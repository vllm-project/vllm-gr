# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the prefill -> step0 in-process chain.

Covers the boundary that removes the prefill/decode step-update round-trip:

- ``_handle_beam_session_template`` caches the session's step template
  (and ``_cache_beam_request`` preserves an out-of-order template).
- ``_beam_advance_from_decision``'s prefill branch builds the step-0
  BeamRequestStepUpdate from a worker prefill decision (parents all zero)
  plus the cached template, applies it in-process and marks the decision.
- Both the prefill and decode branches are gated by ``beam_engine_driven``.
- Skip paths: no template / flag off / width mismatch / single-step
  session (max_decode_steps < 2) / cross-session decision ownership.
- Wire plumbing: BEAM_SESSION_TEMPLATE enum value + payload decode routes
  through the BeamRequestStepUpdate decoder.
- API-side hardening: outputs arriving between collectors are buffered by
  the patched ``OutputProcessor.process_outputs`` and replayed by
  ``register_beam_output_fn``.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequestType, FinishReason

from vllm_gr.v1.engine.core import (
    _beam_advance_from_decision,
    _cache_beam_request,
    _handle_beam_session_template,
    _handle_submit_session,
)
from vllm_gr.v1.engine.types import BeamRequestStepUpdate
from vllm_gr.v1.engine.wire import (
    BEAM_SESSION_TEMPLATE_WIRE_VALUE,
    register_engine_core_request_types,
)

register_engine_core_request_types()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sampling_params(**overrides: Any) -> SamplingParams:
    defaults = dict(max_tokens=4, logprobs=2, temperature=0.0, detokenize=False)
    defaults.update(overrides)
    return SamplingParams(**defaults)


def _make_engine_self(
    beam_cache: dict[str, Any] | None = None,
    *,
    enabled: bool = True,
) -> SimpleNamespace:
    eng = SimpleNamespace(
        beam_cache=beam_cache if beam_cache is not None else {},
        beam_cache_lock=threading.Lock(),
        request_block_hasher=None,
        input_queue=queue.SimpleQueue(),
        output_queue=queue.SimpleQueue(),
        aborts_queue=queue.SimpleQueue(),
        engine_index=0,
    )
    eng._vllm_gr_self_advance = enabled
    return eng


def _session_entry(token_ids: list[int] | None = None) -> dict[str, Any]:
    return {
        "all_token_ids": token_ids if token_ids is not None else [10, 20, 30],
        "block_hashes": [0xAA],
        "eos_token_id": 2,
        "lora_request": None,
        "cache_salt": None,
        "mm_features": None,
        "prompt_embeds": None,
        "prompt_is_token_ids": None,
        "num_prompt_tokens": 3,
    }


def _step_template(
    session_id: str = "s0",
    *,
    beam_width: int = 2,
    prefix_len: int = 3,
    max_decode_steps: int = 3,
    constraint_step: int = 1,
    sampling_params: SamplingParams | None = None,
) -> BeamRequestStepUpdate:
    return BeamRequestStepUpdate(
        session_id=session_id,
        parent_beam_ids=[],
        child_beam_ids=[],
        beam_tokens=[],
        pruned_ids=[],
        prefix_len=prefix_len,
        beam_width=beam_width,
        sampling_params=sampling_params or _sampling_params(),
        eos_token_id=9,
        client_index=4,
        current_wave=2,
        priority=1,
        data_parallel_rank=None,
        max_decode_steps=max_decode_steps,
        constraint_step=constraint_step,
    )


def _prefill_decision(
    tokens: list[int],
    *,
    session_id: str = "s0",
    parents: list[int] | None = None,
) -> dict[str, Any]:
    """Prefill worker decision as it arrives post-IPC (plain lists)."""
    return {
        "out_token_ids": [tokens],
        "out_token_index": [parents if parents is not None else [0] * len(tokens)],
        "out_log_probs": [[0.0] * len(tokens)],
        "is_last_step": False,
        "stage": "prefill",
        "session_id": session_id,
    }


def _decode_decision(parents: list[int], tokens: list[int]) -> dict[str, Any]:
    return {
        "out_token_ids": [tokens],
        "out_token_index": [parents],
        "out_log_probs": [[0.0] * len(tokens)],
        "is_last_step": False,
        "stage": "decode",
    }


def _running_suffix(beam_tokens: list[list[int]] | None = None) -> dict[str, Any]:
    return {
        "beam_tokens": beam_tokens if beam_tokens is not None else [[50], [60]],
        "prefix_len": 3,
        "beam_width": 2,
        "sampling_params": _sampling_params(),
        "eos_token_id": 2,
        "client_index": 0,
        "current_wave": 0,
        "priority": 0,
        "data_parallel_rank": None,
        "lora_request": None,
        "cache_salt": None,
        "trace_headers": None,
        "max_decode_steps": 3,
        "is_constrained": False,
    }


def _fake_prefill_request(
    request_id: str = "s0", token_ids: list[int] | None = None
) -> SimpleNamespace:
    tokens = token_ids if token_ids is not None else [10, 20, 30]
    return SimpleNamespace(
        request_id=request_id,
        _all_token_ids=list(tokens),
        block_hashes=[0xAA],
        cache_salt=None,
        sampling_params=SimpleNamespace(eos_token_id=2),
        lora_request=None,
        mm_features=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        num_prompt_tokens=len(tokens),
        update_block_hashes=lambda: None,
    )


# ===========================================================================
# Section 1 — template caching
# ===========================================================================


class TestSessionTemplateCaching:
    def test_template_cached_in_session_entry(self) -> None:
        eng = _make_engine_self()
        template = _step_template()
        _handle_beam_session_template(eng, template)
        assert eng.beam_cache["s0"]["step_template"] is template

    def test_template_out_of_order_creates_stub(self) -> None:
        eng = _make_engine_self()
        template = _step_template()
        _handle_beam_session_template(eng, template)  # ADD_BATCH not seen yet
        assert eng.beam_cache["s0"]["step_template"] is template

    def test_zero_width_template_ignored(self) -> None:
        eng = _make_engine_self()
        template = _step_template()
        template.beam_width = 0
        _handle_beam_session_template(eng, template)
        assert "s0" not in eng.beam_cache

    def test_cache_beam_request_preserves_out_of_order_template(self) -> None:
        eng = _make_engine_self()
        template = _step_template()
        _handle_beam_session_template(eng, template)
        _cache_beam_request(eng, _fake_prefill_request())
        entry = eng.beam_cache["s0"]
        assert entry["step_template"] is template
        assert entry["all_token_ids"] == [10, 20, 30]

    def test_cache_beam_request_then_template(self) -> None:
        """FIFO order (ADD_BATCH first, template second)."""
        eng = _make_engine_self()
        _cache_beam_request(eng, _fake_prefill_request())
        template = _step_template()
        _handle_beam_session_template(eng, template)
        assert eng.beam_cache["s0"]["step_template"] is template


# ===========================================================================
# Section 2 — prefill branch of _beam_advance_from_decision
# ===========================================================================


class TestPrefillSelfAdvanceChain:
    def _engine_with_template(self, **template_kwargs: Any) -> SimpleNamespace:
        entry = _session_entry()
        entry["step_template"] = _step_template(**template_kwargs)
        return _make_engine_self(beam_cache={"s0": entry})

    def test_prefill_decision_chains_step0(self) -> None:
        eng = self._engine_with_template()
        decision = _prefill_decision(tokens=[61, 71])

        with (
            patch("vllm.v1.request.Request") as MockReq,
            patch(
                "vllm_gr.v1.engine.core._engine_constraint_table_active",
                return_value=True,
            ),
        ):
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.qsize() == 1
        req_type, payload = eng.input_queue.get_nowait()
        assert req_type is EngineCoreRequestType.ADD
        chained_req, wave = payload
        assert chained_req is mock_req
        assert wave == 2  # current_wave from the template

        _, kwargs = MockReq.call_args
        # beam_token_ids = prefill prefix + one new token per beam (identity
        # mapping: prefill decision parents are all zero).
        assert kwargs["request_id"] == "s0"
        assert kwargs["prompt_token_ids"] == [10, 20, 30, 61, 71]
        # Beam attributes are stamped post-construction by _apply_beam_step.
        assert mock_req.prefix_len == 3
        assert mock_req.beam_width == 2
        assert mock_req.max_decode_steps == 3
        assert mock_req.beam_decode_steps == 1
        assert mock_req.beam_decode_step == 0  # step 0
        assert mock_req.parent_beam_indices == [0, 0]
        assert mock_req.constraint_step == 1  # template value replayed

        # The chained step seeds running_suffix for later decode chaining.
        running = eng.beam_cache["s0"]["running_suffix"]
        assert running["beam_tokens"] == [[61], [71]]
        assert running["max_decode_steps"] == 3

        assert decision.get("beam_self_advance_chained") is True

    def test_prefill_without_template_is_noop(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        decision = _prefill_decision(tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()
        assert "beam_self_advance_chained" not in decision

    def test_prefill_flag_off_is_noop(self) -> None:
        entry = _session_entry()
        entry["step_template"] = _step_template()
        eng = _make_engine_self(
            beam_cache={"s0": entry},
            enabled=False,
        )
        decision = _prefill_decision(tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_single_decode_step_session_is_noop(self) -> None:
        """max_decode_steps < 2 means the frontend loop ends right after the
        prefill candidates; chaining step0 would orphan the session."""
        eng = self._engine_with_template(max_decode_steps=1)
        decision = _prefill_decision(tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_width_mismatch_is_noop(self) -> None:
        eng = self._engine_with_template(beam_width=2)
        decision = _prefill_decision(tokens=[61, 71, 81])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_prefill_decision_containing_eos_falls_back_to_frontend(self) -> None:
        eng = self._engine_with_template()
        decision = _prefill_decision(tokens=[61, 9])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()
        assert "beam_self_advance_chained" not in decision

    def test_cross_session_decision_is_noop(self) -> None:
        """Mixed batches: a decision owned by another session must never
        chain onto this one (review P5)."""
        eng = self._engine_with_template()
        decision = _prefill_decision(tokens=[61, 71], session_id="other-session")

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_unmarked_decision_with_template_chains_as_prefill(self) -> None:
        """Legacy payloads without stage fall back to the prefill branch when
        no running_suffix exists but a template does."""
        eng = self._engine_with_template()
        decision = _prefill_decision(tokens=[61, 71])
        decision.pop("stage")
        decision.pop("session_id")

        with patch("vllm.v1.request.Request") as MockReq:
            MockReq.return_value.block_hashes = []
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.qsize() == 1


# ===========================================================================
# Section 3 — engine-driven on/off gate
# ===========================================================================


class TestDualFlagScoping:
    def _decode_seeded_engine(self, **flags: bool) -> SimpleNamespace:
        entry = _session_entry()
        entry["running_suffix"] = _running_suffix()
        return _make_engine_self(beam_cache={"s0": entry}, **flags)

    def test_disabled_does_not_chain_decode(self) -> None:
        """engine_driven=False keeps ALL transitions frontend-driven."""
        eng = self._decode_seeded_engine(enabled=False)
        decision = _decode_decision(parents=[0, 1], tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_enabled_chains_decode(self) -> None:
        """engine_driven=True chains decode steps in-process."""
        eng = self._decode_seeded_engine(enabled=True)
        decision = _decode_decision(parents=[0, 1], tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            MockReq.return_value.block_hashes = []
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.qsize() == 1

    def test_decode_decision_containing_eos_falls_back_to_frontend(self) -> None:
        eng = self._decode_seeded_engine(enabled=True)
        decision = _decode_decision(parents=[0, 1], tokens=[61, 2])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()
        assert "beam_self_advance_chained" not in decision

    def test_decode_flag_still_chains_decode(self) -> None:
        eng = self._decode_seeded_engine(enabled=True)
        decision = _decode_decision(parents=[0, 1], tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            MockReq.return_value.block_hashes = []
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.qsize() == 1

    def test_prefill_only_flag_leaves_decode_decision_fallback(self) -> None:
        """Prefill decision arriving for a session that already has
        running_suffix treats it as decode-shaped and skips (not chained)."""
        eng = self._decode_seeded_engine(enabled=True)
        decision = _prefill_decision(tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()


# ===========================================================================
# Section 4 — wire plumbing
# ===========================================================================


class TestWirePlumbing:
    def test_wire_value_registered(self) -> None:
        assert BEAM_SESSION_TEMPLATE_WIRE_VALUE == b"\x08"
        assert EngineCoreRequestType.BEAM_SESSION_TEMPLATE.value == b"\x08"
        # Existing members untouched.
        assert EngineCoreRequestType.ADD_BATCH.value == b"\x06"
        assert EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE.value == b"\x07"
        # SUBMIT_SESSION (merged submission) registered on the next free
        # wire value.
        from vllm_gr.v1.engine.wire import SUBMIT_SESSION_WIRE_VALUE

        assert SUBMIT_SESSION_WIRE_VALUE == b"\x09"
        assert EngineCoreRequestType.SUBMIT_SESSION.value == b"\x09"

    def test_template_decodes_via_step_update_decoder(self) -> None:
        import msgspec

        from vllm_gr.v1.engine.codec import (
            build_engine_core_decoders,
            decode_engine_core_payload,
        )

        template = _step_template(max_decode_steps=5)
        buf = msgspec.msgpack.encode(template)
        decoders = build_engine_core_decoders(None)
        decoded = decode_engine_core_payload(
            decoders, EngineCoreRequestType.BEAM_SESSION_TEMPLATE, [memoryview(buf)]
        )
        assert isinstance(decoded, BeamRequestStepUpdate)
        assert decoded.prefix_len == 3
        assert decoded.max_decode_steps == 5
        assert decoded.constraint_step == 1


# ===========================================================================
# Section 4b — SUBMIT_SESSION merged submission (wire-message merge, C3)
# ===========================================================================


class TestSubmitSessionHandler:
    """_handle_submit_session = ADD_BATCH handling + template caching."""

    def _engine(self) -> SimpleNamespace:
        eng = _make_engine_self()
        eng.preprocess_add_request = lambda req: (req, 7)
        eng._handle_request_preproc_error = MagicMock()
        eng._cache_beam_request = lambda req: _cache_beam_request(eng, req)
        eng._handle_beam_session_template = lambda tpl, **kw: _handle_beam_session_template(
            eng, tpl, **kw
        )
        return eng

    def test_caches_request_and_template_and_enqueues_add(self) -> None:
        eng = self._engine()
        template = _step_template()

        _handle_submit_session(eng, [_fake_prefill_request()], template)

        entry = eng.beam_cache["s0"]
        assert entry["step_template"] is template
        assert entry["all_token_ids"] == [10, 20, 30]
        assert eng.input_queue.qsize() == 1
        req_type, payload = eng.input_queue.get_nowait()
        assert req_type is EngineCoreRequestType.ADD
        preprocessed, wave = payload
        assert preprocessed.request_id == "s0"
        assert wave == 7

    def test_preproc_failure_skips_request_but_keeps_template(self) -> None:
        """Mirrors the ADD_BATCH branch: a failing request is reported and
        dropped; the template (which rides the same message) is unaffected."""
        eng = self._engine()
        eng.preprocess_add_request = MagicMock(side_effect=ValueError("bad prompt"))
        template = _step_template()

        _handle_submit_session(eng, [_fake_prefill_request()], template)

        assert eng.input_queue.empty()
        eng._handle_request_preproc_error.assert_called_once()
        assert eng.beam_cache["s0"]["step_template"] is template

    def test_none_template_still_submits_batch(self) -> None:
        eng = self._engine()

        _handle_submit_session(eng, [_fake_prefill_request()], None)

        assert eng.input_queue.qsize() == 1
        assert "s0" in eng.beam_cache
        assert "step_template" not in eng.beam_cache["s0"]

    def test_multiple_requests_all_cached(self) -> None:
        eng = self._engine()
        requests = [_fake_prefill_request("s0"), _fake_prefill_request("s1")]

        _handle_submit_session(eng, requests, _step_template())

        assert eng.input_queue.qsize() == 2
        assert "s0" in eng.beam_cache and "s1" in eng.beam_cache

    def test_template_is_cached_before_add_is_enqueued(self) -> None:
        """Busy-loop must not see ADD while the template is still absent."""
        import queue as queue_mod

        eng = self._engine()
        order: list[str] = []
        recorded: queue_mod.SimpleQueue[Any] = queue_mod.SimpleQueue()

        def put(item: Any) -> None:
            order.append("add")
            recorded.put_nowait(item)

        orig_template = eng._handle_beam_session_template

        def handle_template(template: Any, **kwargs: Any) -> None:
            order.append("template")
            orig_template(template, **kwargs)

        eng.input_queue = SimpleNamespace(put_nowait=put)
        eng._handle_beam_session_template = handle_template

        _handle_submit_session(eng, [_fake_prefill_request()], _step_template())

        assert order == ["template", "add"]
        assert eng.beam_cache["s0"]["step_template"] is not None
        assert recorded.qsize() == 1


class TestSubmitSessionClients:
    """AsyncMP / SyncMP / Inproc client paths + AsyncLLM dispatch."""

    def test_sync_mp_sends_merged_payload(self) -> None:
        from vllm_gr.v1.engine.core_client_patch import sync_mp_submit_session

        client = MagicMock()
        template = _step_template()
        requests = [_fake_prefill_request()]

        sync_mp_submit_session(client, requests, template)

        (req_type, payload), _kwargs = client._send_input.call_args
        assert req_type is EngineCoreRequestType.SUBMIT_SESSION
        assert payload[0] is requests
        assert payload[1] is template

    def test_sync_mp_empty_requests_is_noop(self) -> None:
        from vllm_gr.v1.engine.core_client_patch import sync_mp_submit_session

        client = MagicMock()
        sync_mp_submit_session(client, [], _step_template())
        client._send_input.assert_not_called()

    def test_inproc_calls_engine_handler(self) -> None:
        from vllm_gr.v1.engine.core_client_patch import inproc_submit_session

        engine_core = MagicMock()
        client = SimpleNamespace(engine_core=engine_core)
        template = _step_template()
        requests = [_fake_prefill_request()]

        inproc_submit_session(client, requests, template)

        engine_core._handle_submit_session.assert_called_once_with(requests, template)

    def test_async_client_stamps_and_sends_tuple(self) -> None:
        from vllm_gr.v1.engine.core_client import submit_session_async

        sent: list[tuple[Any, Any, Any]] = []

        async def fake_send(req_type: Any, payload: Any, engine: Any = None) -> None:
            sent.append((req_type, payload, engine))

        client = SimpleNamespace(
            client_index=3,
            _send_input=fake_send,
            _ensure_output_queue_task=lambda: None,
        )
        template = _step_template()
        requests = [_fake_prefill_request()]

        asyncio.run(submit_session_async(client, requests, template))

        assert len(sent) == 1
        req_type, payload, engine = sent[0]
        assert req_type is EngineCoreRequestType.SUBMIT_SESSION
        assert payload[0] is requests
        assert payload[1] is template
        assert engine is None  # non-DP client
        # Routing stamps mirror add_requests_async.
        assert requests[0].client_index == 3
        assert template.client_index == 3

    def test_async_client_empty_requests_is_noop(self) -> None:
        from vllm_gr.v1.engine.core_client import submit_session_async

        client = SimpleNamespace(
            client_index=0,
            _send_input=MagicMock(side_effect=AssertionError("must not send")),
            _ensure_output_queue_task=lambda: None,
        )
        asyncio.run(submit_session_async(client, [], _step_template()))

    def test_async_client_dp_routing_registers_in_flight(self) -> None:
        from vllm_gr.v1.engine.core_client import submit_session_async

        chosen = object()
        sent: list[tuple[Any, Any, Any]] = []

        async def fake_send(req_type: Any, payload: Any, engine: Any = None) -> None:
            sent.append((req_type, payload, engine))

        client = SimpleNamespace(
            client_index=1,
            current_wave=9,
            _send_input=fake_send,
            _ensure_output_queue_task=lambda: None,
            get_core_engine_for_request=lambda request: chosen,
            reqs_in_flight={},
        )
        requests = [_fake_prefill_request("a"), _fake_prefill_request("b")]

        asyncio.run(submit_session_async(client, requests, _step_template()))

        # The first request is registered by get_core_engine_for_request;
        # the rest are registered here (mirrors add_requests_async).
        assert client.reqs_in_flight == {"b": chosen}
        assert sent[0][2] is chosen
        assert [r.current_wave for r in requests] == [9, 9]

    def test_async_llm_prefers_merged_send(self) -> None:
        from vllm_gr.v1.engine.async_llm import submit_session_fn

        sent: list[tuple[Any, Any]] = []

        async def fake_async_submit(requests: Any, template: Any) -> None:
            sent.append((requests, template))

        client = SimpleNamespace(submit_session_async=fake_async_submit)
        llm = SimpleNamespace(engine_core=client)
        template = _step_template()
        requests = [_fake_prefill_request()]

        asyncio.run(submit_session_fn(llm, requests, template))

        assert sent == [(requests, template)]

    def test_async_llm_falls_back_to_inproc_direct_call(self) -> None:
        from vllm_gr.v1.engine.async_llm import submit_session_fn

        called: list[tuple[Any, Any]] = []
        client = SimpleNamespace(submit_session=lambda r, t: called.append((r, t)))
        llm = SimpleNamespace(engine_core=client)
        template = _step_template()
        requests = [_fake_prefill_request()]

        asyncio.run(submit_session_fn(llm, requests, template))

        assert called == [(requests, template)]

    def test_async_llm_legacy_fallback_sends_two_messages(self) -> None:
        from vllm_gr.v1.engine.async_llm import submit_session_fn

        batch_calls: list[tuple[Any, bool]] = []
        template_calls: list[Any] = []

        async def fake_batch(requests: Any, use_batch_message: bool) -> None:
            batch_calls.append((requests, use_batch_message))

        async def fake_template(template: Any) -> None:
            template_calls.append(template)

        client = SimpleNamespace(beam_session_template=fake_template)
        llm = SimpleNamespace(engine_core=client, _add_requests_batch=fake_batch)
        template = _step_template()
        requests = [_fake_prefill_request()]

        asyncio.run(submit_session_fn(llm, requests, template))

        assert batch_calls == [(requests, True)]
        assert template_calls == [template]

    def test_async_llm_tracks_session_before_merged_send(self) -> None:
        from vllm_gr.v1.engine.async_llm import submit_session_fn

        sent: list[tuple[Any, Any]] = []

        async def fake_async_submit(requests: Any, template: Any) -> None:
            sent.append((requests, template))

        processor = SimpleNamespace(_gr_beam_session_ids=None)
        client = SimpleNamespace(submit_session_async=fake_async_submit)
        llm = SimpleNamespace(engine_core=client, output_processor=processor)
        template = _step_template(session_id="s0")
        requests = [_fake_prefill_request()]

        asyncio.run(submit_session_fn(llm, requests, template))

        assert sent == [(requests, template)]
        assert "s0" in processor._gr_beam_session_ids


# ===========================================================================
# Section 5 — API-side output buffering + collector replay
# ===========================================================================


class TestOutputBuffering:
    def _processor(self, sessions: set[str], states: dict[str, Any]) -> SimpleNamespace:
        calls: list[list[Any]] = []

        def original(*args: Any, **_kwargs: Any) -> Any:
            # Accept both call shapes: wrapper (self, outputs) and the
            # bound-style replay processor.process_outputs(outputs).
            outputs = args[-1]
            calls.append(list(outputs))
            return []

        # NOTE: a plain function (not MagicMock) — MagicMock auto-creates
        # attributes, which would spoof the patch marker check.
        return SimpleNamespace(
            request_states=states,
            _gr_beam_session_ids=set(sessions),
            _gr_beam_pending={},
            process_outputs=original,
            add_request=MagicMock(),
            _calls=calls,
        )

    def test_beam_output_without_state_is_buffered(self) -> None:
        from vllm_gr.v1.engine.async_llm import make_patched_process_outputs

        processor = self._processor({"s0"}, {})
        wrapped = make_patched_process_outputs(processor.process_outputs)
        beam_out = EngineCoreOutput(request_id="s0", new_token_ids=[1])
        other_out = EngineCoreOutput(request_id="other", new_token_ids=[2])

        wrapped(processor, [beam_out, other_out])

        # The non-beam output passed through; the beam one was buffered.
        assert [[o.request_id for o in call] for call in processor._calls] == [["other"]]
        assert processor._gr_beam_pending["s0"] == [beam_out]

    def test_beam_output_with_state_passes_through(self) -> None:
        from vllm_gr.v1.engine.async_llm import make_patched_process_outputs

        processor = self._processor({"s0"}, {"s0": object()})
        wrapped = make_patched_process_outputs(processor.process_outputs)
        beam_out = EngineCoreOutput(request_id="s0", new_token_ids=[1])

        wrapped(processor, [beam_out])

        assert [[o.request_id for o in call] for call in processor._calls] == [["s0"]]
        assert processor._gr_beam_pending == {}

    def test_untracked_output_is_not_buffered(self) -> None:
        from vllm_gr.v1.engine.async_llm import make_patched_process_outputs

        processor = self._processor(set(), {})
        wrapped = make_patched_process_outputs(processor.process_outputs)
        beam_out = EngineCoreOutput(request_id="s0", new_token_ids=[1])

        wrapped(processor, [beam_out])

        assert processor._gr_beam_pending == {}
        assert [[o.request_id for o in call] for call in processor._calls] == [["s0"]]

    def test_track_session_before_submit_buffers_chained_output(self) -> None:
        """Prefill dispatch can chain step 0 before register_beam_output."""
        from vllm_gr.v1.engine.async_llm import (
            make_patched_process_outputs,
            track_beam_session_fn,
        )

        processor = self._processor(set(), {})
        llm = SimpleNamespace(output_processor=processor)
        track_beam_session_fn(llm, "s0")
        wrapped = make_patched_process_outputs(processor.process_outputs)
        chained = EngineCoreOutput(request_id="s0", new_token_ids=[9])

        wrapped(processor, [chained])

        assert "s0" in processor._gr_beam_session_ids
        assert processor._gr_beam_pending["s0"] == [chained]
        assert [[o.request_id for o in call] for call in processor._calls] == [[]]

    def test_buffer_overflow_keeps_live_session_output(self) -> None:
        from vllm_gr.v1.engine.async_llm import (
            _GR_BEAM_PENDING_PER_ID,
            make_patched_process_outputs,
        )

        processor = self._processor({"s0"}, {})
        wrapped = make_patched_process_outputs(processor.process_outputs)
        outputs = [EngineCoreOutput(request_id="s0", new_token_ids=[i]) for i in range(8)]

        wrapped(processor, outputs)

        # Live sessions keep every output; the cap is an overflow metric.
        assert len(processor._gr_beam_pending["s0"]) == 8
        assert processor._gr_beam_pending_overflows == 8 - _GR_BEAM_PENDING_PER_ID

    def test_register_replays_buffered_outputs(self) -> None:
        from vllm_gr.v1.engine.async_llm import register_beam_output_fn

        buffered = [EngineCoreOutput(request_id="s0", new_token_ids=[1])]
        processor = self._processor({"s0"}, {})
        processor._gr_beam_pending["s0"] = buffered
        llm = SimpleNamespace(
            output_processor=processor,
            _run_output_handler=lambda: None,
        )

        returned = register_beam_output_fn(
            llm,
            "s0",
            [10, 20],
            _sampling_params(),
        )

        # Session tracked, buffer drained and replayed through the
        # (patched) process_outputs, collector returned.
        assert "s0" in processor._gr_beam_session_ids
        assert processor._gr_beam_pending == {}
        assert any([o.request_id for o in call] == ["s0"] for call in processor._calls)
        processor.add_request.assert_called_once()
        assert returned is not None


# ===========================================================================
# Section 6 — runtime auto-default (backend promoted after create_engine_config)
# ===========================================================================


class TestRuntimeAutoDefault:
    def _config(self, **overrides: Any) -> Any:
        from vllm_gr.config import GRConfig

        base = dict(
            catalog_path="/tmp/fake-catalog.json",
            constraint_backend=None,
            beam_max_width=128,
            beam_worker_decision=True,
        )
        base.update(overrides)
        from vllm_gr.config import ConstraintBackend

        if base["constraint_backend"] is None:
            base["constraint_backend"] = ConstraintBackend.CONSTRAINT_TABLE
        return GRConfig(
            catalog_path=base["catalog_path"],
            constraint_backend=base["constraint_backend"],
            beam_max_width=base["beam_max_width"],
            beam_worker_decision=base["beam_worker_decision"],
            beam_engine_driven=base.get("beam_engine_driven"),
        )

    def test_default_on_for_accelerator_beam_deployments(self) -> None:
        """Unset flag + worker decision + beam width -> engine-driven ON."""
        from vllm_gr.v1.engine.core import _engine_self_advance_active

        cfg = self._config()
        assert cfg.beam_engine_driven is None  # truly unset
        with patch(
            "vllm_gr.config.GRConfig.from_vllm_config",
            classmethod(lambda cls, vc: cfg),
        ):
            eng = SimpleNamespace(vllm_config=object())
            assert _engine_self_advance_active(eng) is True

    def test_explicit_false_wins_over_default(self) -> None:
        from vllm_gr.v1.engine.core import _engine_self_advance_active

        cfg = self._config(beam_engine_driven=False)
        assert cfg.beam_engine_driven is False
        with patch(
            "vllm_gr.config.GRConfig.from_vllm_config",
            classmethod(lambda cls, vc: cfg),
        ):
            eng = SimpleNamespace(vllm_config=object())
            assert _engine_self_advance_active(eng) is False

    def test_without_worker_decision_stays_off(self) -> None:
        from vllm_gr.v1.engine.core import _engine_self_advance_active

        cfg = self._config(beam_worker_decision=None)
        with patch(
            "vllm_gr.config.GRConfig.from_vllm_config",
            classmethod(lambda cls, vc: cfg),
        ):
            eng = SimpleNamespace(vllm_config=object())
            assert _engine_self_advance_active(eng) is False


# ===========================================================================
# Section 7 — unified engine-driven mode (scheduling refactor Phase 2)
# ===========================================================================


class TestEngineDrivenRuntimeDefault:
    """beam_engine_driven: explicit value / runtime default."""

    def _flags(self, cfg: Any) -> bool:
        from vllm_gr.v1.engine.core import _engine_self_advance_active

        with patch(
            "vllm_gr.config.GRConfig.from_vllm_config",
            classmethod(lambda cls, vc: cfg),
        ):
            eng = SimpleNamespace(vllm_config=object())
            return cast(bool, _engine_self_advance_active(eng))

    def _config(self, **overrides: Any) -> Any:
        from vllm_gr.config import GRConfig

        base = dict(beam_max_width=128, beam_worker_decision=True)
        base.update(overrides)
        return GRConfig(**base)

    def test_explicit_true(self) -> None:
        cfg = self._config(beam_engine_driven=True)
        assert self._flags(cfg) is True

    def test_explicit_false(self) -> None:
        cfg = self._config(beam_engine_driven=False)
        assert self._flags(cfg) is False

    def test_runtime_default(self) -> None:
        """Unset flag auto-enables for accelerator beam with worker decisions."""
        cfg = self._config()
        assert cfg.beam_engine_driven is None
        assert self._flags(cfg) is True

    def test_requires_worker_decision(self) -> None:
        from vllm_gr.config import GRConfig

        with pytest.raises(ValueError, match="beam_engine_driven=True requires"):
            GRConfig(beam_engine_driven=True)

        # With the worker decision the same config validates.
        GRConfig(beam_engine_driven=True, beam_worker_decision=True)


class TestFifoReplayAndCap:
    """R1: a session's buffered outputs are released ONE per register.

    Every beam step Request runs with max_tokens=1, so each EngineCoreOutput
    arrives finished=True and _collect_beam_result consumes exactly one
    output per beam_search loop iteration (register:consume = 1:1).  Replaying
    the whole buffer in ONE process_outputs call would let the first finished
    output pop the request state and the rest would hit upstream's
    req_state-is-None drop — the next iteration would then await a collector
    that never fires (bounded only by the 120s collect timeout).
    """

    def _llm(
        self,
        sessions: Any = (),
        states: Any = None,
        vllm_config: Any = None,
    ) -> Any:
        processor = TestOutputBuffering()._processor(set(sessions), states or {})
        return (
            SimpleNamespace(
                output_processor=processor,
                _run_output_handler=lambda: None,
                vllm_config=vllm_config if vllm_config is not None else object(),
            ),
            processor,
        )

    def test_three_buffered_outputs_release_one_per_register_in_order(self) -> None:
        from vllm_gr.v1.engine.async_llm import register_beam_output_fn

        llm, processor = self._llm({"s0"})
        outputs = [EngineCoreOutput(request_id="s0", new_token_ids=[i]) for i in range(3)]
        processor._gr_beam_pending["s0"] = list(outputs)

        for _ in range(3):
            register_beam_output_fn(llm, "s0", [10, 20], _sampling_params())

        # One output per register call, FIFO order, none lost or replayed twice.
        released = [[o.new_token_ids[0] for o in call] for call in processor._calls]
        assert released == [[0], [1], [2]]
        assert processor._gr_beam_pending == {}

    def test_terminal_output_released_by_final_register(self) -> None:
        from vllm_gr.v1.engine.async_llm import register_beam_output_fn

        llm, processor = self._llm({"s0"})
        step_out = EngineCoreOutput(request_id="s0", new_token_ids=[5])
        terminal = EngineCoreOutput(
            request_id="s0", new_token_ids=[9], finish_reason=FinishReason.STOP
        )
        processor._gr_beam_pending["s0"] = [step_out, terminal]

        register_beam_output_fn(llm, "s0", [10, 20], _sampling_params())
        # Only the oldest releases; the terminal output stays held back.
        assert processor._calls == [[step_out]]
        assert processor._gr_beam_pending["s0"] == [terminal]

        register_beam_output_fn(llm, "s0", [10, 20], _sampling_params())
        assert processor._calls[-1] == [terminal]
        assert "s0" not in processor._gr_beam_pending

    def test_register_stamps_dynamic_cap_from_max_decode_steps(self) -> None:
        from vllm_gr.v1.engine.async_llm import register_beam_output_fn

        vllm_config = SimpleNamespace(additional_config={"vllm_gr_beam_max_decode_steps": 8})
        llm, processor = self._llm(vllm_config=vllm_config)
        register_beam_output_fn(llm, "s0", [10], _sampling_params())
        # Live-session lead allowance: one output per engine-led decode step
        # (<= max_decode_steps) plus the terminal output and a margin.
        assert processor._gr_beam_pending_cap == 10

    def test_buffer_respects_dynamic_cap(self) -> None:
        from vllm_gr.v1.engine.async_llm import make_patched_process_outputs

        processor = TestOutputBuffering()._processor({"s0"}, {})
        processor._gr_beam_pending_cap = 3
        wrapped = make_patched_process_outputs(processor.process_outputs)
        outputs = [EngineCoreOutput(request_id="s0", new_token_ids=[i]) for i in range(5)]

        wrapped(processor, outputs)

        assert len(processor._gr_beam_pending["s0"]) == 5
        assert processor._gr_beam_pending_overflows == 2
        # Overflow is kept, not forwarded to upstream drop.
        assert not any(processor._calls)

    def test_bootstrap_cap_without_register(self) -> None:
        """Before the first register stamps the dynamic cap, the module
        bootstrap cap (4) bounds the buffer."""
        from vllm_gr.v1.engine.async_llm import make_patched_process_outputs

        processor = TestOutputBuffering()._processor({"s0"}, {})
        wrapped = make_patched_process_outputs(processor.process_outputs)
        outputs = [EngineCoreOutput(request_id="s0", new_token_ids=[i]) for i in range(9)]

        wrapped(processor, outputs)

        assert len(processor._gr_beam_pending["s0"]) == 9
        assert processor._gr_beam_pending_overflows == 5

    def test_cleanup_update_discards_buffer_and_untracks(self) -> None:
        from vllm_gr.v1.engine.async_llm import beam_request_step_update_fn

        llm, processor = self._llm({"s0"})
        processor._gr_beam_pending["s0"] = [EngineCoreOutput(request_id="s0", new_token_ids=[1])]
        sent: list[BeamRequestStepUpdate] = []

        async def fake_update(update: BeamRequestStepUpdate) -> None:
            sent.append(update)

        llm.engine_core = SimpleNamespace(beam_request_step_update_async=fake_update)
        cleanup = BeamRequestStepUpdate(
            session_id="s0",
            parent_beam_ids=[],
            child_beam_ids=[],
            beam_tokens=[],
            pruned_ids=["x"],
            prefix_len=0,
            beam_width=0,
            sampling_params=_sampling_params(),
        )

        asyncio.run(beam_request_step_update_fn(llm, cleanup))

        assert processor._gr_beam_pending == {}
        assert "s0" not in processor._gr_beam_session_ids
        assert sent == [cleanup]  # the teardown still reaches the engine

    def test_non_cleanup_update_keeps_buffer(self) -> None:
        from vllm_gr.v1.engine.async_llm import beam_request_step_update_fn

        llm, processor = self._llm({"s0"})
        buffered = [EngineCoreOutput(request_id="s0", new_token_ids=[1])]
        processor._gr_beam_pending["s0"] = buffered

        async def fake_update(update: BeamRequestStepUpdate) -> None:
            pass

        llm.engine_core = SimpleNamespace(beam_request_step_update_async=fake_update)
        step = BeamRequestStepUpdate(
            session_id="s0",
            parent_beam_ids=["p"],
            child_beam_ids=["c"],
            beam_tokens=[[7]],
            pruned_ids=[],
            prefix_len=3,
            beam_width=1,
            sampling_params=_sampling_params(),
        )

        asyncio.run(beam_request_step_update_fn(llm, step))

        assert processor._gr_beam_pending["s0"] is buffered


# ===========================================================================
# Section 9 — Y4: unconstrained-deployment invariants under the new default
# ===========================================================================


class TestUnconstrainedDefaultInvariants:
    """What the engine-driven default actually does WITHOUT Constraint Tables.

    The prefill boundary never chains on unconstrained deployments (no
    template/prefill decision exists), but decode steps DO chain once
    seeded by the frontend's first step update.
    """

    def test_flags_resolve_engine_driven_without_constraint_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vllm_gr.config import GRConfig
        from vllm_gr.v1.engine.core import _engine_self_advance_active

        cfg = GRConfig(beam_worker_decision=True, beam_max_width=128)
        monkeypatch.setattr(
            GRConfig,
            "from_vllm_config",
            classmethod(lambda cls, vc: cfg),
        )
        eng = SimpleNamespace(vllm_config=object())
        assert _engine_self_advance_active(eng) is True

    def test_prefill_never_chains_without_template(self) -> None:
        """Unconstrained sessions receive no step template, so even a stray
        prefill-shaped decision cannot chain (fallback stays frontend)."""
        entry = _session_entry()  # no step_template key at all
        eng = _make_engine_self(beam_cache={"s0": entry}, enabled=True)
        decision = _prefill_decision(tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()
        assert "beam_self_advance_chained" not in decision

    def test_decode_chains_once_frontend_seeded_running_suffix(self) -> None:
        """The seed for unconstrained sessions comes from the frontend's own
        first step update; from then on decode decisions chain."""
        entry = _session_entry()
        entry["running_suffix"] = _running_suffix()
        eng = _make_engine_self(beam_cache={"s0": entry}, enabled=True)
        decision = _decode_decision(parents=[0, 1], tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            MockReq.return_value.block_hashes = []
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.qsize() == 1
        assert decision.get("beam_self_advance_chained") is True


# ===========================================================================
# Section 10 — G-nit2: template-cache log names its wire source
# ===========================================================================


class TestTemplateCacheLogSource:
    def test_submit_session_log_names_merged_source(self) -> None:
        """The merged SUBMIT_SESSION path reuses the template handler; the
        log line must attribute the source so log-based triage is not
        misled (G-nit2), while the stable prefix keeps existing greps
        working."""
        import logging

        eng = _make_engine_self()
        records: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
        logger = logging.getLogger("vllm_gr.v1.engine.core")
        old_level, old_propagate = logger.level, logger.propagate
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            _handle_beam_session_template(eng, _step_template(), via="SUBMIT_SESSION")
            _handle_beam_session_template(eng, _step_template(session_id="s1"))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
            logger.propagate = old_propagate

        merged = [m for m in records if "s0" in m]
        standalone = [m for m in records if "s1" in m]
        assert any("via SUBMIT_SESSION" in m for m in merged)
        assert all("via" not in m for m in standalone)
        # Stable prefix substring for existing greps.
        assert all(m.startswith("vllm-gr: BEAM_SESSION_TEMPLATE received") for m in records)
