# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the BeamRequest beam search protocol (no GPU required).

Sections:
  1  — BeamRequestStepUpdate serialization / msgpack round-trip
  2  — EngineCoreRequestType enum patching (_add_enum_member)
  3  — EngineCore beam cache (_cache_beam_request / _handle_beam_request_step_update)
  4  — Client-side patches (add_requests_async / beam_request_step_update_async /
         apply_batch_fork_patches / sync + inproc variants)
  5  — Beam search orchestration (_beam_request_step / _beam_request_cleanup /
         use_beam_request flag detection)
  6  — Patch wiring (patch_batch_and_fork / EngineCoreProc.run_engine_core)
  7  — Integration / cross-process (enum no-collision, encode-decode)
"""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import asyncio
import queue
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import msgspec
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    EngineCoreRequestType,
    FinishReason,
)

from vllm_gr.entrypoints.openai.serving_engine import (
    _beam_request_cleanup,
    _beam_request_step,
)
from vllm_gr.v1.engine.core import _cache_beam_request, _handle_beam_request_step_update
from vllm_gr.v1.engine.core_client import (
    add_requests_async,
    beam_request_step_update_async,
)
from vllm_gr.v1.engine.core_client_patch import (
    _add_requests_batch_fn,
    apply_batch_fork_patches,
    beam_request_step_update_fn,
    inproc_add_requests,
    inproc_beam_request_step_update,
    prepare_request_fn,
    register_beam_output_fn,
    sync_mp_add_requests,
    sync_mp_beam_request_step_update,
)
from vllm_gr.v1.engine.engine_core_patch import _add_enum_member
from vllm_gr.v1.engine.types import BeamRequestStepUpdate

# Ensure custom enum members exist in this process for all tests below.
_add_enum_member("ADD_BATCH", b"\x06")
_add_enum_member("BEAM_REQUEST_STEP_UPDATE", b"\x07")

# ---------------------------------------------------------------------------
# Module-level picklable targets for subprocess tests (§6)
# ---------------------------------------------------------------------------


def _sub_patch_batch_fork_registers_enum() -> None:
    """Verify patch_batch_and_fork adds BEAM_REQUEST_STEP_UPDATE to the enum."""
    from vllm.v1.engine import EngineCoreRequestType as ERT

    from vllm_gr.entrypoints.openai.beam_search_patch import patch_batch_and_fork

    patch_batch_and_fork()
    assert hasattr(ERT, "ADD_BATCH"), "ADD_BATCH missing after patch"
    assert hasattr(ERT, "BEAM_REQUEST_STEP_UPDATE"), "BEAM_REQUEST_STEP_UPDATE missing after patch"
    assert ERT.BEAM_REQUEST_STEP_UPDATE.value == b"\x07"
    assert ERT.ADD_BATCH.value == b"\x06"


def _sub_patch_batch_fork_wires_run_engine_core() -> None:
    """Verify patch_batch_and_fork wires EngineCoreProc.run_engine_core."""
    from vllm.v1.engine.core import EngineCoreProc

    from vllm_gr.entrypoints.openai.beam_search_patch import patch_batch_and_fork
    from vllm_gr.v1.engine.engine_core_patch import run_engine_core

    patch_batch_and_fork()
    assert EngineCoreProc.run_engine_core is run_engine_core, (
        f"run_engine_core not wired; got {EngineCoreProc.run_engine_core}"
    )


def _sub_patch_batch_fork_idempotent() -> None:
    """Calling patch_batch_and_fork twice must not double-wrap run_engine_core."""
    from vllm.v1.engine.core import EngineCoreProc

    from vllm_gr.entrypoints.openai.beam_search_patch import patch_batch_and_fork
    from vllm_gr.v1.engine.engine_core_patch import run_engine_core

    patch_batch_and_fork()
    first = EngineCoreProc.run_engine_core
    patch_batch_and_fork()
    second = EngineCoreProc.run_engine_core
    assert first is second is run_engine_core, "Second call must be a no-op"


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _sampling_params(**overrides: Any) -> SamplingParams:
    defaults = dict(max_tokens=4, logprobs=3, temperature=0.0, detokenize=False)
    defaults.update(overrides)
    return SamplingParams(**defaults)


def _make_mock_request(
    request_id: str = "req-0",
    token_ids: list[int] | None = None,
    block_hashes: list[int] | None = None,
    eos_token_id: int = 2,
) -> MagicMock:
    m = MagicMock()
    m.request_id = request_id
    m._all_token_ids = token_ids if token_ids is not None else [10, 20, 30]
    m.block_hashes = block_hashes if block_hashes is not None else [0xAA, 0xBB]
    m.sampling_params = _sampling_params()
    m.sampling_params.update_from_generation_config({}, eos_token_id)
    m.lora_request = None
    m.cache_salt = None
    m.mm_features = None
    m.prompt_embeds = None
    m.prompt_is_token_ids = None
    m.num_prompt_tokens = len(token_ids or [10, 20, 30])
    return m


def _make_engine_self(
    beam_cache: dict | None = None,
    block_hasher: Any | None = None,
    engine_index: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        beam_cache=beam_cache if beam_cache is not None else {},
        beam_cache_lock=threading.Lock(),
        request_block_hasher=block_hasher,
        input_queue=queue.SimpleQueue(),
        output_queue=queue.SimpleQueue(),
        aborts_queue=queue.SimpleQueue(),
        engine_index=engine_index,
    )


def _session_entry(
    token_ids: list[int] | None = None,
    block_hashes: list[int] | None = None,
    eos_token_id: int = 2,
) -> dict:
    return {
        "all_token_ids": token_ids if token_ids is not None else [10, 20, 30],
        "block_hashes": block_hashes if block_hashes is not None else [0xAA],
        "eos_token_id": eos_token_id,
        "lora_request": None,
        "cache_salt": None,
        "mm_features": None,
        "prompt_embeds": None,
        "prompt_is_token_ids": None,
        "num_prompt_tokens": len(token_ids or [10, 20, 30]),
    }


def _make_beam_update(
    session_id: str = "session-0",
    parent_beam_ids: list[str] | None = None,
    child_beam_ids: list[str] | None = None,
    beam_tokens: list[list[int]] | None = None,
    pruned_ids: list[str] | None = None,
    prefix_len: int = 3,
    beam_width: int = 1,
    sampling_params: SamplingParams | None = None,
    eos_token_id: int | None = None,
    client_index: int = 0,
    current_wave: int = 0,
    data_parallel_rank: int | None = None,
) -> BeamRequestStepUpdate:
    return BeamRequestStepUpdate(
        session_id=session_id,
        parent_beam_ids=parent_beam_ids if parent_beam_ids is not None else ["old-0"],
        child_beam_ids=child_beam_ids if child_beam_ids is not None else ["new-0"],
        beam_tokens=beam_tokens if beam_tokens is not None else [[50]],
        pruned_ids=pruned_ids if pruned_ids is not None else [],
        prefix_len=prefix_len,
        beam_width=beam_width,
        sampling_params=sampling_params or _sampling_params(),
        eos_token_id=eos_token_id,
        client_index=client_index,
        current_wave=current_wave,
        data_parallel_rank=data_parallel_rank,
    )


def _make_non_dp_client_mock(*, client_index: int = 0) -> MagicMock:
    mock_self = MagicMock()
    mock_self.client_index = client_index
    mock_self._send_input = AsyncMock()
    mock_self._ensure_output_queue_task = MagicMock()
    del mock_self.get_core_engine_for_request
    del mock_self.reqs_in_flight
    del mock_self.first_req_send_socket
    del mock_self.current_wave
    del mock_self.engines_running
    return mock_self


def _make_dp_client_mock(
    *,
    client_index: int = 0,
    engines_running: bool = True,
    num_engines: int = 2,
    has_current_wave: bool = False,
    current_wave: int = 0,
) -> MagicMock:
    mock_self = MagicMock()
    mock_self.client_index = client_index
    mock_self.engines_running = engines_running
    mock_self._send_input = AsyncMock()
    mock_self._ensure_output_queue_task = MagicMock()
    mock_self._ensure_stats_update_task = MagicMock()
    mock_self.first_req_send_socket = MagicMock()
    mock_self.first_req_send_socket.send = AsyncMock()
    mock_self.reqs_in_flight = {}
    engines = [f"engine-{i}" for i in range(num_engines)]
    mock_self.core_engines = engines
    mock_self.get_core_engine_for_request = MagicMock(
        side_effect=lambda req: engines[getattr(req, "data_parallel_rank", 0) or 0]
    )
    if has_current_wave:
        mock_self.current_wave = current_wave
    else:
        del mock_self.current_wave
    return mock_self


def _finished_queue() -> MagicMock:
    finished = MagicMock()
    finished.finished = True
    q = MagicMock()
    q.get = AsyncMock(return_value=finished)
    return q


def _run_subprocess(target: Any) -> None:
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=target)
    p.start()
    p.join(timeout=30)
    assert p.exitcode == 0, f"Subprocess {target.__name__} exited with code {p.exitcode}"


# ===========================================================================
# Section 1 — Types Serialization
# ===========================================================================


class TestTypesSerialization:
    def test_roundtrip_all_fields(self) -> None:
        sp = _sampling_params()
        original = BeamRequestStepUpdate(
            session_id="sess-abc",
            parent_beam_ids=["p0", "p1"],
            child_beam_ids=["c0", "c1"],
            beam_tokens=[[10, 11], [20, 21]],
            pruned_ids=["dead-0"],
            prefix_len=7,
            beam_width=2,
            sampling_params=sp,
            eos_token_id=99,
            client_index=3,
            current_wave=5,
            data_parallel_rank=1,
        )
        encoded = msgspec.msgpack.encode(original)
        decoded = msgspec.msgpack.decode(encoded, type=BeamRequestStepUpdate)

        assert decoded.session_id == "sess-abc"
        assert decoded.parent_beam_ids == ["p0", "p1"]
        assert decoded.child_beam_ids == ["c0", "c1"]
        assert decoded.beam_tokens == [[10, 11], [20, 21]]
        assert decoded.pruned_ids == ["dead-0"]
        assert decoded.prefix_len == 7
        assert decoded.beam_width == 2
        assert decoded.eos_token_id == 99
        assert decoded.client_index == 3
        assert decoded.current_wave == 5
        assert decoded.data_parallel_rank == 1
        # sampling_params must survive the round-trip intact
        assert decoded.sampling_params.max_tokens == sp.max_tokens
        assert decoded.sampling_params.logprobs == sp.logprobs
        assert decoded.sampling_params.temperature == sp.temperature
        # optional fields not explicitly set — must arrive at their defaults
        assert decoded.priority == 0
        assert decoded.lora_request is None
        assert decoded.cache_salt is None
        assert decoded.trace_headers is None

    def test_optional_fields_preserve_defaults_through_roundtrip(self) -> None:
        # All eight optional fields are omitted from the constructor call.
        # After encode → decode they must arrive at their declared defaults,
        # not as None/0 noise or missing keys.
        msg = BeamRequestStepUpdate(
            session_id="s",
            parent_beam_ids=[],
            child_beam_ids=[],
            beam_tokens=[],
            pruned_ids=[],
            prefix_len=0,
            beam_width=0,
            sampling_params=SamplingParams(),
        )
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(msg), type=BeamRequestStepUpdate)

        assert decoded.eos_token_id is None
        assert decoded.client_index == 0
        assert decoded.current_wave == 0
        assert decoded.priority == 0
        assert decoded.data_parallel_rank is None
        assert decoded.lora_request is None
        assert decoded.cache_salt is None
        assert decoded.trace_headers is None

    def test_empty_beam_tokens_survives_roundtrip(self) -> None:
        update = _make_beam_update(
            beam_width=0,
            parent_beam_ids=[],
            child_beam_ids=[],
            beam_tokens=[],
            pruned_ids=["final-0"],
        )
        encoded = msgspec.msgpack.encode(update)
        decoded = msgspec.msgpack.decode(encoded, type=BeamRequestStepUpdate)
        assert decoded.session_id == update.session_id
        assert decoded.beam_tokens == []
        assert decoded.parent_beam_ids == []
        assert decoded.child_beam_ids == []
        assert decoded.pruned_ids == ["final-0"]
        assert decoded.beam_width == 0


# ===========================================================================
# Section 2 — Enum Patching
# ===========================================================================


class TestEnumPatching:
    def test_add_enum_member_creates_member(self) -> None:
        _add_enum_member("ADD_BATCH", b"\x06")
        member = EngineCoreRequestType.ADD_BATCH
        assert member.value == b"\x06"
        assert member.name == "ADD_BATCH"
        # Must be accessible via reverse value lookup
        assert EngineCoreRequestType(b"\x06") is member
        # Must appear exactly once in the name list
        assert sum(1 for n in EngineCoreRequestType._member_names_ if n == "ADD_BATCH") == 1

    def test_add_enum_member_idempotent(self) -> None:
        _add_enum_member("ADD_BATCH", b"\x06")
        _add_enum_member("ADD_BATCH", b"\x06")
        assert EngineCoreRequestType.ADD_BATCH.value == b"\x06"
        count = sum(1 for name in EngineCoreRequestType._member_names_ if name == "ADD_BATCH")
        assert count == 1, "Duplicate member added"
        # Reverse lookup must still work after idempotent call
        assert EngineCoreRequestType(b"\x06") is EngineCoreRequestType.ADD_BATCH

    def test_beam_request_step_update_enum_value(self) -> None:
        _add_enum_member("BEAM_REQUEST_STEP_UPDATE", b"\x07")
        member = EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE
        assert member.value == b"\x07"
        assert member.name == "BEAM_REQUEST_STEP_UPDATE"
        # Must be accessible via reverse value lookup
        assert EngineCoreRequestType(b"\x07") is member
        assert (
            sum(1 for n in EngineCoreRequestType._member_names_ if n == "BEAM_REQUEST_STEP_UPDATE")
            == 1
        )

    def test_enum_lookup_by_value(self) -> None:
        _add_enum_member("ADD_BATCH", b"\x06")
        _add_enum_member("BEAM_REQUEST_STEP_UPDATE", b"\x07")
        assert EngineCoreRequestType(b"\x06") is EngineCoreRequestType.ADD_BATCH
        assert EngineCoreRequestType(b"\x07") is EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE

    def test_rejects_upstream_wire_value_collision(self) -> None:
        try:
            _add_enum_member("COLLIDES_WITH_WAKEUP", b"\x05")
            raise AssertionError("expected wire collision to be rejected")
        except RuntimeError as exc:
            assert "already used by WAKEUP" in str(exc)


# ===========================================================================
# Section 3 — EngineCore Beam Cache
# ===========================================================================


class TestBeamCache:
    def test_cache_beam_request_stores_dict(self) -> None:
        eng = _make_engine_self()
        req = _make_mock_request(request_id="r0", token_ids=[1, 2, 3], block_hashes=[0xAA, 0xBB])
        _cache_beam_request(eng, req)

        entry = eng.beam_cache["r0"]
        assert entry["all_token_ids"] == [1, 2, 3]
        assert entry["block_hashes"] == [0xAA, 0xBB]
        assert entry["eos_token_id"] == 2
        assert entry["lora_request"] is None
        assert entry["cache_salt"] is None
        assert entry["mm_features"] is None
        assert entry["prompt_embeds"] is None
        assert entry["num_prompt_tokens"] == 3

    def test_cache_beam_request_copies_lists(self) -> None:
        eng = _make_engine_self()
        original_tokens = [10, 20, 30]
        original_hashes = [0x11, 0x22]
        req = _make_mock_request(token_ids=original_tokens, block_hashes=original_hashes)
        _cache_beam_request(eng, req)

        # Mutating originals should not affect the stored copy.
        original_tokens.append(99)
        original_hashes.append(0xFF)
        entry = eng.beam_cache[req.request_id]
        assert 99 not in entry["all_token_ids"]
        assert 0xFF not in entry["block_hashes"]

    def test_cache_beam_request_overwrites_existing(self) -> None:
        eng = _make_engine_self()
        req1 = _make_mock_request(request_id="r0", token_ids=[1], block_hashes=[0x11])
        req2 = _make_mock_request(request_id="r0", token_ids=[9, 9, 9], block_hashes=[0xAA, 0xBB])
        _cache_beam_request(eng, req1)
        _cache_beam_request(eng, req2)
        entry = eng.beam_cache["r0"]
        assert entry["all_token_ids"] == [9, 9, 9]
        assert entry["block_hashes"] == [0xAA, 0xBB]
        assert entry["num_prompt_tokens"] == 3

    def test_handle_creates_single_add_request(self) -> None:
        eng = _make_engine_self(beam_cache={"session-0": _session_entry()})
        update = _make_beam_update(session_id="session-0")

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        assert not eng.input_queue.empty()
        req_type, payload = eng.input_queue.get_nowait()
        assert req_type is EngineCoreRequestType.ADD
        assert payload[0] is mock_req
        assert eng.input_queue.empty(), "Expected exactly one request"

    def test_handle_beam_token_ids_composition(self) -> None:
        prefill = [10, 20, 30]
        beam0 = [50, 51]
        beam1 = [60, 61]
        eng = _make_engine_self(beam_cache={"session-0": _session_entry(token_ids=prefill)})
        update = _make_beam_update(
            session_id="session-0",
            parent_beam_ids=["p0", "p1"],
            child_beam_ids=["c0", "c1"],
            beam_tokens=[beam0, beam1],
            beam_width=2,
        )

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        assert not eng.input_queue.empty(), "Request must be pushed to input_queue"
        _, kwargs = MockReq.call_args
        assert kwargs["prompt_token_ids"] == prefill + beam0 + beam1
        assert kwargs["request_id"] == "session-0"
        assert mock_req.is_beam is True

    def test_handle_decode_steps_from_beam_token_length(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        # 2 beams, 3 decode tokens each
        update = _make_beam_update(
            session_id="s0",
            parent_beam_ids=["p0", "p1"],
            child_beam_ids=["c0", "c1"],
            beam_tokens=[[50, 51, 52], [60, 61, 62]],
            beam_width=2,
        )

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        assert not eng.input_queue.empty()
        assert mock_req.beam_decode_steps == 3

    def test_handle_sets_meta_attributes(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        update = _make_beam_update(
            session_id="s0",
            parent_beam_ids=["p0", "p1"],
            child_beam_ids=["c0", "c1"],
            beam_tokens=[[50], [60]],
            beam_width=2,
            prefix_len=5,
        )

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        assert not eng.input_queue.empty()
        assert mock_req.is_beam is True
        assert mock_req.beam_width == 2
        assert mock_req.is_beam_decode is True
        assert mock_req.beam_decode_steps == 1  # each beam_tokens list has length 1
        assert mock_req.prefix_len == 5

    def test_handle_block_hashes_from_session(self) -> None:
        eng = _make_engine_self(
            beam_cache={"s0": _session_entry(block_hashes=[0xAA, 0xBB])},
            block_hasher=None,
        )
        update = _make_beam_update(session_id="s0")

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        assert not eng.input_queue.empty()
        assert mock_req.block_hashes == [0xAA, 0xBB]

    def test_handle_block_hasher_extends_hashes(self) -> None:
        extra_hash = [0xCC]
        hasher = MagicMock(return_value=extra_hash)
        eng = _make_engine_self(
            beam_cache={"s0": _session_entry(block_hashes=[0xAA])},
            block_hasher=hasher,
        )
        update = _make_beam_update(session_id="s0")

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            mock_req.update_block_hashes.side_effect = lambda: mock_req.block_hashes.extend(
                hasher(mock_req)
            )
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        assert not eng.input_queue.empty()
        assert mock_req.block_hashes == [0xAA, 0xCC]

    def test_handle_no_hasher_no_extension(self) -> None:
        eng = _make_engine_self(
            beam_cache={"s0": _session_entry(block_hashes=[0xAA])},
            block_hasher=None,
        )
        update = _make_beam_update(session_id="s0")

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        assert not eng.input_queue.empty()
        # block_hashes should be exactly the prefill hashes, not extended
        assert mock_req.block_hashes == [0xAA]

    def test_handle_inherits_eos_from_session(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry(eos_token_id=42)})
        update = _make_beam_update(session_id="s0", eos_token_id=None)

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        assert not eng.input_queue.empty()
        assert update.sampling_params.eos_token_id == 42

    def test_handle_overrides_eos_from_update(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry(eos_token_id=42)})
        update = _make_beam_update(session_id="s0", eos_token_id=99)

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        assert not eng.input_queue.empty()
        assert update.sampling_params.eos_token_id == 99

    def test_handle_missing_session_emits_error_outputs(self) -> None:
        eng = _make_engine_self(beam_cache={})
        update = _make_beam_update(
            session_id="unknown",
            child_beam_ids=["c0", "c1"],
            parent_beam_ids=["p0", "p1"],
            beam_tokens=[[50], [60]],
            beam_width=2,
        )

        with patch("vllm.v1.request.Request"):
            _handle_beam_request_step_update(eng, update)

        assert eng.input_queue.empty()
        assert eng.output_queue.qsize() == 2
        error_req_ids = set()
        for _ in range(2):
            _client_idx, outputs = eng.output_queue.get_nowait()
            assert len(outputs.outputs) == 1
            assert outputs.outputs[0].finish_reason is FinishReason.ERROR
            error_req_ids.add(outputs.outputs[0].request_id)
        assert error_req_ids == {"c0", "c1"}

    def test_handle_length_mismatch_emits_error_outputs(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        update = _make_beam_update(
            session_id="s0",
            parent_beam_ids=["p0", "p1"],  # 2 parents
            child_beam_ids=["c0", "c1"],  # 2 children
            beam_tokens=[[50]],  # only 1 token list — mismatch!
            beam_width=2,
        )

        _handle_beam_request_step_update(eng, update)

        assert eng.input_queue.empty()
        assert eng.output_queue.qsize() == 2
        error_req_ids = set()
        for _ in range(2):
            _client_idx, outputs = eng.output_queue.get_nowait()
            assert outputs.outputs[0].finish_reason is FinishReason.ERROR
            error_req_ids.add(outputs.outputs[0].request_id)
        assert error_req_ids == {"c0", "c1"}

    def test_handle_parent_count_mismatch_emits_error_outputs(self) -> None:
        # beam_width=2 but parent_beam_ids has 3 entries → mismatch
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        update = _make_beam_update(
            session_id="s0",
            parent_beam_ids=["p0", "p1", "p2"],  # 3 parents
            child_beam_ids=["c0", "c1"],
            beam_tokens=[[50], [60]],
            beam_width=2,
        )

        _handle_beam_request_step_update(eng, update)

        assert eng.input_queue.empty()
        assert eng.output_queue.qsize() == 2  # one error per child
        error_req_ids = set()
        for _ in range(2):
            _client_idx, outputs = eng.output_queue.get_nowait()
            assert outputs.outputs[0].finish_reason is FinishReason.ERROR
            error_req_ids.add(outputs.outputs[0].request_id)
        assert error_req_ids == {"c0", "c1"}

    def test_handle_beam_width_child_count_mismatch_emits_error_outputs(self) -> None:
        # beam_width=3 but only 2 children and 2 token lists → mismatch
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        update = _make_beam_update(
            session_id="s0",
            parent_beam_ids=["p0", "p1"],
            child_beam_ids=["c0", "c1"],
            beam_tokens=[[50], [60]],
            beam_width=3,  # claims 3 beams but arrays have 2
        )

        _handle_beam_request_step_update(eng, update)

        assert eng.input_queue.empty()
        assert eng.output_queue.qsize() == 2  # one error per child
        error_req_ids = set()
        for _ in range(2):
            _client_idx, outputs = eng.output_queue.get_nowait()
            assert outputs.outputs[0].finish_reason is FinishReason.ERROR
            error_req_ids.add(outputs.outputs[0].request_id)
        assert error_req_ids == {"c0", "c1"}

    def test_handle_b0_removes_session_from_cache(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        update = _make_beam_update(
            session_id="s0",
            beam_width=0,
            pruned_ids=[],
        )
        _handle_beam_request_step_update(eng, update)
        assert "s0" not in eng.beam_cache
        assert eng.input_queue.empty()

    def test_handle_b0_puts_pruned_ids_in_aborts(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        update = _make_beam_update(
            session_id="s0",
            beam_width=0,
            pruned_ids=["final-0", "final-1"],
        )
        _handle_beam_request_step_update(eng, update)
        assert eng.input_queue.empty()
        aborted = set()
        while not eng.aborts_queue.empty():
            aborted.add(eng.aborts_queue.get_nowait())
        assert aborted == {"s0", "final-0", "final-1"}

    def test_handle_pruned_ids_go_to_aborts_session_stays(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        update = _make_beam_update(
            session_id="s0",
            parent_beam_ids=["p0"],
            child_beam_ids=["c0"],
            beam_tokens=[[50]],
            pruned_ids=["dead-0"],
            beam_width=1,
        )

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        assert not eng.input_queue.empty(), "Normal step must push request to input_queue"
        # Session must still be in cache (only B=0 removes it)
        assert "s0" in eng.beam_cache
        # Pruned IDs go to aborts_queue; session_id does NOT (only B=0 aborts session)
        aborted = eng.aborts_queue.get_nowait()
        assert aborted == "dead-0"
        assert eng.aborts_queue.empty(), "Only the pruned ID must be aborted, not the session"

    def test_handle_session_evolves_across_steps(self) -> None:
        """Three-step lifecycle: step → step → B=0 teardown."""
        session_tokens = [10, 20, 30]
        eng = _make_engine_self(beam_cache={"s0": _session_entry(token_ids=session_tokens)})

        def _run_step(update):
            with patch("vllm.v1.request.Request") as MockReq:
                mock_req = MagicMock()
                mock_req.block_hashes = []
                MockReq.return_value = mock_req
                _handle_beam_request_step_update(eng, update)
            return MockReq

        # Step 1: first decode step, both beams survive
        MockReq1 = _run_step(
            _make_beam_update(
                session_id="s0",
                parent_beam_ids=["parent-0", "parent-1"],
                child_beam_ids=["child-1a", "child-1b"],
                beam_tokens=[[40], [50]],
                beam_width=2,
            )
        )
        assert "s0" in eng.beam_cache, "Session must survive step 1"
        assert eng.input_queue.qsize() == 1
        _, kwargs1 = MockReq1.call_args
        assert kwargs1["prompt_token_ids"][:3] == session_tokens  # prefill reused

        # Step 2: second decode step, same session, same prefill
        MockReq2 = _run_step(
            _make_beam_update(
                session_id="s0",
                parent_beam_ids=["child-1a", "child-1b"],
                child_beam_ids=["child-2a", "child-2b"],
                beam_tokens=[[40, 60], [50, 70]],
                beam_width=2,
            )
        )
        assert "s0" in eng.beam_cache, "Session must survive step 2"
        assert eng.input_queue.qsize() == 2
        _, kwargs2 = MockReq2.call_args
        assert kwargs2["prompt_token_ids"][:3] == session_tokens  # same prefill reused

        # Step 3: teardown — session removed, all ids aborted
        _handle_beam_request_step_update(
            eng,
            _make_beam_update(
                session_id="s0",
                beam_width=0,
                pruned_ids=["child-2a", "child-2b"],
            ),
        )
        assert "s0" not in eng.beam_cache
        aborted = set()
        while not eng.aborts_queue.empty():
            aborted.add(eng.aborts_queue.get_nowait())
        assert {"s0", "child-2a", "child-2b"} == aborted

    def test_handle_current_wave_stored_in_queue_payload(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        update = _make_beam_update(session_id="s0", current_wave=7)

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        _, payload = eng.input_queue.get_nowait()
        _req, wave = payload
        assert wave == 7


# ===========================================================================
# Section 4 — Client-Side Patches
# ===========================================================================


class TestClientPatches:
    # ── add_requests_async ──────────────────────────────────────────────────

    def test_add_requests_async_empty_is_noop(self) -> None:
        mock_self = _make_non_dp_client_mock()
        asyncio.run(add_requests_async(mock_self, []))
        mock_self._send_input.assert_not_awaited()
        mock_self._ensure_output_queue_task.assert_not_called()

    def test_add_requests_async_single_no_force_delegates(self) -> None:
        mock_self = _make_non_dp_client_mock()
        mock_self.add_request_async = AsyncMock()
        req = MagicMock()
        asyncio.run(add_requests_async(mock_self, [req], force_batch=False))
        mock_self.add_request_async.assert_awaited_once_with(req)

    def test_add_requests_async_single_force_sends_add_batch(self) -> None:
        mock_self = _make_non_dp_client_mock()
        req = MagicMock()
        req.client_index = 0
        asyncio.run(add_requests_async(mock_self, [req], force_batch=True))
        mock_self._send_input.assert_awaited_once()
        assert mock_self._send_input.call_args[0][0] is EngineCoreRequestType.ADD_BATCH

    def test_add_requests_async_multiple_sends_add_batch(self) -> None:
        mock_self = _make_non_dp_client_mock()
        reqs = [MagicMock(), MagicMock()]
        for r in reqs:
            r.client_index = 0
        asyncio.run(add_requests_async(mock_self, reqs))
        mock_self._send_input.assert_awaited_once()
        assert mock_self._send_input.call_args[0][0] is EngineCoreRequestType.ADD_BATCH

    # ── beam_request_step_update_async ────────────────────────────────────

    def test_beam_request_step_update_async_sets_client_index(self) -> None:
        mock_self = _make_non_dp_client_mock(client_index=7)
        update = _make_beam_update(client_index=0)
        asyncio.run(beam_request_step_update_async(mock_self, update))
        assert update.client_index == 7
        # Must still have dispatched to the engine
        mock_self._send_input.assert_awaited_once()
        assert (
            mock_self._send_input.call_args[0][0] is EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE
        )

    def test_beam_request_step_update_async_sends_correct_type(self) -> None:
        mock_self = _make_non_dp_client_mock()
        update = _make_beam_update()
        asyncio.run(beam_request_step_update_async(mock_self, update))
        mock_self._send_input.assert_awaited_once()
        assert (
            mock_self._send_input.call_args[0][0] is EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE
        )
        assert mock_self._send_input.call_args[0][1] is update
        mock_self._ensure_output_queue_task.assert_called_once()

    def test_beam_request_step_update_async_sets_current_wave(self) -> None:
        mock_self = _make_non_dp_client_mock()
        mock_self.current_wave = 3
        update = _make_beam_update(current_wave=0)
        asyncio.run(beam_request_step_update_async(mock_self, update))
        assert update.current_wave == 3

    def test_beam_request_step_update_async_no_wave_if_missing(self) -> None:
        mock_self = _make_non_dp_client_mock()
        update = _make_beam_update(current_wave=42)
        asyncio.run(beam_request_step_update_async(mock_self, update))
        # Without current_wave attribute, the update's value is left as-is
        assert update.current_wave == 42

    # ── register_beam_output_fn ──────────────────────────────────────────

    def test_register_beam_output_sets_external_req_id(self) -> None:
        mock_self = MagicMock()
        mock_self.errored = False
        mock_self.log_requests = False

        register_beam_output_fn(
            mock_self,
            request_id="child-beam-7",
            prompt_token_ids=[10, 20],
            sampling_params=_sampling_params(),
        )

        mock_self.output_processor.add_request.assert_called_once()
        ec_req = mock_self.output_processor.add_request.call_args[0][0]
        assert ec_req.external_req_id == "child-beam-7"
        assert ec_req.data_parallel_rank is None  # not passed → default

    def test_register_beam_output_passes_data_parallel_rank(self) -> None:
        mock_self = MagicMock()
        mock_self.errored = False
        mock_self.log_requests = False

        register_beam_output_fn(
            mock_self,
            request_id="child-0",
            prompt_token_ids=[1, 2],
            sampling_params=_sampling_params(),
            data_parallel_rank=3,
        )

        mock_self.output_processor.add_request.assert_called_once()
        ec_req = mock_self.output_processor.add_request.call_args[0][0]
        assert ec_req.data_parallel_rank == 3
        assert ec_req.external_req_id == "child-0"

    # ── _add_requests_batch_fn ───────────────────────────────────────────

    def test_add_requests_batch_fn_delegates(self) -> None:
        mock_self = MagicMock()
        mock_self.engine_core.add_requests_async = AsyncMock()
        reqs = [MagicMock(), MagicMock()]

        asyncio.run(_add_requests_batch_fn(mock_self, reqs, use_batch_message=True))

        mock_self.engine_core.add_requests_async.assert_awaited_once_with(reqs, force_batch=True)

    def test_add_requests_batch_fn_no_force(self) -> None:
        mock_self = MagicMock()
        mock_self.engine_core.add_requests_async = AsyncMock()
        reqs = [MagicMock()]

        asyncio.run(_add_requests_batch_fn(mock_self, reqs, use_batch_message=False))

        mock_self.engine_core.add_requests_async.assert_awaited_once_with(reqs, force_batch=False)

    # ── beam_request_step_update_fn ──────────────────────────────────────

    def test_beam_request_step_update_fn_delegates(self) -> None:
        mock_self = MagicMock()
        mock_self.engine_core.beam_request_step_update_async = AsyncMock()
        update = _make_beam_update()

        asyncio.run(beam_request_step_update_fn(mock_self, update))

        mock_self.engine_core.beam_request_step_update_async.assert_awaited_once_with(update)

    # ── SyncMPClient patches ─────────────────────────────────────────────

    def test_sync_mp_add_requests_empty_is_noop(self) -> None:
        mock_self = MagicMock()
        sync_mp_add_requests(mock_self, [])
        mock_self._send_input.assert_not_called()
        mock_self.add_request.assert_not_called()

    def test_sync_mp_add_requests_single_no_force_delegates(self) -> None:
        mock_self = MagicMock()
        req = MagicMock()
        sync_mp_add_requests(mock_self, [req], force_batch=False)
        mock_self.add_request.assert_called_once_with(req)
        mock_self._send_input.assert_not_called()

    def test_sync_mp_add_requests_single_force_sends_add_batch(self) -> None:
        mock_self = MagicMock()
        mock_self.is_dp = False
        req = MagicMock()
        sync_mp_add_requests(mock_self, [req], force_batch=True)
        mock_self._send_input.assert_called_once()
        assert mock_self._send_input.call_args[0][0] is EngineCoreRequestType.ADD_BATCH

    def test_sync_mp_add_requests_multiple_sends_add_batch(self) -> None:
        mock_self = MagicMock()
        mock_self.is_dp = False
        reqs = [MagicMock(), MagicMock()]
        sync_mp_add_requests(mock_self, reqs)
        mock_self._send_input.assert_called_once()
        assert mock_self._send_input.call_args[0][0] is EngineCoreRequestType.ADD_BATCH

    def test_sync_mp_beam_request_step_update_sends_correct_type(self) -> None:
        mock_self = MagicMock()
        update = _make_beam_update()
        sync_mp_beam_request_step_update(mock_self, update)
        mock_self._send_input.assert_called_once()
        assert (
            mock_self._send_input.call_args[0][0] is EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE
        )
        assert mock_self._send_input.call_args[0][1] is update

    # ── InprocClient patches ─────────────────────────────────────────────

    def test_inproc_add_requests_empty_is_noop(self) -> None:
        mock_self = MagicMock()
        inproc_add_requests(mock_self, [])
        mock_self.add_request.assert_not_called()
        mock_self.engine_core.preprocess_add_request.assert_not_called()

    def test_inproc_add_requests_single_no_force_delegates(self) -> None:
        mock_self = MagicMock()
        req = MagicMock()
        inproc_add_requests(mock_self, [req], force_batch=False)
        mock_self.add_request.assert_called_once_with(req)

    def test_inproc_add_requests_multiple_caches_and_queues(self) -> None:
        inproc_q = queue.SimpleQueue()
        mock_core = MagicMock()
        mock_core.input_queue = inproc_q
        preprocessed = (MagicMock(), "extra")
        mock_core.preprocess_add_request.return_value = preprocessed

        mock_self = MagicMock()
        mock_self.engine_core = mock_core

        reqs = [MagicMock(), MagicMock()]
        inproc_add_requests(mock_self, reqs)  # multiple requests → always batch path

        assert mock_core._cache_beam_request.call_count == 2
        assert inproc_q.qsize() == 2
        # Each queue entry must be (ADD, preprocessed_tuple)
        for _ in range(2):
            req_type, payload = inproc_q.get_nowait()
            assert req_type is EngineCoreRequestType.ADD
            assert payload is preprocessed

    def test_inproc_add_requests_preproc_error_skips_bad_request(self) -> None:
        inproc_q = queue.SimpleQueue()
        mock_core = MagicMock()
        mock_core.input_queue = inproc_q

        good_preprocessed = (MagicMock(), "extra")
        mock_core.preprocess_add_request.side_effect = [
            RuntimeError("bad input"),  # first request fails
            good_preprocessed,  # second request succeeds
        ]

        mock_self = MagicMock()
        mock_self.engine_core = mock_core

        reqs = [MagicMock(), MagicMock()]
        inproc_add_requests(mock_self, reqs)

        # Error handler called for the failing request
        mock_core._handle_request_preproc_error.assert_called_once_with(reqs[0])
        # Only the successful request was queued, with the correct type and payload
        assert inproc_q.qsize() == 1
        req_type, payload = inproc_q.get_nowait()
        assert req_type is EngineCoreRequestType.ADD
        assert payload is good_preprocessed

    def test_inproc_beam_request_step_update_delegates(self) -> None:
        mock_self = MagicMock()
        update = _make_beam_update()
        inproc_beam_request_step_update(mock_self, update)
        mock_self.engine_core._handle_beam_request_step_update.assert_called_once_with(update)

    # ── apply_batch_fork_patches ─────────────────────────────────────────

    def test_apply_batch_fork_patches_adds_enum_members(self) -> None:
        apply_batch_fork_patches()
        assert hasattr(EngineCoreRequestType, "ADD_BATCH")
        assert hasattr(EngineCoreRequestType, "BEAM_REQUEST_STEP_UPDATE")
        assert EngineCoreRequestType.ADD_BATCH.value == b"\x06"
        assert EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE.value == b"\x07"

    def test_apply_batch_fork_patches_four_classes(self) -> None:
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient

        apply_batch_fork_patches()

        assert AsyncMPClient.add_requests_async is add_requests_async
        assert AsyncMPClient.beam_request_step_update_async is beam_request_step_update_async
        assert AsyncLLM.prepare_request is prepare_request_fn
        assert AsyncLLM._add_requests_batch is _add_requests_batch_fn
        assert AsyncLLM.register_beam_output is register_beam_output_fn
        assert AsyncLLM.beam_request_step_update is beam_request_step_update_fn
        assert SyncMPClient.add_requests is sync_mp_add_requests
        assert SyncMPClient.beam_request_step_update is sync_mp_beam_request_step_update
        assert InprocClient.add_requests is inproc_add_requests
        assert InprocClient.beam_request_step_update is inproc_beam_request_step_update

    def test_apply_batch_fork_patches_patches_engine_core_for_inproc(self) -> None:
        from vllm.v1.engine.core import EngineCore

        apply_batch_fork_patches()

        assert hasattr(EngineCore, "_cache_beam_request")
        assert hasattr(EngineCore, "_handle_beam_request_step_update")
        assert getattr(EngineCore, "_patched_for_beam_requests", False) is True


# ===========================================================================
# Section 5 — Beam Search Orchestration
# ===========================================================================


class TestBeamSearchOrchestration:
    def test_use_beam_request_true_when_client_has_method(self) -> None:
        ec = MagicMock(spec=["beam_request_step_update"])
        assert hasattr(ec, "beam_request_step_update")

    def test_use_beam_request_false_when_method_absent(self) -> None:
        ec = MagicMock(spec=[])
        assert not hasattr(ec, "beam_request_step_update")

    def test_beam_request_step_builds_correct_update(self) -> None:
        ec = MagicMock()
        ec.register_beam_output = MagicMock(return_value=_finished_queue())
        ec.beam_request_step_update = AsyncMock()

        all_beams = [
            SimpleNamespace(tokens=[10, 20, 50]),
            SimpleNamespace(tokens=[10, 20, 60]),
        ]
        fork_info = [(0, 50), (1, 60)]

        asyncio.run(
            _beam_request_step(
                ec,
                session_id="sess-0",
                fork_info=fork_info,
                prev_beam_internal_ids=["prev-0", "prev-1"],
                all_beams=all_beams,
                request_id_batch="batch-1",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                prefix_len=2,
                priority=0,
                data_parallel_rank=None,
            )
        )

        ec.beam_request_step_update.assert_awaited_once()
        update: BeamRequestStepUpdate = ec.beam_request_step_update.call_args[0][0]
        assert isinstance(update, BeamRequestStepUpdate)
        assert update.session_id == "sess-0"
        assert update.beam_width == 2
        assert update.prefix_len == 2
        assert update.parent_beam_ids == ["prev-0", "prev-1"]
        assert update.child_beam_ids == ["batch-1-beam-0", "batch-1-beam-1"]
        assert update.beam_tokens == [[50], [60]]  # full_tokens[prefix_len:]
        # Both parents used → nothing pruned
        assert update.pruned_ids == []
        assert update.eos_token_id == 2
        assert update.data_parallel_rank is None
        assert update.sampling_params.n == 2  # W beams

    def test_beam_request_step_beam_tokens_are_delta_from_prefix(self) -> None:
        ec = MagicMock()
        ec.register_beam_output = MagicMock(return_value=_finished_queue())
        ec.beam_request_step_update = AsyncMock()

        all_beams = [SimpleNamespace(tokens=[1, 2, 3, 50, 51])]
        asyncio.run(
            _beam_request_step(
                ec,
                session_id="s",
                fork_info=[(0, 50)],
                prev_beam_internal_ids=["p0"],
                all_beams=all_beams,
                request_id_batch="r",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                prefix_len=3,  # strip first 3 tokens
                priority=0,
                data_parallel_rank=None,
            )
        )

        update = ec.beam_request_step_update.call_args[0][0]
        assert update.session_id == "s"
        assert update.beam_width == 1
        assert update.prefix_len == 3
        assert update.parent_beam_ids == ["p0"]
        assert update.child_beam_ids == ["r-beam-0"]
        assert update.beam_tokens == [[50, 51]]
        assert update.pruned_ids == []

    def test_beam_request_step_pruned_ids_are_unused_parents(self) -> None:
        ec = MagicMock()
        ec.register_beam_output = MagicMock(return_value=_finished_queue())
        ec.beam_request_step_update = AsyncMock()

        # Both new beams expand from parent 0 → parent 1 is pruned
        all_beams = [
            SimpleNamespace(tokens=[10, 50]),
            SimpleNamespace(tokens=[10, 60]),
        ]
        asyncio.run(
            _beam_request_step(
                ec,
                session_id="s",
                fork_info=[(0, 50), (0, 60)],
                prev_beam_internal_ids=["parent-0", "parent-1"],
                all_beams=all_beams,
                request_id_batch="r",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                prefix_len=1,
                priority=0,
                data_parallel_rank=None,
            )
        )

        update = ec.beam_request_step_update.call_args[0][0]
        assert update.session_id == "s"
        assert update.beam_width == 2
        assert update.parent_beam_ids == ["parent-0", "parent-0"]
        assert update.child_beam_ids == ["r-beam-0", "r-beam-1"]
        assert update.beam_tokens == [[50], [60]]
        assert update.pruned_ids == ["parent-1"]

    def test_beam_request_step_calls_register_beam_output(self) -> None:
        ec = MagicMock()
        ec.register_beam_output = MagicMock(return_value=_finished_queue())
        ec.beam_request_step_update = AsyncMock()

        all_beams = [SimpleNamespace(tokens=[10, 50])]
        asyncio.run(
            _beam_request_step(
                ec,
                session_id="sess-1",
                fork_info=[(0, 50)],
                prev_beam_internal_ids=["p0"],
                all_beams=all_beams,
                request_id_batch="r",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                prefix_len=1,
                priority=0,
                data_parallel_rank=None,
            )
        )

        ec.register_beam_output.assert_called_once()
        reg_args, reg_kwargs = ec.register_beam_output.call_args
        assert reg_args[0] == "sess-1"  # session_id
        assert reg_args[1] == [10, 50]  # all_beams[0].tokens
        assert reg_kwargs["data_parallel_rank"] is None

    def test_beam_request_step_returns_single_output_list(self) -> None:
        ec = MagicMock()
        ec.register_beam_output = MagicMock(return_value=_finished_queue())
        ec.beam_request_step_update = AsyncMock()

        all_beams = [SimpleNamespace(tokens=[10, 50])]
        output, child_ids = asyncio.run(
            _beam_request_step(
                ec,
                session_id="s",
                fork_info=[(0, 50)],
                prev_beam_internal_ids=["p0"],
                all_beams=all_beams,
                request_id_batch="r",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                prefix_len=1,
                priority=0,
                data_parallel_rank=None,
            )
        )

        assert len(output) == 1
        assert output[0].finished is True
        assert child_ids == ["r-beam-0"]

    def test_beam_request_step_returns_correct_child_ids(self) -> None:
        ec = MagicMock()
        ec.register_beam_output = MagicMock(return_value=_finished_queue())
        ec.beam_request_step_update = AsyncMock()

        all_beams = [
            SimpleNamespace(tokens=[10, 50]),
            SimpleNamespace(tokens=[10, 60]),
        ]
        _, child_ids = asyncio.run(
            _beam_request_step(
                ec,
                session_id="s",
                fork_info=[(0, 50), (1, 60)],
                prev_beam_internal_ids=["p0", "p1"],
                all_beams=all_beams,
                request_id_batch="mybatch",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                prefix_len=1,
                priority=0,
                data_parallel_rank=None,
            )
        )

        assert child_ids == ["mybatch-beam-0", "mybatch-beam-1"]

    def test_beam_request_step_passes_data_parallel_rank(self) -> None:
        ec = MagicMock()
        ec.register_beam_output = MagicMock(return_value=_finished_queue())
        ec.beam_request_step_update = AsyncMock()

        all_beams = [SimpleNamespace(tokens=[10, 50])]
        asyncio.run(
            _beam_request_step(
                ec,
                session_id="s",
                fork_info=[(0, 50)],
                prev_beam_internal_ids=["p0"],
                all_beams=all_beams,
                request_id_batch="r",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                prefix_len=1,
                priority=0,
                data_parallel_rank=2,
            )
        )

        update = ec.beam_request_step_update.call_args[0][0]
        assert update.session_id == "s"
        assert update.beam_width == 1
        assert update.data_parallel_rank == 2
        # register_beam_output must also receive the rank
        reg_args, reg_kwargs = ec.register_beam_output.call_args
        assert reg_kwargs["data_parallel_rank"] == 2

    def test_single_protocol_message_per_decode_step(self) -> None:
        """Key BeamRequest optimization: all beams in a decode step are sent
        as exactly one protocol message, not one per beam."""
        ec = MagicMock()
        ec.register_beam_output = MagicMock(return_value=_finished_queue())
        ec.beam_request_step_update = AsyncMock()

        all_beams = [
            SimpleNamespace(tokens=[10, 50]),
            SimpleNamespace(tokens=[10, 60]),
            SimpleNamespace(tokens=[10, 70]),
        ]
        asyncio.run(
            _beam_request_step(
                ec,
                session_id="s",
                fork_info=[(0, 50), (1, 60), (2, 70)],
                prev_beam_internal_ids=["p0", "p1", "p2"],
                all_beams=all_beams,
                request_id_batch="batch",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                prefix_len=1,
                priority=0,
                data_parallel_rank=None,
            )
        )

        # All 3 beams packed into exactly one protocol message
        ec.beam_request_step_update.assert_awaited_once()
        update = ec.beam_request_step_update.call_args[0][0]
        assert update.session_id == "s"
        assert update.beam_width == 3
        assert update.parent_beam_ids == ["p0", "p1", "p2"]
        assert update.child_beam_ids == ["batch-beam-0", "batch-beam-1", "batch-beam-2"]
        assert update.beam_tokens == [[50], [60], [70]]
        assert update.pruned_ids == []

    def test_beam_request_cleanup_sends_b0(self) -> None:
        ec = MagicMock()
        ec.beam_request_step_update = AsyncMock()

        asyncio.run(_beam_request_cleanup(ec, "sess-0", ["old-0", "old-1"], rank=None))

        ec.beam_request_step_update.assert_awaited_once()
        update: BeamRequestStepUpdate = ec.beam_request_step_update.call_args[0][0]
        assert isinstance(update, BeamRequestStepUpdate)
        assert update.session_id == "sess-0"
        assert update.beam_width == 0
        assert update.pruned_ids == ["old-0", "old-1"]
        assert update.parent_beam_ids == []
        assert update.child_beam_ids == []
        assert update.beam_tokens == []
        assert update.prefix_len == 0
        assert update.data_parallel_rank is None

    def test_beam_request_cleanup_passes_rank(self) -> None:
        ec = MagicMock()
        ec.beam_request_step_update = AsyncMock()

        asyncio.run(_beam_request_cleanup(ec, "sess-r", ["final-x"], rank=3))

        ec.beam_request_step_update.assert_awaited_once()
        update = ec.beam_request_step_update.call_args[0][0]
        assert update.session_id == "sess-r"
        assert update.beam_width == 0
        assert update.pruned_ids == ["final-x"]
        assert update.parent_beam_ids == []
        assert update.child_beam_ids == []
        assert update.beam_tokens == []
        assert update.prefix_len == 0
        assert update.data_parallel_rank == 3

    def test_beam_request_cleanup_noop_when_empty_final_ids(self) -> None:
        ec = MagicMock()
        ec.beam_request_step_update = AsyncMock()

        asyncio.run(_beam_request_cleanup(ec, "s", [], rank=None))

        ec.beam_request_step_update.assert_not_awaited()

    def test_beam_request_cleanup_swallows_exception(self) -> None:
        ec = MagicMock()
        ec.beam_request_step_update = AsyncMock(side_effect=RuntimeError("net error"))
        # Should not raise
        asyncio.run(_beam_request_cleanup(ec, "s", ["x"], rank=None))


# ===========================================================================
# Section 6 — Patch Wiring
# ===========================================================================


class TestPatchWiring:
    def test_patch_batch_fork_registers_beam_request_step_update(self) -> None:
        _run_subprocess(_sub_patch_batch_fork_registers_enum)

    def test_patch_batch_fork_wires_run_engine_core(self) -> None:
        _run_subprocess(_sub_patch_batch_fork_wires_run_engine_core)

    def test_patch_batch_fork_idempotent(self) -> None:
        _run_subprocess(_sub_patch_batch_fork_idempotent)


# ===========================================================================
# Section 7 — Integration / Cross-Process
# ===========================================================================


class TestIntegrationCrossProcess:
    def test_enum_values_no_collision(self) -> None:
        _add_enum_member("ADD_BATCH", b"\x06")
        _add_enum_member("BEAM_REQUEST_STEP_UPDATE", b"\x07")

        assert EngineCoreRequestType.ADD_BATCH.value == b"\x06"
        assert EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE.value == b"\x07"

        # Our two values must not collide with any upstream member
        for name in EngineCoreRequestType._member_names_:
            if name in ("ADD_BATCH", "BEAM_REQUEST_STEP_UPDATE"):
                continue
            member = getattr(EngineCoreRequestType, name)
            assert member.value not in (b"\x06", b"\x07"), (
                f"Collision: upstream member {name!r} already uses value {member.value!r}"
            )

    def test_beam_request_step_update_encode_decode(self) -> None:
        from vllm.v1.serial_utils import MsgpackDecoder

        sp = _sampling_params()
        original = BeamRequestStepUpdate(
            session_id="session-xyz",
            parent_beam_ids=["p0", "p1"],
            child_beam_ids=["c0", "c1"],
            beam_tokens=[[100, 200], [300, 400]],
            pruned_ids=["old-0"],
            prefix_len=5,
            beam_width=2,
            sampling_params=sp,
            eos_token_id=2,
            data_parallel_rank=1,
        )
        data = msgspec.msgpack.encode(original)
        decoder = MsgpackDecoder(BeamRequestStepUpdate)
        decoded = decoder.decode(data)

        assert decoded.session_id == "session-xyz"
        assert decoded.parent_beam_ids == ["p0", "p1"]
        assert decoded.child_beam_ids == ["c0", "c1"]
        assert decoded.beam_tokens == [[100, 200], [300, 400]]
        assert decoded.pruned_ids == ["old-0"]
        assert decoded.prefix_len == 5
        assert decoded.beam_width == 2
        assert decoded.eos_token_id == 2
        assert decoded.data_parallel_rank == 1

    def test_beam_request_step_update_wire_byte_is_x07(self) -> None:
        _add_enum_member("BEAM_REQUEST_STEP_UPDATE", b"\x07")
        assert EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE.value == b"\x07"

    def test_add_batch_wire_byte_is_x06(self) -> None:
        _add_enum_member("ADD_BATCH", b"\x06")
        assert EngineCoreRequestType.ADD_BATCH.value == b"\x06"
