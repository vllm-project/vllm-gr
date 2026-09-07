# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-any-return"
"""Unit tests for EngineCore engine-driven chaining.

Covers the chain that removes the per-step frontend RPC round-trip:

- ``_apply_beam_step`` records ``running_suffix`` so the engine can replay
  the routing fields on internally chained steps.
- ``_beam_advance_from_decision`` builds the next BeamRequestStepUpdate
  from the worker's beam decision (host tensors from
  ``kv_transfer_params['beam_search_result']``) and applies it in-process.
- ``_beam_self_advance`` scans a step's EngineCoreOutputs for the decision
  and chains every independent session decision in that engine iteration.
- ``make_patched_process_engine_step`` runs the hook between ``step_fn()``
  and the output-queue forward, with an exception fallback that keeps the
  frontend step-update loop usable.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)

import vllm_gr.v1.engine.core as core_module
from vllm_gr.entrypoints.openai.serving_engine import _beam_request_step
from vllm_gr.v1.beam import BeamAttentionContext, beam_decision
from vllm_gr.v1.engine.core import (
    _beam_advance_from_decision,
    _beam_self_advance,
    _handle_beam_request_step_update,
)
from vllm_gr.v1.engine.engine_core_patch import make_patched_process_engine_step
from vllm_gr.v1.engine.scheduler_metadata import build_beam_request_metadata
from vllm_gr.v1.engine.types import BeamRequestStepUpdate
from vllm_gr.v1.worker import model_runner_common

# ---------------------------------------------------------------------------
# Shared test helpers (mirrors test_beam_request_unit.py)
# ---------------------------------------------------------------------------


def _sampling_params(**overrides: Any) -> SamplingParams:
    defaults = {"max_tokens": 4, "logprobs": 2, "temperature": 0.0, "detokenize": False}
    defaults.update(overrides)
    return SamplingParams(**defaults)


def _make_engine_self(
    beam_cache: dict[str, Any] | None = None,
    block_hasher: Any | None = None,
    engine_index: int = 0,
    self_advance: bool = True,
) -> SimpleNamespace:
    eng = SimpleNamespace(
        beam_cache=beam_cache if beam_cache is not None else {},
        beam_cache_lock=threading.Lock(),
        request_block_hasher=block_hasher,
        input_queue=queue.SimpleQueue(),
        output_queue=queue.SimpleQueue(),
        aborts_queue=queue.SimpleQueue(),
        engine_index=engine_index,
    )
    # Pre-seed the per-engine gate cache so tests do not depend on the
    # process-global vLLM config (the read path has its own test below).
    eng._vllm_gr_self_advance = self_advance
    return eng


def _session_entry(
    token_ids: list[int] | None = None,
    block_hashes: list[int] | None = None,
    eos_token_id: int = 2,
) -> dict[str, Any]:
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


def _running_suffix(
    beam_tokens: list[list[int]] | None = None,
    *,
    beam_width: int = 2,
    prefix_len: int = 3,
    sampling_params: SamplingParams | None = None,
    eos_token_id: int = 2,
    client_index: int = 5,
    current_wave: int = 3,
    priority: int = 2,
    data_parallel_rank: int | None = 1,
    max_decode_steps: int = 7,
    is_constrained: bool = False,
) -> dict[str, Any]:
    return {
        "beam_tokens": beam_tokens if beam_tokens is not None else [[50], [60]],
        "prefix_len": prefix_len,
        "beam_width": beam_width,
        "sampling_params": sampling_params or _sampling_params(),
        "eos_token_id": eos_token_id,
        "client_index": client_index,
        "current_wave": current_wave,
        "priority": priority,
        "data_parallel_rank": data_parallel_rank,
        "lora_request": None,
        "cache_salt": None,
        "trace_headers": None,
        "max_decode_steps": max_decode_steps,
        "is_constrained": is_constrained,
    }


def _seeded_engine(session_id: str = "s0", **running_kwargs: Any) -> SimpleNamespace:
    entry = _session_entry()
    entry["running_suffix"] = _running_suffix(**running_kwargs)
    return _make_engine_self(beam_cache={session_id: entry})


def _decision(
    parents: list[int],
    tokens: list[int],
    *,
    is_last_step: bool = False,
) -> dict[str, Any]:
    """Worker decision dict as it arrives via kv_transfer_params.

    Mirrors the return-channel normalization in
    ``_maybe_run_worker_beam_decision``: nested Python lists, never tensors —
    kv_transfer_params values are decoded against ``dict[str, Any]`` and only
    plain containers round-trip the msgpack IPC boundary.
    """
    decision: dict[str, Any] = {
        "out_token_ids": [tokens],
        "out_token_index": [parents],
        "out_log_probs": [[0.0] * len(tokens)],
        "is_last_step": is_last_step,
    }
    if is_last_step:
        decision["out_sequence"] = [[[0] * (len(tokens) + 1)] * len(tokens)]
    return decision


def _outputs_with_decision(
    decision: Any,
    request_id: str = "s0",
    client_index: int = 0,
) -> dict[int, EngineCoreOutputs]:
    return {
        client_index: EngineCoreOutputs(
            engine_index=0,
            outputs=[
                EngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=[],
                    kv_transfer_params={"beam_search_result": decision},
                )
            ],
        )
    }


# ===========================================================================
# Composed lifecycle continuity
# ===========================================================================


@pytest.mark.parametrize("transport", ["list", "compact"])
def test_parent_scores_survive_two_worker_decisions_and_self_advance(transport: str) -> None:
    """Exercise the score protocol as one continuous, multi-step lifecycle."""
    frontend = MagicMock()
    finished = SimpleNamespace(finished=True)
    frontend.register_beam_output.return_value = SimpleNamespace(
        get=AsyncMock(return_value=finished)
    )
    frontend.beam_request_step_update = AsyncMock()
    parents = [
        SimpleNamespace(tokens=[10, 20, 30, 50], cum_logprob=-0.1),
        SimpleNamespace(tokens=[10, 20, 30, 60], cum_logprob=-5.0),
    ]

    asyncio.run(
        _beam_request_step(
            frontend,
            session_id="s0",
            fork_info=[(0, 50), (1, 60)],
            prev_beam_internal_ids=["p0", "p1"],
            all_beams=parents,
            request_id_batch="batch-1",
            beam_search_params=_sampling_params(),
            eos_token_id=2,
            lora_request=None,
            trace_headers=None,
            prefix_len=3,
            max_decode_steps=4,
        )
    )
    await_args = frontend.beam_request_step_update.await_args
    assert await_args is not None
    first_update = await_args.args[0]
    assert first_update.parent_cumulative_logprobs == [-0.1, -5.0]

    engine = _make_engine_self(beam_cache={"s0": _session_entry()})
    with patch("vllm.v1.request.Request") as Request:
        first_request = MagicMock()
        first_request.block_hashes = []
        first_request.is_constrained = False
        first_request.is_constraint_prefill = False
        first_request.result_width = None
        Request.return_value = first_request
        _handle_beam_request_step_update(engine, first_update)
    engine.input_queue.get_nowait()
    first_request.num_computed_tokens = 5
    first_metadata = build_beam_request_metadata(first_request, num_scheduled_tokens=2)
    assert first_metadata is not None
    assert first_metadata["beam_suffix"] == [[50], [60]]
    assert first_metadata["parent_cumulative_logprobs"] == [-0.1, -5.0]

    context = BeamAttentionContext(
        max_batch=1,
        max_beams=2,
        num_kv_heads=1,
        max_decode_steps=4,
        head_dim=2,
        num_layers=1,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        _beam_context=context,
        _beam_sessions={},
        _vllm_gr_config=SimpleNamespace(beam_max_width=2, beam_max_decode_steps=4),
    )
    beam_data = {"s0": first_metadata}
    session = model_runner_common._get_or_create_beam_session(
        runner,
        beam_data,
        SimpleNamespace(beam_finished_session_ids=[]),
    )
    assert session is not None
    assert context.decision.beam_scores[0, :2].tolist() == pytest.approx([-0.1, -5.0])

    runner._current_beam_info = {"beam_data": beam_data, "beam_session": session}
    first_sampler = SimpleNamespace(
        logprobs_tensors=SimpleNamespace(
            logprob_token_ids=torch.tensor([[51, 52], [61, 62]], dtype=torch.int32),
            logprobs=torch.tensor([[-0.1, -0.2], [-0.1, -0.2]], dtype=torch.float32),
        )
    )
    first_decision = beam_decision.run_beam_decision(
        runner, session, first_sampler, is_last_step=False
    )
    assert first_decision is not None
    assert first_decision["out_log_probs"].reshape(-1).tolist() == pytest.approx([-0.2, -0.3])
    parent_indices = first_decision["out_token_index"].reshape(-1).tolist()
    assert parent_indices == [0, 0]
    assert session.decode_step == 1
    session._state.kv_reorder = [int(parent) for parent in parent_indices]

    if transport == "compact":
        encoded = beam_decision._encode_cuda_beam_result(first_decision, is_last_step=False)
        assert encoded is not None
        payload = encoded[0]
    else:
        payload = {
            key: first_decision[key].detach().cpu().tolist()
            for key in ("out_token_ids", "out_token_index", "out_log_probs")
        }
        payload["is_last_step"] = False
    payload["stage"] = "decode"
    payload["session_id"] = "s0"

    with patch("vllm.v1.request.Request") as Request:
        second_request = MagicMock()
        second_request.block_hashes = []
        second_request.is_constrained = False
        second_request.is_constraint_prefill = False
        second_request.result_width = None
        Request.return_value = second_request
        _beam_advance_from_decision(engine, "s0", payload)
    assert payload["beam_self_advance_chained"] is True
    _, (queued_request, _) = engine.input_queue.get_nowait()
    assert queued_request is second_request
    assert second_request.parent_cumulative_logprobs == pytest.approx([-0.2, -0.3])

    second_request.num_computed_tokens = 5
    second_metadata = build_beam_request_metadata(second_request, num_scheduled_tokens=2)
    assert second_metadata is not None
    assert second_metadata["beam_suffix"] == [[50, 51], [50, 52]]
    assert second_metadata["parent_beam_ids"] == [0, 0]
    assert second_metadata["parent_cumulative_logprobs"] == pytest.approx([-0.2, -0.3])
    second_beam_data = {"s0": second_metadata}
    same_session = model_runner_common._get_or_create_beam_session(
        runner,
        second_beam_data,
        SimpleNamespace(beam_finished_session_ids=[]),
    )
    assert same_session is session
    assert session._state.kv_reorder is None

    runner._current_beam_info = {
        "beam_data": second_beam_data,
        "beam_session": session,
    }
    second_sampler = SimpleNamespace(
        logprobs_tensors=SimpleNamespace(
            logprob_token_ids=torch.tensor([[71, 72], [81, 82]], dtype=torch.int32),
            logprobs=torch.tensor([[-0.5, -2.0], [-0.1, -3.0]], dtype=torch.float32),
        )
    )
    second_decision = beam_decision.run_beam_decision(
        runner, session, second_sampler, is_last_step=True
    )
    assert second_decision is not None
    assert sorted(second_decision["out_log_probs"].reshape(-1).tolist()) == pytest.approx(
        [-0.7, -0.4]
    )


# ===========================================================================
# Section 1 — _apply_beam_step records running_suffix
# ===========================================================================


class TestApplyBeamStepRunningSuffix:
    def test_apply_beam_step_records_running_suffix(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        params = _sampling_params()
        update = BeamRequestStepUpdate(
            session_id="s0",
            parent_beam_ids=["p0", "p1"],
            child_beam_ids=["c0", "c1"],
            beam_tokens=[[50], [60]],
            pruned_ids=[],
            prefix_len=3,
            beam_width=2,
            sampling_params=params,
            eos_token_id=9,
            client_index=5,
            current_wave=3,
            priority=2,
            data_parallel_rank=1,
            max_decode_steps=7,
        )

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            mock_req.is_constrained = False
            MockReq.return_value = mock_req
            _handle_beam_request_step_update(eng, update)

        running = eng.beam_cache["s0"]["running_suffix"]
        assert running["beam_tokens"] == [[50], [60]]
        assert running["prefix_len"] == 3
        assert running["beam_width"] == 2
        assert running["sampling_params"] is params
        assert running["eos_token_id"] == 9
        assert running["client_index"] == 5
        assert running["current_wave"] == 3
        assert running["priority"] == 2
        assert running["data_parallel_rank"] == 1
        assert running["max_decode_steps"] == 7
        assert running["is_constrained"] is False

    def test_running_suffix_updated_each_step(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        first = BeamRequestStepUpdate(
            session_id="s0",
            parent_beam_ids=["p0", "p1"],
            child_beam_ids=["c0", "c1"],
            beam_tokens=[[50], [60]],
            pruned_ids=[],
            prefix_len=3,
            beam_width=2,
            sampling_params=_sampling_params(),
        )
        second = BeamRequestStepUpdate(
            session_id="s0",
            parent_beam_ids=["c0", "c1"],
            child_beam_ids=["d0", "d1"],
            beam_tokens=[[50, 61], [60, 71]],
            pruned_ids=[],
            prefix_len=3,
            beam_width=2,
            sampling_params=first.sampling_params,
        )

        with patch("vllm.v1.request.Request"):
            _handle_beam_request_step_update(eng, first)
            _handle_beam_request_step_update(eng, second)

        assert eng.beam_cache["s0"]["running_suffix"]["beam_tokens"] == [
            [50, 61],
            [60, 71],
        ]

    def test_running_suffix_beam_tokens_are_copies(self) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        beam_tokens = [[50], [60]]
        update = BeamRequestStepUpdate(
            session_id="s0",
            parent_beam_ids=["p0", "p1"],
            child_beam_ids=["c0", "c1"],
            beam_tokens=beam_tokens,
            pruned_ids=[],
            prefix_len=3,
            beam_width=2,
            sampling_params=_sampling_params(),
        )

        with patch("vllm.v1.request.Request"):
            _handle_beam_request_step_update(eng, update)

        beam_tokens[0].append(99)
        assert eng.beam_cache["s0"]["running_suffix"]["beam_tokens"] == [[50], [60]]


# ===========================================================================
# Section 2 — _beam_advance_from_decision
# ===========================================================================


class TestBeamAdvanceFromDecision:
    def test_chains_next_step_from_decision(self) -> None:
        eng = _seeded_engine("s0")
        running = eng.beam_cache["s0"]["running_suffix"]
        decision = _decision(parents=[1, 0], tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _beam_advance_from_decision(eng, "s0", decision)

        # Exactly one ADD lands in input_queue (same busy-loop iteration).
        assert eng.input_queue.qsize() == 1
        req_type, payload = eng.input_queue.get_nowait()
        assert req_type is EngineCoreRequestType.ADD
        chained_req, wave = payload
        assert chained_req is mock_req
        assert wave == 3  # current_wave replayed from running_suffix

        _, kwargs = MockReq.call_args
        # New suffixes extend the recorded ones: parent 1 -> [60, 61],
        # parent 0 -> [50, 71]; incremental decode sends only the newest
        # token per beam after the prefill prefix.
        assert kwargs["request_id"] == "s0"
        assert kwargs["prompt_token_ids"] == [10, 20, 30, 61, 71]
        assert kwargs["sampling_params"] is running["sampling_params"]
        assert kwargs["client_index"] == 5
        assert kwargs["priority"] == 2

        assert mock_req.is_beam is True
        assert mock_req.is_beam_decode is True
        assert mock_req.beam_width == 2
        assert mock_req.prefix_len == 3
        assert mock_req.beam_decode_steps == 2
        assert mock_req.beam_decode_step == 1
        assert mock_req.parent_beam_indices == [1, 0]
        assert mock_req.max_decode_steps == 7

        # The chained step's running_suffix now carries the extended beams.
        assert eng.beam_cache["s0"]["running_suffix"]["beam_tokens"] == [
            [60, 61],
            [50, 71],
        ]

    def test_terminal_decision_stops_chain(self) -> None:
        eng = _seeded_engine("s0")
        decision = _decision(parents=[0, 1], tokens=[61, 71], is_last_step=True)

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()
        assert eng.beam_cache["s0"]["running_suffix"]["beam_tokens"] == [[50], [60]]

    def test_non_mapping_decision_is_noop(self) -> None:
        eng = _seeded_engine("s0")

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", ["not", "a", "mapping"])

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_decision_without_candidate_tensors_is_noop(self) -> None:
        eng = _seeded_engine("s0")
        decision = {"is_last_step": False, "out_log_probs": [[0.0, 0.0]]}

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_missing_running_suffix_is_graceful_noop(self) -> None:
        """Session cached before engine-driven chaining was enabled has no
        running_suffix; the chain must skip, not raise (the frontend
        step-update loop remains the fallback)."""
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        decision = _decision(parents=[0, 1], tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)  # must not raise

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_unknown_session_is_noop(self) -> None:
        eng = _make_engine_self(beam_cache={})
        decision = _decision(parents=[0, 1], tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_decision_width_mismatch_is_noop(self) -> None:
        eng = _seeded_engine("s0", beam_width=2)
        # Decision carries 3 candidates but the session records width 2.
        decision = _decision(parents=[0, 1, 1], tokens=[61, 71, 81])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_parent_index_out_of_range_is_noop(self) -> None:
        eng = _seeded_engine("s0")  # 2 previous suffixes
        decision = _decision(parents=[0, 2], tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_constrained_step_replayed(self) -> None:
        eng = _seeded_engine("s0", is_constrained=True)
        decision = _decision(parents=[0, 1], tokens=[61, 71])

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
        # Step 2 (two tokens per beam) becomes constraint_step for the
        # constrained decode path.
        assert mock_req.is_constrained is True
        assert mock_req.constraint_step == 2
        assert mock_req.beam_suffix == [[50, 61], [60, 71]]
        assert mock_req.is_constraint_prefill is False
        assert eng.beam_cache["s0"]["running_suffix"]["is_constrained"] is True

    def test_routing_fields_default_when_unset(self) -> None:
        entry = _session_entry()
        params = _sampling_params()
        entry["running_suffix"] = {
            "beam_tokens": [[50]],
            "prefix_len": 3,
            "beam_width": 1,
            "sampling_params": params,
            "eos_token_id": 2,
            "client_index": 0,
            "current_wave": 0,
            "priority": 0,
            "data_parallel_rank": None,
            "lora_request": None,
            "cache_salt": None,
            "trace_headers": None,
            "max_decode_steps": 0,
            "is_constrained": False,
        }
        eng = _make_engine_self(beam_cache={"s0": entry})
        decision = _decision(parents=[0], tokens=[61])

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.qsize() == 1
        _, payload = eng.input_queue.get_nowait()
        _chained_req, wave = payload
        assert wave == 0
        # eos falls back to the session-cached token id.
        assert params.eos_token_id == 2

    def test_chained_decision_sets_frontend_flag(self) -> None:
        """Successful chaining marks the decision dict so the frontend loop
        skips its own beam_request_step_update for the next step (otherwise
        scheduler.add_request asserts "duplicate request id")."""
        eng = _seeded_engine("s0")
        decision = _decision(parents=[1, 0], tokens=[61, 71])

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _beam_advance_from_decision(eng, "s0", decision)

        assert decision.get("beam_self_advance_chained") is True

    def test_skip_paths_leave_frontend_flag_unset(self) -> None:
        """Every skip path (terminal / non-mapping / missing fields /
        unknown session / width mismatch) must NOT set the flag: the
        frontend-driven step-update loop is the fallback."""
        skip_cases = [
            _decision(parents=[0, 1], tokens=[61, 71], is_last_step=True),
            ["not", "a", "mapping"],
            {"is_last_step": False, "out_log_probs": [[0.0, 0.0]]},
        ]
        for decision in skip_cases:
            eng = _seeded_engine("s0")
            with patch("vllm.v1.request.Request"):
                _beam_advance_from_decision(eng, "s0", decision)
            if isinstance(decision, dict):
                assert "beam_self_advance_chained" not in decision

        eng = _make_engine_self(beam_cache={"s0": _session_entry()})
        decision = _decision(parents=[0, 1], tokens=[61, 71])
        with patch("vllm.v1.request.Request"):
            _beam_advance_from_decision(eng, "s0", decision)
        assert "beam_self_advance_chained" not in decision

        eng = _make_engine_self(beam_cache={})
        decision = _decision(parents=[0, 1], tokens=[61, 71])
        with patch("vllm.v1.request.Request"):
            _beam_advance_from_decision(eng, "s0", decision)
        assert "beam_self_advance_chained" not in decision

        eng = _seeded_engine("s0", beam_width=2)
        decision = _decision(parents=[0, 1, 1], tokens=[61, 71, 81])
        with patch("vllm.v1.request.Request"):
            _beam_advance_from_decision(eng, "s0", decision)
        assert "beam_self_advance_chained" not in decision


# ===========================================================================
# Section 2b — engine-driven decode chaining (unified producer, Phase 2)
# ===========================================================================


class TestBeamSelfAdvance:
    def test_advances_from_engine_core_outputs(self) -> None:
        eng = _seeded_engine("s0")
        outputs = _outputs_with_decision(_decision(parents=[1, 0], tokens=[61, 71]))

        with patch("vllm.v1.request.Request"):
            _beam_self_advance(eng, outputs)

        assert eng.input_queue.qsize() == 1

    def test_drains_all_independent_session_decisions(self) -> None:
        """One engine iteration chains every session that produced a decision."""
        eng = _seeded_engine("s0")
        entry1 = _session_entry()
        entry1["running_suffix"] = _running_suffix(beam_tokens=[[50]])
        entry1["running_suffix"]["beam_width"] = 1
        eng.beam_cache["s1"] = entry1
        outputs = {
            0: _outputs_with_decision(_decision(parents=[1, 0], tokens=[61, 71]), request_id="s0")[
                0
            ],
            1: _outputs_with_decision(
                _decision(parents=[0], tokens=[61]), request_id="s1", client_index=1
            )[1],
        }

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _beam_self_advance(eng, outputs)

        assert eng.input_queue.qsize() == 2
        request_ids = [kwargs["request_id"] for _, kwargs in MockReq.call_args_list]
        assert request_ids == ["s0", "s1"]

    def test_disabled_flag_is_noop(self) -> None:
        entry = _session_entry()
        entry["running_suffix"] = _running_suffix()
        eng = _make_engine_self(beam_cache={"s0": entry}, self_advance=False)
        outputs = _outputs_with_decision(_decision(parents=[1, 0], tokens=[61, 71]))

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_self_advance(eng, outputs)

        assert eng.input_queue.empty()
        MockReq.assert_not_called()

    def test_empty_outputs_is_noop(self) -> None:
        eng = _seeded_engine("s0")
        _beam_self_advance(eng, {})
        assert eng.input_queue.empty()

    def test_outputs_without_decision_are_noop(self) -> None:
        eng = _seeded_engine("s0")
        outputs = {
            0: EngineCoreOutputs(
                engine_index=0,
                outputs=[EngineCoreOutput(request_id="s0", new_token_ids=[])],
            )
        }

        _beam_self_advance(eng, outputs)

        assert eng.input_queue.empty()

    def test_two_step_chain_then_terminal(self) -> None:
        """End-to-end engine-driven chain: two chained steps recorded from the
        running_suffix, then a terminal decision stops chaining."""
        eng = _seeded_engine("s0")
        step1 = _outputs_with_decision(_decision(parents=[1, 0], tokens=[61, 71]))
        step2 = _outputs_with_decision(_decision(parents=[0, 1], tokens=[62, 72]))
        terminal = _outputs_with_decision(
            _decision(parents=[0, 1], tokens=[63, 73], is_last_step=True)
        )

        with patch("vllm.v1.request.Request") as MockReq:
            MockReq.return_value.block_hashes = []
            _beam_self_advance(eng, step1)
            assert eng.input_queue.qsize() == 1
            assert eng.beam_cache["s0"]["running_suffix"]["beam_tokens"] == [
                [60, 61],
                [50, 71],
            ]

            _beam_self_advance(eng, step2)
            assert eng.input_queue.qsize() == 2
            assert eng.beam_cache["s0"]["running_suffix"]["beam_tokens"] == [
                [60, 61, 62],
                [50, 71, 72],
            ]

            _beam_self_advance(eng, terminal)
            assert eng.input_queue.qsize() == 2  # terminal adds nothing
            # Final ADD still carries the full 3-step suffixes.
            while not eng.input_queue.empty():
                _req_type, _payload = eng.input_queue.get_nowait()
            _, kwargs = MockReq.call_args
            assert kwargs["prompt_token_ids"] == [10, 20, 30, 62, 72]
            assert MockReq.return_value.beam_decode_steps == 3


# ===========================================================================
# Section 4 — engine-driven config gate
# ===========================================================================


def _step_engine(
    outputs: dict[int, EngineCoreOutputs],
    model_executed: bool = True,
    has_requests: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        step_fn=lambda: (outputs, model_executed),
        output_queue=queue.SimpleQueue(),
        post_step=MagicMock(),
        scheduler=SimpleNamespace(has_requests=lambda: has_requests),
    )


class TestProcessEngineStepWrapper:
    def test_self_advance_runs_before_outputs_are_forwarded(self) -> None:
        outputs = _outputs_with_decision(_decision(parents=[1, 0], tokens=[61, 71]))
        eng = _step_engine(outputs)
        observed: list[tuple[Any, Any, bool]] = []

        def fake_advance(engine: Any, outs: dict[int, EngineCoreOutputs]) -> None:
            observed.append((engine, outs, engine.output_queue.empty()))

        monkey = pytest.MonkeyPatch()
        monkey.setattr(core_module, "_beam_self_advance", fake_advance)
        try:
            wrapper = make_patched_process_engine_step(lambda self: None)
            result = wrapper(eng)
        finally:
            monkey.undo()

        # Hook saw the step's outputs BEFORE they reached the output queue,
        # so the chained ADD lands in input_queue within this iteration.
        assert observed == [(eng, outputs, True)]
        # The step's output envelope was forwarded (one queue entry per
        # client_index; the two EngineCoreOutputs ride inside it).
        assert eng.output_queue.qsize() == 1
        forwarded = eng.output_queue.get_nowait()
        assert forwarded == (0, outputs[0])
        # post_step ran with the model_executed flag and the flag returned.
        eng.post_step.assert_called_once_with(True)
        assert result is True

    def test_wrapper_survives_self_advance_failure(self) -> None:
        outputs = _outputs_with_decision(_decision(parents=[1, 0], tokens=[61, 71]))
        eng = _step_engine(outputs)

        def broken_advance(engine: Any, outs: dict[int, EngineCoreOutputs]) -> None:
            raise RuntimeError("boom")

        monkey = pytest.MonkeyPatch()
        monkey.setattr(core_module, "_beam_self_advance", broken_advance)
        try:
            wrapper = make_patched_process_engine_step(lambda self: None)
            result = wrapper(eng)  # must not raise
        finally:
            monkey.undo()

        # Outputs and post_step still run: the frontend loop stays usable.
        assert eng.output_queue.qsize() == 1
        eng.post_step.assert_called_once_with(True)
        assert result is True

    def test_wrapper_skips_hook_when_no_outputs(self) -> None:
        eng = _step_engine({}, model_executed=False)
        called: list[Any] = []

        monkey = pytest.MonkeyPatch()
        monkey.setattr(core_module, "_beam_self_advance", lambda e, o: called.append(o))
        try:
            wrapper = make_patched_process_engine_step(lambda self: None)
            result = wrapper(eng)
        finally:
            monkey.undo()

        assert called == []
        assert eng.output_queue.empty()
        eng.post_step.assert_called_once_with(False)
        assert result is False

    def test_wrapper_is_idempotent(self) -> None:
        def original(self: Any) -> bool:
            return True

        wrapper = make_patched_process_engine_step(original)
        assert make_patched_process_engine_step(wrapper) is wrapper


class TestInprocGetOutputHook:
    def test_inproc_get_output_runs_self_advance(self) -> None:
        from vllm.v1.engine import EngineCoreOutputs

        from vllm_gr.v1.engine.core_client_patch import make_patched_inproc_get_output

        outputs = _outputs_with_decision(_decision(parents=[1, 0], tokens=[61, 71]))
        eng = _step_engine(outputs)
        observed: list[Any] = []

        def fake_advance(engine: Any, outs: dict[int, EngineCoreOutputs]) -> None:
            observed.append((engine, outs))

        client = SimpleNamespace(engine_core=eng)
        monkey = pytest.MonkeyPatch()
        monkey.setattr(core_module, "_beam_self_advance", fake_advance)
        try:
            wrapped = make_patched_inproc_get_output(lambda self: None)
            result = wrapped(client)
        finally:
            monkey.undo()

        assert observed == [(eng, outputs)]
        eng.post_step.assert_called_once_with(model_executed=True)
        assert isinstance(result, EngineCoreOutputs)

    def test_inproc_get_output_is_idempotent(self) -> None:
        from vllm_gr.v1.engine.core_client_patch import make_patched_inproc_get_output

        def original(self: Any) -> None:
            return None

        wrapper = make_patched_inproc_get_output(original)
        assert make_patched_inproc_get_output(wrapper) is wrapper


# ===========================================================================
# Section 4 — decision payload shapes (post-IPC normalization)
# ===========================================================================


class TestDecisionPayloadShapes:
    def test_list_payload_chains_next_step(self) -> None:
        """Post-IPC decisions arrive as nested lists, not tensors.

        kv_transfer_params values decode against ``dict[str, Any]``; the
        worker therefore normalizes to plain lists at the D2H boundary, and
        engine-driven chaining must consume them (the old ``reshape(-1).tolist()``
        crashed with AttributeError: 'list' object has no attribute
        'reshape').
        """
        eng = _seeded_engine("s0")
        decision = _decision(parents=[1, 0], tokens=[61, 71])
        assert isinstance(decision["out_token_ids"], list)

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.qsize() == 1
        _, kwargs = MockReq.call_args
        assert kwargs["prompt_token_ids"] == [10, 20, 30, 61, 71]

    def test_tensor_payload_still_chains(self) -> None:
        """In-process callers may still hand over host tensors."""
        eng = _seeded_engine("s0")
        decision = {
            "out_token_ids": torch.tensor([[61, 71]], dtype=torch.int32),
            "out_token_index": torch.tensor([[1, 0]], dtype=torch.int32),
            "out_log_probs": torch.tensor([[-1.0, -2.0]], dtype=torch.float32),
            "is_last_step": False,
        }

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.qsize() == 1
        _, kwargs = MockReq.call_args
        assert kwargs["prompt_token_ids"] == [10, 20, 30, 61, 71]

    def test_compact_cuda_payload_chains_next_step(self) -> None:
        from vllm_gr.v1.beam import beam_decision

        eng = _seeded_engine("s0")
        encoded = beam_decision._encode_cuda_beam_result(
            {
                "out_token_ids": torch.tensor([[61, 71]], dtype=torch.int32),
                "out_token_index": torch.tensor([[1, 0]], dtype=torch.int32),
                "out_log_probs": torch.tensor([[-1.0, -2.0]], dtype=torch.float32),
            },
            is_last_step=False,
        )
        assert encoded is not None
        decision, _ = encoded

        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            MockReq.return_value = mock_req
            _beam_advance_from_decision(eng, "s0", decision)

        assert eng.input_queue.qsize() == 1
        _, kwargs = MockReq.call_args
        assert kwargs["prompt_token_ids"] == [10, 20, 30, 61, 71]

    def test_serialized_tensor_triple_is_graceful_noop(self) -> None:
        """An unnormalized (encoder-triple) payload is unrecoverable data —
        skip, never crash; the frontend step-update loop remains the
        fallback."""
        eng = _seeded_engine("s0")
        decision = {
            "out_token_ids": ["int64", [1, 2], 0],
            "out_token_index": ["int64", [1, 2], 1],
            "is_last_step": False,
        }

        with patch("vllm.v1.request.Request") as MockReq:
            _beam_advance_from_decision(eng, "s0", decision)  # must not raise

        assert eng.input_queue.empty()
        MockReq.assert_not_called()


# ===========================================================================
# Section — parent cumulative score guard (EOS-masked -inf beams)
# ===========================================================================


class TestApplyBeamStepParentScoreGuard:
    """EOS-masked beams legitimately carry -inf cumulative scores.

    The protocol may transport the EOS-mask sentinel at its lifecycle
    boundary. The guard must accept it, reject only NaN/+inf/length
    mismatches, and publish ERROR outputs for the registered session — a bare
    rejection would leave the frontend waiting on the session forever.
    """

    @staticmethod
    def _update(parent_scores: Any) -> BeamRequestStepUpdate:
        return BeamRequestStepUpdate(
            session_id="s0",
            parent_beam_ids=["p0", "p1"],
            child_beam_ids=["c0", "c1"],
            beam_tokens=[[50], [60]],
            pruned_ids=[],
            prefix_len=3,
            beam_width=2,
            sampling_params=_sampling_params(),
            eos_token_id=9,
            client_index=5,
            max_decode_steps=7,
            parent_cumulative_logprobs=parent_scores,
        )

    def _run(self, eng: Any, update: BeamRequestStepUpdate) -> bool:
        with patch("vllm.v1.request.Request") as MockReq:
            mock_req = MagicMock()
            mock_req.block_hashes = []
            mock_req.is_constrained = False
            MockReq.return_value = mock_req
            return core_module._apply_beam_step(eng, update)

    @pytest.mark.parametrize(
        "scores",
        [
            [-float("inf"), -1.25],
            [-float("inf"), -float("inf")],
        ],
        ids=["mixed", "all-masked"],
    )
    def test_eos_masked_neg_inf_parents_are_accepted(self, scores: list[float]) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})

        assert self._run(eng, self._update(scores)) is True

        assert eng.input_queue.qsize() == 1
        assert eng.output_queue.empty()

    @pytest.mark.parametrize(
        "scores",
        [
            [float("nan"), -1.25],
            [float("inf"), -1.25],
            [-1.0, -0.5, -3.0],
        ],
        ids=["nan", "plus-inf", "length-mismatch"],
    )
    def test_invalid_parent_scores_reject_and_publish_error(self, scores: list[float]) -> None:
        eng = _make_engine_self(beam_cache={"s0": _session_entry()})

        assert self._run(eng, self._update(scores)) is False

        assert eng.input_queue.empty()
        published: list[tuple[str, Any]] = []
        while not eng.output_queue.empty():
            client_index, outputs = eng.output_queue.get_nowait()
            assert client_index == 5
            for out in outputs.outputs:
                published.append((out.request_id, out.finish_reason))
        assert published == [("s0", FinishReason.ERROR)]

    @pytest.mark.parametrize("failure", ["nan", "inf", "score-count", "arrays", "cache"])
    @pytest.mark.parametrize("with_collector", [True, False], ids=["serving", "offline"])
    def test_validation_error_reaches_registered_session(
        self, failure: str, with_collector: bool
    ) -> None:
        from vllm.v1.engine.output_processor import OutputProcessor

        from vllm_gr.entrypoints.openai.serving_engine import _collect_beam_result
        from vllm_gr.v1.engine.async_llm import register_beam_output_fn

        processor = OutputProcessor(tokenizer=None, log_stats=False)
        frontend = SimpleNamespace(output_processor=processor, _run_output_handler=lambda: None)
        collector = register_beam_output_fn(
            frontend, "s0", [10, 20, 30], _sampling_params(logprobs=None)
        )
        if not with_collector:
            processor.request_states["s0"].queue = None
        eng = _make_engine_self(beam_cache={} if failure == "cache" else {"s0": _session_entry()})
        update = self._update([-1.0, -2.0])
        if failure in {"nan", "inf"}:
            update.parent_cumulative_logprobs = [float(failure), -2.0]
        elif failure == "score-count":
            update.parent_cumulative_logprobs = [-1.0]
        elif failure == "arrays":
            update.child_beam_ids = []  # Even zero child IDs must terminate the session.
        assert self._run(eng, update) is False
        client_index, outputs = eng.output_queue.get_nowait()
        assert client_index == 5
        assert outputs.finished_requests == {"s0"}
        assert eng.output_queue.empty()
        processed = processor.process_outputs(outputs.outputs)
        if with_collector:

            async def collect() -> Any:
                return await asyncio.wait_for(_collect_beam_result(collector), timeout=1.0)

            result = asyncio.run(collect())
        else:
            assert len(processed.request_outputs) == 1
            result = processed.request_outputs[0]
        assert result.request_id == "s0"
        assert result.finished
        assert result.outputs[0].finish_reason == "error"
        assert "s0" not in processor.request_states
