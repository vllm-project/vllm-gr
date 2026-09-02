# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

import queue
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
)

import vllm_gr.v1.engine.core as core_module
from vllm_gr.v1.engine.core import (
    _beam_advance_from_decision,
    _beam_self_advance,
    _handle_beam_request_step_update,
)
from vllm_gr.v1.engine.engine_core_patch import make_patched_process_engine_step
from vllm_gr.v1.engine.types import BeamRequestStepUpdate

# ---------------------------------------------------------------------------
# Shared test helpers (mirrors test_beam_request_unit.py)
# ---------------------------------------------------------------------------


def _sampling_params(**overrides: Any) -> SamplingParams:
    defaults = dict(max_tokens=4, logprobs=2, temperature=0.0, detokenize=False)
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
                _req_type, payload = eng.input_queue.get_nowait()
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
