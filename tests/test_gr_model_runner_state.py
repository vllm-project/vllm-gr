# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.engine.types import GRRequestStage
from vllm_gr.v1.worker.gr_model_runner_state import (
    GRModelRunnerState,
    acquire_gr_model_runner_state,
    get_gr_model_runner_state_stats,
    get_gr_memory_stats,
    release_gr_model_runner_states,
    unpack_gr_final_output,
)


class _Session:
    def __init__(self) -> None:
        self.end_calls = 0

    def end_session(self) -> None:
        self.end_calls += 1


def _state(session: _Session | None = None) -> GRModelRunnerState:
    return GRModelRunnerState.allocate(
        request_id="gr-0",
        session=session or _Session(),
        beam_width=2,
        max_tokens=2,
        continue_token_id=7,
        finish_token_id=8,
        device=torch.device("cpu"),
    )


def test_state_keeps_beam_history_until_one_final_copy() -> None:
    state = _state()
    state.bootstrap(
        torch.tensor([[10, 11]], dtype=torch.int32),
        torch.tensor([[-1.0, -2.0]], dtype=torch.float32),
    )
    assert state.stage is GRRequestStage.PREFILL_BOOTSTRAPPED

    state.advance(
        torch.tensor([[20, 21]], dtype=torch.int32),
        torch.tensor([[-1.5, -2.5]], dtype=torch.float32),
        torch.tensor([[1, 0]], dtype=torch.int32),
    )
    state.set_control_token(True)

    output = state.make_final_output()
    assert output["request_id"] == "gr-0"
    unpacked = unpack_gr_final_output(output)
    assert unpacked is not None
    sequences, parents, logprobs, scores = unpacked
    assert sequences == [[11, 20], [10, 21]]
    assert parents == [[0, 1], [0, 0]]
    assert logprobs[0] == pytest.approx([-2.0, 0.5])
    assert logprobs[1] == pytest.approx([-1.0, -1.5])
    assert scores == pytest.approx([-1.5, -2.5])
    assert state.control_token_ids.tolist() == [[8]]
    assert state.stage is GRRequestStage.FINAL_READY


def test_release_is_idempotent() -> None:
    session = _Session()
    state = _state(session)

    assert state.release() is True
    assert state.release() is False
    assert state.stage is GRRequestStage.RELEASED
    assert session.end_calls == 1


def test_one_step_state_is_final_after_bootstrap() -> None:
    state = GRModelRunnerState.allocate(
        request_id="gr-0",
        session=_Session(),
        beam_width=1,
        max_tokens=1,
        continue_token_id=7,
        finish_token_id=8,
        device=torch.device("cpu"),
    )

    state.bootstrap(torch.tensor([10]), torch.tensor([-1.0]))

    assert state.stage is GRRequestStage.FINAL_READY


def test_release_worker_states_removes_state_and_session() -> None:
    state = _state()
    runner = SimpleNamespace(
        _gr_model_runner_states={"gr-0": state},
        _beam_sessions={"gr-0": state.session},
    )
    worker = SimpleNamespace(model_runner=runner)

    assert release_gr_model_runner_states(worker, ["gr-0"]) == 1
    assert release_gr_model_runner_states(worker, ["gr-0"]) == 0
    assert runner._gr_model_runner_states == {}
    assert runner._beam_sessions == {}
    assert state.session.end_calls == 1
    assert get_gr_model_runner_state_stats(runner) == {
        "created": 0,
        "freed": 1,
        "active": 0,
    }


def test_released_state_allocation_is_reused() -> None:
    state = _state()
    state.bootstrap(
        torch.tensor([10, 11], dtype=torch.int32),
        torch.tensor([-1.0, -2.0], dtype=torch.float32),
    )
    state.advance(
        torch.tensor([20, 21], dtype=torch.int32),
        torch.tensor([-1.5, -2.5], dtype=torch.float32),
        torch.tensor([1, 0], dtype=torch.int32),
    )
    runner = SimpleNamespace(
        _gr_model_runner_states={"gr-0": state},
        _beam_sessions={"gr-0": state.session},
    )
    assert release_gr_model_runner_states(runner, ["gr-0"]) == 1

    session = _Session()
    reused = acquire_gr_model_runner_state(
        runner,
        request_id="gr-1",
        session=session,
        beam_width=2,
        max_tokens=2,
        continue_token_id=17,
        finish_token_id=18,
        device=torch.device("cpu"),
    )

    assert reused is state
    assert reused.request_id == "gr-1"
    assert reused.session is session
    assert reused.step == 0
    assert reused.device_step == 0
    assert reused.released is False
    assert reused.control_token_ids.tolist() == [[17]]

    reused.bootstrap(
        torch.tensor([30, 31], dtype=torch.int32),
        torch.tensor([-3.0, -4.0], dtype=torch.float32),
    )
    reused.advance(
        torch.tensor([40, 41], dtype=torch.int32),
        torch.tensor([-3.5, -4.5], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.int32),
    )
    final = unpack_gr_final_output(reused.make_final_output())
    assert final is not None
    sequences, parents, logprobs, scores = final
    assert sequences == [[30, 40], [31, 41]]
    assert parents == [[0, 0], [0, 1]]
    assert logprobs[0] == pytest.approx([-3.0, -0.5])
    assert logprobs[1] == pytest.approx([-4.0, -0.5])
    assert scores == pytest.approx([-3.5, -4.5])


def test_memory_stats_separate_native_beam_and_state_storage() -> None:
    state = _state()
    state.prefix_block_table = torch.tensor([1, 2], dtype=torch.int32)
    pool = SimpleNamespace(
        unshared_key=torch.empty(2, dtype=torch.float16),
        unshared_value=torch.empty(3, dtype=torch.float16),
    )
    runner = SimpleNamespace(
        kv_caches=[torch.empty(4, dtype=torch.float16)],
        _beam_context=SimpleNamespace(pools=[pool]),
        _beam_postprocess_workspaces=SimpleNamespace(device_memory_bytes=24),
        _gr_state_peak_bytes=state.device_memory_bytes,
    )

    stats = get_gr_memory_stats(runner)

    assert stats == {
        "native_kv_bytes": 8,
        "beam_kv_bytes": 10,
        "workspace_bytes": 24,
        "gr_state_peak_bytes": state.device_memory_bytes,
        "graph_pool_allocated_bytes": 0,
        "graph_pool_reserved_bytes": 0,
    }


def test_release_worker_states_handles_multiple_requests() -> None:
    first = _state()
    second = GRModelRunnerState.allocate(
        request_id="gr-1",
        session=_Session(),
        beam_width=3,
        max_tokens=2,
        continue_token_id=7,
        finish_token_id=8,
        device=torch.device("cpu"),
    )
    runner = SimpleNamespace(
        _gr_model_runner_states={"gr-0": first, "gr-1": second},
        _beam_sessions={"gr-0": first.session, "gr-1": second.session},
    )

    assert release_gr_model_runner_states(runner) == 2
    assert runner._gr_model_runner_states == {}
    assert runner._beam_sessions == {}
    assert first.released is True
    assert second.released is True


def test_state_cache_is_replaced_for_a_different_shape() -> None:
    state = _state()
    runner = SimpleNamespace(
        _gr_model_runner_states={"gr-0": state},
        _beam_sessions={"gr-0": state.session},
    )
    assert release_gr_model_runner_states(runner, ["gr-0"]) == 1

    replacement = acquire_gr_model_runner_state(
        runner,
        request_id="gr-1",
        session=_Session(),
        beam_width=3,
        max_tokens=2,
        continue_token_id=7,
        finish_token_id=8,
        device=torch.device("cpu"),
    )

    assert replacement is not state
    assert replacement.beam_width == 3
    assert runner._gr_model_runner_state_cache is None


def test_shutdown_clears_released_state_cache() -> None:
    runner = SimpleNamespace(
        _gr_model_runner_states={},
        _gr_model_runner_state_cache=_state(),
    )

    assert release_gr_model_runner_states(runner) == 0
    assert runner._gr_model_runner_state_cache is None
