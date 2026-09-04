# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.engine.types import GRRequestStage
from vllm_gr.v1.worker.gr_model_runner_state import (
    GRBackendStep,
    GRDeviceBackend,
    GRModelRunnerState,
    acquire_gr_model_runner_state,
    get_gr_memory_stats,
    get_gr_model_runner_state_stats,
    release_gr_model_runner_states,
    require_gr_device_backend,
)


class _Session:
    def __init__(self, *, already_ended: bool = False) -> None:
        self.end_calls = 0
        self.already_ended = already_ended

    def end_session(self) -> None:
        self.end_calls += 1
        if self.already_ended:
            raise KeyError("slot already released")


class _FakeBackend:
    def __init__(self, *, bad_device_step: bool = False, fail_release: bool = False) -> None:
        self.events: list[Any] = []
        self.bad_device_step = bad_device_step
        self.fail_release = fail_release
        self.controls = [object(), object()]

    def _result(self, state: GRModelRunnerState) -> GRBackendStep:
        device_step = state.device_step + (2 if self.bad_device_step else 1)
        return GRBackendStep(
            device_step=device_step,
            next_token_ids=("tokens", device_step),
            beam_scores=("scores", device_step),
            active_mask=("active", device_step),
            control_output=self.controls[device_step % 2],
        )

    def bootstrap_from_prefill(self, runner, state, sampler_output, metadata):
        self.events.append(("bootstrap", sampler_output, metadata))
        state.backend_state = SimpleNamespace(device_memory_bytes=17)
        state.prefix_binding = "backend-owned-prefix"
        return self._result(state)

    def prepare_decode_inputs(
        self, runner, state, scheduler_output, num_scheduled_tokens, metadata
    ) -> None:
        self.events.append(("prepare", scheduler_output, num_scheduled_tokens, metadata))

    def advance_beam(self, runner, state, sampler_output, metadata, *, is_last_step):
        self.events.append(("advance", sampler_output, metadata, is_last_step))
        return self._result(state)

    def reorder_kv(self, runner, state, step_result, *, is_last_step) -> None:
        self.events.append(("reorder", step_result.device_step, is_last_step))

    def finalize_output(self, runner, state):
        self.events.append(("finalize", state.device_step))
        return {
            "request_id": state.request_id,
            "is_last_step": True,
            "step": state.step,
            "device_step": state.device_step,
        }

    def release(self, state) -> None:
        self.events.append(("release", state.request_id))
        if self.fail_release:
            raise RuntimeError("backend release failed")


def _state(
    *,
    backend: _FakeBackend | None = None,
    session: _Session | None = None,
    max_tokens: int = 3,
) -> GRModelRunnerState:
    state = GRModelRunnerState(
        request_id="gr-0",
        session=session or _Session(),
        backend=backend or _FakeBackend(),
        beam_width=2,
        max_tokens=max_tokens,
        continue_token_id=7,
        finish_token_id=8,
    )
    state.start_prefill()
    return state


def test_fake_backend_drives_complete_state_machine_in_order() -> None:
    backend = _FakeBackend()
    state = _state(backend=backend, max_tokens=2)
    assert state.stage is GRRequestStage.PREFILL_RUNNING

    state.bootstrap_from_prefill("runner", "prefill-output", {"stage": "prefill"})
    assert state.stage is GRRequestStage.PREFILL_BOOTSTRAPPED
    state.prepare_decode_inputs("runner", "scheduled", [1], {"generation_index": 1})
    assert state.advance_beam("runner", "decode-output", {"generation_index": 1}) is True
    assert state.stage is GRRequestStage.FINAL_READY
    assert state.finalize_output("runner")["request_id"] == "gr-0"
    assert state.release() is True
    assert state.stage is GRRequestStage.RELEASED
    assert [event[0] for event in backend.events] == [
        "bootstrap",
        "prepare",
        "advance",
        "reorder",
        "finalize",
        "release",
    ]


def test_illegal_transitions_and_generation_mismatch_fail_before_backend() -> None:
    backend = _FakeBackend()
    state = _state(backend=backend)
    with pytest.raises(RuntimeError, match="illegal persistent GR state transition"):
        state.start_prefill()
    state.bootstrap_from_prefill(None, None, {})
    with pytest.raises(RuntimeError, match="generation/device step mismatch"):
        state.prepare_decode_inputs(None, None, None, {"generation_index": 2})
    assert [event[0] for event in backend.events] == ["bootstrap"]
    with pytest.raises(RuntimeError, match="requires prepare_decode_inputs"):
        state.advance_beam(None, None, {"generation_index": 1})
    state.prepare_decode_inputs(None, None, None, {"generation_index": 1})
    with pytest.raises(RuntimeError, match="already prepared"):
        state.prepare_decode_inputs(None, None, None, {"generation_index": 1})
    with pytest.raises(RuntimeError, match="generation/device step mismatch"):
        state.advance_beam(None, None, {"generation_index": 2})
    with pytest.raises(RuntimeError, match="final output requested"):
        state.finalize_output(None)


def test_backend_device_step_mismatch_does_not_advance_common_state() -> None:
    backend = _FakeBackend(bad_device_step=True)
    state = _state(backend=backend)
    with pytest.raises(RuntimeError, match="backend/device step mismatch"):
        state.bootstrap_from_prefill(None, None, {})
    assert state.step == state.device_step == 0
    assert state.stage is GRRequestStage.PREFILL_RUNNING


def test_two_control_slots_alternate_without_overwriting_inflight_reference() -> None:
    backend = _FakeBackend()
    state = _state(backend=backend, max_tokens=3)
    state.bootstrap_from_prefill(None, None, {})
    first = state.control_token_ids
    state.prepare_decode_inputs(None, None, None, {"generation_index": 1})
    state.advance_beam(None, None, {"generation_index": 1})
    second = state.control_token_ids
    state.prepare_decode_inputs(None, None, None, {"generation_index": 2})
    state.advance_beam(None, None, {"generation_index": 2})
    third = state.control_token_ids

    assert first is third
    assert first is not second
    assert state.control_slots == (second, first)


def test_backend_cannot_replace_or_alias_fixed_control_slots() -> None:
    class ReplacingBackend(_FakeBackend):
        def _result(self, state: GRModelRunnerState) -> GRBackendStep:
            result = super()._result(state)
            return GRBackendStep(
                device_step=result.device_step,
                next_token_ids=result.next_token_ids,
                beam_scores=result.beam_scores,
                active_mask=result.active_mask,
                control_output=object(),
            )

    replacing = ReplacingBackend()
    state = _state(backend=replacing, max_tokens=3)
    state.bootstrap_from_prefill(None, None, {})
    state.prepare_decode_inputs(None, None, None, {"generation_index": 1})
    state.advance_beam(None, None, {"generation_index": 1})
    state.prepare_decode_inputs(None, None, None, {"generation_index": 2})
    with pytest.raises(RuntimeError, match="replaced a fixed control buffer slot"):
        state.advance_beam(None, None, {"generation_index": 2})

    aliasing = _FakeBackend()
    aliasing.controls[0] = aliasing.controls[1]
    state = _state(backend=aliasing, max_tokens=2)
    state.bootstrap_from_prefill(None, None, {})
    state.prepare_decode_inputs(None, None, None, {"generation_index": 1})
    with pytest.raises(RuntimeError, match="two distinct control buffer slots"):
        state.advance_beam(None, None, {"generation_index": 1})


def test_bad_decode_device_step_is_rejected_before_kv_reorder() -> None:
    backend = _FakeBackend()
    state = _state(backend=backend)
    state.bootstrap_from_prefill(None, None, {})
    state.prepare_decode_inputs(None, None, None, {"generation_index": 1})
    backend.bad_device_step = True

    with pytest.raises(RuntimeError, match="backend/device step mismatch"):
        state.advance_beam(None, None, {"generation_index": 1})

    assert [event[0] for event in backend.events] == ["bootstrap", "prepare", "advance"]
    assert state.step == state.device_step == 1


def test_release_is_idempotent_and_already_ended_session_is_tolerated() -> None:
    backend = _FakeBackend()
    session = _Session(already_ended=True)
    state = _state(backend=backend, session=session)
    assert state.release() is True
    assert state.release() is False
    assert session.end_calls == 1
    assert [event[0] for event in backend.events] == ["release"]


def test_backend_release_failure_still_releases_session_and_all_worker_states() -> None:
    failing = _state(backend=_FakeBackend(fail_release=True))
    healthy = GRModelRunnerState(
        request_id="gr-1",
        session=_Session(),
        backend=_FakeBackend(),
        beam_width=2,
        max_tokens=3,
        continue_token_id=7,
        finish_token_id=8,
    )
    healthy.start_prefill()
    runner = SimpleNamespace(
        _gr_model_runner_states={"gr-0": failing, "gr-1": healthy},
        _beam_sessions={"gr-0": failing.session, "gr-1": healthy.session},
    )

    with pytest.raises(RuntimeError, match="backend release failed"):
        release_gr_model_runner_states(runner)

    assert runner._gr_model_runner_states == {}
    assert runner._beam_sessions == {}
    assert failing.session.end_calls == healthy.session.end_calls == 1
    assert failing.released is healthy.released is True
    assert get_gr_model_runner_state_stats(runner) == {
        "created": 0,
        "freed": 2,
        "active": 0,
    }


def test_released_state_is_reused_only_with_same_backend_and_shape() -> None:
    backend = _FakeBackend()
    state = _state(backend=backend)
    runner = SimpleNamespace(
        _gr_model_runner_states={"gr-0": state},
        _beam_sessions={"gr-0": state.session},
    )
    assert release_gr_model_runner_states(runner, ["gr-0"]) == 1
    reused = acquire_gr_model_runner_state(
        runner,
        request_id="gr-1",
        session=_Session(),
        backend=backend,
        beam_width=2,
        max_tokens=3,
        continue_token_id=17,
        finish_token_id=18,
    )
    assert reused is state
    assert reused.stage is GRRequestStage.PREFILL_RUNNING
    assert reused.request_id == "gr-1"
    assert reused.step == reused.device_step == 0

    reused.release()
    runner._gr_model_runner_state_cache = reused
    replacement = acquire_gr_model_runner_state(
        runner,
        request_id="gr-2",
        session=_Session(),
        backend=_FakeBackend(),
        beam_width=2,
        max_tokens=3,
        continue_token_id=7,
        finish_token_id=8,
    )
    assert replacement is not reused


def test_protocol_validation_and_missing_backend_fail_fast() -> None:
    backend = _FakeBackend()
    assert isinstance(backend, GRDeviceBackend)
    assert require_gr_device_backend(SimpleNamespace(_gr_device_backend=backend)) is backend
    with pytest.raises(RuntimeError, match="not configured"):
        require_gr_device_backend(SimpleNamespace(_gr_device_backend=None))


def test_common_state_source_has_no_accelerator_resources() -> None:
    from vllm_gr.v1.worker import gr_model_runner_state as module

    source = inspect.getsource(module)
    for forbidden in ("torch.cuda", "torch.npu", "torch.Event", "pin_memory"):
        assert forbidden not in source
    assert "torch_npu" not in sys.modules
    state_fields = GRModelRunnerState.__dataclass_fields__
    for duplicate_history in ("sequences", "parent_ids", "selected_logprobs"):
        assert duplicate_history not in state_fields
    for device_resource in ("final_copy_event", "host_final_result", "graph_handle"):
        assert device_resource not in state_fields


def test_memory_stats_separate_native_beam_and_backend_state_storage() -> None:
    state = _state()
    state.backend_state = SimpleNamespace(device_memory_bytes=17)
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
    assert get_gr_memory_stats(runner) == {
        "native_kv_bytes": 8,
        "beam_kv_bytes": 10,
        "workspace_bytes": 24,
        "gr_state_peak_bytes": 17,
        "graph_pool_allocated_bytes": 0,
        "graph_pool_reserved_bytes": 0,
    }
