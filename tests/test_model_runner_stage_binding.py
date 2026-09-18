# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner entrypoints bind stage hooks without runtime version/protocol selection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_RETIRE_OUTPUT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
)
from vllm_gr.v1.worker import gpu_beam_stage_runner as stages
from vllm_gr.v1.worker import model_runner_common
from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import V2StageExecutionHooks


def _unexpected_native(*args: object, **kwargs: object) -> None:
    raise AssertionError("GR control frame dispatched to native execution")


def _scheduled_frame(**metadata: object) -> SimpleNamespace:
    return SimpleNamespace(
        num_scheduled_tokens={"session": 1},
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        **metadata,
    )


@pytest.mark.parametrize("use_v2", [False, True])  # type: ignore[misc]
def test_execute_entry_binds_hooks_independently_of_environment(
    monkeypatch: pytest.MonkeyPatch, use_v2: bool
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0" if use_v2 else "1")
    worker = SimpleNamespace(device="cuda")
    if not use_v2:
        worker.vllm_config = SimpleNamespace(
            additional_config={"vllm_gr_beam_execution_mode": "v1"}
        )
    # V2 deliberately has no vllm_config: the execution entry must not read
    # execution_mode again after Engine admission.
    output = _scheduled_frame(**{GR_STAGE_OUTPUT_ATTR: {"session": object()}})
    owner = SimpleNamespace(execute=Mock(return_value="stage"))
    get_owner = Mock(return_value=owner)
    monkeypatch.setattr(stages, "get_stage_runner", get_owner)
    native = _unexpected_native
    factory = (
        model_runner_common.make_patched_execute_model_v2
        if use_v2
        else model_runner_common.make_patched_execute_model
    )
    assert factory(native)(worker, output) == "stage"
    local = get_owner.call_args.args[1]
    assert local is not output
    assert local.gr_stage_metadata is output.gr_stage_metadata
    get_owner.assert_called_once_with(
        worker,
        local,
        hooks_type=V2StageExecutionHooks if use_v2 else stages._StageExecutionHooks,
    )
    owner.execute.assert_called_once_with(local, native)
    local.num_scheduled_tokens["session"] = 128
    local.finished_req_ids.add("session")
    assert output.num_scheduled_tokens == {"session": 1}
    assert output.finished_req_ids == set()


def test_v2_retirement_does_not_read_execution_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = SimpleNamespace(device="cuda")
    output = _scheduled_frame(**{GR_RETIRE_OUTPUT_ATTR: object()})
    retire = Mock(return_value="retired")
    monkeypatch.setattr(stages, "execute_gr_retirement", retire)
    native = _unexpected_native
    assert model_runner_common.make_patched_execute_model_v2(native)(worker, output) == "retired"
    local = retire.call_args.args[1]
    assert local is not output
    retire.assert_called_once_with(worker, local)
    local.finished_req_ids.add("session")
    assert output.finished_req_ids == set()


@pytest.mark.parametrize("frame_attr", [GR_STAGE_OUTPUT_ATTR, GR_RETIRE_OUTPUT_ATTR])  # type: ignore[misc]
def test_v1_rejects_persistent_frames_in_legacy_mode_before_dispatch(
    monkeypatch: pytest.MonkeyPatch, frame_attr: str
) -> None:
    worker = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))
    output = SimpleNamespace(**{frame_attr: {"session": object()}})
    get_owner = Mock(side_effect=AssertionError("invalid protocol allocated an owner"))
    retire = Mock(side_effect=AssertionError("invalid protocol retired an owner"))
    monkeypatch.setattr(stages, "get_stage_runner", get_owner)
    monkeypatch.setattr(stages, "execute_gr_retirement", retire)
    with pytest.raises(ValueError, match="execution_mode='v1'"):
        model_runner_common.make_patched_execute_model(_unexpected_native)(worker, output)
    get_owner.assert_not_called()
    retire.assert_not_called()


@pytest.mark.parametrize("hooks_type", [stages._StageExecutionHooks, V2StageExecutionHooks])  # type: ignore[misc]
def test_active_owner_keeps_bound_hooks_without_reading_config(
    monkeypatch: pytest.MonkeyPatch, hooks_type: type[stages._StageExecutionHooks]
) -> None:
    worker = SimpleNamespace(constraint_backend=object())
    hooks = hooks_type(worker)
    owner = SimpleNamespace(closed=False, hooks=hooks, backend=worker.constraint_backend)
    worker._gr_gpu_stage_runner = owner
    for flag in ("0", "1"):
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", flag)
        assert stages.get_stage_runner(worker, object(), hooks_type=hooks_type) is owner
        assert owner.hooks is hooks


@pytest.mark.parametrize("closed", [False, True])  # type: ignore[misc]
def test_owner_hooks_cannot_change_on_active_or_released_owner(closed: bool) -> None:
    worker = SimpleNamespace()
    worker._gr_gpu_stage_runner = SimpleNamespace(
        closed=closed, hooks=V2StageExecutionHooks(worker)
    )
    with pytest.raises(ValueError, match="hooks cannot change"):
        stages.get_stage_runner(worker, object(), hooks_type=stages._StageExecutionHooks)


@pytest.mark.parametrize("hooks_type", [stages._StageExecutionHooks, V2StageExecutionHooks])  # type: ignore[misc]
def test_stage_capture_passes_bound_model_install_policy(
    monkeypatch: pytest.MonkeyPatch, hooks_type: type[stages._StageExecutionHooks]
) -> None:
    from vllm_gr.v1.worker import beam_decode_graph

    worker = SimpleNamespace()
    owner = object.__new__(stages.GPUBeamStageRunner)
    owner.worker = worker
    owner.hooks = hooks_type(worker)
    graph = Mock()
    monkeypatch.setattr(beam_decode_graph, "get_beam_decode_graph", lambda _: graph)
    owner.graph = None
    assert owner.capture_forward() is graph
    graph.capture.assert_called_once_with(
        owner._forward_hidden, persistent_install=hooks_type is stages._StageExecutionHooks
    )
    assert owner.graph is None
