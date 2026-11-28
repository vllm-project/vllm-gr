# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Runner entrypoints bind stage hooks without runtime version/protocol selection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.helpers.gr_config import engine_config
from vllm_gr.v1.beam.dispatch import GR_RETIRE_OUTPUT_ATTR, GR_STAGE_OUTPUT_ATTR
from vllm_gr.v1.worker import model_runner_common
from vllm_gr.v1.worker.cuda import gpu_beam_stage_runner as stages
from vllm_gr.v1.worker.cuda.gpu_stage_hooks import CudaStageHooks
from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1
from vllm_gr.v1.worker.cuda.v2 import gpu_model_runner_v2_patch as _v2_integration
from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2


def _unexpected_native(*args: object, **kwargs: object) -> None:
    raise AssertionError("GR control frame dispatched to native execution")


def _bind_cuda_stage(worker: SimpleNamespace, use_v2: bool) -> None:
    from functools import partial

    worker._gr_ready = True
    worker._gr_pending_stage_sample = False
    worker.execute_gr_stage = partial(
        stages.execute_cuda_stage,
        worker,
        hooks_type=CudaStageHooksV2 if use_v2 else CudaStageHooksV1,
    )
    worker.retire_gr_stage = partial(stages.execute_gr_retirement, worker)
    worker.cancel_finished_gr = partial(stages.cancel_finished_stage, worker)


def _scheduled_frame(**metadata: object) -> SimpleNamespace:
    return SimpleNamespace(
        num_scheduled_tokens={"session": 1},
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        **metadata,
    )


@pytest.mark.parametrize("use_v2", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("mixed", [False, True])  # type: ignore[misc]
def test_old_beam_frames_rejected_before_native_execution(use_v2: bool, mixed: bool) -> None:
    worker = SimpleNamespace()
    output = _scheduled_frame(beam_data={"session": {"is_beam_decode": True}})
    if mixed:
        output.num_scheduled_tokens["ordinary"] = 1
    factory = (
        _v2_integration.make_patched_execute_model_v2
        if use_v2
        else model_runner_common.make_patched_execute_model
    )
    with pytest.raises(ValueError, match="(?i)legacy"):
        factory(_unexpected_native)(worker, output)
    assert not hasattr(worker, "_current_beam_data")
    assert not hasattr(worker, "execute_model_state")


@pytest.mark.parametrize("use_v2", [False, True])  # type: ignore[misc]
def test_execute_entry_binds_hooks_independently_of_environment(
    monkeypatch: pytest.MonkeyPatch, use_v2: bool
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0" if use_v2 else "1")
    worker = SimpleNamespace(device="cuda")
    if not use_v2:
        worker.vllm_config = SimpleNamespace(
            additional_config={
                "vllm_gr": {"gr": {"beam_width": [128] * 3, "candidate_width": [128] * 3}}
            }
        )
    # V2 deliberately has no vllm_config: the execution entry must not read
    # startup configuration again after Engine admission.
    output = _scheduled_frame(**{GR_STAGE_OUTPUT_ATTR: {"session": object()}})
    owner = SimpleNamespace(execute=Mock(return_value="stage"))
    get_owner = Mock(return_value=owner)
    monkeypatch.setattr(stages, "get_stage_runner", get_owner)
    native = _unexpected_native
    factory = (
        _v2_integration.make_patched_execute_model_v2
        if use_v2
        else model_runner_common.make_patched_execute_model
    )
    _bind_cuda_stage(worker, use_v2)
    patched = factory(native)
    assert patched(worker, output) == "stage"
    local = get_owner.call_args.args[1]
    assert local is not output
    assert local.gr_stage_metadata is output.gr_stage_metadata
    get_owner.assert_called_once_with(
        worker,
        local,
        hooks_type=CudaStageHooksV2 if use_v2 else CudaStageHooksV1,
    )
    owner.execute.assert_called_once_with(local, native)
    local.num_scheduled_tokens["session"] = 128
    local.finished_req_ids.add("session")
    assert output.num_scheduled_tokens == {"session": 1}
    assert output.finished_req_ids == set()


@pytest.mark.l1  # type: ignore[misc]
def test_cuda_dispatch_requires_bound_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.worker.gr_execution import execute_gr_stage

    get_owner = Mock(side_effect=AssertionError("unbound dispatch allocated an owner"))
    monkeypatch.setattr(stages, "get_stage_runner", get_owner)
    with pytest.raises(RuntimeError, match="before model initialization"):
        execute_gr_stage(SimpleNamespace(device="cuda"), _scheduled_frame(), _unexpected_native)
    get_owner.assert_not_called()


def test_v2_retirement_does_not_read_startup_config(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = SimpleNamespace(device="cuda")
    output = _scheduled_frame(**{GR_RETIRE_OUTPUT_ATTR: object()})
    retire = Mock(return_value="retired")
    monkeypatch.setattr(stages, "execute_gr_retirement", retire)
    _bind_cuda_stage(worker, True)
    native = _unexpected_native
    assert _v2_integration.make_patched_execute_model_v2(native)(worker, output) == "retired"
    local = retire.call_args.args[1]
    assert local is not output
    retire.assert_called_once_with(worker, local)
    local.finished_req_ids.add("session")
    assert output.finished_req_ids == set()


@pytest.mark.parametrize("frame_attr", [GR_STAGE_OUTPUT_ATTR, GR_RETIRE_OUTPUT_ATTR])  # type: ignore[misc]
def test_v1_rejects_persistent_frames_without_v1_before_dispatch(
    monkeypatch: pytest.MonkeyPatch, frame_attr: str
) -> None:
    worker = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))
    output = SimpleNamespace(**{frame_attr: {"session": object()}})
    get_owner = Mock(side_effect=AssertionError("invalid protocol allocated an owner"))
    retire = Mock(side_effect=AssertionError("invalid protocol retired an owner"))
    monkeypatch.setattr(stages, "get_stage_runner", get_owner)
    monkeypatch.setattr(stages, "execute_gr_retirement", retire)
    with pytest.raises(ValueError, match="gr.beam_width"):
        model_runner_common.make_patched_execute_model(_unexpected_native)(worker, output)
    get_owner.assert_not_called()
    retire.assert_not_called()


@pytest.mark.parametrize("hooks_type", [CudaStageHooksV1, CudaStageHooksV2])  # type: ignore[misc]
def test_active_owner_keeps_bound_hooks_without_reading_config(
    monkeypatch: pytest.MonkeyPatch, hooks_type: type[CudaStageHooks]
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
    worker._gr_gpu_stage_runner = SimpleNamespace(closed=closed, hooks=CudaStageHooksV2(worker))
    with pytest.raises(ValueError, match="hooks cannot change"):
        stages.get_stage_runner(worker, object(), hooks_type=CudaStageHooksV1)


@pytest.mark.parametrize("hooks_type", [CudaStageHooksV1, CudaStageHooksV2])  # type: ignore[misc]
@pytest.mark.parametrize("input_width", [None, 2])  # type: ignore[misc]
def test_stage_capture_passes_bound_model_install_policy(
    monkeypatch: pytest.MonkeyPatch,
    hooks_type: type[CudaStageHooks],
    input_width: int | None,
) -> None:
    from vllm_gr.v1.worker.cuda import beam_decode_graph

    worker = SimpleNamespace()
    owner = object.__new__(stages.GPUBeamStageRunner)
    owner.worker = worker
    owner.hooks = hooks_type(worker)
    owner.bindings = (object(),)
    owner.beam = SimpleNamespace(state=SimpleNamespace(beam_width=4))
    owner.inputs = None if input_width is None else SimpleNamespace(width=input_width)
    graph = Mock()
    get_graph = Mock(return_value=graph)
    monkeypatch.setattr(beam_decode_graph, "get_beam_decode_graph", get_graph)
    owner.graph = None
    assert owner.capture_forward() is graph
    assert get_graph.call_args.kwargs["beam_width"] == (input_width or 4)
    graph.capture.assert_called_once_with(
        owner._forward_hidden, persistent_install=hooks_type is CudaStageHooksV1
    )
    assert owner.graph is None


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("use_v2", [False, True])  # type: ignore[misc]
def test_stage_none_result_never_falls_through_to_native(
    monkeypatch: pytest.MonkeyPatch, use_v2: bool
) -> None:
    worker = SimpleNamespace(device="cuda")
    if not use_v2:
        worker.vllm_config = engine_config()
    scheduled = _scheduled_frame(**{GR_STAGE_OUTPUT_ATTR: {"session": object()}})
    owner = SimpleNamespace(execute=Mock(return_value=None))
    monkeypatch.setattr(stages, "get_stage_runner", Mock(return_value=owner))
    factory = (
        _v2_integration.make_patched_execute_model_v2
        if use_v2
        else model_runner_common.make_patched_execute_model
    )
    _bind_cuda_stage(worker, use_v2)
    wrapper = factory(_unexpected_native)
    assert wrapper(worker, scheduled) is None
    assert owner.execute.call_count == 1
