# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified V2 Prefill plan, capture publication, fallback and instance isolation."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode as Mode
from vllm.v1.worker.gpu import cudagraph_utils as native

from vllm_gr.v1.worker import gpu_prefill_graph_runner_v2 as pg2
from vllm_gr.v1.worker.prefill_graph_common import PrefillGraphEnv


@pytest.fixture  # type: ignore[misc]
def runner(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(PrefillGraphEnv, "enabled", True)
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        use_ubatching=False,
    )
    compilation = SimpleNamespace(
        cudagraph_capture_sizes=[1, 2],
        adjust_cudagraph_sizes_for_spec_decode=lambda *args: None,
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        compilation_config=compilation,
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=512),
    )
    monkeypatch.setattr(
        native,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_first_rank=True,
            is_last_rank=True,
        ),
    )
    monkeypatch.setattr(native.current_platform, "get_global_graph_pool", lambda: None)
    manager = native.ModelCudaGraphManager(config, torch.device("cpu"), Mode.FULL_AND_PIECEWISE, 1)
    return SimpleNamespace(
        cudagraph_manager=manager,
        scheduler_config=config.scheduler_config,
        model_config=SimpleNamespace(max_model_len=2048, enforce_eager=False),
        parallel_config=parallel,
        lora_config=None,
        speculator=None,
        use_aux_hidden_state_outputs=False,
        supports_mm_inputs=False,
        req_states=SimpleNamespace(
            req_id_to_index={"r0": 0},
            num_computed_tokens_np=[512],
            prompt_len=SimpleNamespace(np=[1024]),
        ),
    )


def scheduled() -> SimpleNamespace:
    return SimpleNamespace(num_scheduled_tokens={"r0": 128})


def make_manager(
    runner: SimpleNamespace, *, piecewise: bool = True
) -> pg2.GRPrefillCudaGraphManager:
    return pg2.GRPrefillCudaGraphManager(
        runner, pg2.PrefillCapturePlan((1, 16, 128, 512), True, piecewise)
    )


def fake_capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail: Callable[[native.BatchExecutionDescriptor], None] | None = None,
) -> list[tuple[Mode, int]]:
    observed: list[tuple[Mode, int]] = []

    def capture(
        manager: Any, *args: Any, **kwargs: Any
    ) -> dict[native.BatchExecutionDescriptor, object]:
        descs = next(iter(manager._capture_descs.values()))
        for desc in descs:
            observed.append((desc.cg_mode, desc.num_tokens))
            assert manager.dispatch(1, 128, None).cg_mode == Mode.NONE
            if fail:
                fail(desc)
            if desc.cg_mode == Mode.FULL:
                manager.graphs[desc] = object()
        manager._graphs_captured = True
        return {desc: object() for desc in descs}

    monkeypatch.setattr(native.ModelCudaGraphManager, "capture", capture)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args: None)
    return observed


def test_plan_union_covers_native_gr_and_short_tail(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pg2, "get_default_buckets", lambda: [128, 256, 1024])
    runner.cudagraph_manager.compilation_config.cudagraph_capture_sizes = [1, 2, 384, 2048]
    plan = pg2.PrefillCapturePlan.for_runner(runner)
    assert plan.buckets == (1, 2, 16, 128, 256, 384, 512)
    descs = plan.descriptors()
    assert [d.num_tokens for d in descs[Mode.FULL]] == list(reversed(plan.buckets))
    assert all(d.num_reqs == 1 and d.uniform_token_count is None for d in descs[Mode.FULL])
    assert all(d.num_reqs is None for d in descs[Mode.PIECEWISE])
    runner.model_config.max_model_len = 200
    assert pg2.PrefillCapturePlan.for_runner(runner).buckets[-1] == 200


@pytest.mark.parametrize(  # type: ignore[misc]
    "mode,full,pw",
    [
        (Mode.NONE, False, False),
        (Mode.FULL, True, False),
        (Mode.PIECEWISE, False, True),
        (Mode.FULL_AND_PIECEWISE, True, True),
    ],
)
def test_plan_honors_native_modes(
    runner: SimpleNamespace, mode: Mode, full: bool, pw: bool
) -> None:
    runner.cudagraph_manager.cudagraph_mode = mode
    plan = pg2.PrefillCapturePlan.for_runner(runner)
    assert (plan.full, plan.piecewise) == (full, pw)


def test_install_is_local_idempotent_and_before_binding(runner: SimpleNamespace) -> None:
    original_dispatch = native.CudaGraphManager.dispatch
    pg2.install_prefill_graph_manager_v2(runner)
    installed = runner.cudagraph_manager
    assert isinstance(installed, pg2.GRPrefillCudaGraphManager)
    pg2.install_prefill_graph_manager_v2(runner)
    assert runner.cudagraph_manager is installed
    assert native.CudaGraphManager.dispatch is original_dispatch


@pytest.mark.parametrize(  # type: ignore[misc]
    "gate", ["env", "eager", "none", "tp", "dp", "pp", "ubatch", "lora", "spec", "aux", "mm"]
)
def test_install_keeps_native_for_disabled_or_unsupported(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    before = runner.cudagraph_manager
    if gate == "env":
        monkeypatch.setattr(PrefillGraphEnv, "enabled", False)
    elif gate == "eager":
        runner.model_config.enforce_eager = True
    elif gate == "none":
        before.cudagraph_mode = Mode.NONE
    elif gate in ("tp", "dp", "pp"):
        key = {"tp": "tensor", "dp": "data", "pp": "pipeline"}[gate]
        setattr(runner.parallel_config, key + "_parallel_size", 2)
    elif gate == "ubatch":
        runner.parallel_config.use_ubatching = True
    else:
        key = {
            "lora": "lora_config",
            "spec": "speculator",
            "aux": "use_aux_hidden_state_outputs",
            "mm": "supports_mm_inputs",
        }[gate]
        setattr(runner, key, True)
    pg2.install_prefill_graph_manager_v2(runner)
    assert runner.cudagraph_manager is before


def test_replacing_bound_manager_is_error(runner: SimpleNamespace) -> None:
    runner.cudagraph_manager.hidden_states = torch.empty(512, 4)
    with pytest.raises(RuntimeError, match="after output storage is bound"):
        pg2.install_prefill_graph_manager_v2(runner)


def test_capture_order_publication_registry_and_instance_scope(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = fake_capture(monkeypatch)
    manager = make_manager(runner)
    other = make_manager(runner)
    manager.capture()
    assert observed == [(m, k) for m in (Mode.PIECEWISE, Mode.FULL) for k in (512, 128, 16, 1)]
    assert manager.dispatch(1, 127, None).cg_mode == Mode.PIECEWISE
    with pytest.raises(ValueError), manager.prefill_scope(scheduled()):
        assert manager.dispatch(1, 127, None).cg_mode == Mode.FULL
        # A cached Prefill suffix can use FULL, while a Decode batch cannot.
        runner.req_states.num_computed_tokens_np[0] = 1024
        assert manager.dispatch(1, 127, None).cg_mode == Mode.PIECEWISE
        runner.req_states.num_computed_tokens_np[0] = 512
        assert other.dispatch(1, 127, None).cg_mode == Mode.NONE
        with manager.prefill_scope(None):
            assert manager.dispatch(1, 127, None).cg_mode == Mode.PIECEWISE
        desc = manager.dispatch(1, 127, None)
        del manager.graphs[desc]
        assert manager.dispatch(1, 127, None).cg_mode == Mode.PIECEWISE
        assert manager.dispatch(2, 127, None).cg_mode == Mode.PIECEWISE
        assert manager.dispatch(1, 513, None).cg_mode == Mode.NONE
        raise ValueError()
    assert manager._prefill_scope is None
    manager.capture()
    assert len(observed) == 8


def test_full_capture_unsupported_without_piecewise_falls_back_to_none(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(desc: native.BatchExecutionDescriptor) -> None:
        if desc.cg_mode == Mode.FULL and desc.num_tokens == 128:
            raise RuntimeError("CUDA error: operation not permitted when stream is capturing")

    fake_capture(monkeypatch, fail=fail)
    manager = make_manager(runner, piecewise=False)
    manager.capture()
    assert manager._graphs_captured
    assert set(manager.full_capture_failures) == {128}
    with manager.prefill_scope(scheduled()):
        assert manager.dispatch(1, 128, None).cg_mode == Mode.NONE
        assert manager.dispatch(1, 16, None).cg_mode == Mode.FULL
        assert manager.dispatch(1, 512, None).cg_mode == Mode.FULL


@pytest.mark.parametrize(  # type: ignore[misc]
    "message", ["shape mismatch", "illegal memory access", "device-side assert", "out of memory"]
)
def test_unknown_or_fatal_full_errors_abort_without_publication(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    def fail(desc: native.BatchExecutionDescriptor) -> None:
        if desc.cg_mode == Mode.FULL:
            raise RuntimeError(message)

    fake_capture(monkeypatch, fail=fail)
    manager = make_manager(runner)
    with pytest.raises(RuntimeError, match=message):
        manager.capture()
    assert not manager._graphs_captured
    with manager.prefill_scope(scheduled()):
        assert manager.dispatch(1, 128, None).cg_mode == Mode.NONE


def test_piecewise_failure_is_not_published(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(desc: native.BatchExecutionDescriptor) -> None:
        raise RuntimeError("PW failed")

    fake_capture(monkeypatch, fail=fail)
    manager = make_manager(runner)
    with pytest.raises(RuntimeError, match="PW failed"):
        manager.capture()
    assert not manager._successful and not manager._graphs_captured


def test_capture_error_chain_rejects_fatal_context() -> None:
    error = RuntimeError("operation not supported when stream is capturing")
    error.__context__ = RuntimeError(
        "operation not permitted when stream is capturing\n"
        "Compile with TORCH_USE_CUDA_DSA to enable device-side assertions."
    )
    assert pg2.is_recoverable_capture_error(error)
    error.__context__.__context__ = RuntimeError("illegal memory access")
    assert not pg2.is_recoverable_capture_error(error)


def test_unknown_error_wrapping_capture_error_is_not_recoverable() -> None:
    error = RuntimeError("output shape mismatch")
    error.__context__ = RuntimeError("operation not permitted when stream is capturing")
    assert not pg2.is_recoverable_capture_error(error)


# ---------------------------------------------------------------------------
# Capture guards
# ---------------------------------------------------------------------------


def test_invalidated_cuda_capture_is_not_recoverable() -> None:
    error = RuntimeError("operation failed due to a previous error during capture")
    error.__context__ = RuntimeError("operation not permitted when stream is capturing")
    assert not pg2.is_recoverable_capture_error(error)


def test_native_need_eager_bypasses_prefill_manager(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

    fake_capture(monkeypatch)
    manager = make_manager(runner)
    manager.capture()
    with manager.prefill_scope(scheduled()):
        desc, _ = dispatch_cg_and_sync_dp(manager, 1, 127, None, 1, 0, need_eager=True)
        assert desc.cg_mode == Mode.NONE and desc.num_tokens == 127
        assert manager.dispatch_counts["FULL"] == 0


def test_failed_cuda_health_check_aborts(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(desc: native.BatchExecutionDescriptor) -> None:
        if desc.cg_mode == Mode.FULL:
            raise RuntimeError("operation not permitted when stream is capturing")

    fake_capture(monkeypatch, fail=fail)

    def unhealthy(*args: Any) -> None:
        raise RuntimeError("device unhealthy")

    monkeypatch.setattr(torch.accelerator, "synchronize", unhealthy)
    manager = make_manager(runner)
    with pytest.raises(RuntimeError, match="device unhealthy"):
        manager.capture()
    assert not manager._graphs_captured
