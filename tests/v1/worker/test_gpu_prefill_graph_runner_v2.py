# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Unified V2 Prefill plan, capture publication, fallback and instance isolation."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode as Mode
from vllm.v1.worker.gpu import cudagraph_utils as native
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from tests.helpers.gr_config import set_deployment
from vllm_gr.v1.worker import prefill_graph_common as common
from vllm_gr.v1.worker.cuda.v2 import gpu_prefill_graph_runner_v2 as pg2
from vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_resources import close_v2_runtime
from vllm_gr.v1.worker.model_runner_patch import (
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)


@pytest.fixture  # type: ignore[misc]
def runner(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        use_ubatching=False,
    )
    compilation = SimpleNamespace(
        cudagraph_capture_sizes=[1, 2],
        max_cudagraph_capture_size=2,
        adjust_cudagraph_sizes_for_spec_decode=lambda *args: None,
    )
    config = SimpleNamespace(
        additional_config={"vllm_gr": {"gr": {"prefill_token_buckets": None}}},
        parallel_config=parallel,
        compilation_config=compilation,
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=512),
        speculative_config=None,
        cache_config=SimpleNamespace(use_kda_recoverssm=False),
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
        vllm_config=config,
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

    def capture(manager: Any, *args: Any, **kwargs: Any) -> None:
        descs = next(iter(manager._capture_descs.values()))
        for desc in descs:
            observed.append((desc.cg_mode, desc.num_tokens))
            assert manager.dispatch(1, 128, None).cg_mode == Mode.NONE
            if fail:
                fail(desc)
            if desc.cg_mode == Mode.FULL:
                manager.graphs[desc] = object()
        manager._graphs_captured = True

    monkeypatch.setattr(native.ModelCudaGraphManager, "capture", capture)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args: None)
    return observed


@pytest.mark.parametrize(  # type: ignore[misc]
    "sizes,max_seqs,expected",
    [
        (None, 1, {1}),
        (None, 3, {1, 2}),
        (None, 8, {1, 2, 4}),
        ((1, 3, 7), 7, {1, 3, 7}),
        ((1, 6, 8), 8, {1, 6, 8}),
    ],
)
def test_capture_policy_is_not_engine_capacity(
    runner: SimpleNamespace, sizes: tuple[int, ...] | None, max_seqs: int, expected: set[int]
) -> None:
    runner.scheduler_config.max_num_seqs = max_seqs
    runner.cudagraph_manager.compilation_config.cudagraph_capture_sizes = [128]
    if sizes is not None:
        runner.vllm_config.additional_config = {
            "vllm_gr": {
                "gr": {"beam_batch_sizes": list(sizes)},
            }
        }
    plan = pg2.PrefillCapturePlan.for_runner(runner)
    assert set(plan.batch_sizes) == expected
    assert {d.num_reqs for d in plan.descriptors()[Mode.FULL]} == expected


@pytest.mark.parametrize("override", ["1,2,4", "3,7", "8"])  # type: ignore[misc]
def test_prefill_capture_respects_configured_prompt_counts(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    runner.cudagraph_manager.compilation_config.cudagraph_capture_sizes = [128]
    set_deployment(runner.vllm_config, beam_batch_sizes=list(map(int, override.split(","))))
    plan = pg2.PrefillCapturePlan.for_runner(runner)
    expected = {int(b) for b in override.split(",") if int(b) <= 4}
    assert {d.num_reqs for d in plan.descriptors()[Mode.FULL]} == expected
    assert plan.descriptors()[Mode.PIECEWISE]
    assert runner.scheduler_config.max_num_seqs == 4


def test_plan_preserves_native_capture_budget(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(common, "get_default_buckets", lambda config=None: [128, 256, 1024])
    set_deployment(runner.vllm_config, prefill_kv_bound=2048)
    runner.cudagraph_manager.compilation_config.cudagraph_capture_sizes = [1, 2, 384, 2048]
    plan = pg2.PrefillCapturePlan.for_runner(runner)
    assert plan.buckets == (1, 2, 384)
    descs = plan.descriptors()
    assert descs[Mode.FULL][0].num_tokens == descs[Mode.PIECEWISE][0].num_tokens == 384
    assert [d.num_tokens for d in descs[Mode.FULL] if d.num_reqs == 1] == list(
        reversed(plan.buckets)
    )
    assert {d.num_reqs for d in descs[Mode.FULL]} == {1, 2, 4}
    assert all(
        d.num_tokens >= d.num_reqs and d.uniform_token_count is None for d in descs[Mode.FULL]
    )
    assert all(d.num_reqs is None for d in descs[Mode.PIECEWISE])
    runner.model_config.max_model_len = 200
    plan = pg2.PrefillCapturePlan.for_runner(runner)
    assert plan.buckets[-1] == 384
    assert max(d.num_tokens for d in plan.descriptors()[Mode.FULL] if d.num_reqs == 1) == 2
    assert native.BatchExecutionDescriptor(Mode.FULL, 384, 2) in plan.descriptors()[Mode.FULL]


@pytest.mark.parametrize("piecewise", [False, True])  # type: ignore[misc]
def test_full_capture_admits_native_dummy_lengths_and_preserves_fallback(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, piecewise: bool
) -> None:
    sizes = [8187, 8189, 8190, 8191, 8192]
    runner.scheduler_config.max_num_batched_tokens = 8192
    runner.cudagraph_manager.compilation_config.cudagraph_capture_sizes = sizes
    runner.cudagraph_manager.cudagraph_mode = Mode.FULL_AND_PIECEWISE if piecewise else Mode.FULL
    plan = pg2.PrefillCapturePlan.for_runner(runner)
    assert plan.buckets == tuple(sizes)
    buffers = InputBuffers(4, 8192, torch.device("cpu"))
    full = plan.descriptors()[Mode.FULL]
    assert {(desc.num_reqs, desc.num_tokens) for desc in full} == {(4, size) for size in sizes}
    # Use upstream's actual split as the oracle instead of copying its formula.
    for size in sizes:
        dummy = InputBatch.make_dummy(4, size, buffers)
        fits = int(dummy.num_scheduled_tokens.max()) <= runner.model_config.max_model_len
        assert (native.BatchExecutionDescriptor(Mode.FULL, size, 4) in full) is fits

    observed = fake_capture(monkeypatch)
    pg2.install_prefill_graph_manager_v2(runner)
    manager = runner.cudagraph_manager
    manager.capture()
    assert [size for mode, size in observed if mode == Mode.FULL] == list(reversed(sizes))
    assert not manager.full_capture_failures
    runner.req_states.req_id_to_index = {f"r{i}": i for i in range(4)}
    runner.req_states.num_computed_tokens_np = [0] * 4
    runner.req_states.prompt_len.np = [2048] * 4
    for size in sizes:
        # Real requests can distribute the remainder without exceeding 2048.
        counts = {f"r{i}": size // 4 + int(i < size % 4) for i in range(4)}
        with manager.prefill_scope(SimpleNamespace(num_scheduled_tokens=counts)):
            desc = manager.dispatch(4, size, None)
            assert desc.cg_mode == Mode.FULL
            # Preserve the exact bucket rather than selecting a larger graph.
            assert desc.num_tokens == size


@pytest.mark.parametrize("sizes", [[], [1, 2, 16, 128, 1024, 2048, 4096]])  # type: ignore[misc]
def test_v2_native_capture_sizes_do_not_change_v1_grid(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, sizes: list[int]
) -> None:
    from vllm_gr.config import GRConfig
    from vllm_gr.config.models import DeploymentOptions
    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    monkeypatch.setattr(
        GRConfig,
        "from_vllm_config",
        lambda _: SimpleNamespace(
            beam_max_width=128,
            deployment=DeploymentOptions(prefill_token_buckets=None),
        ),
    )
    runner.device = torch.device("cpu")
    runner.model_config.max_model_len = 1000
    runner.scheduler_config.max_num_batched_tokens = 4096
    runner.cudagraph_manager.compilation_config.cudagraph_capture_sizes = sizes
    v1 = GPUPrefillGraphRunner(runner)
    v2 = pg2.PrefillCapturePlan.for_runner(runner)
    assert v1.buckets == [128, 256, 384, 512, 640, 768, 896, 1000]
    assert v1.match_bucket(900) == 1000
    assert v1.match_bucket(1001) is None
    assert v2.buckets == tuple(t for t in sizes if t <= 4000)
    shapes = {(d.num_reqs, d.num_tokens) for d in v2.descriptors().get(Mode.FULL, [])}
    assert (1, 1024) not in shapes
    assert (2, 2048) not in shapes
    assert ((2, 1024) in shapes) == (1024 in sizes)
    assert ((4, 2048) in shapes) == (2048 in sizes)


@pytest.mark.l1  # type: ignore[misc]
def test_explicit_token_buckets_override_native_capture_sizes(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_deployment(runner.vllm_config, prefill_token_buckets=[128, 256])
    runner.cudagraph_manager.compilation_config.cudagraph_capture_sizes = [1, 2, 512]
    assert pg2.PrefillCapturePlan.for_runner(runner).buckets == (128, 256)


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
        set_deployment(runner.vllm_config, prefill_graph_enabled=False)
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


@pytest.mark.parametrize("v2", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("operation", ["reload", "shutdown"])  # type: ignore[misc]
def test_prefill_cleanup_precedes_failed_native_lifecycle(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, v2: bool, operation: str
) -> None:
    def fail(desc: native.BatchExecutionDescriptor) -> None:
        if desc.cg_mode == Mode.FULL and desc.num_tokens == 128:
            raise RuntimeError("operation not supported when stream is capturing")

    fake_capture(monkeypatch, fail=fail)
    manager = runner.cudagraph_manager = make_manager(runner)
    manager.capture()
    storage = manager.hidden_states = torch.empty(512, 4)
    manager.aux_hidden_states = [storage]
    manager.intermediate_tensors = SimpleNamespace(tensors={"hidden": storage})
    manager.use_aux_hidden_state_outputs = True
    manager._prefill_scope = scheduled()
    assert manager.dispatch(1, 128, None).cg_mode == Mode.PIECEWISE

    def native_operation(self: Any) -> None:
        # Invalidation must happen before native loading/shutdown can fail.
        assert (self.cudagraph_manager is manager) is not v2
        assert manager._capture_finished is not v2
        raise RuntimeError("native failure")

    options = {"close_runtime": close_v2_runtime} if v2 else {}
    wrapped = (
        make_load_model_with_constraint_backend(native_operation, **options)
        if operation == "reload"
        else make_shutdown_with_constraint_backend(native_operation, **options)
    )
    with pytest.raises(RuntimeError, match="native failure"):
        wrapped(runner)
    if not v2:
        assert manager.hidden_states is storage
        assert manager.graphs and manager._graphs_captured and manager._successful
        assert manager.full_capture_failures and manager.dispatch_counts["PIECEWISE"] == 1
        return

    assert not manager.graphs and manager.hidden_states is None
    assert not manager.aux_hidden_states and manager.intermediate_tensors is None
    assert not manager.use_aux_hidden_state_outputs
    assert not manager._graphs_captured and not manager.needs_capture()
    assert not manager._successful and not manager.full_capture_failures
    assert manager._prefill_scope is None and not any(manager.dispatch_counts.values())
    assert manager.dispatch(1, 128, None).cg_mode == Mode.NONE
    replacement = runner.cudagraph_manager
    assert type(replacement) is native.ModelCudaGraphManager
    assert not replacement.graphs and replacement.hidden_states is None
    assert not replacement._graphs_captured
    close_v2_runtime(runner)
    assert runner.cudagraph_manager is replacement


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


@pytest.mark.parametrize("limit", [0, 1, 2, None])  # type: ignore[misc]
def test_profiling_limit_applies_to_whole_full_plan(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, limit: int | None
) -> None:
    observed = fake_capture(monkeypatch)
    manager = make_manager(runner)
    plan = manager._capture_descs
    manager._max_full_descs_to_capture = limit
    manager.capture()
    expected_full = [512, 128, 16, 1][:limit]
    assert observed == [(Mode.PIECEWISE, n) for n in (512, 128, 16, 1)] + [
        (Mode.FULL, n) for n in expected_full
    ]
    assert manager._capture_descs is plan
    assert len(plan[Mode.FULL]) == 4  # Native extrapolation still sees the total.


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
    assert set(manager.full_capture_failures) == {
        native.BatchExecutionDescriptor(Mode.FULL, 128, 1)
    }
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


def test_full_capture_failure_and_dispatch_use_exact_request_count(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(desc: native.BatchExecutionDescriptor) -> None:
        if desc == native.BatchExecutionDescriptor(Mode.FULL, 128, 2):
            raise RuntimeError("operation not supported when stream is capturing")

    fake_capture(monkeypatch, fail=fail)
    manager = pg2.GRPrefillCudaGraphManager(
        runner, pg2.PrefillCapturePlan((16, 128, 512), True, True, (1, 2, 4), 2048)
    )
    runner.req_states.req_id_to_index = {f"r{i}": i for i in range(4)}
    runner.req_states.num_computed_tokens_np = [0] * 4
    runner.req_states.prompt_len.np = [128] * 4
    manager.capture()
    assert set(manager.full_capture_failures) == {
        native.BatchExecutionDescriptor(Mode.FULL, 128, 2)
    }
    for b in (1, 2, 4):
        with manager.prefill_scope(
            SimpleNamespace(num_scheduled_tokens={f"r{i}": 32 for i in range(b)})
        ):
            desc = manager.dispatch(b, 128, None)
            assert desc.cg_mode == (Mode.PIECEWISE if b == 2 else Mode.FULL)
            if b != 2:
                assert desc.num_reqs == b


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


@pytest.mark.l1  # type: ignore[misc]
def test_dense_prefill_capture_scope_and_fresh_replay(
    runner: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_gr.v1 import forward_context

    scopes: list[bool] = []

    def capture(manager: Any, *args: Any, **kwargs: Any) -> None:
        scopes.append(forward_context.in_fresh_prefill_capture())
        for desc in next(iter(manager._capture_descs.values())):
            if desc.cg_mode == Mode.FULL:
                manager.graphs[desc] = object()
        manager._graphs_captured = True

    monkeypatch.setattr(native.ModelCudaGraphManager, "capture", capture)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args: None)
    manager = make_manager(runner)
    manager.capture()
    # Every GR Prefill sub-plan captures inside the fresh-Prefill scope.
    assert scopes and all(scopes)
    assert not forward_context.in_fresh_prefill_capture()
    with manager.prefill_scope(scheduled()):
        runner.req_states.num_computed_tokens_np[0] = 512  # cached Prefill suffix
        assert manager.dispatch(1, 127, None).cg_mode == Mode.FULL
    set_deployment(runner.vllm_config, dense_prefill="fa2")
    manager = make_manager(runner)
    manager.capture()
    with manager.prefill_scope(scheduled()):
        assert manager.dispatch(1, 127, None).cg_mode == Mode.PIECEWISE
        runner.req_states.num_computed_tokens_np[0] = 0
        assert manager.dispatch(1, 127, None).cg_mode == Mode.FULL


pytestmark = pytest.mark.cpu_test
