# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"
"""CPU regression coverage for the instance-local NPU projection hook."""

from __future__ import annotations

import sys
import weakref
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_gr.ops.npu_stage_logits import (
    NPUStageProjection,
    can_install_npu_stage_projection,
    install_npu_sid_stage_projection,
    install_npu_stage_projection,
    install_npu_stage_weight_update_hooks,
    make_npu_stage_reload_weights,
)


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    # The Ascend processor can be installed by earlier integration tests.
    # This TP=1 CPU fixture has no engine-wide Ascend config. Disable TP
    # and reduced sampling locally, keeping the real projection/processor.
    ascend_logits = sys.modules.get("vllm_ascend.ops.vocab_parallel_embedding")
    if ascend_logits is not None:
        monkeypatch.setattr(ascend_logits, "lmhead_tp_enable", lambda: False)
        monkeypatch.setattr(
            ascend_logits,
            "get_ascend_config",
            lambda: SimpleNamespace(enable_reduce_sample=False),
        )
    processor = LogitsProcessor(384)
    monkeypatch.setattr(processor, "_gather_logits", lambda logits: logits)
    generator = torch.Generator().manual_seed(41)
    head = SimpleNamespace(
        weight=torch.randn(384, 8, generator=generator),
        bias=torch.randn(384, generator=generator),
        quant_method=UnquantizedEmbeddingMethod(),
        tp_size=1,
    )
    ids = (
        torch.tensor([17, 100, 311], dtype=torch.int64),
        torch.tensor([3, 217], dtype=torch.int64),
        torch.tensor([1, 89, 197, 321], dtype=torch.int64),
    )
    plan = SimpleNamespace(execution=SimpleNamespace(stage_index=0, num_execution_rows=2))
    processors = [processor]
    model = SimpleNamespace(lm_head=head, modules=lambda: iter(processors))
    runner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
            lora_config=None,
            speculative_config=None,
        ),
        lora_config=None,
        _gr_npu_stage_plan=plan,
        _gr_npu_stage_runner=SimpleNamespace(pending_plan=plan),
    )
    backend = SimpleNamespace(
        stage_vocab=SimpleNamespace(vocab_size=128, stage_token_ids=ids),
        info=SimpleNamespace(table_metadata=SimpleNamespace(vocab_size=384)),
    )
    ctx = SimpleNamespace(
        runner=runner,
        backend=backend,
        processor=processor,
        processors=processors,
        head=head,
        hidden=torch.arange(16, dtype=torch.float32).reshape(2, 8) / 20,
        ids=ids,
    )
    monkeypatch.setenv("VLLM_GR_NPU_STAGE_LM_HEAD", "1")
    yield ctx
    installed = getattr(processor._get_logits, "__self__", None)
    if isinstance(installed, NPUStageProjection):
        installed.close()


def _install(ctx: Any) -> NPUStageProjection:
    projection = install_npu_stage_projection(ctx.runner, ctx.backend)
    assert projection is not None
    ctx.runner._gr_npu_stage_projection = projection
    return projection


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("scale,soft_cap", [(1.0, None), (0.7, None), (1.0, 13.0), (0.7, 13.0)])
def test_three_stage_projection_preserves_original_processor_semantics(
    context: Any, use_bias: bool, scale: float, soft_cap: float | None
) -> None:
    ctx = context
    processor, head = ctx.processor, ctx.head
    processor.scale, processor.soft_cap = scale, soft_cap
    if not use_bias:
        head.bias = None
    expected_full = F.linear(ctx.hidden, head.weight, head.bias)
    if soft_cap is not None:
        expected_full = torch.tanh(expected_full / soft_cap) * soft_cap
    expected_full *= scale
    torch.testing.assert_close(processor.forward(head, ctx.hidden, head.bias), expected_full)
    source_weight = head.weight.clone()
    projection = _install(ctx)

    for stage, ids in enumerate(ctx.ids):
        ctx.runner._gr_npu_stage_plan.execution.stage_index = stage
        actual = processor.forward(head, ctx.hidden, head.bias)
        assert actual.shape == (2, 128)
        torch.testing.assert_close(actual[:, : ids.numel()], expected_full[:, ids])
        torch.testing.assert_close(actual[:, ids.numel() :], torch.zeros(2, 128 - ids.numel()))
        torch.testing.assert_close(projection.stage_weights[stage][: ids.numel()], head.weight[ids])
    torch.testing.assert_close(head.weight, source_weight)
    snapshot = projection.snapshot()
    assert snapshot["projection_counts"] == [1, 1, 1]
    assert snapshot["stage_token_counts"] == [3, 2, 4]
    assert snapshot["weight_shapes"] == [[128, 8]] * 3
    assert snapshot["fallback_counts"] == {}


def test_steady_state_reuses_prepared_weights(
    context: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context
    projection = _install(ctx)
    pointers = tuple(weight.data_ptr() for weight in projection.stage_weights)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("steady-state projection rebuilt a stage weight")

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "clone", forbidden)
        scoped.setattr(torch.Tensor, "index_select", forbidden)
        for _ in range(2):
            for stage in range(3):
                ctx.runner._gr_npu_stage_plan.execution.stage_index = stage
                assert ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias).shape == (2, 128)
    assert tuple(weight.data_ptr() for weight in projection.stage_weights) == pointers
    assert projection.projection_counts == [2, 2, 2]


@pytest.mark.parametrize(
    "case,reason",
    [
        ("no_plan", "no_active_stage"),
        ("different_owner", "no_active_stage"),
        ("stage_out_of_range", "unsupported_stage"),
        ("boolean_stage", "unsupported_stage"),
        ("fractional_stage", "unsupported_stage"),
        ("row_count", "row_mismatch"),
        ("hidden_rank", "row_mismatch"),
        ("head", "different_lm_head"),
        ("bias", "different_bias"),
        ("hidden_dtype", "hidden_layout"),
    ],
)
def test_inactive_or_unsupported_dispatch_uses_original_full_projection(
    context: Any, case: str, reason: str
) -> None:
    ctx = context
    projection = _install(ctx)
    hidden, head, bias = ctx.hidden, ctx.head, ctx.head.bias
    if case == "no_plan":
        ctx.runner._gr_npu_stage_plan = None
    elif case == "different_owner":
        ctx.runner._gr_npu_stage_runner.pending_plan = SimpleNamespace()
    elif case == "stage_out_of_range":
        ctx.runner._gr_npu_stage_plan.execution.stage_index = 3
    elif case == "boolean_stage":
        ctx.runner._gr_npu_stage_plan.execution.stage_index = True
    elif case == "fractional_stage":
        ctx.runner._gr_npu_stage_plan.execution.stage_index = 0.5
    elif case == "row_count":
        ctx.runner._gr_npu_stage_plan.execution.num_execution_rows = 3
    elif case == "hidden_rank":
        hidden = hidden.reshape(1, 2, 8)
    elif case == "head":
        head = SimpleNamespace(**vars(head))
        head.weight = head.weight + 0.5
    elif case == "bias":
        bias = bias + 0.25
    elif case == "hidden_dtype":
        # The original head accepts this new dtype, while the prepared stage
        # tensors keep their original dtype and must not be selected.
        hidden = hidden.double()
        head.weight = head.weight.double()
        bias = head.bias = bias.double()
    expected = F.linear(hidden, head.weight, bias)
    actual = ctx.processor.forward(head, hidden, bias)
    torch.testing.assert_close(actual, expected)
    assert actual.shape[-1] == 384
    assert projection.projection_counts == [0, 0, 0]
    assert projection.snapshot()["fallback_counts"] == {reason: 1}


@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "invalid",
        "no_stage_vocab",
        "tensor_parallel",
        "pipeline_parallel",
        "lora",
        "speculative",
        "no_weight",
        "weight_rank",
        "weight_dtype",
        "quantized",
        "no_reduction",
        "bias_type",
        "bias_rank",
        "bias_size",
        "no_processor",
        "multiple_processors",
    ],
)
def test_install_rejects_unsupported_configs_without_changing_processor(
    context: Any, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context
    original = ctx.processor._get_logits
    ctx.processor._gr_vocab_slice = (17, 12)
    if case == "disabled":
        monkeypatch.setenv("VLLM_GR_NPU_STAGE_LM_HEAD", "0")
    elif case == "invalid":
        monkeypatch.setenv("VLLM_GR_NPU_STAGE_LM_HEAD", "true")
    elif case == "no_stage_vocab":
        ctx.backend.stage_vocab = None
    elif case == "tensor_parallel":
        ctx.runner.vllm_config.parallel_config.tensor_parallel_size = 2
    elif case == "pipeline_parallel":
        ctx.runner.vllm_config.parallel_config.pipeline_parallel_size = 2
    elif case == "lora":
        ctx.runner.lora_config = SimpleNamespace()
        ctx.runner.vllm_config.lora_config = ctx.runner.lora_config
    elif case == "speculative":
        ctx.runner.vllm_config.speculative_config = SimpleNamespace()
    elif case == "no_weight":
        ctx.head.weight = None
    elif case == "weight_rank":
        ctx.head.weight = ctx.head.weight.reshape(-1)
    elif case == "weight_dtype":
        ctx.head.weight = ctx.head.weight.to(torch.int8)
    elif case == "quantized":
        ctx.head.quant_method = SimpleNamespace()
    elif case == "no_reduction":
        ctx.backend.stage_vocab.vocab_size = 384
    elif case == "bias_type":
        ctx.head.bias = [0.0] * 384
    elif case == "bias_rank":
        ctx.head.bias = ctx.head.bias.unsqueeze(0)
    elif case == "bias_size":
        ctx.head.bias = ctx.head.bias[:8]
    elif case == "no_processor":
        ctx.processors.clear()
    elif case == "multiple_processors":
        ctx.processors.append(LogitsProcessor(384))
    assert install_npu_stage_projection(ctx.runner, ctx.backend) is None
    assert ctx.processor._get_logits == original
    assert "_get_logits" not in vars(ctx.processor)
    assert ctx.processor._gr_vocab_slice == (17, 12)


@pytest.mark.parametrize("previous_instance_method", [False, True])
@pytest.mark.parametrize("previous_window", [False, True])
def test_close_is_idempotent_and_restores_processor_state(
    context: Any, previous_instance_method: bool, previous_window: bool
) -> None:
    ctx = context
    original = ctx.processor._get_logits
    if previous_instance_method:
        ctx.processor._get_logits = original
    if previous_window:
        ctx.processor._gr_vocab_slice = (17, 12)
    projection = _install(ctx)
    assert ctx.processor._gr_vocab_slice is None
    weights = tuple(weakref.ref(weight) for weight in projection.stage_weights)
    projection.close()
    projection.close()
    assert ctx.processor._get_logits == original
    assert ("_get_logits" in vars(ctx.processor)) is previous_instance_method
    if previous_window:
        assert ctx.processor._gr_vocab_slice == (17, 12)
    else:
        assert "_gr_vocab_slice" not in vars(ctx.processor)
    assert projection.stage_weights == ()
    assert projection.runner is None and projection.processor is None and projection.lm_head is None
    assert all(ref() is None for ref in weights)
    assert not projection.snapshot()["enabled"]


def test_close_then_install_new_owner_uses_reloaded_weights(context: Any) -> None:
    ctx = context
    original = ctx.processor._get_logits
    old = _install(ctx)
    old_values = ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias).clone()
    old.close()
    torch.testing.assert_close(
        ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias),
        F.linear(ctx.hidden, ctx.head.weight, ctx.head.bias),
    )
    # Complete model replacement closes the previous owner before installing
    # a new hook. Supported hot-update APIs are exercised separately below.
    ctx.head = SimpleNamespace(
        weight=ctx.head.weight + 1.25,
        bias=ctx.head.bias + 0.5,
        quant_method=UnquantizedEmbeddingMethod(),
        tp_size=1,
    )
    ctx.runner.model.lm_head = ctx.head
    new = _install(ctx)
    assert new is not old
    expected_full = F.linear(ctx.hidden, ctx.head.weight, ctx.head.bias)
    for stage, ids in enumerate(ctx.ids):
        ctx.runner._gr_npu_stage_plan.execution.stage_index = stage
        actual = ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias)
        torch.testing.assert_close(actual[:, : ids.numel()], expected_full[:, ids])
        if stage == 0:
            assert not torch.allclose(actual, old_values)
    new.close()
    assert ctx.processor._get_logits == original
    torch.testing.assert_close(
        ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias), expected_full
    )


@pytest.mark.parametrize(
    "value,enabled", [(None, True), ("1", True), ("0", False), ("", False), ("true", False)]
)
def test_automatic_stage_projection_can_be_explicitly_disabled(
    context: Any, monkeypatch: pytest.MonkeyPatch, value: str | None, enabled: bool
) -> None:
    ctx = context
    if value is None:
        monkeypatch.delenv("VLLM_GR_NPU_STAGE_LM_HEAD")
    else:
        monkeypatch.setenv("VLLM_GR_NPU_STAGE_LM_HEAD", value)
    original = ctx.processor._get_logits
    assert can_install_npu_stage_projection(ctx.runner) is enabled
    owner = install_npu_stage_projection(ctx.runner, ctx.backend)
    assert (owner is not None) is enabled
    if owner is None:
        assert ctx.processor._get_logits == original
        assert "_get_logits" not in vars(ctx.processor)
    else:
        owner.close()


@pytest.mark.parametrize(
    "case",
    [
        "tensor_parallel",
        "pipeline_parallel",
        "lora",
        "speculative",
        "quantized",
        "multiple_processors",
    ],
)
def test_ineligible_runner_is_rejected_before_projection_allocation(
    context: Any, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    ctx = context
    if case == "tensor_parallel":
        ctx.runner.vllm_config.parallel_config.tensor_parallel_size = 2
    elif case == "pipeline_parallel":
        ctx.runner.vllm_config.parallel_config.pipeline_parallel_size = 2
    elif case == "lora":
        ctx.runner.lora_config = SimpleNamespace()
        ctx.runner.vllm_config.lora_config = ctx.runner.lora_config
    elif case == "speculative":
        ctx.runner.vllm_config.speculative_config = SimpleNamespace()
    elif case == "quantized":
        ctx.head.quant_method = SimpleNamespace()
    else:
        ctx.processors.append(LogitsProcessor(384))

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("An unsupported runner must not allocate compact projection weights")

    monkeypatch.setattr("vllm_gr.ops.npu_stage_logits.NPUStageProjection", forbidden)
    original = ctx.processor._get_logits
    assert not can_install_npu_stage_projection(ctx.runner)
    assert install_npu_stage_projection(ctx.runner, ctx.backend) is None
    assert ctx.processor._get_logits == original


def _projection_addresses(owner: NPUStageProjection) -> tuple[Any, ...]:
    return (
        tuple(weight.data_ptr() for weight in owner.stage_weights),
        tuple(None if bias is None else bias.data_ptr() for bias in owner._stage_biases),
        tuple((id(proxy), proxy.weight.data_ptr()) for proxy in owner._proxies),
    )


def _assert_current_stage_logits(ctx: Any) -> None:
    expected = F.linear(ctx.hidden, ctx.head.weight, ctx.head.bias)
    for stage, ids in enumerate(ctx.ids):
        ctx.runner._gr_npu_stage_plan.execution.stage_index = stage
        actual = ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias)
        assert actual.shape == (ctx.hidden.shape[0], 128)
        torch.testing.assert_close(actual[:, : ids.numel()], expected[:, ids])
        torch.testing.assert_close(
            actual[:, ids.numel() :], torch.zeros_like(actual[:, ids.numel() :])
        )


@pytest.mark.parametrize("replace_tensors", [False, True])
@pytest.mark.parametrize("install_in_inference_mode", [False, True])
def test_runner_reload_refreshes_cached_weights_and_bias_without_replacing_buffers(
    context: Any, replace_tensors: bool, install_in_inference_mode: bool
) -> None:
    ctx = context
    with torch.inference_mode(install_in_inference_mode):
        owner = _install(ctx)
    addresses = _projection_addresses(owner)
    original_method = ctx.processor._get_logits
    calls = []

    def native_reload(
        self: Any,
        weights_iterator: Any = None,
        weights_path: Any = None,
        is_checkpoint_format: bool = True,
    ) -> str:
        calls.append((weights_iterator, weights_path, is_checkpoint_format))
        with pytest.raises(RuntimeError):
            ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias)
        weight, bias = weights_iterator
        with torch.no_grad():
            if replace_tensors:
                self.model.lm_head.weight = weight.clone()
                self.model.lm_head.bias = bias.clone()
            else:
                self.model.lm_head.weight.copy_(weight)
                self.model.lm_head.bias.copy_(bias)
        return "native-reloaded"

    reload_weights = make_npu_stage_reload_weights(native_reload)
    for generation in (1, 2):
        weight = ctx.head.weight + generation * 0.25
        bias = ctx.head.bias + generation * 0.5
        assert (
            reload_weights(ctx.runner, (weight, bias), "unused", generation == 1)
            == "native-reloaded"
        )
        _assert_current_stage_logits(ctx)
        assert _projection_addresses(owner) == addresses
        assert ctx.processor._get_logits == original_method
        assert owner.snapshot()["weights_valid"]
        assert not owner.snapshot()["update_active"]
        assert not owner.snapshot()["update_failed"]
    assert len(calls) == 2
    assert owner.snapshot()["refresh_count"] == 2


def _streaming_worker(ctx: Any, *, fail_at: str | None = None) -> Any:
    """Double the public Ascend Worker transaction, including finish postprocessing."""

    class Worker:
        def __init__(self) -> None:
            self.model_runner = ctx.runner
            self.active = False
            self.events: list[str] = []

        def start_weight_update(self, is_checkpoint_format: bool = True) -> str:
            if self.active:
                raise RuntimeError("already active")
            self.active = True
            self.checkpoint = is_checkpoint_format
            self.events.append("start")
            if fail_at == "start":
                raise RuntimeError("start failed")
            return "started"

        def update_weights(self, update_info: dict[str, Any]) -> str:
            assert self.active
            self.events.append("checkpoint-chunk" if self.checkpoint else "kernel-chunk")
            if self.checkpoint:
                self.model_runner.model.load_weights(update_info["weights"])
            else:
                with torch.no_grad():
                    for name, value in update_info["weights"]:
                        getattr(ctx.head, name).copy_(value)
            if fail_at == "chunk":
                raise RuntimeError("chunk failed")
            return "chunk-loaded"

        def finish_weight_update(self) -> str:
            assert self.active
            self.events.append("finish")
            if self.checkpoint:
                # Represents finalize_layerwise_reload, after the final chunk.
                with torch.no_grad():
                    ctx.head.weight.mul_(1.1)
                    ctx.head.bias.add_(0.3)
            if fail_at == "finish":
                raise RuntimeError("finish failed")
            self.active = False
            return "finished"

    def checkpoint_load(weights: Any) -> None:
        with torch.no_grad():
            for name, value in weights:
                getattr(ctx.head, name).copy_(value)

    ctx.runner.model.load_weights = checkpoint_load
    install_npu_stage_weight_update_hooks(Worker)
    return Worker()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_worker_multiple_chunks_refresh_only_after_native_finish(
    context: Any, checkpoint: bool
) -> None:
    ctx = context
    owner = _install(ctx)
    addresses = _projection_addresses(owner)
    worker = _streaming_worker(ctx)
    for generation in (1, 2):
        assert worker.start_weight_update(is_checkpoint_format=checkpoint) == "started"
        assert owner.snapshot()["update_active"]
        for name, value in (
            ("weight", ctx.head.weight + generation),
            ("bias", ctx.head.bias + generation / 2),
        ):
            assert worker.update_weights({"weights": [(name, value)]}) == "chunk-loaded"
            assert owner.snapshot()["refresh_count"] == generation - 1
            with pytest.raises(RuntimeError):
                ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias)
        assert worker.finish_weight_update() == "finished"
        _assert_current_stage_logits(ctx)
        assert _projection_addresses(owner) == addresses
        assert owner.snapshot()["refresh_count"] == generation
    assert (
        worker.events
        == [
            "start",
            "checkpoint-chunk" if checkpoint else "kernel-chunk",
            "checkpoint-chunk" if checkpoint else "kernel-chunk",
            "finish",
        ]
        * 2
    )


@pytest.mark.parametrize("failure", ["start", "chunk", "finish"])
def test_failed_worker_update_cannot_serve_partial_weights(context: Any, failure: str) -> None:
    ctx = context
    owner = _install(ctx)
    worker = _streaming_worker(ctx, fail_at=failure)
    if failure == "start":
        with pytest.raises(RuntimeError, match="start failed"):
            worker.start_weight_update()
    else:
        worker.start_weight_update()
        if failure == "chunk":
            with pytest.raises(RuntimeError, match="chunk failed"):
                worker.update_weights({"weights": [("weight", ctx.head.weight + 1)]})
        else:
            worker.update_weights({"weights": [("weight", ctx.head.weight + 1)]})
            with pytest.raises(RuntimeError, match="finish failed"):
                worker.finish_weight_update()
    assert owner.snapshot()["update_failed"]
    assert not owner.snapshot()["weights_valid"]
    with pytest.raises(RuntimeError):
        ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias)
    # Re-entering public APIs cannot turn a partially updated model into a
    # serving projection. Complete model load is covered by lifecycle tests.
    with pytest.raises(RuntimeError):
        worker.start_weight_update()
    with pytest.raises(RuntimeError):
        ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias)


@pytest.mark.parametrize(
    "replacement", ["weight_shape", "weight_dtype", "bias_removed", "bias_added"]
)
def test_successful_but_incompatible_reload_uses_current_full_projection(
    context: Any, replacement: str
) -> None:
    ctx = context
    if replacement == "bias_added":
        ctx.head.bias = None
    owner = _install(ctx)
    addresses = _projection_addresses(owner)

    def native_reload(self: Any) -> None:
        if replacement == "weight_shape":
            ctx.head.weight = ctx.head.weight[:, :4].clone()
            ctx.hidden = ctx.hidden[:, :4]
        elif replacement == "weight_dtype":
            ctx.head.weight = ctx.head.weight.double()
            ctx.head.bias = ctx.head.bias.double()
            ctx.hidden = ctx.hidden.double()
        elif replacement == "bias_removed":
            ctx.head.bias = None
        else:
            ctx.head.bias = torch.ones(384, dtype=ctx.head.weight.dtype)

    make_npu_stage_reload_weights(native_reload)(ctx.runner)
    actual = ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias)
    torch.testing.assert_close(actual, F.linear(ctx.hidden, ctx.head.weight, ctx.head.bias))
    assert _projection_addresses(owner) == addresses
    assert not owner.snapshot()["weights_valid"]
    assert not owner.snapshot()["update_failed"]


@pytest.mark.parametrize("closed_owner", [False, True])
def test_weight_update_hooks_preserve_native_behavior_without_active_projection(
    context: Any, closed_owner: bool
) -> None:
    ctx = context
    if closed_owner:
        _install(ctx).close()
    calls = []

    def native_reload(self: Any, *args: Any, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return "untouched-native-result"

    reload_weights = make_npu_stage_reload_weights(native_reload)
    assert make_npu_stage_reload_weights(reload_weights) is reload_weights
    assert (
        reload_weights(ctx.runner, "iterator", is_checkpoint_format=False)
        == "untouched-native-result"
    )
    assert calls == [(("iterator",), {"is_checkpoint_format": False})]
    worker = _streaming_worker(ctx)
    wrapped_start = type(worker).start_weight_update
    install_npu_stage_weight_update_hooks(type(worker))
    assert type(worker).start_weight_update is wrapped_start
    assert worker.start_weight_update(False) == "started"
    assert worker.update_weights({"weights": [("weight", ctx.head.weight + 1)]}) == "chunk-loaded"
    assert worker.finish_weight_update() == "finished"


@pytest.mark.npu_integration
@pytest.mark.parametrize("update_api", ["runner", "worker"])
def test_npu_hot_update_refreshes_inference_buffers_at_native_boundary(
    context: Any, update_api: str
) -> None:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    ctx = context
    ctx.head.weight = ctx.head.weight.to("npu")
    ctx.head.bias = ctx.head.bias.to("npu")
    ctx.hidden = ctx.hidden.to("npu")
    ctx.ids = tuple(ids.to("npu") for ids in ctx.ids)
    ctx.backend.stage_vocab.stage_token_ids = ctx.ids
    with torch.inference_mode():
        owner = _install(ctx)
    addresses = _projection_addresses(owner)
    if update_api == "worker":
        worker = _streaming_worker(ctx)
    else:

        def native_reload(self: Any, weight: torch.Tensor, bias: torch.Tensor) -> None:
            with torch.no_grad():
                self.model.lm_head.weight.copy_(weight)
                self.model.lm_head.bias.copy_(bias)

        reload_weights = make_npu_stage_reload_weights(native_reload)
    for generation in (1, 2):
        weight, bias = ctx.head.weight + generation / 10, ctx.head.bias + generation / 20
        if update_api == "worker":
            worker.start_weight_update(is_checkpoint_format=generation == 1)
            worker.update_weights({"weights": [("weight", weight)]})
            worker.update_weights({"weights": [("bias", bias)]})
            worker.finish_weight_update()
        else:
            reload_weights(ctx.runner, weight, bias)
        _assert_current_stage_logits(ctx)
        assert _projection_addresses(owner) == addresses
        assert owner.snapshot()["refresh_count"] == generation


def _install_sid(ctx: Any) -> tuple[NPUStageProjection, Any]:
    from vllm_gr.ops.npu.sid_stage_vocab import NPUSIDStageVocab

    vocab = NPUSIDStageVocab(384, max(ids.numel() for ids in ctx.ids), ctx.ids)
    # This path is intentionally independent of the table Backend.
    ctx.backend.stage_vocab = None
    owner = install_npu_sid_stage_projection(ctx.runner, vocab)
    assert owner is not None
    ctx.runner._gr_npu_stage_projection = owner
    return owner, vocab


@pytest.mark.parametrize("use_bias", [False, True])
def test_sid_projection_preserves_stage_scores_padding_and_ordinary_fallback(
    context: Any, use_bias: bool
) -> None:
    from vllm_gr.ops.npu.sid_stage_vocab import select_sid_stage_candidates

    ctx = context
    if not use_bias:
        ctx.head.bias = None
    ctx.processor.scale, ctx.processor.soft_cap = 0.7, 13.0
    full = ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias)
    owner, vocab = _install_sid(ctx)
    for stage, ids in enumerate(ctx.ids):
        ctx.runner._gr_npu_stage_plan.execution.stage_index = stage
        compact = ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias)
        assert compact.shape == (2, vocab.vocab_size)
        torch.testing.assert_close(compact[:, : ids.numel()], full.index_select(1, ids))
        torch.testing.assert_close(
            compact[:, ids.numel() :], torch.zeros_like(compact[:, ids.numel() :])
        )
        temperature = torch.tensor([0.0, 0.5])
        baseline = select_sid_stage_candidates(full, temperature, 6, stage, vocab)
        selected = select_sid_stage_candidates(
            compact, temperature, 6, stage, vocab, compact_logits=True
        )
        torch.testing.assert_close(selected.token_ids, baseline.token_ids, atol=0, rtol=0)
        torch.testing.assert_close(selected.logprobs, baseline.logprobs)
        assert torch.equal(selected.valid_mask, baseline.valid_mask)
    assert owner.projection_counts == [1, 1, 1]
    ctx.runner._gr_npu_stage_plan = None
    torch.testing.assert_close(ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias), full)
    assert owner.snapshot()["fallback_counts"] == {"no_active_stage": 1}
    owner.close()
    torch.testing.assert_close(ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias), full)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_sid_projection_reuses_transactional_weight_cache_refresh(
    context: Any, checkpoint: bool
) -> None:
    ctx = context
    owner, vocab = _install_sid(ctx)
    addresses = _projection_addresses(owner)
    worker = _streaming_worker(ctx)
    worker.start_weight_update(is_checkpoint_format=checkpoint)
    worker.update_weights({"weights": [("weight", ctx.head.weight + 0.25)]})
    worker.update_weights({"weights": [("bias", ctx.head.bias + 0.5)]})
    with pytest.raises(RuntimeError, match="cannot run during"):
        ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias)
    worker.finish_weight_update()
    full = F.linear(ctx.hidden, ctx.head.weight, ctx.head.bias)
    for stage, ids in enumerate(ctx.ids):
        ctx.runner._gr_npu_stage_plan.execution.stage_index = stage
        actual = ctx.processor.forward(ctx.head, ctx.hidden, ctx.head.bias)
        assert actual.shape[-1] == vocab.vocab_size
        torch.testing.assert_close(actual[:, : ids.numel()], full.index_select(1, ids))
        assert not actual[:, ids.numel() :].count_nonzero()
    assert _projection_addresses(owner) == addresses
    assert owner.snapshot()["refresh_count"] == 1
    assert owner.snapshot()["weights_valid"]


@pytest.mark.parametrize("reason", ["disabled", "lora", "quantized"])
def test_sid_projection_remains_optional_for_ineligible_or_disabled_models(
    context: Any, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    from vllm_gr.ops.npu.sid_stage_vocab import NPUSIDStageVocab

    ctx = context
    vocab = NPUSIDStageVocab(384, 4, ctx.ids)
    original = ctx.processor._get_logits
    if reason == "disabled":
        monkeypatch.setenv("VLLM_GR_NPU_STAGE_LM_HEAD", "0")
    elif reason == "lora":
        ctx.runner.lora_config = SimpleNamespace()
    else:
        ctx.head.quant_method = SimpleNamespace()
    assert install_npu_sid_stage_projection(ctx.runner, vocab) is None
    assert ctx.processor._get_logits == original
