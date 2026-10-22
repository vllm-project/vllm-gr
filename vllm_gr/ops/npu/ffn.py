# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, instance-local Qwen Gate/Up + SwiGLU fusion on Ascend.

The entire MLP is opaque to Dynamo so the real shape and stream capture state
select the route at execution/capture time, independently of the decoding phase.
Unsupported inputs call the saved original MLP, including its native modules.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)
ENV_ENABLE = "VLLM_GR_NPU_FUSED_FFN"
_LOAD_MARKER = "_gr_npu_fused_ffn_load_hook"
_BINDING_ATTR = "_gr_npu_fused_ffn_binding"
_COMPILE_MARKER_KEY = "_vllm_gr_npu_ffn"
_COMPILE_OWNER_ATTR = "_gr_npu_fused_ffn_compile_marker"
# Bump this when the opaque implementation's compilation contract changes.
_COMPILE_SCHEMA = 7
_OPERATOR_LIBRARY: Any = None
_KERNEL: Callable[..., torch.Tensor] | None = None
_MISSING = object()


@dataclass
class _Binding:
    layer: Any
    name: str
    context: dict[str, Any]
    original_forward: Callable[[torch.Tensor], torch.Tensor]
    previous_forward: Any
    fused_forward: Any
    prefetch_method: Any
    is_capturing: Callable[[], bool] | None


@dataclass
class _CompileMarker:
    config: Any
    previous: dict[str, Any]
    value: dict[str, Any]


def _prefetch_reason(method: Any) -> str | None:
    mlp = getattr(method, "mlp", None)
    if (
        method is None
        or getattr(mlp, "enable", None) is not False
        or getattr(mlp, "is_active_this_forward", None) is not False
    ):
        return "MLP weight prefetch is active or its no-op state is unknown"
    return None


def _current_prefetch_method() -> Any:
    from vllm_ascend.utils import get_weight_prefetch_method

    return get_weight_prefetch_method()


def _has_hooks(module: Any) -> bool:
    return any(
        bool(getattr(module, name, None))
        for name in ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks")
    )


def _layer_reason(layer: Any, config: Any, qwen_mlp: Any, torch_npu: Any) -> str | None:
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm_ascend.ops.activation import AscendSiluAndMul

    if type(layer) is not qwen_mlp:
        return "different MLP implementation"
    if "forward" in layer.__dict__:
        return "MLP already has an instance forward override"
    if getattr(config, "lora_config", None) is not None or getattr(config, "quant_config", None):
        return "LoRA or quantization configuration"
    parallel = getattr(config, "parallel_config", None)
    if getattr(parallel, "tensor_parallel_size", None) != 1:
        return "TP is not one"
    if type(layer.act_fn) not in (SiluAndMul, AscendSiluAndMul):
        return "different activation implementation"
    for projection, shape in (
        (layer.gate_up_proj, (6144, 1024)),
        (layer.down_proj, (1024, 3072)),
    ):
        weight = getattr(projection, "weight", None)
        if (
            getattr(projection, "tp_size", None) != 1
            or getattr(projection, "bias", None) is not None
            or not isinstance(getattr(projection, "quant_method", None), UnquantizedLinearMethod)
        ):
            return "projection TP, bias or quantization method"
        if (
            not isinstance(weight, torch.Tensor)
            or tuple(weight.shape) != shape
            or weight.device.type != "npu"
            or weight.dtype != torch.bfloat16
        ):
            return "weight shape, device or dtype"
    weight = layer.gate_up_proj.weight
    if not weight.is_contiguous() or torch_npu.get_npu_format(weight) not in (0, 2):
        return "Gate/Up weight is not contiguous base ND"
    if layer.down_proj.weight.device != weight.device:
        return "projection devices differ"
    if _has_hooks(layer.gate_up_proj) or _has_hooks(layer.act_fn):
        return "Gate/Up or activation has registered hooks"
    return None


def _hardware_reason(weight: torch.Tensor, torch_npu: Any) -> str | None:
    device_name = torch_npu.npu.get_device_name(weight.device)
    if device_name != "Ascend910B4":
        return f"AscendC FFN requires validated Ascend910B4 hardware, got {device_name!r}"
    return None


def _runtime_supported(x: torch.Tensor, binding: _Binding) -> bool:
    import torch_npu

    layer = binding.layer
    weight = layer.gate_up_proj.weight
    # M1024 has a measured native-GEMM rounding mismatch. Keep the
    # complete original MLP; this is a routing safety exclusion, not
    # a repair of the experimental AscendC kernel.
    if x.ndim == 2 and x.shape[0] == 1024:
        return False
    if (
        x.ndim != 2
        or not 1 <= x.shape[0] <= 16384
        or x.shape[1] != 1024
        or x.device.type != "npu"
        or x.device != weight.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (6144, 1024)
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        return False
    if torch_npu.get_npu_format(x) not in (0, 2) or torch_npu.get_npu_format(weight) not in (0, 2):
        return False
    method = _current_prefetch_method()
    if method is not binding.prefetch_method or _prefetch_reason(method) is not None:
        return False
    return not (_has_hooks(layer.gate_up_proj) or _has_hooks(layer.act_fn))


def _ffn_impl(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    from vllm.forward_context import get_forward_context

    layer = get_forward_context().no_compile_layers[layer_name]
    binding: _Binding = getattr(layer, _BINDING_ATTR)
    if not _runtime_supported(x, binding) or not _route_supported(x, binding):
        return binding.original_forward(x)
    if _KERNEL is None:
        raise RuntimeError("NPU fused FFN kernel was not prepared before model execution")
    intermediate = _KERNEL(x, layer.gate_up_proj.weight)
    # Keep the original RowParallelLinear implementation, its hooks and weights.
    output: torch.Tensor
    output, _ = layer.down_proj(intermediate)
    return output


def _route_supported(x: torch.Tensor, binding: _Binding) -> bool:
    rows = x.shape[0]
    # Exclude M1024 until a kernel-level numerical fix is validated.
    if rows in (512, 4096, 16384):
        return True
    # The query remains inside the opaque runtime implementation. AOT cannot
    # freeze a warmup answer; graph replay uses the nodes selected at capture.
    return rows == 128 and binding.is_capturing is not None and binding.is_capturing() is False


def _routing_policy() -> dict[str, Any]:
    return {
        "name": "ascendc_graph_config_exact_rows_m1024_native_v1",
        "uncaptured_rows": [128, 512, 4096, 16384],
        "capture_rows": [512, 4096, 16384],
        "requires_prefill_graph": True,
    }


def _ffn_fake(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    del layer_name
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _load_backend() -> Any:
    library_path = os.environ.get("GR_FFN_OPAPI_LIBRARY", "")
    if not os.path.isabs(library_path) or not os.path.isfile(library_path):
        raise ImportError(
            "AscendC FFN requires GR_FFN_OPAPI_LIBRARY to name the existing "
            "absolute path to its dedicated libgr_ffn_opapi.so"
        )
    # Check the same handle and symbols execution will use before any MLP
    # binding or compile marker is published. This does not run a device op.
    import gr_ffn_ops

    fused_gate_up_swiglu = getattr(gr_ffn_ops, "fused_gate_up_swiglu", None)
    if not callable(fused_gate_up_swiglu):
        raise ImportError("AscendC FFN extension has no callable fused_gate_up_swiglu")
    preflight = getattr(gr_ffn_ops, "preflight", None)
    if not callable(preflight):
        raise ImportError("AscendC FFN extension has no callable preflight; rebuild gr_ffn_ops")
    preflight()
    return fused_gate_up_swiglu


def _prepare_operator() -> None:
    global _KERNEL, _OPERATOR_LIBRARY

    if _OPERATOR_LIBRARY is not None:
        return
    kernel: Callable[..., torch.Tensor] = _load_backend()
    library = torch.library.Library("vllm_gr", "FRAGMENT")
    library.define("npu_fused_ffn(Tensor x, str layer_name) -> Tensor")
    # This opaque boundary is NPU-specific. Do not expose a composite Python
    # decomposition that can inspect forward_context during AOT tracing.
    library.impl("npu_fused_ffn", _ffn_impl, "PrivateUse1")
    library.impl("npu_fused_ffn", _ffn_fake, "Meta")
    _KERNEL = kernel
    _OPERATOR_LIBRARY = library


def _restore_compile_marker(runner: Any) -> None:
    owner = getattr(runner, _COMPILE_OWNER_ATTR, None)
    if not isinstance(owner, _CompileMarker):
        return
    current = owner.config.additional_config
    if isinstance(current, dict) and current.get(_COMPILE_MARKER_KEY) is owner.value:
        restored = dict(current)
        del restored[_COMPILE_MARKER_KEY]
        # Preserve the caller's mapping when it has not otherwise changed. An
        # identity comparison also works for config values such as tensors.
        unchanged = len(restored) == len(owner.previous) and all(
            key in restored and restored[key] is value for key, value in owner.previous.items()
        )
        owner.config.additional_config = owner.previous if unchanged else restored
    delattr(runner, _COMPILE_OWNER_ATTR)


def _publish_compile_marker(
    runner: Any, config: Any, previous: dict[str, Any], bindings: list[_Binding]
) -> None:
    if config.additional_config is not previous:
        raise ValueError("additional_config changed while preparing NPU fused FFN")
    marker = {
        "schema": _COMPILE_SCHEMA,
        "backend": "ascendc",
        "policy": _routing_policy(),
        "layers": sorted(binding.name for binding in bindings),
    }
    # A config dict can be shared by callers/engines. Publish a per-engine copy,
    # never mutate their mapping, and retain identity-based ownership for undo.
    updated = dict(previous)
    updated[_COMPILE_MARKER_KEY] = marker
    config.additional_config = updated
    setattr(runner, _COMPILE_OWNER_ATTR, _CompileMarker(config, previous, marker))


def _restore_bindings(runner: Any) -> None:
    for binding in getattr(runner, "_gr_npu_fused_ffn_bindings", ()):
        layer = binding.layer
        if layer.__dict__.get("forward") is binding.fused_forward:
            if binding.previous_forward is _MISSING:
                delattr(layer, "forward")
            else:
                layer.forward = binding.previous_forward
        if binding.context.get(binding.name) is layer:
            del binding.context[binding.name]
        if getattr(layer, _BINDING_ATTR, None) is binding:
            delattr(layer, _BINDING_ATTR)
    runner._gr_npu_fused_ffn_bindings = ()
    runner._gr_npu_fused_ffn_backend = None
    _restore_compile_marker(runner)


def _graph_configuration_reason(config: Any) -> str | None:
    from vllm.config.compilation import CUDAGraphMode

    from vllm_gr.v1.worker.prefill_graph_common import is_prefill_graph_enabled

    if getattr(getattr(config, "model_config", None), "enforce_eager", None) is not False:
        return "eager execution is enabled or graph eligibility is unknown"
    mode = getattr(getattr(config, "compilation_config", None), "cudagraph_mode", None)
    if not isinstance(mode, CUDAGraphMode) or mode == CUDAGraphMode.NONE:
        return "graph execution is disabled or its mode is unknown"
    if not is_prefill_graph_enabled():
        return "Prefill graph execution is disabled"
    return None


def install_npu_fused_ffn(runner: Any) -> None:
    """Bind eligible loaded Qwen2/Qwen3 MLP instances before the first trace.

    No parameter or weight storage is replaced. Dependency/configuration misses
    leave the original model callable and produce one startup warning.
    The isolated AscendC extension is only enabled for graph configurations
    with GR Prefill capture enabled. Their uncaptured warmup calls retain the
    same validated shape policy; standalone eager configurations stay native.
    Actual kernel execution and graph hits require external validation.
    Actual bindings enter additional_config before the first forward so both
    vLLM AOT and backend caches distinguish fusion from native fallback.
    """
    _restore_bindings(runner)
    if os.environ.get(ENV_ENABLE) != "1":
        return
    bindings: list[_Binding] = []
    try:
        config = runner.vllm_config
        graph_reason = _graph_configuration_reason(config)
        if graph_reason is not None:
            logger.warning("NPU fused FFN requested; retaining native MLP: %s", graph_reason)
            return
        import torch_npu
        from vllm.model_executor.models.qwen2 import Qwen2MLP

        is_capturing = getattr(torch_npu.npu, "is_current_stream_capturing", None)
        if not callable(is_capturing):
            raise ImportError("AscendC FFN requires torch_npu.npu.is_current_stream_capturing")
        additional_config = config.additional_config
        if not isinstance(additional_config, dict):
            raise ValueError("NPU fused FFN requires a dict additional_config for cache isolation")
        if _COMPILE_MARKER_KEY in additional_config:
            raise ValueError(f"additional_config already owns reserved key {_COMPILE_MARKER_KEY!r}")
        context = config.compilation_config.static_forward_context
        method = _current_prefetch_method()
        prefetch_reason = _prefetch_reason(method)
        if prefetch_reason is not None:
            logger.warning("NPU fused FFN requested; retaining native MLP: %s", prefetch_reason)
            return
        candidates = []
        reasons: Counter[str] = Counter()
        for name, layer in runner.model.named_modules():
            if type(layer) is not Qwen2MLP:
                continue
            reason = _layer_reason(layer, config, Qwen2MLP, torch_npu)
            if reason is None:
                reason = _hardware_reason(layer.gate_up_proj.weight, torch_npu)
            if reason is not None:
                reasons[reason] += 1
                continue
            key = f"{name}.vllm_gr_npu_fused_ffn"
            if key in context:
                raise ValueError(f"NPU fused FFN context key is already registered: {key}")
            candidates.append((key, layer))
        if not candidates:
            logger.warning("NPU fused FFN requested; no eligible MLP instances: %s", dict(reasons))
            return
        _prepare_operator()
        for key, layer in candidates:
            fused_forward = partial(torch.ops.vllm_gr.npu_fused_ffn, layer_name=key)
            binding = _Binding(
                layer=layer,
                name=key,
                context=context,
                original_forward=layer.forward,
                previous_forward=layer.__dict__.get("forward", _MISSING),
                fused_forward=fused_forward,
                prefetch_method=method,
                is_capturing=is_capturing,
            )
            context[key] = layer
            setattr(layer, _BINDING_ATTR, binding)
            layer.forward = fused_forward
            bindings.append(binding)
        runner._gr_npu_fused_ffn_bindings = tuple(bindings)
        runner._gr_npu_fused_ffn_backend = "ascendc"
        _publish_compile_marker(runner, config, additional_config, bindings)
        logger.info(
            "NPU fused FFN backend=ascendc prepared for %d Qwen MLP layers: BF16 H=1024 I=3072, "
            "ND routing=%s; native Down and unsupported-input fallback. "
            "Actual kernel hits require validation; excluded layers=%s",
            len(bindings),
            _routing_policy(),
            dict(reasons),
        )
    except Exception as exc:
        runner._gr_npu_fused_ffn_bindings = tuple(bindings)
        _restore_bindings(runner)
        logger.warning("NPU fused FFN unavailable; retaining native MLP: %s", exc, exc_info=True)


def install_npu_fused_ffn_worker_hook(worker_class: Any) -> bool:
    """Install the opt-in hook after the native worker has loaded its model."""
    if os.environ.get(ENV_ENABLE) != "1":
        return False
    original = getattr(worker_class, "load_model", None)
    if not callable(original):
        logger.warning("NPU fused FFN requested, but NPUWorker has no load_model hook")
        return False
    if getattr(original, _LOAD_MARKER, False):
        return False

    @wraps(original)
    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        runner = getattr(self, "model_runner", None)
        if runner is not None:
            _restore_bindings(runner)
        result = original(self, *args, **kwargs)
        runner = getattr(self, "model_runner", None)
        if runner is None:
            logger.warning("NPU fused FFN requested, but loaded worker has no model_runner")
        else:
            install_npu_fused_ffn(runner)
        return result

    setattr(load_model, _LOAD_MARKER, True)
    worker_class.load_model = load_model
    return True
