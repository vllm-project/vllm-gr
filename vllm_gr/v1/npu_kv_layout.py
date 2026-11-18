# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Narrow Ascend 0.30 allocation-policy bridge for dense Beam prefix KV."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable, Mapping
from functools import wraps
from importlib import import_module
from importlib.metadata import version
from typing import Any, cast

logger = logging.getLogger(__name__)
_MARKER = "_vllm_gr_contiguous_prefix_kv_policy"
_ORIGINAL = "_vllm_gr_contiguous_prefix_kv_original"
_HELPER = "requires_contiguous_pa_kv_cache"
_UTILS = "vllm_ascend.attention.utils"
_RUNNER = "vllm_ascend.worker.model_runner_v1"
_MISSING = object()
_ASCEND_COMPILATION_DEFAULTS = {
    "enable_npugraph_ex": False,
    "enable_static_kernel": False,
    "enable_super_kernel": False,
    "fuse_norm_quant": True,
    "fuse_qknorm_rope": True,
    "fuse_muls_add": True,
}
NativePolicy = Callable[[Any, Any, Any], bool]


def _supported_signature(function: Any) -> bool:
    if not callable(function):
        return False
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return False
    return tuple(parameter.name for parameter in parameters) == (
        "layer",
        "vllm_config",
        "spec",
    ) and all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters
    )


def _plain_beam_impl(layer: Any, beam_module: Any) -> bool:
    impl = getattr(layer, "impl", None)
    return (
        type(impl) is beam_module.BeamAttentionImpl
        and impl.attn_type == beam_module.AttentionType.DECODER
        and impl.sliding_window in (None, (-1, -1))
        and impl.kv_sharing_target_layer_name is None
        and getattr(layer, "kv_sharing_target_layer_name", None) is None
        and impl.alibi_slopes is None
        and impl.sinks is None
        and impl.logits_soft_cap == 0
        and impl.kv_cache_dtype in ("auto", "bfloat16", "float16")
        and not getattr(impl, "fa_quant_layer", False)
    )


def _eligible_beam_layer(layer: Any, vllm_config: Any, spec: Any) -> bool:
    """Decide only from this engine's explicit config and actual layer/spec.

    Unknown vendor features/layouts keep the native policy. In particular this
    does not turn native PA on or reinterpret an existing interleaved cache.
    """
    if getattr(vllm_config, "use_v2_model_runner", _MISSING) is not False:
        return False
    config_module = import_module("vllm_gr.config")
    gr = config_module.GRConfig.from_vllm_config(vllm_config)
    if (
        gr.attention_backend is not config_module.GRAttentionBackend.CUSTOM
        or gr.beam_execution_mode is not config_module.BeamExecutionMode.V1
        or type(gr.beam_max_width) is not int
        or gr.beam_max_width <= 0
        or (
            gr.npu_decode_execution is not None
            and gr.npu_decode_execution is not config_module.NPUDecodeExecution.XLLM
        )
    ):
        return False
    if os.environ.get("VLLM_KV_CACHE_LAYOUT") not in (None, "NHD", "LBNHC"):
        return False
    model = vllm_config.model_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    if (
        model.runner_type != "generate"
        or getattr(model, "is_hybrid", _MISSING) is not False
        or getattr(model, "quantization", _MISSING) is not None
        or getattr(vllm_config, "quant_config", _MISSING) is not None
        or getattr(vllm_config, "speculative_config", _MISSING) is not None
        or getattr(vllm_config, "kv_transfer_config", _MISSING) is not None
        or cache.cache_dtype not in ("auto", "bfloat16", "float16")
        or getattr(cache, "kv_sharing_fast_prefill", _MISSING) is not False
        or any(
            getattr(parallel, field, _MISSING) != 1
            for field in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        )
    ):
        return False
    additional = getattr(vllm_config, "additional_config", None)
    # Ascend 0.30 injects this compiler-default mapping during initialization.
    # It does not select a different KV allocation mode. Other vendor keys,
    # unknown compiler keys and nondefault values remain unqualified.
    if additional is not None and (
        not isinstance(additional, Mapping)
        or any(
            not isinstance(key, str)
            or (
                key != "vllm_gr"
                and not key.startswith("vllm_gr_")
                and (
                    key != "ascend_compilation_config"
                    or not isinstance(value, Mapping)
                    or any(
                        name not in _ASCEND_COMPILATION_DEFAULTS
                        or type(setting) is not bool
                        or setting is not _ASCEND_COMPILATION_DEFAULTS[name]
                        for name, setting in value.items()
                    )
                )
            )
            for key, value in additional.items()
        )
    ):
        return False

    # Resolve only after NPU CUSTOM Beam has been selected. Importing this
    # module or checking ordinary/CUDA configs never imports Beam's backends.
    beam = import_module("vllm_gr.v1.attention.backends.beam_attn")
    if (
        beam._attention_service().backend_id != "ascend"
        or beam._get_npu_decode_execution() != "xllm"
        or not _plain_beam_impl(layer, beam)
    ):
        return False
    if not import_module("vllm_gr.ops.npu.beam_ops").is_available():
        return False
    full_spec = import_module("vllm.v1.kv_cache_interface").FullAttentionSpec
    torch = import_module("torch")
    if (
        type(spec) is not full_spec
        or spec.dtype not in (torch.bfloat16, torch.float16)
        or spec.dtype != model.dtype
        # The exact upstream 0.30 FullAttentionSpec has no cache_dtype_str;
        # vendor extensions that do expose it still must be unquantized.
        or getattr(spec, "cache_dtype_str", None) not in (None, "auto", "bfloat16", "float16")
        or spec.page_size_bytes != spec.real_page_size_bytes
        or spec.real_page_size_bytes <= 0
        or spec.head_size != layer.impl.head_size
        or spec.num_kv_heads != layer.impl.num_kv_heads
    ):
        return False
    layers = vllm_config.compilation_config.static_forward_context
    return (
        isinstance(layers, Mapping)
        and bool(layers)
        and any(candidate is layer for candidate in layers.values())
        and all(_plain_beam_impl(candidate, beam) for candidate in layers.values())
    )


def _wrap_policy(original: NativePolicy) -> NativePolicy:
    @wraps(original)
    def policy(layer: Any, vllm_config: Any, spec: Any) -> bool:
        # Native results and native exceptions remain authoritative for its
        # ordinary Attention path, including already contiguous PA caches.
        native = original(layer, vllm_config, spec)
        if native:
            return native
        try:
            if _eligible_beam_layer(layer, vllm_config, spec):
                return True
        except Exception:
            logger.debug("NPU contiguous Beam KV policy kept native layout", exc_info=True)
        return native

    setattr(policy, _MARKER, True)
    setattr(policy, _ORIGINAL, original)
    return policy


def apply_npu_contiguous_kv_cache_patch() -> int:
    """Patch both known 0.30 aliases before native KV allocation, or do nothing."""
    try:
        if import_module("vllm.platforms").current_platform.device_type != "npu":
            return 0
        if version("vllm-ascend") != "0.30.0rc2":
            return 0
        utils = import_module(_UTILS)
        runner = import_module(_RUNNER)
        utility_policy = getattr(utils, _HELPER, None)
        runner_policy = getattr(runner, _HELPER, None)
        wrapped = [
            function
            for function in (utility_policy, runner_policy)
            if getattr(function, _MARKER, False)
        ]
        if wrapped:
            wrapper = wrapped[0]
            original = getattr(wrapper, _ORIGINAL, None)
            if (
                not _supported_signature(original)
                or any(function is not wrapper for function in wrapped)
                or any(
                    function is not original and function is not wrapper
                    for function in (utility_policy, runner_policy)
                )
            ):
                return 0
        else:
            if utility_policy is not runner_policy or not _supported_signature(utility_policy):
                return 0
            wrapper = _wrap_policy(cast(NativePolicy, utility_policy))
        changed = utility_policy is not wrapper or runner_policy is not wrapper
        setattr(utils, _HELPER, wrapper)
        setattr(runner, _HELPER, wrapper)
        return int(changed)
    except Exception:
        # This is an optional, version-specific policy extension. Never
        # replace a missing/unknown vendor allocation implementation.
        logger.debug("NPU contiguous Beam KV bridge unavailable", exc_info=True)
        return 0


__all__ = ["apply_npu_contiguous_kv_cache_patch"]
