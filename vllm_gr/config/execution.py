# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Beam startup requirements shared by frontends and workers."""

from typing import Any

from vllm_gr.config.models import GRConfig


def require_beam_config(config: object) -> None:
    if GRConfig.from_vllm_config(config).beam_max_width is None:
        raise ValueError("Beam search requires gr.beam_width configured at engine startup")


def validate_gr_execution_config(config: Any) -> None:
    """Reject unsupported Asynchronous scheduling runtimes before native/Worker request allocation."""
    require_beam_config(config)
    validate_gr_execution_runtime(config)


def _validate_hybrid_beam_runtime(config: Any) -> None:
    model = config.model_config
    device = getattr(getattr(config, "device_config", None), "device", None)
    if str(device).split(":", 1)[0] != "cuda":
        raise ValueError("Hybrid Beam requires CUDA")
    cache = config.cache_config
    cache_mode = "align" if cache.enable_prefix_caching else "none"
    if cache.mamba_cache_mode != cache_mode:
        raise ValueError(f"Hybrid Beam requires mamba_cache_mode={cache_mode}")
    if cache.cache_dtype not in ("auto", str(model.dtype).removeprefix("torch.")):
        raise ValueError("Hybrid Beam requires native unquantized attention cache")
    attention = getattr(config, "attention_config", None)
    backend = getattr(attention, "backend", None)
    gr = GRConfig.from_vllm_config(config)
    if getattr(backend, "name", backend) == "CUSTOM" or gr.attention_backend is not None:
        raise ValueError("Hybrid Beam requires native attention backends")
    scheduler = config.scheduler_config
    if gr.beam_max_width is not None and gr.beam_max_width > min(
        scheduler.max_num_seqs, scheduler.max_num_batched_tokens
    ):
        raise ValueError("Hybrid Beam width must fit the native sequence and token budgets")


def validate_gr_execution_runtime(config: Any) -> None:
    """Validate runtime capabilities after the persistent protocol is admitted."""
    if not config.scheduler_config.async_scheduling:
        raise ValueError("Asynchronous scheduling requires async_scheduling=True")
    model = getattr(config, "model_config", None)
    cache = getattr(config, "cache_config", None)
    hybrid_checkpoints = getattr(model, "is_hybrid", False) and (
        getattr(cache, "mamba_cache_mode", None) == "align"
    )
    if getattr(config.scheduler_config, "enable_chunked_prefill", False) and not hybrid_checkpoints:
        raise ValueError("beam_search requires enable_chunked_prefill=False")
    if getattr(config.scheduler_config, "long_prefill_token_threshold", 0):
        raise ValueError("beam_search does not support partial Prefill scheduling")
    parallel = getattr(config, "parallel_config", None)
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError("Asynchronous scheduling requires TP/PP/DP/DCP/PCP=1")
    if getattr(parallel, "use_ubatching", False):
        raise ValueError("Asynchronous scheduling does not support microbatching")
    for name in ("speculative_config", "lora_config", "kv_transfer_config", "ec_transfer_config"):
        if getattr(config, name, None) is not None:
            raise ValueError(f"Asynchronous scheduling does not support {name}")
    if any(getattr(model, name, False) for name in ("is_encoder_decoder", "is_multimodal_model")):
        raise ValueError("Asynchronous scheduling requires a text decoder model")
    if getattr(model, "is_hybrid", False):
        _validate_hybrid_beam_runtime(config)
    dtype = getattr(model, "dtype", None)
    if dtype is not None and str(dtype) not in ("torch.float16", "torch.bfloat16"):
        raise ValueError("Asynchronous scheduling requires FP16 or BF16")
    if getattr(model, "quantization", None) is not None:
        raise ValueError("Asynchronous scheduling does not support quantization")
