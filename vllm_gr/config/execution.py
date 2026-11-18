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


def validate_gr_execution_runtime(config: Any) -> None:
    """Validate runtime capabilities after the persistent protocol is admitted."""
    if not config.scheduler_config.async_scheduling:
        raise ValueError("Asynchronous scheduling requires async_scheduling=True")
    if getattr(config.scheduler_config, "enable_chunked_prefill", False):
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
    model = getattr(config, "model_config", None)
    if any(
        getattr(model, name, False)
        for name in ("is_encoder_decoder", "is_hybrid", "is_multimodal_model")
    ):
        raise ValueError("Asynchronous scheduling requires a uniform text decoder model")
    dtype = getattr(model, "dtype", None)
    if dtype is not None and str(dtype) not in ("torch.float16", "torch.bfloat16"):
        raise ValueError("Asynchronous scheduling requires FP16 or BF16")
    if getattr(model, "quantization", None) is not None:
        raise ValueError("Asynchronous scheduling does not support quantization")
