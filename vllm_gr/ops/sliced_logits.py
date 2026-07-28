# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Sliced Logits Projection for vLLM / GR decoding.

Optimizes output projection (lm_head) by slicing the weight matrix to a contiguous
subset of the vocabulary (e.g., 24k tokens out of 170k).
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger("vllm.ops.sliced_logits")

# Global configuration variables for static offset mode
_VOCAB_OFFSET: Optional[int] = None
_VOCAB_SLICE_SIZE: Optional[int] = None
_QUANT_GROUP_SIZE: int = 128


class SlicedLMHeadProxy(nn.Module):
    """Lightweight proxy object for quantized lm_head apply."""

    def __init__(self, original_lm_head, sliced_weight, sliced_num_embeddings):
        super().__init__()
        if not isinstance(sliced_weight, nn.Parameter):
            sliced_weight = nn.Parameter(sliced_weight, requires_grad=False)
        self.weight = sliced_weight
        self.quant_method = getattr(original_lm_head, "quant_method", None)
        self.num_embeddings_per_partition = sliced_num_embeddings
        self.org_vocab_size = sliced_num_embeddings


def set_vocab_slice_config(
    offset: Optional[int], slice_size: Optional[int], quant_group_size: int = 128
) -> None:
    """Configures global vocabulary slice settings."""
    global _VOCAB_OFFSET, _VOCAB_SLICE_SIZE, _QUANT_GROUP_SIZE
    _VOCAB_OFFSET = offset
    _VOCAB_SLICE_SIZE = slice_size
    _QUANT_GROUP_SIZE = quant_group_size


def get_vocab_slice_config() -> Tuple[Optional[int], Optional[int], int]:
    """Retrieves current global vocabulary slice settings."""
    return _VOCAB_OFFSET, _VOCAB_SLICE_SIZE, _QUANT_GROUP_SIZE


def is_vocab_slice_enabled() -> bool:
    """Returns True if a valid vocabulary slice is configured."""
    return _VOCAB_OFFSET is not None and _VOCAB_SLICE_SIZE is not None and _VOCAB_SLICE_SIZE > 0


def get_rank_vocab_bounds(lm_head: Any) -> Tuple[int, int]:
    """Retrieves the global vocabulary index range [start, end) owned by the current TP rank."""
    if hasattr(lm_head, "shard_indices") and hasattr(
        lm_head.shard_indices, "org_vocab_start_index"
    ):
        return (
            int(lm_head.shard_indices.org_vocab_start_index),
            int(lm_head.shard_indices.org_vocab_end_index),
        )
    if hasattr(lm_head, "vocab_start_index") and hasattr(lm_head, "vocab_end_index"):
        return int(lm_head.vocab_start_index), int(lm_head.vocab_end_index)

    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
    except Exception:
        tp_rank = 0
        tp_size = 1

    shard_size = lm_head.weight.shape[0]
    if tp_size > 1:
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return start_idx, end_idx

    return 0, shard_size


def is_supported_unquantized_lm_head(lm_head: Any) -> bool:
    """Returns True if lm_head is unquantized and safe for weight slicing."""
    quant_method = getattr(lm_head, "quant_method", None)
    if quant_method is None:
        return True

    quant_cls_name = quant_method.__class__.__name__
    return quant_cls_name in (
        "UnquantizedEmbeddingMethod",
        "UnquantizedLinearMethod",
        "UnquantizedBaseMethod",
    )


def compute_sliced_logits(
    hidden_states: torch.Tensor,
    lm_head: Any,
    embedding_bias: Optional[torch.Tensor] = None,
    offset: Optional[int] = None,
    slice_size: Optional[int] = None,
    quant_group_size: int = 128,
    logits_processor: Optional[Any] = None,
    orig_get_logits: Optional[Any] = None,
) -> torch.Tensor:
    """
    Computes logits using pre-sliced, contiguous vocabulary projection weights for unquantized models,
    or via a safe logits-post-slicing fallback for quantized models.
    Fully optimized for torch.compile and CUDA graph execution.
    """
    cfg_offset, cfg_slice_size, _ = get_vocab_slice_config()
    off = offset if offset is not None else cfg_offset
    sz = slice_size if slice_size is not None else cfg_slice_size

    if off is None or sz is None:
        raise ValueError("Vocabulary slice offset and slice_size must be specified.")

    # Safe fallback for quantized models where weight slicing is unsupported/unsafe
    if not is_supported_unquantized_lm_head(lm_head):
        if not getattr(lm_head, "_quant_slice_fallback_warned", False):
            logger.info(
                "vllm_gr: Quantized lm_head detected (%s). Sliced weight optimization "
                "is disabled for quantized models; using safe output-logits slicing fallback.",
                getattr(getattr(lm_head, "quant_method", None), "__class__", {}).__name__,
            )
            lm_head._quant_slice_fallback_warned = True

        if logits_processor is not None and orig_get_logits is not None:
            full_logits = orig_get_logits(logits_processor, hidden_states, lm_head, embedding_bias)
        else:
            quant_method = getattr(lm_head, "quant_method", None)
            if quant_method is not None:
                full_logits = quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
            else:
                full_logits = F.linear(hidden_states, lm_head.weight, embedding_bias)

        if full_logits is not None and full_logits.numel() > 0:
            return full_logits[..., off : off + sz]
        return full_logits

    rank_start, rank_end = get_rank_vocab_bounds(lm_head)
    global_start = off
    global_end = off + sz

    inter_start = max(global_start, rank_start)
    inter_end = min(global_end, rank_end)

    if inter_start >= inter_end:
        # Rank has no overlap with the requested global slice range
        batch_shape = hidden_states.shape[:-1]
        empty_shape = (*batch_shape, 0)
        return torch.empty(empty_shape, dtype=hidden_states.dtype, device=hidden_states.device)

    local_start = inter_start - rank_start
    local_end = inter_end - rank_start
    local_sz = local_end - local_start

    cache_key = (off, sz, rank_start, rank_end)
    if (
        not hasattr(lm_head, "_sliced_weight")
        or getattr(lm_head, "_sliced_cache_key", None) != cache_key
    ):
        offset_aligned = (local_start // quant_group_size) * quant_group_size
        end_idx = ((local_end + quant_group_size - 1) // quant_group_size) * quant_group_size
        aligned_slice_size = end_idx - offset_aligned

        sliced_w = (
            lm_head.weight[offset_aligned : offset_aligned + aligned_slice_size]
            .detach()
            .clone()
            .contiguous()
        )
        if not isinstance(sliced_w, nn.Parameter):
            sliced_w = nn.Parameter(sliced_w, requires_grad=False)

        lm_head._sliced_weight = sliced_w
        lm_head._sliced_bias = (
            embedding_bias[offset_aligned : offset_aligned + aligned_slice_size]
            .detach()
            .clone()
            .contiguous()
            if embedding_bias is not None
            else None
        )
        lm_head._sliced_cache_key = cache_key
        lm_head._sliced_rel_start = local_start - offset_aligned
        lm_head._sliced_rel_end = lm_head._sliced_rel_start + local_sz
        lm_head._sliced_need_crop = aligned_slice_size > local_sz
        lm_head._sliced_proxy = SlicedLMHeadProxy(
            lm_head, lm_head._sliced_weight, aligned_slice_size
        )
        logger.info(
            f"vllm_gr: Cached pre-sliced contiguous lm_head weights (global range=[{inter_start}:{inter_end}], "
            f"local offset={offset_aligned}, aligned slice_size={aligned_slice_size})"
        )

    proxy = lm_head._sliced_proxy
    if logits_processor is not None and orig_get_logits is not None:
        logits = orig_get_logits(logits_processor, hidden_states, proxy, lm_head._sliced_bias)
    else:
        quant_method = getattr(lm_head, "quant_method", None)
        if quant_method is not None:
            logits = quant_method.apply(proxy, hidden_states, bias=lm_head._sliced_bias)
        else:
            logits = F.linear(hidden_states, lm_head._sliced_weight, lm_head._sliced_bias)

    if lm_head._sliced_need_crop and logits is not None and logits.numel() > 0:
        logits = logits[..., lm_head._sliced_rel_start : lm_head._sliced_rel_end]

    return logits
