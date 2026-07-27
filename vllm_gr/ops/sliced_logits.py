# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Sliced Logits Projection for vLLM / GR decoding.

Optimizes output projection (lm_head) by slicing the weight matrix to a contiguous
subset of the vocabulary (e.g., 24k tokens out of 170k).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from vllm.logger import init_logger

logger = init_logger(__name__)

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


def init_vocab_slice_config_from_env() -> None:
    """Reads vocabulary slice configuration from environment variables once."""
    global _VOCAB_OFFSET, _VOCAB_SLICE_SIZE
    _VOCAB_OFFSET = None
    if "VLLM_GR_VOCAB_OFFSET" in os.environ:
        try:
            _VOCAB_OFFSET = int(os.environ["VLLM_GR_VOCAB_OFFSET"])
        except ValueError:
            pass

    _VOCAB_SLICE_SIZE = None
    if "VLLM_GR_VOCAB_SLICE_SIZE" in os.environ:
        try:
            _VOCAB_SLICE_SIZE = int(os.environ["VLLM_GR_VOCAB_SLICE_SIZE"])
        except ValueError:
            pass


# Read initial environment variables once at module load
init_vocab_slice_config_from_env()


def set_vocab_slice_config(
    offset: Optional[int],
    slice_size: Optional[int],
    quant_group_size: int = 128
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


def compute_sliced_logits(
    hidden_states: torch.Tensor,
    lm_head: Any,
    embedding_bias: Optional[torch.Tensor] = None,
    offset: Optional[int] = None,
    slice_size: Optional[int] = None,
    quant_group_size: int = 128
) -> torch.Tensor:
    """
    Computes logits using pre-sliced, contiguous vocabulary projection weights.
    Fully optimized for torch.compile and CUDA graph execution.
    """
    cfg_offset, cfg_slice_size, _ = get_vocab_slice_config()
    off = offset if offset is not None else cfg_offset
    sz = slice_size if slice_size is not None else cfg_slice_size

    if not hasattr(lm_head, "_sliced_weight") or getattr(lm_head, "_sliced_offset", None) != off or getattr(lm_head, "_sliced_size", None) != sz:

        if off is None or sz is None:
            raise ValueError("Vocabulary slice offset and slice_size must be specified.")

        offset_aligned = (off // quant_group_size) * quant_group_size
        end_idx = ((off + sz + quant_group_size - 1) // quant_group_size) * quant_group_size
        aligned_slice_size = end_idx - offset_aligned

        sliced_w = lm_head.weight[offset_aligned : offset_aligned + aligned_slice_size].detach().clone().contiguous()
        if not isinstance(sliced_w, nn.Parameter):
            sliced_w = nn.Parameter(sliced_w, requires_grad=False)

        lm_head._sliced_weight = sliced_w
        lm_head._sliced_bias = (
            embedding_bias[offset_aligned : offset_aligned + aligned_slice_size].detach().clone().contiguous()
            if embedding_bias is not None else None
        )
        lm_head._sliced_offset = off
        lm_head._sliced_size = sz
        lm_head._sliced_rel_start = off - offset_aligned
        lm_head._sliced_rel_end = lm_head._sliced_rel_start + sz
        lm_head._sliced_need_crop = (aligned_slice_size > sz)        
        lm_head._sliced_proxy = SlicedLMHeadProxy(lm_head, lm_head._sliced_weight, aligned_slice_size)
        logger.info(
            f"vllm_gr: Cached pre-sliced contiguous lm_head weights (offset={off}, "
            f"slice_size={sz}, aligned range=[{offset_aligned}:{end_idx}])"
        )

    # Check if a custom quantization method is active
    quant_method = getattr(lm_head, "quant_method", None)
    if quant_method is not None:
        logits = quant_method.apply(lm_head._sliced_proxy, hidden_states, bias=lm_head._sliced_bias)
    else:
        logits = F.linear(hidden_states, lm_head._sliced_weight, lm_head._sliced_bias)

    if lm_head._sliced_need_crop:
        logits = logits[..., lm_head._sliced_rel_start : lm_head._sliced_rel_end]
    return logits