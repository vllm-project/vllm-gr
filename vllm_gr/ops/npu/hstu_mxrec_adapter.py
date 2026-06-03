# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from typing import Optional

import torch

from vllm_gr.logger import init_logger
from vllm_gr.utils.npu_adapter import load_mxrec_ops

logger = init_logger(__name__)

MASK_TYPE_TRIL = 0
MASK_TYPE_NONE = 2


def _debug_validate_hstu_jagged_inputs(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_seq_len: Optional[int],
    contextual_seq_len: int,
    target_group_size: int,
    batch_size: int,
    num_targets: Optional[torch.Tensor],
) -> bool:
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: tensor dims must be 3D, got q=%s k=%s v=%s",
            q.shape,
            k.shape,
            v.shape,
        )
        return False
    if seq_offsets.dim() != 1 or seq_offsets.numel() < 2:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: seq_offsets must be 1D with >=2 elements, got shape=%s",
            tuple(seq_offsets.shape),
        )
        return False
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: token count mismatch q=%s k=%s v=%s",
            q.shape[0],
            k.shape[0],
            v.shape[0],
        )
        return False
    if k.shape[1] != v.shape[1]:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: K/V head count mismatch k=%s v=%s",
            k.shape[1],
            v.shape[1],
        )
        return False
    if q.shape[1] < k.shape[1] or q.shape[1] % k.shape[1] != 0:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: unsupported GQA head layout q_heads=%s k_heads=%s",
            q.shape[1],
            k.shape[1],
        )
        return False
    if q.shape[2] != k.shape[2]:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: QK head_dim mismatch q=%s k=%s",
            q.shape[2],
            k.shape[2],
        )
        return False
    if q.dtype != k.dtype or k.dtype != v.dtype:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: q/k/v dtype mismatch q=%s k=%s v=%s",
            q.dtype,
            k.dtype,
            v.dtype,
        )
        return False
    if q.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: unsupported dtype=%s",
            q.dtype,
        )
        return False
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        logger.debug(
            "[HSTU] mxrec.hstu_jagged note: q/k/v are not all contiguous; call site will materialize contiguous tensors before kernel launch",
        )
    q_heads = int(q.shape[1])
    k_heads = int(k.shape[1])
    qk_dim = int(q.shape[2])
    v_dim = int(v.shape[2])
    if not (1 <= q_heads <= 16):
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: q_heads=%s out of supported range [1, 16]",
            q_heads,
        )
        return False
    if not (1 <= k_heads <= 16):
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: k_heads=%s out of supported range [1, 16]",
            k_heads,
        )
        return False
    if not (1 <= qk_dim <= 512):
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: qk_dim=%s out of supported range [1, 512]",
            qk_dim,
        )
        return False
    if qk_dim % 16 != 0:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: qk_dim=%s must be a multiple of 16",
            qk_dim,
        )
        return False
    if not (16 <= v_dim <= 512):
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: v_dim=%s out of supported range [16, 512]",
            v_dim,
        )
        return False
    if v_dim % 16 != 0:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: v_dim=%s must be a multiple of 16",
            v_dim,
        )
        return False
    if not (1 <= batch_size <= 2048):
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: batch_size=%s out of supported range [1, 2048]",
            batch_size,
        )
        return False
    if max_seq_len is not None and not (1 <= int(max_seq_len) <= 20480):
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: max_seq_len=%s out of supported range [1, 20480]",
            max_seq_len,
        )
        return False
    if contextual_seq_len < 0 or contextual_seq_len > 256:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: contextual_seq_len=%s outside documented range [0, 256]",
            contextual_seq_len,
        )
        return False
    if target_group_size not in (0, 1, 3):
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: target_group_size=%s unsupported (expected one of {0, 1, 3})",
            target_group_size,
        )
        return False
    seq_offsets_cpu = seq_offsets.detach().to("cpu", dtype=torch.int64)
    if int(seq_offsets_cpu[0].item()) != 0:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: seq_offsets must start from 0, got first=%s",
            int(seq_offsets_cpu[0].item()),
        )
        return False
    if torch.any(seq_offsets_cpu[1:] < seq_offsets_cpu[:-1]).item():
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: seq_offsets must be nondecreasing, got %s",
            seq_offsets_cpu.tolist(),
        )
        return False
    if int(seq_offsets_cpu[-1].item()) != int(q.shape[0]):
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: seq_offsets[-1]=%s must equal token_count=%s",
            int(seq_offsets_cpu[-1].item()),
            int(q.shape[0]),
        )
        return False
    if num_targets is None:
        return True
    if num_targets.dim() != 1 or int(num_targets.numel()) != batch_size:
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: num_targets must be 1D with batch_size=%s, got shape=%s",
            batch_size,
            tuple(num_targets.shape),
        )
        return False
    num_targets_cpu = num_targets.detach().to("cpu", dtype=torch.int64)
    if torch.any(num_targets_cpu < 0).item():
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: num_targets must be nonnegative, got %s",
            num_targets_cpu.tolist(),
        )
        return False
    if torch.any(num_targets_cpu > 512).item():
        logger.debug(
            "[HSTU] mxrec.hstu_jagged debug reject: num_targets exceeds documented range [0, 512], got %s",
            num_targets_cpu.tolist(),
        )
        return False
    return True


def _mxrec_hstu_jagged_available() -> bool:
    if not load_mxrec_ops():
        logger.warning("[HSTU] mxrec.hstu_jagged unavailable: mxrec ops not loaded")
        return False
    try:
        torch.ops.mxrec.hstu_jagged
        return True
    except (RuntimeError, AttributeError) as e:
        logger.warning("[HSTU] mxrec.hstu_jagged unavailable: %s", e)
        return False


def can_run_hstu_jagged(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_seq_len: Optional[int] = None,
    num_targets: Optional[torch.Tensor] = None,
    contextual_seq_len: int = 0,
    target_group_size: int = 1,
) -> bool:
    logger.debug(
        "[HSTU] checking mxrec.hstu_jagged availability: "
        "q=%s k=%s v=%s seq_offsets=%s max_seq_len=%s contextual_seq_len=%s target_group_size=%s",
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        tuple(seq_offsets.shape),
        max_seq_len,
        contextual_seq_len,
        target_group_size,
    )
    if not _mxrec_hstu_jagged_available():
        return False
    if q.device.type != "npu" or k.device.type != "npu" or v.device.type != "npu":
        logger.warning(
            "[HSTU] mxrec.hstu_jagged skipped: q/k/v must be on NPU, got q=%s k=%s v=%s",
            q.device,
            k.device,
            v.device,
        )
        return False
    if logger.isEnabledFor(logging.DEBUG):
        batch_size = int(seq_offsets.numel()) - 1
        return _debug_validate_hstu_jagged_inputs(
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            contextual_seq_len=contextual_seq_len,
            target_group_size=target_group_size,
            batch_size=batch_size,
            num_targets=num_targets,
        )
    return True


def run_hstu_jagged(
    *,
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    num_targets: Optional[torch.Tensor],
    contextual_seq_len: int,
    target_group_size: int = 1,
) -> torch.Tensor:
    if not can_run_hstu_jagged(
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        max_seq_len=max_seq_len,
        num_targets=num_targets,
        contextual_seq_len=contextual_seq_len,
        target_group_size=target_group_size,
    ):
        raise RuntimeError("mxrec.hstu_jagged is unavailable or inputs are unsupported")
    batch_size = int(seq_offsets.numel()) - 1
    if batch_size <= 0:
        return q.new_empty((0, q.shape[1], v.shape[2]))

    seq_offsets_i32 = seq_offsets.to(device=q.device, dtype=torch.int32)

    if num_targets is None:
        num_targets_i32 = torch.zeros(batch_size, device=q.device, dtype=torch.int32)
    else:
        num_targets_i32 = num_targets.to(device=q.device, dtype=torch.int32)

    if contextual_seq_len > 0:
        num_context_i32 = torch.full(
            (batch_size,),
            int(contextual_seq_len),
            device=q.device,
            dtype=torch.int32,
        )
    else:
        num_context_i32 = torch.zeros(batch_size, device=q.device, dtype=torch.int32)

    mask_type = MASK_TYPE_TRIL if causal else MASK_TYPE_NONE
    silu_scale = 1.0 / max_seq_len

    logger.debug(
        "[HSTU] Calling mxrec.hstu_jagged: batch=%d max_seq=%d alpha=%.4f mask_type=%d",
        batch_size,
        max_seq_len,
        alpha,
        mask_type,
    )

    return torch.ops.mxrec.hstu_jagged(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        None,
        None,
        mask_type,
        int(max_seq_len),
        float(silu_scale),
        seq_offsets_i32,
        num_context_i32,
        num_targets_i32,
        int(target_group_size),
        float(alpha),
    )
