# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""BeamAttentionPool — 5-D continuous KV pool for incremental decode.

Each pool holds a single contiguous tensor pair (key/value) of shape
``[max_batch, max_beams, num_kv_heads, max_decode_steps, head_dim]``.
The address is fixed at allocation time and never changes across steps
or sessions.

Each session occupies one slot (batch dimension).  ``reset_slot(slot)``
zeros only the KV for a single session, leaving other sessions untouched.
"""

from __future__ import annotations

import torch

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _validate_pool_dtype(dtype: torch.dtype) -> None:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"BeamAttentionPool dtype must be torch.float16 or torch.bfloat16, got {dtype}"
        )


class BeamAttentionPool:
    """Continuous decode KV storage pool (one per layer, one pair).

    Design principles:
    - 5-D contiguous tensor, fixed device address, never re-allocated.
    - Pre-allocated at init time with max configuration.
    - max_batch dimension: each session occupies one slot.
    """

    def __init__(
        self,
        max_batch: int,
        max_beams: int,
        num_kv_heads: int,
        max_decode_steps: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        _validate_pool_dtype(dtype)

        self.max_batch = max_batch
        self.max_beams = max_beams
        self.num_kv_heads = num_kv_heads
        self.max_decode_steps = max_decode_steps
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # 5-D contiguous tensor, fixed device address
        # [max_batch, max_beams, num_kv_heads, max_decode_steps, head_dim]
        pool_shape = (max_batch, max_beams, num_kv_heads, max_decode_steps, head_dim)
        self.unshared_key = torch.zeros(pool_shape, dtype=dtype, device=device)
        self.unshared_value = torch.zeros(pool_shape, dtype=dtype, device=device)

    def reset_slot(self, slot: int):
        """Zero out a single slot's KV for a new session."""
        self.unshared_key[slot].zero_()
        self.unshared_value[slot].zero_()


def write_eager_step(
    unshared_key: torch.Tensor,  # [max_batch, W, H, max_steps, D] pool key
    unshared_value: torch.Tensor,  # [max_batch, W, H, max_steps, D] pool value
    curr_key: torch.Tensor,  # [W_cur, H, D] current step's key
    curr_value: torch.Tensor,  # [W_cur, H, D] current step's value
    slot: int,  # batch slot index
    decode_step: int,  # 0-indexed, host int
    beam_session: object = None,  # Accepted by the platform KV-write interface.
) -> None:
    """Write the current step's W_cur beam KV into the pool's decode_step slice.

    Write address = pool base + slot offset + decode_step offset (fixed).

    Within the selected slot, the write covers the first ``curr_key.shape[0]``
    beam rows. If the live beam count shrinks, the suffix ``cu_seqlens_k`` only
    spans the live rows, so attention skips any stale KV in the trailing rows.
    """
    if unshared_key.dtype != unshared_value.dtype:
        raise TypeError(
            "Beam KV pool key/value dtype mismatch: "
            f"key={unshared_key.dtype}, value={unshared_value.dtype}"
        )
    if curr_key.dtype != curr_value.dtype:
        raise TypeError(
            "Current attention key/value dtype mismatch: "
            f"key={curr_key.dtype}, value={curr_value.dtype}"
        )
    if unshared_key.dtype != curr_key.dtype:
        raise TypeError(
            "Beam KV pool dtype must match the current attention K/V dtype: "
            f"pool={unshared_key.dtype}, current={curr_key.dtype}"
        )

    n = curr_key.shape[0]
    # Write into the pool's step slice for the given slot
    unshared_key[slot, :n, :, decode_step : decode_step + 1, :].copy_(
        curr_key.unsqueeze(2)  # [W_cur, H, 1, D] → broadcast into [W_cur, H, 1, D]
    )
    unshared_value[slot, :n, :, decode_step : decode_step + 1, :].copy_(curr_value.unsqueeze(2))
