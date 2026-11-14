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
        # Optional second bank, owned by BeamAttentionContext. Both addresses
        # stay fixed; CUDA kernels select the bank from the device Decode step.
        self.standby_key: torch.Tensor | None = None
        self.standby_value: torch.Tensor | None = None

    def reset_slot(self, slot: int):
        """Zero out a single slot's KV for a new session."""
        self.unshared_key[slot].zero_()
        self.unshared_value[slot].zero_()
        if self.standby_key is not None:
            assert self.standby_value is not None
            self.standby_key[slot].zero_()
            self.standby_value[slot].zero_()
