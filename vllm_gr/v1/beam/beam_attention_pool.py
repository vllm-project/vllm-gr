# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.max_batch = max_batch
        self.max_beams = max_beams
        self.num_kv_heads = num_kv_heads
        self.max_decode_steps = max_decode_steps
        self.head_dim = head_dim
        self.device = device

        # 5-D contiguous tensor, fixed device address
        # [max_batch, max_beams, num_kv_heads, max_decode_steps, head_dim]
        pool_shape = (max_batch, max_beams, num_kv_heads, max_decode_steps, head_dim)
        self.unshared_key = torch.zeros(pool_shape, dtype=dtype, device=device)
        self.unshared_value = torch.zeros(pool_shape, dtype=dtype, device=device)

    def reset_slot(self, slot: int):
        """Zero out a single slot's KV for a new session."""
        self.unshared_key[slot].zero_()
        self.unshared_value[slot].zero_()
