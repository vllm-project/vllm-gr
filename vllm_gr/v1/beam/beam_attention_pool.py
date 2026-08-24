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

        # issue #39 §4.2 decode-buffer reservation bookkeeping.  The pool is sized for
        # ``max_batch * max_beams * max_decode_steps`` decode rows; ``reserve``
        # marks rows as held so the Scheduler can track decode-buffer capacity
        # before dispatch.  Reservations are host-side counters (the pool tensor
        # is fixed-address and never re-allocated).
        self._reserved_rows: int = 0

    @property
    def capacity_rows(self) -> int:
        """Total decode rows the pool can hold (issue #39 §4.2 B×W×max_steps)."""
        return self.max_batch * self.max_beams * self.max_decode_steps

    @property
    def reserved_rows(self) -> int:
        """Number of decode rows currently reserved by the Scheduler."""
        return self._reserved_rows

    def reserve(self, num_rows: int) -> bool:
        """Reserve ``num_rows`` decode rows, all-or-none.

        Returns ``False`` (without changing state) when the request exceeds
        the remaining capacity.  Matching the issue #39 all-or-none
        reservation invariant, a partial reservation is never returned.
        """
        if num_rows < 0:
            raise ValueError(f"reserve: num_rows must be >= 0, got {num_rows}")
        if self._reserved_rows + num_rows > self.capacity_rows:
            return False
        self._reserved_rows += num_rows
        return True

    def release_reserved(self, num_rows: int) -> None:
        """Release ``num_rows`` previously reserved decode rows back to the pool."""
        if num_rows < 0:
            raise ValueError(f"release_reserved: num_rows must be >= 0, got {num_rows}")
        if num_rows > self._reserved_rows:
            raise ValueError(
                f"release_reserved: {num_rows} > reserved_rows={self._reserved_rows}"
            )
        self._reserved_rows -= num_rows

    def reset_slot(self, slot: int):
        """Zero out a single slot's KV for a new session."""
        self.unshared_key[slot].zero_()
        self.unshared_value[slot].zero_()
