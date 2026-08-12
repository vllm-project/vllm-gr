# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV operators for incremental beam decode.

cache_unshared_kv: Write the current step's W beam KV into the pool's
    ``[slot, :, :, decode_step, *]`` slice.

select_unshared_kv: In-place reorder pool KV according to ``parent[W]``,
    covering the range ``[0:decode_step+1]`` for a specific slot.

Both are implemented as pure Python (D1 / E1) for the minimum viable version.
They will be optimized to Triton / CUDA kernels later.
"""

from __future__ import annotations

from typing import List

import torch


def cache_unshared_kv(
    unshared_key: torch.Tensor,  # [max_batch, W, H, max_steps, D] pool key
    unshared_value: torch.Tensor,  # [max_batch, W, H, max_steps, D] pool value
    curr_key: torch.Tensor,  # [W_cur, H, D] current step's key
    curr_value: torch.Tensor,  # [W_cur, H, D] current step's value
    slot: int,  # batch slot index
    decode_step: int,  # 0-indexed, host int
) -> None:
    """Write the current step's W_cur beam KV into the pool's decode_step slice.

    Write address = pool base + slot offset + decode_step offset (fixed).

    Only the first ``W_cur = curr_key.shape[0]`` pool slots are written.  When
    the live beam count shrinks (e.g. catalog pruning drops dead-end beams),
    the trailing pool slots hold stale KV that no scheduled query references,
    because the suffix ``cu_seqlens_k`` only spans the first ``W_cur`` beams.
    When the beam count stays constant (the fixed-width case) ``W_cur`` equals
    the pool width and this is a full-width write.
    """
    n = curr_key.shape[0]
    # Write into the pool's step slice for the given slot
    unshared_key[slot, :n, :, decode_step : decode_step + 1, :].copy_(
        curr_key.unsqueeze(2)  # [W_cur, H, 1, D] → broadcast into [W_cur, H, 1, D]
    )
    unshared_value[slot, :n, :, decode_step : decode_step + 1, :].copy_(curr_value.unsqueeze(2))


def select_unshared_kv(
    beam_index: torch.Tensor,  # [W_cur] int32, source beam index for each beam
    slot: int,  # batch slot index
    unshared_key_caches: List[torch.Tensor],  # per-layer pool key
    unshared_value_caches: List[torch.Tensor],  # per-layer pool value
    decode_step: int,  # 0-indexed, reorder range [0:decode_step+1]
) -> None:
    """In-place reorder pool KV according to beam_index for a specific slot.

    ``beam_index[i] = j`` means: new beam i's KV history should come from
    old beam j.

    Only the first ``n = len(beam_index)`` pool slots are reordered; the
    trailing slots (leftover from a larger beam count on an earlier step) are
    left stale and unreferenced.  ``beam_index`` values index into the *full*
    previous pool, so a shrinking beam count correctly rescues the surviving
    beams' histories into the front slots.  When the beam count stays constant
    (the fixed-width case) ``n`` equals the pool width and this is a full
    reorder.

    For each layer pool, clones the old KV for ``[0:decode_step+1]`` range,
    then reorders by beam_index.
    """
    n = beam_index.shape[0]
    for key_cache, value_cache in zip(unshared_key_caches, unshared_value_caches):
        # key_cache: [max_batch, W, H, max_steps, D]
        # Only reorder [0:decode_step+1] range for the given slot
        old_key = key_cache[slot, :, :, : decode_step + 1, :].clone()  # [W, H, step+1, D]
        old_value = value_cache[slot, :, :, : decode_step + 1, :].clone()  # [W, H, step+1, D]

        # Reorder by beam_index (write into the first n pool slots)
        key_cache[slot, :n, :, : decode_step + 1, :] = old_key[beam_index]
        value_cache[slot, :n, :, : decode_step + 1, :] = old_value[beam_index]
