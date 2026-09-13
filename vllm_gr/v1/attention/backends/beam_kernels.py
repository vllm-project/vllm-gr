# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV operators for incremental beam decode.

cache_unshared_kv: Write the current step's W beam KV into the pool's
    ``[slot, :, :, decode_step, *]`` slice.

select_unshared_kv: In-place reorder pool KV according to ``parent[W]``,
    covering the range ``[0:decode_step+1]`` for a specific slot.

select_unshared_kv_fused: Triton fast path for the reorder (CUDA only;
    returns False everywhere else so callers fall back to the reference).
"""

from __future__ import annotations

import torch
import triton

from vllm_gr.ops.cuda.beam_kv_reorder import (
    _select_kv_gather_kernel,
    _select_kv_scatter_kernel,
)


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


def select_unshared_kv(
    beam_index: torch.Tensor,  # [W_cur] int32, source beam index for each beam
    slot: int,  # batch slot index
    unshared_key_caches: list[torch.Tensor],  # per-layer pool key
    unshared_value_caches: list[torch.Tensor],  # per-layer pool value
    decode_step: int,  # 0-indexed, reorder range [0:decode_step+1]
    aux_key_bufs: list[torch.Tensor] | None = None,  # pre-allocated [W, H, max_steps, D]
    aux_value_bufs: list[torch.Tensor] | None = None,  # pre-allocated [W, H, max_steps, D]
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

    When aux_key_bufs/aux_value_bufs are provided, they are used instead of
    .clone() to avoid per-step new allocations (CUDA Graph compatibility).
    Otherwise, the original .clone() path is used (NPU or non-Graph).

    Note: select_unshared_kv is called outside the CUDA Graph (after beam
    decision), so decode_step remains a host int.
    """
    n = beam_index.shape[0]
    for layer_idx, (key_cache, value_cache) in enumerate(
        zip(unshared_key_caches, unshared_value_caches)
    ):
        # key_cache: [max_batch, W, H, max_steps, D]
        # Only reorder [0:decode_step+1] range for the given slot
        if aux_key_bufs is not None and aux_value_bufs is not None:
            # Graph-compatible path: use pre-allocated aux_buf instead of .clone()
            aux_key = aux_key_bufs[layer_idx]
            aux_value = aux_value_bufs[layer_idx]
            # Copy only the [0:decode_step+1] range into aux_buf
            aux_key[:, :, : decode_step + 1, :].copy_(key_cache[slot, :, :, : decode_step + 1, :])
            aux_value[:, :, : decode_step + 1, :].copy_(
                value_cache[slot, :, :, : decode_step + 1, :]
            )
            # Reorder by beam_index and write back (only first n beams, only valid steps)
            key_cache[slot, :n, :, : decode_step + 1, :] = aux_key[
                beam_index, :, : decode_step + 1, :
            ]
            value_cache[slot, :n, :, : decode_step + 1, :] = aux_value[
                beam_index, :, : decode_step + 1, :
            ]
        else:
            # Original path: .clone() creates new allocation (not Graph-compatible)
            old_key = key_cache[slot, :, :, : decode_step + 1, :].clone()
            old_value = value_cache[slot, :, :, : decode_step + 1, :].clone()
            key_cache[slot, :n, :, : decode_step + 1, :] = old_key[beam_index]
            value_cache[slot, :n, :, : decode_step + 1, :] = old_value[beam_index]


# ── Triton fused fast path (CUDA) ──────────────────────────────────────────
# The reorder runs as TWO launches (gather pool→aux, then scatter aux→pool):
# beam_index is not a permutation (fan-out — several beams may select the
# same parent), so an in-place single kernel would race cross-program reads
# against writes.


def select_unshared_kv_fused(
    beam_index: torch.Tensor,  # [n] source beam index for each beam
    slot: int,
    unshared_key_caches: list[torch.Tensor],
    unshared_value_caches: list[torch.Tensor],
    decode_step: int,
    aux_key: torch.Tensor,  # [L, W, H, max_steps, D] per-layer scratch
    aux_value: torch.Tensor,
    pool_key_ptrs: torch.Tensor | None = None,
    pool_value_ptrs: torch.Tensor | None = None,
    *,
    validate_layout: bool = True,
) -> bool:
    """Triton fast path for select_unshared_kv; returns False when unavailable.

    Same semantics as select_unshared_kv's aux-buf path — reorder the first
    ``n`` beams' ``[0:decode_step+1]`` KV through the scratch buffer — as one
    gather + one scatter launch across all layers (per-layer pool base
    addresses passed as an int64 pointer table; uint16 pointer casts make
    the copy bit-exact for any 2-byte dtype). Requires the per-layer aux
    buffers: every layer stages through its own slice, so the single
    all-layer launch cannot cross-clobber staging data.
    """
    if not unshared_key_caches or len(unshared_value_caches) != len(unshared_key_caches):
        return False
    n = beam_index.shape[0]
    first = unshared_key_caches[0]
    if not first.is_cuda or first.element_size() != 2:
        return False
    if aux_key.dim() != 5 or aux_key.shape[0] != len(unshared_key_caches):
        return False  # unexpected aux layout → Python fallback
    _, _, H, _, D = aux_key.shape
    steps = decode_step + 1
    # All layers must share layout (they do: same-shape contiguous pools) and
    # the head_dim axis must be unit-stride in both pool and aux.
    ref_strides = first.stride()
    if ref_strides[4] != 1 or aux_key.stride(4) != 1:
        return False
    if validate_layout and (
        any(p.stride() != ref_strides for p in unshared_key_caches)
        or any(v.stride() != ref_strides for v in unshared_value_caches)
    ):
        return False

    if not beam_index.is_contiguous():
        return False
    device = first.device
    if pool_key_ptrs is None or pool_value_ptrs is None:
        k_ptrs = torch.tensor(
            [p.data_ptr() for p in unshared_key_caches], dtype=torch.int64, device=device
        )
        v_ptrs = torch.tensor(
            [p.data_ptr() for p in unshared_value_caches], dtype=torch.int64, device=device
        )
    else:
        k_ptrs = pool_key_ptrs
        v_ptrs = pool_value_ptrs
        if validate_layout:
            expected = (len(unshared_key_caches),)
            for ptrs in (k_ptrs, v_ptrs):
                if (
                    ptrs.device != device
                    or ptrs.dtype != torch.int64
                    or tuple(ptrs.shape) != expected
                    or not ptrs.is_contiguous()
                ):
                    return False
    # The batch-slot offset advances the byte addresses directly instead of
    # passing a stride_pb argument into the kernels.
    slot_off = slot * ref_strides[0] * first.element_size()
    grid = (len(unshared_key_caches) * n * H,)
    base_args = (
        k_ptrs,
        v_ptrs,
        aux_key.data_ptr(),
        aux_value.data_ptr(),
        slot_off,
        n,
        steps,
    )
    kernel_kwargs = {
        "H": H,
        "BLOCK_D": triton.next_power_of_2(D),
        "D": D,
        "stride_pw": ref_strides[1],
        "stride_ph": ref_strides[2],
        "stride_ps": ref_strides[3],
        "stride_al": aux_key.stride(0),
        "stride_aw": aux_key.stride(1),
        "stride_ah": aux_key.stride(2),
        "stride_as": aux_key.stride(3),
    }
    _select_kv_gather_kernel[grid](*base_args, beam_index, **kernel_kwargs)
    _select_kv_scatter_kernel[grid](*base_args, **kernel_kwargs)
    return True
