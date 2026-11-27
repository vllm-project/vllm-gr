# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""KV pool reorder dispatcher for incremental beam decode.

``BeamSearchSession.step_complete`` calls ``reorder_unshared_kv`` after each
beam decision.  The mode follows ``vllm_gr_npu_decode_execution``:

- ``cascade`` (default, any device): the Triton fused fast path
  ``ops.cuda.beam_kv_reorder.select_unshared_kv_fused`` (one gather + one scatter launch
  across all layers, CUDA only), falling back to the per-layer reference
  ``select_unshared_kv`` indexed by ``slot`` with 0-based
  ``decode_step`` semantics when the fused path declines (non-CUDA,
  non-Triton, or unexpected pool/aux layouts).
- ``xllm`` (NPU only): the fused ``beam_ops.select_unshared_kv`` operator,
  which consumes a ``block_table`` plus ``group_token_num`` prefix sums and a
  1-based step, and reorders every layer's pool in one call.

The fused kernel's mapping is derived from ``group_token_num``
(``bincount(parent).cumsum``), which can only express survivors *grouped by
source beam in ascending order* at *full beam width* — the ordering
``beam_search_group`` emits.  When the caller's ``parent`` is unsorted or
shrunk (e.g. catalog pruning drops beams), the dispatcher falls back to the
cascade kernel, which handles arbitrary ``beam_index`` orderings.
"""

from __future__ import annotations

import torch

_FALLBACK_LOGGED = False


def _resolve_mode() -> str:
    """Read the process-global decode execution mode (cascade | xllm)."""
    from vllm_gr.v1.attention.backends.beam_attn import _get_npu_decode_execution

    return _get_npu_decode_execution()


def _group_token_num_from_parents(parent: torch.Tensor) -> torch.Tensor:
    """Request-local prefix sums of surviving-child counts per source beam.

    ``group_token_num[j] = #{{ i : parent[i] <= j }}`` within the request,
    i.e. ``bincount(parent, minlength=beam_size).cumsum`` flattened — the
    exact representation the xllm ``select_unshared_kv`` kernel parses back
    into per-target source beams (``_parent_beams_from_prefix`` is the
    inverse; it round-trips exactly for sorted parents).
    """
    width = int(parent.max().item()) + 1 if parent.numel() else 1
    width = max(width, parent.numel())
    counts = torch.bincount(parent.to(torch.int64), minlength=width)
    return counts.cumsum(0).to(torch.int32).reshape(-1)


def _fused_reorder_applies(parent: torch.Tensor, pool_beam_width: int) -> bool:
    """Whether the fused kernel can express this selection.

    The kernel derives its mapping purely from ``group_token_num`` prefix
    sums, so it requires (a) one survivor per pool beam row (full width) and
    (b) survivors grouped by source beam ascending.  Both hold for
    ``beam_search_group`` output; anything else must use the cascade kernel.
    """
    if parent.numel() != pool_beam_width:
        return False
    parent_cpu = parent.detach().to("cpu")
    if parent_cpu.numel() < 2:
        return True
    return bool((parent_cpu[1:] >= parent_cpu[:-1]).all().item())


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


def reorder_unshared_kv(
    parent: torch.Tensor,
    slot: int,
    pools: list,
    decode_step: int,
    aux_key_bufs: list[torch.Tensor] | None = None,
    aux_value_bufs: list[torch.Tensor] | None = None,
    aux_key: torch.Tensor | None = None,
    aux_value: torch.Tensor | None = None,
    pool_key_ptrs: torch.Tensor | None = None,
    pool_value_ptrs: torch.Tensor | None = None,
    *,
    layout_prevalidated: bool = False,
) -> None:
    """Reorder every layer's unshared pool KV in place after selection.

    Args:
        parent: ``[W_cur]`` int tensor; ``parent[i] = j`` means new beam i
            inherits old beam j's KV history.
        slot: pool slot (batch row) of the owning session.
        pools: per-layer ``BeamAttentionPool`` objects.
        decode_step: 0-based step; the reorder covers ``[0, decode_step]``.
        aux_key_bufs / aux_value_bufs: optional pre-allocated per-layer scratch
            buffers forwarded to the cascade kernel to avoid per-step
            allocations (CUDA Graph compatibility).  The fused xllm path
            reorders in place and ignores them.
        aux_key / aux_value: optional per-layer scratch buffers shaped
            ``[num_layers, W, H, max_steps, D]`` consumed by the fused Triton
            cascade fast path (one gather + one scatter launch across all
            layers).  When present and the fused path applies, this is used in
            preference to the per-layer ``aux_key_bufs``; otherwise it falls
            back to the per-layer cascade kernel.
    """
    pools = list(pools)
    if _resolve_mode() == "xllm":
        pool_beam_width = pools[0].unshared_key.shape[1]
        if _fused_reorder_applies(parent, pool_beam_width):
            from vllm_gr.ops.npu import beam_ops

            beam_ops.select_unshared_kv(
                parent.to(torch.int32).contiguous(),
                [p.unshared_key for p in pools],
                [p.unshared_value for p in pools],
                torch.tensor([slot], dtype=torch.int32, device=parent.device),
                _group_token_num_from_parents(parent).to(parent.device),
                decode_step + 1,
                pool_beam_width,
                len(pools),
            )
            return

        global _FALLBACK_LOGGED
        if not _FALLBACK_LOGGED:
            from vllm.logger import init_logger

            init_logger(__name__).debug(
                "xllm decode execution fell back to the cascade KV reorder: "
                "the fused select_unshared_kv requires full-width parents "
                "sorted by source beam (got width=%d, pool width=%d)",
                parent.numel(),
                pool_beam_width,
            )
            _FALLBACK_LOGGED = True

    # Cascade mode (GPU / non-xllm).  Try the Triton fused fast path first:
    # one gather + one scatter launch covering every layer at once.  It
    # declines (returns False) on non-CUDA / non-Triton / unexpected pool or
    # aux layouts, falling through to the per-layer reference below.
    key_caches = [p.unshared_key for p in pools]
    value_caches = [p.unshared_value for p in pools]
    if aux_key is not None and aux_value is not None:
        # The fused kernel is Triton; other devices use the reference below.
        if parent.is_cuda:
            from vllm_gr.ops.cuda.beam_kv_reorder import select_unshared_kv_fused

            if select_unshared_kv_fused(
                parent,
                slot,
                key_caches,
                value_caches,
                decode_step,
                aux_key=aux_key,
                aux_value=aux_value,
                pool_key_ptrs=pool_key_ptrs,
                pool_value_ptrs=pool_value_ptrs,
                validate_layout=not layout_prevalidated,
            ):
                return

        # Build per-layer aux slices from the [num_layers, ...] buffer so the
        # reference path reuses the same allocation instead of .clone().
        num_layers = len(pools)
        aux_key_bufs = [aux_key[layer] for layer in range(num_layers)]
        aux_value_bufs = [aux_value[layer] for layer in range(num_layers)]

    select_unshared_kv(
        parent,
        slot,
        key_caches,
        value_caches,
        decode_step,
        aux_key_bufs=aux_key_bufs,
        aux_value_bufs=aux_value_bufs,
    )


__all__ = ["reorder_unshared_kv", "select_unshared_kv"]
