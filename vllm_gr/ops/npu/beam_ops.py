# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""NPU dispatch wrappers for the xLLM beam-search decode operators.

Covers all six operators used by the asynchronous Decode pipeline. ``cache_unshared_kv``,
``x_attention`` and ``select_unshared_kv`` form the execution path ported by the
NPU decode migration plan; ``beam_search_group`` and ``beam_search_rec_final_select``
are the worker-side beam-decision operators kept as building blocks for a future
migration (their precision contracts are locked down by the
``tests/kernels/test_*_npu.py`` suite).  ``rec_constrained_topk`` lives in its
own module and is intentionally not re-exported here.

ABI notes (verified on device, 2026-08-22; recorded in the workspace
operator survey ``xllm_ops-decode*.md`` and locked by ``tests/kernels/``):
- ``decode_step`` is heterogeneous across the library: ``cache_unshared_kv``
  and ``x_attention`` require an int32 ``[1]`` device tensor, while ``select_unshared_kv``,
  ``beam_search_group`` and ``beam_search_rec_final_select`` take a host int.
  This wrapper takes a host int everywhere and converts as needed.
- ``x_attention`` ignores an explicit ``unshared_block_tables`` argument: the
  Python wrapper passes ``None``,
  which selects the contiguous-unshared layout with implicit row mapping
  ``batch = shared_block_table.shape[0]``, ``beam = num_tokens / batch``.
- ``beam_search_rec_final_select`` tiling constraints: ``result_width`` must
  be a multiple of 32 and ``candidate_top_k`` a multiple of 8.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm_gr.ops.npu._xllm_loader import custom_ops as _custom_ops
from vllm_gr.ops.npu._xllm_loader import has_op as _has_op

_EXECUTION_OPS = ("cache_unshared_kv", "x_attention", "select_unshared_kv")


def is_available() -> bool:
    """Return whether the execution-path fused operators are all present."""
    return all(_has_op(name) for name in _EXECUTION_OPS)


def decision_ops_available() -> bool:
    """Check the two auxiliary selection operators independently of attention."""
    return _has_op("beam_search_group") and _has_op("beam_search_rec_final_select")


def unavailable_reason() -> str:
    """Describe why the execution-path fused operators cannot be called."""
    if _custom_ops() is None:
        return "custom_ops_lib cannot be imported"
    missing = [name for name in _EXECUTION_OPS if not _has_op(name)]
    if missing:
        return f"custom_ops_lib does not export {', '.join(missing)}"
    return ""


def _step_tensor(decode_step: int | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Build the int32 ``[1]`` step tensor required by cache/attention ops.

    A pre-allocated device tensor is passed through unchanged so callers can
    use a persistent fixed-address buffer (ACL graph capture/replay reads the
    refreshed value at every replay; a fresh per-call allocation would break
    address stability).
    """
    if isinstance(decode_step, torch.Tensor):
        return decode_step
    return torch.tensor([decode_step], dtype=torch.int32, device=device)


def cache_unshared_kv(
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    curr_k: torch.Tensor,
    curr_v: torch.Tensor,
    block_table: torch.Tensor,
    decode_step_1based: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write the current step's per-beam K/V into the unshared beam pool.

    Args:
        pool_k / pool_v: ``[num_blocks, beam_size, kv_heads, max_decode_steps,
            head_dim]`` bf16 whole-pool tensors, updated in place.
        curr_k / curr_v: ``[batch * beam_size, kv_heads, head_dim]``.
        block_table: int32 ``[batch]``, pool slot per request.
        decode_step_1based: 1-based step (writes time slice ``step - 1``);
            accepts a host int or a persistent int32 ``[1]`` device tensor
            (graph capture/replay contract).
    """
    custom_ops = _custom_ops()
    if custom_ops is None or not callable(getattr(custom_ops, "cache_unshared_kv", None)):
        raise RuntimeError(
            "NPU cache_unshared_kv is unavailable "
            f"({unavailable_reason()}); use the Torch implementation instead"
        )
    return custom_ops.cache_unshared_kv(
        pool_k,
        pool_v,
        curr_k,
        curr_v,
        block_table,
        _step_tensor(decode_step_1based, pool_k.device),
    )


def x_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    shared_block_tables: torch.Tensor,
    actual_shared_kvlen: torch.Tensor,
    decode_step_1based: int | torch.Tensor,
    scale_value: float = 0.0,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused shared-paged + unshared-contiguous beam attention.

    Args:
        query: ``[batch * beam_size, num_heads, head_dim]``.
        key_cache / value_cache: paged shared cache
            ``[num_blocks, block_size, kv_heads, head_dim]`` (standard non-NZ layout).
        unshared_key / unshared_value: contiguous unshared pool view
            ``[batch * beam_size, kv_heads, max_decode_steps, head_dim]``.
        shared_block_tables: int32 ``[batch, max_blocks_per_seq]``.
        actual_shared_kvlen: int32 ``[batch]``.
        decode_step_1based: 1-based step (unshared valid length); accepts a
            host int or a persistent int32 ``[1]`` device tensor (graph
            capture/replay contract).
        scale_value: ``<= 0`` lets the kernel use ``1 / sqrt(head_dim)``.
        out: Optional contiguous caller-owned output. The optional native
            binding writes directly; installations without it allocate and copy.

    Returns:
        ``[batch * beam_size, num_heads, head_dim]``.
    """
    custom_ops = _custom_ops()
    if custom_ops is None or not callable(getattr(custom_ops, "x_attention", None)):
        raise RuntimeError(
            f"NPU x_attention is unavailable ({unavailable_reason()}); use the cascade path instead"
        )
    if out is not None:
        if out.shape != query.shape:
            raise ValueError("x_attention out must have the same shape as query")
        if out.dtype != query.dtype or out.device != query.device:
            raise ValueError("x_attention out must match query dtype and device")
        if not out.is_contiguous():
            raise ValueError("x_attention out must be contiguous")
    # Preserve the deployed contiguous-unshared ACLNN argument order.
    args = (
        query,
        key_cache,
        value_cache,
        unshared_key,
        unshared_value,
        shared_block_tables,
        None,
        actual_shared_kvlen,
        _step_tensor(decode_step_1based, query.device),
        scale_value,
    )
    if out is not None:
        output_storage = out.untyped_storage().data_ptr()
        if any(
            isinstance(value, torch.Tensor) and value.untyped_storage().data_ptr() == output_storage
            for value in args
        ):
            raise ValueError("x_attention out must not share storage with inputs")
        from vllm_gr.ops.npu._xattention_out import get_x_attention_out

        out_op = get_x_attention_out(out)
        if out_op is not None:
            # A native failure may follow a partial write. Never retry it.
            out_op(*args, out)
            return out
    result = custom_ops.x_attention(*args)
    if out is not None:
        out.copy_(result)
        return out
    return result


def select_unshared_kv(
    beam_index: torch.Tensor,
    pools_k: Sequence[torch.Tensor],
    pools_v: Sequence[torch.Tensor],
    block_table: torch.Tensor,
    group_token_num: torch.Tensor,
    decode_step_1based: int,
    num_beams: int,
    num_layers: int,
) -> None:
    """Rearrange every layer's unshared beam pool in place after selection.

    Args:
        beam_index: int32 ``[batch * beam_size]``, source beam per target slot
            (sorted ascending within each request).
        pools_k / pools_v: per-layer ``[num_blocks, beam_size, kv_heads,
            max_decode_steps, head_dim]`` tensors, updated in place.
        block_table: int32 ``[batch]``.
        group_token_num: int32 ``[batch * beam_size]``, request-local prefix
            sums of the surviving-child counts per source beam.
        decode_step_1based: 1-based step (reorders the time axis ``[0, step)``).
    """
    custom_ops = _custom_ops()
    if custom_ops is None or not callable(getattr(custom_ops, "select_unshared_kv", None)):
        raise RuntimeError(
            "NPU select_unshared_kv is unavailable "
            f"({unavailable_reason()}); use the Torch implementation instead"
        )
    return custom_ops.select_unshared_kv(
        beam_index,
        list(pools_k),
        list(pools_v),
        block_table,
        group_token_num,
        decode_step_1based,
        num_beams,
        num_layers,
    )


def beam_search_group(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """NPU beam TopK + sequence-history rearrangement (non-final decode steps).

    Args:
        log_probs: fp32 ``[request_num * beam_width, 1]`` cumulative logprobs.
        top_tokens: int32 ``[request_num * beam_width, beam_width]``.
        top_probs: fp32 ``[request_num * beam_width, beam_width]``.
        sequence: int32 ``[request_num, beam_width, current_step + 1]``.
        current_step: 0-based step (0 is a legacy kernel-initialization path
            that only writes the flat sequence head; asynchronous scheduling uses
            ``init_from_prefill``).
        top_k: number of beams to keep.

    Returns:
        ``(out_token_ids, out_token_index, out_log_probs,
        out_beam_count_prefix_sums, out_sequence)``; the index and prefix
        outputs carry global ``request * beam_width`` offsets.
    """
    custom_ops = _custom_ops()
    if custom_ops is None or not callable(getattr(custom_ops, "beam_search_group", None)):
        raise RuntimeError(
            "NPU beam_search_group is unavailable "
            f"({unavailable_reason()}); use the Torch implementation instead"
        )
    return custom_ops.beam_search_group(
        log_probs,
        top_tokens,
        top_probs,
        sequence,
        current_step,
        top_k,
    )


def beam_search_rec_final_select(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Final-step beam selection without KV reordering.

    Args:
        log_probs: fp32 ``[request_num * beam_width, 1]``.
        top_tokens / top_probs: ``[request_num * beam_width, candidate_top_k]``
            with ``candidate_top_k`` a multiple of 8 (tiling constraint).
        sequence: int32 ``[request_num, beam_width, current_step + 1]``.
        current_step: 1-based step of the final decode round.
        result_width: number of output sequences; must be a multiple of 32.

    Returns:
        ``(out_token_ids, out_token_index, out_log_probs, out_sequence)`` with
        shapes ``[request_num * result_width, 1]`` x3 and
        ``[request_num, result_width, current_step + 1]``.
    """
    custom_ops = _custom_ops()
    if custom_ops is None or not callable(
        getattr(custom_ops, "beam_search_rec_final_select", None)
    ):
        raise RuntimeError(
            "NPU beam_search_rec_final_select is unavailable "
            f"({unavailable_reason()}); use the Torch implementation instead"
        )
    if result_width % 32 != 0:
        raise ValueError(
            f"result_width must be a multiple of 32 for the NPU kernel, got {result_width}"
        )
    candidate_top_k = top_tokens.shape[-1] if top_tokens.ndim == 2 else 0
    if candidate_top_k and candidate_top_k % 8 != 0:
        raise ValueError(
            f"candidate_top_k must be a multiple of 8 for the NPU kernel, got {candidate_top_k}"
        )
    return custom_ops.beam_search_rec_final_select(
        log_probs,
        top_tokens,
        top_probs,
        sequence,
        current_step,
        result_width,
    )


__all__ = [
    "beam_search_group",
    "beam_search_rec_final_select",
    "cache_unshared_kv",
    "is_available",
    "select_unshared_kv",
    "unavailable_reason",
    "x_attention",
]
