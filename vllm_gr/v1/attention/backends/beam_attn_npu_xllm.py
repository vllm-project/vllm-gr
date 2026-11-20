# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""xLLM fused-operator execution primitives for the Beam Attention backend.

NPU-only execution module selected by the Ascend provider when
``vllm_gr_npu_decode_execution == "xllm"``.  It replaces the cascade
per-operator path only for the incremental beam decode branch; every other
branch (std requests, prefill) forwards to
``beam_attn_npu`` unchanged, and the CUDA path is never touched.

The incremental branch becomes a single fused ``x_attention`` call that
attends over the paged shared prefix and the contiguous unshared beam pool
in one kernel, replacing "two npu_fused_infer_attention_score_v2 calls +
pool->buffer copies + the merge operator chain".
"""

from __future__ import annotations

import math
import os
from typing import cast

import torch
import torch_npu
from vllm.logger import init_logger

from vllm_gr.ops.npu import beam_ops
from vllm_gr.v1.attention.backends import beam_attn_npu
from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
    require_beam_decode_session,
)
from vllm_gr.v1.attention.backends.merge_tnd_attention_outputs import (
    interleaved_pool_base,
)
from vllm_gr.v1.attention.backends.npu_cache_layout import restore_npu_cache_pool_layout

logger = init_logger(__name__)

# Fail fast when the mode selects this module but the operator library is
# missing: without this check the failure would surface at the first beam
# decode step, deep inside model execution.
if not beam_ops.is_available():
    raise RuntimeError(
        "vllm_gr_npu_decode_execution='xllm' requires the xllm-ops library "
        f"({beam_ops.unavailable_reason()}); install it with "
        "tools/install_xllm_ops.sh or select 'cascade' explicitly via "
        "gr.npu_decode_execution=cascade"
    )

# Forwarded primitives: std requests, prefill and cache writes behave
# exactly like the cascade module.
is_available = beam_attn_npu.is_available
get_flash_attn_version = beam_attn_npu.get_flash_attn_version
supports_sinks = beam_attn_npu.supports_sinks
reshape_and_cache = beam_attn_npu.reshape_and_cache
# xllm owns decode. Prefill uses the native Ascend cascade primitives.
prefill_attention = beam_attn_npu.prefill_attention
prefill_attention_bsnd = beam_attn_npu.prefill_attention_bsnd
_GRAPH_CAPTURE_STATE = beam_attn_npu._GRAPH_CAPTURE_STATE


def select_beam_rows(tensor: torch.Tensor, attn_metadata: BeamAttentionMetadata) -> torch.Tensor:
    """Reuse dense NPU rows only when their routing is proven to be identity.

    Fancy indexing also materializes contiguous storage. Keep it for strided,
    blocked-format, padded or unproven inputs, even if the row counts match.
    """
    rows = attn_metadata.num_prefix_indices
    if (
        tensor.device.type == "npu"
        and getattr(attn_metadata, "npu_prefix_rows_are_identity", False)
        and attn_metadata.is_beam_decode
        and attn_metadata.std_indices is None
        and rows > 0
        and rows == attn_metadata.num_actual_tokens
        and tensor.shape[0] == rows
        and tensor.is_contiguous()
        and tensor.storage_offset() == 0
        and torch_npu.get_npu_format(tensor) == 2  # standard unblocked format
    ):
        return tensor[:rows]
    return tensor[attn_metadata.prefix_indices]


def cache_unshared_kv(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    slot: int,
    decode_step: int,
    beam_session: object = None,
) -> None:
    """Fused pool write: slot -> block_table, 0-based step -> 1-based.

    When ``beam_session`` carries the context's fixed-address device buffers
    (``xllm_block_table_buf`` / ``decode_step_1based_buf``), those are passed
    to the kernel directly instead of allocating fresh tensors — required for
    ACL graph capture/replay (address stability + fresh values at replay).
    """
    context = getattr(beam_session, "context", None) if beam_session is not None else None
    block_table = getattr(context, "xllm_block_table_buf", None)
    step = getattr(context, "decode_step_1based_buf", None)
    if block_table is None or step is None:
        # Eager fallback: per-call tensors (previous behavior).
        sessions = getattr(beam_session, "sessions", None)
        slots = [session.slot for session in sessions] if sessions else [slot]
        block_table = torch.tensor(slots, dtype=torch.int32, device=unshared_key.device)
        step = decode_step + 1
    request_width = int(
        getattr(beam_session, "request_width", getattr(beam_session, "beam_width", 0))
    )
    if request_width < 1 or curr_key.shape[0] % request_width:
        raise RuntimeError("xllm Beam rows are not divisible by the request Beam width")
    active_batch = curr_key.shape[0] // request_width
    block_table = block_table[:active_batch]
    unshared_key = restore_npu_cache_pool_layout(unshared_key, request_width)
    unshared_value = restore_npu_cache_pool_layout(unshared_value, request_width)

    beam_ops.cache_unshared_kv(
        unshared_key,
        unshared_value,
        curr_key,
        curr_value,
        block_table,
        step,
    )


def _npu_stream_capturing() -> bool:
    """Whether an ACL graph capture is running on the current stream.

    Mirrors the discriminator vllm-ascend's own backends use to route between
    capture-safe and eager execution paths (``torch.npu.NPUGraph`` capture is
    stream-capture based).
    """
    try:
        return bool(torch_npu.npu.is_current_stream_capturing())
    except Exception:
        # Unknown capture state: assume capture so the captured-shape path is
        # taken.
        return True


def _can_use_direct_shared_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query: torch.Tensor,
    unshared_key: torch.Tensor,
) -> bool:
    """Recognize the x_attention NHD ABI without reading device values.

    Separate native allocations can have an aligned storage offset: their
    data_ptr still starts a valid dense span. Interleaved, blocked-format or
    head-major views must continue through the existing materialization path.
    """
    if os.environ.get("VLLM_KV_CACHE_LAYOUT") not in (None, "", "NHD", "LBNHC"):
        return False
    if key_cache.device.type != "npu" or key_cache.ndim != 4:
        return False
    if (
        value_cache.shape != key_cache.shape
        or value_cache.device != key_cache.device
        or value_cache.dtype != key_cache.dtype
        or key_cache.dtype not in (torch.float16, torch.bfloat16)
        or not key_cache.is_contiguous()
        or not value_cache.is_contiguous()
        or any(size <= 0 for size in key_cache.shape)
    ):
        return False
    _, block_size, heads, dim = key_cache.shape
    if (
        query.ndim != 3
        or unshared_key.ndim != 4
        or query.device != key_cache.device
        or unshared_key.device != key_cache.device
        or query.dtype != key_cache.dtype
        or unshared_key.dtype != key_cache.dtype
        or query.shape[-1] != dim
        or query.shape[-2] <= 0
        or query.shape[-2] % heads
        or unshared_key.shape[1] != heads
        or unshared_key.shape[-1] != dim
    ):
        return False
    # Be explicit about the physical axis order, including degenerate sizes.
    expected_stride = (block_size * heads * dim, heads * dim, dim, 1)
    if key_cache.stride() != expected_stride or value_cache.stride() != expected_stride:
        return False
    return cast(
        bool,
        torch_npu.get_npu_format(key_cache) == 2 and torch_npu.get_npu_format(value_cache) == 2,
    )


def _gather_shared_blocks(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    max_valid_cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather only the shared-prefix blocks the block table references.

    x_attention reads its shared key/value buffers through a block table,
    so only the referenced blocks need to be contiguous. On the 0.30
    K/V-interleaved layout the per-layer cache views are strided halves of
    one pool, and on such non-contiguous sources ``index_select``
    materializes the entire pool before selecting (same cost as
    ``.contiguous()``), defeating the point of the gather. The gather
    therefore runs against the contiguous pool base reconstructed by
    ``interleaved_pool_base``; caches that are already contiguous take the
    direct per-tensor ``index_select`` fallback. Either way the copy is
    proportional to the blocks the table references rather than to the
    pool.

    ``max_valid_cols`` is ``attn_metadata.max_prefix_blocks_count``: the
    number of block-table columns actually needed by the longest prefix in
    this batch.  The fixed-width buffer is allocated at ``max_model_len //
    block_size`` columns, but the tail columns are padding.  Slicing them
    off before flattening reduces the gather size from O(max_model_len /
    block_size) to O(actual_prefix_len / block_size).
    A zero bound retains the full table width for graph capture or callers
    without a tighter prefix bound.

    The remapped table ``row * width + col`` matches the flatten order of
    the gather, so the kernel sees exactly the blocks the original table
    named.
    """
    width = block_table.shape[1]
    active = block_table[:, :max_valid_cols] if 0 < max_valid_cols < width else block_table
    flat = active.reshape(-1)
    base = interleaved_pool_base(key_cache, value_cache)
    if base is not None:
        # Gather K blocks followed by V blocks into one contiguous allocation.
        # Selecting [block, K/V] first would leave strided halves and require
        # copying both gathered tensors again before x_attention can read them.
        block_indices = flat
        if flat.dtype == torch.int32 and base.shape[0] > (torch.iinfo(torch.int32).max + 1) // 2:
            block_indices = flat.to(torch.int64)
        key_indices = block_indices * 2
        packed_indices = torch.cat((key_indices, key_indices + 1))
        packed = base.view(-1, *base.shape[2:]).index_select(0, packed_indices)
        gathered_key = packed[: flat.numel()]
        gathered_value = packed[flat.numel() :]
    else:
        # Contiguous or non-interleaved caches: direct per-tensor gather.
        gathered_key = key_cache.index_select(0, flat)
        gathered_value = value_cache.index_select(0, flat)
    remapped_table = torch.arange(
        flat.numel(), dtype=block_table.dtype, device=block_table.device
    ).reshape(active.shape)
    return gathered_key, gathered_value, remapped_table


def _direct_attention_output(
    output: torch.Tensor, query: torch.Tensor, attn_metadata: BeamAttentionMetadata
) -> torch.Tensor | None:
    """Only pure Beam rows have a caller-owned contiguous leading view."""
    if attn_metadata.std_indices is not None:
        return None
    from vllm_gr.ops.npu._xattention_out import x_attention_out_available

    # Most deployments initially use the original allocating binding. Its
    # cached absence must not add per-layer output views or format/alias checks.
    if not x_attention_out_available():
        return None
    if not output.is_contiguous() or query.dtype not in (torch.float16, torch.bfloat16):
        return None
    view = output[: attn_metadata.num_prefix_indices]
    if view.shape != query.shape or view.dtype != query.dtype or view.device != query.device:
        return None
    return view


def _xllm_incremental_decode(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata,
    beam_session,
    beam_pool,
    scale: float,
) -> None:
    """One fused x_attention call for the incremental beam decode branch.

    Graph-friendly (ACL graph capture/replay): every runtime-varying kernel
    input is either a fixed-address buffer refreshed outside the graph
    (prefix_indices / prefix_seqlens_kv_device / prefix_block_table from the
    metadata builder's fixed-buffer pool; decode_step_1based_buf from the
    beam context) or a stable pool slice.  Host-int fallbacks are kept for the
    eager path (no fixed buffers configured).
    """
    step = beam_session.decode_step  # 0-based host int
    slot = beam_session.slot
    beam_width = int(getattr(beam_session, "attention_width", beam_session.beam_width))
    head_dim = key_cache.shape[-1]

    if abs(scale - (1.0 / math.sqrt(head_dim))) > 1e-6:
        # The kernel derives its own scale when scale_value <= 0; only the
        # standard 1/sqrt(head_dim) matches the model contract.
        raise RuntimeError(
            f"xllm decode execution only supports softmax_scale = 1/sqrt(head_dim), got {scale}"
        )

    # Beam queries: [W, num_heads, D] (already gathered at prefix_indices).
    beam_query = select_beam_rows(query, attn_metadata)

    # Contiguous unshared pool view: [W, H_kv, max_steps, D] -- exactly the
    # layout x_attention expects for the contiguous unshared branch
    # (unshared_block_tables=None, implicit row mapping).
    unshared_key = beam_pool.unshared_key[slot, :beam_width]
    unshared_value = beam_pool.unshared_value[slot, :beam_width]

    # Shared prefix KV length: fixed-address device buffer when the metadata
    # builder runs with fixed buffers (the NPU default whenever beam_max_width
    # is configured, eager decodes included); CPU-list -> fresh tensor
    # otherwise.
    seqlens_kv = getattr(attn_metadata, "prefix_seqlens_kv_device", None)
    if seqlens_kv is None:
        seqlens_kv = torch.tensor(
            attn_metadata.prefix_seqlens_kv_cpu,
            dtype=torch.int32,
            device=beam_query.device,
        )

    # 1-based step count: fixed-address device buffer when available.
    step_count = getattr(beam_session.context, "decode_step_1based_buf", None)
    if step_count is None:
        step_count = step + 1  # dev is 0-based; kernel takes 1-based count

    # Native separate NHD K/V allocations can be read through the original
    # paged table. Preserve their addresses; this branch captures no gather,
    # remapping or copy and replay consumes the refreshed table/sequence data.
    # When the source layout is not proven, gather its blocks in both
    # eager and ACL-graph capture to avoid copying the whole interleaved pool.
    #
    # Capture uses the fixed table width so the gather shape stays constant
    # across prefix lengths. Its input buffers and captured intermediates must
    # retain stable addresses; replay reads updated block IDs from the same
    # metadata buffer. The graph owns the captured allocation lifetime, so
    # this does not require copying every block in the KV pool.
    # Eager can use ``max_prefix_blocks_count`` to omit padding columns.
    shared_table = attn_metadata.prefix_block_table
    if _can_use_direct_shared_kv(key_cache, value_cache, beam_query, unshared_key):
        shared_key, shared_value = key_cache, value_cache
    elif shared_table is not None:
        max_valid_cols = (
            0 if _npu_stream_capturing() else getattr(attn_metadata, "max_prefix_blocks_count", 0)
        )
        shared_key, shared_value, shared_table = _gather_shared_blocks(
            key_cache,
            value_cache,
            shared_table,
            max_valid_cols,
        )
    else:
        # No paged shared prefix in this batch: the kernel still needs
        # contiguous buffers, and there is no table to gather through.
        shared_key = key_cache.contiguous()
        shared_value = value_cache.contiguous()

    direct_output = _direct_attention_output(output, beam_query, attn_metadata)
    out = beam_ops.x_attention(
        beam_query,
        shared_key,
        shared_value,
        unshared_key,
        unshared_value,
        shared_table,
        seqlens_kv,
        step_count,
        0.0,  # auto 1/sqrt(head_dim), validated above
        out=direct_output,
    )
    if direct_output is not None:
        return
    # Scatter into the layer output: slice copy for a pure beam batch (rows
    # [0, num_prefix_indices) are exactly the beam rows; index_put_ here was
    # a host-blocking op), fancy-index scatter otherwise.
    if attn_metadata.std_indices is None:
        output[: attn_metadata.num_prefix_indices] = out
    else:
        output[attn_metadata.prefix_indices] = out


def _cascade_fallback_reason(
    attn_metadata: BeamAttentionMetadata,
    sliding_window_size: list[int] | tuple[int, int] | None,
    logits_soft_cap: float,
    s_aux: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    beam_pool: object | None,
) -> str | None:
    """Return why fused xLLM decode cannot run, or None when eligible."""

    if not getattr(attn_metadata, "is_beam_decode", False):
        return "not_beam_decode"
    beam_session = getattr(attn_metadata, "beam_session", None)
    if beam_session is None:
        return "missing_beam_session"
    if not beam_session.is_active:
        return "inactive_beam_session"
    if attn_metadata.std_indices is not None:
        return "mixed_std_beam_batch"
    if attn_metadata.prefix_indices is None:
        return "missing_prefix_indices"
    # (-1, -1) is the "no sliding window" sentinel used by models without one.
    if sliding_window_size is not None and any(
        window is not None and window >= 0 for window in sliding_window_size
    ):
        return "sliding_window"
    if logits_soft_cap:
        return "logits_soft_cap"
    if s_aux is not None:
        return "attention_sinks"
    if alibi_slopes is not None:
        return "alibi"
    if beam_pool is None:
        return "missing_beam_pool"
    return None


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata,
    sliding_window_size: list[int] | tuple[int, int] | None,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    beam_pool: object = None,
) -> None:
    """Dispatch incremental beam decode to x_attention; forward the rest.

    Mixed std/beam batches and missing prefix_indices/pool are delegated
    to ``beam_attn_npu.cascade_attention``, which handles those layouts.
    Invalid Beam decode metadata is rejected before either execution path.
    Sliding-window, softcap, ALiBi and sink configurations are unsupported
    by both NPU execution paths and rejected by ``BeamAttentionImpl`` at
    initialization; delegation does not add support for those semantics.
    """

    def _fallback_to_cascade() -> None:
        beam_attn_npu.cascade_attention(
            output,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            sliding_window_size,
            scale,
            logits_soft_cap,
            vllm_flash_attn_version,
            s_aux=s_aux,
            alibi_slopes=alibi_slopes,
            beam_pool=beam_pool,
        )

    if getattr(attn_metadata, "is_beam_decode", False) or attn_metadata.prefix_indices is not None:
        require_beam_decode_session(attn_metadata)

    fallback_reason = _cascade_fallback_reason(
        attn_metadata,
        sliding_window_size,
        logits_soft_cap,
        s_aux,
        alibi_slopes,
        beam_pool,
    )
    if fallback_reason is not None:
        logger.debug("NPU Beam xLLM decode fallback: %s", fallback_reason)
        _fallback_to_cascade()
        return

    beam_session = attn_metadata.beam_session
    assert beam_session is not None
    _xllm_incremental_decode(
        output,
        query,
        key_cache,
        value_cache,
        attn_metadata,
        beam_session,
        beam_pool,
        scale,
    )


__all__ = [
    "cache_unshared_kv",
    "cascade_attention",
    "get_flash_attn_version",
    "is_available",
    "prefill_attention",
    "reshape_and_cache",
    "select_beam_rows",
    "supports_sinks",
]
