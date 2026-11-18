# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Device tensor construction for Beam Attention metadata."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = [
    "BeamTensorBuffers",
    "BeamTensorContext",
    "allocate_tensor_buffers",
    "cached_seqlens_views",
    "create_beam_tensors",
    "create_std_tensors",
    "get_schedule",
    "pack_int32_to_dev",
    "upload_incremental_seqlens",
]


@dataclass
class BeamTensorBuffers:
    prefix_cu_seqlens_q: torch.Tensor | None = None
    suffix_cu_seqlens_q: torch.Tensor | None = None
    prefix_seqlens_kv: torch.Tensor | None = None
    suffix_seqlens_kv: torch.Tensor | None = None
    suffix_cu_seqlens_k: torch.Tensor | None = None
    prefix_block_table: torch.Tensor | None = None
    std_block_table: torch.Tensor | None = None
    std_indices: torch.Tensor | None = None
    prefix_indices: torch.Tensor | None = None
    std_cu_seqlens_q: torch.Tensor | None = None
    std_seqlens_kv: torch.Tensor | None = None
    h2d_host: torch.Tensor | None = None
    h2d_dev: torch.Tensor | None = None


@dataclass
class BeamTensorContext:
    device: torch.device
    block_size: int
    fixed_buffers: bool
    is_cuda: bool
    is_npu: bool
    cache: dict
    buffers: BeamTensorBuffers
    num_heads_q: int = 0
    num_kv_heads: int = 0
    headdim: int = 0
    kv_cache_dtype: torch.dtype = torch.float16
    aot_schedule: bool = False
    aot_sliding_window: tuple[int, int] | None = None
    cache_dtype: str = "auto"


def _get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


def cached_seqlens_views(
    ctx: BeamTensorContext,
    geom_key,
    *,
    n_pk: int,
    n_pc: int,
    n_sk: int,
    n_scq: int,
    n_sck: int,
):
    """Return the five fixed seqlens/cu_seqlens device views for a geometry.

    When ``geom_key`` has already produced these views, they are returned from
    a cache instead of re-slicing the address-stable buffers every step.
    Because a cache hit means the beam geometry (widths / prefix_len /
    cache_len) is unchanged, the slice extents ``n_*`` are identical on every
    step, so the reused view objects stay valid (their storage is the
    fixed-address device buffer; per-step values are updated in place by the
    caller).

    Returns ``(prefix_seqlens_kv, prefix_cu_seqlens_q, suffix_seqlens_kv,
    suffix_cu_seqlens_q, suffix_cu_seqlens_k)`` or None when ``geom_key`` is
    None (fall back to full per-step slicing).
    """
    if geom_key is None:
        return None
    views_cache = ctx.cache.setdefault("views", {})
    cached = views_cache.get(geom_key)
    if cached is not None:
        return cached
    buffers = ctx.buffers
    assert buffers.prefix_seqlens_kv is not None
    assert buffers.prefix_cu_seqlens_q is not None
    assert buffers.suffix_seqlens_kv is not None
    assert buffers.suffix_cu_seqlens_q is not None
    assert buffers.suffix_cu_seqlens_k is not None
    views = (
        buffers.prefix_seqlens_kv[:n_pk],
        buffers.prefix_cu_seqlens_q[:n_pc],
        buffers.suffix_seqlens_kv[:n_sk],
        buffers.suffix_cu_seqlens_q[:n_scq],
        buffers.suffix_cu_seqlens_k[:n_sck],
    )
    views_cache[geom_key] = views
    return views


def upload_incremental_seqlens(
    ctx: BeamTensorContext,
    geom_key,
    decode_step: int,
    *,
    prefix_seqlens_kv_view,
    prefix_cu_seqlens_q_view,
    suffix_seqlens_kv_view,
    suffix_cu_seqlens_q_view,
    suffix_cu_seqlens_k_view,
    prefix_seqlens_kv_cpu,
    prefix_cu_seqlens_q_cpu,
    suffix_cu_seqlens_q_cpu,
) -> bool:
    """Upload fixed beam seqlens once per geometry, refresh step-derived ones.

    Incremental decode: only ``suffix_seqlens_kv`` and ``suffix_cu_seqlens_k``
    depend on ``decode_step``. The three fixed payloads are written to the
    address-stable buffers exactly once per ``geom_key``; later steps skip
    them. The step-derived suffix arrays are rebuilt on device with ``fill_``
    / ``arange`` from ``decode_step`` — no host list, no H2D.

    Returns True when the caller should skip the generic ``pack`` path
    (geom_key admitted the cache), False to fall back to full construction.
    """
    if geom_key is None or decode_step is None:
        return False
    uploaded = ctx.cache.setdefault("uploaded_geometries", set())
    if geom_key not in uploaded:
        pack_int32_to_dev(
            ctx,
            [
                (prefix_seqlens_kv_view, list(prefix_seqlens_kv_cpu)),
                (prefix_cu_seqlens_q_view, [0] + list(prefix_cu_seqlens_q_cpu)),
                (suffix_cu_seqlens_q_view, [0] + list(suffix_cu_seqlens_q_cpu)),
            ],
            non_blocking=ctx.is_npu,
        )
        uploaded.add(geom_key)

    # Device-side step update (no H2D):
    #   suffix_seqlens_kv   = [s+1]*w
    #   suffix_cu_seqlens_k = [0, s+1, 2(s+1), ..., w*(s+1)]
    # A single strided arange emits the cu_seqlens_k array directly (its i-th
    # element is i*s), avoiding the separate arange+mul_ pair — just two small
    # launches per step instead of three.
    s = decode_step + 1
    num_suffix = suffix_seqlens_kv_view.numel()
    suffix_seqlens_kv_view.fill_(s)
    if suffix_cu_seqlens_k_view is not None:
        torch.arange(
            0,
            (num_suffix + 1) * s,
            step=s,
            dtype=torch.int32,
            device=ctx.device,
            out=suffix_cu_seqlens_k_view,
        )
    return True


def get_schedule(
    ctx: BeamTensorContext,
    *,
    batch_size: int,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    seqlens: torch.Tensor,
    max_seq_len: int,
    causal: bool,
    max_num_splits: int,
    scheduler: Callable[..., dict | None],
) -> dict | None:
    if not ctx.aot_schedule:
        return None

    if ctx.cache_dtype.startswith("fp8"):
        qkv_dtype = _get_fp8_dtype_for_flashattn(ctx.cache_dtype)
    else:
        qkv_dtype = ctx.kv_cache_dtype

    return scheduler(
        batch_size=batch_size,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_seq_len,
        num_heads_q=ctx.num_heads_q,
        num_heads_kv=ctx.num_kv_heads,
        headdim=ctx.headdim,
        cache_seqlens=seqlens,
        qkv_dtype=qkv_dtype,
        cu_seqlens_q=cu_query_lens,
        page_size=ctx.block_size,
        causal=causal,
        window_size=ctx.aot_sliding_window,
        num_splits=max_num_splits,
    )


def pack_int32_to_dev(
    ctx: BeamTensorContext,
    entries: list[tuple[Any, list[int]]],
    *,
    non_blocking: bool = False,
) -> None:
    """Write several small host int lists to device in ONE H2D.

    ``entries`` is a list of ``(dst_slice, host_list)``. All the step's
    seqlens / cu_seqlens / index metadata is staged into a single pinned host
    buffer, moved across the PCIe bus once, then scattered into the
    preallocated (address-stable) device buffers with device-to-device copies.
    This replaces N independent ``async torch.tensor(...).copy_`` HtoD calls,
    whose per-copy launch latency dominated the host time of
    ``_build_attention_metadata`` for beam decode.
    """
    total = sum(len(host_list) for _dst, host_list in entries)
    if total == 0:
        return
    h2d_host = ctx.buffers.h2d_host
    h2d_dev = ctx.buffers.h2d_dev
    assert h2d_host is not None and h2d_dev is not None
    assert total <= h2d_host.numel(), (
        f"beam metadata host staging overflow: {total} > {h2d_host.numel()}"
    )

    # Build the host payload with one numpy concatenate instead of a per-entry
    # ``torch.as_tensor(list)`` + ``copy_`` pair. The Python list -> tensor
    # conversion per entry is the dominant host cost at this size (the arrays
    # total only a few KB), so a single bulk write into the pinned staging
    # keeps the PCIe transfer as the only real work.
    np_dtype = np.int32 if h2d_host.dtype == torch.int32 else np.int64
    segments = [np.asarray(hl, dtype=np_dtype) for _dst, hl in entries]
    packed = np.concatenate(segments) if len(segments) > 1 else segments[0]
    if non_blocking:
        # A first NPU Beam geometry must not drain the preceding Prefill and
        # TopK before the rest of Decode preparation can be submitted. Do not
        # reuse h2d_host here: a following request could overwrite it before
        # this asynchronous H2D has consumed the payload. The pinned allocator
        # tracks the copy and delays recycling this independent allocation
        # until its transfer has completed.
        host = torch.empty(total, dtype=h2d_host.dtype, pin_memory=True)
    else:
        host = h2d_host[:total]
    host.copy_(torch.from_numpy(packed))
    # Single host -> device transfer for the whole payload.
    h2d_dev[:total].copy_(host, non_blocking=non_blocking)

    off = 0
    for dst, host_list in entries:
        n = len(host_list)
        if n:
            dst.copy_(h2d_dev[off : off + n])
            off += n


def allocate_tensor_buffers(
    *,
    device: torch.device,
    block_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    max_beams: int,
) -> BeamTensorBuffers:
    """Allocate the fixed-address metadata buffers used by graph replay."""
    max_segs = max_num_seqs * max_beams + 1
    max_blocks = max(1, (max_model_len + block_size - 1) // block_size)
    max_token_indices = max(max_num_seqs * max_beams, max_num_batched_tokens)
    return BeamTensorBuffers(
        prefix_cu_seqlens_q=torch.zeros(max_segs, dtype=torch.int32, device=device),
        suffix_cu_seqlens_q=torch.zeros(max_segs, dtype=torch.int32, device=device),
        prefix_seqlens_kv=torch.zeros(max_num_seqs, dtype=torch.int32, device=device),
        suffix_seqlens_kv=torch.zeros(max_num_seqs * max_beams, dtype=torch.int32, device=device),
        suffix_cu_seqlens_k=torch.zeros(max_segs, dtype=torch.int32, device=device),
        prefix_block_table=torch.zeros(
            (max_num_seqs, max_blocks), dtype=torch.int32, device=device
        ),
        std_block_table=torch.zeros((max_num_seqs, max_blocks), dtype=torch.int32, device=device),
        std_indices=torch.zeros(max_token_indices, dtype=torch.long, device=device),
        prefix_indices=torch.zeros(max_token_indices, dtype=torch.long, device=device),
        std_cu_seqlens_q=torch.zeros(max_num_seqs + 1, dtype=torch.int32, device=device),
        std_seqlens_kv=torch.zeros(max_num_seqs, dtype=torch.int32, device=device),
        h2d_host=torch.zeros(4 * max_segs, dtype=torch.int32, pin_memory=True),
        h2d_dev=torch.zeros(4 * max_segs, dtype=torch.int32, device=device),
    )


def create_std_tensors(
    ctx: BeamTensorContext,
    std_indices,
    std_seqlens_q_cpu,
    std_cu_seqlens_q_cpu,
    std_seqlens_kv_cpu,
    std_req_idxs,
    common_attn_metadata,
    device,
):
    if not std_seqlens_q_cpu:
        return None, None, None, 0, 0, None

    num_std = len(std_indices)
    std_max_q_len = max(std_seqlens_q_cpu)
    std_max_seq_len = max(std_seqlens_kv_cpu)
    num_std_blocks = (std_max_seq_len + ctx.block_size - 1) // ctx.block_size

    if ctx.fixed_buffers:
        assert ctx.buffers.std_indices is not None
        assert ctx.buffers.std_block_table is not None
        assert ctx.buffers.std_cu_seqlens_q is not None
        assert ctx.buffers.std_seqlens_kv is not None
        std_indices_tensor = ctx.buffers.std_indices[:num_std]
        std_indices_tensor.copy_(torch.as_tensor(std_indices, dtype=torch.long, device="cpu"))
        std_block_table = ctx.buffers.std_block_table[: len(std_req_idxs), :num_std_blocks]
        std_block_table.copy_(
            common_attn_metadata.block_table_tensor[std_req_idxs, :num_std_blocks]
        )
        num_cu = len(std_cu_seqlens_q_cpu) + 1  # +1 for leading 0
        std_cu_seqlens_q_device = ctx.buffers.std_cu_seqlens_q[:num_cu]
        std_seqlens_kv_device = ctx.buffers.std_seqlens_kv[: len(std_seqlens_kv_cpu)]
        pack_int32_to_dev(
            ctx,
            [
                (std_cu_seqlens_q_device, [0] + list(std_cu_seqlens_q_cpu)),
                (std_seqlens_kv_device, list(std_seqlens_kv_cpu)),
            ],
        )
    else:
        std_indices_tensor = torch.tensor(std_indices, dtype=torch.long, device=device)
        std_block_table = common_attn_metadata.block_table_tensor[
            std_req_idxs, :num_std_blocks
        ].contiguous()
        if ctx.is_cuda:
            # Tests / kernel harnesses that skip __init__ buffer alloc.
            std_cu_seqlens_q_device = torch.tensor(
                [0] + std_cu_seqlens_q_cpu, dtype=torch.int32, device=device
            )
            std_seqlens_kv_device = torch.tensor(
                std_seqlens_kv_cpu, dtype=torch.int32, device=device
            )
        else:
            # NPU: npu_fused_infer_attention_score_v2 currently reads CPU lists
            # (std_cu_seqlens_q_cpu / std_seqlens_kv_cpu).
            # TODO(ACL-graph): switch NPU to device tensors so this branch
            # disappears and seqlens handling is unified.
            std_cu_seqlens_q_device = None
            std_seqlens_kv_device = None

    return (
        std_indices_tensor,
        std_cu_seqlens_q_device,
        std_seqlens_kv_device,
        std_max_q_len,
        std_max_seq_len,
        std_block_table,
    )


def create_beam_tensors(
    ctx: BeamTensorContext,
    common_attn_metadata,
    prefix_indices,
    prefix_seqlens_q_cpu,
    prefix_cu_seqlens_q_cpu,
    prefix_seqlens_kv_cpu,
    suffix_seqlens_q_cpu,
    suffix_cu_seqlens_q_cpu,
    suffix_cu_seqlens_kv_cpu,
    suffix_seqlens_kv_cpu,
    prefix_req_idxs_cpu,
    max_prefix_blocks_count,
    device,
    max_num_splits: int,
    *,
    scheduler: Callable[..., dict | None] | None = None,
    geom_key=None,
    decode_step: int | None = None,
    prefix_indices_contiguous: bool | None = None,
    prefix_max_q_len: int | None = None,
    suffix_max_q_len: int | None = None,
    suffix_max_seq_len: int | None = None,
):
    if not prefix_seqlens_q_cpu:
        # Full structured initializations for profiling warm-up model passes
        prefix_indices_tensor = prefix_seqlens_kv_device = prefix_cu_seqlens_q_device = (
            prefix_block_table
        ) = None
        prefix_max_q_len = prefix_max_seq_len = 0
        suffix_cu_seqlens_q_device = suffix_cu_seqlens_k_device = suffix_seqlens_kv_device = None
        suffix_max_q_len = suffix_max_seq_len = 0
        prefix_scheduler_metadata = None
        return (
            prefix_indices_tensor,
            prefix_seqlens_kv_device,
            prefix_max_seq_len,
            prefix_cu_seqlens_q_device,
            prefix_max_q_len,
            prefix_block_table,
            suffix_seqlens_kv_device,
            suffix_max_seq_len,
            suffix_cu_seqlens_q_device,
            suffix_cu_seqlens_k_device,
            suffix_max_q_len,
            prefix_scheduler_metadata,
        )

    num_prefix = len(prefix_indices)
    # Scalar reductions may arrive precomputed from _build_beam_mappings
    # (the incremental path knows them analytically); fall back to the
    # list max for callers that only pass the raw lists.
    if prefix_max_q_len is None:
        prefix_max_q_len = max(prefix_seqlens_q_cpu) if prefix_seqlens_q_cpu else 0
    if suffix_max_q_len is None:
        suffix_max_q_len = max(suffix_seqlens_q_cpu) if suffix_seqlens_q_cpu else 0
    if suffix_max_seq_len is None:
        suffix_max_seq_len = max(suffix_seqlens_kv_cpu) if suffix_seqlens_kv_cpu else 0

    if ctx.fixed_buffers:
        prefix_indices_tensor = ctx.buffers.prefix_indices[:num_prefix]
        # Incremental decode produces a contiguous prefix index range
        # (q_start .. q_start+w).  Emit it on device with a strided arange
        # instead of converting the Python list every step (no host list,
        # no H2D).  ``prefix_indices_contiguous`` is the EXPLICIT verdict
        # from _build_beam_mappings (True only on the incremental path,
        # where each request contributes one ``range(q_start, q_start+w)``);
        # callers that do not supply it fall back to the per-element scan,
        # and anything non-contiguous (multi-request recompute layout)
        # falls back to the host list transfer.
        if num_prefix and (
            prefix_indices_contiguous
            if prefix_indices_contiguous is not None
            else all(prefix_indices[i] == prefix_indices[0] + i for i in range(num_prefix))
        ):
            torch.arange(
                prefix_indices[0],
                prefix_indices[0] + num_prefix,
                dtype=torch.long,
                device=device,
                out=prefix_indices_tensor,
            )
        else:
            prefix_indices_tensor.copy_(
                torch.as_tensor(prefix_indices, dtype=torch.long, device="cpu")
            )
        # G1 constant prefix geometry: use a FIXED max_seqlen_k upper
        # bound (the full block-table capacity) and the FULL-WIDTH
        # block-table view, so the FA2 launch parameters baked into a
        # captured CUDA graph are identical for every prompt length.
        # FA2 reads the true KV extent from the device-side seqused_k,
        # so the over-set bound and the never-read tail columns are
        # harmless (verified: a graph captured at prefix_len=1024
        # replays bit-identically for prefix_len=5120).
        prefix_max_seq_len = ctx.buffers.prefix_block_table.shape[1] * ctx.block_size
        prefix_block_table = ctx.buffers.prefix_block_table[: len(prefix_req_idxs_cpu)]
        # The prefix block rows are frozen for the lifetime of an
        # incremental beam session (decode KV goes to the beam pool, not
        # the paged cache), so on a geometry hit the staged copy is
        # already in the address-stable buffer and can be skipped.
        copied = ctx.cache.setdefault("block_table_copied", set())
        if geom_key is None or geom_key not in copied:
            source = common_attn_metadata.block_table_tensor
            if (
                ctx.is_npu
                and geom_key is not None
                and all(row == index for index, row in enumerate(prefix_req_idxs_cpu))
            ):
                # A request-major incremental batch already has its
                # prefix blocks in this order. Basic slicing avoids the
                # implicit CPU index tensor upload of list indexing, so
                # first Decode preparation can continue during Prefill.
                source = source[: len(prefix_req_idxs_cpu), :max_prefix_blocks_count]
            else:
                source = source[prefix_req_idxs_cpu, :max_prefix_blocks_count]
            prefix_block_table[:, :max_prefix_blocks_count].copy_(source)
            if geom_key is not None:
                copied.add(geom_key)
    else:
        prefix_indices_tensor = torch.tensor(prefix_indices, dtype=torch.long, device=device)
        prefix_max_seq_len = max(prefix_seqlens_kv_cpu) if prefix_seqlens_kv_cpu else 0
        prefix_block_table = common_attn_metadata.block_table_tensor[
            prefix_req_idxs_cpu, :max_prefix_blocks_count
        ].contiguous()

    if ctx.fixed_buffers:
        num_prefix_groups = len(prefix_seqlens_kv_cpu)
        num_prefix_cu = len(prefix_cu_seqlens_q_cpu) + 1
        num_suffix_beams = len(suffix_seqlens_kv_cpu)
        num_suffix_cu_q = len(suffix_cu_seqlens_q_cpu) + 1
        num_suffix_cu_k = len(suffix_cu_seqlens_kv_cpu) + 1

        # Reuse the fixed-address seqlens views for a known geometry
        # instead of re-slicing the address-stable buffers every step.
        cached_views = cached_seqlens_views(
            ctx,
            geom_key,
            n_pk=num_prefix_groups,
            n_pc=num_prefix_cu,
            n_sk=num_suffix_beams,
            n_scq=num_suffix_cu_q,
            n_sck=num_suffix_cu_k,
        )
        if cached_views is not None:
            (
                prefix_seqlens_kv_device,
                prefix_cu_seqlens_q_device,
                suffix_seqlens_kv_device,
                suffix_cu_seqlens_q_device,
                suffix_cu_seqlens_k_device,
            ) = cached_views
        else:
            prefix_seqlens_kv_device = ctx.buffers.prefix_seqlens_kv[:num_prefix_groups]
            prefix_cu_seqlens_q_device = ctx.buffers.prefix_cu_seqlens_q[:num_prefix_cu]
            suffix_seqlens_kv_device = ctx.buffers.suffix_seqlens_kv[:num_suffix_beams]
            suffix_cu_seqlens_q_device = ctx.buffers.suffix_cu_seqlens_q[:num_suffix_cu_q]
            suffix_cu_seqlens_k_device = ctx.buffers.suffix_cu_seqlens_k[:num_suffix_cu_k]

        # Incremental decode admits the fixed-geometry cache: write the three
        # step-invariant payloads once per geometry, then refresh the
        # decode_step-derived suffix arrays on device (zero H2D).
        if upload_incremental_seqlens(
            ctx,
            geom_key,
            decode_step,
            prefix_seqlens_kv_view=prefix_seqlens_kv_device,
            prefix_cu_seqlens_q_view=prefix_cu_seqlens_q_device,
            suffix_seqlens_kv_view=suffix_seqlens_kv_device,
            suffix_cu_seqlens_q_view=suffix_cu_seqlens_q_device,
            suffix_cu_seqlens_k_view=suffix_cu_seqlens_k_device,
            prefix_seqlens_kv_cpu=prefix_seqlens_kv_cpu,
            prefix_cu_seqlens_q_cpu=prefix_cu_seqlens_q_cpu,
            suffix_cu_seqlens_q_cpu=suffix_cu_seqlens_q_cpu,
        ):
            pass
        else:
            # One pinned host->device transfer carries prefix + suffix
            # seqlens and cu_seqlens payloads; D2D scatter then fills the
            # fixed buffers.
            pack_int32_to_dev(
                ctx,
                [
                    (prefix_seqlens_kv_device, list(prefix_seqlens_kv_cpu)),
                    (prefix_cu_seqlens_q_device, [0] + list(prefix_cu_seqlens_q_cpu)),
                    (suffix_seqlens_kv_device, list(suffix_seqlens_kv_cpu)),
                    (suffix_cu_seqlens_q_device, [0] + list(suffix_cu_seqlens_q_cpu)),
                    (suffix_cu_seqlens_k_device, [0] + list(suffix_cu_seqlens_kv_cpu)),
                ],
            )
    elif ctx.is_cuda:
        # Tests / kernel harnesses that skip __init__ buffer alloc.
        prefix_seqlens_kv_device = torch.tensor(
            prefix_seqlens_kv_cpu, dtype=torch.int32, device=device
        )
        prefix_cu_seqlens_q_device = torch.tensor(
            [0] + prefix_cu_seqlens_q_cpu, dtype=torch.int32, device=device
        )
        suffix_seqlens_kv_device = torch.tensor(
            suffix_seqlens_kv_cpu, dtype=torch.int32, device=device
        )
        suffix_cu_seqlens_q_device = torch.tensor(
            [0] + suffix_cu_seqlens_q_cpu, dtype=torch.int32, device=device
        )
        suffix_cu_seqlens_k_device = torch.tensor(
            [0] + suffix_cu_seqlens_kv_cpu, dtype=torch.int32, device=device
        )
    else:
        # NPU: npu_fused_infer_attention_score_v2 currently reads
        # CPU lists (prefix_cu_seqlens_q_cpu / suffix_cu_seqlens_q_cpu etc.).
        # TODO(ACL-graph): switch NPU to device tensors so this
        # branch disappears and seqlens handling is unified.
        prefix_seqlens_kv_device = None
        prefix_cu_seqlens_q_device = None
        suffix_seqlens_kv_device = None
        suffix_cu_seqlens_q_device = None
        suffix_cu_seqlens_k_device = None

    # The FA2 scheduler metadata is a pure function of geometry-fixed
    # inputs on the incremental path (cached cu_seqlens views, constant
    # max lens, constant beam-width token count), so it is cached per
    # geom_key instead of being rebuilt every decode step.
    schedule_cache = ctx.cache.setdefault("schedule", {})
    if geom_key is not None and geom_key in schedule_cache:
        prefix_scheduler_metadata = schedule_cache[geom_key]
    else:
        prefix_scheduler_metadata = get_schedule(
            ctx,
            batch_size=len(prefix_seqlens_q_cpu),
            cu_query_lens=prefix_cu_seqlens_q_device,
            max_query_len=prefix_max_q_len,
            seqlens=prefix_seqlens_kv_device,
            max_seq_len=prefix_max_seq_len,
            causal=False,
            max_num_splits=max_num_splits,
            scheduler=scheduler,
        )
        if geom_key is not None:
            schedule_cache[geom_key] = prefix_scheduler_metadata

    return (
        prefix_indices_tensor,
        prefix_seqlens_kv_device,
        prefix_max_seq_len,
        prefix_cu_seqlens_q_device,
        prefix_max_q_len,
        prefix_block_table,
        suffix_seqlens_kv_device,
        suffix_max_seq_len,
        suffix_cu_seqlens_q_device,
        suffix_cu_seqlens_k_device,
        suffix_max_q_len,
        prefix_scheduler_metadata,
    )
