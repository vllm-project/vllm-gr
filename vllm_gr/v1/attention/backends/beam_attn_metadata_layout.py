# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Pure Beam Attention layout and mapping helpers."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


def _geom_cache_key(
    num_reqs: int,
    req_ids: list,
    beam_data: dict,
    q_starts: list,
    seq_lens_cpu_list: list[int],
    is_incremental: bool,
    block_size: int,
):
    """Return the geometry cache key for the SHAPE-ONLY (step-invariant)
    part of the batch.

    The build() geometry (prefix/suffix seqlens structure, indices,
    cu_seqlens, block table layout) depends only on the batch composition
    and the beam request geometry (widths / prefix_len / cache_len).  It
    does NOT depend on ``seq_lens_cpu_list``: for a pure beam batch the
    standard (non-beam) partition is empty and every beam length is derived
    from ``prefix_len``/``cache_len``/``beam_width``, which are constant for
    the lifetime of an incremental beam session.  The optimistic sequence
    length vLLM feeds in grows every decode step, so it must not be part of
    the key or the cache would miss every frame.  ``q_starts`` (the per-
    step query layout) is likewise excluded: it only offsets
    ``prefix_indices`` / block-table rows, which are written per step
    outside this cache; the cached payloads (fixed prefix/suffix seqlens
    + cu_seqlens_q) do not change with the query start offset.
    ``decode_step`` is intentionally excluded — only the two suffix
    step-derived arrays are rebuilt per step.  Returns None for non-beam or
    mixed batches, which then fall back to the full per-step construction
    path.
    """
    if not is_incremental or not beam_data or num_reqs == 0:
        return None
    try:
        beam_view = []
        for req_id in req_ids:
            mdata = beam_data.get(req_id)
            if mdata is None:
                return None
            if not mdata.get("is_beam_decode", False):
                return None
            beam_view.append(
                (
                    req_id,
                    mdata["beam_width"],
                    mdata["prefix_len"],
                    mdata["cache_len"],
                )
            )
    except (KeyError, TypeError):
        return None
    # Serializable key: request ids are stable within a session; keep the
    # tuple hashable and deterministic across steps.
    return (
        tuple(beam_view),
        block_size,
    )


def _evict_stale_geometry_entries(geom_key, cache) -> None:
    """Drop geometry-cache entries left behind by finished sessions.

    geom keys embed the request ids of their beam session.  When the ids
    of the CURRENT key share no request with a cached key, that entry
    belongs to a finished session (beam_finished_session_ids tore it down)
    and its host lists / device views / schedule will never be read again.
    Without eviction a long-running server accumulates one entry per
    historical request in each sub-cache (cu_lists, views, layout,
    schedule, block_table_copied), so the builder sweeps them whenever a
    NEW geometry is admitted.
    """
    if geom_key is None:
        return
    current_ids = {view[0] for view in geom_key[0]}
    for sub in (
        "cu_lists",
        "views",
        "layout",
        "schedule",
        "block_table_copied",
        "uploaded_geometries",
    ):
        bucket = cache.get(sub)
        if not bucket:
            continue
        stale = [k for k in bucket if not ({v[0] for v in k[0]} & current_ids)]
        for k in stale:
            # ``block_table_copied`` stores its keys in a set (values are
            # only membership markers); the other sub-caches are dicts.
            if isinstance(bucket, set):
                bucket.discard(k)
            else:
                del bucket[k]


def _segment_requests(
    num_reqs: int,
    req_ids: list,
    beam_data: dict,
    q_starts: list,
    seq_lens_cpu_list: list[int],
    beam_session=None,
    *,
    block_size: int,
):
    std_reqs = []
    beam_req_idxs = []
    beam_prefix_lens = []
    beam_widths = []
    beam_q_start_offsets = []
    beam_decode_step_list = []  # incremental decode step
    beam_pool_slots = []  # per beam-request slot index

    for i in range(num_reqs):
        req_id = req_ids[i] if i < len(req_ids) else None
        q_len = q_starts[i + 1] - q_starts[i]
        mdata = beam_data.get(req_id, {}) if req_id else {}
        if mdata.get("is_beam_decode", False):
            beam_req_idxs.append(i)
            prefix_len = mdata["prefix_len"]
            cache_len = mdata["cache_len"]
            delta = prefix_len - cache_len
            q_offset = 0
            # if prefix was cached out - run standard attention till prefix size
            if (delta > 0 and delta > block_size) or (cache_len == 0):
                std_q_len = min(delta, q_len)
                std_kv_len = cache_len + std_q_len
                cache_len = cache_len + std_q_len
                std_reqs.append((i, std_q_len, std_kv_len))
                q_offset = std_q_len

            beam_prefix_lens.append(prefix_len)
            beam_widths.append(mdata["beam_width"])
            beam_q_start_offsets.append(q_offset)
            beam_decode_step_list.append(mdata["decode_step"])

            # Get pool slot for this request from the session registry
            pool_slot = 0  # default for max_batch=1
            if beam_session is not None and req_id is not None:
                try:
                    pool_slot = beam_session.context.registry.get_slot(req_id)
                except KeyError:
                    pass  # session not yet registered, use default
            beam_pool_slots.append(pool_slot)
        else:
            kv_len = seq_lens_cpu_list[i]
            std_reqs.append((i, q_len, kv_len))

    return (
        std_reqs,
        beam_req_idxs,
        beam_prefix_lens,
        beam_widths,
        beam_q_start_offsets,
        beam_decode_step_list,
        beam_pool_slots,
    )


def _build_std_mappings(std_reqs: list, q_starts: list):
    std_indices = []
    std_seqlens_q_cpu = []
    std_seqlens_kv_cpu = []
    std_req_idxs = []

    for req_idx, q_len, kv_len in std_reqs:
        if q_len == 0:
            continue

        q_start = q_starts[req_idx]
        std_indices.extend(range(q_start, q_start + q_len))
        std_seqlens_q_cpu.append(q_len)
        std_seqlens_kv_cpu.append(kv_len)
        std_req_idxs.append(req_idx)

    std_cu_seqlens_q_cpu = None
    std_cu_seqlens_kv_cpu = None
    if std_seqlens_q_cpu:
        std_cu_seqlens_q_cpu = np.cumsum(std_seqlens_q_cpu).tolist()
        std_cu_seqlens_kv_cpu = np.cumsum(std_seqlens_kv_cpu).tolist()

    return (
        std_indices,
        std_seqlens_q_cpu,
        std_seqlens_kv_cpu,
        std_req_idxs,
        std_cu_seqlens_q_cpu,
        std_cu_seqlens_kv_cpu,
    )


class BeamMappings(NamedTuple):
    """Named field contract between layout construction and metadata binding."""

    prefix_indices: list[int]
    prefix_seqlens_q: list[int]
    prefix_seqlens_kv: list[int]
    suffix_seqlens_q: list[int]
    suffix_seqlens_kv: list[int]
    prefix_req_idxs: list[int]
    prefix_cu_seqlens_q: list[int] | None
    prefix_cu_seqlens_kv: list[int] | None
    suffix_cu_seqlens_q: list[int] | None
    suffix_cu_seqlens_kv: list[int] | None
    max_prefix_blocks_count: int
    prefix_indices_contiguous: bool
    prefix_max_q_len: int
    suffix_max_q_len: int
    suffix_max_seq_len: int


def _build_beam_mappings(
    beam_req_idxs,
    beam_prefix_lens,
    beam_widths,
    beam_q_start_offsets,
    q_starts,
    beam_decode_step_list,
    *,
    geom_key=None,
    block_size: int,
    cache: dict,
) -> BeamMappings:
    """Build one query per beam, sharing prefix KV and reading suffix KV from pools."""
    prefix_indices = []
    prefix_q = []
    prefix_kv = []
    suffix_q = []
    suffix_kv = []
    prefix_req_idxs = []
    contiguous = True
    previous_end = None
    for req_idx, prefix_len, width, offset, step in zip(
        beam_req_idxs,
        beam_prefix_lens,
        beam_widths,
        beam_q_start_offsets,
        beam_decode_step_list,
        strict=True,
    ):
        start = q_starts[req_idx] + offset
        end = q_starts[req_idx + 1]
        if start == end:
            # A prefix-only chunk has no beam queries yet.
            continue
        if end - start != width:
            raise ValueError("Beam decode requires exactly one query token per beam")
        contiguous = contiguous and (previous_end is None or previous_end == start)
        previous_end = end
        prefix_indices.extend(range(start, end))
        prefix_q.append(width)
        prefix_kv.append(prefix_len)
        prefix_req_idxs.append(req_idx)
        suffix_q.extend([1] * width)
        suffix_kv.extend([step + 1] * width)

    fixed_cache = cache.setdefault("cu_lists", {})
    fixed = fixed_cache.get(geom_key) if geom_key is not None else None
    if fixed is None:
        fixed = (
            np.cumsum(prefix_q).tolist() if prefix_q else None,
            np.cumsum(prefix_kv).tolist() if prefix_kv else None,
            list(range(1, len(suffix_q) + 1)) if suffix_q else None,
        )
        if geom_key is not None and suffix_q:
            fixed_cache[geom_key] = fixed
    # Mixed batches need not have the same Decode step for every request.
    suffix_cu_kv = None
    if suffix_kv:
        if geom_key is not None:
            step_length = suffix_kv[0]
            suffix_cu_kv = list(range(step_length, (len(suffix_kv) + 1) * step_length, step_length))
        else:
            suffix_cu_kv = np.cumsum(suffix_kv).tolist()
    max_prefix_len = max(beam_prefix_lens, default=0)
    return BeamMappings(
        prefix_indices=prefix_indices,
        prefix_seqlens_q=prefix_q,
        prefix_seqlens_kv=prefix_kv,
        suffix_seqlens_q=suffix_q,
        suffix_seqlens_kv=suffix_kv,
        prefix_req_idxs=prefix_req_idxs,
        prefix_cu_seqlens_q=fixed[0],
        prefix_cu_seqlens_kv=fixed[1],
        suffix_cu_seqlens_q=fixed[2],
        suffix_cu_seqlens_kv=suffix_cu_kv,
        max_prefix_blocks_count=(max_prefix_len + block_size - 1) // block_size,
        prefix_indices_contiguous=contiguous,
        prefix_max_q_len=max(prefix_q, default=0),
        suffix_max_q_len=1 if suffix_q else 0,
        suffix_max_seq_len=max(suffix_kv, default=0),
    )
