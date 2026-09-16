# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-neutral ModelRunner helpers for grouped Beam execution."""

from __future__ import annotations

import gc
import threading
import time
from functools import wraps
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.beam.geometry import _compute_beam_bounds

logger = init_logger(__name__)

_WORKER_EXECUTE_PATCH_MARKER = "_vllm_gr_beam_execute_patch"
_WORKER_PREPARE_PATCH_MARKER = "_vllm_gr_beam_prepare_patch"
_WORKER_BOOKKEEPING_PATCH_MARKER = "_vllm_gr_beam_bookkeeping_patch"
# Optional: only installed when the runner exposes ``_sample`` (NPU builds may
# not).  is_worker_runner_patched() deliberately does not require it.
_WORKER_SAMPLE_PATCH_MARKER = "_vllm_gr_beam_sample_patch"
_WORKER_SAMPLE_TOKENS_PATCH_MARKER = "_vllm_gr_beam_sample_tokens_patch"
_WORKER_DETERMINE_PATCH_MARKER = "_vllm_gr_beam_determine_patch"
_WORKER_CAPTURE_PATCH_MARKER = "_vllm_gr_beam_capture_patch"
_WORKER_MODEL_FORWARD_PATCH_MARKER = "_vllm_gr_prefill_model_forward_patch"


def _get_positions_buffers(runner: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Safely extracts underlying hardware and host indexing position tensors."""
    gpu_positions = None
    cpu_positions = None
    if hasattr(runner, "positions") and runner.positions is not None:
        if isinstance(runner.positions, torch.Tensor):
            # vLLM-Ascend 0.22 uses a plain device tensor in PCP paths.
            gpu_positions = runner.positions
        else:
            gpu_positions = getattr(runner.positions, "gpu", None)
            cpu_positions = getattr(runner.positions, "cpu", None)
    if cpu_positions is None:
        # vLLM-Ascend 0.22 keeps a dedicated pinned host buffer even when
        # ``positions`` itself is a plain device tensor.
        cpu_positions = getattr(runner, "_positions_cpu_buf", None)
    return gpu_positions, cpu_positions


def _write_position_segments(
    token_offset: int,
    seg_starts: list[int],
    seg_lens: list[int],
    gpu_pos: torch.Tensor | None,
    cpu_pos: torch.Tensor | None,
) -> None:
    """Write concatenated per-beam position-id segments in one pass.

    Each active beam contributes ``range(st_b, st_b + len_b)``.  The result is
    element-wise identical to the historical per-beam ``torch.arange`` writes,
    but the concatenation is built with numpy on the host (microseconds) and
    committed with a single host->device copy for ``gpu_pos``.

    NOTE: the previous implementation used ``torch.repeat_interleave`` to
    expand per-segment offsets; on Ascend that aten op is a pure host-side
    stall (profiling bw128_1k_t3: 45.5ms + 90.6ms per decode step with zero
    device activity), because the ACL backend materializes the repeat on
    device. Asynchronous scheduling instead builds positions with plain NumPy
    writes before uploading the completed layout.
    """
    if not seg_lens:
        return
    import numpy as np

    lens = np.asarray(seg_lens, dtype=np.int64)
    total = int(lens.sum())
    if total == 0:
        return
    starts = np.asarray(seg_starts, dtype=np.int64)
    # Output offset where each segment begins (prefix sum minus own length).
    seg_off = np.cumsum(lens) - lens
    gidx = np.arange(total, dtype=np.int64)
    # positions[start:end] = st + arange(len) for each segment, vectorized:
    # repeat the per-segment start/offset by the segment length and add the
    # within-segment index (gidx - seg_off_rep).
    starts_rep = np.repeat(starts, lens)
    seg_off_rep = np.repeat(seg_off, lens)
    positions = torch.from_numpy(starts_rep + (gidx - seg_off_rep))
    if gpu_pos is not None:
        gpu_pos[token_offset : token_offset + total] = positions.to(gpu_pos.device)
    if cpu_pos is not None:
        cpu_pos[token_offset : token_offset + total] = positions.to(cpu_pos.device)


def _process_beam_decode_request(
    mdata: dict[str, Any],
    seq_len: int,
    token_offset: int,
    gpu_pos: torch.Tensor | None,
    cpu_pos: torch.Tensor | None,
    new_logits_indices: list[int],
    is_incremental: bool = False,
) -> tuple[int, int, int, list[int]]:
    """Traces allocation arrays and mutates position buffers for a speculative request.

    When is_incremental=True, each beam inputs 1 token with position_id = prefix_len + decode_step.
    """
    prefix_len = mdata["prefix_len"]
    beam_width = mdata["beam_width"]
    decode_steps = mdata["beam_decode_steps"]
    cache_len = mdata["cache_len"]
    active_beam_ids = []  # Track structural beam locations

    # --- Incremental decode path ---
    if is_incremental:
        decode_step = mdata.get("decode_step", 0)
        # W identical single-token rows: contiguous logits rows and the full
        # beam set.  Built with C-level list ops instead of a per-beam Python
        # append loop (the loop dominated the host time of this function for
        # beam_width=128 on every decode step).
        new_logits_indices.extend(range(token_offset, token_offset + beam_width))
        active_beam_ids = list(range(beam_width))

        # All beams share one constant position id this step, so a scalar
        # slice-fill is enough: a single device-side fill kernel, no host
        # build, no H2D copy (matches the recompute path's no-stall goal).
        pos_id = prefix_len + decode_step
        if gpu_pos is not None:
            gpu_pos[token_offset : token_offset + beam_width] = pos_id
        if cpu_pos is not None:
            cpu_pos[token_offset : token_offset + beam_width] = pos_id

        return beam_width, beam_width, token_offset + beam_width, active_beam_ids

    # --- Standard (recompute) path ---
    chunk_budget = seq_len
    curr_offset = token_offset
    valid_rows_for_req = 0
    logits_row_cnt = 0
    b_cache_len = cache_len
    seg_starts = []
    seg_lens = []
    for b in range(beam_width):
        if chunk_budget <= 0:
            break

        b_uncached, b_suffix_len, b_past_suffix, b_physical = _compute_beam_bounds(
            b, prefix_len, b_cache_len, decode_steps, chunk_budget
        )

        if b_uncached > 0:
            st = cache_len if (prefix_len >= b_cache_len) else (prefix_len + b_past_suffix)
            seg_starts.append(st)
            seg_lens.append(b_uncached)

            if b_past_suffix + b_uncached >= decode_steps:
                new_logits_indices.append(curr_offset + b_uncached - 1)
                active_beam_ids.append(b_physical)

                logits_row_cnt += 1

            valid_rows_for_req += 1
            curr_offset += b_uncached
            chunk_budget -= b_uncached
            b_cache_len += b_uncached

    # Batched position-id write: one vectorized build (CPU) + single H2D copy
    # instead of one torch.arange per beam (up to beam_width GPU launches/call).
    _write_position_segments(token_offset, seg_starts, seg_lens, gpu_pos, cpu_pos)

    return valid_rows_for_req, logits_row_cnt, curr_offset, active_beam_ids


def _remux_logprobs_tensors(
    request_row_configs: list[tuple[Any, int, int]],
    max_chained_dim: int,
    target_K: int,
    orig_lps: torch.Tensor,
    orig_lp_ids: torch.Tensor | None,
    orig_ranks: torch.Tensor | None,
    st_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Merges and aligns sampler tensor slices sequentially across runtime waves."""
    if max_chained_dim == 0 or orig_lps.numel() == 0:
        return st_ids, orig_lps, orig_lp_ids, orig_ranks

    new_st_ids, new_lps, new_lp_ids, new_ranks = [], [], [], []
    gpu_row_cursor = 0
    logical_row_cursor = 0
    logical_st_ids = st_ids.shape[0] == sum(
        req_id is not None for req_id, *_ in request_row_configs
    )

    for req_id, _, W_logits, *_ in request_row_configs:
        # CASE A: INTERMEDIATE CHUNKED PREFILL (No output rows generated)
        if W_logits == 0:
            dummy_st = torch.tensor([[-1]], dtype=st_ids.dtype, device=st_ids.device)
            new_st_ids.append(dummy_st)

            new_lps.append(
                torch.full(
                    (1, max_chained_dim),
                    float("-inf"),
                    dtype=orig_lps.dtype,
                    device=orig_lps.device,
                )
            )
            if orig_lp_ids is not None:
                new_lp_ids.append(
                    torch.zeros(
                        (1, max_chained_dim), dtype=orig_lp_ids.dtype, device=orig_lp_ids.device
                    )
                )
            if orig_ranks is not None:
                new_ranks.append(
                    torch.zeros(
                        (1, max_chained_dim), dtype=orig_ranks.dtype, device=orig_ranks.device
                    )
                )
            continue

        # CASE B: ACTIVE SAMPLING PATH (Captures the full multi-beam footprint)
        sampled_row = logical_row_cursor if logical_st_ids else gpu_row_cursor
        new_st_ids.append(st_ids[sampled_row : sampled_row + 1, :])
        logical_row_cursor += 1

        req_lps = orig_lps[gpu_row_cursor : gpu_row_cursor + W_logits, :target_K].reshape(
            1, W_logits * target_K
        )
        if W_logits * target_K < max_chained_dim:
            req_lps = torch.nn.functional.pad(
                req_lps, (0, max_chained_dim - (W_logits * target_K)), value=float("-inf")
            )
        new_lps.append(req_lps)

        if orig_lp_ids is not None:
            req_lp_ids = orig_lp_ids[gpu_row_cursor : gpu_row_cursor + W_logits, :target_K].reshape(
                1, W_logits * target_K
            )
            if W_logits * target_K < max_chained_dim:
                req_lp_ids = torch.nn.functional.pad(
                    req_lp_ids, (0, max_chained_dim - (W_logits * target_K)), value=0
                )
            new_lp_ids.append(req_lp_ids)

        if orig_ranks is not None:
            req_ranks = orig_ranks[gpu_row_cursor : gpu_row_cursor + W_logits].reshape(1, W_logits)
            if W_logits < max_chained_dim:
                req_ranks = torch.nn.functional.pad(
                    req_ranks, (0, max_chained_dim - W_logits), value=0
                )
            new_ranks.append(req_ranks)

        gpu_row_cursor += W_logits

    return (
        torch.cat(new_st_ids, dim=0),
        torch.cat(new_lps, dim=0),
        torch.cat(new_lp_ids, dim=0) if orig_lp_ids is not None else None,
        torch.cat(new_ranks, dim=0) if orig_ranks is not None else None,
    )


class BeamReqIdsProxy(list):
    def __init__(self, original_list, req_id_map):
        super().__init__(original_list)
        self.req_id_map = req_id_map

    def __getitem__(self, index):
        try:
            return self.req_id_map[index]
        except IndexError as e:
            logger.error("Out of bounds in BeamReqIdsProxy for index %s", index)
            raise e

    def copy(self):
        return list(self.req_id_map)


class Mock1DArray:
    def __init__(self, original, batch_idx_map):
        self.original = original
        self.batch_idx_map = batch_idx_map

    def __getitem__(self, idx):
        try:
            return self.original[self.batch_idx_map[idx]]
        except IndexError as e:
            logger.error("Out of bounds in Mock1DArray getitem for index %s", idx)
            raise e

    def __setitem__(self, idx, value):
        try:
            self.original[self.batch_idx_map[idx]] = value
        except IndexError as e:
            logger.error("Out of bounds in Mock1DArray setitem for index %s", idx)
            raise e


class Mock2DArray:
    def __init__(self, original, batch_idx_map):
        self.original = original
        self.batch_idx_map = batch_idx_map

    def __setitem__(self, key, value):
        idx, slc = key
        try:
            self.original[self.batch_idx_map[idx], slc] = value
        except IndexError as e:
            logger.error("Out of bounds in Mock2DArray setitem for key %s", key)
            raise e

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, key):
        idx, slc = key
        try:
            return self.original[self.batch_idx_map[idx], slc]
        except IndexError as e:
            logger.error("Out of bounds in Mock2DArray getitem for key %s", key)
            raise e


# Per-request sampling tensors remapped onto per-beam logits rows by
# InputBatchProxy. Shared across the remap pass so one cached device index
# tensor serves every attribute (instead of one host->device list transfer
# per attribute, the `aten::index` + `aten::to/copy_` pairs in profiling
# bw128_1k_t3).
_SAMPLING_META_ATTRS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalties",
    "presence_penalties",
    "repetition_penalties",
)


def _first_device_of(s_meta) -> Any:
    """Return the device of the first tensor sampling attribute, or None."""
    for attr in _SAMPLING_META_ATTRS:
        val = getattr(s_meta, attr, None)
        if isinstance(val, torch.Tensor) and val.numel() > 0:
            return val.device
    return None


class InputBatchProxy:
    def __init__(
        self,
        obj,
        req_map,
        idx_map,
        logits_map,
        shared_idx_cache=None,
        shared_sampling_cache=None,
        req_ids_key=None,
    ):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_req_id_map", req_map)
        object.__setattr__(self, "_batch_idx_map", idx_map)
        object.__setattr__(self, "_logit_idx_map", logits_map)
        # Frozen tuple of the logits map, computed once per proxy and shared
        # by the idx-tensor and sampling-remap cache keys (two keys per frame
        # would otherwise materialize the 128-element tuple twice).
        object.__setattr__(
            self, "_logit_idx_tuple", tuple(logits_map) if logits_map is not None else None
        )
        # Frozen tuple of the batch's request ids — the request-identity
        # component of the sampling-remap cache key.  Beam-row expansions of
        # the sampling attributes are only reusable WITHIN one request's
        # decode session; a later request with the same beam width and batch
        # layout would otherwise silently inherit the previous request's
        # temperature/top-p/penalty tensors.
        object.__setattr__(
            self, "_req_ids_key", tuple(req_ids_key) if req_ids_key is not None else None
        )
        # Runner-owned single-slot cache shared across per-frame proxy
        # instances.  Consecutive incremental decode steps carry the same
        # logits_map (e.g. ``[0]*beam_width`` for a single beam request), so
        # the host->device logits-row index tensor is rebuilt once per
        # geometry instead of once per frame.
        object.__setattr__(self, "_shared_idx_cache", shared_idx_cache)
        # Runner-owned single-slot cache for the REMAPPED sampling tensors.
        # vLLM v1 per-request sampling params are immutable for a request's
        # lifetime (written once by add_request), and a beam decode session is
        # one request, so the beam-row expansion of the 7 sampling attrs is a
        # per-geometry constant WITHIN that request.  The cache key carries
        # the request ids so a different request with the same layout always
        # misses and re-gathers from its own params.  A hit skips the
        # per-frame stack + index_select + unbind gather chain entirely.
        object.__setattr__(self, "_shared_sampling_cache", shared_sampling_cache)
        # Remapped sampling metadata, cached for this proxy's lifetime.
        # Holds the underlying ``sampling_metadata`` object after its
        # per-request tensors were remapped onto logits rows; validated by
        # identity so a mid-step rebuild (refresh_metadata) is picked up.
        object.__setattr__(self, "_sampling_metadata_cache", None)
        # (device, tensor) cache for the shared logits-row index tensor.
        object.__setattr__(self, "_logits_idx_tensor_cache", None)

    def _logits_idx_tensor(self, device):
        cache = object.__getattribute__(self, "_logits_idx_tensor_cache")
        if cache is not None and cache[0] == device:
            return cache[1]
        shared = object.__getattribute__(self, "_shared_idx_cache")
        key = (device, object.__getattribute__(self, "_logit_idx_tuple"))
        if shared is not None:
            hit = shared.get(key)
            if hit is not None:
                object.__setattr__(self, "_logits_idx_tensor_cache", (device, hit))
                return hit
        idx = torch.tensor(
            object.__getattribute__(self, "_logit_idx_map"),
            dtype=torch.long,
            device=device,
        )
        if shared is not None:
            # Keep only the current geometry's entry; the hot path is
            # frame-to-frame reuse within one beam session.
            shared.clear()
            shared[key] = idx
        object.__setattr__(self, "_logits_idx_tensor_cache", (device, idx))
        return idx

    def __getattr__(self, name):
        obj = object.__getattribute__(self, "_obj")
        idx_map = object.__getattribute__(self, "_batch_idx_map")
        logits_map = object.__getattribute__(self, "_logit_idx_map")
        req_map = object.__getattribute__(self, "_req_id_map")

        if name == "sampling_metadata":
            s_meta = getattr(obj, "sampling_metadata", None)
            if s_meta is None or logits_map is None:
                return s_meta

            cached = object.__getattribute__(self, "_sampling_metadata_cache")
            if cached is not None and cached is s_meta:
                # Fast path: this exact metadata object was already remapped.
                # Repeated accesses within the step (sampler, constraint
                # hook, bookkeeping) skip the per-attribute loop entirely.
                return cached

            orig_req_len = len(obj.req_ids) if hasattr(obj, "req_ids") else 1
            shared_sampling = object.__getattribute__(self, "_shared_sampling_cache")
            remap_key = None
            if shared_sampling is not None and orig_req_len != len(logits_map):
                # Beam-row expansion is a per-REQUEST constant: vLLM v1
                # per-request sampling params are immutable once the request
                # is admitted, and the logits_map pins the batch layout.  The
                # request ids in the key prevent a LATER request with the
                # same beam width and layout from inheriting this request's
                # gathered tensors.
                remap_key = (
                    _first_device_of(s_meta),
                    orig_req_len,
                    object.__getattribute__(self, "_req_ids_key"),
                    object.__getattribute__(self, "_logit_idx_tuple"),
                )
                hit = shared_sampling.get(remap_key)
                if hit is not None:
                    for attr_name, tensor in hit.items():
                        object.__setattr__(s_meta, attr_name, tensor)
                    object.__setattr__(self, "_sampling_metadata_cache", s_meta)
                    return s_meta
            # Group per-request sampling tensors by dtype and remap each group
            # with a single device gather instead of one ``aten::index`` kernel
            # per attribute (7 kernels per decode step in profiling).  The
            # per-attribute results are views into the gathered buffer, so no
            # extra kernels are emitted when the sampler consumes them.
            pending: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
            for attr in _SAMPLING_META_ATTRS:
                val = getattr(s_meta, attr, None)
                if isinstance(val, torch.Tensor) and val.numel() > 0:
                    if val.shape[0] == orig_req_len and val.shape[0] != len(logits_map):
                        pending.setdefault(val.dtype, []).append((attr, val))
            if pending:
                first_val = next(iter(next(iter(pending.values()))))[1]
                idx_tensor = self._logits_idx_tensor(first_val.device)
                remapped: dict[str, torch.Tensor] = {}
                for dtype, items in pending.items():
                    names = [n for n, _ in items]
                    # Stack attributes on dim 0 ([num_attrs, num_reqs]) and
                    # gather on dim 1, so each remapped attribute comes out
                    # contiguous (stride 1).  Stacking on dim -1 then unbinding
                    # dim -1 yields strided column views (e.g. top_p with
                    # stride 5); vLLM 0.22.1's Triton top-p kernel loads the
                    # row pointer without a stride, so a non-contiguous column
                    # would silently read interleaved penalties/temperature as
                    # top_p values and corrupt filtering.
                    stacked = torch.stack([v for _, v in items], dim=0)
                    gathered = stacked.index_select(1, idx_tensor)
                    for attr_name, col in zip(names, gathered.unbind(dim=0)):
                        object.__setattr__(s_meta, attr_name, col)
                        remapped[attr_name] = col
                if shared_sampling is not None and remap_key is None:
                    remap_key = (
                        first_val.device,
                        orig_req_len,
                        object.__getattribute__(self, "_req_ids_key"),
                        object.__getattribute__(self, "_logit_idx_tuple"),
                    )
                if shared_sampling is not None and remap_key is not None:
                    # Single slot: the hot path is frame-to-frame reuse within
                    # one beam session; a geometry change replaces the entry.
                    shared_sampling.clear()
                    shared_sampling[remap_key] = remapped
            object.__setattr__(self, "_sampling_metadata_cache", s_meta)
            return s_meta

        elif name == "req_ids":
            return BeamReqIdsProxy(obj.req_ids, req_map)
        elif name == "num_tokens_no_spec":
            return Mock1DArray(obj.num_tokens_no_spec, idx_map)
        elif name == "num_tokens":
            # NPU's _bookkeeping_sync writeback loop also updates num_tokens
            # (the GPU path does not). Map it to the logical batch slot like
            # num_tokens_no_spec. Inert when never accessed (e.g. GPU).
            return Mock1DArray(obj.num_tokens, idx_map)
        elif name == "token_ids_cpu":
            return Mock2DArray(obj.token_ids_cpu, idx_map)
        elif name == "is_token_ids":
            return (
                Mock2DArray(obj.is_token_ids, idx_map)
                if hasattr(obj, "is_token_ids")
                else Mock2DArray(obj.token_ids_cpu, idx_map)
            )
        return getattr(obj, name)

    def __setattr__(self, name, value):
        if name == "sampling_metadata":
            # The underlying metadata object is being replaced; drop the
            # remapped cache so the next read reindexes fresh tensors.
            object.__setattr__(self, "_sampling_metadata_cache", None)
        setattr(object.__getattribute__(self, "_obj"), name, value)


def _beam_remap_inputs(runner, scheduler_output, num_scheduled_tokens) -> list[int]:
    """Expand a packed beam-decode batch into per-beam rows.

    Device-neutral. Overwrites per-beam RoPE position ids (Job B), builds the
    per-beam ``logits_indices`` list, installs an ``InputBatchProxy`` on
    ``runner.input_batch`` and stashes ``scheduler_output._gr_request_row_configs``
    (Job C). Job A (populating ``beam_info["req_ids"]``) is left to the caller
    because its ordering relative to the original ``_prepare_inputs`` differs
    between GPU and NPU.

    Assumes ``scheduler_output.beam_data`` is non-empty (caller checks).

    Returns:
        The new per-beam logits-index list (caller converts to a tensor).
    """
    beam_data = getattr(scheduler_output, "beam_data", {})

    runner.input_batch.logprob_token_ids = {}
    orig_req_ids = runner.input_batch.req_ids

    # Fast path: a pure incremental-beam batch replays a cached row layout.
    # When every scheduled request is an incremental beam decode with the same
    # widths and token counts as the cached frame, the four host lists and the
    # logits index list are identical frame to frame; only the position ids
    # depend on decode_step.  The per-beam Python loop below is skipped
    # entirely on the hot path.
    layout_cache = getattr(runner, "_gr_remap_layout_cache", None)
    if layout_cache is None:
        layout_cache = {}
        runner._gr_remap_layout_cache = layout_cache
    try:
        layout_key = (
            tuple(orig_req_ids),
            tuple(int(x) for x in num_scheduled_tokens[: len(orig_req_ids)]),
        )
        all_incremental = (
            all(
                (m := beam_data.get(rid)) is not None
                and m.get("is_beam_decode", False)
                and "decode_step" in m
                for rid in orig_req_ids
                if rid is not None
            )
            and len(orig_req_ids) > 0
            and not any(rid is None for rid in orig_req_ids)
        )
    except (TypeError, ValueError):
        layout_key = None
        all_incremental = False
    if layout_key is not None and all_incremental:
        cached_layout = layout_cache.get(layout_key)
        if cached_layout is not None:
            (
                row_to_req_id,
                row_to_batch_idx,
                logits_to_batch_idx,
                request_row_configs,
                new_logits_indices,
                widths_offsets,
            ) = cached_layout
            # Refresh only the step-dependent position ids.
            gpu_positions, cpu_positions = _get_positions_buffers(runner)
            token_offset = 0
            for req_id, width, prefix_len in widths_offsets:
                mdata = beam_data[req_id]
                pos_id = prefix_len + int(mdata.get("decode_step", 0))
                if gpu_positions is not None:
                    gpu_positions[token_offset : token_offset + width] = pos_id
                if cpu_positions is not None:
                    cpu_positions[token_offset : token_offset + width] = pos_id
                token_offset += width
            shared_idx_cache = getattr(runner, "_gr_logits_idx_cache", None)
            if shared_idx_cache is None:
                shared_idx_cache = {}
                runner._gr_logits_idx_cache = shared_idx_cache
            shared_sampling_cache = getattr(runner, "_gr_sampling_remap_cache", None)
            if shared_sampling_cache is None:
                shared_sampling_cache = {}
                runner._gr_sampling_remap_cache = shared_sampling_cache
            runner.input_batch = InputBatchProxy(
                runner.input_batch,
                row_to_req_id,
                row_to_batch_idx,
                logits_to_batch_idx,
                shared_idx_cache=shared_idx_cache,
                shared_sampling_cache=shared_sampling_cache,
                req_ids_key=tuple(orig_req_ids),
            )
            scheduler_output._gr_request_row_configs = request_row_configs
            _ = runner.input_batch.sampling_metadata
            return new_logits_indices

    new_logits_indices: list[int] = []
    token_offset = 0
    row_to_req_id, row_to_batch_idx, logits_to_batch_idx, request_row_configs = ([], [], [], [])
    input_row_idx = 0
    incremental_widths: list[tuple[Any, int, int]] = []

    gpu_positions, cpu_positions = _get_positions_buffers(runner)

    for batch_idx, req_id in enumerate(orig_req_ids):
        seq_len = num_scheduled_tokens[batch_idx] if batch_idx < len(num_scheduled_tokens) else 0
        if req_id is None:
            row_to_req_id.append(None)
            row_to_batch_idx.append(batch_idx)
            input_row_idx += 1
            token_offset += seq_len
            continue

        mdata = beam_data.get(req_id)
        if mdata and mdata.get("is_beam_decode", False):
            # Check if incremental decode is active
            is_incremental = "decode_step" in mdata
            beams_count, beams_count_with_logits, next_offset, active_beam_ids = (
                _process_beam_decode_request(
                    mdata,
                    seq_len,
                    token_offset,
                    gpu_positions,
                    cpu_positions,
                    new_logits_indices,
                    is_incremental=is_incremental,
                )
            )
            row_to_req_id.extend([req_id] * beams_count)
            row_to_batch_idx.extend([batch_idx] * beams_count)
            logits_to_batch_idx.extend([batch_idx] * beams_count_with_logits)

            request_row_configs.append(
                (req_id, input_row_idx, beams_count_with_logits, active_beam_ids)
            )
            input_row_idx += beams_count
            if is_incremental:
                incremental_widths.append((req_id, mdata["beam_width"], mdata["prefix_len"]))
        else:
            num_computed_tokens = runner.input_batch.num_computed_tokens_cpu[batch_idx]
            num_prompt_tokens = runner.input_batch.num_prompt_tokens[batch_idx]

            is_last_chunk_prefill = bool(
                num_computed_tokens < num_prompt_tokens
                and num_computed_tokens + seq_len >= num_prompt_tokens
            )
            beams_count_with_logits = 0
            beams_count = 1
            if is_last_chunk_prefill:
                new_logits_indices.append(token_offset + seq_len - 1)
                beams_count_with_logits = 1
                logits_to_batch_idx.extend([batch_idx] * beams_count_with_logits)
            row_to_req_id.extend([req_id] * beams_count)
            row_to_batch_idx.extend([batch_idx] * beams_count)

            request_row_configs.append((req_id, input_row_idx, beams_count_with_logits, []))
            input_row_idx += beams_count

        token_offset += seq_len

    shared_idx_cache = getattr(runner, "_gr_logits_idx_cache", None)
    if shared_idx_cache is None:
        shared_idx_cache = {}
        runner._gr_logits_idx_cache = shared_idx_cache
    shared_sampling_cache = getattr(runner, "_gr_sampling_remap_cache", None)
    if shared_sampling_cache is None:
        shared_sampling_cache = {}
        runner._gr_sampling_remap_cache = shared_sampling_cache
    runner.input_batch = InputBatchProxy(
        runner.input_batch,
        row_to_req_id,
        row_to_batch_idx,
        logits_to_batch_idx,
        shared_idx_cache=shared_idx_cache,
        shared_sampling_cache=shared_sampling_cache,
        req_ids_key=tuple(orig_req_ids),
    )
    scheduler_output._gr_request_row_configs = request_row_configs

    # Register the layout for the pure incremental fast path.  Only batches
    # where EVERY scheduled request went through the incremental beam branch
    # are eligible (mixed prefill/None rows keep the per-frame loop).
    if (
        layout_key is not None
        and len(incremental_widths) == len(orig_req_ids)
        and len(orig_req_ids) > 0
    ):
        layout_cache.clear()
        layout_cache[layout_key] = (
            row_to_req_id,
            row_to_batch_idx,
            logits_to_batch_idx,
            request_row_configs,
            new_logits_indices,
            incremental_widths,
        )

    # Warm the remapped sampling metadata now, in the prepare phase, instead
    # of lazily at the first ``_sample`` access: the per-request tensor remap
    # is host-side work that would otherwise land right after the model
    # forward, where the device sits idle waiting for the sampler (the
    # post-forward host stall in profiling bw128_1k_t3).  refresh_metadata
    # has already run (it is the tail of _update_states, executed at the top
    # of execute_model), so this is the step's final metadata object.  Harmless
    # no-op when the metadata is not built yet (returns None, caches nothing).
    _ = runner.input_batch.sampling_metadata

    return new_logits_indices


def _materialize_logits_indices(runner: Any, indices: list[int]) -> torch.Tensor:
    """Copy per-beam ``logits_indices`` into a runner-owned reusable buffer.

    The legacy path built a fresh ``torch.tensor(list, device=...)`` on every
    beam step, paying a CUDA allocation + host->device transfer each time.
    This keeps one int32 device buffer on the runner (sized to the first
    request's footprint) and reuses it in place, so repeated decode steps only
    pay the host->device ``copy_`` of the newly expanded indices.

    Fully incremental beam decode fills ``indices`` as the contiguous range
    ``[0, num_logits)``; that shape is materialized directly on the device with
    ``torch.arange``, eliminating the host->device copy (and the Python list
    itself) entirely.  A non-contiguous index list (chunked prefill / standard
    recompute rows) falls back to the host copy path.

    Returns a view of the buffer containing exactly the given indices.  The
    caller must consume the value before the next call to this helper (the
    buffer is intentionally overwritten on the next beam step), matching the
    one-step lifetime of ``logits_indices`` inside ``execute_model``.
    """
    n = len(indices)
    buffer = getattr(runner, "_gr_logits_indices_buffer", None)
    if buffer is None or buffer.numel() < n:
        buffer = torch.empty(n, dtype=torch.int32, device=runner.device)
        runner._gr_logits_indices_buffer = buffer

    # Contiguous incremental decode: materialize on device, no H2D.
    if n and indices[-1] == n - 1 and _is_contiguous_range(indices):
        torch.arange(n, dtype=torch.int32, device=runner.device, out=buffer[:n])
        return buffer[:n]

    buffer[:n].copy_(torch.as_tensor(indices, dtype=torch.int32))
    return buffer[:n]


def _is_contiguous_range(indices: list[int]) -> bool:
    """Return whether ``indices`` equals ``range(0, len(indices))``.

    Pure incremental beam decode produces exactly this shape; detecting it
    lets ``_materialize_logits_indices`` emit a device-side ``arange`` and
    skip the host->device transfer entirely.  The ``[-1]`` sentinel is cheap
    and covers the common (single-request) case; the full loop only runs when
    the tail matches.
    """
    if not indices:
        return True
    for i, value in enumerate(indices):
        if value != i:
            return False
    return True


def _beam_collapse_bookkeeping(
    runner, orig_input_batch, request_row_configs, beam_data, sampler_output
) -> tuple[list[Any], dict[Any, int], int]:
    """Collapse per-beam sampler output back onto logical (session) requests.

    Device-neutral. Remuxes the sampler logprobs tensors in place so that each
    logical request owns one flattened ``W * K`` row, and resizes the sampler
    output tensors accordingly.

    Returns:
        ``(collapsed_req_ids, collapsed_req_id_to_index, num_logical_requests)``.
    """
    collapsed_req_ids_copy = [r[0] for r in request_row_configs if r[0] is not None]
    collapsed_req_id_to_index = {req_id: idx for idx, req_id in enumerate(collapsed_req_ids_copy)}
    num_logical_requests = len(collapsed_req_ids_copy)

    if sampler_output is not None:
        st_ids = sampler_output.sampled_token_ids
        lp_tensors = sampler_output.logprobs_tensors
        if getattr(sampler_output, "_gr_beam_result_transport", False):
            # Worker-side selection already reduced the candidate matrix to
            # one O(W) row per logical request before upstream bookkeeping.
            # The selected parent mapping travels independently in
            # ModelRunnerOutput.beam_search_result; do not expand this small
            # compatibility row back into the legacy W-by-W payload.
            if st_ids is None or lp_tensors is None:
                raise RuntimeError("worker Beam result transport is missing sampler tensors")
            if st_ids.shape[0] != num_logical_requests:
                raise RuntimeError(
                    "worker Beam sampled row count does not match logical requests: "
                    f"{st_ids.shape[0]} != {num_logical_requests}"
                )
            if lp_tensors.logprobs.shape[0] != num_logical_requests:
                raise RuntimeError(
                    "worker Beam logprob row count does not match logical requests: "
                    f"{lp_tensors.logprobs.shape[0]} != {num_logical_requests}"
                )
            return collapsed_req_ids_copy, collapsed_req_id_to_index, num_logical_requests
        if st_ids is not None and lp_tensors is not None and lp_tensors.logprobs is not None:
            orig_lps = lp_tensors.logprobs
            orig_lp_ids = lp_tensors.logprob_token_ids
            orig_ranks = getattr(lp_tensors, "selected_token_ranks", None)
            actual_K = orig_lps.shape[1]

            max_chained_dim = 0
            for req_id, _, W, *_ in request_row_configs:
                if req_id is not None and req_id in beam_data:
                    m_width = beam_data[req_id].get("beam_width", 1)
                    needed_dim = m_width * actual_K
                else:
                    needed_dim = W * actual_K
                max_chained_dim = max(max_chained_dim, needed_dim)

            final_st_ids, final_lps, final_lp_ids, final_ranks = _remux_logprobs_tensors(
                request_row_configs,
                max_chained_dim,
                actual_K,
                orig_lps,
                orig_lp_ids,
                orig_ranks,
                st_ids,
            )
            if not hasattr(runner, "_beam_logprobs_cache"):
                runner._beam_logprobs_cache = {}

            req_id_to_config = {r[0]: (r[2], r[3]) for r in request_row_configs if r[0] is not None}

            for idx, req_id in enumerate(collapsed_req_ids_copy):
                mdata = beam_data.get(req_id)
                if mdata and mdata.get("is_beam_decode", False):
                    logits_num, active_beams = req_id_to_config.get(req_id, (0, []))
                    total_beam_width = mdata.get("beam_width", 1)

                    if logits_num > 0:
                        # FAST-PATH: This chunk contains ALL beams directly.
                        if logits_num == total_beam_width:
                            if req_id in runner._beam_logprobs_cache:
                                del runner._beam_logprobs_cache[req_id]
                            continue

                        # SLOW-PATH: Handle split chunked prefills across steps.
                        actual_lps = final_lps[idx, : logits_num * actual_K].view(
                            logits_num, actual_K
                        )
                        actual_lp_ids = (
                            final_lp_ids[idx, : logits_num * actual_K].view(logits_num, actual_K)
                            if final_lp_ids is not None
                            else None
                        )
                        actual_ranks = (
                            final_ranks[idx, :logits_num] if final_ranks is not None else None
                        )

                        if req_id not in runner._beam_logprobs_cache:
                            runner._beam_logprobs_cache[req_id] = {
                                "lps": torch.full(
                                    (total_beam_width, actual_K),
                                    float("-inf"),
                                    device=orig_lps.device,
                                    dtype=orig_lps.dtype,
                                ),
                                "lp_ids": (
                                    torch.zeros(
                                        (total_beam_width, actual_K),
                                        dtype=(
                                            orig_lp_ids.tensor.dtype
                                            if hasattr(orig_lp_ids, "tensor")
                                            else torch.long
                                        ),
                                        device=orig_lps.device,
                                    )
                                    if orig_lp_ids is not None
                                    else None
                                ),
                                "ranks": (
                                    torch.zeros(
                                        total_beam_width, dtype=torch.long, device=orig_lps.device
                                    )
                                    if orig_ranks is not None
                                    else None
                                ),
                                "completed_beams": set(),
                            }

                        cache_entry = runner._beam_logprobs_cache[req_id]

                        for local_row_idx, physical_beam_id in enumerate(active_beams):
                            if physical_beam_id not in cache_entry["completed_beams"]:
                                cache_entry["lps"][physical_beam_id] = actual_lps[local_row_idx]
                                if actual_lp_ids is not None:
                                    cache_entry["lp_ids"][physical_beam_id] = actual_lp_ids[
                                        local_row_idx
                                    ]
                                if actual_ranks is not None:
                                    cache_entry["ranks"][physical_beam_id] = actual_ranks[
                                        local_row_idx
                                    ]
                                cache_entry["completed_beams"].add(physical_beam_id)

                        if len(cache_entry["completed_beams"]) >= total_beam_width:
                            final_lps[idx].copy_(cache_entry["lps"].view(-1))
                            if final_lp_ids is not None:
                                final_lp_ids[idx].copy_(cache_entry["lp_ids"].view(-1))
                            if final_ranks is not None:
                                final_ranks[idx, :total_beam_width].copy_(cache_entry["ranks"])

                            del runner._beam_logprobs_cache[req_id]

            try:
                st_ids.resize_(final_st_ids.shape).copy_(final_st_ids)
                lp_tensors.logprobs.resize_(final_lps.shape).copy_(final_lps)

                if lp_tensors.logprob_token_ids is not None and final_lp_ids is not None:
                    lp_tensors.logprob_token_ids.resize_(final_lp_ids.shape).copy_(final_lp_ids)
                if orig_ranks is not None and final_ranks is not None:
                    clean_ranks_slice = final_ranks[:num_logical_requests, 0].contiguous()
                    lp_tensors.selected_token_ranks.resize_(clean_ranks_slice.shape).copy_(
                        clean_ranks_slice
                    )
            except Exception as e:
                logger.critical("Inplace copy failed during beam collapse: %s", e)
                raise RuntimeError("Inplace copy failed during beam collapse") from e

    return collapsed_req_ids_copy, collapsed_req_id_to_index, num_logical_requests


def _reduce_sampler_output_for_worker_decision(runner: Any, sampler_output: Any) -> bool:
    """Reduce a CUDA worker-decision sampler payload from O(W^2) to O(W).

    vLLM's synchronous bookkeeping copies ``sampler_output.logprobs_tensors``
    to the host before :func:`_beam_collapse_bookkeeping` runs.  Once the
    worker has selected the W survivors, copying the original W-by-W
    candidate matrix is redundant: token/parent/score winners are carried by
    ``beam_search_result``.  Keep one ordinary selected token/score row only
    so upstream can still materialize a valid RequestOutput, and mark it so
    the later collapse does not recreate the large matrix.

    The parent indices are deliberately *not* packed into token ids.  This is
    the normal sampler representation plus the canonical beam_search_result
    channel, not the removed FlatLogprobs packed-wire protocol.
    """
    decision = getattr(runner, "_beam_decision_tensors", None)
    if not isinstance(decision, dict) or sampler_output is None:
        return False
    lp_tensors = getattr(sampler_output, "logprobs_tensors", None)
    if lp_tensors is None:
        return False
    token_ids = decision.get("out_token_ids")
    scores = decision.get("out_log_probs")
    if not torch.is_tensor(token_ids) or not torch.is_tensor(scores):
        return False
    if token_ids.ndim != 2 or scores.shape != token_ids.shape or token_ids.shape[0] < 1:
        return False

    ranks = getattr(lp_tensors, "selected_token_ranks", None)
    rank_dtype = ranks.dtype if torch.is_tensor(ranks) else torch.int64
    workspace = getattr(runner, "_beam_postprocess_workspaces", None)
    selected_ids = token_ids
    selected_scores = scores
    selected_sampled = None
    selected_ranks = None
    host_ids = decision.get("_host_token_ids")
    host_scores = decision.get("_host_log_probs")
    host_ranks = decision.get("_host_selected_token_ranks")
    host_ready = bool(
        getattr(runner, "_beam_result_host_ready", False)
        and torch.is_tensor(host_ids)
        and torch.is_tensor(host_scores)
        and torch.is_tensor(host_ranks)
        and host_ids.device.type == "cpu"
        and host_scores.device.type == "cpu"
        and host_ranks.device.type == "cpu"
    )
    if host_ready:
        selected_ids = host_ids
        selected_scores = host_scores
        selected_sampled = host_ids[:, :1]
        selected_ranks = host_ranks
    elif bool(getattr(runner, "use_async_scheduling", False)):
        # AsyncGPUModelRunnerOutput starts its D2H copies only after this
        # function returns.  The decision tensors may be views into a
        # runner-owned workspace that the next execute_model can overwrite in
        # the meantime, so publish private device storage for the async
        # wrapper.  Its copy stream waits for these clones on the producer
        # stream, and retaining the tensors keeps their storage alive until
        # the copy has completed.
        selected_ids = token_ids.clone()
        selected_scores = scores.clone()
        selected_sampled = selected_ids[:, :1]
        selected_ranks = torch.zeros(
            selected_ids.shape[0],
            dtype=rank_dtype,
            device=selected_ids.device,
        )
    elif workspace is not None:
        from vllm_gr.v1.beam.beam_decision import get_beam_postprocess_workspace

        fixed = get_beam_postprocess_workspace(runner, int(token_ids.shape[1]))
        if fixed is not None and token_ids.shape[0] <= fixed.max_batch:
            selected_sampled = fixed.sampled_token_ids[: token_ids.shape[0]]
            selected_sampled.copy_(token_ids[:, :1])
            if fixed.selected_token_ranks.dtype == rank_dtype:
                selected_ranks = fixed.selected_token_ranks[: token_ids.shape[0]]
    if selected_sampled is None:
        selected_ids = token_ids.contiguous()
        selected_scores = scores.contiguous()
        selected_sampled = selected_ids[:, :1].contiguous()
    if selected_ranks is None:
        selected_ranks = torch.zeros(
            selected_ids.shape[0],
            dtype=rank_dtype,
            device=selected_ids.device,
        )
    sampler_output.sampled_token_ids = selected_sampled
    sampler_output.logprobs_tensors = lp_tensors._replace(
        logprob_token_ids=selected_ids,
        logprobs=selected_scores,
        selected_token_ranks=selected_ranks,
    )
    sampler_output._gr_beam_result_transport = True
    return True


def _attach_beam_search_result(runner: Any, model_output: Any) -> None:
    """Attach the worker beam decision onto a ModelRunnerOutput, if present.

    ``ModelRunnerOutput`` is a mutable dataclass (no ``__slots__``), so the
    decision dict is attached as a dynamic ``beam_search_result`` attribute.
    Under async scheduling the runner returns an ``AsyncModelRunnerOutput``
    wrapper; the scheduler eventually sees ``get_output()`` which returns the
    inner ``ModelRunnerOutput``, so the attribute must land on that inner
    object.  This matches the asynchronous scheduling ``ModelRunnerOutput.beam_search_result``
    save-flow used by the scheduler ``kv_transfer_params`` attachment.

    Async ``execute_model`` returns ``None`` / an ``ExecuteModelState`` (the
    sampling + decision happen later in ``sample_tokens``), which are skipped
    here; the ``sample_tokens`` hook attaches the decision to the final output.
    """
    decision = getattr(runner, "_beam_decision_result", None)
    if decision is None:
        return
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

    if not isinstance(model_output, (ModelRunnerOutput, AsyncModelRunnerOutput)):
        # Async execute_model: sampling (and the decision) has not run yet; the
        # sample_tokens hook attaches the decision to the real output.
        return

    target = model_output
    if isinstance(target, AsyncModelRunnerOutput):
        target = getattr(target, "model_runner_output", None) or getattr(
            target, "_model_runner_output", None
        )
        if target is None:
            logger.warning(
                "vllm-gr: cannot attach beam_search_result to AsyncModelRunnerOutput "
                "(no inner ModelRunnerOutput); decision dropped"
            )
            return
    target.beam_search_result = decision


def _sampling_params_is_structured(params: Any) -> bool:
    """True when the row carries guided / structured-output constraints."""
    if params is None:
        return False
    if getattr(params, "structured_outputs", None) is not None:
        return True
    return getattr(params, "guided_decoding", None) is not None


def _batch_has_non_beam_or_structured_row(scheduler_output: Any) -> bool:
    """True when M2 must stay off: ordinary, mixed, or guided rows present."""
    if scheduler_output is None:
        return False
    if getattr(scheduler_output, "pending_structured_output_tokens", False):
        return True
    beam_data = getattr(scheduler_output, "beam_data", None) or {}
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    scheduled_ids = set(scheduled.keys()) if scheduled else set(beam_data.keys())
    if not scheduled_ids:
        # Ordinary empty/unknown batch: do not inline.
        return True
    for req_id in scheduled_ids:
        meta = beam_data.get(req_id) or {}
        if not (meta.get("is_beam_decode") or meta.get("is_constraint_prefill")):
            return True
        if meta.get("use_structured_output"):
            return True
    for req in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
        if _sampling_params_is_structured(getattr(req, "sampling_params", None)):
            return True
    return False


def _should_inline_sample_tokens(runner: Any, scheduler_output: Any = None) -> bool:
    """Whether ``execute_model`` should finish sampling in-process (M2).

    The upstream short-circuit (``vllm/v1/engine/core.py``: ``future.result()
    is None`` → ``model_executor.sample_tokens``) only exists on the plain
    ``step()`` path.  It therefore requires:

    - the GRConfig opt-ins: ``beam_engine_driven`` (M2 rides the engine-
      driven pipeline) and ``beam_worker_decision`` (its validation already
      requires ``beam_engine_driven → beam_worker_decision``);
    - ``use_async_scheduling`` off: with async scheduling the engine runs
      ``step_with_batch_queue``, which issues the second ``sample_tokens``
      RPC unconditionally (the returned Future is fed through the batch
      queue); an inlined output would collide with that call;
    - pipeline-parallel off: with PP>1 the worker contract (IntermediateTensors
      handoff / per-rank responses) is undefined for an inlined output, and
      the engine also switches to ``step_with_batch_queue``.
    - per-request metadata: every scheduled row must be a beam decode or
      constraint prefill.  Guided / structured-output rows need the engine
      grammar bitmask computed *after* execute_model; inlining
      ``sample_tokens(None)`` would sample them unmasked.  Mixed ordinary
      batches keep the two-RPC path.

    When ``scheduler_output`` is omitted the config/topology gates are
    returned alone (unit tests of those flags).
    """
    gr_cfg = _get_gr_config(runner)
    if not (gr_cfg.beam_engine_driven and gr_cfg.beam_worker_decision):
        return False
    if getattr(runner, "use_async_scheduling", False):
        return False
    parallel_config = getattr(runner, "parallel_config", None)
    if parallel_config is not None and getattr(parallel_config, "pipeline_parallel_size", 1) > 1:
        return False
    return not (
        scheduler_output is not None and _batch_has_non_beam_or_structured_row(scheduler_output)
    )


def _inline_sample_tokens(runner: Any) -> Any:
    """Run the runner's ``sample_tokens`` in-process and return its output.

    Beam decode steps never carry grammar output (constraints run worker-side
    via the Constraint Tables hook), so ``grammar_output=None`` matches the
    engine's own second-RPC call.  ``sample_tokens`` consumes and clears the
    runner's ephemeral ``execute_model_state`` itself; the defensive reset
    below only guards runners whose ``sample_tokens`` returns early (e.g. a
    pooling / non-final-PP edge) without consuming the state.
    """
    try:
        out = runner.sample_tokens(None)
    finally:
        runner.execute_model_state = None
    return out


def _get_gr_config(runner) -> Any:
    """Return the worker's GRConfig, cached on the runner (frozen dataclass).

    Configuration flows exclusively through the engine config chain
    (CLI flags / additional_config -> GRConfig -> VllmConfig.additional_config);
    no environment variables are consulted at runtime.
    """
    cached = getattr(runner, "_vllm_gr_config", None)
    if cached is not None:
        return cached
    from vllm_gr.config import GRConfig

    cfg = GRConfig.from_vllm_config(getattr(runner, "vllm_config", None))
    runner._vllm_gr_config = cfg
    return cfg


def _beam_full_graph_ready(runner: Any, num_tokens: int) -> bool:
    """Query the GPU Beam component; retain the device-neutral NPU key contract."""
    graph = getattr(runner, "_beam_decode_graph", None)
    if graph is not None:
        return bool(graph.ready and num_tokens == graph.width)
    beam_keys = getattr(runner, "_beam_graph_keys", None)
    return beam_keys is not None and num_tokens in beam_keys


def _beam_graph_enabled_for_run(runner) -> bool:
    """Beam graphs are only used when the config enables them (GRConfig)."""
    return _get_gr_config(runner).beam_graph_enabled


def get_beam_graph_runtime_stats(worker: Any) -> dict[str, Any]:
    """Return capture/dispatch state through ``collective_rpc``.

    vLLM utility RPC callables receive the Worker, while Beam graph state is
    owned by its ModelRunner.  Keeping this probe in an importable module makes
    it safe under both ``fork`` and ``spawn`` and lets profiling fail loudly
    when startup capture succeeded but the measured request ran eager.
    """

    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("worker has no model_runner for Beam graph diagnostics")
    config = _get_gr_config(runner)
    return {
        "runner": type(runner).__name__,
        "beam_graph_enabled": bool(config.beam_graph_enabled),
        "beam_max_width": config.beam_max_width,
        "captured_widths": sorted(int(width) for width in getattr(runner, "_beam_graph_keys", ())),
        "full_dispatches": int(getattr(runner, "_beam_graph_full_dispatch_count", 0)),
    }


_GC_FROZEN_AFTER_CAPTURE_MARKER = "_vllm_gr_gc_frozen_after_capture"
# Process-level guard for the one-time post-warmup freeze.  GC state is
# per-process (not per-runner), so a module-level flag — rather than a runner
# attribute — correctly deduplicates across the engine/worker layers and
# across in-proc vs. spawned EngineCore deployments.
#
# Two states are distinguished so that a freeze is only ever *acknowledged*
# after it actually completed:
# - _GC_WARMUP_FREEZE_DONE: the one-shot has been claimed (scheduled by the
#   automatic daemon or executed/attempted by the RPC).  Blocks re-scheduling.
# - _GC_WARMUP_FREEZE_COMPLETE: a collect+freeze actually finished
#   successfully (via the daemon or the RPC).  Gates dedupe acknowledgement.
# A claim whose attempt failed rolls DONE back so a later path can retry.
_GC_WARMUP_FREEZE_DONE = False
_GC_WARMUP_FREEZE_COMPLETE = False
_GC_WARMUP_FREEZE_LOCK = threading.Lock()

# Bounded wait for a quiescent boundary (no execute_model in flight) before
# the automatic one-shot collection runs.  An engine that stays busy past
# this deadline gets the collection anyway (once), degrading to the
# pre-barrier placement rather than never freezing the heap.
_QUIESCENT_WAIT_TIMEOUT_S = 30.0

# In-flight (patched) execute_model calls in this worker process.
# gc.collect() holds the GIL for the whole collection, so the automatic
# warmup freeze — scheduled from the teardown branch that runs at the START
# of the next execute_model — must not run its collection there: it would
# stall that request's critical path for the whole pause.  The freeze daemon
# waits for this depth to reach zero and starts the collection at that
# quiescent boundary instead.  A step admitted while the collection is
# already running can still see the tail of that one-shot pause.
_WORKER_EXECUTE_DEPTH = 0
_WORKER_EXECUTE_COND = threading.Condition()


def _enter_worker_execute() -> None:
    global _WORKER_EXECUTE_DEPTH
    with _WORKER_EXECUTE_COND:
        _WORKER_EXECUTE_DEPTH += 1


def _exit_worker_execute() -> None:
    global _WORKER_EXECUTE_DEPTH
    with _WORKER_EXECUTE_COND:
        _WORKER_EXECUTE_DEPTH -= 1
        if _WORKER_EXECUTE_DEPTH <= 0:
            _WORKER_EXECUTE_COND.notify_all()


def _freeze_serving_gc(*, raise_thresholds: bool) -> list[int]:
    """Collect all generations, freeze the live heap into the permanent
    generation, and (optionally) raise the gen-1/gen-2 thresholds.

    Process-level operation shared by the capture-time freeze, the warmup
    RPC freeze, and the automatic post-first-request freeze.  ``gc.freeze()``
    moves every currently-tracked live object into the permanent generation
    so that gen-0/gen-1 collections no longer scan the persistent beam
    infrastructure (beam-session pools, captured graph state, scheduler beam
    bookkeeping).  When ``raise_thresholds`` is set, gen-1/gen-2 full scans
    are additionally disabled during serving (they measured ``collected=0``,
    i.e. pure 17-36 ms pauses); gen-0 keeps its default threshold so
    per-request temporaries are still reclaimed cheaply.
    """
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()
    if raise_thresholds:
        g0, _g1, _g2 = gc.get_threshold()
        gc.set_threshold(g0, 100_000, 100_000)
    return list(gc.get_threshold())


def schedule_warmup_gc_freeze() -> bool:
    """Schedule the one-time post-warmup GC freeze at a quiescent boundary.

    Called from the beam-session teardown path after the first beam-search
    request finishes in the EngineCore/worker process.  The teardown signal
    is consumed at the START of the next ``execute_model``, so a plain
    background thread would begin its collection while that request is in
    flight — and ``gc.collect()`` holds the GIL for the whole collection,
    stalling the request's critical path.  Instead the daemon thread first
    waits (bounded by ``_QUIESCENT_WAIT_TIMEOUT_S``) until no
    ``execute_model`` is running, then runs the ``collect + freeze`` (a few
    tens of ms) at that quiescent boundary — never inside a model step.
    Idempotent and thread-safe: only the first call schedules the freeze;
    subsequent calls are no-ops.  If the explicit RPC freeze has already
    completed the heap by then, the daemon collects nothing.

    This is the production equivalent of the benchmark's
    ``collective_rpc("gr_freeze_gc_after_warmup")``: the cross-request beam
    infrastructure is built lazily by the first beam request, so freezing
    once it has been torn down captures the whole persistent set.
    """
    global _GC_WARMUP_FREEZE_DONE
    with _GC_WARMUP_FREEZE_LOCK:
        if _GC_WARMUP_FREEZE_DONE:
            return False
        _GC_WARMUP_FREEZE_DONE = True

    def _run() -> None:
        global _GC_WARMUP_FREEZE_DONE, _GC_WARMUP_FREEZE_COMPLETE
        try:
            with _WORKER_EXECUTE_COND:
                deadline = time.monotonic() + _QUIESCENT_WAIT_TIMEOUT_S
                while _WORKER_EXECUTE_DEPTH > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.info(
                            "vllm-gr: engine stayed busy for %.0fs; running "
                            "the one-shot GC freeze without a quiescent "
                            "boundary",
                            _QUIESCENT_WAIT_TIMEOUT_S,
                        )
                        break
                    _WORKER_EXECUTE_COND.wait(remaining)
            with _GC_WARMUP_FREEZE_LOCK:
                if _GC_WARMUP_FREEZE_COMPLETE:
                    # The explicit warmup RPC already froze this heap.
                    return
            thresholds = _freeze_serving_gc(raise_thresholds=True)
            with _GC_WARMUP_FREEZE_LOCK:
                _GC_WARMUP_FREEZE_COMPLETE = True
            logger.info(
                "vllm-gr: auto-froze GC heap at a quiescent boundary after "
                "the first beam request (gen thresholds now %s)",
                thresholds,
            )
        except Exception:
            # Roll back the guard so a later teardown/RPC can retry.
            with _GC_WARMUP_FREEZE_LOCK:
                _GC_WARMUP_FREEZE_DONE = False
            # Never let a GC hint failure surface into the serving process.
            logger.exception(
                "vllm-gr: automatic warmup GC freeze failed; continuing with default GC"
            )

    threading.Thread(
        target=_run,
        name="vllm-gr-gc-freeze",
        daemon=True,
    ).start()
    return True


def freeze_gc_after_graph_capture(runner: Any) -> None:
    """Freeze the serving-time heap once CUDA-graph capture has finished.

    vLLM's startup ``freeze_gc_heap()`` runs at the end of ``EngineCore.__init__``
    — *before* KV-cache allocation and CUDA-graph capture.  The long-lived
    objects created afterwards (KV caches, captured native/beam/prefill graph
    state, beam-session infrastructure) therefore stay in the tracked young/
    middle generations.  During serving, gen-1 collections scan them all every
    few requests for cycles that never exist (measured 29-36 ms pauses with
    ``collected=0``, landing on a fixed 4-request cadence and spiking that
    request's latency).  Running a full collection and ``gc.freeze()`` here
    moves this post-startup heap into the permanent generation so gen-0/gen-1
    collections during serving no longer scan it.

    Idempotent: only the first invocation after capture runs the collection.
    Runs in the EngineCore/worker process (``capture_model`` executes there).
    """
    if getattr(runner, _GC_FROZEN_AFTER_CAPTURE_MARKER, False):
        return
    try:
        _freeze_serving_gc(raise_thresholds=False)
        setattr(runner, _GC_FROZEN_AFTER_CAPTURE_MARKER, True)
        logger.info(
            "vllm-gr: froze GC heap after CUDA-graph capture "
            "(gen thresholds now scan serving allocations only)"
        )
    except Exception:
        logger.exception(
            "vllm-gr: freeze_gc_after_graph_capture failed; continuing with default GC"
        )


def freeze_gc_after_warmup(worker: Any) -> dict[str, Any]:
    """Collect + freeze the heap in the EngineCore process after warmup.

    Reachable via ``collective_rpc("gr_freeze_gc_after_warmup")``.  The
    cross-request beam infrastructure (beam-session pools, beam graph runner
    state, scheduler beam bookkeeping) is built lazily by the first
    beam-search request — i.e. *after* the startup/capture freeze — and then
    stays live for the whole process.  Without a second freeze here, every
    gen-1 collection during serving rescans that persistent set (measured
    29-36 ms pauses with ``collected=0`` on a fixed 4-request cadence).

    This RPC is a *deterministic warmup barrier*: it reports ``frozen=True``
    only after a freeze has actually completed — either one that already
    finished (``deduped: True``, e.g. the automatic daemon ran first) or the
    synchronous collect+freeze performed here.  It never acknowledges a
    merely-scheduled (still pending) collection, so a benchmark using it as
    a barrier cannot start timing during the GC pause, and a collector that
    subsequently fails cannot invalidate an acknowledgement.  If the
    automatic daemon happens to be collecting concurrently, the two GIL-
    serialized idempotent freezes at worst double the one-shot pause.

    ``gc.freeze()`` is safe to call repeatedly.
    """
    runner = getattr(worker, "model_runner", None)
    global _GC_WARMUP_FREEZE_DONE, _GC_WARMUP_FREEZE_COMPLETE
    try:
        with _GC_WARMUP_FREEZE_LOCK:
            if _GC_WARMUP_FREEZE_COMPLETE:
                # A freeze (automatic daemon or an earlier RPC) already
                # completed; nothing left to do.
                return {"frozen": True, "thresholds": list(gc.get_threshold()), "deduped": True}
            # Claim the one-shot even if the automatic daemon has not fired
            # yet, so it does not schedule a second collection afterwards.
            _GC_WARMUP_FREEZE_DONE = True
        # After freeze, the persistent heap (weights, KV caches, captured
        # graphs, beam-session pools) lives in the permanent generation and is
        # never scanned.  Anything still promoted to gen-1/gen-2 during serving
        # is measured to be collected=0 (no cyclic garbage in the beam path),
        # so gen-1/gen-2 collections are pure 17-36ms pauses that land mid-
        # request on a fixed cadence.  Keep gen-0 at its default (it cheaply
        # reclaims per-request temporaries) but push the gen-1/gen-2 thresholds
        # high enough that full-heap scans do not fire during serving; they can
        # still run eventually / at shutdown if real cross-generation garbage
        # ever accumulates.
        thresholds = _freeze_serving_gc(raise_thresholds=True)
        with _GC_WARMUP_FREEZE_LOCK:
            _GC_WARMUP_FREEZE_COMPLETE = True
        g0 = thresholds[0]
        if runner is not None:
            setattr(runner, _GC_FROZEN_AFTER_CAPTURE_MARKER, True)
        logger.info(
            "vllm-gr: froze GC heap after warmup and raised gen-1/gen-2 "
            "thresholds to 100000 (gen-0 stays at %d)",
            g0,
        )
        return {"frozen": True, "thresholds": thresholds}
    except Exception as e:  # never fail the request over a GC hint
        # Roll back the one-shot claim so the automatic path (or a later RPC)
        # can retry instead of leaving the heap permanently unfrozen.
        with _GC_WARMUP_FREEZE_LOCK:
            _GC_WARMUP_FREEZE_DONE = False
        logger.exception("vllm-gr: freeze_gc_after_warmup failed")
        return {"frozen": False, "error": str(e)}


def _create_beam_context(runner, gr_cfg, max_batch: int = 1):
    """Create a BeamAttentionContext with sizing derived from model + GRConfig.

    Shared by the dummy-capture paths (GPU/NPU ``capture_beam_decode_graph``)
    and the runtime lazy-creation path (``_get_or_create_beam_session``) so
    that the pool dimensions, dtype, and device are always derived from the
    same sources.  A single source of truth prevents silent size mismatches
    that would invalidate captured graphs if the runtime path rebuilt the
    context with different parameters.

    Args:
        runner: The ModelRunner (must have ``model_config``,
            ``parallel_config``, ``device``).
        gr_cfg: The GRConfig (provides ``beam_max_width``,
            ``beam_max_decode_steps``).  ``beam_max_width`` must not be
            None — callers must guard before reaching this function.
        max_batch: Maximum concurrent beam search sessions.

    Returns:
        A new ``BeamAttentionContext``.

    Raises:
        TypeError: If ``gr_cfg.beam_max_width`` is None.
    """
    if gr_cfg.beam_max_width is None:
        raise TypeError(
            "beam_max_width must be set before creating a BeamAttentionContext "
            "(beam KV pool is disabled when beam_max_width is None)"
        )
    from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext

    model_config = runner.model_config
    parallel_config = runner.parallel_config

    return BeamAttentionContext(
        max_batch=max_batch,
        max_beams=int(gr_cfg.beam_max_width),
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        max_decode_steps=int(gr_cfg.beam_max_decode_steps),
        head_dim=model_config.get_head_size(),
        num_layers=model_config.get_num_layers(parallel_config),
        device=runner.device,
        dtype=model_config.dtype,
    )


def _seed_beam_decision_from_metadata(session: Any, mdata: dict[str, Any]) -> None:
    """Synchronize decision state with the authoritative selected parents."""
    parent_scores = mdata.get("parent_cumulative_logprobs")
    if parent_scores is None:
        # A frontend fallback can advance or reorder beams without a worker
        # decision. Never resume from old buffers when their state is unknown.
        session.context.decision.worker_decode_steps = 0
        return
    parent_sequences = mdata.get("beam_suffix")
    if not parent_sequences:
        raise RuntimeError(
            "worker beam decision received parent scores without parent beam suffixes"
        )
    from vllm_gr.v1.beam.beam_decision import init_beam_decision_from_parents

    init_beam_decision_from_parents(session, parent_sequences, parent_scores)


def _get_or_create_beam_session(runner, beam_data: dict, scheduler_output=None) -> Any | None:
    """Get or create per-request BeamSearchSession for incremental decode.

    Manages per-request sessions via runner._beam_sessions dict, keyed by
    session_id (== req_id).  Each session references a shared
    BeamAttentionContext (runner._beam_context) which owns the pools and
    the BeamSessionRegistry.

    Session lifecycle:
    - Creation: on the first beam decode step (decode_step=0) for a req_id.
    - Step progression: step_complete (select_unshared_kv + advance) on
      subsequent steps.
    - Teardown: the authoritative signal is
      ``SchedulerOutput.beam_finished_session_ids``, populated by
      ``patched_schedule`` from
      ``Scheduler._beam_finished_session_ids`` which is written by
      ``_handle_beam_request_step_update`` when it receives a
      ``BeamRequestStepUpdate(beam_width=0)`` — the beam search loop's
      explicit "generation finished" message.  This is distinct from
      ``finished_req_ids`` which fires on every decode step.
    - Fallback: if allocate fails and the registry is full, stale sessions
      (whose req_ids are no longer in beam_data) are released as a safety
      net for unexpected abort/error paths.

    Concurrency: the shared context is created with ``max_batch=1`` — a
    single active beam session by design (pool slots, suffix buffers and the
    captured FULL graphs all bake one session).  Batches carrying more than
    one incremental beam-decode request are rejected with an explicit
    RuntimeError below.
    """
    runner._beam_sessions = getattr(runner, "_beam_sessions", {})

    # 1. Clean up finished sessions (beam_width=0 teardown signal)
    finished_ids = getattr(scheduler_output, "beam_finished_session_ids", [])
    tore_down_session = False
    for sid in finished_ids:
        session = runner._beam_sessions.pop(sid, None)
        if session is not None:
            session.end_session()
            tore_down_session = True
    if tore_down_session:
        # A full beam-search request just finished in this (EngineCore/worker)
        # process.  Schedule the one-time post-warmup GC freeze on a background
        # daemon thread: the lazily-built, process-lifetime beam infrastructure
        # is now fully populated, and freezing it outside any request's
        # critical path removes the fixed-cadence gen-1 pause spikes without
        # any client/script coordination.  Idempotent and cross-deployment
        # (in-proc and spawned EngineCore both tear sessions down here).
        schedule_warmup_gc_freeze()
    # 2. Check if we have any beam decode requests with decode_step
    incremental_ids = [
        req_id
        for req_id, mdata in beam_data.items()
        if mdata.get("is_beam_decode", False) and "decode_step" in mdata
    ]
    if not incremental_ids:
        return None

    # The shared beam context serves a SINGLE active session: the pools, the
    # suffix buffers, the xllm block-table buffer and the captured FULL
    # graphs are all sized/created with max_batch=1 (see step 3 below and the
    # capture paths).  A second concurrent beam-decode request would exhaust
    # the slot registry mid-batch, and the stale-session retry inside the
    # allocate path below cannot free a slot because BOTH ids are active —
    # the batch would die with the generic "No free beam session slots"
    # error.  Reject the unsupported concurrency explicitly instead.
    if len(incremental_ids) > 1:
        raise RuntimeError(
            "vllm-gr concurrent incremental beam decode requests are not "
            "supported: the beam KV pool holds a single active session "
            f"(max_batch=1), but this batch carries {len(incremental_ids)} "
            f"({incremental_ids}). Serialize beam search requests or wait "
            "for the active one to finish."
        )

    gr_cfg = _get_gr_config(runner)

    # Beam decode requests are scheduled incrementally (the scheduler freezes
    # the paged cache at the prefix; decode KV belongs in the beam pool).  With
    # no pool the attention would fall back to the standard paged path, which
    # cannot represent per-beam divergent suffix KV — silently wrong logprobs.
    # Fail loudly instead, mirroring the over-capacity error below.
    if gr_cfg.beam_max_width is None:
        raise RuntimeError(
            "vllm-gr beam decode requests require a persistent beam KV pool, "
            "but beam_max_width is not configured. Set --beam-max-width (or "
            "vllm_gr_beam_max_width in additional_config) to the maximum "
            "served beam width."
        )

    # 3. Ensure worker-level context exists (lazy creation).
    #    Uses the shared _create_beam_context so pool sizing (max_beams,
    #    max_decode_steps, dtype, device) is derived from the same sources
    #    as the dummy-capture paths — a single source of truth prevents
    #    silent size mismatches that would invalidate captured graphs.
    if not hasattr(runner, "_beam_context"):
        runner._beam_context = _create_beam_context(runner, gr_cfg, max_batch=1)

    # 4. Per-request session management
    active_session = None

    # The context above is sized once and is never resized (a regrow would
    # move every buffer's data_ptr and silently invalidate any captured CUDA
    # graph), so requests beyond its capacity are a hard configuration error.
    # Without this check the Triton pool kernels write out of bounds silently
    # (the eager Python path's IndexError safety net does not exist on the
    # device-side path).
    context = runner._beam_context
    for req_id, mdata in beam_data.items():
        if not mdata.get("is_beam_decode") or "decode_step" not in mdata:
            continue

        req_width = int(mdata["beam_width"])
        req_step = int(mdata.get("decode_step", 0))
        if req_width > context.max_beams or req_step >= context.max_decode_steps:
            raise ValueError(
                f"vllm-gr beam request {req_id!r} exceeds the persistent beam "
                f"KV pool capacity (pool: max_beams={context.max_beams}, "
                f"max_decode_steps={context.max_decode_steps}; request: "
                f"beam_width={req_width}, decode_step={req_step}). Raise "
                f"--beam-max-width / --beam-max-decode-steps to cover the "
                f"served request population."
            )

        session = runner._beam_sessions.get(req_id)
        mdata_step = mdata.get("decode_step", 0)

        # Terminal-step detection.  ``max_decode_steps`` on the
        # metadata is the frontend LOOP length (``max_tokens -
        # pre_calc``): iteration 0 is the prefill, so the request
        # runs ``loop_len - 1`` decode steps and the 0-indexed,
        # decode-only ``decode_step`` tops out at ``loop_len - 2``.
        # Comparing against ``loop_len - 1`` fired the terminal flag
        # one step late: the 3-iteration constrained FSM never saw a
        # terminal step, engine-driven chaining produced a 3rd decode, and
        # its constraint_step=3 was rejected by
        # build_beam_request_metadata (decode steps are 1/2 only).
        # With Prefill counted as the first generation step, the final Decode
        # is ``decode_step >= loop_len - 2``.
        # When the host omitted the loop length, fall back to the
        # pool bound (``gr_cfg.beam_max_decode_steps``), which counts
        # DECODE-ONLY steps — never to beam_decode_steps, which is
        # "steps so far", not the planned length.
        # Computed for EVERY decode step, the first included: a loop
        # of 2 (prefill + 1 decode) runs a single, terminal decode,
        # and ``begin_session`` resets ``is_last_step`` — so the flag
        # must be (re)applied after every session (re)initialisation,
        # not only on subsequent steps.
        loop_len = mdata.get("max_decode_steps", 0)
        planned_decode_steps = loop_len - 1 if loop_len > 0 else gr_cfg.beam_max_decode_steps
        is_last_step = mdata_step >= planned_decode_steps - 1

        if session is not None:
            # Existing session: step_complete or reset
            if mdata_step == 0:
                # First decode step of a request: full reset.
                # There is no previous beam decision yet, so select_unshared_kv
                # must not run, and decode KV must be written to pool slice 0.
                # (parent_beam_indices is non-None even on the first step, so it
                # cannot be used to distinguish the first step — decode_step can.)
                session.begin_session(mdata["beam_width"], mdata["prefix_len"])
                _seed_beam_decision_from_metadata(session, mdata)
                session._state.is_last_step = is_last_step
            else:
                # Subsequent decode steps: apply the previous step's beam decision.
                # step_complete (select_unshared_kv + advance_step) must run BEFORE
                # the new KV is written, using the current decode_step to select
                # the KV range to reorder.
                parent_beam_ids = mdata.get("parent_beam_ids")
                state = session._state
                state.is_last_step = is_last_step

                # Worker-side same-step reorder record: the worker beam
                # decision already applied ``select_unshared_kv`` inside the
                # previous execute_model and
                # advanced decode_step there.  Consume the record —
                # re-applying the same parent mapping is NOT idempotent
                # (p . p != p), so an exact match with the frontend's
                # parent_beam_ids means the KV reorder is already done.
                recorded = state.kv_reorder
                if recorded is not None:
                    state.kv_reorder = None
                    if parent_beam_ids is not None and list(parent_beam_ids) != recorded:
                        # Frontend mapping differs from what the worker
                        # applied (e.g. EOS re-ordering).  The pool currently
                        # holds pre-decision row `recorded[i]` at row i; the
                        # frontend wants pre-decision row parent[j] at row j.
                        # Composite via the cascade kernel (gather semantics
                        # tolerate mappings the fused kernel cannot express).
                        comp = []
                        for p in parent_beam_ids:
                            try:
                                comp.append(recorded.index(p))
                            except ValueError:
                                logger.warning(
                                    "vllm-gr: frontend parent %s not in worker "
                                    "reorder record %r; keeping identity",
                                    p,
                                    recorded,
                                )
                                comp.append(p)
                        from vllm_gr.v1.beam.reorder import reorder_unshared_kv

                        reorder_unshared_kv(
                            torch.tensor(comp, dtype=torch.int32, device=runner.device),
                            session.slot,
                            context.pools,
                            state.decode_step - 1,  # KV range written through the previous step
                        )
                    # Exact matches were already applied by the worker.
                elif parent_beam_ids is not None:
                    parent = torch.tensor(parent_beam_ids, dtype=torch.int32, device=runner.device)
                    # This mapping belongs to the previous decision and must
                    # be applied before the current, possibly terminal, step.
                    state.is_last_step = False
                    try:
                        session.step_complete(parent)
                    finally:
                        state.is_last_step = is_last_step
                else:
                    # No previous parent mapping: just advance decode_step.
                    state.decode_step += 1
                # EOS pruning and frontend selection can change both width
                # and row order. KV reconciliation above does not synchronize
                # decision scores/sequences. Restore them on every update,
                # including the first worker decision after a frontend fallback.
                # Only change the active width; keep pool addresses and KV state.
                state.beam_width = req_width
                _seed_beam_decision_from_metadata(session, mdata)
        else:
            # New session
            from vllm_gr.v1.beam.beam_search_session import BeamSearchSession

            try:
                slot = runner._beam_context.registry.allocate(
                    req_id, mdata["beam_width"], mdata["prefix_len"]
                )
            except RuntimeError:
                # Fallback: release stale sessions whose req_ids are no
                # longer in beam_data.  This handles unexpected abort/error
                # paths where the beam_width=0 cleanup signal never arrived.
                # The primary teardown path is beam_finished_session_ids
                # (set by _handle_beam_request_step_update on beam_width=0).
                active_req_ids = set(beam_data.keys())
                stale_ids = [
                    sid for sid in list(runner._beam_sessions.keys()) if sid not in active_req_ids
                ]
                for sid in stale_ids:
                    old_session = runner._beam_sessions.pop(sid)
                    old_session.end_session()
                # Retry allocation after cleanup
                slot = runner._beam_context.registry.allocate(
                    req_id, mdata["beam_width"], mdata["prefix_len"]
                )
            session = BeamSearchSession(session_id=req_id, slot=slot, context=runner._beam_context)
            session.begin_session(mdata["beam_width"], mdata["prefix_len"])
            _seed_beam_decision_from_metadata(session, mdata)
            # Set initial decode_step from scheduler
            session._state.decode_step = mdata_step
            session._state.is_last_step = is_last_step
            runner._beam_sessions[req_id] = session

        if active_session is None:
            active_session = session

    return active_session


def make_patched_execute_model(_orig_execute_model):
    """Wrap execute_model to publish beam-request context for the run.

    Shared by GPU and NPU runners.
    """
    if getattr(_orig_execute_model, _WORKER_EXECUTE_PATCH_MARKER, False):
        return _orig_execute_model

    @wraps(_orig_execute_model)
    def _patched_execute_model_step(self, scheduler_output, *args, **kwargs):
        from vllm_gr.config import BeamExecutionMode
        from vllm_gr.config.execution import require_beam_execution_mode
        from vllm_gr.v1.engine.gr_async_scheduler import (
            GR_RETIRE_OUTPUT_ATTR,
            GR_STAGE_OUTPUT_ATTR,
        )

        if getattr(scheduler_output, GR_RETIRE_OUTPUT_ATTR, None) is not None:
            from vllm_gr.v1.worker.gpu_beam_stage_runner import execute_gr_retirement

            require_beam_execution_mode(self.vllm_config, BeamExecutionMode.V1)
            return execute_gr_retirement(self, scheduler_output)

        if getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None):
            if str(self.device).split(":")[0] != "cuda":
                raise ValueError("persistent GR stage execution requires GPU V1")
            from vllm_gr.v1.worker.gpu_beam_stage_runner import get_stage_runner

            stage_runner = get_stage_runner(self, scheduler_output)
            try:
                return stage_runner.execute(scheduler_output, _orig_execute_model, *args, **kwargs)
            except Exception as error:
                # Native BatchQueue associates the execute and sample Futures.
                # Sampling must also fail, otherwise valid-looking stale logits
                # could hide a failed execute Future from that native path.
                stage_runner.execution_error = error
                raise
        if getattr(scheduler_output, "beam_data", {}):
            require_beam_execution_mode(self.vllm_config, BeamExecutionMode.LEGACY)
        gr_stage = getattr(self, "_gr_gpu_stage_runner", None)
        if gr_stage is not None and not gr_stage.closed:
            if getattr(scheduler_output, "beam_data", {}):
                raise ValueError("legacy Beam cannot use an active GR execution context")
            if gr_stage.binding.native_request_id in getattr(
                scheduler_output, "finished_req_ids", ()
            ):
                gr_stage.cancel(gr_stage.binding)
        # Clear the previous step's beam context at the START of the next
        # execute_model, not in a finally: with async scheduling the sampled
        # tokens are produced by ``sample_tokens`` AFTER execute_model returns,
        # and the worker-side beam decision needs ``_current_beam_data`` /
        # ``_current_beam_info`` (session + decision buffers) still alive there.
        # A stale previous-step context is overwritten here before the new
        # step's values are installed.
        self._current_beam_data = getattr(scheduler_output, "beam_data", {})
        self._current_beam_info = {"beam_data": self._current_beam_data, "req_ids": []}
        beam_info = self._current_beam_info

        # Worker-side beam decision output (NPU xllm path).  Reset each step so
        # a stale result from a previous run can never leak into this step's
        # ModelRunnerOutput.beam_search_result.
        self._beam_decision_result = None
        self._beam_decision_tensors = None
        self._beam_result_host_ready = False
        # CUDA Prefill prelaunch state is single-step and single-consumer.
        # Reset it before upstream input preparation may publish this run's
        # prepared descriptor.
        self._gr_prepared_cuda_prefill_constraint = None
        self._gr_prelaunched_cuda_prefill = None

        # Manage BeamSearchSession for incremental decode
        beam_session = _get_or_create_beam_session(self, self._current_beam_data, scheduler_output)
        if beam_session is not None:
            beam_info["beam_session"] = beam_session

            # Refresh device-side buffers before model forward / Graph replay.
            # These fill_() calls are on the same CUDA stream as the model
            # forward, so they are ordered before any kernel that reads them.
            context = beam_session.context  # BeamAttentionContext
            decode_step = beam_session.decode_step

            # decode_step_buf: device scalar read by the Triton
            # cache_unshared_kv / copy_suffix_kv kernels (the per-step KV
            # extent for suffix FA2 is device-side data via the metadata
            # builder's fixed-address cu_seqlens_k buffer, refreshed by
            # build() each step).
            if hasattr(context, "decode_step_buf"):
                context.decode_step_buf.fill_(decode_step)

            # decode_step_1based_buf / xllm_block_table_buf: device inputs for
            # the xllm fused kernels (NPU).  Same refresh-outside-the-graph
            # contract — a captured ACL graph reads fresh values at replay.
            # Only NPU kernels consume them; on GPU (Triton cache_unshared /
            # copy_suffix reads decode_step_buf only) skip the two redundant
            # fill_ launches to cut per-step stream setup time.
            if getattr(context, "_gr_uses_xllm_buffers", True):
                if hasattr(context, "decode_step_1based_buf"):
                    context.decode_step_1based_buf.fill_(decode_step + 1)
                if hasattr(context, "xllm_block_table_buf"):
                    # max_batch-sized; request 0 maps to the active session's
                    # pool slot (the registry is max_batch=1, so one active
                    # session).
                    context.xllm_block_table_buf.fill_(0)
                    context.xllm_block_table_buf[0] = beam_session.slot

        token = BEAM_DATA_VAR.set(beam_info)
        self._current_beam_info = beam_info
        # Published for the optional Constraint Tables ``_sample`` hook
        # (constraint_hook): it reads ``_gr_request_row_configs`` to locate the
        # constrained logits rows.  Lives until the next execute_model (which
        # overwrites it at the top); with async scheduling the constraint hook
        # and worker beam decision run in ``sample_tokens`` AFTER
        # execute_model returns, so the context must stay alive past the
        # execute_model frame.
        self._current_scheduler_output = scheduler_output
        try:
            out = _orig_execute_model(self, scheduler_output, *args, **kwargs)
            if out is None and (
                getattr(self, "_gr_prepared_cuda_prefill_constraint", None) is not None
            ):
                # The prepared descriptor exists only when the CUDA
                # constraint-table prepare path ran for THIS step, so the
                # import stays behind a real capability gate: Triton-less
                # (NPU/CPU/trimmed) workers never publish a descriptor and
                # never reach this import, whose chain pulls
                # vllm_gr.ops.cuda (triton).  Upstream has just published
                # LM-Head logits in ExecuteModelState, so enqueue the
                # complete P->D device chain before returning control to the
                # scheduler.
                from vllm_gr.v1.worker.constraint_hook import (
                    launch_prepared_cuda_prefill_constraint,
                )

                with torch.autograd.profiler.record_function(
                    "prefill_constraint_post_forward_enqueue"
                ):
                    launch_prepared_cuda_prefill_constraint(self)
            if out is None and _should_inline_sample_tokens(self, scheduler_output):
                # M2: finish sampling in-process and return the
                # ModelRunnerOutput through the execute_model channel, so the
                # engine's None short-circuit (core.py ``future.result()``)
                # skips the second collective_rpc.  Beam decode steps carry no
                # grammar output (constraints run worker-side).  Failure falls
                # back to the two-RPC path unchanged.
                try:
                    out = _inline_sample_tokens(self)
                except Exception:
                    logger.exception(
                        "vllm-gr: inlined sample_tokens failed; falling back "
                        "to the second sample_tokens RPC"
                    )
                    out = None
            _attach_beam_search_result(self, out)
            return out
        finally:
            # Deliberately keep ``_current_beam_data`` / ``_current_beam_info``
            # / ``_current_scheduler_output`` alive for the async sampling
            # phase (sample_tokens runs after execute_model returns).  They are
            # overwritten at the top of the next execute_model call.
            BEAM_DATA_VAR.reset(token)

    @wraps(_orig_execute_model)
    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        # Track in-flight model execution so the automatic post-warmup GC
        # freeze finds a quiescent boundary: the teardown signal (and hence
        # the freeze scheduling) happens at the START of this wrapper, and
        # gc.collect() holds the GIL for the whole collection — running it
        # here would stall this request's critical path.  The freeze daemon
        # waits for the depth to reach zero before collecting.
        _enter_worker_execute()
        try:
            return _patched_execute_model_step(self, scheduler_output, *args, **kwargs)
        finally:
            _exit_worker_execute()

    setattr(patched_execute_model, _WORKER_EXECUTE_PATCH_MARKER, True)
    return patched_execute_model


def is_worker_runner_patched(runner_class) -> bool:
    """Return whether all grouped-beam hooks are present on a runner class.

    The ``_determine_batch_execution_and_padding`` hook stamps the beam
    determine marker on all platforms.  On GPU it overrides the dispatch
    result to force FULL CUDA Graph for beam decode; on NPU it is a no-op
    wrapper that preserves the original result — but the marker must still
    be present for the gate-check to pass.
    """
    has_determine = hasattr(runner_class, "_determine_batch_execution_and_padding")
    return bool(
        getattr(runner_class.execute_model, _WORKER_EXECUTE_PATCH_MARKER, False)
        and getattr(runner_class._prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False)
        and getattr(
            runner_class._bookkeeping_sync,
            _WORKER_BOOKKEEPING_PATCH_MARKER,
            False,
        )
        and (
            not has_determine
            or getattr(
                runner_class._determine_batch_execution_and_padding,
                _WORKER_DETERMINE_PATCH_MARKER,
                False,
            )
        )
    )
