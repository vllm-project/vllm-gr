# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Cross-device native ModelRunner V1 adaptation.

Owns V1 input proxies, row/logprob bookkeeping and its execute wrapper.
Configuration/context, GC coordination, dispatch and installation contracts
have separate owners; native V2 does not depend on this module.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, cast

import torch
from vllm.logger import init_logger

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.worker import gr_execution, gr_gc
from vllm_gr.v1.worker import model_runner_contracts as contracts
from vllm_gr.v1.worker.gpu_stage_hooks import CudaStageHooksType

logger = init_logger(__name__)


def _get_positions_buffers(runner: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Safely extracts underlying hardware and host indexing position tensors."""
    gpu_positions = None
    cpu_positions = None
    if hasattr(runner, "positions") and runner.positions is not None:
        if isinstance(runner.positions, torch.Tensor):
            # The Ascend runner uses a plain device tensor in PCP paths;
            # accepted as-is because that path keeps its own pinned host
            # buffer, resolved below.
            gpu_positions = runner.positions
        else:
            gpu_positions = getattr(runner.positions, "gpu", None)
            cpu_positions = getattr(runner.positions, "cpu", None)
    if cpu_positions is None:
        # The historical Ascend runner keeps a dedicated pinned host buffer
        # even when ``positions`` itself is a plain device tensor.
        cpu_positions = getattr(runner, "_positions_cpu_buf", None)
    return gpu_positions, cpu_positions


def _process_beam_decode_request(
    mdata: dict[str, Any],
    seq_len: int,
    token_offset: int,
    gpu_pos: torch.Tensor | None,
    cpu_pos: torch.Tensor | None,
    new_logits_indices: list[int],
) -> tuple[int, int, int, list[int]]:
    """Prepare one new token per beam at position prefix_len + decode_step."""
    prefix_len = mdata["prefix_len"]
    beam_width = mdata["beam_width"]
    decode_step = mdata.get("decode_step")
    if type(decode_step) is not int or decode_step < 0:
        raise ValueError("Beam decode requires a nonnegative integer decode_step")
    if type(beam_width) is not int or beam_width < 1:
        raise ValueError("Beam decode requires a positive integer beam_width")
    if mdata["cache_len"] >= prefix_len and seq_len != beam_width:
        raise ValueError("Beam decode requires exactly one query token per beam")
    new_logits_indices.extend(range(token_offset, token_offset + beam_width))
    pos_id = prefix_len + decode_step
    if gpu_pos is not None:
        gpu_pos[token_offset : token_offset + beam_width] = pos_id
    if cpu_pos is not None:
        cpu_pos[token_offset : token_offset + beam_width] = pos_id
    return beam_width, beam_width, token_offset + beam_width, list(range(beam_width))


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


class BeamReqIdsProxy(list[Any]):
    def __init__(self, original_list: Any, req_id_map: Any) -> None:
        super().__init__(original_list)
        self.req_id_map = req_id_map

    def __getitem__(self, index: Any) -> Any:
        try:
            return self.req_id_map[index]
        except IndexError as e:
            logger.error("Out of bounds in BeamReqIdsProxy for index %s", index)
            raise e

    def copy(self) -> Any:
        return list(self.req_id_map)


class Mock1DArray:
    def __init__(self, original: Any, batch_idx_map: Any) -> None:
        self.original = original
        self.batch_idx_map = batch_idx_map

    def __getitem__(self, idx: Any) -> Any:
        try:
            return self.original[self.batch_idx_map[idx]]
        except IndexError as e:
            logger.error("Out of bounds in Mock1DArray getitem for index %s", idx)
            raise e

    def __setitem__(self, idx: Any, value: Any) -> None:
        try:
            self.original[self.batch_idx_map[idx]] = value
        except IndexError as e:
            logger.error("Out of bounds in Mock1DArray setitem for index %s", idx)
            raise e


class Mock2DArray:
    def __init__(self, original: Any, batch_idx_map: Any) -> None:
        self.original = original
        self.batch_idx_map = batch_idx_map

    def __setitem__(self, key: Any, value: Any) -> None:
        idx, slc = key
        try:
            self.original[self.batch_idx_map[idx], slc] = value
        except IndexError as e:
            logger.error("Out of bounds in Mock2DArray setitem for key %s", key)
            raise e

    def __getattr__(self, name: Any) -> Any:
        return getattr(self.original, name)

    def __getitem__(self, key: Any) -> Any:
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


def _first_device_of(s_meta: Any) -> Any:
    """Return the device of the first tensor sampling attribute, or None."""
    for attr in _SAMPLING_META_ATTRS:
        val = getattr(s_meta, attr, None)
        if isinstance(val, torch.Tensor) and val.numel() > 0:
            return val.device
    return None


class InputBatchProxy:
    def __init__(
        self,
        obj: Any,
        req_map: Any,
        idx_map: Any,
        logits_map: Any,
        shared_idx_cache: Any = None,
        shared_sampling_cache: Any = None,
        req_ids_key: Any = None,
    ) -> None:
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

    def _logits_idx_tensor(self, device: Any) -> Any:
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

    def __getattr__(self, name: Any) -> Any:
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
                    # stride 5); vLLM 0.30's Triton top-p kernel loads the
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

    def __setattr__(self, name: Any, value: Any) -> None:
        if name == "sampling_metadata":
            # The underlying metadata object is being replaced; drop the
            # remapped cache so the next read reindexes fresh tensors.
            object.__setattr__(self, "_sampling_metadata_cache", None)
        setattr(object.__getattribute__(self, "_obj"), name, value)


def _beam_remap_inputs(runner: Any, scheduler_output: Any, num_scheduled_tokens: Any) -> list[int]:
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
    new_logits_indices: list[int]
    if layout_key is not None and all_incremental:
        cached_layout = layout_cache.get(layout_key)
        if cached_layout is not None and any(
            beam_data[rid]["beam_width"] != width or beam_data[rid]["prefix_len"] != prefix_len
            for rid, width, prefix_len in cached_layout[-1]
        ):
            # A new session may reuse a request id with different geometry.
            # Rebuild its layout instead of applying stale positions/row maps.
            cached_layout = None
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
                step = mdata.get("decode_step")
                if type(step) is not int or step < 0:
                    raise ValueError("Beam decode requires a nonnegative integer decode_step")
                pos_id = prefix_len + step
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

    new_logits_indices = []
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
            beams_count, beams_count_with_logits, _next_offset, active_beam_ids = (
                _process_beam_decode_request(
                    mdata,
                    seq_len,
                    token_offset,
                    gpu_positions,
                    cpu_positions,
                    new_logits_indices,
                )
            )
            row_to_req_id.extend([req_id] * beams_count)
            row_to_batch_idx.extend([batch_idx] * beams_count)
            logits_to_batch_idx.extend([batch_idx] * beams_count_with_logits)

            request_row_configs.append(
                (req_id, input_row_idx, beams_count_with_logits, active_beam_ids)
            )
            input_row_idx += beams_count
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
    itself) entirely. A non-contiguous index list (chunked prefill or mixed
    standard/beam rows) falls back to the host copy path.

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
    runner: Any,
    orig_input_batch: Any,
    request_row_configs: Any,
    beam_data: Any,
    sampler_output: Any,
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
    token_ids = cast(torch.Tensor, token_ids)
    scores = cast(torch.Tensor, scores)
    if token_ids.ndim != 2 or scores.shape != token_ids.shape or token_ids.shape[0] < 1:
        return False

    ranks = getattr(lp_tensors, "selected_token_ranks", None)
    rank_dtype = cast(torch.Tensor, ranks).dtype if torch.is_tensor(ranks) else torch.int64
    workspace = getattr(runner, "_beam_postprocess_workspaces", None)
    selected_ids = token_ids
    selected_scores = scores
    selected_sampled = None
    selected_ranks = None
    host_ids = decision.get("_host_token_ids")
    host_scores = decision.get("_host_log_probs")
    host_ranks = decision.get("_host_selected_token_ranks")
    # The host-staged reduction is a V1 async-output optimization: V1's
    # bookkeeping already copied the sampler payload to host, so publishing the
    # O(W) compatibility row from the host copy avoids a second D2H.  (This
    # helper is V1-only: the V2 sample path was removed with the LEGACY frame
    # stack.)
    host_ready = bool(
        getattr(runner, "_beam_result_host_ready", False)
        and torch.is_tensor(host_ids)
        and torch.is_tensor(host_scores)
        and torch.is_tensor(host_ranks)
        and cast(torch.Tensor, host_ids).device.type == "cpu"
        and cast(torch.Tensor, host_scores).device.type == "cpu"
        and cast(torch.Tensor, host_ranks).device.type == "cpu"
    )
    if host_ready:
        selected_ids = host_ids
        selected_scores = host_scores
        selected_sampled = cast(torch.Tensor, host_ids)[:, :1]
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
        from vllm_gr.v1.beam.gpu_beam_postprocess_workspace import (
            get_beam_postprocess_workspace,
        )

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


def make_patched_execute_model(
    _orig_execute_model: Callable[..., Any],
    *,
    hooks_type: CudaStageHooksType | None = None,
) -> Callable[..., Any]:
    """Wrap execute_model to publish beam-request context for the run.

    Shared by GPU and NPU runners.
    """
    if getattr(_orig_execute_model, contracts._WORKER_EXECUTE_PATCH_MARKER, False):
        return _orig_execute_model

    @wraps(_orig_execute_model)
    def _patched_execute_model_step(
        self: Any, scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> Any:
        from vllm_gr.config import BeamExecutionMode
        from vllm_gr.config.execution import require_beam_execution_mode
        from vllm_gr.v1.engine.gr_async_scheduler import (
            GR_RETIRE_OUTPUT_ATTR,
            GR_STAGE_OUTPUT_ATTR,
        )

        # Shared stage and retirement code receives an admitted V1 frame.
        if getattr(scheduler_output, GR_RETIRE_OUTPUT_ATTR, None) is not None or getattr(
            scheduler_output, GR_STAGE_OUTPUT_ATTR, None
        ):
            require_beam_execution_mode(self.vllm_config, BeamExecutionMode.V1)

        retirement = gr_execution._dispatch_retirement_frame(self, scheduler_output)
        if retirement is not None:
            return retirement

        stage = gr_execution._dispatch_stage_frame(
            self, scheduler_output, _orig_execute_model, *args, hooks_type=hooks_type, **kwargs
        )
        if stage is not gr_execution._NOT_A_STAGE_FRAME:
            return stage
        if getattr(scheduler_output, "beam_data", {}):
            raise ValueError("Legacy Beam frames have been removed; use beam.execution_mode='v1'.")
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
            return _orig_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            # Deliberately keep ``_current_beam_data`` / ``_current_beam_info``
            # / ``_current_scheduler_output`` alive for the async sampling
            # phase (sample_tokens runs after execute_model returns).  They are
            # overwritten at the top of the next execute_model call.
            BEAM_DATA_VAR.reset(token)

    @wraps(_orig_execute_model)
    def patched_execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        # Track in-flight model execution so the automatic post-warmup GC
        # freeze finds a quiescent boundary: the teardown signal (and hence
        # the freeze scheduling) happens at the START of this wrapper, and
        # gc.collect() holds the GIL for the whole collection — running it
        # here would stall this request's critical path.  The freeze daemon
        # waits for the depth to reach zero before collecting.
        gr_gc._enter_worker_execute()
        try:
            return _patched_execute_model_step(self, scheduler_output, *args, **kwargs)
        finally:
            gr_gc._exit_worker_execute()

    setattr(patched_execute_model, contracts._WORKER_EXECUTE_PATCH_MARKER, True)
    return patched_execute_model
