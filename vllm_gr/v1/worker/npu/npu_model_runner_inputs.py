# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Beam input remapping and per-beam InputBatch views for the NPU V1 runner."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_positions_buffers(runner: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return the runner's device and pinned-host position buffers."""
    positions = getattr(runner, "positions", None)
    cpu_positions = getattr(runner, "_positions_cpu_buf", None)
    if positions is None or isinstance(positions, torch.Tensor):
        return positions, cpu_positions
    device_positions = getattr(positions, "gpu", None)
    buffered_cpu = getattr(positions, "cpu", None)
    if buffered_cpu is not None:
        cpu_positions = buffered_cpu
    return device_positions, cpu_positions


def _process_beam_decode_request(
    mdata: dict[str, Any],
    seq_len: int,
    token_offset: int,
    device_pos: torch.Tensor | None,
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
    if device_pos is not None:
        device_pos[token_offset : token_offset + beam_width] = pos_id
    if cpu_pos is not None:
        cpu_pos[token_offset : token_offset + beam_width] = pos_id
    return beam_width, beam_width, token_offset + beam_width, list(range(beam_width))


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


# Reuse one device index tensor across all sampling attributes when
# remapping per-request tensors to per-beam logits rows.
_SAMPLING_META_ATTRS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalties",
    "presence_penalties",
    "repetition_penalties",
)


class InputBatchProxy:
    def __init__(
        self,
        obj: Any,
        req_map: Any,
        idx_map: Any,
        logits_map: Any,
        *,
        device: Any,
        shared_idx_cache: Any = None,
        shared_sampling_cache: Any = None,
        req_ids_key: Any = None,
    ) -> None:
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_device", torch.device(device))
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

    def _logits_idx_tensor(self) -> Any:
        device = object.__getattribute__(self, "_device")
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
                    object.__getattribute__(self, "_device"),
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
                idx_tensor = self._logits_idx_tensor()
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
                        object.__getattribute__(self, "_device"),
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
            # NPU's _bookkeeping_sync writeback loop also updates num_tokens.
            # Map it to the logical batch slot like num_tokens_no_spec.
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

    Overwrites per-beam RoPE position ids (Job B), builds the per-beam
    ``logits_indices`` list, installs an ``InputBatchProxy`` on
    ``runner.input_batch`` and stashes ``scheduler_output._gr_request_row_configs``
    (Job C). Job A (populating ``beam_info["req_ids"]``) is left to the caller
    because its ordering relative to the original ``_prepare_inputs`` differs
    between runner implementations.

    Assumes ``scheduler_output.beam_data`` is non-empty (caller checks).

    Returns:
        The new per-beam logits-index list (caller converts to a tensor).
    """
    beam_data = getattr(scheduler_output, "beam_data", {})

    runner.input_batch.logprob_token_ids = {}
    orig_req_ids = runner.input_batch.req_ids
    device_positions, cpu_positions = _get_positions_buffers(runner)
    if device_positions is None:
        raise RuntimeError("NPU runner has no device position buffer")

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
            token_offset = 0
            for req_id, width, prefix_len in widths_offsets:
                mdata = beam_data[req_id]
                step = mdata.get("decode_step")
                if type(step) is not int or step < 0:
                    raise ValueError("Beam decode requires a nonnegative integer decode_step")
                pos_id = prefix_len + step
                if device_positions is not None:
                    device_positions[token_offset : token_offset + width] = pos_id
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
                device=device_positions.device,
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
                    device_positions,
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
        device=device_positions.device,
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
    """Materialize logits indices in a runner-owned reusable int32 device buffer.

    The buffer grows when needed. A nonempty index list equal to ``range(n)`` is
    materialized with a device-side ``arange``; other lists are copied from the
    host. The input Python list is still constructed and checked by this path.

    The returned view must be consumed before the next call, which may overwrite
    the buffer.
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
