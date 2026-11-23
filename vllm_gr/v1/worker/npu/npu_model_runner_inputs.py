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


def _materialize_logits_indices(runner: Any, indices: list[int]) -> torch.Tensor:
    """Copy per-beam ``logits_indices`` into a runner-owned reusable buffer.

    The legacy path built a fresh ``torch.tensor(list, device=...)`` on every
    beam step, paying a device allocation + host->device transfer each time.
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
