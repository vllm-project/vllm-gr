# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam Attention metadata and builder."""

from __future__ import annotations

import copy
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm import envs
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.beam.decode_geometry import BeamDecodeGeometry
from vllm_gr.v1.beam.geometry import _compute_beam_bounds


def _persistent_prefix_binding() -> torch.Tensor | None:
    """Return the backend-bound prefix row for the active B=1 session."""

    beam_info = BEAM_DATA_VAR.get()
    if not beam_info:
        return None
    beam_data = beam_info.get("beam_data") or {}
    persistent = [
        metadata
        for metadata in beam_data.values()
        if metadata.get("persistent_runner_state") and metadata.get("is_beam_decode")
    ]
    if not persistent:
        return None
    if len(persistent) != 1:
        raise RuntimeError("persistent GR prefix binding requires exactly one B=1 request")
    session = beam_info.get("beam_session")
    binding = getattr(getattr(session, "context", None), "_gr_persistent_prefix_binding", None)
    if not isinstance(binding, tuple) or len(binding) != 1:
        raise RuntimeError("persistent GR prefix binding is missing or has multiple KV groups")
    prefix_row = binding[0]
    if not torch.is_tensor(prefix_row) or prefix_row.ndim != 1:
        raise RuntimeError("persistent GR prefix binding must be one device block-table row")
    return prefix_row


def get_scheduler_metadata(*args, **kwargs):
    from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available

    if not is_flash_attn_varlen_func_available():
        raise RuntimeError("BeamAttentionMetadataBuilder requires FlashAttention scheduler support")
    from vllm.v1.attention.backends.fa_utils import (
        get_scheduler_metadata as _get_scheduler_metadata,
    )

    return _get_scheduler_metadata(*args, **kwargs)


def _get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


@dataclass
class BeamAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True
    npu_causal_atten_mask: torch.Tensor | None = None

    # Standard Requests batch routing
    std_indices: torch.Tensor | None = None
    std_cu_seqlens_q_device: torch.Tensor | None = None
    std_seqlens_kv_device: torch.Tensor | None = None
    std_max_q_len: int = 0
    std_max_seq_len: int = 0
    std_block_table: torch.Tensor | None = None
    std_seqlens_q_cpu: list | None = None
    std_cu_seqlens_q_cpu: list | None = None
    std_seqlens_kv_cpu: list | None = None
    std_cu_seqlens_kv_cpu: list | None = None

    # Shared prefix context arrays
    prefix_indices: torch.Tensor | None = None
    prefix_cu_seqlens_q_device: torch.Tensor | None = None
    prefix_seqlens_kv_device: torch.Tensor | None = None
    prefix_max_q_len: int = 0
    prefix_max_seq_len: int = 0
    prefix_block_table: torch.Tensor | None = None
    prefix_seqlens_q_cpu: list | None = None
    prefix_cu_seqlens_q_cpu: list | None = None
    prefix_seqlens_kv_cpu: list | None = None
    prefix_cu_seqlens_kv_cpu: list | None = None

    # Unique suffix split-gather channels
    suffix_cu_seqlens_q_device: torch.Tensor | None = None
    suffix_cu_seqlens_k_device: torch.Tensor | None = None
    suffix_seqlens_kv_device: torch.Tensor | None = None
    suffix_max_q_len: int = 0
    suffix_max_seq_len: int = 0
    suffix_total_tokens: int = 0
    suffix_slot_mapping_out: torch.Tensor | None = None
    suffix_seqlens_q_cpu: list | None = None
    suffix_cu_seqlens_q_cpu: list | None = None
    suffix_seqlens_kv_cpu: list | None = None
    suffix_cu_seqlens_kv_cpu: list | None = None

    # Slicing variables for block table updates
    std_req_idxs_cpu: list | None = None
    prefix_req_idxs_cpu: list | None = None
    max_prefix_blocks_count: int = 0

    num_std_indices: int = 0
    num_prefix_indices: int = 0

    scheduler_metadata: dict | None = None
    prefix_scheduler_metadata: dict | None = None
    max_num_splits: int = 0
    block_size: int = 16
    num_heads: int = 0
    num_kv_heads: int = 0

    # --- Incremental decode fields ---
    # When is_beam_decode is True and beam_session is available,
    # the attention forward uses the incremental decode path.
    is_beam_decode: bool = False
    beam_session: Any = None  # BeamSearchSession reference
    # DEPRECATED: beam_layer_idx is no longer the primary pool resolution
    # mechanism.  Prefer passing ``beam_pool`` directly as an explicit
    # parameter to ``cascade_attention``.  This field is kept only for
    # backward compatibility with callers that have not yet migrated.
    beam_layer_idx: int = 0
    beam_pool_slots: list = None  # Per beam-request slot index (for max_batch>1)
    incremental_suffix_kv_len: int = 0  # decode_step + 1


class BeamAttentionMetadataBuilder(AttentionMetadataBuilder[BeamAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        device_type = getattr(device, "type", str(device).partition(":")[0])
        self._is_cuda = device_type == "cuda"
        self._is_npu = device_type == "npu"
        # Default: no pre-allocated fixed-address buffers.  Overridden to True
        # below when CUDA (or, in the future, ACL Graph on NPU) needs them.
        # Tests / kernel harnesses that skip __init__ keep the default and
        # allocate device tensors per build() call.
        self._fixed_buffers = False
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(self.parallel_config)
        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.max_num_splits = 0
        self.aot_schedule = False

        self.use_full_cuda_graph = self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            if hasattr(vllm_config, "attention_config") and hasattr(
                vllm_config.attention_config, "flash_attn_max_num_splits_for_cuda_graph"
            ):
                self.max_num_splits = (
                    vllm_config.attention_config.flash_attn_max_num_splits_for_cuda_graph
                )

        self.aot_sliding_window: tuple[int, int] | None = None

        if self._is_npu:
            self.npu_causal_atten_mask = (
                torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(self.device)
            )
        else:
            self.npu_causal_atten_mask = None

        # --- Fixed-address metadata buffers for graph capture compatibility ---
        # All buffers are pre-allocated at init time with maximum sizes.
        # build() writes into these buffers and returns views (stable addresses).
        # This ensures that tensor addresses captured during CUDA/ACL Graph
        # capture remain valid across replays.  CUDA and NPU allocate these
        # (NPU needs them for ACL Graph FULL-mode beam decode via the xllm
        # path; the cascade path ignores the *_device fields).
        self._fixed_buffers = self._is_cuda or self._is_npu
        if self._fixed_buffers:
            max_reqs = vllm_config.scheduler_config.max_num_seqs
            max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            # Upper bound from GRConfig so the fixed-address metadata buffers
            # always cover the configured beam_max_width (single source of
            # truth; CLI --beam-max-width / additional_config).  When
            # beam_max_width is None (beam decode disabled), no fixed buffers
            # are needed — fall back to the per-request dynamic path.
            from vllm_gr.config import GRConfig

            max_beams = GRConfig.from_vllm_config(vllm_config).beam_max_width
            if max_beams is None:
                self._fixed_buffers = False
            else:
                max_segs = max_reqs * max_beams + 1
                max_model_len = vllm_config.model_config.max_model_len
                max_blocks = max(1, (max_model_len + self.block_size - 1) // self.block_size)

                # cu_seqlens_q buffers (prefix + suffix)
                # These include the leading 0 element: size = num_groups + 1
                self._prefix_cu_seqlens_q_buf = torch.zeros(
                    max_segs, dtype=torch.int32, device=self.device
                )
                self._suffix_cu_seqlens_q_buf = torch.zeros(
                    max_segs, dtype=torch.int32, device=self.device
                )

                # seqused_k / seqlens_kv buffers (per-group, no leading 0)
                self._prefix_seqlens_kv_buf = torch.zeros(
                    max_reqs, dtype=torch.int32, device=self.device
                )
                self._suffix_seqlens_kv_buf = torch.zeros(
                    max_reqs * max_beams, dtype=torch.int32, device=self.device
                )

                # cu_seqlens_k for suffix. Includes leading 0: size = num_suffix_groups + 1
                self._suffix_cu_seqlens_k_buf = torch.zeros(
                    max_segs, dtype=torch.int32, device=self.device
                )

                # block_table buffers
                self._prefix_block_table_buf = torch.zeros(
                    (max_reqs, max_blocks), dtype=torch.int32, device=self.device
                )
                self._std_block_table_buf = torch.zeros(
                    (max_reqs, max_blocks), dtype=torch.int32, device=self.device
                )

                # indices buffers
                # Per-token index buffers: each std/prefix token contributes one
                # index (see _build_std_mappings: std_indices.extend(range(...))),
                # so the upper bound is max_num_batched_tokens, not max_reqs*max_beams.
                # When max_num_batched_tokens > max_reqs*max_beams (e.g. seqs=64,
                # beams=128, budget=16384) the old sizing caused a silent slice
                # truncation and copy_ shape mismatch → RuntimeError.
                max_token_indices = max(max_reqs * max_beams, max_num_batched_tokens)
                self._std_indices_buf = torch.zeros(
                    max_token_indices, dtype=torch.long, device=self.device
                )
                self._prefix_indices_buf = torch.zeros(
                    max_token_indices, dtype=torch.long, device=self.device
                )

                # std cu_seqlens_q / seqlens_kv
                self._std_cu_seqlens_q_buf = torch.zeros(
                    max_reqs + 1, dtype=torch.int32, device=self.device
                )
                self._std_seqlens_kv_buf = torch.zeros(
                    max_reqs, dtype=torch.int32, device=self.device
                )

                # Single pinned host + device staging pair so every per-step
                # seqlens / cu_seqlens int32 array is moved device-side in ONE
                # H2D transfer instead of N independent small ones.  With W=128
                # the per-step build() was launching ~6 tiny HtoD copies whose
                # launch latency dominated the host time of _build_attention_metadata.
                self._h2d_host = torch.zeros(4 * max_segs, dtype=torch.int32, pin_memory=True)
                self._h2d_dev = torch.zeros(4 * max_segs, dtype=torch.int32, device=self.device)

    def _update_aot_schedule(self, fast_build: bool):
        self.aot_schedule = self.aot_schedule and not fast_build
        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)

    def _get_max_num_splits(self, num_actual_tokens: int) -> int:
        max_num_splits = 0
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            max_num_splits = self.max_num_splits

        if self._is_cuda and envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1
        return max_num_splits

    def _geom_cache_key(
        self,
        num_reqs: int,
        req_ids: list,
        beam_data: dict,
        q_starts: list,
        seq_lens_cpu_list: list[int],
        is_incremental: bool,
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
            self.block_size,
        )

    def _geometry_cache(self) -> dict:
        cache = getattr(self, "_geom_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_geom_cache", cache)
        return cache

    def _invalidate_geometry_cache(self) -> None:
        self._geometry_cache().clear()
        self._resident_geometry_key = None

    def _select_geometry_cache(self, geom_key) -> dict:
        # There is one physical set of fixed buffers, not one per key. An
        # initialized bit for an older request cannot survive another layout
        # overwriting those buffers (including Graph warmup/capture inputs).
        if getattr(self, "_resident_geometry_key", None) != geom_key:
            self._invalidate_geometry_cache()
            self._resident_geometry_key = geom_key
        return self._geometry_cache()

    def _cached_seqlens_views(
        self,
        geom_key,
        *,
        n_pk: int,
        n_pc: int,
        n_sk: int,
        n_scq: int,
        n_sck: int,
    ):
        """Return the five fixed seqlens/cu_seqlens device views for a geometry.

        When ``geom_key`` has already produced these views, they are returned
        from a cache instead of re-slicing the address-stable buffers every
        step.  Because a cache hit means the beam geometry (widths /
        prefix_len / cache_len) is unchanged, the slice extents ``n_*`` are
        identical on every step, so the reused view objects stay valid (their
        storage is the fixed-address device buffer; per-step values are updated
        in place by the caller).

        Returns ``(prefix_seqlens_kv, prefix_cu_seqlens_q, suffix_seqlens_kv,
        suffix_cu_seqlens_q, suffix_cu_seqlens_k)`` or None when ``geom_key``
        is None (fall back to full per-step slicing).
        """
        if geom_key is None:
            return None
        views_cache = self._select_geometry_cache(geom_key).setdefault("views", {})
        cached = views_cache.get(geom_key)
        if cached is not None:
            sizes = (n_pk, n_pc, n_sk, n_scq, n_sck)
            if all(view.numel() == size for view, size in zip(cached, sizes)):
                return cached
            # A caller-provided key cannot authorize reusing incompatible
            # slices. Rebuild the views and their static contents together.
            self._invalidate_geometry_cache()
            views_cache = self._select_geometry_cache(geom_key).setdefault("views", {})
        views = (
            self._prefix_seqlens_kv_buf[:n_pk],
            self._prefix_cu_seqlens_q_buf[:n_pc],
            self._suffix_seqlens_kv_buf[:n_sk],
            self._suffix_cu_seqlens_q_buf[:n_scq],
            self._suffix_cu_seqlens_k_buf[:n_sck],
        )
        views_cache[geom_key] = views
        return views

    def _upload_incremental_seqlens(
        self,
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

        Incremental decode: only ``suffix_seqlens_kv`` and
        ``suffix_cu_seqlens_k`` depend on ``decode_step``.  The three fixed
        payloads are written once while ``geom_key`` remains resident in the
        address-stable buffers; later steps skip them until another layout
        overwrites the storage. The step-derived suffix arrays
        are rebuilt on device with ``fill_`` / ``arange`` from
        ``decode_step`` — no host list, no H2D.

        Returns True when the caller should skip the generic
        ``_pack_int32_to_dev`` path (geom_key admitted the cache), False to
        fall back to full construction.
        """
        if geom_key is None or decode_step is None:
            return False
        cache = self._select_geometry_cache(geom_key)
        if not cache.get(geom_key, False):
            if _persistent_prefix_binding() is not None:
                # Admission/binding currently restricts persistent execution
                # to B=1. Keep the writer itself request-group based: future
                # multi-session support must not reuse a single-prefix layout.
                offsets = [0] + list(prefix_cu_seqlens_q_cpu)
                try:
                    geometry = BeamDecodeGeometry(
                        tuple(b - a for a, b in zip(offsets, offsets[1:])),
                        tuple(prefix_seqlens_kv_cpu),
                    )
                except ValueError as exc:
                    raise RuntimeError("persistent GR seqlens geometry mismatch") from exc
                if (
                    geometry.num_rows != suffix_seqlens_kv_view.numel()
                    or list(suffix_cu_seqlens_q_cpu) != list(range(1, geometry.num_rows + 1))
                ):
                    raise RuntimeError("persistent GR seqlens geometry mismatch")
                try:
                    geometry.write_static_seqlens(
                        prefix_seqlens_kv_view,
                        prefix_cu_seqlens_q_view,
                        suffix_cu_seqlens_q_view,
                    )
                except ValueError as exc:
                    raise RuntimeError("persistent GR seqlens geometry mismatch") from exc
            else:
                self._pack_int32_to_dev(
                    [
                        (prefix_seqlens_kv_view, list(prefix_seqlens_kv_cpu)),
                        (prefix_cu_seqlens_q_view, [0] + list(prefix_cu_seqlens_q_cpu)),
                        (suffix_cu_seqlens_q_view, [0] + list(suffix_cu_seqlens_q_cpu)),
                    ]
                )
            cache[geom_key] = True

        # Device-side step update (no H2D):
        #   suffix_seqlens_kv   = [s+1]*w
        #   suffix_cu_seqlens_k = [0, s+1, 2(s+1), ..., w*(s+1)]
        # A single strided arange emits the cu_seqlens_k array directly (its
        # i-th element is i*s), avoiding the separate arange+mul_ pair — just
        # two small launches per step instead of three.
        s = decode_step + 1
        num_suffix = suffix_seqlens_kv_view.numel()
        suffix_seqlens_kv_view.fill_(s)
        if suffix_cu_seqlens_k_view is not None:
            torch.arange(
                0,
                (num_suffix + 1) * s,
                step=s,
                dtype=torch.int32,
                device=self.device,
                out=suffix_cu_seqlens_k_view,
            )
        return True

    def _get_schedule(
        self,
        batch_size: int,
        cu_query_lens: torch.Tensor,
        max_query_len: int,
        seqlens: torch.Tensor,
        max_seq_len: int,
        causal: bool,
        max_num_splits: int,
    ) -> dict | None:
        if not self.aot_schedule:
            return None

        cache_dtype = self.cache_config.cache_dtype
        if cache_dtype.startswith("fp8"):
            qkv_dtype = _get_fp8_dtype_for_flashattn(cache_dtype)
        else:
            qkv_dtype = self.kv_cache_dtype

        return get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            num_heads_q=self.num_heads_q,
            num_heads_kv=self.num_kv_heads,
            headdim=self.headdim,
            cache_seqlens=seqlens,
            qkv_dtype=qkv_dtype,
            cu_seqlens_q=cu_query_lens,
            page_size=self.block_size,
            causal=causal,
            window_size=self.aot_sliding_window,
            num_splits=max_num_splits,
        )

    def _segment_requests(
        self,
        num_reqs: int,
        req_ids: list,
        beam_data: dict,
        q_starts: list,
        seq_lens_cpu_list: list[int],
        beam_session=None,
    ):
        std_reqs = []
        beam_req_idxs = []
        beam_prefix_lens = []
        beam_widths = []
        beam_decode_steps = []
        beam_cache_lens = []
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
                if (delta > 0 and delta > self.block_size) or (cache_len == 0):
                    std_q_len = min(delta, q_len)
                    std_kv_len = cache_len + std_q_len
                    cache_len = cache_len + std_q_len
                    std_reqs.append((i, std_q_len, std_kv_len))
                    q_offset = std_q_len

                beam_prefix_lens.append(prefix_len)
                beam_widths.append(mdata["beam_width"])
                beam_decode_steps.append(mdata["beam_decode_steps"])
                beam_cache_lens.append(cache_len)
                beam_q_start_offsets.append(q_offset)
                beam_decode_step_list.append(mdata.get("decode_step", 0))

                # Get pool slot for this request from the session registry
                pool_slot = 0  # default for max_batch=1
                if beam_session is not None and req_id is not None:
                    with suppress(KeyError):
                        pool_slot = beam_session.context.registry.get_slot(req_id)
                beam_pool_slots.append(pool_slot)
            else:
                kv_len = seq_lens_cpu_list[i]
                std_reqs.append((i, q_len, kv_len))

        return (
            std_reqs,
            beam_req_idxs,
            beam_prefix_lens,
            beam_widths,
            beam_decode_steps,
            beam_cache_lens,
            beam_q_start_offsets,
            beam_decode_step_list,
            beam_pool_slots,
        )

    def _build_std_mappings(self, std_reqs: list, q_starts: list):
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

    def _build_beam_mappings(
        self,
        beam_req_idxs,
        beam_prefix_lens,
        beam_widths,
        beam_decode_steps,
        beam_cache_lens,
        beam_q_start_offsets,
        q_starts,
        common_attn_metadata,
        is_incremental: bool = False,
        beam_decode_step_list: list | None = None,
    ):
        """Builds attention mapping metadata for all active beams in a batch.

        This method decomposes the query token allocations (chunk budgets) for
        each request into prefix (shared history) and suffix (beam-specific lookahead)
        computations.

        When is_incremental=True (incremental decode path), each beam contributes
        1 query token and decode_step+1 suffix KV tokens. The suffix slot mapping
        is not needed (pool slice is used instead of gather).
        """
        # Prefix attention tracking lists
        beam_prefix_indices = []
        prefix_seqlens_q_cpu = []
        prefix_seqlens_kv_cpu = []
        prefix_req_idxs_cpu = []

        # Suffix attention tracking lists
        suffix_seqlens_q_cpu = []
        suffix_seqlens_kv_cpu = []
        beam_suffix_slot_mapping_out_list = []

        max_prefix_len = max(beam_prefix_lens) if beam_prefix_lens else 0
        max_prefix_blocks_count = (max_prefix_len + self.block_size - 1) // self.block_size

        out_slot = 0
        for idx, (req_idx, prefix_len, w, steps, cache_len, q_offset) in enumerate(
            zip(
                beam_req_idxs,
                beam_prefix_lens,
                beam_widths,
                beam_decode_steps,
                beam_cache_lens,
                beam_q_start_offsets,
            )
        ):
            q_start = q_starts[req_idx] + q_offset
            q_end = q_starts[req_idx + 1]
            q_len = q_end - q_start
            if w == 0 or steps == 0 or q_len <= 0:
                continue

            # --- Incremental decode path ---
            if is_incremental and beam_decode_step_list is not None:
                decode_step = beam_decode_step_list[idx]
                # Each beam: 1 query token, decode_step+1 suffix KV tokens
                # Prefix: all beams share the same prefix_len
                # List-multiplication extends in C (vs a per-beam Python append
                # loop that dominates the host time of _build_beam_mappings for
                # beam_width=128 on every decode step).
                suffix_seqlens_q_cpu.extend([1] * w)
                suffix_seqlens_kv_cpu.extend([decode_step + 1] * w)

                # All beams share the same prefix group
                beam_prefix_indices.extend(range(q_start, q_start + w))
                prefix_seqlens_q_cpu.append(w)  # total query tokens for prefix
                # Remainder-block analysis (PREFIX_REMAINDER_BLOCK_ANALYSIS.md):
                # the trailing partial-block tokens belong to the prefix
                # (shared KV), NOT to the suffix. Their KV is written to the
                # paged cache during prefill, and prefix FA reads exactly
                # prefix_len KV via seqused_k, so FA ignores the remainder-block
                # padding automatically. Using the exact prefix_len (not the
                # block-aligned length) preserves those trailing prefix tokens.
                prefix_seqlens_kv_cpu.append(prefix_len)
                prefix_req_idxs_cpu.append(req_idx)
                continue

            # --- Standard (recompute) path ---
            q_curr = q_start
            chunk_budget = q_len
            prefix_groups = []  # Elements: [group_q_start, group_q_len, group_cache_len]

            b_prefix_len = min(cache_len, prefix_len)
            b_cache_len = cache_len
            # Loop 1: Partition query tokens across the w active beams for this request
            for b in range(w):
                if chunk_budget <= 0:
                    break

                b_uncached, b_suffix_len, _b_past_suffix, _b_physical = _compute_beam_bounds(
                    b, prefix_len, b_cache_len, steps, chunk_budget
                )

                full_mapping = np.arange(out_slot, out_slot + b_suffix_len, dtype=np.int32)

                # Invariant: Merge adjacent beams that share the exact same prefix length
                # into a single prefix group to parallelize key/value lookup during attention.
                if prefix_groups and prefix_groups[-1][2] == b_prefix_len:
                    prefix_groups[-1][1] += b_uncached
                else:
                    prefix_groups.append([q_curr, b_uncached, b_prefix_len])

                beam_suffix_slot_mapping_out_list.extend(full_mapping)
                suffix_seqlens_q_cpu.append(b_uncached)
                suffix_seqlens_kv_cpu.append(b_suffix_len)

                # Advance state for the next beam
                b_cache_len += b_uncached
                q_curr += b_uncached
                chunk_budget -= b_uncached
                out_slot += b_uncached
                b_prefix_len = min(b_cache_len, prefix_len)

            # Loop 2: Record grouped prefix indexing lists for the attention kernel
            for group_q_start, group_q_len, group_cache_len in prefix_groups:
                beam_prefix_indices.extend(range(group_q_start, group_q_start + group_q_len))
                prefix_seqlens_q_cpu.append(group_q_len)
                prefix_seqlens_kv_cpu.append(group_cache_len)
                prefix_req_idxs_cpu.append(req_idx)

        # Helper to compute prefix cumulative sums
        prefix_cu_seqlens_q_cpu = None
        prefix_cu_seqlens_kv_cpu = None
        if prefix_seqlens_q_cpu:
            prefix_cu_seqlens_q_cpu = np.cumsum(prefix_seqlens_q_cpu).tolist()
            prefix_cu_seqlens_kv_cpu = np.cumsum(prefix_seqlens_kv_cpu).tolist()

        # Helper to compute suffix cumulative sums
        suffix_cu_seqlens_q_cpu = None
        suffix_cu_seqlens_kv_cpu = None
        if suffix_seqlens_q_cpu:
            suffix_cu_seqlens_q_cpu = np.cumsum(suffix_seqlens_q_cpu).tolist()
            suffix_cu_seqlens_kv_cpu = np.cumsum(suffix_seqlens_kv_cpu).tolist()

        return (
            beam_prefix_indices,
            prefix_seqlens_q_cpu,
            prefix_seqlens_kv_cpu,
            suffix_seqlens_q_cpu,
            suffix_seqlens_kv_cpu,
            prefix_req_idxs_cpu,
            beam_suffix_slot_mapping_out_list,
            prefix_cu_seqlens_q_cpu,
            prefix_cu_seqlens_kv_cpu,
            suffix_cu_seqlens_q_cpu,
            suffix_cu_seqlens_kv_cpu,
            max_prefix_blocks_count,
        )

    def _pack_int32_to_dev(self, entries: list[tuple[Any, list[int]]]) -> None:
        """Write several small host int lists to device in ONE H2D.

        ``entries`` is a list of ``(dst_slice, host_list)``.  All the step's
        seqlens / cu_seqlens / index metadata is staged into a single pinned
        host buffer, moved across the PCIe bus once, then scattered into the
        preallocated (address-stable) device buffers with device-to-device
        copies.  This replaces N independent ``async torch.tensor(...).copy_``
        HtoD calls, whose per-copy launch latency dominated the host time of
        ``_build_attention_metadata`` for beam decode.

        Needs ``self._fixed_buffers`` and the ``_h2d_host`` / ``_h2d_dev``
        staging pair allocated by ``__init__``.
        """
        total = sum(len(host_list) for _dst, host_list in entries)
        if total == 0:
            return
        assert total <= self._h2d_host.numel(), (
            f"beam metadata host staging overflow: {total} > {self._h2d_host.numel()}"
        )

        # Build the host payload with one numpy concatenate instead of a
        # per-entry ``torch.as_tensor(list)`` + ``copy_`` pair.  The Python
        # list -> tensor conversion per entry is the dominant host cost at this
        # size (the arrays total only a few KB), so a single bulk write into
        # the pinned staging keeps the PCIe transfer as the only real work.
        np_dtype = np.int32 if self._h2d_host.dtype == torch.int32 else np.int64
        segments = [np.asarray(hl, dtype=np_dtype) for _dst, hl in entries]
        packed = np.concatenate(segments) if len(segments) > 1 else segments[0]
        self._h2d_host[:total].copy_(torch.from_numpy(packed))
        # Single host -> device transfer for the whole payload.
        self._h2d_dev[:total].copy_(self._h2d_host[:total])

        off = 0
        for dst, host_list in entries:
            n = len(host_list)
            if n:
                dst.copy_(self._h2d_dev[off : off + n])
                off += n

    def _create_std_tensors(
        self,
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
        num_std_blocks = (std_max_seq_len + self.block_size - 1) // self.block_size

        if self._fixed_buffers:
            std_indices_tensor = self._std_indices_buf[:num_std]
            std_indices_tensor.copy_(torch.as_tensor(std_indices, dtype=torch.long, device="cpu"))
            std_block_table = self._std_block_table_buf[: len(std_req_idxs), :num_std_blocks]
            std_block_table.copy_(
                common_attn_metadata.block_table_tensor[std_req_idxs, :num_std_blocks]
            )
            num_cu = len(std_cu_seqlens_q_cpu) + 1  # +1 for leading 0
            std_cu_seqlens_q_device = self._std_cu_seqlens_q_buf[:num_cu]
            std_seqlens_kv_device = self._std_seqlens_kv_buf[: len(std_seqlens_kv_cpu)]
            self._pack_int32_to_dev(
                [
                    (std_cu_seqlens_q_device, [0] + list(std_cu_seqlens_q_cpu)),
                    (std_seqlens_kv_device, list(std_seqlens_kv_cpu)),
                ]
            )
        else:
            std_indices_tensor = torch.tensor(std_indices, dtype=torch.long, device=device)
            std_block_table = common_attn_metadata.block_table_tensor[
                std_req_idxs, :num_std_blocks
            ].contiguous()
            if self._is_cuda:
                # Tests / kernel harnesses that skip __init__ buffer alloc.
                std_cu_seqlens_q_device = torch.tensor(
                    [0] + std_cu_seqlens_q_cpu, dtype=torch.int32, device=device
                )
                std_seqlens_kv_device = torch.tensor(
                    std_seqlens_kv_cpu, dtype=torch.int32, device=device
                )
            else:
                # NPU: npu_fused_infer_attention_score_v2 currently reads
                # CPU lists (std_cu_seqlens_q_cpu / std_seqlens_kv_cpu).
                # TODO(ACL-graph): switch NPU to device tensors so this
                # branch disappears and seqlens handling is unified.
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

    def _create_beam_tensors(
        self,
        common_attn_metadata,
        prefix_indices,
        prefix_seqlens_q_cpu,
        prefix_cu_seqlens_q_cpu,
        prefix_seqlens_kv_cpu,
        suffix_seqlens_q_cpu,
        suffix_cu_seqlens_q_cpu,
        suffix_cu_seqlens_kv_cpu,
        suffix_seqlens_kv_cpu,
        suffix_slot_mapping_out_list,
        prefix_req_idxs_cpu,
        max_prefix_blocks_count,
        device,
        max_num_splits: int,
        *,
        geom_key=None,
        decode_step: int | None = None,
    ):
        if not prefix_seqlens_q_cpu:
            # Full structured initializations for profiling warm-up model passes
            prefix_indices_tensor = prefix_seqlens_kv_device = prefix_cu_seqlens_q_device = (
                prefix_block_table
            ) = None
            prefix_max_q_len = prefix_max_seq_len = 0
            suffix_cu_seqlens_q_device = suffix_cu_seqlens_k_device = suffix_seqlens_kv_device = (
                suffix_slot_mapping_out
            ) = None
            suffix_max_q_len = suffix_max_seq_len = suffix_total_tokens = 0
            prefix_scheduler_metadata = None
            return (
                prefix_indices_tensor,
                prefix_seqlens_kv_device,
                prefix_max_seq_len,
                prefix_cu_seqlens_q_device,
                prefix_max_q_len,
                prefix_block_table,
                suffix_seqlens_kv_device,
                suffix_slot_mapping_out,
                suffix_max_seq_len,
                suffix_total_tokens,
                suffix_cu_seqlens_q_device,
                suffix_cu_seqlens_k_device,
                suffix_max_q_len,
                prefix_scheduler_metadata,
            )

        num_prefix = len(prefix_indices)
        prefix_max_q_len = max(prefix_seqlens_q_cpu) if prefix_seqlens_q_cpu else 0
        suffix_max_seq_len = max(suffix_seqlens_kv_cpu) if suffix_seqlens_kv_cpu else 0
        suffix_max_q_len = max(suffix_seqlens_q_cpu) if suffix_seqlens_q_cpu else 0

        if self._fixed_buffers:
            prefix_indices_tensor = self._prefix_indices_buf[:num_prefix]
            # Incremental decode produces a contiguous prefix index range
            # (q_start .. q_start+w).  Emit it on device with a strided arange
            # instead of converting the Python list every step (no host list,
            # no H2D).  Only take this path when the whole list is verified
            # contiguous; anything else (multi-request recompute layout) falls
            # back to the host list transfer.
            if num_prefix and all(
                prefix_indices[i] == prefix_indices[0] + i for i in range(num_prefix)
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
            prefix_max_seq_len = self._prefix_block_table_buf.shape[1] * self.block_size
            prefix_block_table = self._prefix_block_table_buf[: len(prefix_req_idxs_cpu)]
            persistent_binding = _persistent_prefix_binding()
            if persistent_binding is not None:
                if len(prefix_req_idxs_cpu) != 1 or persistent_binding.numel() < (
                    max_prefix_blocks_count
                ):
                    raise RuntimeError("persistent GR prefix binding geometry mismatch")
                prefix_block_table[0, :max_prefix_blocks_count].copy_(
                    persistent_binding[:max_prefix_blocks_count]
                )
            else:
                prefix_block_table[:, :max_prefix_blocks_count].copy_(
                    common_attn_metadata.block_table_tensor[
                        prefix_req_idxs_cpu, :max_prefix_blocks_count
                    ]
                )
        else:
            prefix_indices_tensor = torch.tensor(prefix_indices, dtype=torch.long, device=device)
            prefix_max_seq_len = max(prefix_seqlens_kv_cpu) if prefix_seqlens_kv_cpu else 0
            persistent_binding = _persistent_prefix_binding()
            if persistent_binding is not None:
                if len(prefix_req_idxs_cpu) != 1 or persistent_binding.numel() < (
                    max_prefix_blocks_count
                ):
                    raise RuntimeError("persistent GR prefix binding geometry mismatch")
                prefix_block_table = persistent_binding[
                    :max_prefix_blocks_count
                ].unsqueeze(0)
            else:
                prefix_block_table = common_attn_metadata.block_table_tensor[
                    prefix_req_idxs_cpu, :max_prefix_blocks_count
                ].contiguous()

        if suffix_slot_mapping_out_list:
            suffix_total_tokens = len(suffix_slot_mapping_out_list)
            if num_prefix > 0:
                suffix_map_tensor = torch.tensor(
                    suffix_slot_mapping_out_list, dtype=torch.long, device=device
                ).clamp_(0, num_prefix - 1)
                global_chunk_token_indices = prefix_indices_tensor[suffix_map_tensor]
                suffix_slot_mapping_out = common_attn_metadata.slot_mapping[
                    global_chunk_token_indices
                ].to(dtype=torch.int32)
            else:
                suffix_slot_mapping_out = torch.empty(0, dtype=torch.int32, device=device)
        else:
            suffix_total_tokens = 0
            suffix_slot_mapping_out = torch.empty(0, dtype=torch.int32, device=device)

        if self._fixed_buffers:
            num_prefix_groups = len(prefix_seqlens_kv_cpu)
            num_prefix_cu = len(prefix_cu_seqlens_q_cpu) + 1
            num_suffix_beams = len(suffix_seqlens_kv_cpu)
            num_suffix_cu_q = len(suffix_cu_seqlens_q_cpu) + 1
            num_suffix_cu_k = len(suffix_cu_seqlens_kv_cpu) + 1

            # Reuse the fixed-address seqlens views for a known geometry
            # instead of re-slicing the address-stable buffers every step.
            cached_views = self._cached_seqlens_views(
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
                prefix_seqlens_kv_device = self._prefix_seqlens_kv_buf[:num_prefix_groups]
                prefix_cu_seqlens_q_device = self._prefix_cu_seqlens_q_buf[:num_prefix_cu]
                suffix_seqlens_kv_device = self._suffix_seqlens_kv_buf[:num_suffix_beams]
                suffix_cu_seqlens_q_device = self._suffix_cu_seqlens_q_buf[:num_suffix_cu_q]
                suffix_cu_seqlens_k_device = self._suffix_cu_seqlens_k_buf[:num_suffix_cu_k]

            # Incremental decode admits the fixed-geometry cache: write the three
            # step-invariant payloads once per geometry, then refresh the
            # decode_step-derived suffix arrays on device (zero H2D).
            if self._upload_incremental_seqlens(
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
                self._invalidate_geometry_cache()
                self._pack_int32_to_dev(
                    [
                        (prefix_seqlens_kv_device, list(prefix_seqlens_kv_cpu)),
                        (prefix_cu_seqlens_q_device, [0] + list(prefix_cu_seqlens_q_cpu)),
                        (suffix_seqlens_kv_device, list(suffix_seqlens_kv_cpu)),
                        (suffix_cu_seqlens_q_device, [0] + list(suffix_cu_seqlens_q_cpu)),
                        (suffix_cu_seqlens_k_device, [0] + list(suffix_cu_seqlens_kv_cpu)),
                    ]
                )
        elif self._is_cuda:
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

        prefix_scheduler_metadata = self._get_schedule(
            batch_size=len(prefix_seqlens_q_cpu),
            cu_query_lens=prefix_cu_seqlens_q_device,
            max_query_len=prefix_max_q_len,
            seqlens=prefix_seqlens_kv_device,
            max_seq_len=prefix_max_seq_len,
            causal=False,
            max_num_splits=max_num_splits,
        )

        return (
            prefix_indices_tensor,
            prefix_seqlens_kv_device,
            prefix_max_seq_len,
            prefix_cu_seqlens_q_device,
            prefix_max_q_len,
            prefix_block_table,
            suffix_seqlens_kv_device,
            suffix_slot_mapping_out,
            suffix_max_seq_len,
            suffix_total_tokens,
            suffix_cu_seqlens_q_device,
            suffix_cu_seqlens_k_device,
            suffix_max_q_len,
            prefix_scheduler_metadata,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> BeamAttentionMetadata:
        self._update_aot_schedule(fast_build)
        max_num_splits = self._get_max_num_splits(common_attn_metadata.num_actual_tokens)
        device = self.device
        block_size = self.block_size
        num_reqs = common_attn_metadata.num_reqs
        beam_info = BEAM_DATA_VAR.get()
        beam_data = beam_info["beam_data"] if beam_info else {}
        req_ids = beam_info["req_ids"] if beam_info else []

        q_starts = common_attn_metadata.query_start_loc_cpu.tolist()
        # Prefer the runner's CPU mirror (populated by _prepare_inputs) to avoid
        # a device->host .cpu() round-trip on every build().
        seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        if seq_lens_cpu is not None and seq_lens_cpu.numel() >= num_reqs:
            seq_lens_cpu_list = seq_lens_cpu[:num_reqs].tolist()
        else:
            seq_lens_cpu_list = common_attn_metadata.seq_lens.cpu().tolist()

        # Get beam_session early so _segment_requests can use it for slot lookup
        beam_session = beam_info.get("beam_session") if beam_info else None

        (
            std_reqs,
            beam_req_idxs,
            beam_prefix_lens,
            beam_widths,
            beam_decode_steps,
            beam_cache_lens,
            beam_q_start_offsets,
            beam_decode_step_list,
            beam_pool_slots,
        ) = self._segment_requests(
            num_reqs, req_ids, beam_data, q_starts, seq_lens_cpu_list, beam_session
        )

        (
            std_indices,
            std_seqlens_q_cpu,
            std_seqlens_kv_cpu,
            std_req_idxs,
            std_cu_seqlens_q_cpu,
            std_cu_seqlens_kv_cpu,
        ) = self._build_std_mappings(std_reqs, q_starts)

        # Check if we have an active beam session for incremental decode
        is_incremental = (
            beam_session is not None and beam_session.is_active and len(beam_decode_step_list) > 0
        )
        geom_key = self._geom_cache_key(
            num_reqs, req_ids, beam_data, q_starts, seq_lens_cpu_list, is_incremental
        )
        decode_step = beam_decode_step_list[0] if beam_decode_step_list else None

        (
            prefix_indices,
            prefix_seqlens_q_cpu,
            prefix_seqlens_kv_cpu,
            suffix_seqlens_q_cpu,
            suffix_seqlens_kv_cpu,
            prefix_req_idxs_cpu,
            suffix_slot_mapping_out_list,
            prefix_cu_seqlens_q_cpu,
            prefix_cu_seqlens_kv_cpu,
            suffix_cu_seqlens_q_cpu,
            suffix_cu_seqlens_kv_cpu,
            max_prefix_blocks_count,
        ) = self._build_beam_mappings(
            beam_req_idxs,
            beam_prefix_lens,
            beam_widths,
            beam_decode_steps,
            beam_cache_lens,
            beam_q_start_offsets,
            q_starts,
            common_attn_metadata,
            is_incremental=is_incremental,
            beam_decode_step_list=beam_decode_step_list,
        )

        (
            std_indices_tensor,
            std_cu_seqlens_q_device,
            std_seqlens_kv_device,
            std_max_q_len,
            std_max_seq_len,
            std_block_table,
        ) = self._create_std_tensors(
            std_indices,
            std_seqlens_q_cpu,
            std_cu_seqlens_q_cpu,
            std_seqlens_kv_cpu,
            std_req_idxs,
            common_attn_metadata,
            device,
        )

        (
            prefix_indices_tensor,
            prefix_seqlens_kv_device,
            prefix_max_seq_len,
            prefix_cu_seqlens_q_device,
            prefix_max_q_len,
            prefix_block_table,
            suffix_seqlens_kv_device,
            suffix_slot_mapping_out,
            suffix_max_seq_len,
            suffix_total_tokens,
            suffix_cu_seqlens_q_device,
            suffix_cu_seqlens_k_device,
            suffix_max_q_len,
            prefix_scheduler_metadata,
        ) = self._create_beam_tensors(
            common_attn_metadata,
            prefix_indices,
            prefix_seqlens_q_cpu,
            prefix_cu_seqlens_q_cpu,
            prefix_seqlens_kv_cpu,
            suffix_seqlens_q_cpu,
            suffix_cu_seqlens_q_cpu,
            suffix_cu_seqlens_kv_cpu,
            suffix_seqlens_kv_cpu,
            suffix_slot_mapping_out_list,
            prefix_req_idxs_cpu,
            max_prefix_blocks_count,
            device,
            max_num_splits,
            geom_key=geom_key,
            decode_step=decode_step,
        )

        # Determine incremental decode metadata
        incremental_suffix_kv_len = 0
        if is_incremental and beam_decode_step_list:
            # In incremental decode, suffix KV length = decode_step + 1
            incremental_suffix_kv_len = beam_decode_step_list[0] + 1

        meta = BeamAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
            npu_causal_atten_mask=getattr(self, "npu_causal_atten_mask", None),
            std_indices=std_indices_tensor,
            std_cu_seqlens_q_device=std_cu_seqlens_q_device,
            std_seqlens_kv_device=std_seqlens_kv_device,
            std_max_q_len=std_max_q_len,
            std_max_seq_len=std_max_seq_len,
            std_block_table=std_block_table,
            std_seqlens_q_cpu=std_seqlens_q_cpu,
            std_cu_seqlens_q_cpu=std_cu_seqlens_q_cpu,
            std_seqlens_kv_cpu=std_seqlens_kv_cpu,
            std_cu_seqlens_kv_cpu=std_cu_seqlens_kv_cpu,
            std_req_idxs_cpu=std_req_idxs,
            prefix_req_idxs_cpu=prefix_req_idxs_cpu,
            max_prefix_blocks_count=max_prefix_blocks_count,
            prefix_indices=prefix_indices_tensor,
            prefix_cu_seqlens_q_device=prefix_cu_seqlens_q_device,
            prefix_seqlens_kv_device=prefix_seqlens_kv_device,
            prefix_max_q_len=prefix_max_q_len,
            prefix_max_seq_len=prefix_max_seq_len,
            prefix_block_table=prefix_block_table,
            prefix_seqlens_q_cpu=prefix_seqlens_q_cpu,
            prefix_cu_seqlens_q_cpu=prefix_cu_seqlens_q_cpu,
            prefix_seqlens_kv_cpu=prefix_seqlens_kv_cpu,
            prefix_cu_seqlens_kv_cpu=prefix_cu_seqlens_kv_cpu,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            suffix_cu_seqlens_q_device=suffix_cu_seqlens_q_device,
            suffix_cu_seqlens_k_device=suffix_cu_seqlens_k_device,
            suffix_seqlens_kv_device=suffix_seqlens_kv_device,
            suffix_max_q_len=suffix_max_q_len,
            suffix_max_seq_len=suffix_max_seq_len,
            suffix_total_tokens=suffix_total_tokens,
            suffix_slot_mapping_out=suffix_slot_mapping_out,
            suffix_seqlens_q_cpu=suffix_seqlens_q_cpu,
            suffix_cu_seqlens_q_cpu=suffix_cu_seqlens_q_cpu,
            suffix_seqlens_kv_cpu=suffix_seqlens_kv_cpu,
            suffix_cu_seqlens_kv_cpu=suffix_cu_seqlens_kv_cpu,
            num_std_indices=len(std_indices) if std_indices else 0,
            num_prefix_indices=len(prefix_indices) if prefix_indices else 0,
            max_num_splits=max_num_splits,
            block_size=block_size,
            num_heads=self.num_heads_q,
            num_kv_heads=self.num_kv_heads,
            # Incremental decode fields
            is_beam_decode=is_incremental,
            beam_session=beam_session,
            beam_pool_slots=beam_pool_slots,
            incremental_suffix_kv_len=incremental_suffix_kv_len,
        )
        return meta

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> BeamAttentionMetadata:
        """Build metadata for CUDA Graph capture.

        Uses the same build() method with fixed-address buffers.
        The returned metadata references pre-allocated buffers whose addresses
        are stable across steps, making them compatible with CUDA Graph capture.

        During replay, buffer contents are refreshed via fill_()/copy_()
        in patched_execute_model before the graph is replayed.
        """
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

    def update_block_table(
        self,
        metadata: BeamAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> BeamAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping

        if metadata.std_req_idxs_cpu:
            if self._fixed_buffers:
                num_std = len(metadata.std_req_idxs_cpu)
                num_std_blocks = (
                    metadata.std_block_table.shape[1] if metadata.std_block_table is not None else 0
                )
                std_block_table = self._std_block_table_buf[:num_std, :num_std_blocks]
                std_block_table.copy_(blk_table[metadata.std_req_idxs_cpu, :num_std_blocks])
                new_metadata.std_block_table = std_block_table
            else:
                new_metadata.std_block_table = blk_table[metadata.std_req_idxs_cpu].contiguous()
        if metadata.prefix_req_idxs_cpu:
            persistent_binding = _persistent_prefix_binding()
            if self._fixed_buffers:
                num_prefix = len(metadata.prefix_req_idxs_cpu)
                # G1: full-width view (must match the view used at capture time
                # so the replayed graph reads the same tensor geometry).
                prefix_block_table = self._prefix_block_table_buf[:num_prefix]
                if persistent_binding is not None:
                    if num_prefix != 1 or persistent_binding.numel() < (
                        metadata.max_prefix_blocks_count
                    ):
                        raise RuntimeError("persistent GR prefix binding geometry mismatch")
                    prefix_block_table[0, : metadata.max_prefix_blocks_count].copy_(
                        persistent_binding[: metadata.max_prefix_blocks_count]
                    )
                else:
                    prefix_block_table[:, : metadata.max_prefix_blocks_count].copy_(
                        blk_table[
                            metadata.prefix_req_idxs_cpu,
                            : metadata.max_prefix_blocks_count,
                        ]
                    )
                new_metadata.prefix_block_table = prefix_block_table
            else:
                if persistent_binding is not None:
                    new_metadata.prefix_block_table = persistent_binding[
                        : metadata.max_prefix_blocks_count
                    ].unsqueeze(0)
                else:
                    new_metadata.prefix_block_table = blk_table[
                        metadata.prefix_req_idxs_cpu,
                        : metadata.max_prefix_blocks_count,
                    ].contiguous()

        return new_metadata
