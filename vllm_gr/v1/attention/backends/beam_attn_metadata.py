# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Beam Attention metadata and builder."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.beam.geometry import _compute_beam_bounds

logger = init_logger(__name__)


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
    prefix_length_bucket: int = 0
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

    # --- Pure-prefill fast path (prefill graph capture/replay) ---
    # When is_prefill_graph_exec is True the attention forward uses the
    # streamlined single-call prefill path that consumes only the common
    # fields (query_start_loc, seq_lens, block_table, slot_mapping); all the
    # grouped std_*/prefix_*/suffix_* fields stay None and must not be read.
    is_prefill_graph_exec: bool = False
    # Only dedicated padded captures use BSND; live/native metadata stays TND.
    prefill_fia_layout: str = "TND"
    # Graph-owned scratch, never shared across independent model runners.
    bsnd_resources: dict = field(default_factory=dict)
    num_reqs: int = 0
    query_start_loc_cpu: list | None = None
    seq_lens_cpu: list | None = None

    # --- Incremental decode fields ---
    # When is_beam_decode is True and beam_session is available,
    # the attention forward uses the incremental decode path.
    is_beam_decode: bool = False
    beam_active_mask: torch.Tensor | None = None
    beam_session: Any = None  # BeamSearchSession reference
    # DEPRECATED: beam_layer_idx is no longer the primary pool resolution
    # mechanism.  Prefer passing ``beam_pool`` directly as an explicit
    # parameter to ``cascade_attention``.  This field is kept only for
    # backward compatibility with callers that have not yet migrated.
    beam_layer_idx: int = 0
    beam_pool_slots: list = None  # Per beam-request slot index (for max_batch>1)
    incremental_suffix_kv_len: int = 0  # decode_step + 1
    # Host proof that the current NPU batch routes every row in order. Graph
    # dispatch must preserve this layout: replay does not re-run Python guards.
    npu_prefix_rows_are_identity: bool = False


class BeamAttentionMetadataBuilder(AttentionMetadataBuilder[BeamAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True

    @staticmethod
    def build_async_batch_layout(
        *,
        width: int,
        prompt_lengths: list[int],
        decode_step: int,
        block_ids_by_request: list[list[int]],
        max_blocks: int,
    ) -> dict[str, list[int]]:
        """Build request-major metadata for B logical requests and B*W rows."""
        batch = len(prompt_lengths)
        if (
            type(width) is not int
            or width < 1
            or not batch
            or len(block_ids_by_request) != batch
            or any(type(length) is not int or length < 1 for length in prompt_lengths)
        ):
            raise ValueError("Asynchronous scheduling batch geometry is invalid")
        query = [item * width for item in range(batch + 1)]
        rows = batch * width
        blocks: list[int] = []
        for request_blocks in block_ids_by_request:
            if len(request_blocks) > max_blocks:
                raise ValueError("Asynchronous scheduling prefix block table exceeds capacity")
            blocks.extend(request_blocks)
            blocks.extend([0] * (max_blocks - len(request_blocks)))
        suffix_length = decode_step + 1
        return {
            "positions": [prompt + decode_step for prompt in prompt_lengths for _ in range(width)],
            "query": query,
            "sequence": [prompt + suffix_length for prompt in prompt_lengths],
            "prefix_lengths": list(prompt_lengths),
            "blocks": blocks,
            "logits": list(range(rows)),
            "suffix_query": list(range(rows + 1)),
            "suffix_lengths": [suffix_length] * rows,
            "suffix_offsets": [item * suffix_length for item in range(rows + 1)],
        }

    @staticmethod
    def build_async_layout(
        *,
        width: int,
        prompt_len: int,
        decode_step: int,
        block_ids: list[int],
        max_blocks: int,
        query_start_loc_cpu: list[int] | tuple[int, ...],
        seq_lens_cpu: list[int] | tuple[int, ...],
    ) -> dict[str, list[int]]:
        """Complete CPU layout for B=1 incremental Decode, with no cache lookup.

        CPU mirrors are mandatory. This entry never falls back to device reads,
        geometry caches, cached views, cached schedules or the legacy uploader.
        Common attention has one packed request; suffix attention has W beams.
        """
        if any(
            not isinstance(values, (list, tuple)) or any(type(value) is not int for value in values)
            for values in (query_start_loc_cpu, seq_lens_cpu)
        ):
            raise ValueError(
                "Asynchronous scheduling attention requires explicit CPU query/sequence mirrors"
            )
        if list(query_start_loc_cpu) != [0, width] or list(seq_lens_cpu) != [
            prompt_len + decode_step + 1
        ]:
            raise ValueError(
                "Asynchronous scheduling common CPU mirrors disagree with the planned Decode"
            )
        if len(block_ids) > max_blocks:
            raise ValueError("Asynchronous scheduling prefix block table exceeds model capacity")
        return BeamAttentionMetadataBuilder.build_async_batch_layout(
            width=width,
            prompt_lengths=[prompt_len],
            decode_step=decode_step,
            block_ids_by_request=[block_ids],
            max_blocks=max_blocks,
        )

    @staticmethod
    def build_async(
        *,
        layout: dict[str, list[int]],
        views: dict[str, torch.Tensor],
        session: Any,
        mask: torch.Tensor,
        slots: torch.Tensor,
        block_size: int,
        max_model_len: int,
        num_heads: int,
        num_kv_heads: int,
    ) -> BeamAttentionMetadata:
        """Build a fresh metadata description over stable device storage each step.

        Native execution uses block-aligned model/pool bounds. A matching
        measured profile enables bounded prefix-length graph buckets.
        """
        rows = len(layout["positions"])
        batch = len(layout["prefix_lengths"])
        width = rows // batch
        blocks = views["blocks"].view(batch, -1)
        from vllm_gr.v1.attention.backends.beam_prefix_policy import (
            PrefixShape,
            available_prefix_versions,
            device_fingerprint,
            explicit_fa_version,
            multi_processor_count,
            policy_mode,
            prefix_bucket,
            resolve_prefix_plan,
        )

        bucket = 0
        if not envs.VLLM_BATCH_INVARIANT and policy_mode() != "off":
            candidate = prefix_bucket(max(layout["prefix_lengths"]), block_size, max_model_len)
            versions = available_prefix_versions(blocks.device.index)
            explicit = explicit_fa_version()
            plan = None
            if versions:
                plan = resolve_prefix_plan(
                    PrefixShape(
                        batch=batch,
                        width=width,
                        length=candidate,
                        heads=num_heads,
                        kv_heads=num_kv_heads,
                        dim=session.context.pools[0].unshared_key.shape[-1],
                        dtype=str(session.context.dtype),
                        block_size=block_size,
                    ),
                    # An explicit choice decides the backend; otherwise the best
                    # supported one (FA3/FA4) keeps its own split-K heuristic and
                    # does not need a length-scoped graph bound.
                    default_version=explicit if explicit is not None else max(versions),
                    fingerprint=device_fingerprint(blocks.device.index),
                    supported_versions=versions,
                    sm_count=multi_processor_count(blocks.device.index),
                    explicit_version=explicit,
                )
            if plan is not None and plan.segments > 1:
                pages = (candidate + block_size - 1) // block_size
                chunk = max(1, (pages + plan.segments - 1) // plan.segments)
                if chunk * plan.segments <= blocks.shape[1]:
                    bucket = candidate
        return BeamAttentionMetadata(
            num_actual_tokens=rows,
            num_reqs=batch,
            max_query_len=width,
            query_start_loc=views["query"],
            query_start_loc_cpu=list(layout["query"]),
            max_seq_len=max_model_len,
            seq_lens=views["sequence"],
            seq_lens_cpu=list(layout["sequence"]),
            block_table=blocks,
            slot_mapping=slots,
            is_beam_decode=True,
            beam_session=session,
            beam_active_mask=mask,
            prefix_indices=views["logits"],
            num_prefix_indices=rows,
            prefix_cu_seqlens_q_device=views["query"],
            prefix_seqlens_kv_device=views["prefix_lengths"],
            prefix_max_q_len=width,
            prefix_max_seq_len=bucket
            or ((max_model_len + block_size - 1) // block_size) * block_size,
            prefix_length_bucket=bucket,
            prefix_block_table=blocks,
            prefix_seqlens_q_cpu=[width] * batch,
            prefix_cu_seqlens_q_cpu=list(layout["query"]),
            prefix_seqlens_kv_cpu=list(layout["prefix_lengths"]),
            prefix_cu_seqlens_kv_cpu=None,
            suffix_cu_seqlens_q_device=views["suffix_query"],
            suffix_cu_seqlens_k_device=views["suffix_offsets"],
            suffix_seqlens_kv_device=views["suffix_lengths"],
            suffix_max_q_len=1,
            suffix_max_seq_len=session.context.max_decode_steps,
            suffix_seqlens_q_cpu=[1] * rows,
            suffix_cu_seqlens_q_cpu=list(layout["suffix_query"]),
            suffix_seqlens_kv_cpu=list(layout["suffix_lengths"]),
            suffix_cu_seqlens_kv_cpu=list(layout["suffix_offsets"]),
            suffix_total_tokens=layout["suffix_offsets"][-1],
            incremental_suffix_kv_len=layout["suffix_lengths"][0],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
        )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Metadata construction policy belongs to the selected backend, so this
        # module carries no device-name flags. The profile is a plain object
        # obtained once here; build() reads its answers without provider lookups.
        from vllm_gr.platforms.loader import active_backend_service

        self._profile = active_backend_service("create_beam_metadata_profile")
        # Default: no pre-allocated fixed-address buffers.  Overridden to True
        # below when the backend needs them for graph capture.
        # Tests / kernel harnesses that skip __init__ keep the default and
        # allocate device tensors per build() call.
        self._fixed_buffers = False
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        from vllm_gr.config import GRConfig

        self._v2_gr_prefill_capture = (
            self._profile.uses_v2_prefill_capture()
            and envs.VLLM_USE_V2_MODEL_RUNNER
            and GRConfig.from_vllm_config(vllm_config).beam_worker_decision
        )

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

        if self._profile.needs_causal_mask_buffer():
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
        self._fixed_buffers = self._profile.allocates_fixed_buffers()
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

        if self._profile.forces_single_reduction_split() and envs.VLLM_BATCH_INVARIANT:
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

    def _evict_stale_geometry_entries(self, geom_key) -> None:
        """Drop geometry-cache entries left behind by finished sessions.

        geom keys embed the request ids of their beam session.  When the ids
        of the CURRENT key share no request with a cached key, that entry
        belongs to a finished session (beam_finished_session_ids tore it down)
        and its host lists / device views / schedule will never be read again.
        Without eviction a long-running server accumulates one entry per
        historical request in each sub-cache (cu_lists, views, layout,
        schedule, block_table_copied), so the builder sweeps them whenever a
        NEW geometry is admitted.  ``empty_slot_out`` is keyed by device and
        is deliberately left alone.
        """
        if geom_key is None:
            return
        current_ids = {view[0] for view in geom_key[0]}
        cache = self._geometry_cache()
        for sub in ("cu_lists", "views", "layout", "schedule", "block_table_copied"):
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

    def _empty_slot_out(self, device) -> torch.Tensor:
        """Per-device cached empty slot-mapping tensor.

        Incremental decode never produces a suffix slot mapping (the pool
        slice replaces the gather), so every decode step would otherwise pay
        a fresh ``torch.empty(0)`` allocation for the same shapeless tensor.
        """
        cache = self._geometry_cache().setdefault("empty_slot_out", {})
        empty = cache.get(device)
        if empty is None:
            empty = torch.empty(0, dtype=torch.int32, device=device)
            cache[device] = empty
        return empty

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
        views_cache = self._geometry_cache().setdefault("views", {})
        cached = views_cache.get(geom_key)
        if cached is not None:
            return cached
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
        payloads are written to the address-stable buffers exactly once per
        ``geom_key``; later steps skip them.  The step-derived suffix arrays
        are rebuilt on device with ``fill_`` / ``arange`` from
        ``decode_step`` — no host list, no H2D.

        Returns True when the caller should skip the generic
        ``_pack_int32_to_dev`` path (geom_key admitted the cache), False to
        fall back to full construction.
        """
        if geom_key is None or decode_step is None:
            return False
        cache = self._geometry_cache()
        if not cache.get(geom_key, False):
            self._pack_int32_to_dev(
                [
                    (prefix_seqlens_kv_view, list(prefix_seqlens_kv_cpu)),
                    (prefix_cu_seqlens_q_view, [0] + list(prefix_cu_seqlens_q_cpu)),
                    (suffix_cu_seqlens_q_view, [0] + list(suffix_cu_seqlens_q_cpu)),
                ],
                non_blocking=self._profile.packs_seqlens_non_blocking(),
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
        geom_key=None,
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
        # Incremental decode emits ``prefix_indices`` as a single contiguous
        # ``range(q_start, q_start + w)`` per request, so the device-side
        # strided-arange fast path in ``_create_beam_tensors`` is always valid
        # and its per-element ``all()`` scan can be skipped.  The recompute
        # path merges arbitrary prefix groups and must keep the scan.
        incremental_only = True
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
            incremental_only = False
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
        suffix_cu_seqlens_q_cpu = None
        suffix_cu_seqlens_kv_cpu = None

        # Pure incremental decode keeps the three non-step cumulative sums
        # (prefix_cu_seqlens_q / prefix_cu_seqlens_kv / suffix_cu_seqlens_q)
        # fixed per geometry, so a geom_key hit reuses them instead of paying a
        # numpy cumsum + tolist on every decode step.  The step-dependent
        # suffix_cu_seqlens_kv is always recomputed below.
        fixed_cached = None
        if geom_key is not None:
            fixed_cached = self._geometry_cache().setdefault("cu_lists", {}).get(geom_key)
        if fixed_cached is not None:
            prefix_cu_seqlens_q_cpu, prefix_cu_seqlens_kv_cpu, suffix_cu_seqlens_q_cpu = (
                fixed_cached
            )
        else:
            if prefix_seqlens_q_cpu:
                prefix_cu_seqlens_q_cpu = np.cumsum(prefix_seqlens_q_cpu).tolist()
                prefix_cu_seqlens_kv_cpu = np.cumsum(prefix_seqlens_kv_cpu).tolist()
            if suffix_seqlens_q_cpu:
                suffix_cu_seqlens_q_cpu = np.cumsum(suffix_seqlens_q_cpu).tolist()
            if geom_key is not None and suffix_seqlens_kv_cpu:
                # Only cache when suffix_seqlens_kv is geometry-fixed (incremental
                # decode would make it step-dependent and cache it wrongly).
                cache = self._geometry_cache().setdefault("cu_lists", {})
                cache[geom_key] = (
                    prefix_cu_seqlens_q_cpu,
                    prefix_cu_seqlens_kv_cpu,
                    suffix_cu_seqlens_q_cpu,
                )
        if suffix_seqlens_kv_cpu:
            if incremental_only and beam_decode_step_list:
                # cumsum([s+1]*w) == [s+1, 2(s+1), ..., w(s+1)]: a strided
                # range materializes it directly, skipping the numpy cumsum
                # + tolist pair (the only step-dependent host list on the
                # hot decode path).
                s = beam_decode_step_list[0] + 1
                w_kv = len(suffix_seqlens_kv_cpu)
                suffix_cu_seqlens_kv_cpu = list(range(s, (w_kv + 1) * s, s))
            else:
                suffix_cu_seqlens_kv_cpu = np.cumsum(suffix_seqlens_kv_cpu).tolist()

        # Scalar reductions for ``_create_beam_tensors``.  The incremental
        # path knows them analytically (single prefix group of w tokens, one
        # query token per beam); the recompute path falls back to the list
        # max / contiguity scan (it is not the hot decode loop).
        if incremental_only:
            prefix_indices_contiguous = True
            prefix_max_q_len = prefix_seqlens_q_cpu[0] if prefix_seqlens_q_cpu else 0
            suffix_max_q_len = 1
            # suffix_seqlens_kv is [decode_step+1]*w on this path, so the max
            # is the step value itself (no list scan in _create_beam_tensors).
            suffix_max_seq_len = (
                beam_decode_step_list[0] + 1
                if beam_decode_step_list and suffix_seqlens_kv_cpu
                else 0
            )
        else:
            prefix_indices_contiguous = bool(beam_prefix_indices) and all(
                beam_prefix_indices[i] == beam_prefix_indices[0] + i
                for i in range(len(beam_prefix_indices))
            )
            prefix_max_q_len = max(prefix_seqlens_q_cpu) if prefix_seqlens_q_cpu else 0
            suffix_max_q_len = max(suffix_seqlens_q_cpu) if suffix_seqlens_q_cpu else 0
            suffix_max_seq_len = max(suffix_seqlens_kv_cpu) if suffix_seqlens_kv_cpu else 0

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
            prefix_indices_contiguous,
            prefix_max_q_len,
            suffix_max_q_len,
            suffix_max_seq_len,
        )

    def _pack_int32_to_dev(
        self,
        entries: list[tuple[Any, list[int]]],
        *,
        non_blocking: bool = False,
    ) -> None:
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
        if non_blocking:
            # A first NPU Beam geometry must not drain the preceding Prefill
            # and TopK before the rest of Decode preparation can be submitted.
            # Do not reuse _h2d_host here: a following request could overwrite
            # it before this asynchronous H2D has consumed the payload. The
            # pinned allocator tracks the copy and delays recycling this
            # independent allocation until its transfer has completed.
            host = torch.empty(total, dtype=self._h2d_host.dtype, pin_memory=True)
        else:
            host = self._h2d_host[:total]
        host.copy_(torch.from_numpy(packed))
        # Single host -> device transfer for the whole payload.
        self._h2d_dev[:total].copy_(host, non_blocking=non_blocking)

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
            if self._profile.writes_seqlens_per_tensor():
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
        # Scalar reductions may arrive precomputed from _build_beam_mappings
        # (the incremental path knows them analytically); fall back to the
        # list max for callers that only pass the raw lists.
        if prefix_max_q_len is None:
            prefix_max_q_len = max(prefix_seqlens_q_cpu) if prefix_seqlens_q_cpu else 0
        if suffix_max_q_len is None:
            suffix_max_q_len = max(suffix_seqlens_q_cpu) if suffix_seqlens_q_cpu else 0
        if suffix_max_seq_len is None:
            suffix_max_seq_len = max(suffix_seqlens_kv_cpu) if suffix_seqlens_kv_cpu else 0

        if self._fixed_buffers:
            prefix_indices_tensor = self._prefix_indices_buf[:num_prefix]
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
            prefix_max_seq_len = self._prefix_block_table_buf.shape[1] * self.block_size
            prefix_block_table = self._prefix_block_table_buf[: len(prefix_req_idxs_cpu)]
            # The prefix block rows are frozen for the lifetime of an
            # incremental beam session (decode KV goes to the beam pool, not
            # the paged cache), so on a geometry hit the staged copy is
            # already in the address-stable buffer and can be skipped.
            copied = self._geometry_cache().setdefault("block_table_copied", set())
            if geom_key is None or geom_key not in copied:
                source = common_attn_metadata.block_table_tensor
                if (
                    self._profile.allows_identity_block_table_slice()
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
                suffix_slot_mapping_out = self._empty_slot_out(device)
        else:
            suffix_total_tokens = 0
            suffix_slot_mapping_out = self._empty_slot_out(device)

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
                self._pack_int32_to_dev(
                    [
                        (prefix_seqlens_kv_device, list(prefix_seqlens_kv_cpu)),
                        (prefix_cu_seqlens_q_device, [0] + list(prefix_cu_seqlens_q_cpu)),
                        (suffix_seqlens_kv_device, list(suffix_seqlens_kv_cpu)),
                        (suffix_cu_seqlens_q_device, [0] + list(suffix_cu_seqlens_q_cpu)),
                        (suffix_cu_seqlens_k_device, [0] + list(suffix_cu_seqlens_kv_cpu)),
                    ]
                )
        elif self._profile.writes_seqlens_per_tensor():
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
        schedule_cache = self._geometry_cache().setdefault("schedule", {})
        if geom_key is not None and geom_key in schedule_cache:
            prefix_scheduler_metadata = schedule_cache[geom_key]
        else:
            prefix_scheduler_metadata = self._get_schedule(
                batch_size=len(prefix_seqlens_q_cpu),
                cu_query_lens=prefix_cu_seqlens_q_device,
                max_query_len=prefix_max_q_len,
                seqlens=prefix_seqlens_kv_device,
                max_seq_len=prefix_max_seq_len,
                causal=False,
                max_num_splits=max_num_splits,
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
        # Preserve V1's exact CPU mirror. Native V2 supplies an upper bound
        # instead, which is exact without speculative decoding (including
        # cached/chunked Prefill). Legacy Beam rows retain their own lengths.
        seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        if seq_lens_cpu is None or seq_lens_cpu.numel() < num_reqs:
            upper_bound = getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
            if (
                upper_bound is not None
                and self.vllm_config.speculative_config is None
                and not any(m and m.get("is_beam_decode", False) for m in beam_data.values())
            ):
                seq_lens_cpu = upper_bound
        if seq_lens_cpu is not None and seq_lens_cpu.numel() >= num_reqs:
            seq_lens_cpu_list = seq_lens_cpu[:num_reqs].tolist()
        else:
            seq_lens_cpu_list = common_attn_metadata.seq_lens.cpu().tolist()

        # Get beam_session early so _segment_requests can use it for slot lookup
        beam_session = beam_info.get("beam_session") if beam_info else None

        # Check if we have an active beam session for incremental decode.
        # Derived directly from the SCHEDULED requests' beam_data (not from
        # _segment_requests output) so the geometry-hit fast path below can
        # skip segmentation entirely.
        is_incremental = (
            beam_session is not None
            and beam_session.is_active
            and any(
                (m := beam_data.get(rid)) is not None
                and m.get("is_beam_decode", False)
                and "decode_step" in m
                for rid in req_ids
            )
        )

        geom_key = self._geom_cache_key(
            num_reqs, req_ids, beam_data, q_starts, seq_lens_cpu_list, is_incremental
        )
        # Sweep entries left behind by finished sessions: keys whose request
        # ids no longer overlap the current batch will never be read again.
        self._evict_stale_geometry_entries(geom_key)

        # Geometry-hit fast path: for a pure incremental beam batch the whole
        # mapping layout (segmentation, std partition — always empty — and the
        # beam mapping lists) is a per-geometry constant.  Only three things
        # vary per step: the suffix step value s=decode_step+1 (analytic
        # lists below), the query-start offsets (rebuilt from q_starts), and
        # the device-side refresh in _create_beam_tensors.  A hit skips
        # _segment_requests / _build_std_mappings / _build_beam_mappings
        # entirely: this legacy Decode path reuses the layout and only updates
        # step-dependent scalars. The asynchronous build_async entrypoint does
        # not use these caches.
        geom_layout_cache = self._geometry_cache().setdefault("layout", {})
        cached_layout = geom_layout_cache.get(geom_key) if geom_key is not None else None
        if cached_layout is not None:
            (
                beam_req_idxs,
                beam_prefix_lens,
                beam_widths,
                beam_decode_steps,
                beam_cache_lens,
                beam_q_start_offsets,
                beam_pool_slots,
                prefix_seqlens_q_cpu,
                prefix_seqlens_kv_cpu,
                suffix_seqlens_q_cpu,
                prefix_req_idxs_cpu,
                prefix_cu_seqlens_q_cpu,
                prefix_cu_seqlens_kv_cpu,
                suffix_cu_seqlens_q_cpu,
                max_prefix_blocks_count,
                prefix_indices_contiguous,
                prefix_max_q_len,
                suffix_max_q_len,
                w_total,
            ) = cached_layout
            # The CURRENT decode_step comes from this frame's beam_data (the
            # cached layout only pins the geometry, never the step).  Rebuild
            # the per-request step list for the cached beam requests.
            beam_decode_step_list = [
                beam_data[req_ids[beam_req_idxs[i]]]["decode_step"]
                for i in range(len(beam_req_idxs))
            ]
            decode_step = beam_decode_step_list[0] if beam_decode_step_list else None
            # Step-derived suffix lists (analytic; no numpy, no per-beam loop).
            s = decode_step + 1 if decode_step is not None else 1
            suffix_seqlens_kv_cpu = [s] * w_total
            suffix_cu_seqlens_kv_cpu = list(range(s, (w_total + 1) * s, s))
            suffix_max_seq_len = s if w_total else 0
            # Query-start-dependent prefix indices (one range per request).
            prefix_indices = []
            for _b_idx, _req_idx in enumerate(beam_req_idxs):
                q_offset = beam_q_start_offsets[_b_idx]
                w = beam_widths[_b_idx]
                q_start = q_starts[_req_idx] + q_offset
                prefix_indices.extend(range(q_start, q_start + w))
            std_indices = []
            std_seqlens_q_cpu = []
            std_seqlens_kv_cpu = []
            std_req_idxs = []
            std_cu_seqlens_q_cpu = None
            std_cu_seqlens_kv_cpu = None
            suffix_slot_mapping_out_list = []
        else:
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
            decode_step = beam_decode_step_list[0] if beam_decode_step_list else None

            # Fast-path for pure prefill: skip grouped beam metadata computation.
            if not beam_req_idxs:
                logger.debug(
                    "BeamAttentionMetadataBuilder: Pure prefill batch detected "
                    "(%d actual tokens). Skipping grouped beam metadata build.",
                    common_attn_metadata.num_actual_tokens,
                )
                return BeamAttentionMetadata(
                    num_actual_tokens=common_attn_metadata.num_actual_tokens,
                    max_query_len=common_attn_metadata.max_query_len,
                    query_start_loc=common_attn_metadata.query_start_loc,
                    max_seq_len=common_attn_metadata.max_seq_len,
                    seq_lens=common_attn_metadata.seq_lens,
                    block_table=common_attn_metadata.block_table_tensor,
                    slot_mapping=common_attn_metadata.slot_mapping,
                    causal=True,
                    npu_causal_atten_mask=self.npu_causal_atten_mask,
                    is_prefill_graph_exec=True,
                    num_reqs=common_attn_metadata.num_reqs,
                    num_heads=self.num_heads_q,
                    num_kv_heads=self.num_kv_heads,
                    block_size=self.block_size,
                    query_start_loc_cpu=q_starts,
                    seq_lens_cpu=seq_lens_cpu_list,
                )

            (
                std_indices,
                std_seqlens_q_cpu,
                std_seqlens_kv_cpu,
                std_req_idxs,
                std_cu_seqlens_q_cpu,
                std_cu_seqlens_kv_cpu,
            ) = self._build_std_mappings(std_reqs, q_starts)

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
                prefix_indices_contiguous,
                prefix_max_q_len,
                suffix_max_q_len,
                suffix_max_seq_len,
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
                geom_key=geom_key,
            )
            # Register the layout for the geometry-hit fast path.  Only the
            # pure incremental case is eligible (mixed/recompute batches keep
            # the per-frame reconstruction).
            if geom_key is not None and is_incremental and not std_reqs:
                geom_layout_cache.clear()
                geom_layout_cache[geom_key] = (
                    beam_req_idxs,
                    beam_prefix_lens,
                    beam_widths,
                    beam_decode_steps,
                    beam_cache_lens,
                    beam_q_start_offsets,
                    beam_pool_slots,
                    prefix_seqlens_q_cpu,
                    prefix_seqlens_kv_cpu,
                    suffix_seqlens_q_cpu,
                    prefix_req_idxs_cpu,
                    prefix_cu_seqlens_q_cpu,
                    prefix_cu_seqlens_kv_cpu,
                    suffix_cu_seqlens_q_cpu,
                    max_prefix_blocks_count,
                    prefix_indices_contiguous,
                    prefix_max_q_len,
                    suffix_max_q_len,
                    len(suffix_seqlens_q_cpu),
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
            prefix_indices_contiguous=prefix_indices_contiguous,
            prefix_max_q_len=prefix_max_q_len,
            suffix_max_q_len=suffix_max_q_len,
            suffix_max_seq_len=suffix_max_seq_len,
        )

        # Determine incremental decode metadata
        incremental_suffix_kv_len = 0
        if is_incremental and beam_decode_step_list:
            # In incremental decode, suffix KV length = decode_step + 1
            incremental_suffix_kv_len = beam_decode_step_list[0] + 1

        # Use the current CPU routing, including geometry-hit q_start updates.
        # Counts or contiguity alone do not prove a zero-based identity range.
        npu_prefix_rows_are_identity = (
            self._profile.marks_identity_prefix_rows()
            and is_incremental
            and not std_indices
            and len(prefix_indices) > 0
            and len(prefix_indices) == common_attn_metadata.num_actual_tokens
            and len(prefix_indices) == getattr(beam_session, "attention_width", 0)
            and all(index == row for row, index in enumerate(prefix_indices))
        )

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
            npu_prefix_rows_are_identity=npu_prefix_rows_are_identity,
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
        beam_info = BEAM_DATA_VAR.get()
        beam_data = beam_info.get("beam_data", {}) if beam_info else {}
        if getattr(self, "_v2_gr_prefill_capture", False) and not any(
            row and row.get("is_beam_decode", False) for row in beam_data.values()
        ):
            # Native dummy requests split T nearly evenly. Real Prefills can
            # put nearly all T tokens in one member and can reuse long cached
            # prefixes. Set bounds before build creates dependent metadata.
            common_attn_metadata = replace(
                common_attn_metadata,
                max_query_len=min(
                    common_attn_metadata.num_actual_tokens, self.model_config.max_model_len
                ),
                max_seq_len=self.model_config.max_model_len,
            )
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
            if self._fixed_buffers:
                num_prefix = len(metadata.prefix_req_idxs_cpu)
                # G1: full-width view (must match the view used at capture time
                # so the replayed graph reads the same tensor geometry).
                prefix_block_table = self._prefix_block_table_buf[:num_prefix]
                prefix_block_table[:, : metadata.max_prefix_blocks_count].copy_(
                    blk_table[metadata.prefix_req_idxs_cpu, : metadata.max_prefix_blocks_count]
                )
                new_metadata.prefix_block_table = prefix_block_table
            else:
                new_metadata.prefix_block_table = blk_table[
                    metadata.prefix_req_idxs_cpu, : metadata.max_prefix_blocks_count
                ].contiguous()

        return new_metadata
