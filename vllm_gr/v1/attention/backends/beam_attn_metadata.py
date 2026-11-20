# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Beam Attention metadata and builder."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

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

from vllm_gr.config import GRConfig
from vllm_gr.config.models import DEFAULT_DEPLOYMENT
from vllm_gr.v1.attention.backends.beam_attn_metadata_layout import (
    _build_beam_mappings,
    _build_std_mappings,
    _evict_stale_geometry_entries,
    _geom_cache_key,
    _segment_requests,
)
from vllm_gr.v1.attention.backends.beam_attn_metadata_tensors import (
    BeamTensorBuffers,
    BeamTensorContext,
    allocate_tensor_buffers,
    create_beam_tensors,
    create_std_tensors,
)
from vllm_gr.v1.attention.backends.beam_prefix_policy import configured_fa_version
from vllm_gr.v1.beam import BEAM_DATA_VAR

logger = init_logger(__name__)


def get_scheduler_metadata(*args, **kwargs):
    from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available

    if not is_flash_attn_varlen_func_available():
        raise RuntimeError("BeamAttentionMetadataBuilder requires FlashAttention scheduler support")
    from vllm.v1.attention.backends.fa_utils import (
        get_scheduler_metadata as _get_scheduler_metadata,
    )

    return _get_scheduler_metadata(*args, **kwargs)


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

    # Per-beam incremental suffix lengths
    suffix_cu_seqlens_q_device: torch.Tensor | None = None
    suffix_cu_seqlens_k_device: torch.Tensor | None = None
    suffix_seqlens_kv_device: torch.Tensor | None = None
    suffix_max_q_len: int = 0
    suffix_max_seq_len: int = 0
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
    # Beam decode requires an active session and persistent suffix KV pool.
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
    prefix_policy: str = DEFAULT_DEPLOYMENT.prefix_policy
    prefix_min_chain_us: float = DEFAULT_DEPLOYMENT.prefix_min_chain_us
    prefix_calibrations: dict | None = None
    explicit_fa_version: int | None = None


def require_beam_decode_session(metadata: BeamAttentionMetadata) -> Any:
    """Reject obsolete beam metadata before any attention/cache kernel runs."""
    session = getattr(metadata, "beam_session", None)
    if not getattr(metadata, "is_beam_decode", False) or session is None or not session.is_active:
        raise RuntimeError("Beam decode requires an active beam session and KV pool")
    if metadata.prefix_indices is not None and metadata.suffix_max_q_len != 1:
        raise ValueError("Beam decode requires exactly one query token per beam")
    return session


def resolve_beam_pool(metadata: BeamAttentionMetadata, beam_pool: Any = None) -> Any:
    """Resolve the explicit layer pool, or the session pool for direct callers."""
    session = require_beam_decode_session(metadata)
    if beam_pool is not None:
        return beam_pool
    layer_idx = getattr(metadata, "beam_layer_idx", 0)
    pools = getattr(session, "pools", ())
    if not 0 <= layer_idx < len(pools) or pools[layer_idx] is None:
        raise RuntimeError("Beam decode requires a KV pool for the attention layer")
    return pools[layer_idx]


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

        self._v2_gr_prefill_capture = (
            self._is_cuda
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

        if self._is_npu:
            self.npu_causal_atten_mask = (
                torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(self.device)
            )
        else:
            self.npu_causal_atten_mask = None

        # --- Fixed-address metadata buffers for graph capture compatibility ---
        # CUDA and NPU pre-allocate maximum-sized buffers so build() can return
        # address-stable views for graph replay.  Thread the buffers through a
        # BeamTensorContext instead of reading them individually in build().
        self._fixed_buffers = self._is_cuda or self._is_npu
        self._tensor_buffers = BeamTensorBuffers()
        if self._fixed_buffers:
            max_beams = GRConfig.from_vllm_config(vllm_config).beam_max_width
            if max_beams is None:
                self._fixed_buffers = False
            else:
                self._tensor_buffers = allocate_tensor_buffers(
                    device=self.device,
                    block_size=self.block_size,
                    max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
                    max_num_batched_tokens=(vllm_config.scheduler_config.max_num_batched_tokens),
                    max_model_len=vllm_config.model_config.max_model_len,
                    max_beams=max_beams,
                )

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

    def _geometry_cache(self) -> dict:
        cache = getattr(self, "_geom_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_geom_cache", cache)
        return cache

    def _tensor_context(self) -> BeamTensorContext:
        cache_config = getattr(self, "cache_config", None)
        return BeamTensorContext(
            device=getattr(self, "device", torch.device("cpu")),
            block_size=self.block_size,
            fixed_buffers=getattr(self, "_fixed_buffers", False),
            is_cuda=getattr(self, "_is_cuda", False),
            is_npu=getattr(self, "_is_npu", False),
            cache=self._geometry_cache(),
            buffers=(getattr(self, "_tensor_buffers", None) or BeamTensorBuffers()),
            num_heads_q=getattr(self, "num_heads_q", 0),
            num_kv_heads=getattr(self, "num_kv_heads", 0),
            headdim=getattr(self, "headdim", 0),
            kv_cache_dtype=getattr(self, "kv_cache_dtype", torch.float16),
            aot_schedule=getattr(self, "aot_schedule", False),
            aot_sliding_window=getattr(self, "aot_sliding_window", None),
            cache_dtype=getattr(cache_config, "cache_dtype", "auto"),
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

        # Validate scheduled beam requests before either metadata construction or
        # a geometry-cache hit. Old suffix-recompute metadata is not supported.
        is_incremental = False
        for index, req_id in enumerate(req_ids[:num_reqs]):
            mdata = beam_data.get(req_id)
            if not mdata or not mdata.get("is_beam_decode", False):
                continue
            is_incremental = True
            if beam_session is None or not beam_session.is_active:
                raise RuntimeError("Beam decode requires an active beam session and KV pool")
            step = mdata.get("decode_step")
            width = mdata.get("beam_width")
            if type(step) is not int or step < 0:
                raise ValueError("Beam decode requires a nonnegative integer decode_step")
            if type(width) is not int or width < 1:
                raise ValueError("Beam decode requires a positive integer beam_width")
            # Pure Decode must fit its physical rows even when a cached layout
            # would otherwise bypass _build_beam_mappings. Prefix chunks are
            # segmented separately below, preserving their standard rows.
            if (
                mdata["cache_len"] >= mdata["prefix_len"]
                and q_starts[index + 1] - q_starts[index] != width
            ):
                raise ValueError("Beam decode requires exactly one query token per beam")

        geom_key = _geom_cache_key(
            num_reqs,
            req_ids,
            beam_data,
            q_starts,
            seq_lens_cpu_list,
            is_incremental,
            self.block_size,
        )
        if (
            is_incremental
            and len(
                {
                    beam_data[rid]["decode_step"]
                    for rid in req_ids[:num_reqs]
                    if beam_data.get(rid, {}).get("is_beam_decode", False)
                }
            )
            > 1
        ):
            geom_key = None
        # Sweep entries left behind by finished sessions: keys whose request
        # ids no longer overlap the current batch will never be read again.
        _evict_stale_geometry_entries(geom_key, self._geometry_cache())

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
        else:
            (
                std_reqs,
                beam_req_idxs,
                beam_prefix_lens,
                beam_widths,
                beam_q_start_offsets,
                beam_decode_step_list,
                beam_pool_slots,
            ) = _segment_requests(
                num_reqs,
                req_ids,
                beam_data,
                q_starts,
                seq_lens_cpu_list,
                beam_session,
                block_size=self.block_size,
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
            ) = _build_std_mappings(std_reqs, q_starts)

            mappings = _build_beam_mappings(
                beam_req_idxs,
                beam_prefix_lens,
                beam_widths,
                beam_q_start_offsets,
                q_starts,
                beam_decode_step_list,
                geom_key=geom_key,
                block_size=self.block_size,
                cache=self._geometry_cache(),
            )
            prefix_indices = mappings.prefix_indices
            prefix_seqlens_q_cpu = mappings.prefix_seqlens_q
            prefix_seqlens_kv_cpu = mappings.prefix_seqlens_kv
            suffix_seqlens_q_cpu = mappings.suffix_seqlens_q
            suffix_seqlens_kv_cpu = mappings.suffix_seqlens_kv
            prefix_req_idxs_cpu = mappings.prefix_req_idxs
            prefix_cu_seqlens_q_cpu = mappings.prefix_cu_seqlens_q
            prefix_cu_seqlens_kv_cpu = mappings.prefix_cu_seqlens_kv
            suffix_cu_seqlens_q_cpu = mappings.suffix_cu_seqlens_q
            suffix_cu_seqlens_kv_cpu = mappings.suffix_cu_seqlens_kv
            max_prefix_blocks_count = mappings.max_prefix_blocks_count
            prefix_indices_contiguous = mappings.prefix_indices_contiguous
            prefix_max_q_len = mappings.prefix_max_q_len
            suffix_max_q_len = mappings.suffix_max_q_len
            suffix_max_seq_len = mappings.suffix_max_seq_len
            # Register the layout for the geometry-hit fast path.  Only the
            # pure incremental case is eligible (mixed batches keep
            # the per-frame reconstruction).
            if geom_key is not None and is_incremental and not std_reqs:
                geom_layout_cache.clear()
                geom_layout_cache[geom_key] = (
                    beam_req_idxs,
                    beam_prefix_lens,
                    beam_widths,
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

        tensor_context = self._tensor_context()
        (
            std_indices_tensor,
            std_cu_seqlens_q_device,
            std_seqlens_kv_device,
            std_max_q_len,
            std_max_seq_len,
            std_block_table,
        ) = create_std_tensors(
            tensor_context,
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
            suffix_max_seq_len,
            suffix_cu_seqlens_q_device,
            suffix_cu_seqlens_k_device,
            suffix_max_q_len,
            prefix_scheduler_metadata,
        ) = create_beam_tensors(
            tensor_context,
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
            max_num_splits,
            geom_key=geom_key,
            decode_step=decode_step,
            prefix_indices_contiguous=prefix_indices_contiguous,
            prefix_max_q_len=prefix_max_q_len,
            suffix_max_q_len=suffix_max_q_len,
            suffix_max_seq_len=suffix_max_seq_len,
            scheduler=get_scheduler_metadata,
        )

        # Determine incremental decode metadata
        incremental_suffix_kv_len = 0
        if is_incremental and beam_decode_step_list:
            # In incremental decode, suffix KV length = decode_step + 1
            incremental_suffix_kv_len = beam_decode_step_list[0] + 1

        # Use the current CPU routing, including geometry-hit q_start updates.
        # Counts or contiguity alone do not prove a zero-based identity range.
        npu_prefix_rows_are_identity = (
            getattr(self, "_is_npu", False)
            and is_incremental
            and not std_indices
            and len(prefix_indices) > 0
            and len(prefix_indices) == common_attn_metadata.num_actual_tokens
            and len(prefix_indices) == getattr(beam_session, "attention_width", 0)
            and all(index == row for row, index in enumerate(prefix_indices))
        )

        deployment = GRConfig.from_vllm_config(self.vllm_config).deployment
        meta = BeamAttentionMetadata(
            explicit_fa_version=configured_fa_version(self.vllm_config),
            prefix_policy=deployment.prefix_policy,
            prefix_min_chain_us=deployment.prefix_min_chain_us,
            prefix_calibrations=getattr(self.vllm_config, "_gr_prefix_calibrations", None),
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
                std_block_table = self._tensor_buffers.std_block_table[:num_std, :num_std_blocks]
                std_block_table.copy_(blk_table[metadata.std_req_idxs_cpu, :num_std_blocks])
                new_metadata.std_block_table = std_block_table
            else:
                new_metadata.std_block_table = blk_table[metadata.std_req_idxs_cpu].contiguous()
        if metadata.prefix_req_idxs_cpu:
            if self._fixed_buffers:
                num_prefix = len(metadata.prefix_req_idxs_cpu)
                # G1: full-width view (must match the view used at capture time
                # so the replayed graph reads the same tensor geometry).
                prefix_block_table = self._tensor_buffers.prefix_block_table[:num_prefix]
                prefix_block_table[:, : metadata.max_prefix_blocks_count].copy_(
                    blk_table[metadata.prefix_req_idxs_cpu, : metadata.max_prefix_blocks_count]
                )
                new_metadata.prefix_block_table = prefix_block_table
            else:
                new_metadata.prefix_block_table = blk_table[
                    metadata.prefix_req_idxs_cpu, : metadata.max_prefix_blocks_count
                ].contiguous()

        return new_metadata
