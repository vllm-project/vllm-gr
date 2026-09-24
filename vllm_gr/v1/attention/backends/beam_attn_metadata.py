# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam Attention metadata and builder."""

from __future__ import annotations

import copy
from dataclasses import dataclass
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
    # Sparse coordinates behind suffix_slot_mapping_out (see
    # _project_suffix_slot_mapping), persisted so update_block_table can
    # re-project it against a new block table/slot_mapping instead of
    # silently reusing the stale mapping from the original build().
    suffix_past_coords: torch.Tensor | None = None
    suffix_curr_coords: torch.Tensor | None = None
    # CPU-side maxima over the coordinate lists above (see build()), so
    # update_block_table's geometry validation can compare against them
    # without a device sync.
    suffix_max_past_req: int | None = None
    suffix_max_past_blk: int | None = None
    suffix_max_q_idx: int | None = None

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
    num_reqs: int = 0
    query_start_loc_cpu: list | None = None
    seq_lens_cpu: list | None = None


class BeamAttentionMetadataBuilder(AttentionMetadataBuilder[BeamAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True
    _fixed_buffers: bool = False

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

    def _empty_slot_out(self, device) -> torch.Tensor:
        """Per-device cached empty slot-mapping tensor."""
        cache = getattr(self, "_empty_slot_cache", None)
        if cache is None:
            cache = {}
            self._empty_slot_cache = cache
        empty = cache.get(device)
        if empty is None:
            empty = torch.empty(0, dtype=torch.int32, device=device)
            cache[device] = empty
        return empty

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
    ):
        std_reqs = []
        beam_req_idxs = []
        beam_prefix_lens = []
        beam_widths = []
        beam_decode_steps = []
        beam_cache_lens = []
        beam_q_start_offsets = []

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
    ):
        """Builds attention mapping metadata for all active beams in a batch.

        This method decomposes the query token allocations (chunk budgets) for
        each request into prefix (shared history) and suffix (beam-specific lookahead)
        computations.
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

        for req_idx, prefix_len, w, steps, cache_len, q_offset in zip(
            beam_req_idxs,
            beam_prefix_lens,
            beam_widths,
            beam_decode_steps,
            beam_cache_lens,
            beam_q_start_offsets,
        ):
            q_start = q_starts[req_idx] + q_offset
            q_end = q_starts[req_idx + 1]
            q_len = q_end - q_start
            if w == 0 or steps == 0 or q_len <= 0:
                continue

            q_curr = q_start
            chunk_budget = q_len
            prefix_groups = []  # Elements: [group_q_start, group_q_len, group_cache_len]

            b_prefix_len = min(cache_len, prefix_len)
            b_cache_len = cache_len
            # Loop 1: Partition query tokens across the w active beams for this request
            for b in range(w):
                if chunk_budget <= 0:
                    break

                b_uncached, b_suffix_len, b_past_suffix, b_physical = _compute_beam_bounds(
                    b, prefix_len, b_cache_len, steps, chunk_budget
                )

                # Positions carried over from a prior decode step (already
                # cached) are tagged with their owning req_idx so
                # _create_beam_tensors can resolve them through the
                # persistent block table; freshly computed positions are
                # tagged -1 and resolve directly through this step's
                # slot_mapping.
                if b_past_suffix > 0:
                    if prefix_len >= b_cache_len:
                        b_suffix_start = b_cache_len
                    else:
                        b_suffix_start = prefix_len + b_physical * steps
                    beam_suffix_slot_mapping_out_list.extend(
                        (req_idx, b_suffix_start + i) for i in range(b_past_suffix)
                    )
                beam_suffix_slot_mapping_out_list.extend(
                    (-1, q_curr + i) for i in range(b_uncached)
                )

                # Invariant: Merge adjacent beams that share the exact same prefix length
                # into a single prefix group to parallelize key/value lookup during attention.
                if prefix_groups and prefix_groups[-1][2] == b_prefix_len:
                    prefix_groups[-1][1] += b_uncached
                else:
                    prefix_groups.append([q_curr, b_uncached, b_prefix_len])

                suffix_seqlens_q_cpu.append(b_uncached)
                suffix_seqlens_kv_cpu.append(b_suffix_len)

                # Advance state for the next beam
                b_cache_len += b_uncached
                q_curr += b_uncached
                chunk_budget -= b_uncached
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

        if prefix_seqlens_q_cpu:
            prefix_cu_seqlens_q_cpu = np.cumsum(prefix_seqlens_q_cpu).tolist()
            prefix_cu_seqlens_kv_cpu = np.cumsum(prefix_seqlens_kv_cpu).tolist()
        if suffix_seqlens_q_cpu:
            suffix_cu_seqlens_q_cpu = np.cumsum(suffix_seqlens_q_cpu).tolist()
        if suffix_seqlens_kv_cpu:
            suffix_cu_seqlens_kv_cpu = np.cumsum(suffix_seqlens_kv_cpu).tolist()

        # Scalar reductions for ``_create_beam_tensors``.
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
            ) = suffix_past_coords = suffix_curr_coords = None
            suffix_max_past_req = suffix_max_past_blk = suffix_max_q_idx = None
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
                suffix_past_coords,
                suffix_curr_coords,
                suffix_max_past_req,
                suffix_max_past_blk,
                suffix_max_q_idx,
            )

        num_prefix = len(prefix_indices)
        # Scalar reductions may arrive precomputed from _build_beam_mappings;
        # fall back to the list max for callers that only pass the raw lists.
        if prefix_max_q_len is None:
            prefix_max_q_len = max(prefix_seqlens_q_cpu) if prefix_seqlens_q_cpu else 0
        if suffix_max_q_len is None:
            suffix_max_q_len = max(suffix_seqlens_q_cpu) if suffix_seqlens_q_cpu else 0
        if suffix_max_seq_len is None:
            suffix_max_seq_len = max(suffix_seqlens_kv_cpu) if suffix_seqlens_kv_cpu else 0

        if self._fixed_buffers:
            prefix_indices_tensor = self._prefix_indices_buf[:num_prefix]
            # When the caller's ``prefix_indices`` form one contiguous run,
            # emit it on device with a strided arange instead of converting
            # the Python list every step (no host list, no H2D).
            # ``prefix_indices_contiguous`` is the EXPLICIT verdict from
            # _build_beam_mappings when supplied; callers that do not supply
            # it fall back to the per-element scan, and anything
            # non-contiguous falls back to the host list transfer.
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
            prefix_block_table[:, :max_prefix_blocks_count].copy_(
                common_attn_metadata.block_table_tensor[
                    prefix_req_idxs_cpu, :max_prefix_blocks_count
                ]
            )
        else:
            prefix_indices_tensor = torch.tensor(prefix_indices, dtype=torch.long, device=device)
            prefix_max_seq_len = max(prefix_seqlens_kv_cpu) if prefix_seqlens_kv_cpu else 0
            prefix_block_table = common_attn_metadata.block_table_tensor[
                prefix_req_idxs_cpu, :max_prefix_blocks_count
            ].contiguous()

        if suffix_slot_mapping_out_list:
            suffix_total_tokens = len(suffix_slot_mapping_out_list)
            block_size = self.block_size
            # Split past-vs-current on the CPU-side Python list (cheap: no
            # kernel launch) rather than via a GPU boolean mask + compaction,
            # since b_past_suffix > 0 (mixed past/current) is the common case
            # for every decode step after the first, not a rare edge case.
            past_coords_list = []
            curr_coords_list = []
            for idx, (tag, pos) in enumerate(suffix_slot_mapping_out_list):
                if tag == -1:
                    curr_coords_list.append((idx, pos))
                else:
                    past_coords_list.append((idx, tag, pos // block_size, pos % block_size))
            suffix_past_coords = (
                torch.tensor(past_coords_list, dtype=torch.long, device=device)
                if past_coords_list
                else None
            )
            suffix_curr_coords = (
                torch.tensor(curr_coords_list, dtype=torch.long, device=device)
                if curr_coords_list
                else None
            )
            # Track these maxima on the CPU now, while the plain Python
            # coordinate lists are already in hand, so update_block_table's
            # geometry validation can compare against them directly instead
            # of paying a device sync (.max().item()) on its hot path.
            suffix_max_past_req = max((c[1] for c in past_coords_list), default=None)
            suffix_max_past_blk = max((c[2] for c in past_coords_list), default=None)
            suffix_max_q_idx = max((c[1] for c in curr_coords_list), default=None)
            suffix_slot_mapping_out = self._project_suffix_slot_mapping(
                common_attn_metadata.slot_mapping,
                common_attn_metadata.block_table_tensor,
                block_size,
                suffix_total_tokens,
                suffix_past_coords,
                suffix_curr_coords,
                device,
            )
        else:
            suffix_total_tokens = 0
            suffix_slot_mapping_out = self._empty_slot_out(device)
            suffix_past_coords = None
            suffix_curr_coords = None
            suffix_max_past_req = None
            suffix_max_past_blk = None
            suffix_max_q_idx = None

        if self._fixed_buffers:
            num_prefix_groups = len(prefix_seqlens_kv_cpu)
            num_prefix_cu = len(prefix_cu_seqlens_q_cpu) + 1
            num_suffix_beams = len(suffix_seqlens_kv_cpu)
            num_suffix_cu_q = len(suffix_cu_seqlens_q_cpu) + 1
            num_suffix_cu_k = len(suffix_cu_seqlens_kv_cpu) + 1

            prefix_seqlens_kv_device = self._prefix_seqlens_kv_buf[:num_prefix_groups]
            prefix_cu_seqlens_q_device = self._prefix_cu_seqlens_q_buf[:num_prefix_cu]
            suffix_seqlens_kv_device = self._suffix_seqlens_kv_buf[:num_suffix_beams]
            suffix_cu_seqlens_q_device = self._suffix_cu_seqlens_q_buf[:num_suffix_cu_q]
            suffix_cu_seqlens_k_device = self._suffix_cu_seqlens_k_buf[:num_suffix_cu_k]

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
            suffix_past_coords,
            suffix_curr_coords,
            suffix_max_past_req,
            suffix_max_past_blk,
            suffix_max_q_idx,
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

        (
            std_reqs,
            beam_req_idxs,
            beam_prefix_lens,
            beam_widths,
            beam_decode_steps,
            beam_cache_lens,
            beam_q_start_offsets,
        ) = self._segment_requests(num_reqs, req_ids, beam_data, q_starts, seq_lens_cpu_list)

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
            suffix_past_coords,
            suffix_curr_coords,
            suffix_max_past_req,
            suffix_max_past_blk,
            suffix_max_q_idx,
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
            prefix_indices_contiguous=prefix_indices_contiguous,
            prefix_max_q_len=prefix_max_q_len,
            suffix_max_q_len=suffix_max_q_len,
            suffix_max_seq_len=suffix_max_seq_len,
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
            suffix_past_coords=suffix_past_coords,
            suffix_curr_coords=suffix_curr_coords,
            suffix_max_past_req=suffix_max_past_req,
            suffix_max_past_blk=suffix_max_past_blk,
            suffix_max_q_idx=suffix_max_q_idx,
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
        )
        return meta

    def _project_suffix_slot_mapping(
        self,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        suffix_total_tokens: int,
        past_coords: torch.Tensor | None,
        curr_coords: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Projects physical suffix slot mappings using sparse past and current token coordinates.

        Memory & Compute Complexity: O(suffix_total_tokens)
        Avoids materializing or concatenating full-capacity block tables.
        """
        if suffix_total_tokens == 0:
            return torch.empty(0, dtype=torch.int32, device=device)

        has_past = past_coords is not None and past_coords.numel() > 0
        has_curr = curr_coords is not None and curr_coords.numel() > 0

        # Fast path 1: Current tokens only (no past block table lookups)
        if has_curr and not has_past:
            curr_slots = slot_mapping[curr_coords[:, 1]].to(torch.int32)
            # curr_coords_list is built via enumerate() over the full
            # suffix_slot_mapping_out_list in order; when every entry is
            # "current" (no past entries to skip), its col-0 output indices
            # are therefore exactly [0, 1, ..., suffix_total_tokens - 1] by
            # construction -- no need for a torch.equal() device sync to
            # confirm it, just the (host-side) length check.
            if curr_coords.shape[0] == suffix_total_tokens:
                return curr_slots
            out = torch.empty(suffix_total_tokens, dtype=torch.int32, device=device)
            out[curr_coords[:, 0]] = curr_slots
            return out

        # Fast path 2: Past tokens only
        if has_past and not has_curr:
            past_blocks = block_table[past_coords[:, 1], past_coords[:, 2]].to(torch.int32)
            past_slots = past_blocks * block_size + past_coords[:, 3].to(torch.int32)
            # Same construction guarantee as fast path 1, mirrored for the
            # all-past case.
            if past_coords.shape[0] == suffix_total_tokens:
                return past_slots
            out = torch.empty(suffix_total_tokens, dtype=torch.int32, device=device)
            out[past_coords[:, 0]] = past_slots
            return out

        # Mixed path: past and current tokens
        out = torch.empty(suffix_total_tokens, dtype=torch.int32, device=device)
        if has_past:
            past_blocks = block_table[past_coords[:, 1], past_coords[:, 2]].to(torch.int32)
            out[past_coords[:, 0]] = past_blocks * block_size + past_coords[:, 3].to(torch.int32)
        if has_curr:
            out[curr_coords[:, 0]] = slot_mapping[curr_coords[:, 1]].to(torch.int32)

        return out

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
        # vLLM reuses one built BeamAttentionMetadata across every hybrid
        # KV-cache group that shares this builder's (kv_cache_spec, type)
        # (see gpu_model_runner.py's _build_attn_group_metadata), swapping in
        # each group's own block_table/slot_mapping via this method instead
        # of a full rebuild. Validate up front that the new tensors are wide
        # enough for the indices this metadata already carries, matching the
        # single-build-multi-group contract (ubatching / sliced block tables
        # are not supported by this fast path).
        if metadata.std_req_idxs_cpu:
            max_std_req = max(metadata.std_req_idxs_cpu)
            if max_std_req >= blk_table.shape[0]:
                raise ValueError(
                    f"Cannot update block table: blk_table row count ({blk_table.shape[0]}) is "
                    f"incompatible with std_req_idxs_cpu max index ({max_std_req}). "
                    f"update_block_table requires full-table tensors with identical batch geometry "
                    f"(ubatching/micro-batching with sliced block tables is not supported)."
                )
        if metadata.prefix_req_idxs_cpu:
            max_prefix_req = max(metadata.prefix_req_idxs_cpu)
            if max_prefix_req >= blk_table.shape[0]:
                raise ValueError(
                    f"Cannot update block table: blk_table row count ({blk_table.shape[0]}) is "
                    f"incompatible with prefix_req_idxs_cpu max index ({max_prefix_req}). "
                    f"update_block_table requires full-table tensors with identical batch geometry "
                    f"(ubatching/micro-batching with sliced block tables is not supported)."
                )
        # max_past_req/max_past_blk/max_q_idx are cached CPU ints computed
        # alongside the coordinate tensors in build() -- reading them here
        # avoids a .max().item() device sync on this hot path. Fall back to
        # computing directly from the tensor for metadata that did not go
        # through build() (e.g. hand-constructed in tests) rather than
        # silently skipping the check.
        if metadata.suffix_past_coords is not None and metadata.suffix_past_coords.numel() > 0:
            max_past_req = (
                metadata.suffix_max_past_req
                if metadata.suffix_max_past_req is not None
                else int(metadata.suffix_past_coords[:, 1].max().item())
            )
            if max_past_req >= blk_table.shape[0]:
                raise ValueError(
                    f"Cannot update block table: blk_table row count ({blk_table.shape[0]}) is "
                    f"incompatible with suffix_past_coords max req_idx ({max_past_req}). "
                    f"update_block_table requires full-table tensors with identical batch geometry "
                    f"(ubatching/micro-batching with sliced block tables is not supported)."
                )
            max_past_blk = (
                metadata.suffix_max_past_blk
                if metadata.suffix_max_past_blk is not None
                else int(metadata.suffix_past_coords[:, 2].max().item())
            )
            if max_past_blk >= blk_table.shape[1]:
                raise ValueError(
                    f"Cannot update block table: blk_table max_blocks ({blk_table.shape[1]}) is "
                    f"incompatible with suffix_past_coords max block_idx ({max_past_blk})."
                )
        if metadata.suffix_curr_coords is not None and metadata.suffix_curr_coords.numel() > 0:
            max_q_idx = (
                metadata.suffix_max_q_idx
                if metadata.suffix_max_q_idx is not None
                else int(metadata.suffix_curr_coords[:, 1].max().item())
            )
            if max_q_idx >= slot_mapping.shape[0]:
                raise ValueError(
                    f"Cannot update block table: slot_mapping length ({slot_mapping.shape[0]}) is "
                    f"incompatible with suffix_curr_coords max q_idx ({max_q_idx})."
                )

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

        if metadata.suffix_total_tokens > 0:
            # The suffix slot mapping was projected from the block table this
            # metadata was originally built() with; re-project it against the
            # new group's block_table/slot_mapping instead of silently
            # carrying forward physical KV slots that belong to a DIFFERENT
            # KV-cache group.
            new_metadata.suffix_slot_mapping_out = self._project_suffix_slot_mapping(
                slot_mapping,
                blk_table,
                metadata.block_size,
                metadata.suffix_total_tokens,
                metadata.suffix_past_coords,
                metadata.suffix_curr_coords,
                blk_table.device,
            )

        return new_metadata
