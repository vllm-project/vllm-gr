# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam Attention metadata and builder."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.beam.geometry import _compute_beam_bounds


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

    # Incremental decode is opt-in. These scalar fields describe the active
    # region of the fixed-address per-layer Pool without changing the legacy
    # suffix-gather metadata contract.
    is_incremental_decode: bool = False
    incremental_beam_width: int = 0
    incremental_suffix_kv_len: int = 0
    beam_session: BeamSearchSession | None = None


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

        Key Invariants:
        1. Query Length Partitioning: A request's total query length `q_len` represents
           the `chunk_budget`. We process lookahead for each beam `b` up to width `w`.
           Each beam consumes `b_uncached` query tokens, decrementing `chunk_budget`
           until it reaches 0.
        2. Prefix Grouping: Since multiple beams within a request may share the same
           prefix length, they are grouped in `prefix_groups` to run attention
           queries in parallel, maximizing hardware occupancy.
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
        if std_seqlens_q_cpu:
            std_indices_tensor = torch.tensor(std_indices, dtype=torch.long, device=device)
            std_max_q_len = max(std_seqlens_q_cpu)
            std_max_seq_len = max(std_seqlens_kv_cpu)
            num_std_blocks = (std_max_seq_len + self.block_size - 1) // self.block_size
            std_block_table = common_attn_metadata.block_table_tensor[
                std_req_idxs, :num_std_blocks
            ].contiguous()

            if self._is_cuda:
                std_cu_seqlens_q_device = torch.tensor(
                    [0] + std_cu_seqlens_q_cpu, dtype=torch.int32, device=device
                )
                std_seqlens_kv_device = torch.tensor(
                    std_seqlens_kv_cpu, dtype=torch.int32, device=device
                )
            else:  # NPU
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
        return None, None, None, 0, 0, None

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
    ):
        if prefix_seqlens_q_cpu:
            prefix_indices_tensor = torch.tensor(prefix_indices, dtype=torch.long, device=device)
            prefix_max_seq_len = max(prefix_seqlens_kv_cpu) if prefix_seqlens_kv_cpu else 0
            prefix_max_q_len = max(prefix_seqlens_q_cpu) if prefix_seqlens_q_cpu else 0
            prefix_block_table = common_attn_metadata.block_table_tensor[
                prefix_req_idxs_cpu, :max_prefix_blocks_count
            ].contiguous()

            suffix_max_seq_len = max(suffix_seqlens_kv_cpu) if suffix_seqlens_kv_cpu else 0
            suffix_max_q_len = max(suffix_seqlens_q_cpu) if suffix_seqlens_q_cpu else 0

            if suffix_slot_mapping_out_list:
                suffix_total_tokens = len(suffix_slot_mapping_out_list)
                global_chunk_token_indices = [
                    prefix_indices[idx] for idx in suffix_slot_mapping_out_list
                ]
                suffix_slot_mapping_out = common_attn_metadata.slot_mapping[
                    global_chunk_token_indices
                ].to(dtype=torch.int32, device=device)
            else:
                suffix_total_tokens = 0
                suffix_slot_mapping_out = torch.empty(0, dtype=torch.int32, device=device)

            if self._is_cuda:
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
            else:  # NPU
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
        else:
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

    @staticmethod
    def _require_incremental_int(name: str, value: Any, *, positive: bool = False) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if positive and value <= 0:
            raise ValueError(f"{name} must be greater than zero, got {value}")
        if not positive and value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return value

    def _build_incremental_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        request_metadata: dict[str, Any],
        beam_session: BeamSearchSession | None,
        max_num_splits: int,
    ) -> BeamAttentionMetadata:
        """Build the CUDA-only, one-token-per-beam incremental metadata."""

        if not self._is_cuda:
            raise NotImplementedError("Incremental Beam Attention is currently CUDA-only")
        if beam_session is None:
            raise RuntimeError("incremental Beam Attention requires a worker-local beam_session")
        if not isinstance(beam_session, BeamSearchSession):
            raise TypeError("beam_session must be a BeamSearchSession")
        if not beam_session.is_active:
            raise RuntimeError("incremental Beam Attention requires an active beam_session")
        if common_attn_metadata.num_reqs != 1:
            raise ValueError("incremental Beam Attention currently supports one logical request")

        beam_width = self._require_incremental_int(
            "beam_width", request_metadata.get("beam_width"), positive=True
        )
        decode_step = self._require_incremental_int(
            "decode_step", request_metadata.get("decode_step")
        )
        prefix_len = self._require_incremental_int("prefix_len", request_metadata.get("prefix_len"))
        cache_len = self._require_incremental_int("cache_len", request_metadata.get("cache_len"))
        if beam_width != beam_session.beam_width:
            raise ValueError("scheduler beam_width does not match the active beam_session")
        if decode_step != beam_session.current_step:
            raise ValueError("scheduler decode_step does not match beam_session.current_step")
        if prefix_len != beam_session.prefix_len:
            raise ValueError("scheduler prefix_len does not match beam_session.prefix_len")
        if cache_len != prefix_len:
            raise ValueError("incremental Beam Attention requires cache_len == prefix_len")
        if prefix_len == 0:
            raise ValueError("incremental Beam Attention currently requires a non-empty prefix")
        if common_attn_metadata.num_actual_tokens != beam_width:
            raise ValueError("incremental Beam Attention requires one query token per beam")

        q_starts = common_attn_metadata.query_start_loc_cpu.tolist()
        if q_starts != [0, beam_width]:
            raise ValueError("incremental query rows must be one contiguous logical request")

        suffix_kv_len = beam_session.current_step + 1
        pool = beam_session.pools[0]
        max_decode_steps = pool.unshared_key.shape[3]
        if suffix_kv_len > max_decode_steps:
            raise ValueError("incremental suffix length exceeds Pool capacity")
        if beam_width * suffix_kv_len > beam_session.suffix_k_buf.shape[0]:
            raise ValueError("incremental suffix length exceeds fixed Buffer capacity")

        device = self.device
        prefix_indices = torch.arange(beam_width, dtype=torch.long, device=device)
        prefix_cu_seqlens_q_device = torch.tensor([0, beam_width], dtype=torch.int32, device=device)
        prefix_seqlens_kv_device = beam_session.actual_shared_kvlen_buf
        max_prefix_blocks_count = (prefix_len + self.block_size - 1) // self.block_size
        if common_attn_metadata.block_table_tensor.shape[1] < max_prefix_blocks_count:
            raise ValueError("block table does not cover the complete shared prefix")
        prefix_block_table = common_attn_metadata.block_table_tensor[
            :1, :max_prefix_blocks_count
        ].contiguous()

        suffix_cu_seqlens_q_device = torch.arange(beam_width + 1, dtype=torch.int32, device=device)
        suffix_cu_seqlens_k_device = torch.arange(
            0,
            (beam_width + 1) * suffix_kv_len,
            suffix_kv_len,
            dtype=torch.int32,
            device=device,
        )
        suffix_seqlens_kv_device = torch.full(
            (beam_width,), suffix_kv_len, dtype=torch.int32, device=device
        )
        prefix_scheduler_metadata = self._get_schedule(
            batch_size=1,
            cu_query_lens=prefix_cu_seqlens_q_device,
            max_query_len=beam_width,
            seqlens=prefix_seqlens_kv_device,
            max_seq_len=prefix_len,
            causal=False,
            max_num_splits=max_num_splits,
        )

        suffix_cu_cpu = list(range(1, beam_width + 1))
        suffix_kv_cu_cpu = [suffix_kv_len * index for index in range(1, beam_width + 1)]
        return BeamAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
            npu_causal_atten_mask=getattr(self, "npu_causal_atten_mask", None),
            std_indices=None,
            std_seqlens_q_cpu=[],
            std_cu_seqlens_q_cpu=None,
            std_seqlens_kv_cpu=[],
            std_cu_seqlens_kv_cpu=None,
            std_req_idxs_cpu=[],
            prefix_req_idxs_cpu=[0],
            max_prefix_blocks_count=max_prefix_blocks_count,
            prefix_indices=prefix_indices,
            prefix_cu_seqlens_q_device=prefix_cu_seqlens_q_device,
            prefix_seqlens_kv_device=prefix_seqlens_kv_device,
            prefix_max_q_len=beam_width,
            prefix_max_seq_len=prefix_len,
            prefix_block_table=prefix_block_table,
            prefix_seqlens_q_cpu=[beam_width],
            prefix_cu_seqlens_q_cpu=[beam_width],
            prefix_seqlens_kv_cpu=[prefix_len],
            prefix_cu_seqlens_kv_cpu=[prefix_len],
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            suffix_cu_seqlens_q_device=suffix_cu_seqlens_q_device,
            suffix_cu_seqlens_k_device=suffix_cu_seqlens_k_device,
            suffix_seqlens_kv_device=suffix_seqlens_kv_device,
            suffix_max_q_len=1,
            suffix_max_seq_len=suffix_kv_len,
            suffix_total_tokens=beam_width * suffix_kv_len,
            suffix_slot_mapping_out=None,
            suffix_seqlens_q_cpu=[1] * beam_width,
            suffix_cu_seqlens_q_cpu=suffix_cu_cpu,
            suffix_seqlens_kv_cpu=[suffix_kv_len] * beam_width,
            suffix_cu_seqlens_kv_cpu=suffix_kv_cu_cpu,
            num_std_indices=0,
            num_prefix_indices=beam_width,
            max_num_splits=max_num_splits,
            block_size=self.block_size,
            num_heads=self.num_heads_q,
            num_kv_heads=self.num_kv_heads,
            is_incremental_decode=True,
            incremental_beam_width=beam_width,
            incremental_suffix_kv_len=suffix_kv_len,
            beam_session=beam_session,
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

        active_request_metadata = [
            beam_data.get(req_ids[index], {}) for index in range(min(num_reqs, len(req_ids)))
        ]
        incremental_requests = [
            metadata
            for metadata in active_request_metadata
            if metadata.get("is_incremental_decode", False)
        ]
        if incremental_requests:
            if len(incremental_requests) != 1 or num_reqs != 1:
                raise ValueError("incremental Beam Attention cannot be mixed with other requests")
            beam_session = beam_info.get("beam_session") if beam_info else None
            return self._build_incremental_metadata(
                common_attn_metadata,
                incremental_requests[0],
                beam_session,
                max_num_splits,
            )

        q_starts = common_attn_metadata.query_start_loc_cpu.tolist()
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
        )
        return meta

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
            new_metadata.std_block_table = blk_table[metadata.std_req_idxs_cpu].contiguous()
        if metadata.prefix_req_idxs_cpu:
            new_metadata.prefix_block_table = blk_table[
                metadata.prefix_req_idxs_cpu, : metadata.max_prefix_blocks_count
            ].contiguous()

        return new_metadata
