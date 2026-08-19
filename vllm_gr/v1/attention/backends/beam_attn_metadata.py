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
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

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
        # capture remain valid across replays.  Currently only CUDA allocates
        # these; when ACL Graph support lands, NPU will set _fixed_buffers=True
        # here too and the rest of the code needs no change.
        self._fixed_buffers = self._is_cuda
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
                for b in range(w):
                    suffix_seqlens_q_cpu.append(1)
                    suffix_seqlens_kv_cpu.append(decode_step + 1)

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
        if not std_seqlens_q_cpu:
            return None, None, None, 0, 0, None

        num_std = len(std_indices)
        std_max_q_len = max(std_seqlens_q_cpu)
        std_max_seq_len = max(std_seqlens_kv_cpu)
        num_std_blocks = (std_max_seq_len + self.block_size - 1) // self.block_size

        if self._fixed_buffers:
            std_indices_tensor = self._std_indices_buf[:num_std]
            std_indices_tensor.copy_(torch.tensor(std_indices, dtype=torch.long, device=device))
            std_block_table = self._std_block_table_buf[: len(std_req_idxs), :num_std_blocks]
            std_block_table.copy_(
                common_attn_metadata.block_table_tensor[std_req_idxs, :num_std_blocks]
            )
            num_cu = len(std_cu_seqlens_q_cpu) + 1  # +1 for leading 0
            std_cu_seqlens_q_device = self._std_cu_seqlens_q_buf[:num_cu]
            std_cu_seqlens_q_device.copy_(
                torch.tensor([0] + std_cu_seqlens_q_cpu, dtype=torch.int32, device=device)
            )
            std_seqlens_kv_device = self._std_seqlens_kv_buf[: len(std_seqlens_kv_cpu)]
            std_seqlens_kv_device.copy_(
                torch.tensor(std_seqlens_kv_cpu, dtype=torch.int32, device=device)
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
            prefix_indices_tensor.copy_(
                torch.tensor(prefix_indices, dtype=torch.long, device=device)
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
            prefix_seqlens_kv_device = self._prefix_seqlens_kv_buf[:num_prefix_groups]
            prefix_seqlens_kv_device.copy_(
                torch.tensor(prefix_seqlens_kv_cpu, dtype=torch.int32, device=device)
            )
            num_prefix_cu = len(prefix_cu_seqlens_q_cpu) + 1
            prefix_cu_seqlens_q_device = self._prefix_cu_seqlens_q_buf[:num_prefix_cu]
            prefix_cu_seqlens_q_device.copy_(
                torch.tensor([0] + prefix_cu_seqlens_q_cpu, dtype=torch.int32, device=device)
            )

            num_suffix_beams = len(suffix_seqlens_kv_cpu)
            suffix_seqlens_kv_device = self._suffix_seqlens_kv_buf[:num_suffix_beams]
            suffix_seqlens_kv_device.copy_(
                torch.tensor(suffix_seqlens_kv_cpu, dtype=torch.int32, device=device)
            )
            num_suffix_cu_q = len(suffix_cu_seqlens_q_cpu) + 1
            suffix_cu_seqlens_q_device = self._suffix_cu_seqlens_q_buf[:num_suffix_cu_q]
            suffix_cu_seqlens_q_device.copy_(
                torch.tensor([0] + suffix_cu_seqlens_q_cpu, dtype=torch.int32, device=device)
            )
            num_suffix_cu_k = len(suffix_cu_seqlens_kv_cpu) + 1
            suffix_cu_seqlens_k_device = self._suffix_cu_seqlens_k_buf[:num_suffix_cu_k]
            suffix_cu_seqlens_k_device.copy_(
                torch.tensor([0] + suffix_cu_seqlens_kv_cpu, dtype=torch.int32, device=device)
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
