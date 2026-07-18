# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam-Consecutive Suffix Gather Attention Backend."""

import contextvars
import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache

from vllm_gr.utils.npu_adapter import is_cuda_available, is_npu_available
from vllm_gr.v1.attention.backends.merge_tnd_attention_outputs import (
    extract_suffix_kv_pytorch,
    merge_tnd_attention_outputs,
)

logger = init_logger(__name__)
BEAM_DATA_VAR = contextvars.ContextVar("beam_data", default=None)
# Device flag – must be checked after torch is loaded.
NPU_DEVICE = is_npu_available()
CUDA_DEVICE = is_cuda_available()

if NPU_DEVICE:
    import torch_npu  # noqa: F401
if CUDA_DEVICE:
    from vllm_gr.v1.attention.backends.beam_attn_triton import (
        extract_suffix_kv,
    )

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
else:

    def _flash_attn_varlen_unavailable(*args, **kwargs):
        raise RuntimeError("BeamAttentionBackend requires FlashAttention varlen support")

    flash_attn_varlen_func = _flash_attn_varlen_unavailable
    reshape_and_cache_flash = _flash_attn_varlen_unavailable
    get_scheduler_metadata = _flash_attn_varlen_unavailable
    merge_attn_states = _flash_attn_varlen_unavailable
    flash_attn_supports_sinks = _flash_attn_varlen_unavailable


def _compute_beam_bounds(
    b: int,
    b_prefix_len: int,
    b_cache_len: int,
    decode_steps: int,
    chunk_budget: int,
) -> Tuple[int, int, int, int]:
    """Calculates active lookahead allocations and historical trace lengths."""
    delta = max(0, b_cache_len - b_prefix_len)
    remaining_prefix = b_prefix_len - b_cache_len
    b_physical = delta // decode_steps
    b_uncached = 0
    b_suffix_len = 0

    b_past_suffix = max(0, min(decode_steps, delta - b_physical * decode_steps))
    if remaining_prefix >= 0:
        # Scenario A: Prefix Incomplete - compute remaining prefix + suffix of first beam
        b_uncached = min(chunk_budget, remaining_prefix + decode_steps)
    else:
        # Scenario B: Compute only suffix of beams
        b_uncached = min(chunk_budget, max(decode_steps - b_past_suffix, 0))
    b_suffix_len = b_uncached + b_past_suffix
    return b_uncached, b_suffix_len, b_past_suffix, b_physical


# ---------------------------------------------------------------------------
# METADATA STRUCTURE DEFINITIONS
# ---------------------------------------------------------------------------
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

        if NPU_DEVICE:
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

        if CUDA_DEVICE and envs.VLLM_BATCH_INVARIANT:
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
            qkv_dtype = BeamAttentionBackend.get_fp8_dtype_for_flashattn(cache_dtype)
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

            if CUDA_DEVICE:
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

            if CUDA_DEVICE:
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


# ---------------------------------------------------------------------------
# CORE EXECUTION REFACTOR
# ---------------------------------------------------------------------------


class BeamAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type
        self.vllm_flash_attn_version = (
            get_flash_attn_version(
                requires_alibi=alibi_slopes is not None,
                head_size=head_size,
            )
            if not NPU_DEVICE
            else 0
        )
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT if CUDA_DEVICE else False

        if is_quantized_kv_cache(self.kv_cache_dtype):
            if NPU_DEVICE:
                raise NotImplementedError("BeamAttention does not support fp8 kv-cache on NPU.")
            if not BeamAttentionBackend.supports_kv_cache_dtype(self.kv_cache_dtype):
                raise NotImplementedError(
                    "BeamAttention does not support fp8 kv-cache on this device."
                )

        self.sinks = sinks
        if self.sinks is not None:
            if NPU_DEVICE:
                raise NotImplementedError("Sinks are not yet supported on NPU.")
            assert flash_attn_supports_sinks(), "Sinks are only supported in FlashAttention 3"
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of heads in the layer"
            )
        self.supports_quant_query_input = True
        self.use_cascade = True

    @staticmethod
    def cascade_attention_gpu(
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        sliding_window_size: list[int] | tuple[int, int] | None,
        scale: float,
        logits_soft_cap: float,
        vllm_flash_attn_version: int,
        s_aux: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
    ):
        if attn_metadata.std_indices is not None:
            if attn_metadata.prefix_indices is None:
                std_query = query[: attn_metadata.num_std_indices]
            else:
                std_query = query[attn_metadata.std_indices]
            std_out = flash_attn_varlen_func(
                q=std_query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=attn_metadata.std_cu_seqlens_q_device,
                max_seqlen_q=attn_metadata.std_max_q_len,
                seqused_k=attn_metadata.std_seqlens_kv_device,
                max_seqlen_k=attn_metadata.std_max_seq_len,
                softmax_scale=scale,
                causal=attn_metadata.causal,
                alibi_slopes=alibi_slopes,
                window_size=sliding_window_size,
                block_table=attn_metadata.std_block_table,
                softcap=logits_soft_cap,
                fa_version=vllm_flash_attn_version,
            )
            if attn_metadata.prefix_indices is None:
                output[: attn_metadata.num_std_indices] = std_out
            else:
                output[attn_metadata.std_indices] = std_out

        if attn_metadata.prefix_indices is not None:
            if attn_metadata.std_indices is None:
                beam_query = query[: attn_metadata.num_prefix_indices]
            else:
                beam_query = query[attn_metadata.prefix_indices]

            prefix_out, prefix_lse = flash_attn_varlen_func(
                q=beam_query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=attn_metadata.prefix_cu_seqlens_q_device,
                seqused_k=attn_metadata.prefix_seqlens_kv_device,
                max_seqlen_q=attn_metadata.prefix_max_q_len,
                max_seqlen_k=attn_metadata.prefix_max_seq_len,
                softmax_scale=scale,
                causal=False,
                alibi_slopes=alibi_slopes,
                window_size=sliding_window_size,
                block_table=attn_metadata.prefix_block_table,
                softcap=logits_soft_cap,
                return_softmax_lse=True,
                fa_version=vllm_flash_attn_version,
                scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
                num_splits=attn_metadata.max_num_splits,
                s_aux=s_aux,
            )

            contig_k, contig_v = extract_suffix_kv(
                key_cache,
                value_cache,
                attn_metadata.suffix_slot_mapping_out,
                attn_metadata.suffix_total_tokens,
                attn_metadata.block_size,
            )

            suffix_out, suffix_lse = flash_attn_varlen_func(
                q=beam_query,
                k=contig_k,
                v=contig_v,
                cu_seqlens_q=attn_metadata.suffix_cu_seqlens_q_device,
                cu_seqlens_k=attn_metadata.suffix_cu_seqlens_k_device,
                seqused_k=None,
                max_seqlen_q=attn_metadata.suffix_max_q_len,
                max_seqlen_k=attn_metadata.suffix_max_seq_len,
                softmax_scale=scale,
                causal=True,
                alibi_slopes=alibi_slopes,
                window_size=sliding_window_size,
                block_table=None,
                softcap=logits_soft_cap,
                return_softmax_lse=True,
                fa_version=vllm_flash_attn_version,
            )
            group_size = attn_metadata.num_heads // attn_metadata.num_kv_heads
            if (
                attn_metadata.suffix_max_q_len == 1
                and attn_metadata.suffix_max_seq_len == 1
                and group_size > 1
                and not vllm_flash_attn_version == 3
            ):
                suffix_lse = (
                    suffix_lse.view(attn_metadata.num_kv_heads, group_size, -1)
                    .transpose(1, 2)
                    .contiguous()
                    .view(attn_metadata.num_heads, -1)
                )

            out = torch.empty_like(beam_query)
            merge_attn_states(out, prefix_out, prefix_lse, suffix_out, suffix_lse)
            if attn_metadata.std_indices is None:
                output[: attn_metadata.num_prefix_indices] = out
            else:
                output[attn_metadata.prefix_indices] = out

    @staticmethod
    def cascade_attention_npu(
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        sliding_window_size: list[int] | tuple[int, int] | None,
        scale: float,
        logits_soft_cap: float,
        vllm_flash_attn_version: int,
        s_aux: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
    ):
        # Standard attention
        if attn_metadata.std_indices is not None:
            if attn_metadata.prefix_indices is None:
                std_query = query[: attn_metadata.num_std_indices]
            else:
                std_query = query[attn_metadata.std_indices]
            std_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query=std_query,
                key=key_cache.view(*key_cache.shape[:-2], -1),
                value=value_cache.view(*value_cache.shape[:-2], -1),
                atten_mask=attn_metadata.npu_causal_atten_mask if attn_metadata.causal else None,
                block_table=attn_metadata.std_block_table,
                input_layout="TND",
                block_size=attn_metadata.block_size,
                actual_seq_qlen=attn_metadata.std_cu_seqlens_q_cpu,
                actual_seq_kvlen=attn_metadata.std_seqlens_kv_cpu,
                num_key_value_heads=attn_metadata.num_kv_heads,
                num_query_heads=attn_metadata.num_heads,
                softmax_scale=scale,
                sparse_mode=3 if attn_metadata.causal else 0,
            )
            if attn_metadata.prefix_indices is None:
                output[: attn_metadata.num_std_indices] = std_out
            else:
                output[attn_metadata.std_indices] = std_out
        # Prefix/Suffix attention
        if attn_metadata.prefix_indices is not None:
            if attn_metadata.std_indices is None:
                beam_query = query[: attn_metadata.num_prefix_indices]
            else:
                beam_query = query[attn_metadata.prefix_indices]
            prefix_out, prefix_lse = torch_npu.npu_fused_infer_attention_score_v2(
                query=beam_query,
                key=key_cache.view(*key_cache.shape[:-2], -1),
                value=value_cache.view(*value_cache.shape[:-2], -1),
                block_table=attn_metadata.prefix_block_table,
                input_layout="TND",
                atten_mask=None,
                block_size=attn_metadata.block_size,
                actual_seq_qlen=attn_metadata.prefix_cu_seqlens_q_cpu,
                actual_seq_kvlen=attn_metadata.prefix_seqlens_kv_cpu,
                num_key_value_heads=attn_metadata.num_kv_heads,
                num_query_heads=attn_metadata.num_heads,
                softmax_scale=scale,
                return_softmax_lse=True,
                sparse_mode=0,  # prefix attention is non-causal (full)
            )

            contig_k, contig_v = extract_suffix_kv_pytorch(
                key_cache, value_cache, attn_metadata.suffix_slot_mapping_out
            )

            suffix_out, suffix_lse = torch_npu.npu_fused_infer_attention_score_v2(
                beam_query,
                contig_k,
                contig_v,
                num_query_heads=attn_metadata.num_heads,
                num_key_value_heads=attn_metadata.num_kv_heads,
                actual_seq_qlen=attn_metadata.suffix_cu_seqlens_q_cpu,
                actual_seq_kvlen=attn_metadata.suffix_cu_seqlens_kv_cpu,
                softmax_scale=scale,
                atten_mask=attn_metadata.npu_causal_atten_mask,
                block_table=None,
                block_size=0,
                sparse_mode=3,
                inner_precise=0,
                input_layout="TND",
                return_softmax_lse=True,
            )
            out, _ = merge_tnd_attention_outputs(prefix_out, prefix_lse, suffix_out, suffix_lse)
            if attn_metadata.std_indices is None:
                output[: attn_metadata.num_prefix_indices] = out
            else:
                output[attn_metadata.prefix_indices] = out

    @staticmethod
    def cascade_attention(
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        sliding_window_size: list[int] | tuple[int, int] | None,
        scale: float,
        logits_soft_cap: float,
        vllm_flash_attn_version: int,
        s_aux: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
    ):
        if CUDA_DEVICE and not NPU_DEVICE:
            BeamAttentionImpl.cascade_attention_gpu(
                output=output,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                attn_metadata=attn_metadata,
                sliding_window_size=sliding_window_size,
                scale=scale,
                logits_soft_cap=logits_soft_cap,
                vllm_flash_attn_version=vllm_flash_attn_version,
                s_aux=s_aux,
                alibi_slopes=alibi_slopes,
            )
        elif NPU_DEVICE and not CUDA_DEVICE:
            BeamAttentionImpl.cascade_attention_npu(
                output=output,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                attn_metadata=attn_metadata,
                sliding_window_size=sliding_window_size,
                scale=scale,
                logits_soft_cap=logits_soft_cap,
                vllm_flash_attn_version=vllm_flash_attn_version,
                s_aux=s_aux,
                alibi_slopes=alibi_slopes,
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if CUDA_DEVICE:
            assert self.vllm_flash_attn_version is not None, "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for BeamAttentionImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)

        # NPU provides kv_cache as a tuple (key_cache, value_cache);
        # GPU provides a stacked tensor [2, num_blocks, ...]
        if isinstance(kv_cache, tuple):
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        else:
            key_cache, value_cache = kv_cache.unbind(0)

        if self.kv_sharing_target_layer_name is None and key is not None and value is not None:
            if NPU_DEVICE:
                torch_npu._npu_reshape_and_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                )
            elif CUDA_DEVICE:
                reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = BeamAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        BeamAttentionImpl.cascade_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            attn_metadata=attn_metadata,
            sliding_window_size=sliding_window_size,
            scale=self.scale,
            logits_soft_cap=self.logits_soft_cap,
            vllm_flash_attn_version=self.vllm_flash_attn_version,
            s_aux=self.sinks,
            alibi_slopes=self.alibi_slopes,
        )

        return output


@register_backend(AttentionBackendEnum.CUSTOM)
class BeamAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    supported_dtypes = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return CUDA_DEVICE

    @staticmethod
    def get_impl_cls() -> type["BeamAttentionImpl"]:
        return BeamAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["BeamAttentionMetadataBuilder"]:
        return BeamAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension: bool = False) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            return (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        raise ValueError(f"Unknown cache layout format {cache_layout}.")

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            if NPU_DEVICE:
                return False  # fp8 kv-cache not yet supported on NPU
            return get_flash_attn_version() == 3 and current_platform.is_device_capability_family(
                90
            )
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if NPU_DEVICE:
            return False  # sinks not yet supported on NPU
        return is_flash_attn_varlen_func_available() and flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # NPU does not use CUDA-style compute capability; accept it.
        if NPU_DEVICE:
            return True
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        device_capability,
    ) -> str | None:
        if not NPU_DEVICE and not is_flash_attn_varlen_func_available():
            return "No FlashAttention varlen support."
        if dtype not in cls.supported_dtypes:
            return f"Unsupported dtype {dtype}."
        if not cls.supports_head_size(head_size):
            return f"Unsupported head_size={head_size}."
        if not NPU_DEVICE and not cls.supports_kv_cache_dtype(kv_cache_dtype):
            return f"Unsupported kv_cache_dtype={kv_cache_dtype}."
        if block_size is not None and block_size % 16 != 0:
            return f"block_size={block_size} not multiple of 16."
        if not cls.supports_compute_capability(device_capability):
            return "Requires GPU compute capability >= 8.0."
        if not NPU_DEVICE and has_sink and not cls.supports_sink():
            return "No FlashAttention sink support."
        if use_mla:
            return "No MLA support."
        if use_sparse:
            return "No sparse support."
        return None
