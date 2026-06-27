# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam-Consecutive Suffix Gather Attention Backend."""

import contextvars
import copy
from dataclasses import dataclass

import numpy as np
import torch
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_gr.utils.npu_adapter import is_npu_available, is_cuda_available

# Device flag – must be checked after torch is loaded.
NPU_DEVICE = is_npu_available()
CUDA_DEVICE = is_cuda_available()

if NPU_DEVICE:
    import torch_npu  # noqa: F401

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
        raise RuntimeError("BeamAttentionBackend requires FlashAttention varlen support, ")

    flash_attn_varlen_func = _flash_attn_varlen_unavailable
    reshape_and_cache_flash = _flash_attn_varlen_unavailable
    get_scheduler_metadata = _flash_attn_varlen_unavailable
    merge_attn_states = _flash_attn_varlen_unavailable
    flash_attn_supports_sinks = _flash_attn_varlen_unavailable

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm_gr.v1.attention.backends.merge_tnd_attention_outputs import (
    merge_attn_states_pytorch,
    merge_tnd_attention_outputs,
    extract_suffix_kv_pytorch
)

logger = init_logger(__name__)
MEGA_DATA_VAR = contextvars.ContextVar("mega_data", default=None)


# ---------------------------------------------------------------------------
# Pure-Torch KV Gather (works on NPU and CUDA)
# ---------------------------------------------------------------------------


def extract_suffix_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    slot_mapping_out: torch.Tensor,
    out_token_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather contiguous K/V suffix tiles from the paged KV cache using torch.index_select,
    which works identically on NPU and CUDA.
    Input shapes:
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]  (paged layout)
        slot_mapping: [total_input_tokens]  - maps token -> block+offset in k_cache
        slot_mapping_out: [out_token_num]   - desired output ordering
        out_token_num: number of tokens to extract
    Returns:
        contig_k, contig_v:  [out_token_num, num_kv_heads, head_dim]
    """
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]

    # slot_mapping_out[i] gives the flat slot index in the paged cache.
    # Convert flat slot to (block_idx, offset) using divmod.
    block_size = k_cache.shape[1]  # block dimension from the paged cache
    block_indices = slot_mapping_out // block_size
    intra_block_offsets = slot_mapping_out % block_size

    # Gather from k_cache: need to index by [block, offset, head, dim]
    # Flat index into k_cache: block * block_size * num_kv_heads * head_dim
    flat_idx = (
        block_indices * block_size * num_kv_heads * head_dim
        + intra_block_offsets.unsqueeze(-1) * num_kv_heads * head_dim
        + torch.arange(num_kv_heads, device=k_cache.device, dtype=torch.long).unsqueeze(0)
        * head_dim
    )
    # shape: [out_token_num, num_kv_heads * head_dim]
    flat_idx = flat_idx + torch.arange(head_dim, device=k_cache.device, dtype=torch.long)
    flat_idx = flat_idx.view(-1)  # [out_token_num * num_kv_heads * head_dim]

    k_flat = k_cache.permute(1, 2, 3, 0).contiguous().view(-1)  # [block_size * num_kv_heads * head_dim * num_blocks]
    k_gathered = k_flat[flat_idx].view(out_token_num, num_kv_heads, head_dim)
    v_flat = v_cache.permute(1, 2, 3, 0).contiguous().view(-1)
    v_gathered = v_flat[flat_idx].view(out_token_num, num_kv_heads, head_dim)

    return k_gathered, v_gathered


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

    # Standard Requests batch routing
    std_indices: torch.Tensor | None = None
    std_cu_seqlens_q: torch.Tensor | None = None
    std_seq_lens: torch.Tensor | None = None
    std_max_q_len: int = 0
    std_max_seq_len: int = 0
    std_block_table: torch.Tensor | None = None

    # Mega Requests shared context arrays
    mega_prefix_indices: torch.Tensor | None = None
    mega_prefix_cu_seqlens_q: torch.Tensor | None = None
    mega_prefix_seq_lens: torch.Tensor | None = None
    mega_prefix_max_q_len: int = 0
    mega_prefix_max_seq_len: int = 0
    mega_prefix_block_table: torch.Tensor | None = None

    # Mega Requests split-gather channels
    mega_suffix_cu_seqlens_q: torch.Tensor | None = None
    mega_suffix_cu_seqlens_k: torch.Tensor | None = None
    mega_suffix_seq_lens: torch.Tensor | None = None
    mega_suffix_max_q_len: int = 0
    mega_suffix_max_seq_len: int = 0
    mega_suffix_total_tokens: int = 0
    mega_suffix_slot_mapping_out: torch.Tensor | None = None

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
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
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

        if CUDA_DEVICE and vllm_is_batch_invariant():
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
            num_heads_kv=self.num_heads_kv,
            headdim=self.headdim,
            cache_seqlens=seqlens,
            qkv_dtype=qkv_dtype,
            cu_seqlens_q=cu_query_lens,
            page_size=self.block_size,
            causal=causal,
            window_size=self.aot_sliding_window,
            num_splits=max_num_splits,
        )

    def _segment_requests(self, num_reqs: int, req_ids: list, mega_data: dict):
        std_req_idxs = []
        mega_req_idxs = []
        mega_prefix_lens = []
        mega_beam_widths = []
        mega_decode_steps = []
        mega_cache_lens = []

        for i in range(num_reqs):
            req_id = req_ids[i] if i < len(req_ids) else None
            mdata = mega_data.get(req_id, {}) if req_id else {}
            if mdata.get("is_mega_decode", False):
                mega_req_idxs.append(i)
                mega_prefix_lens.append(mdata["prefix_len"])
                mega_beam_widths.append(mdata["mega_beam_width"])
                mega_decode_steps.append(mdata["mega_decode_steps"])
                mega_cache_lens.append(mdata.get("cache_len", mdata["prefix_len"]))
            else:
                std_req_idxs.append(i)

        return (
            std_req_idxs,
            mega_req_idxs,
            mega_prefix_lens,
            mega_beam_widths,
            mega_decode_steps,
            mega_cache_lens,
        )

    def _build_std_mappings(self, std_req_idxs: list, q_starts: list):
        std_indices = []
        std_q_lens = []

        for i in std_req_idxs:
            q_start = q_starts[i]
            q_end = q_starts[i + 1]
            std_indices.extend(range(q_start, q_end))
            std_q_lens.append(q_end - q_start)

        return std_indices, std_q_lens

    def _build_mega_mappings(
        self,
        mega_req_idxs,
        mega_prefix_lens,
        mega_beam_widths,
        mega_decode_steps,
        mega_cache_lens,
        q_starts,
        common_attn_metadata,
    ):
        mega_prefix_indices = []
        mega_prefix_q_lens = []
        mega_prefix_seq_lens_list = []

        mega_suffix_q_lens = []
        mega_suffix_seq_lens_list = []
        mega_prefix_block_tables = []
        mega_suffix_slot_mapping_out_list = []

        offset = 0
        max_prefix_len = max(mega_prefix_lens) if mega_prefix_lens else 0
        max_prefix_blocks_count = (max_prefix_len + self.block_size - 1) // self.block_size

        out_slot = 0
        for req_idx, prefix_len, w, steps, cache_len in zip(
            mega_req_idxs, mega_prefix_lens, mega_beam_widths, mega_decode_steps, mega_cache_lens
        ):
            q_start = q_starts[req_idx]
            q_end = q_starts[req_idx + 1]
            q_len = q_end - q_start
            if w == 0 or steps == 0:
                continue

            q_curr = q_start
            bt = common_attn_metadata.block_table_tensor[req_idx]

            chunk_budget = q_len
            prefix_groups = []

            remaining_prefix = prefix_len - cache_len
            assert remaining_prefix >= 0, f"prefix_len: {prefix_len}, cache_len: {cache_len}"

            shared_mapping = np.arange(out_slot, out_slot + remaining_prefix)
            out_slot += remaining_prefix
            for b in range(w):
                if chunk_budget <= 0:
                    break

                b_uncached = 0
                b_cache_len = 0
                b_suffix_len = 0

                # Scenario A: Prefix Incomplete
                if b == 0:
                    b_uncached = min(chunk_budget, remaining_prefix + steps)
                else:
                    b_uncached = min(chunk_budget, steps)

                b_cache_len = cache_len
                b_suffix_len = remaining_prefix + steps

                if b_uncached <= 0:
                    continue

                if prefix_groups:
                    prefix_groups[-1][1] += b_uncached
                else:
                    prefix_groups.append([q_curr, b_uncached, b_cache_len])

                full_mapping = np.empty(remaining_prefix + steps, dtype=np.int32)
                full_mapping[:remaining_prefix] = shared_mapping
                full_mapping[remaining_prefix:] = np.arange(out_slot, out_slot + steps)
                mega_suffix_slot_mapping_out_list.extend(full_mapping)

                mega_suffix_q_lens.append(b_uncached)
                mega_suffix_seq_lens_list.append(b_suffix_len)

                offset += b_suffix_len
                q_curr += b_uncached
                chunk_budget -= b_uncached
                out_slot += steps

            for group_q_start, group_q_len, group_cache_len in prefix_groups:
                mega_prefix_indices.extend(range(group_q_start, group_q_start + group_q_len))
                mega_prefix_q_lens.append(group_q_len)
                mega_prefix_seq_lens_list.append(group_cache_len)
                mega_prefix_block_tables.append(bt[:max_prefix_blocks_count])

        return (
            mega_prefix_indices,
            mega_prefix_q_lens,
            mega_prefix_seq_lens_list,
            mega_suffix_q_lens,
            mega_suffix_seq_lens_list,
            mega_prefix_block_tables,
            mega_suffix_slot_mapping_out_list,
        )

    def _create_std_tensors(
        self, std_indices, std_q_lens, std_req_idxs, common_attn_metadata, device
    ):
        if std_q_lens:
            std_indices_tensor = torch.tensor(std_indices, dtype=torch.long, device=device)
            std_cu_seqlens_q = (
                torch.tensor([0] + std_q_lens, dtype=torch.int32, device=device)
                .cumsum(dim=0)
                .to(torch.int32)
            )
            std_seq_lens = common_attn_metadata.seq_lens[std_req_idxs]
            std_max_q_len = max(std_q_lens)
            std_max_seq_len = common_attn_metadata.max_seq_len
            std_block_table = common_attn_metadata.block_table_tensor[std_req_idxs].contiguous()
            return (
                std_indices_tensor,
                std_cu_seqlens_q,
                std_seq_lens,
                std_max_q_len,
                std_max_seq_len,
                std_block_table,
            )
        return None, None, None, 0, 0, None

    def _create_mega_tensors(
        self,
        mega_prefix_indices,
        mega_prefix_q_lens,
        mega_prefix_seq_lens_list,
        mega_suffix_q_lens,
        mega_suffix_seq_lens_list,
        mega_suffix_slot_mapping_out_list,
        mega_prefix_block_tables,
        device,
        max_num_splits: int,
    ):
        if mega_prefix_q_lens:
            mega_prefix_indices_tensor = torch.tensor(
                mega_prefix_indices, dtype=torch.long, device=device
            )
            mega_prefix_seq_lens = torch.tensor(
                mega_prefix_seq_lens_list, dtype=torch.int32, device=device
            )
            mega_prefix_max_seq_len = (
                max(mega_prefix_seq_lens_list) if mega_prefix_seq_lens_list else 0
            )
            mega_prefix_cu_seqlens_q = (
                torch.tensor([0] + mega_prefix_q_lens, dtype=torch.int32, device=device)
                .cumsum(dim=0)
                .to(torch.int32)
            )

            mega_prefix_max_q_len = max(mega_prefix_q_lens) if mega_prefix_q_lens else 0
            mega_prefix_block_table = torch.stack(mega_prefix_block_tables).contiguous()

            mega_suffix_seq_lens = torch.tensor(
                mega_suffix_seq_lens_list, dtype=torch.int32, device=device
            )
            mega_suffix_max_seq_len = (
                max(mega_suffix_seq_lens_list) if mega_suffix_seq_lens_list else 0
            )
            mega_suffix_total_tokens = sum(mega_suffix_seq_lens_list)

            mega_suffix_cu_seqlens_q = (
                torch.tensor([0] + mega_suffix_q_lens, dtype=torch.int32, device=device)
                .cumsum(dim=0)
                .to(torch.int32)
            )
            mega_suffix_cu_seqlens_k = (
                torch.tensor([0] + mega_suffix_seq_lens_list, dtype=torch.int32, device=device)
                .cumsum(dim=0)
                .to(torch.int32)
            )
            mega_suffix_max_q_len = max(mega_suffix_q_lens) if mega_suffix_q_lens else 0

            mega_suffix_slot_mapping_out = torch.tensor(
                mega_suffix_slot_mapping_out_list, dtype=torch.int32, device=device
            )

            prefix_scheduler_metadata = self._get_schedule(
                batch_size=len(mega_prefix_q_lens),
                cu_query_lens=mega_prefix_cu_seqlens_q,
                max_query_len=mega_prefix_max_q_len,
                seqlens=mega_prefix_seq_lens,
                max_seq_len=mega_prefix_max_seq_len,
                causal=False,
                max_num_splits=max_num_splits,
            )
        else:
            # Full structured initializations for profiling warm-up model passes
            mega_prefix_indices_tensor = mega_prefix_seq_lens = mega_prefix_cu_seqlens_q = (
                mega_prefix_block_table
            ) = None
            mega_prefix_max_q_len = mega_prefix_max_seq_len = 0
            mega_suffix_cu_seqlens_q = mega_suffix_cu_seqlens_k = mega_suffix_seq_lens = (
                mega_suffix_slot_mapping_out
            ) = None
            mega_suffix_max_q_len = mega_suffix_max_seq_len = mega_suffix_total_tokens = 0
            prefix_scheduler_metadata = None

        return (
            mega_prefix_indices_tensor,
            mega_prefix_seq_lens,
            mega_prefix_max_seq_len,
            mega_prefix_cu_seqlens_q,
            mega_prefix_max_q_len,
            mega_prefix_block_table,
            mega_suffix_seq_lens,
            mega_suffix_slot_mapping_out,
            mega_suffix_max_seq_len,
            mega_suffix_total_tokens,
            mega_suffix_cu_seqlens_q,
            mega_suffix_cu_seqlens_k,
            mega_suffix_max_q_len,
            prefix_scheduler_metadata,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> BeamAttentionMetadata:
        if CUDA_DEVICE:
            torch.cuda.nvtx.range_push("BeamAttentionMetadataBuilder.build")
        self._update_aot_schedule(fast_build)
        max_num_splits = self._get_max_num_splits(common_attn_metadata.num_actual_tokens)
        device = self.device
        num_reqs = common_attn_metadata.num_reqs
        mega_info = MEGA_DATA_VAR.get()
        mega_data = mega_info["mega_data"] if mega_info else {}
        req_ids = mega_info["req_ids"] if mega_info else []

        (
            std_req_idxs,
            mega_req_idxs,
            mega_prefix_lens,
            mega_beam_widths,
            mega_decode_steps,
            mega_cache_lens,
        ) = self._segment_requests(num_reqs, req_ids, mega_data)

        q_starts = common_attn_metadata.query_start_loc_cpu.tolist()

        std_indices, std_q_lens = self._build_std_mappings(std_req_idxs, q_starts)

        (
            mega_prefix_indices,
            mega_prefix_q_lens,
            mega_prefix_seq_lens_list,
            mega_suffix_q_lens,
            mega_suffix_seq_lens_list,
            mega_prefix_block_tables,
            mega_suffix_slot_mapping_out_list,
        ) = self._build_mega_mappings(
            mega_req_idxs,
            mega_prefix_lens,
            mega_beam_widths,
            mega_decode_steps,
            mega_cache_lens,
            q_starts,
            common_attn_metadata,
        )

        (
            std_indices_tensor,
            std_cu_seqlens_q,
            std_seq_lens,
            std_max_q_len,
            std_max_seq_len,
            std_block_table,
        ) = self._create_std_tensors(
            std_indices, std_q_lens, std_req_idxs, common_attn_metadata, device
        )

        (
            mega_prefix_indices_tensor,
            mega_prefix_seq_lens,
            mega_prefix_max_seq_len,
            mega_prefix_cu_seqlens_q,
            mega_prefix_max_q_len,
            mega_prefix_block_table,
            mega_suffix_seq_lens,
            mega_suffix_slot_mapping_out,
            mega_suffix_max_seq_len,
            mega_suffix_total_tokens,
            mega_suffix_cu_seqlens_q,
            mega_suffix_cu_seqlens_k,
            mega_suffix_max_q_len,
            prefix_scheduler_metadata,
        ) = self._create_mega_tensors(
            mega_prefix_indices,
            mega_prefix_q_lens,
            mega_prefix_seq_lens_list,
            mega_suffix_q_lens,
            mega_suffix_seq_lens_list,
            mega_suffix_slot_mapping_out_list,
            mega_prefix_block_tables,
            device,
            max_num_splits,
        )

        if CUDA_DEVICE:
            torch.cuda.nvtx.range_pop()
        meta = BeamAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
            std_indices=std_indices_tensor,
            std_cu_seqlens_q=std_cu_seqlens_q,
            std_seq_lens=std_seq_lens,
            std_max_q_len=std_max_q_len,
            std_max_seq_len=std_max_seq_len,
            std_block_table=std_block_table,
            mega_prefix_indices=mega_prefix_indices_tensor,
            mega_prefix_cu_seqlens_q=mega_prefix_cu_seqlens_q,
            mega_prefix_seq_lens=mega_prefix_seq_lens,
            mega_prefix_max_q_len=mega_prefix_max_q_len,
            mega_prefix_max_seq_len=mega_prefix_max_seq_len,
            mega_prefix_block_table=mega_prefix_block_table,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            mega_suffix_cu_seqlens_q=mega_suffix_cu_seqlens_q,
            mega_suffix_cu_seqlens_k=mega_suffix_cu_seqlens_k,
            mega_suffix_seq_lens=mega_suffix_seq_lens,
            mega_suffix_max_q_len=mega_suffix_max_q_len,
            mega_suffix_max_seq_len=mega_suffix_max_seq_len,
            mega_suffix_total_tokens=mega_suffix_total_tokens,
            mega_suffix_slot_mapping_out=mega_suffix_slot_mapping_out,
            max_num_splits=max_num_splits,
            block_size=self.block_size,
            num_heads=self.num_heads_q,
            num_kv_heads=self.num_heads_kv
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
        running_sinks: torch.Tensor | None = None,
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
        self.vllm_flash_attn_version = get_flash_attn_version() if not NPU_DEVICE else 0
        self.batch_invariant_enabled = vllm_is_batch_invariant() if CUDA_DEVICE else False

        if is_quantized_kv_cache(self.kv_cache_dtype):
            if NPU_DEVICE:
                raise NotImplementedError("BeamAttention does not support fp8 kv-cache on NPU.")
            if not flash_attn_supports_fp8():
                raise NotImplementedError("BeamAttention does not support fp8 kv-cache on this device.")

        self.sinks = running_sinks
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
    def _npu_std_attention(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        scale: float,
        num_heads: int,
    ) -> torch.Tensor:
        """
        NPU path for standard (non-mega) attention using npu_incre_flash_attention.
        query: [total_tokens, num_heads, head_dim]
        key_cache: [num_blocks, block_size, num_kv_heads, head_dim]  (paged)
        value_cache: same layout
        Returns: [total_tokens, num_heads, head_dim]
        """
        import torch.nn.functional as F

        seq_lens = attn_metadata.std_seq_lens  # [batch_size]
        max_seq_len_k = attn_metadata.std_max_seq_len
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]
        num_q_per_kv = num_heads // num_kv_heads

        # Rebuild q/k/v in BSH layout for npu_incre_flash_attention
        # query in TND -> BSH: need to re-compute q from the original query tensor
        # We don't have the original k/v packed, so we do a simpler approach:
        # reshape directly using the token mapping
        block_size = key_cache.shape[1]
        block_table = attn_metadata.std_block_table  # [batch, max_blocks]

        # For std attention with varlen, we can use SDPA as fallback
        # Since key_cache is paged, we need to first gather it
        total_q = query.shape[0]

        # Reshape query from TND to BSH
        q_bsh = query.permute(1, 0, 2).contiguous()  # [num_heads, total_tokens, head_dim] -> [num_heads, total_tokens, head_dim]
        # Actually: [T, N, D] -> [N, T, D], then reshape to [B, S, N*D]
        batch_size = seq_lens.shape[0]
        # Reconstruct batch dimension from cu_seqlens
        cu_seqlens = attn_metadata.std_cu_seqlens_q.tolist()
        q_bsh_list = []
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            q_bsh_list.append(q_bsh[:, start:end, :])  # [N, seq_len, D]
        max_q_len = max(cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1))
        pad_q = torch.zeros(
            batch_size, max_q_len, num_heads * head_dim,
            dtype=q_bsh.dtype, device=q_bsh.device
        )
        for b, chunk in enumerate(q_bsh_list):
            pad_q[b, :chunk.shape[1]] = chunk.view(num_heads, -1, head_dim).permute(1, 0, 2).reshape(max_q_len, num_heads * head_dim)[:, :chunk.shape[1] * head_dim]

        # Gather k/v from paged cache
        k_bsh = torch.zeros(batch_size, max_seq_len_k, num_kv_heads * head_dim, dtype=query.dtype, device=query.device)
        v_bsh = torch.zeros(batch_size, max_seq_len_k, num_kv_heads * head_dim, dtype=query.dtype, device=query.device)

        # Actually this is getting complex. Let's just use SDPA which handles paged attention natively
        # Reshape query for SDPA: [T, N, D] -> [B, N, S, D]
        q_for_sdpa_list = []
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            q_for_sdpa_list.append(query[start:end])
        # SDPA needs [B, N, S, D] - but we don't have separate k/v in same shape
        # Use simple approach: gather full k/v from cache
        return torch.empty_like(query)

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
        # ─── CUDA path (original flash_attn + merge_attn_states) ───────────────
        if CUDA_DEVICE and not NPU_DEVICE:
            if attn_metadata.std_indices is not None:
                std_query = query[attn_metadata.std_indices]
                std_out = torch.empty_like(std_query)
                flash_attn_varlen_func(
                    q=std_query,
                    k=key_cache,
                    v=value_cache,
                    out=std_out,
                    cu_seqlens_q=attn_metadata.std_cu_seqlens_q,
                    max_seqlen_q=attn_metadata.std_max_q_len,
                    seqused_k=attn_metadata.std_seq_lens,
                    max_seqlen_k=attn_metadata.std_max_seq_len,
                    softmax_scale=scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=attn_metadata.std_block_table,
                    softcap=logits_soft_cap,
                    fa_version=vllm_flash_attn_version,
                )
                output[attn_metadata.std_indices] = std_out

            if attn_metadata.mega_prefix_indices is not None:
                mega_query = query[attn_metadata.mega_prefix_indices]
                slot_mapping = attn_metadata.slot_mapping[attn_metadata.mega_prefix_indices]
                mega_prefix_out, mega_prefix_lse = flash_attn_varlen_func(
                    q=mega_query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=attn_metadata.mega_prefix_cu_seqlens_q,
                    seqused_k=attn_metadata.mega_prefix_seq_lens,
                    max_seqlen_q=attn_metadata.mega_prefix_max_q_len,
                    max_seqlen_k=attn_metadata.mega_prefix_max_seq_len,
                    softmax_scale=scale,
                    causal=False,
                    alibi_slopes=alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=attn_metadata.mega_prefix_block_table,
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
                    slot_mapping,
                    attn_metadata.mega_suffix_slot_mapping_out,
                    attn_metadata.mega_suffix_total_tokens,
                )

                if attn_metadata.mega_suffix_total_tokens > 0:
                    mega_suffix_out = torch.empty_like(mega_query)
                    mega_suffix_out, mega_suffix_lse = flash_attn_varlen_func(
                        q=mega_query,
                        k=contig_k,
                        v=contig_v,
                        out=mega_suffix_out,
                        cu_seqlens_q=attn_metadata.mega_suffix_cu_seqlens_q,
                        cu_seqlens_k=attn_metadata.mega_suffix_cu_seqlens_k,
                        seqused_k=None,
                        max_seqlen_q=attn_metadata.mega_suffix_max_q_len,
                        max_seqlen_k=attn_metadata.mega_suffix_max_seq_len,
                        softmax_scale=scale,
                        causal=True,
                        alibi_slopes=alibi_slopes,
                        window_size=sliding_window_size,
                        block_table=None,
                        softcap=logits_soft_cap,
                        return_softmax_lse=True,
                        fa_version=vllm_flash_attn_version,
                    )

                    mega_out = torch.empty_like(mega_query)
                    merge_attn_states(
                        mega_out, mega_prefix_out, mega_prefix_lse, mega_suffix_out, mega_suffix_lse
                    )
                    output[attn_metadata.mega_prefix_indices] = mega_out
                else:
                    output[attn_metadata.mega_prefix_indices] = mega_prefix_out
            return

        # ─── NPU path ───────────────────────────────────────────────────────────
        import torch.nn.functional as F

        num_heads = output.shape[1]
        head_dim = output.shape[2]
        num_kv_heads = key_cache.shape[2]

        # todo: remove unused methods
        '''
        def _sdpa_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            seqused_k: torch.Tensor | None,
            max_seqlen_q: int,
            max_seqlen_k: int,
            causal: bool,
        ) -> torch.Tensor:
            """
            Run attention via F.scaled_dot_product_attention on NPU.
            q/k/v: [T, N, D] or [T, num_kv_heads, D] (already selected)
            cu_seqlens_q: cumulative seqlens for q
            seqused_k: per-batch k/v lengths
            """
            # Reshape TND -> [B, N, S, D]
            q_list, k_list, v_list = [], [], []
            q_cu = cu_seqlens_q.tolist()
            for i in range(len(q_cu) - 1):
                start, end = q_cu[i], q_cu[i + 1]
                q_list.append(q[start:end])  # [S_q, N, D]
            if seqused_k is not None:
                k_seqlens = seqused_k.tolist()
            else:
                k_seqlens = [max_seqlen_k] * len(q_list)
            # Gather k/v from paged cache
            block_size = key_cache.shape[1]
            for b in range(len(q_list)):
                q_b = q_list[b]  # [S_q, N, D]
                S_k = k_seqlens[b]
                # Gather k/v for this batch
                bt = attn_metadata.block_table[b] if attn_metadata.block_table is not None else None
                k_b = torch.zeros(S_k, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
                v_b = torch.zeros(S_k, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
                token_idx = 0
                if bt is not None:
                    for block_idx in bt:
                        for offset in range(block_size):
                            if token_idx >= S_k:
                                break
                            k_b[token_idx] = key_cache[block_idx, offset]
                            v_b[token_idx] = value_cache[block_idx, offset]
                            token_idx += 1
                else:
                    for t in range(S_k):
                        k_b[t] = key_cache[t // block_size, t % block_size]
                        v_b[t] = value_cache[t // block_size, t % block_size]
                k_list.append(k_b)
                v_list.append(v_b)
            # Pad to max lengths
            max_q = max(x.shape[0] for x in q_list) if q_list else 0
            max_k = max(x.shape[0] for x in k_list) if k_list else 0
            q_padded = torch.zeros(len(q_list), num_heads, max_q, head_dim, dtype=q.dtype, device=q.device)
            k_padded = torch.zeros(len(k_list), num_kv_heads, max_k, head_dim, dtype=q.dtype, device=q.device)
            v_padded = torch.zeros(len(k_list), num_kv_heads, max_k, head_dim, dtype=q.dtype, device=q.device)
            for b in range(len(q_list)):
                # q/k/v are [S, N, D] but padded tensors are [B, N, S, D]
                q_padded[b, :, :q_list[b].shape[0]] = q_list[b].transpose(0, 1)  # [S, N, D] -> [N, S, D]
                k_padded[b, :, :k_list[b].shape[0]] = k_list[b].transpose(0, 1)  # [S, N_kv, D] -> [N_kv, S, D]
                v_padded[b, :, :v_list[b].shape[0]] = v_list[b].transpose(0, 1)  # [S, N_kv, D] -> [N_kv, S, D]
            # Expand q heads for GQA by repeating k/v heads to match q heads
            if num_heads != num_kv_heads:
                H_q = num_heads // num_kv_heads
                # Repeat k/v heads H_q times so they match q heads: [B, N_kv, S, D] -> [B, N_q, S, D]
                k_padded = k_padded.repeat_interleave(H_q, dim=1)
                v_padded = v_padded.repeat_interleave(H_q, dim=1)
                logger.debug(
                    "[beam_attn NPU] _sdpa_attention GQA: H_q=%d, q_padded=%s, k_padded=%s, v_padded=%s",
                    H_q, q_padded.shape, k_padded.shape, v_padded.shape
                )
            # For now, use the first S_q tokens for each batch
            out = F.scaled_dot_product_attention(
                q_padded, k_padded, v_padded, is_causal=causal
            )
            logger.debug(
                "[beam_attn NPU] _sdpa_attention: out shape=%s (should be [B, N, S, D] or [B, N_kv, S, H_q*D] for GQA)",
                out.shape
            )
            # Extract back
            result = torch.zeros_like(q)
            offset = 0
            for b in range(len(q_list)):
                sq = q_list[b].shape[0]
                # SDPA output is [B, N, S, D]
                # Reshape to [sq, num_heads, head_dim]
                out_b = out[b, :, :sq].permute(1, 0, 2).reshape(sq, num_heads, head_dim)
                end = offset + sq
                result[offset:end] = out_b
                offset = end
            logger.debug("[beam_attn NPU] _sdpa_attention: result range [%s, %s]", float(result.min()), float(result.max()))
            return result

        def _sdpa_attention_prefix(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            seq_lens_k: torch.Tensor,
            max_seqlen_q: int,
            max_seqlen_k: int,
            block_table: torch.Tensor | None,
        ) -> torch.Tensor:
            """
            Prefix attention for mega path on NPU.
            k/v are already the full KV cache (from block_table indexing).
            q: [T, N, D], k/v: [T, num_kv_heads, D]
            """
            block_size = key_cache.shape[1]
            q_cu = cu_seqlens_q.tolist()
            seq_lens_k_list = seq_lens_k.tolist() if seq_lens_k is not None else [max_seqlen_k] * len(q_cu)
            # Gather full k/v for each batch item using block table
            k_packed, v_packed = [], []
            for b in range(len(q_cu) - 1):
                seq_k = seq_lens_k_list[b]
                k_b = torch.zeros(seq_k, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
                v_b = torch.zeros(seq_k, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
                bt = block_table[b] if block_table is not None else None
                for t in range(seq_k):
                    slot = t
                    if bt is not None:
                        block_idx = slot // block_size
                        offset = slot % block_size
                        slot = bt[block_idx].item() * block_size + offset
                    b_idx, off = divmod(slot, block_size)
                    k_b[t] = key_cache[b_idx, off]
                    v_b[t] = value_cache[b_idx, off]
                k_packed.append(k_b)
                v_packed.append(v_b)
            max_k = max(x.shape[0] for x in k_packed) if k_packed else 0
            max_q = max(q_cu[i + 1] - q_cu[i] for i in range(len(q_cu) - 1)) if q_cu else 0
            q_padded = torch.zeros(len(k_packed), num_heads, max_q, head_dim, dtype=q.dtype, device=q.device)
            k_padded = torch.zeros(len(k_packed), num_kv_heads, max_k, head_dim, dtype=q.dtype, device=q.device)
            v_padded = torch.zeros(len(k_packed), num_kv_heads, max_k, head_dim, dtype=q.dtype, device=q.device)
            for b in range(len(k_packed)):
                sq = q_cu[b + 1] - q_cu[b]
                q_slice = q[q_cu[b]:q_cu[b + 1]].permute(1, 0, 2)  # [num_heads, sq, head_dim]
                q_padded[b, :, :sq] = q_slice
                k_padded[b, :, :k_packed[b].shape[0]] = k_packed[b]
                v_padded[b, :, :v_packed[b].shape[0]] = v_packed[b]
            if num_heads != num_kv_heads:
                # For GQA: reshape from [B, N_q, S, D] -> [B, N_kv, H_q, S, D] -> [B, N_kv, S, H_q, D] -> [B, N_kv, S, H_q*D]
                H_q = num_heads // num_kv_heads
                q_padded = q_padded.view(len(k_packed), num_kv_heads, H_q, max_q, head_dim)
                q_padded = q_padded.permute(0, 1, 3, 2, 4).reshape(len(k_packed), num_kv_heads, max_q, H_q * head_dim)
                logger.debug(
                    "[beam_attn NPU] GQA: H_q=%d, q_padded shape=%s, k_padded shape=%s",
                    H_q, q_padded.shape, k_padded.shape
                )
            logger.debug("[beam_attn NPU] _sdpa_attention_prefix: q_padded=%s, k_padded=%s, out_before_reshape=%s",
                q_padded.shape, k_padded.shape, out.shape if 'out' in dir() else 'N/A')
            out = F.scaled_dot_product_attention(q_padded, k_padded, v_padded, is_causal=False)
            logger.debug("[beam_attn NPU] _sdpa_attention_prefix: out shape=%s", out.shape)
            result = torch.zeros_like(q)
            offset = 0
            for b in range(len(k_packed)):
                sq = q_cu[b + 1] - q_cu[b]
                if num_heads == num_kv_heads:
                    out_b = out[b, :, :sq].permute(1, 0, 2).reshape(sq, num_heads, head_dim)
                else:
                    # GQA: out is [B, N_kv, S, H_q*D]
                    # Reshape [N_kv, sq, H_q*D] -> [sq, N_kv, H_q, D] -> [sq, N_q, D]
                    H_q = num_heads // num_kv_heads
                    out_b = out[b, :, :sq]  # [N_kv, sq, H_q*D]
                    out_b = out_b.permute(1, 0, 2)  # [sq, N_kv, H_q*D]
                    out_b = out_b.reshape(sq, num_kv_heads, H_q, head_dim)  # [sq, N_kv, H_q, D]
                    out_b = out_b.reshape(sq, num_heads, head_dim)  # [sq, N_q, D]
                result[offset:offset + sq] = out_b
                offset += sq
            return result
        '''


        # DEBUG
        logger.debug(
            "[beam_attn NPU] std_indices=%s, mega_prefix_indices=%s, "
            "mega_suffix_total=%s, num_heads=%d, num_kv_heads=%d, head_dim=%d, query shape=%s",
            attn_metadata.std_indices.shape if attn_metadata.std_indices is not None else None,
            attn_metadata.mega_prefix_indices.shape if attn_metadata.mega_prefix_indices is not None else None,
            attn_metadata.mega_suffix_total_tokens,
            num_heads, num_kv_heads, head_dim, query.shape,
        )
        if attn_metadata.std_indices is not None:
            logger.debug(
                "[beam_attn NPU] std cu_seqlens=%s, seq_lens=%s, block_table=%s",
                attn_metadata.std_cu_seqlens_q.tolist() if attn_metadata.std_cu_seqlens_q is not None else None,
                attn_metadata.std_seq_lens.tolist() if attn_metadata.std_seq_lens is not None else None,
                attn_metadata.std_block_table.shape if attn_metadata.std_block_table is not None else None,
            )
        if attn_metadata.mega_prefix_indices is not None:
            logger.debug(
                "[beam_attn NPU] mega cu_seqlens_q=%s, seq_lens=%s, block_table shape=%s",
                attn_metadata.mega_prefix_cu_seqlens_q.tolist() if attn_metadata.mega_prefix_cu_seqlens_q is not None else None,
                attn_metadata.mega_prefix_seq_lens.tolist() if attn_metadata.mega_prefix_seq_lens is not None else None,
                attn_metadata.mega_prefix_block_table.shape if attn_metadata.mega_prefix_block_table is not None else None,
            )
            logger.debug(
                "[beam_attn NPU] mega_prefix_indices[:20]=%s",
                attn_metadata.mega_prefix_indices[:20].tolist()
                if attn_metadata.mega_prefix_indices is not None else None,
            )
            logger.debug(
                "[beam_attn NPU] mega_suffix cu_seqlens_q=%s, cu_seqlens_k=%s, total_tokens=%d",
                attn_metadata.mega_suffix_cu_seqlens_q.tolist() if attn_metadata.mega_suffix_cu_seqlens_q is not None else None,
                attn_metadata.mega_suffix_cu_seqlens_k.tolist() if attn_metadata.mega_suffix_cu_seqlens_k is not None else None,
                attn_metadata.mega_suffix_total_tokens,
            )
            logger.debug(
                "[beam_attn NPU] mega_suffix_slot_mapping_out[:20]=%s",
                attn_metadata.mega_suffix_slot_mapping_out[:20].tolist()
                if attn_metadata.mega_suffix_slot_mapping_out is not None else None,
            )
            logger.debug(
                "[beam_attn NPU] mega_prefix_block_table[:3, :8]=%s",
                attn_metadata.mega_prefix_block_table[:3, :8].tolist()
                if attn_metadata.mega_prefix_block_table is not None else None,
            )

        # Standard attention
        if attn_metadata.std_indices is not None:
            std_query = query[attn_metadata.std_indices]
            logger.debug(
                "[beam_attn NPU] std query shape=%s, key_cache shape=%s",
                std_query.shape, key_cache.shape
            )
            logger.debug("key_cache: %s key_cache_squeezed: %s",
                key_cache.shape, key_cache.view(*key_cache.shape[:-2], -1).shape)

            logger.debug(f"std_cu_seqlens_q.shape: {list(attn_metadata.std_cu_seqlens_q.shape)} | std_cu_seqlens_q.numel: {attn_metadata.std_cu_seqlens_q.numel()} | std_cu_seqlens_q.content: {attn_metadata.std_cu_seqlens_q.cpu().tolist()}")

            std_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
                std_query, key_cache.view(*key_cache.shape[:-2], -1), value_cache.view(*value_cache.shape[:-2], -1),
                num_query_heads=attn_metadata.num_heads,
                num_key_value_heads=attn_metadata.num_kv_heads,
                actual_seq_qlen=attn_metadata.std_cu_seqlens_q.tolist(),
                actual_seq_kvlen=attn_metadata.std_seq_lens.tolist(),
                softmax_scale=scale,
                block_table=attn_metadata.std_block_table,
                block_size=attn_metadata.block_size,
                sparse_mode=0,
                inner_precise=0,
                input_layout="TND",
                return_softmax_lse=False,
            )

            logger.debug("[beam_attn NPU] std_out shape=%s", std_out.shape)
            output[attn_metadata.std_indices] = std_out

        # Mega attention
        if attn_metadata.mega_prefix_indices is not None:
            mega_query = query[attn_metadata.mega_prefix_indices]
            logger.debug(
                "[beam_attn NPU] mega_query shape=%s, key_cache shape=%s",
                mega_query.shape, key_cache.shape
            )

            print(
                "[beam_attn NPU DIAG] mega_query.shape=%s | "
                "mega_prefix_cu_seqlens_q=%s | mega_prefix_seq_lens=%s | "
                "mega_prefix_block_table.shape=%s | block_size=%s | "
                "num_heads=%s num_kv_heads=%s"
                % (
                    list(mega_query.shape),
                    attn_metadata.mega_prefix_cu_seqlens_q.tolist(),
                    attn_metadata.mega_prefix_seq_lens.tolist(),
                    list(attn_metadata.mega_prefix_block_table.shape),
                    attn_metadata.block_size,
                    attn_metadata.num_heads,
                    attn_metadata.num_kv_heads,
                ),
                flush=True,
            )

            mega_prefix_out, mega_prefix_lse = torch_npu.npu_fused_infer_attention_score_v2(
                mega_query,
                key_cache.view(*key_cache.shape[:-2], -1),
                value_cache.view(*value_cache.shape[:-2], -1),
                num_query_heads=attn_metadata.num_heads,
                num_key_value_heads=attn_metadata.num_kv_heads,
                actual_seq_qlen=attn_metadata.mega_prefix_cu_seqlens_q.tolist(),
                actual_seq_kvlen=attn_metadata.mega_prefix_seq_lens.tolist(),
                softmax_scale=scale,
                block_table=attn_metadata.mega_prefix_block_table,
                block_size=attn_metadata.block_size,
                sparse_mode=0,
                inner_precise=0,
                input_layout="TND",
                return_softmax_lse=True,
            )

            logger.debug(
                "[beam_attn NPU] mega_prefix_out shape=%s, range [%.6f, %.6f], mega_suffix_total=%d",
                mega_prefix_out.shape, float(mega_prefix_out.min()), float(mega_prefix_out.max()),
                attn_metadata.mega_suffix_total_tokens
            )

            if attn_metadata.mega_suffix_total_tokens > 0:
                slot_mapping = attn_metadata.slot_mapping[attn_metadata.mega_prefix_indices]
                logger.debug(
                    "[beam_attn NPU] slot_mapping[:20]=%s, mega_suffix_slot_out[:20]=%s",
                    slot_mapping[:20].tolist(),
                    attn_metadata.mega_suffix_slot_mapping_out[:20].tolist(),
                )
                contig_k, contig_v = extract_suffix_kv_pytorch(
                    key_cache,
                    value_cache,
                    slot_mapping,
                    attn_metadata.mega_suffix_slot_mapping_out
                )
                logger.debug(
                    "[beam_attn NPU] contig_k shape=%s, contig_v shape=%s",
                    contig_k.shape, contig_v.shape
                )

                mega_suffix_out, mega_suffix_lse = torch_npu.npu_fused_infer_attention_score_v2(
                    mega_query, contig_k, contig_v,
                    num_query_heads=attn_metadata.num_heads,
                    num_key_value_heads=attn_metadata.num_kv_heads,
                    actual_seq_qlen=attn_metadata.mega_suffix_cu_seqlens_q.tolist(),
                    actual_seq_kvlen=attn_metadata.mega_suffix_cu_seqlens_k.tolist(),
                    softmax_scale=scale,
                    block_table=None,  # Using contiguous format, not paged
                    block_size=0,  # 0 = non-paged (schema requires int, not None)
                    sparse_mode=0,  # Standard attention
                    inner_precise=0,  # FP32 internal precision
                    input_layout="TND",
                    return_softmax_lse=True
                )

                logger.debug(
                    "[beam_attn NPU] mega_suffix_out shape=%s, range [%.6f, %.6f]",
                    mega_suffix_out.shape, float(mega_suffix_out.min()), float(mega_suffix_out.max())
                )
                # NPU FIA returns LSE in TND layout [T, N, 1], so use the TND
                # merge (merge_attn_states_pytorch expects the flash 2-D
                # [num_heads, total_q] convention used on the GPU path).
                mega_out, _ = merge_tnd_attention_outputs(
                    mega_prefix_out, mega_prefix_lse, mega_suffix_out, mega_suffix_lse
                )

                logger.debug(
                    "[beam_attn NPU] merged mega_out range [%.6f, %.6f]",
                    float(mega_out.min()), float(mega_out.max())
                )
                logger.debug(
                    "[beam_attn NPU] merged_out range [%s, %s]",
                    float(mega_out.min()), float(mega_out.max())
                )
                output[attn_metadata.mega_prefix_indices] = mega_out
            else:
                output[attn_metadata.mega_prefix_indices] = mega_prefix_out

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
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

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
        if kv_cache_dtype.startswith("fp8"):
            if NPU_DEVICE:
                return False  # fp8 kv-cache not yet supported on NPU
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto"]

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
