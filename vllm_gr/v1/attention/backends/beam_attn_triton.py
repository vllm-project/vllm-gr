# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Beam-Consecutive triton helper."""

import torch
import triton
import triton.language as tl


@triton.jit
def paged_kv_to_contig_suffix_kernel(
    K_CACHE_PTR,
    V_CACHE_PTR,
    SLOT_MAPPING_OUT_PTR,
    OUT_K_PTR,
    OUT_V_PTR,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ok_tok,
    stride_ok_h,
    stride_ok_d,
    stride_ov_tok,
    stride_ov_h,
    stride_ov_d,
    head_dim,
    BLOCK_SIZE: tl.constexpr,  # vLLM cache page block size (e.g., 16)
    BLOCK_SIZE_D: tl.constexpr,  # Next power of 2 greater than or equal to head_dim
):
    pid_tok = tl.program_id(0)
    pid_h = tl.program_id(1)

    # 1. Fetch indirect layout indices safely
    slot_idx = tl.load(SLOT_MAPPING_OUT_PTR + pid_tok)

    # 2. Decompose global slot into robust paged coordinate structures
    block_id = slot_idx // BLOCK_SIZE
    block_offset = slot_idx % BLOCK_SIZE

    # 3. Create masked vector tracking offsets for head_dim
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < head_dim

    # 4. Correct 4D paged coordinate jumping math
    k_ptrs = (
        K_CACHE_PTR
        + block_id * stride_kb
        + block_offset * stride_kt
        + pid_h * stride_kh
        + d_offsets * stride_kd
    )

    v_ptrs = (
        V_CACHE_PTR
        + block_id * stride_vb
        + block_offset * stride_vt
        + pid_h * stride_vh
        + d_offsets * stride_vd
    )

    # Load vectors with mask protection
    k = tl.load(k_ptrs, mask=d_mask, other=0.0)
    v = tl.load(v_ptrs, mask=d_mask, other=0.0)

    # 5. Continuous out-packing target calculation
    ok_ptrs = OUT_K_PTR + pid_tok * stride_ok_tok + pid_h * stride_ok_h + d_offsets * stride_ok_d
    ov_ptrs = OUT_V_PTR + pid_tok * stride_ov_tok + pid_h * stride_ov_h + d_offsets * stride_ov_d

    tl.store(ok_ptrs, k, mask=d_mask)
    tl.store(ov_ptrs, v, mask=d_mask)


# ---------------------------------------------------------------------------
# Python Launcher
# ---------------------------------------------------------------------------
def extract_suffix_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping_out: torch.Tensor,
    out_token_num: int,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract suffix key and value caches from the paged block KV cache.

    This function calls the Triton kernel `paged_kv_to_contig_suffix_kernel`
    to fetch key and value projections for the suffix tokens from the paged
    block caches, yielding contiguous tensor outputs suitable for flash attention.

    Args:
        k_cache: Key cache tensor of shape [num_blocks, block_size, num_heads, head_dim].
        v_cache: Value cache tensor of shape [num_blocks, block_size, num_heads, head_dim].
        slot_mapping_out: A 1D tensor of shape [out_token_num] containing slot mappings
            that map suffix tokens to their corresponding physical slots in the cache.
        out_token_num: Number of output/suffix tokens to extract.
        block_size: Block size of the vLLM cache page (default: 16).

    Returns:
        A tuple (out_k, out_v) of contiguous tensors, each of shape
        [out_token_num, num_heads, head_dim], representing the extracted
        key and value caches for the suffix tokens.
    """
    nheads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    total_tokens = out_token_num

    # Calculate next power of 2 for Triton compiler stability
    block_size_d = 1 << (head_dim - 1).bit_length()

    out_kv = torch.empty(
        (2, total_tokens, nheads, head_dim), device=k_cache.device, dtype=k_cache.dtype
    )
    out_k, out_v = out_kv[0], out_kv[1]

    grid = (total_tokens, nheads)
    paged_kv_to_contig_suffix_kernel[grid](
        k_cache,
        v_cache,
        slot_mapping_out,
        out_k,
        out_v,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        head_dim=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_D=block_size_d,
    )
    return out_k, out_v
