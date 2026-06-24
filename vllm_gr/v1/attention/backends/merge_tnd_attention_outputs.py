import torch


def extract_suffix_kv_pytorch(k_cache, v_cache, slot_mapping, slot_mapping_out):
    """
    Pure PyTorch equivalent of the paged_kv_to_contig_suffix_kernel.
    Replaces indirect pointer math with vectorized advanced indexing.
   
    Args:
        k_cache: Tensor of shape [num_blocks, block_size, num_heads, head_dim]
        v_cache: Tensor of shape [num_blocks, block_size, num_heads, head_dim]
        slot_mapping: Global slot allocation tensor
        slot_mapping_out: Indices indicating which slots to extract
    """
    # 1. Replicate the indirect lookup: src_idx -> slot_idx
    # In Triton: src_idx = tl.load(SLOT_MAPPING_OUT_PTR + pid_tok)
    #            slot_idx = tl.load(SLOT_MAPPING_PTR + src_idx)
    resolved_slots = slot_mapping[slot_mapping_out.long()]
 
    # 2. Collapse paged blocks [num_blocks, block_size, heads, dim]
    # into a flat token space [total_physical_slots, heads, dim]
    num_heads, head_dim = k_cache.shape[2], k_cache.shape[3]
    k_flat = k_cache.view(-1, num_heads, head_dim)
    v_flat = v_cache.view(-1, num_heads, head_dim)
 
    # 3. Vectorized gather: Selects all matching token rows across all heads/dimensions
    out_k = k_flat[resolved_slots]
    out_v = v_flat[resolved_slots]
 
    return out_k, out_v

def merge_tnd_attention_outputs(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
):
    """
    Normalizes and merges two TND attention output pairs.

    Expected input dimensions:
        out1, out2: [T, N, D]  (Token, NumHead, HeadDim)
        lse1, lse2: [T, N, 1]  (output of _npu_normalize_lse)
    """
    import logging as _logging
    _logger = _logging.getLogger("vllm_gr.merge_tnd")
    _logger.info(
        "[merge_tnd] lse1 range [%.4f, %.4f], lse2 range [%.4f, %.4f]",
        float(lse1.min()), float(lse1.max()),
        float(lse2.min()), float(lse2.max()),
    )
    _logger.info(
        "[merge_tnd] out1 range [%.4f, %.4f], out2 range [%.4f, %.4f]",
        float(out1.min()), float(out1.max()),
        float(out2.min()), float(out2.max()),
    )

    # LSE is already [T, N, 1] from _npu_normalize_lse, matching out1/out2 layout [T, N, D]

    # 1. Extract max values for numerical stability
    max_lse = torch.maximum(lse1, lse2)  # [T, N, 1]

    # 2. Compute exponential weights (relative to max for numerical stability)
    w1 = torch.exp(lse1 - max_lse)  # [T, N, 1]
    w2 = torch.exp(lse2 - max_lse)  # [T, N, 1]
    w_total = w1 + w2                # [T, N, 1]

    # 3. Compute weighted attention sum - shapes align: [T, N, D] * [T, N, 1] broadcasts
    normalized_out = (w1 * out1 + w2 * out2) / w_total

    # 4. Compute merged LSE
    normalized_lse = max_lse + torch.log(w_total)  # [T, N, 1]

    _logger.info(
        "[merge_tnd] w1=%.4f..%.4f, w2=%.4f..%.4f, w_total=%.4f..%.4f",
        float(w1.min()), float(w1.max()),
        float(w2.min()), float(w2.max()),
        float(w_total.min()), float(w_total.max()),
    )
    _logger.info(
        "[merge_tnd] normalized_out range [%.4f, %.4f], normalized_lse range [%.4f, %.4f]",
        float(normalized_out.min()), float(normalized_out.max()),
        float(normalized_lse.min()), float(normalized_lse.max()),
    )

    return normalized_out, normalized_lse



def merge_attn_states_pytorch(out, prefix_out, prefix_lse, suffix_out, suffix_lse):
    """
    Pure PyTorch exact replacement for the custom merge_attn_states kernel.
    Modifies the 'out' tensor in-place to match the C++ signature layout.

    Signature Layout:
        out: Pre-allocated target tensor [total_q, num_heads, head_dim]
        prefix_out, suffix_out: Partial attention outputs [total_q, num_heads, head_dim]
        prefix_lse, suffix_lse: Attention scales [num_heads, total_q]
    """
    # 1. Align the LSE tensors for clean 3D broadcasting
    # [num_heads, total_q] -> [total_q, num_heads, 1]
    lse_p = prefix_lse.t().unsqueeze(-1)
    lse_s = suffix_lse.t().unsqueeze(-1)

    # 2. Numerical Stability: Subtract element-wise maximum LSE
    max_lse = torch.maximum(lse_p, lse_s)

    # 3. Compute stabilized exponential weights
    weight_p = torch.exp(lse_p - max_lse)
    weight_s = torch.exp(lse_s - max_lse)
    weight_sum = weight_p + weight_s

    # 4. In-place write to the pre-allocated out tensor
    # .copy_() writes directly to out's existing memory pointer
    out.copy_((prefix_out * weight_p + suffix_out * weight_s) / weight_sum)

    return out
