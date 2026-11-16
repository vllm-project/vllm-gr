# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""GR Prefill inputs and metadata, written to native ModelRunner buffers."""

import triton
import triton.language as tl

# Triton pointer parameters and constexpr annotations are handled by its JIT.
# mypy: disable-error-code="misc,no-untyped-def"


@triton.jit(do_not_specialize=["req_index", "num_tokens", "num_blocks"])
def apply_gr_prefill_state_writes_v2(
    packed,
    total_lens,
    all_token_ids,
    num_computed_tokens,
    block_table,
    req_index,
    num_tokens,
    num_blocks,
    TOKEN_STRIDE: tl.constexpr,
    BLOCK_TABLE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write one fresh request's native state from packed int32 contents.

    all_token_ids retains its native UVA allocation. GPU writes stay ordered
    before subsequent input preparation on the same stream, as in apply_write.
    """
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = req_index.to(tl.int64)
    tokens = tl.load(packed + 2 + index, index < num_tokens, other=0)
    tl.store(all_token_ids + row * TOKEN_STRIDE + index, tokens, index < num_tokens)
    block_ids = tl.load(packed + 2 + num_tokens + index, index < num_blocks, other=0)
    tl.store(block_table + row * BLOCK_TABLE_STRIDE + index, block_ids, index < num_blocks)
    if tl.program_id(0) == 0:
        tl.store(total_lens + row, tl.load(packed))
        tl.store(num_computed_tokens + row, tl.load(packed + 1))


@triton.jit
def apply_gr_prefill_batch_state_writes_v2(
    packed,
    total_lens,
    all_token_ids,
    num_computed_tokens,
    block_table,
    TOKEN_STRIDE: tl.constexpr,
    BLOCK_TABLE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Flush one native registration write per row from a packed batch."""
    header = packed + tl.program_id(0) * 7
    row = tl.load(header).to(tl.int64)
    token_start = tl.load(header + 3)
    token_count = tl.load(header + 4)
    block_start = tl.load(header + 5)
    block_count = tl.load(header + 6)
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tokens = tl.load(packed + token_start + index, index < token_count, other=0)
    tl.store(all_token_ids + row * TOKEN_STRIDE + index, tokens, index < token_count)
    blocks = tl.load(packed + block_start + index, index < block_count, other=0)
    tl.store(block_table + row * BLOCK_TABLE_STRIDE + index, blocks, index < block_count)
    if tl.program_id(1) == 0:
        tl.store(total_lens + row, tl.load(header + 1))
        tl.store(num_computed_tokens + row, tl.load(header + 2))


@triton.jit(do_not_specialize=["computed", "scheduled", "num_tokens"])
def prepare_gr_prefill_metadata(
    positions,
    query_start_loc,
    seq_lens,
    num_computed_tokens,
    req_indices,
    query_pos,
    num_scheduled_tokens,
    discard_request_mask,
    num_accepted_tokens,
    logits_indices,
    computed,
    scheduled,
    num_tokens,
    MAX_REQS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(positions + index, computed + index.to(tl.int64), index < scheduled)
    tl.store(req_indices + index, 0, index < scheduled)
    tl.store(query_pos + index, index, index < scheduled)
    # Native attention requires a nondecreasing padded query_start_loc.
    tl.store(query_start_loc + index, tl.where(index == 0, 0, scheduled), index <= MAX_REQS)
    tl.store(seq_lens + index, tl.where(index == 0, computed + scheduled, 0), index < MAX_REQS)
    tl.store(num_accepted_tokens + index, 1, index < MAX_REQS)
    tl.store(num_computed_tokens + index, computed, index == 0)
    tl.store(num_scheduled_tokens + index, scheduled, index == 0)
    tl.store(discard_request_mask + index, computed + scheduled < num_tokens, index == 0)
    tl.store(logits_indices + index, scheduled - 1, index == 0)


@triton.jit(do_not_specialize=["req_index", "scheduled"])
def prepare_gr_prefill_inputs_v2(
    input_ids,
    positions,
    query_start_loc,
    seq_lens,
    all_token_ids,
    num_computed_tokens,
    prefill_lens,
    next_prefill_tokens,
    batch_indices,
    logits_indices,
    req_index,
    scheduled,
    TOKEN_STRIDE: tl.constexpr,
    MAX_REQS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fuse native V2 pure-prefill input preparation for a single GR row."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Match the native kernels: GPU state, not its optimistic CPU mirror,
    # determines the tokens/positions consumed by this dispatch.
    computed = tl.load(num_computed_tokens + req_index)
    request_tokens = all_token_ids + req_index.to(tl.int64) * TOKEN_STRIDE
    tokens = tl.load(request_tokens + computed + index, index < scheduled, other=0)
    tl.store(input_ids + index, tokens, index < scheduled)
    tl.store(positions + index, computed + index.to(tl.int64), index < scheduled)
    tl.store(query_start_loc + index, tl.where(index == 0, 0, scheduled), index <= MAX_REQS)
    tl.store(seq_lens + index, tl.where(index == 0, computed + scheduled, 0), index < MAX_REQS)

    if tl.program_id(0) == 0:
        # Fresh per-dispatch storage, retained by InputBatch; it is never
        # overwritten while asynchronous sampling/post-update can read it.
        tl.store(batch_indices, req_index)
        tl.store(batch_indices + 1, 0)
        tl.store(batch_indices + 2, 1)
        tl.store(batch_indices + 3, 0)
        tl.store(logits_indices, scheduled - 1)
        next_pos = computed + scheduled
        if next_pos < tl.load(prefill_lens + req_index):
            tl.store(next_prefill_tokens + req_index, tl.load(request_tokens + next_pos))


@triton.jit(do_not_specialize=["num_tokens", "padded_tokens"])
def prepare_gr_prefill_batch_inputs_v2(
    input_ids,
    positions,
    query_start_loc,
    seq_lens,
    is_padding,
    all_token_ids,
    num_computed_tokens,
    prefill_lens,
    next_prefill_tokens,
    descriptors,
    idx_mapping,
    batch_indices,
    logits_indices,
    num_tokens,
    padded_tokens,
    NUM_REQS: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    MAX_REQS: tl.constexpr,
    WRITE_PADDING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Prepare native ordered Prefill rows from one packed descriptor copy."""
    row = tl.program_id(0)
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    req_index = tl.load(descriptors + row)
    start = tl.load(descriptors + NUM_REQS + row)
    end = tl.load(descriptors + NUM_REQS + row + 1)
    scheduled = end - start
    computed = tl.load(num_computed_tokens + req_index)
    request_tokens = all_token_ids + req_index.to(tl.int64) * TOKEN_STRIDE
    tokens = tl.load(request_tokens + computed + index, index < scheduled, other=0)
    tl.store(input_ids + start + index, tokens, index < scheduled)
    tl.store(positions + start + index, computed + index.to(tl.int64), index < scheduled)
    if WRITE_PADDING:
        tl.store(is_padding + start + index, False, index < scheduled)

    if tl.program_id(1) == 0:
        tl.store(query_start_loc + row, start)
        tl.store(seq_lens + row, computed + scheduled)
        tl.store(idx_mapping + row, req_index)
        tl.store(batch_indices + row, row)
        tl.store(batch_indices + NUM_REQS + 1 + row, 0)
        tl.store(logits_indices + row, end - 1)
        next_pos = computed + scheduled
        next_token = tl.load(
            request_tokens + next_pos, next_pos < tl.load(prefill_lens + req_index), other=0
        )
        tl.store(next_prefill_tokens + req_index, next_token)

    if row == 0:
        # One row owns all padding: monotonic query offsets and zero sequence
        # lengths for dummy requests, without racing real rows above.
        tl.store(query_start_loc + index, num_tokens, (index >= NUM_REQS) & (index <= MAX_REQS))
        tl.store(seq_lens + index, 0, (index >= NUM_REQS) & (index < MAX_REQS))
        if WRITE_PADDING:
            tl.store(is_padding + num_tokens + index, True, num_tokens + index < padded_tokens)
        if tl.program_id(1) == 0:
            tl.store(batch_indices + NUM_REQS, NUM_REQS)
