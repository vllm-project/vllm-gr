# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request Prefill metadata, written to native ModelRunner buffers."""

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
