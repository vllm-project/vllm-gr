# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Functional guards for Beam Attention metadata ownership."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
    BeamAttentionMetadataBuilder,
)
from vllm_gr.v1.beam.context import BEAM_DATA_VAR


def _make_builder(block_size: int = 2) -> BeamAttentionMetadataBuilder:
    builder = object.__new__(BeamAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.block_size = block_size
    builder.aot_schedule = False
    builder.aot_sliding_window = None
    builder.use_full_cuda_graph = False
    builder.max_cudagraph_size = None
    builder.max_num_splits = 0
    builder.num_heads_q = 8
    builder.num_kv_heads = 2
    builder._is_cuda = False
    builder._is_npu = False
    builder.npu_causal_atten_mask = None
    return builder


def _common_metadata(
    query_starts: list[int],
    seq_lens: list[int],
    block_table: list[list[int]],
) -> SimpleNamespace:
    num_tokens = query_starts[-1]
    return SimpleNamespace(
        num_reqs=len(seq_lens),
        query_start_loc_cpu=torch.tensor(query_starts, dtype=torch.int32),
        query_start_loc=torch.tensor(query_starts, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        block_table_tensor=torch.tensor(block_table, dtype=torch.int32),
        num_actual_tokens=num_tokens,
        max_query_len=max(
            query_starts[index + 1] - query_starts[index] for index in range(len(query_starts) - 1)
        ),
        max_seq_len=max(seq_lens),
        causal=True,
        slot_mapping=torch.arange(100, 100 + num_tokens, dtype=torch.int64),
    )


def _build_metadata(
    common: SimpleNamespace,
    req_ids: list[str],
    beam_data: dict[str, dict[str, Any]],
) -> BeamAttentionMetadata:
    token = BEAM_DATA_VAR.set({"beam_data": beam_data, "req_ids": req_ids})
    try:
        return _make_builder().build(0, common)
    finally:
        BEAM_DATA_VAR.reset(token)


@pytest.mark.parametrize(  # type: ignore[misc]
    (
        "query_starts",
        "seq_lens",
        "block_table",
        "req_ids",
        "beam_data",
        "expected",
    ),
    [
        pytest.param(
            [0, 4],
            [8],
            [[0, 1, 2, 3]],
            ["beam"],
            {
                "beam": {
                    "is_beam_decode": True,
                    "prefix_len": 4,
                    "beam_width": 2,
                    "beam_decode_steps": 2,
                    "cache_len": 4,
                }
            },
            {
                "std_indices": [],
                "prefix_indices": [0, 1, 2, 3],
                "std_q": [],
                "std_kv": [],
                "prefix_q": [4],
                "prefix_kv": [4],
                "suffix_q": [2, 2],
                "suffix_kv": [2, 2],
                "std_blocks": None,
                "prefix_blocks": [[0, 1]],
                "suffix_slots": [100, 101, 102, 103],
            },
            id="beam-only",
        ),
        pytest.param(
            [0, 2, 6],
            [3, 8],
            [[0, 1, 2, 3], [10, 11, 12, 13]],
            ["ordinary", "beam"],
            {
                "beam": {
                    "is_beam_decode": True,
                    "prefix_len": 4,
                    "beam_width": 2,
                    "beam_decode_steps": 2,
                    "cache_len": 4,
                }
            },
            {
                "std_indices": [0, 1],
                "prefix_indices": [2, 3, 4, 5],
                "std_q": [2],
                "std_kv": [3],
                "prefix_q": [4],
                "prefix_kv": [4],
                "suffix_q": [2, 2],
                "suffix_kv": [2, 2],
                "std_blocks": [[0, 1]],
                "prefix_blocks": [[10, 11]],
                "suffix_slots": [102, 103, 104, 105],
            },
            id="mixed-batch",
        ),
    ],
)
def test_build_routes_standard_and_beam_metadata(
    query_starts: list[int],
    seq_lens: list[int],
    block_table: list[list[int]],
    req_ids: list[str],
    beam_data: dict[str, dict[str, Any]],
    expected: dict[str, Any],
) -> None:
    metadata = _build_metadata(
        _common_metadata(query_starts, seq_lens, block_table),
        req_ids,
        beam_data,
    )

    std_indices = metadata.std_indices.tolist() if metadata.std_indices is not None else []
    prefix_indices = metadata.prefix_indices.tolist() if metadata.prefix_indices is not None else []
    assert std_indices == expected["std_indices"]
    assert prefix_indices == expected["prefix_indices"]
    assert set(std_indices).isdisjoint(prefix_indices)
    assert sorted(std_indices + prefix_indices) == list(range(query_starts[-1]))

    assert metadata.std_seqlens_q_cpu == expected["std_q"]
    assert metadata.std_seqlens_kv_cpu == expected["std_kv"]
    assert metadata.prefix_seqlens_q_cpu == expected["prefix_q"]
    assert metadata.prefix_seqlens_kv_cpu == expected["prefix_kv"]
    assert metadata.suffix_seqlens_q_cpu == expected["suffix_q"]
    assert metadata.suffix_seqlens_kv_cpu == expected["suffix_kv"]

    std_blocks = metadata.std_block_table.tolist() if metadata.std_block_table is not None else None
    prefix_blocks = (
        metadata.prefix_block_table.tolist() if metadata.prefix_block_table is not None else None
    )
    suffix_slots = (
        metadata.suffix_slot_mapping_out.tolist()
        if metadata.suffix_slot_mapping_out is not None
        else None
    )
    assert std_blocks == expected["std_blocks"]
    assert prefix_blocks == expected["prefix_blocks"]
    assert suffix_slots == expected["suffix_slots"]

    if expected["suffix_q"]:
        assert metadata.prefix_cu_seqlens_q_cpu == [
            sum(expected["prefix_q"][: index + 1]) for index in range(len(expected["prefix_q"]))
        ]
        assert metadata.suffix_cu_seqlens_q_cpu == [
            sum(expected["suffix_q"][: index + 1]) for index in range(len(expected["suffix_q"]))
        ]
        assert metadata.suffix_cu_seqlens_kv_cpu == [
            sum(expected["suffix_kv"][: index + 1]) for index in range(len(expected["suffix_kv"]))
        ]


def test_build_pure_prefill_fast_path_carries_common_fields() -> None:
    common = _common_metadata([0, 3, 7], [3, 7], [[0, 1], [2, 3]])
    metadata = _build_metadata(common, ["a", "b"], {})

    assert metadata.is_prefill_graph_exec is True
    assert metadata.num_reqs == 2
    assert metadata.num_heads == 8
    assert metadata.num_kv_heads == 2
    assert metadata.block_size == 2
    assert metadata.std_indices is None
    assert metadata.std_block_table is None
    assert metadata.prefix_indices is None
    assert metadata.query_start_loc_cpu == [0, 3, 7]
    assert metadata.seq_lens_cpu == [3, 7]
    assert torch.equal(metadata.query_start_loc, common.query_start_loc)
    assert torch.equal(metadata.seq_lens, common.seq_lens)
    assert torch.equal(metadata.block_table, common.block_table_tensor)


def test_update_block_table_returns_copy_and_refreshes_subtables() -> None:
    old_table = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    old_slots = torch.tensor([10, 11], dtype=torch.int64)
    metadata = BeamAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_seq_len=6,
        seq_lens=torch.tensor([3, 6], dtype=torch.int32),
        block_table=old_table,
        slot_mapping=old_slots,
        std_block_table=old_table[[1]],
        prefix_block_table=old_table[[0], :2],
        std_req_idxs_cpu=[1],
        prefix_req_idxs_cpu=[0],
        max_prefix_blocks_count=2,
    )
    new_table = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int32)
    new_slots = torch.tensor([20, 21], dtype=torch.int64)

    updated = _make_builder().update_block_table(metadata, new_table, new_slots)

    assert updated is not metadata
    assert metadata.block_table is old_table
    assert metadata.slot_mapping is old_slots
    assert updated.block_table is new_table
    assert updated.slot_mapping is new_slots
    assert updated.std_block_table.tolist() == [[10, 11, 12]]
    assert updated.prefix_block_table.tolist() == [[7, 8]]


def test_build_chunked_prefill_and_multistep_beam_metadata() -> None:
    query_starts = [0, 4]
    seq_lens = [8]
    block_table = [[0, 1, 2, 3]]
    req_ids = ["beam_chunked"]
    beam_data = {
        "beam_chunked": {
            "is_beam_decode": True,
            "prefix_len": 4,
            "beam_width": 2,
            "beam_decode_steps": 2,
            "cache_len": 5,
        }
    }
    common = _common_metadata(query_starts, seq_lens, block_table)
    metadata = _build_metadata(common, req_ids, beam_data)
    assert metadata is not None
    assert metadata.suffix_slot_mapping_out is not None
    # Assert exact expected physical slot sequence:
    # slot 4 for cached-past token at offset 4 + slots [100, 101, 102] for current query tokens
    assert metadata.suffix_slot_mapping_out.tolist() == [4, 100, 101, 102]


def test_chunked_prefill_multi_req_no_cuda_assert() -> None:
    # Scenario matching the crash: Req 0 (beam decode 32 tokens), Req 1 (multistep beam decode 1 token), Req 2 (ordinary 67 tokens)
    query_starts = [0, 32, 33, 100]
    seq_lens = [5191, 3810, 67]
    block_table = [[i for i in range(2500)] for _ in range(3)]
    req_ids = ["req0", "req1", "req2"]
    beam_data = {
        "req0": {
            "is_beam_decode": True,
            "prefix_len": 5159,
            "beam_width": 32,
            "beam_decode_steps": 1,
            "cache_len": 5159,
        },
        "req1": {
            "is_beam_decode": True,
            "prefix_len": 3747,
            "beam_width": 32,
            "beam_decode_steps": 2,
            "cache_len": 3810,
        },
    }
    common = _common_metadata(query_starts, seq_lens, block_table)
    metadata = _build_metadata(common, req_ids, beam_data)
    assert metadata is not None
    assert metadata.suffix_slot_mapping_out is not None
    # Assert exact expected physical slot sequence for multi-req chunked prefill:
    # slots 100..131 for req0 (32 query tokens) + slot 3809 for req1 cached-past token + slot 132 for req1 query token
    expected_slots = list(range(100, 132)) + [3809, 132]
    assert metadata.suffix_slot_mapping_out.tolist() == expected_slots


def test_update_block_table_refreshes_suffix_slots() -> None:
    query_starts = [0, 4]
    seq_lens = [8]
    block_table = [[0, 1, 2, 3]]
    req_ids = ["beam_chunked"]
    beam_data = {
        "beam_chunked": {
            "is_beam_decode": True,
            "prefix_len": 4,
            "beam_width": 2,
            "beam_decode_steps": 2,
            "cache_len": 5,
        }
    }
    common = _common_metadata(query_starts, seq_lens, block_table)
    builder = _make_builder()
    metadata = _build_metadata(common, req_ids, beam_data)
    assert metadata is not None
    assert metadata.suffix_slot_mapping_out is not None
    # Original physical slots: past slot at offset 4 (block 0 -> 4) + current query slots 100, 101, 102
    assert metadata.suffix_slot_mapping_out.tolist() == [4, 100, 101, 102]

    # Update block table (block 0 becomes block 10) and slot mapping (0,1,2 becomes 500,501,502)
    new_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    new_slots = torch.tensor([500, 501, 502, 503], dtype=torch.int64)

    updated = builder.update_block_table(metadata, new_table, new_slots)

    # Past slot offset 4 uses block 4 // 2 = 2 -> block ID 12 -> 12 * 2 + (4 % 2) = 24
    # Current query slots index positions 0, 1, 2 in new_slots -> 500, 501, 502
    assert updated.suffix_slot_mapping_out is not None
    assert updated.suffix_slot_mapping_out.tolist() == [24, 500, 501, 502]


def test_update_block_table_multireq_nonzero_row_suffix() -> None:
    # 2 requests: req0 (std decode), req1 (beam decode with past suffix)
    query_starts = [0, 2, 6]
    seq_lens = [4, 8]
    block_table = [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
    ]
    req_ids = ["req0", "req1_beam"]
    beam_data = {
        "req1_beam": {
            "is_beam_decode": True,
            "prefix_len": 4,
            "beam_width": 2,
            "beam_decode_steps": 2,
            "cache_len": 5,
        }
    }
    common = _common_metadata(query_starts, seq_lens, block_table)
    builder = _make_builder()
    metadata = _build_metadata(common, req_ids, beam_data)
    assert metadata is not None
    assert metadata.suffix_slot_mapping_out is not None
    # req1 past suffix at offset 4 uses req1 row (row 1), block 4 // 2 = 2 -> block ID 12 -> slot 12 * 2 + 0 = 24
    # req1 query slots in common.slot_mapping at indices 2, 3, 4 -> 102, 103, 104
    assert metadata.suffix_slot_mapping_out.tolist() == [24, 102, 103, 104]

    # Update with distinct new block IDs for row 0 vs row 1
    new_table = torch.tensor(
        [
            [100, 101, 102, 103],
            [200, 201, 202, 203],
        ],
        dtype=torch.int32,
    )
    new_slots = torch.tensor([500, 501, 502, 503, 504, 505], dtype=torch.int64)

    updated = builder.update_block_table(metadata, new_table, new_slots)
    assert updated.suffix_slot_mapping_out is not None
    # req1 past suffix at offset 4 uses updated row 1, block 2 -> block ID 202 -> slot 202 * 2 + 0 = 404
    # req1 query slots in new_slots at indices 2, 3, 4 -> 502, 503, 504
    assert updated.suffix_slot_mapping_out.tolist() == [404, 502, 503, 504]


def test_update_block_table_current_only_suffix() -> None:
    # Suffix containing only current query tokens (no past suffix)
    query_starts = [0, 3]
    seq_lens = [6]
    block_table = [[0, 1, 2]]
    req_ids = ["req_curr_suffix"]
    beam_data = {
        "req_curr_suffix": {
            "is_beam_decode": True,
            "prefix_len": 0,
            "beam_width": 3,
            "beam_decode_steps": 1,
            "cache_len": 0,
        }
    }
    common = _common_metadata(query_starts, seq_lens, block_table)
    builder = _make_builder()
    metadata = _build_metadata(common, req_ids, beam_data)
    assert metadata is not None
    assert metadata.suffix_slot_mapping_out is not None
    assert metadata.suffix_slot_mapping_out.tolist() == [100, 101, 102]

    # Update slot mapping
    new_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)
    new_slots = torch.tensor([900, 901, 902], dtype=torch.int64)

    updated = builder.update_block_table(metadata, new_table, new_slots)
    assert updated.suffix_slot_mapping_out is not None
    assert updated.suffix_slot_mapping_out.tolist() == [900, 901, 902]


def test_update_block_table_repeated_updates_immutability() -> None:
    query_starts = [0, 4]
    seq_lens = [8]
    block_table = [[0, 1, 2, 3]]
    req_ids = ["beam_req"]
    beam_data = {
        "beam_req": {
            "is_beam_decode": True,
            "prefix_len": 4,
            "beam_width": 2,
            "beam_decode_steps": 2,
            "cache_len": 5,
        }
    }
    common = _common_metadata(query_starts, seq_lens, block_table)
    builder = _make_builder()
    meta0 = _build_metadata(common, req_ids, beam_data)
    assert meta0 is not None
    assert meta0.suffix_slot_mapping_out.tolist() == [4, 100, 101, 102]

    # Update 1
    table1 = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    slots1 = torch.tensor([500, 501, 502, 503], dtype=torch.int64)
    meta1 = builder.update_block_table(meta0, table1, slots1)

    # Update 2 from meta1
    table2 = torch.tensor([[30, 31, 32, 33]], dtype=torch.int32)
    slots2 = torch.tensor([700, 701, 702, 703], dtype=torch.int64)
    meta2 = builder.update_block_table(meta1, table2, slots2)

    # Prove meta0 remains unchanged
    assert meta0.block_table.tolist() == [[0, 1, 2, 3]]
    assert meta0.suffix_slot_mapping_out.tolist() == [4, 100, 101, 102]

    # Prove meta1 remains unchanged
    assert meta1.block_table.tolist() == [[10, 11, 12, 13]]
    assert meta1.suffix_slot_mapping_out.tolist() == [24, 500, 501, 502]

    # Prove meta2 reflects update 2
    assert meta2.block_table.tolist() == [[30, 31, 32, 33]]
    assert meta2.suffix_slot_mapping_out.tolist() == [64, 700, 701, 702]


def test_update_block_table_two_kv_cache_group_reuse() -> None:
    query_starts = [0, 4]
    seq_lens = [8]
    block_table_group0 = [[1, 2, 3, 4]]
    req_ids = ["group_beam"]
    beam_data = {
        "group_beam": {
            "is_beam_decode": True,
            "prefix_len": 4,
            "beam_width": 2,
            "beam_decode_steps": 2,
            "cache_len": 5,
        }
    }
    common = _common_metadata(query_starts, seq_lens, block_table_group0)
    builder = _make_builder()

    # KV Cache Group 0 build
    meta_group0 = _build_metadata(common, req_ids, beam_data)
    assert meta_group0 is not None

    # KV Cache Group 1 table and slot mapping reuse
    block_table_group1 = torch.tensor([[50, 51, 52, 53]], dtype=torch.int32)
    slots_group1 = torch.tensor([800, 801, 802, 803], dtype=torch.int64)
    meta_group1 = builder.update_block_table(meta_group0, block_table_group1, slots_group1)

    # Both share underlying scheduling descriptors
    assert meta_group1.prefix_indices is meta_group0.prefix_indices
    assert meta_group1.suffix_past_coords is meta_group0.suffix_past_coords

    # Each holds its own physical block tables & suffix slot mappings
    assert meta_group0.suffix_slot_mapping_out.tolist() == [6, 100, 101, 102]
    assert meta_group1.suffix_slot_mapping_out.tolist() == [104, 800, 801, 802]


def test_update_block_table_ubatching_rejection() -> None:
    query_starts = [0, 2, 6]
    seq_lens = [4, 8]
    block_table = [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
    ]
    req_ids = ["req0", "req1_beam"]
    beam_data = {
        "req1_beam": {
            "is_beam_decode": True,
            "prefix_len": 4,
            "beam_width": 2,
            "beam_decode_steps": 2,
            "cache_len": 5,
        }
    }
    common = _common_metadata(query_starts, seq_lens, block_table)
    builder = _make_builder()
    metadata = _build_metadata(common, req_ids, beam_data)
    assert metadata is not None

    # Pass sliced micro-batch table (e.g., ubatching splitting the batch to 1 request)
    microbatch_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    microbatch_slots = torch.tensor([100, 101], dtype=torch.int64)

    with pytest.raises(ValueError, match="Cannot update block table:"):
        builder.update_block_table(metadata, microbatch_table, microbatch_slots)
