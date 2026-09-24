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
    builder._fixed_buffers = False
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


def test_update_block_table_reprojects_suffix_slot_mapping() -> None:
    """update_block_table must re-project suffix_slot_mapping_out against the
    NEW block_table/slot_mapping, not silently carry forward the physical KV
    slots computed at build() time against the OLD one.

    vLLM caches one built BeamAttentionMetadata across every hybrid KV-cache
    group sharing this builder's (kv_cache_spec, type) and swaps in each
    group's own block_table/slot_mapping via update_block_table instead of a
    full rebuild (gpu_model_runner.py's _build_attn_group_metadata). Reusing
    a stale suffix_slot_mapping_out there means beam decode reads/writes the
    wrong physical KV cache slots for the second (and later) group.
    """
    block_size = 2
    old_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    old_slots = torch.tensor([50, 51], dtype=torch.int64)
    # index 0: "past" token -> req_idx=0, block_idx=1, block_offset=0
    #   old: block_table[0, 1] = 2 -> slot = 2 * block_size + 0 = 4
    past_coords = torch.tensor([[0, 0, 1, 0]], dtype=torch.long)
    # index 1: "current" token -> token_idx=0 into slot_mapping
    #   old: slot_mapping[0] = 50
    curr_coords = torch.tensor([[1, 0]], dtype=torch.long)
    old_suffix_slots = torch.tensor([4, 50], dtype=torch.int32)

    metadata = BeamAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_seq_len=6,
        seq_lens=torch.tensor([3, 6], dtype=torch.int32),
        block_table=old_table,
        slot_mapping=old_slots,
        block_size=block_size,
        suffix_total_tokens=2,
        suffix_slot_mapping_out=old_suffix_slots,
        suffix_past_coords=past_coords,
        suffix_curr_coords=curr_coords,
    )

    new_table = torch.tensor([[100, 200], [300, 400]], dtype=torch.int32)
    new_slots = torch.tensor([999, 998], dtype=torch.int64)

    updated = _make_builder(block_size=block_size).update_block_table(
        metadata, new_table, new_slots
    )

    # Old metadata (and its suffix mapping) must be untouched.
    assert metadata.suffix_slot_mapping_out is old_suffix_slots
    assert metadata.suffix_slot_mapping_out.tolist() == [4, 50]
    # New: block_table[0, 1] = 200 -> 200 * 2 + 0 = 400 (past);
    # slot_mapping[0] = 999 (current). NOT the stale [4, 50].
    assert updated.suffix_slot_mapping_out.tolist() == [400, 999]


def test_update_block_table_rejects_incompatible_suffix_geometry() -> None:
    """A block_table/slot_mapping too small for the persisted suffix
    coordinates must raise rather than silently index out of bounds or
    project garbage (ubatching / sliced block tables are not supported by
    this cached-metadata fast path)."""
    past_coords = torch.tensor([[0, 0, 1, 0]], dtype=torch.long)
    metadata = BeamAttentionMetadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=4,
        seq_lens=torch.tensor([4], dtype=torch.int32),
        block_table=torch.tensor([[1, 2]], dtype=torch.int32),
        slot_mapping=torch.tensor([10], dtype=torch.int64),
        block_size=2,
        suffix_total_tokens=1,
        suffix_slot_mapping_out=torch.tensor([4], dtype=torch.int32),
        suffix_past_coords=past_coords,
    )
    # Only 1 block column: past_coords references block_idx=1, out of range.
    too_narrow_table = torch.tensor([[1]], dtype=torch.int32)

    with pytest.raises(ValueError, match="suffix_past_coords"):
        _make_builder().update_block_table(
            metadata, too_narrow_table, torch.tensor([10], dtype=torch.int64)
        )


def test_pack_int32_to_dev_matches_individual_copies() -> None:
    """Single-H2D staging scatters the same values as per-field copies.

    Uses CPU staging buffers (device="cpu") to exercise the fixed-buffer pack
    path without a GPU: ``_pack_int32_to_dev`` must copy exactly the per-field
    host lists into the destination buffers in order.
    """
    builder = _make_builder()
    builder._h2d_host = torch.zeros(16, dtype=torch.int32)
    builder._h2d_dev = torch.zeros(16, dtype=torch.int32)

    dst_a = torch.zeros(4, dtype=torch.int32)
    dst_b = torch.zeros(3, dtype=torch.int32)
    builder._pack_int32_to_dev([(dst_a, [1, 2, 3, 4]), (dst_b, [5, 6, 7])])

    assert dst_a.tolist() == [1, 2, 3, 4]
    assert dst_b.tolist() == [5, 6, 7]
    # Single contiguous staging: values are laid back-to-back on the host
    # staging, so the first transfer covers every field's source.
    assert builder._h2d_host[:7].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert builder._h2d_dev[:7].tolist() == [1, 2, 3, 4, 5, 6, 7]

    # Empty entries are a no-op.
    builder._pack_int32_to_dev([])


def test_pack_int32_to_dev_leading_zero_cus() -> None:
    """cu_seqlens payloads carry the leading 0 element (prefix/suffix groups)."""
    builder = _make_builder()
    builder._h2d_host = torch.zeros(6, dtype=torch.int32)
    builder._h2d_dev = torch.zeros(6, dtype=torch.int32)

    cu = torch.zeros(3, dtype=torch.int32)
    builder._pack_int32_to_dev([(cu, [0, 2, 4])])

    assert cu.tolist() == [0, 2, 4]


def _make_fixed_builder() -> BeamAttentionMetadataBuilder:
    """Builder with fixed-address CPU buffers enabled (CUDA-graph-like path)."""
    builder = _make_builder()
    builder._fixed_buffers = True
    # Preallocate the fixed-address metadata buffers on CPU.
    max_segs = 8
    builder._prefix_indices_buf = torch.zeros(max_segs, dtype=torch.long)
    builder._std_indices_buf = torch.zeros(max_segs, dtype=torch.long)
    builder._prefix_seqlens_kv_buf = torch.zeros(max_segs, dtype=torch.int32)
    builder._suffix_seqlens_kv_buf = torch.zeros(max_segs, dtype=torch.int32)
    builder._prefix_cu_seqlens_q_buf = torch.zeros(max_segs, dtype=torch.int32)
    builder._suffix_cu_seqlens_q_buf = torch.zeros(max_segs, dtype=torch.int32)
    builder._suffix_cu_seqlens_k_buf = torch.zeros(max_segs, dtype=torch.int32)
    builder._std_cu_seqlens_q_buf = torch.zeros(max_segs, dtype=torch.int32)
    builder._std_seqlens_kv_buf = torch.zeros(max_segs, dtype=torch.int32)
    builder._prefix_block_table_buf = torch.zeros((1, 8), dtype=torch.int32)
    builder._std_block_table_buf = torch.zeros((1, 8), dtype=torch.int32)
    builder._h2d_host = torch.zeros(64, dtype=torch.int32)
    builder._h2d_dev = torch.zeros(64, dtype=torch.int32)
    return builder


def test_create_beam_tensors_arrange_contiguity() -> None:
    """The device-side arange fast path is only valid when ``prefix_indices``
    is one contiguous range; the builder must fall back to the host copy for
    a non-contiguous index list."""
    builder = _make_fixed_builder()
    common = _common_metadata([0, 4], [4], [[1, 2, 3, 4]])

    def run(prefix_indices: list[int]) -> Any:
        return builder._create_beam_tensors(
            common,
            prefix_indices=prefix_indices,
            prefix_seqlens_q_cpu=[len(prefix_indices)],
            prefix_cu_seqlens_q_cpu=[len(prefix_indices)],
            prefix_seqlens_kv_cpu=[8],
            suffix_seqlens_q_cpu=[1] * len(prefix_indices),
            suffix_cu_seqlens_q_cpu=list(range(1, len(prefix_indices) + 1)),
            suffix_cu_seqlens_kv_cpu=[1] * len(prefix_indices),
            suffix_seqlens_kv_cpu=[1] * len(prefix_indices),
            suffix_slot_mapping_out_list=[],
            prefix_req_idxs_cpu=[0],
            max_prefix_blocks_count=2,
            device=torch.device("cpu"),
            max_num_splits=0,
        )

    # Contiguous range: emitted with a strided arange.
    m = run([0, 1, 2, 3])
    assert m[0].tolist() == [0, 1, 2, 3]
    # Non-contiguous layout must NOT be wrongly ranged.
    m = run([3, 7, 11])
    assert m[0].tolist() == [3, 7, 11]


def test_create_beam_tensors_caches_suffix_coord_maxima_on_cpu() -> None:
    """_create_beam_tensors must compute suffix_max_past_req/_past_blk/_q_idx
    on the CPU (from the plain Python coordinate lists) so
    update_block_table's geometry validation can read them without a
    .max().item() device sync."""
    builder = _make_fixed_builder()
    # 1 request, 5 blocks so block_idx up to 4 is in range.
    common = _common_metadata([0, 4], [4], [[10, 20, 30, 40, 50]])
    # block_size == 2

    # tag == -1 -> "current" (idx, token_idx into slot_mapping, len 4 here);
    # tag >= 0 -> "past" (idx, req_idx, pos into the paged cache).
    # past: req_idx=0, pos=8 -> block_idx=4, offset=0
    # past: req_idx=0, pos=3 -> block_idx=1, offset=1
    # curr: token_idx=3 ; curr: token_idx=1
    suffix_slot_mapping_out_list = [(-1, 3), (0, 8), (0, 3), (-1, 1)]

    result = builder._create_beam_tensors(
        common,
        prefix_indices=[0, 1, 2, 3],
        prefix_seqlens_q_cpu=[4],
        prefix_cu_seqlens_q_cpu=[4],
        prefix_seqlens_kv_cpu=[8],
        suffix_seqlens_q_cpu=[1, 1, 1, 1],
        suffix_cu_seqlens_q_cpu=[1, 2, 3, 4],
        suffix_cu_seqlens_kv_cpu=[1, 2, 3, 4],
        suffix_seqlens_kv_cpu=[1, 1, 1, 1],
        suffix_slot_mapping_out_list=suffix_slot_mapping_out_list,
        prefix_req_idxs_cpu=[0],
        max_prefix_blocks_count=2,
        device=torch.device("cpu"),
        max_num_splits=0,
    )
    suffix_max_past_req, suffix_max_past_blk, suffix_max_q_idx = result[-3:]

    assert suffix_max_past_req == 0
    assert suffix_max_past_blk == 4  # 8 // block_size(2)
    assert suffix_max_q_idx == 3


def test_create_beam_tensors_explicit_contiguous_flag_skips_scan() -> None:
    """``prefix_indices_contiguous`` is the explicit verdict from
    _build_beam_mappings: when supplied, the per-element ``all()`` scan is
    skipped and the arange fast path is taken directly.  When absent (None),
    the conservative scan runs."""
    builder = _make_fixed_builder()
    common = _common_metadata([0, 4], [4], [[1, 2, 3, 4]])

    def run(prefix_indices: list[int], **kw: Any) -> Any:
        return builder._create_beam_tensors(
            common,
            prefix_indices=prefix_indices,
            prefix_seqlens_q_cpu=[len(prefix_indices)],
            prefix_cu_seqlens_q_cpu=[len(prefix_indices)],
            prefix_seqlens_kv_cpu=[8],
            suffix_seqlens_q_cpu=[1] * len(prefix_indices),
            suffix_cu_seqlens_q_cpu=list(range(1, len(prefix_indices) + 1)),
            suffix_cu_seqlens_kv_cpu=[1] * len(prefix_indices),
            suffix_seqlens_kv_cpu=[1] * len(prefix_indices),
            suffix_slot_mapping_out_list=[],
            prefix_req_idxs_cpu=[0],
            max_prefix_blocks_count=2,
            device=torch.device("cpu"),
            max_num_splits=0,
            **kw,
        )

    # Explicit True: arange fast path without the scan.
    m = run([5, 6, 7, 8], prefix_indices_contiguous=True)
    assert m[0].tolist() == [5, 6, 7, 8]
    # Precomputed scalars are consumed (prefix_max_q_len visible in the tuple).
    m = run([5, 6, 7, 8], prefix_indices_contiguous=True, prefix_max_q_len=4, suffix_max_q_len=1)
    assert m[0].tolist() == [5, 6, 7, 8]
    # Explicit False: host copy even for a visually contiguous list.
    m = run([0, 1, 2, 3], prefix_indices_contiguous=False)
    assert m[0].tolist() == [0, 1, 2, 3]


def test_build_beam_mappings_scalar_reductions() -> None:
    """Scalar reductions (contiguity flag, max q/kv lens) are derived from
    the built lists for a single-group beam layout."""
    builder = _make_builder()
    common = _common_metadata([0, 4], [4], [[1, 2, 3, 4]])

    rec = builder._build_beam_mappings(
        beam_req_idxs=[0],
        beam_prefix_lens=[8],
        beam_widths=[4],
        beam_decode_steps=[2],
        beam_cache_lens=[8],
        beam_q_start_offsets=[0],
        q_starts=[0, 4],
        common_attn_metadata=common,
    )
    # Tail of the tuple: (..., max_prefix_blocks_count, contiguous, pql, sql, skl)
    assert rec[-4] is True  # this single-group layout happens to be contiguous
    assert rec[-3] == 4  # prefix_max_q_len derived from the group list
    assert rec[-2] == 2  # suffix_max_q_len: chunk budget split 2/beam
    assert rec[-1] == 2  # suffix_max_seq_len derived from suffix_seqlens_kv list
