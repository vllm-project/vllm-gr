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


def test_incremental_seqlens_step_updates_device_in_place() -> None:
    """Fixed seqlens upload once; decode_step increments update suffix on device."""
    builder = _make_fixed_builder()
    geom_key = ("beam", (1, 4, 8, 3), (0, 4), (4,), builder.block_size)

    prefix_kv = builder._prefix_seqlens_kv_buf[:1]
    prefix_cu = builder._prefix_cu_seqlens_q_buf[:2]
    suffix_kv = builder._suffix_seqlens_kv_buf[:4]
    suffix_cu_q = builder._suffix_cu_seqlens_q_buf[:5]
    suffix_cu_k = builder._suffix_cu_seqlens_k_buf[:5]

    # First step (decode_step=0 -> suffix KV length 1).
    ok = builder._upload_incremental_seqlens(
        geom_key,
        0,
        prefix_seqlens_kv_view=prefix_kv,
        prefix_cu_seqlens_q_view=prefix_cu,
        suffix_seqlens_kv_view=suffix_kv,
        suffix_cu_seqlens_q_view=suffix_cu_q,
        suffix_cu_seqlens_k_view=suffix_cu_k,
        prefix_seqlens_kv_cpu=[8],
        prefix_cu_seqlens_q_cpu=[4],
        suffix_cu_seqlens_q_cpu=[1, 2, 3, 4],  # cumsum of [1,1,1,1]
    )
    assert ok is True
    assert prefix_kv.tolist() == [8]
    assert prefix_cu.tolist() == [0, 4]
    assert suffix_kv.tolist() == [1, 1, 1, 1]
    assert suffix_cu_q.tolist() == [0, 1, 2, 3, 4]
    assert suffix_cu_k.tolist() == [0, 1, 2, 3, 4]

    # Subsequent step: fixed payloads are not rewritten (H2D skipped), the
    # step-derived suffix arrays refresh on device.
    suffix_kv.fill_(-1)  # poison to prove cache reuse does not repack fixed.
    suffix_cu_k.mul_(0)  # zero out
    ok = builder._upload_incremental_seqlens(
        geom_key,
        2,  # decode_step=2 -> suffix KV length 3
        prefix_seqlens_kv_view=prefix_kv,
        prefix_cu_seqlens_q_view=prefix_cu,
        suffix_seqlens_kv_view=suffix_kv,
        suffix_cu_seqlens_q_view=suffix_cu_q,
        suffix_cu_seqlens_k_view=suffix_cu_k,
        prefix_seqlens_kv_cpu=[8],
        prefix_cu_seqlens_q_cpu=[4],
        suffix_cu_seqlens_q_cpu=[1, 2, 3, 4],
    )
    assert ok is True
    assert suffix_kv.tolist() == [3, 3, 3, 3]
    assert suffix_cu_k.tolist() == [0, 3, 6, 9, 12]
    # Fixed payloads untouched by the second step.
    assert prefix_kv.tolist() == [8]
    assert prefix_cu.tolist() == [0, 4]
    assert suffix_cu_q.tolist() == [0, 1, 2, 3, 4]


def test_incremental_seqlens_rejects_none_key() -> None:
    builder = _make_fixed_builder()
    ok = builder._upload_incremental_seqlens(
        None,
        0,
        prefix_seqlens_kv_view=builder._prefix_seqlens_kv_buf[:1],
        prefix_cu_seqlens_q_view=builder._prefix_cu_seqlens_q_buf[:2],
        suffix_seqlens_kv_view=builder._suffix_seqlens_kv_buf[:4],
        suffix_cu_seqlens_q_view=builder._suffix_cu_seqlens_q_buf[:5],
        suffix_cu_seqlens_k_view=builder._suffix_cu_seqlens_k_buf[:5],
        prefix_seqlens_kv_cpu=[8],
        prefix_cu_seqlens_q_cpu=[4],
        suffix_cu_seqlens_q_cpu=[1, 2, 3, 4],
    )
    assert ok is False


@pytest.mark.parametrize(
    "invalid_field,invalid_value",
    [
        (None, None),
        ("prefix_seqlens_kv_cpu", [8, 8]),
        ("prefix_seqlens_kv_cpu", [-1]),
        ("prefix_cu_seqlens_q_cpu", [3]),
        ("suffix_cu_seqlens_q_cpu", [1, 2, 3, 5]),
        ("prefix_seqlens_kv_view", 2),
        ("prefix_cu_seqlens_q_view", 3),
        ("suffix_cu_seqlens_q_view", 4),
        ("suffix_seqlens_kv_view", 0),
    ],
)
def test_persistent_seqlens_device_constants_without_host_staging(
    monkeypatch, invalid_field, invalid_value
) -> None:
    builder = _make_fixed_builder()
    fields = {
        "prefix_seqlens_kv_view": builder._prefix_seqlens_kv_buf[:1],
        "prefix_cu_seqlens_q_view": builder._prefix_cu_seqlens_q_buf[:2],
        "suffix_seqlens_kv_view": builder._suffix_seqlens_kv_buf[:4],
        "suffix_cu_seqlens_q_view": builder._suffix_cu_seqlens_q_buf[:5],
        "suffix_cu_seqlens_k_view": builder._suffix_cu_seqlens_k_buf[:5],
        "prefix_seqlens_kv_cpu": [8],
        "prefix_cu_seqlens_q_cpu": [4],
        "suffix_cu_seqlens_q_cpu": [1, 2, 3, 4],
    }
    if invalid_field:
        fields[invalid_field] = (
            torch.zeros(invalid_value, dtype=torch.int32)
            if invalid_field.endswith("_view")
            else invalid_value
        )

    def unexpected_upload(*args, **kwargs):
        pytest.fail("persistent static seqlens must not upload host staging")

    monkeypatch.setattr(builder, "_pack_int32_to_dev", unexpected_upload)
    session = SimpleNamespace(
        context=SimpleNamespace(
            _gr_persistent_prefix_binding=(torch.tensor([7, 8], dtype=torch.int32),)
        )
    )
    token = BEAM_DATA_VAR.set(
        {
            "beam_data": {"gr": {"persistent_runner_state": True, "is_beam_decode": True}},
            "beam_session": session,
        }
    )
    key = ("persistent",)
    try:
        if invalid_field:
            with pytest.raises(RuntimeError, match="seqlens geometry mismatch"):
                builder._upload_incremental_seqlens(key, 0, **fields)
            assert not builder._geometry_cache().get(key, False)
            return

        addresses = {name: view.data_ptr() for name, view in fields.items() if name.endswith("_view")}
        for step in (0, 1, 2):
            assert builder._upload_incremental_seqlens(key, step, **fields)
            assert fields["prefix_seqlens_kv_view"].tolist() == [8]
            assert fields["prefix_cu_seqlens_q_view"].tolist() == [0, 4]
            assert fields["suffix_cu_seqlens_q_view"].tolist() == [0, 1, 2, 3, 4]
            assert fields["suffix_seqlens_kv_view"].tolist() == [step + 1] * 4
            assert fields["suffix_cu_seqlens_k_view"].tolist() == [i * (step + 1) for i in range(5)]
            assert all(fields[name].data_ptr() == ptr for name, ptr in addresses.items())
        assert builder._geometry_cache()[key]
    finally:
        BEAM_DATA_VAR.reset(token)


def test_cached_seqlens_views_reuse_same_objects() -> None:
    """Fixed seqlens views are created once per geometry and reused verbatim."""
    builder = _make_fixed_builder()
    key = ("beam", (4, 8, 8), 16)

    v1 = builder._cached_seqlens_views(key, n_pk=1, n_pc=2, n_sk=4, n_scq=5, n_sck=5)
    v2 = builder._cached_seqlens_views(key, n_pk=1, n_pc=2, n_sk=4, n_scq=5, n_sck=5)
    assert v1 is not None and v2 is not None
    # Same object (not just equal values) on the second call.
    for a, b in zip(v1, v2):
        assert a is b
        assert a.data_ptr() == b.data_ptr()
    # Storage is the address-stable buffer.
    assert v1[0].data_ptr() == builder._prefix_seqlens_kv_buf.data_ptr()

    # A different geometry key yields a separate view set.
    other = builder._cached_seqlens_views(
        ("beam2", (8, 16, 16), 16), n_pk=1, n_pc=2, n_sk=4, n_scq=5, n_sck=5
    )
    assert other is not None
    assert other[0] is not v1[0]
    # None key falls back (no caching).
    assert builder._cached_seqlens_views(None, n_pk=1, n_pc=1, n_sk=1, n_scq=1, n_sck=1) is None


def _build_cached_geometry_case(builder, key, prefix_len, width, *, persistent=False):
    common = _common_metadata([0, width], [width], [list(range(8))])
    token = BEAM_DATA_VAR.set(
        {
            "beam_data": {"gr": {"persistent_runner_state": True, "is_beam_decode": True}},
            "beam_session": SimpleNamespace(
                context=SimpleNamespace(
                    _gr_persistent_prefix_binding=(torch.arange(8, dtype=torch.int32),)
                )
            ),
        }
        if persistent else None
    )
    try:
        return builder._create_beam_tensors(
            common,
            prefix_indices=list(range(width)),
            prefix_seqlens_q_cpu=[width],
            prefix_cu_seqlens_q_cpu=[width],
            prefix_seqlens_kv_cpu=[prefix_len],
            suffix_seqlens_q_cpu=[1] * width,
            suffix_cu_seqlens_q_cpu=list(range(1, width + 1)),
            suffix_cu_seqlens_kv_cpu=list(range(1, width + 1)),
            suffix_seqlens_kv_cpu=[1] * width,
            suffix_slot_mapping_out_list=[],
            prefix_req_idxs_cpu=[0],
            max_prefix_blocks_count=(prefix_len + builder.block_size - 1) // builder.block_size,
            device=builder.device,
            max_num_splits=0,
            geom_key=key,
            decode_step=0 if key is not None else None,
        )
    finally:
        BEAM_DATA_VAR.reset(token)


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("overwrite_key", [("session-b",), None])
def test_metadata_cache_reinitializes_after_shared_buffers_overwritten(persistent, overwrite_key):
    """A→B→A and A→uncached→A must not reuse an obsolete initialized bit."""
    builder = _make_fixed_builder()
    first = _build_cached_geometry_case(builder, ("session-a",), 8, 4, persistent=persistent)
    assert first[1].tolist() == [8]
    second = _build_cached_geometry_case(builder, overwrite_key, 12, 3, persistent=persistent)
    assert second[1].tolist() == [12]
    last = _build_cached_geometry_case(builder, ("session-a",), 8, 4, persistent=persistent)
    assert last[1].tolist() == [8]
    assert last[3].tolist() == [0, 4]
    assert last[10].tolist() == [0, 1, 2, 3, 4]


def test_metadata_cache_does_not_accumulate_completed_request_keys():
    builder = _make_fixed_builder()
    for session_id in range(30):
        _build_cached_geometry_case(
            builder, (f"session-{session_id}",), 8 + 2 * (session_id % 2), 4,
            persistent=True,
        )
        cache = builder._geometry_cache()
        assert len(cache.get("views", {})) <= 1
        assert sum(value is True for value in cache.values()) <= 1


def test_std_metadata_does_not_invalidate_resident_beam_lengths(monkeypatch):
    from vllm_gr.v1.beam.decode_geometry import BeamDecodeGeometry

    builder = _make_fixed_builder()
    writes = []
    original = BeamDecodeGeometry.write_static_seqlens

    def write(geometry, *args):
        writes.append(geometry)
        return original(geometry, *args)

    monkeypatch.setattr(BeamDecodeGeometry, "write_static_seqlens", write)
    key = ("session-a",)
    first = _build_cached_geometry_case(builder, key, 8, 4, persistent=True)
    common = _common_metadata([0, 2], [2], [list(range(8))])
    std = builder._create_std_tensors([0, 1], [2], [2], [2], [0], common, builder.device)
    assert std[1].tolist() == [0, 2]
    assert std[2].tolist() == [2]
    last = _build_cached_geometry_case(builder, key, 8, 4, persistent=True)
    assert last[1].tolist() == [8]
    assert last[1] is first[1]
    assert len(writes) == 1


@pytest.mark.parametrize("recovery_key", [("session-a",), ("session-b",)])
def test_failed_geometry_initialization_leaves_no_ready_entry(recovery_key):
    builder = _make_fixed_builder()
    _build_cached_geometry_case(builder, ("session-a",), 8, 4, persistent=True)
    with pytest.raises(RuntimeError, match="seqlens geometry mismatch"):
        _build_cached_geometry_case(builder, ("session-b",), -1, 3, persistent=True)
    assert not any(value is True for value in builder._geometry_cache().values())
    recovered = _build_cached_geometry_case(builder, recovery_key, 12, 3, persistent=True)
    assert recovered[1].tolist() == [12]
    assert recovered[3].tolist() == [0, 3]
    assert recovered[10].tolist() == [0, 1, 2, 3]


def test_same_cache_key_with_different_view_extents_rebuilds_static_values():
    builder = _make_fixed_builder()
    key = ("caller-key",)
    first = _build_cached_geometry_case(builder, key, 8, 4, persistent=True)
    changed = _build_cached_geometry_case(builder, key, 8, 3, persistent=True)
    assert changed[3].tolist() == [0, 3]
    assert changed[10].tolist() == [0, 1, 2, 3]
    assert changed[1].data_ptr() == first[1].data_ptr()
    assert changed[1] is not first[1]
    restored = _build_cached_geometry_case(builder, key, 8, 4, persistent=True)
    assert restored[3].tolist() == [0, 4]
    assert restored[10].tolist() == [0, 1, 2, 3, 4]


def test_geom_cache_key_stable_without_decode_step() -> None:
    """Key ignores decode_step / seq_lens / q_starts (the per-step-varying
    quantities) and is stable across steps of the same request geometry;
    mixed/non-beam batches return None (full rebuild path)."""
    builder = _make_builder()
    beam_data = {
        "beam": {
            "is_beam_decode": True,
            "beam_width": 4,
            "prefix_len": 8,
            "cache_len": 8,
            "beam_decode_steps": 3,  # grows per decode step; must not be keyed
        }
    }
    req_ids = ["beam"]
    k1 = builder._geom_cache_key(1, req_ids, beam_data, [0, 4], [4], True)
    k2 = builder._geom_cache_key(1, req_ids, beam_data, [0, 4], [4], True)
    assert k1 is not None
    assert k1 == k2
    # key[0] is the beam_view; only width/prefix_len/cache_len (not the
    # growing beam_decode_steps) participate.
    assert k1[0] == (("beam", 4, 8, 8),)
    # A later decode frame reports a grown beam_decode_steps but the same
    # geometry -> identical key (cache hits).
    grown = dict(beam_data)
    grown["beam"] = dict(grown["beam"], beam_decode_steps=5)
    k3 = builder._geom_cache_key(1, req_ids, grown, [0, 4], [5], True)
    assert k3 == k1
    # q_starts / seq_lens do not affect the key (per-step-varying, excluded).
    k4 = builder._geom_cache_key(1, req_ids, beam_data, [0, 8], [8], True)
    assert k4 == k1
    # Non-incremental falls back.
    assert builder._geom_cache_key(1, req_ids, beam_data, [0, 4], [4], False) is None
    # Mixed batch (non-beam request) falls back.
    mixed = dict(beam_data)
    mixed["other"] = {"is_beam_decode": False, "beam_width": 1}
    assert builder._geom_cache_key(2, ["beam", "other"], mixed, [0, 4, 6], [4, 2], True) is None


def test_incremental_seqlens_through_create_beam_tensors() -> None:
    """Full _create_beam_tensors with geom cached route keeps device suffix
    arrays consistent across decode_step increments."""
    builder = _make_fixed_builder()
    common = _common_metadata([0, 4], [4], [[1, 2, 3, 4]])
    geom_key = ("beam", (1, 4, 8, 3), (0, 4), (4,), builder.block_size)

    def run(step: int) -> Any:
        return builder._create_beam_tensors(
            common,
            prefix_indices=[0, 1, 2, 3],
            prefix_seqlens_q_cpu=[4],
            prefix_cu_seqlens_q_cpu=[4],
            prefix_seqlens_kv_cpu=[8],
            suffix_seqlens_q_cpu=[1, 1, 1, 1],
            suffix_cu_seqlens_q_cpu=[1, 2, 3, 4],
            suffix_cu_seqlens_kv_cpu=[step + 1, 2 * (step + 1), 3 * (step + 1), 4 * (step + 1)],
            suffix_seqlens_kv_cpu=[step + 1] * 4,
            suffix_slot_mapping_out_list=[],
            prefix_req_idxs_cpu=[0],
            max_prefix_blocks_count=2,
            device=torch.device("cpu"),
            max_num_splits=0,
            geom_key=geom_key,
            decode_step=step,
        )

    m0 = run(0)
    assert m0[1].tolist() == [8]  # prefix_seqlens_kv_device
    assert m0[2] == 16  # prefix_max_seq_len (4 blocks * block_size 2)
    assert m0[6].tolist() == [1, 1, 1, 1]  # suffix_seqlens_kv_device
    assert m0[10].tolist() == [0, 1, 2, 3, 4]  # suffix_cu_seqlens_q_device

    # Step 3: suffix KV length = 4, cu_seqlens_k = [0,4,8,12,16].
    m3 = run(3)
    assert m3[6].tolist() == [4, 4, 4, 4]
    assert m3[11].tolist() == [0, 4, 8, 12, 16]
    # Fixed prefix untouched across the cached step.
    assert m3[1].tolist() == [8]


def test_create_beam_tensors_geom_key_arrange_contiguity() -> None:
    """A geom_key must not bypass the contiguous-index verification.

    The device-side arange fast path is only valid when ``prefix_indices`` is
    one contiguous range.  Supplying a geom_key does not by itself prove that
    (the caller could hand over an arbitrary recompute/gather layout), so the
    builder must still fall back to the host copy for a non-contiguous index
    list.  This guards the fast path against a future optimization that would
    turn a gather layout into a bogus strided arange."""
    builder = _make_fixed_builder()
    common = _common_metadata([0, 4], [4], [[1, 2, 3, 4]])
    geom_key = ("beam", (1, 4, 8, 3), (0, 4), (4,), builder.block_size)

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
            geom_key=geom_key,
            decode_step=0,
        )

    # Contiguous range under a geom_key: emitted with a strided arange.
    m = run([0, 1, 2, 3])
    assert m[0].tolist() == [0, 1, 2, 3]
    # Non-contiguous layout under a geom_key must NOT be wrongly ranged.
    m = run([3, 7, 11])
    assert m[0].tolist() == [3, 7, 11]


@pytest.mark.parametrize("fixed_buffers", [False, True])
def test_persistent_decode_uses_backend_bound_prefix_row(
    fixed_buffers: bool,
) -> None:
    # Prefix ownership must not depend on the CUDA-graph fixed-buffer mode:
    # eager returns a view of the bound row, while graph mode refreshes its
    # address-stable staging row from the same backend-owned binding.
    builder = _make_fixed_builder() if fixed_buffers else _make_builder()
    common = _common_metadata([0, 4], [4], [[91, 92, 93, 94]])
    bound_prefix = torch.tensor([7, 8], dtype=torch.int32)
    session = SimpleNamespace(
        context=SimpleNamespace(_gr_persistent_prefix_binding=(bound_prefix,))
    )
    token = BEAM_DATA_VAR.set(
        {
            "beam_data": {
                "gr": {
                    "persistent_runner_state": True,
                    "is_beam_decode": True,
                }
            },
            "beam_session": session,
        }
    )
    try:
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
            suffix_slot_mapping_out_list=[],
            prefix_req_idxs_cpu=[0],
            max_prefix_blocks_count=2,
            device=torch.device("cpu"),
            max_num_splits=0,
            geom_key=("persistent",),
            decode_step=0,
        )
    finally:
        BEAM_DATA_VAR.reset(token)

    assert result[5][0, :2].tolist() == [7, 8]


@pytest.mark.parametrize("fixed_buffers", [False, True])
def test_persistent_update_block_table_keeps_backend_prefix_binding(
    fixed_buffers: bool,
) -> None:
    """A refreshed native table cannot replace the session's bound prefix.

    vLLM refreshes attention metadata for cache groups after the initial
    build.  A prefix-cache hit is especially sensitive here because its row
    contains a mix of reused blocks and the final recomputed block.
    """
    builder = _make_fixed_builder() if fixed_buffers else _make_builder()
    bound_prefix = torch.tensor([31, 32], dtype=torch.int32)
    session = SimpleNamespace(
        context=SimpleNamespace(_gr_persistent_prefix_binding=(bound_prefix,))
    )
    metadata = BeamAttentionMetadata(
        num_actual_tokens=4,
        max_query_len=4,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        block_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        slot_mapping=torch.arange(4, dtype=torch.int64),
        prefix_block_table=torch.tensor([[31, 32]], dtype=torch.int32),
        prefix_req_idxs_cpu=[0],
        max_prefix_blocks_count=2,
    )
    refreshed_native = torch.tensor([[91, 92, 93, 94]], dtype=torch.int32)
    token = BEAM_DATA_VAR.set(
        {
            "beam_data": {
                "gr": {
                    "persistent_runner_state": True,
                    "is_beam_decode": True,
                }
            },
            "beam_session": session,
        }
    )
    try:
        updated = builder.update_block_table(
            metadata,
            refreshed_native,
            torch.arange(4, dtype=torch.int64),
        )
    finally:
        BEAM_DATA_VAR.reset(token)

    assert updated.prefix_block_table[0, :2].tolist() == [31, 32]
