# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Functional guards for Beam Attention metadata ownership."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
    BeamAttentionMetadataBuilder,
)
from vllm_gr.v1.beam.context import BEAM_DATA_VAR


def _make_builder(block_size: int = 2) -> BeamAttentionMetadataBuilder:
    builder = object.__new__(BeamAttentionMetadataBuilder)
    builder.vllm_config = SimpleNamespace(speculative_config=None)
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
    "query_starts,seq_lens,num_reqs",
    [
        pytest.param([0, 8], [8], 1, id="prefill"),
        pytest.param([0, 1], [8], 1, id="prefix-cache-hit"),
        pytest.param([0, 3], [3], 1, id="first-chunk"),
        pytest.param([0, 3], [6], 1, id="continuation-chunk"),
        pytest.param([0, 3, 3], [6, 0], 1, id="padded-request"),
        pytest.param([0, 3, 3], [6, 0], 2, id="full-graph-padded-request"),
        pytest.param([0, 3, 4], [6, 8], 2, id="multiple-requests"),
    ],
)
def test_native_cpu_lengths_avoid_device_read(
    monkeypatch: pytest.MonkeyPatch,
    query_starts: list[int],
    seq_lens: list[int],
    num_reqs: int,
) -> None:
    common = CommonAttentionMetadata(
        **vars(_common_metadata(query_starts, seq_lens, [[0, 1, 2, 3]] * len(seq_lens)))
    )
    common.num_reqs = num_reqs
    common.seq_lens_cpu_upper_bound = common.seq_lens.clone()

    def forbid_cpu(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native CPU lengths must avoid a device-to-host read")

    monkeypatch.setattr(torch.Tensor, "cpu", forbid_cpu)
    metadata = _make_builder().build(0, common)

    assert metadata.seq_lens_cpu == seq_lens[:num_reqs]
    assert metadata.query_start_loc_cpu == query_starts
    assert metadata.seq_lens is common.seq_lens


@pytest.mark.parametrize(  # type: ignore[misc]
    "source", ["missing", "short", "speculative", "v1-mirror", "short-v1-mirror"]
)
def test_native_cpu_length_fallback_preserves_exact_lengths(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    common = CommonAttentionMetadata(**vars(_common_metadata([0, 1], [8], [[0, 1, 2, 3]])))
    builder = _make_builder()
    if source == "short":
        common.seq_lens_cpu_upper_bound = torch.empty(0, dtype=torch.int32)
    elif source in ("speculative", "v1-mirror"):
        # Async speculation may overestimate the actual context length.
        common.seq_lens_cpu_upper_bound = torch.tensor([11], dtype=torch.int32)
        builder.vllm_config.speculative_config = SimpleNamespace()
        if source == "v1-mirror":
            common._seq_lens_cpu = torch.tensor([8], dtype=torch.int32)
    elif source == "short-v1-mirror":
        common._seq_lens_cpu = torch.empty(0, dtype=torch.int32)
        common.seq_lens_cpu_upper_bound = torch.tensor([8], dtype=torch.int32)

    reads = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        reads.append(tensor)
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    metadata = builder.build(0, common)

    assert metadata.seq_lens_cpu == [8]
    assert len(reads) == (0 if source in ("v1-mirror", "short-v1-mirror") else 1)
    if reads:
        assert reads[0] is common.seq_lens


@pytest.mark.parametrize("native_upper_bound", [False, True])  # type: ignore[misc]
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
    native_upper_bound: bool,
    query_starts: list[int],
    seq_lens: list[int],
    block_table: list[list[int]],
    req_ids: list[str],
    beam_data: dict[str, dict[str, Any]],
    expected: dict[str, Any],
) -> None:
    common = _common_metadata(query_starts, seq_lens, block_table)
    if native_upper_bound:
        # Legacy Beam batches must keep using their exact custom lengths,
        # including ordinary rows mixed into the same batch.
        common.seq_lens_cpu_upper_bound = torch.tensor([99] * len(seq_lens), dtype=torch.int32)
    metadata = _build_metadata(
        common,
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


@pytest.mark.parametrize("enabled", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "case", ["mixed", "cache-lost", "cache-partial", "prefix-only"]
)
def test_incremental_metadata_preserves_standard_segments(case: str, enabled: bool) -> None:
    """Enabling suffix fusion must not reject scheduler-generated std rows."""

    class Registry:
        def get_slot(self, req_id: str) -> int:
            return 0

    prefix_len = 6 if case == "cache-partial" else 4
    cache_len = 4 if case == "mixed" else (2 if case == "cache-partial" else 0)
    beam_query_len = 4 if case == "prefix-only" else (2 if case == "mixed" else 6)
    req_ids = ["ordinary", "beam"] if case == "mixed" else ["beam"]
    query_starts = [0, 2, 4] if case == "mixed" else [0, beam_query_len]
    seq_lens = [3, 6] if case == "mixed" else [cache_len + beam_query_len]
    common = _common_metadata(query_starts, seq_lens, [[0, 1, 2, 3]] * len(req_ids))
    session = SimpleNamespace(
        is_active=True,
        beam_width=2,
        context=SimpleNamespace(enable_short_suffix_attention=enabled, registry=Registry()),
    )
    token = BEAM_DATA_VAR.set(
        {
            "req_ids": req_ids,
            "beam_session": session,
            "beam_data": {
                "beam": {
                    "is_beam_decode": True,
                    "prefix_len": prefix_len,
                    "cache_len": cache_len,
                    "beam_width": 2,
                    "beam_decode_steps": 1,
                    "decode_step": 0,
                }
            },
        }
    )
    try:
        metadata = _make_builder().build(0, common)
    finally:
        BEAM_DATA_VAR.reset(token)

    assert metadata.is_beam_decode
    assert metadata.beam_session is session
    assert metadata.std_indices is not None
    expected_std = [0, 1] if case == "mixed" else [0, 1, 2, 3]
    expected_prefix = [] if case == "prefix-only" else ([2, 3] if case == "mixed" else [4, 5])
    assert metadata.std_indices.tolist() == expected_std
    prefix_indices = metadata.prefix_indices
    assert ([] if prefix_indices is None else prefix_indices.tolist()) == expected_prefix
    assert sorted(expected_std + expected_prefix) == list(range(query_starts[-1]))
    assert metadata.std_seqlens_kv_cpu == ([3] if case == "mixed" else [prefix_len])
    if expected_prefix:
        assert metadata.suffix_max_q_len == 1
        assert metadata.suffix_seqlens_q_cpu == [1, 1]
        assert metadata.prefix_seqlens_kv_cpu == [prefix_len]


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


def test_async_metadata_upload_keeps_each_pending_host_payload_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second request cannot overwrite the source of an enqueued H2D."""
    builder = _make_builder()
    builder._h2d_host = torch.zeros(16, dtype=torch.int32)
    builder._h2d_dev = torch.zeros(16, dtype=torch.int32)
    first, second = torch.zeros(4, dtype=torch.int32), torch.zeros(4, dtype=torch.int32)
    staging_storage = builder._h2d_dev.untyped_storage().data_ptr()
    queued: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
    allocated = []
    original_empty, original_copy = torch.empty, torch.Tensor.copy_

    def allocate(*args: Any, **kwargs: Any) -> torch.Tensor:
        # CPU-only fixture models the private pinned allocation; hardware
        # integration tests exercise the real allocator and transfer events.
        assert kwargs.pop("pin_memory") is True
        tensor = original_empty(*args, **kwargs)
        allocated.append(tensor)
        return tensor

    def enqueue(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        if staging_storage in (dst.untyped_storage().data_ptr(), src.untyped_storage().data_ptr()):
            queued.append((dst, src, non_blocking))
            return dst
        return original_copy(dst, src, non_blocking=non_blocking)

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.Tensor, "copy_", enqueue)
    builder._pack_int32_to_dev([(first, [1, 2, 3, 4])], non_blocking=True)
    builder._h2d_host.fill_(-1)
    builder._pack_int32_to_dev([(second, [5, 6, 7, 8])], non_blocking=True)
    builder._h2d_host.fill_(-2)
    assert len(allocated) == 2
    assert allocated[0].data_ptr() != allocated[1].data_ptr()
    assert queued[0][2] and queued[2][2]
    for dst, src, _ in queued:
        original_copy(dst, src)
    assert first.tolist() == [1, 2, 3, 4]
    assert second.tolist() == [5, 6, 7, 8]


@pytest.mark.parametrize("is_npu", [False, True])  # type: ignore[misc]
def test_only_npu_incremental_geometry_uses_async_upload(
    monkeypatch: pytest.MonkeyPatch, is_npu: bool
) -> None:
    builder = _make_fixed_builder()
    builder._is_npu = is_npu
    transfers = []

    def upload(entries: Any, *, non_blocking: bool = False) -> None:
        transfers.append(non_blocking)

    monkeypatch.setattr(builder, "_pack_int32_to_dev", upload)
    for step in (0, 1):
        assert builder._upload_incremental_seqlens(
            ("beam", 4),
            step,
            prefix_seqlens_kv_view=builder._prefix_seqlens_kv_buf[:1],
            prefix_cu_seqlens_q_view=builder._prefix_cu_seqlens_q_buf[:2],
            suffix_seqlens_kv_view=builder._suffix_seqlens_kv_buf[:4],
            suffix_cu_seqlens_q_view=builder._suffix_cu_seqlens_q_buf[:5],
            suffix_cu_seqlens_k_view=builder._suffix_cu_seqlens_k_buf[:5],
            prefix_seqlens_kv_cpu=[8],
            prefix_cu_seqlens_q_cpu=[4],
            suffix_cu_seqlens_q_cpu=[1, 2, 3, 4],
        )
    assert transfers == [is_npu]


@pytest.mark.npu_test  # type: ignore[misc]
def test_npu_async_metadata_upload_survives_pinned_allocator_reuse() -> None:
    """Queued uploads retain immutable payloads after their local source dies."""
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    builder = _make_builder()
    builder.device = torch.device("npu")
    builder._is_npu = True
    with torch.inference_mode():
        builder._h2d_host = torch.zeros(64, dtype=torch.int32, pin_memory=True)
        builder._h2d_dev = torch.zeros(64, dtype=torch.int32, device="npu")
        work = torch.ones((1024, 1024), dtype=torch.float16, device="npu")
        scratch = torch.empty_like(work)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        results = []
        with torch.npu.stream(stream):
            for request in range(32):
                # Keep preceding work on the owning stream while the CPU
                # submits multiple request payloads and churns pinned memory.
                for _ in range(8):
                    torch.mm(work, work, out=scratch)
                first = torch.empty(8, dtype=torch.int32, device="npu")
                second = torch.empty(12, dtype=torch.int32, device="npu")
                values = list(range(20 * request, 20 * request + 20))
                builder._pack_int32_to_dev(
                    [(first, values[:8]), (second, values[8:])], non_blocking=True
                )
                results.append((first, second, values))
                # The just-released upload source has exactly this size.
                # It must not be recycled until its asynchronous H2D finishes.
                churn = torch.empty(20, dtype=torch.int32, pin_memory=True)
                churn.fill_(-1)
                del churn
                builder._h2d_host.fill_(-2)
        stream.synchronize()
        for first, second, values in results:
            assert first.cpu().tolist() == values[:8]
            assert second.cpu().tolist() == values[8:]


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


@pytest.mark.parametrize("is_npu", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("incremental", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("max_columns", [1, 3])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "request_rows", [[0], [0, 1, 2, 3], [3, 2, 1, 0], [0, 2], [0, 0]]
)
def test_prefix_block_table_view_requires_npu_incremental_identity_rows(
    monkeypatch: pytest.MonkeyPatch,
    is_npu: bool,
    incremental: bool,
    max_columns: int,
    request_rows: list[int],
) -> None:
    """Basic slicing preserves B1/B4 block rows, cache hits and new requests."""
    builder = _make_fixed_builder()
    builder._is_npu = is_npu
    builder._is_cuda = not is_npu
    builder._prefix_block_table_buf = torch.zeros((4, 8), dtype=torch.int32)
    builder._suffix_cu_seqlens_q_buf = torch.zeros(16, dtype=torch.int32)
    builder._suffix_cu_seqlens_k_buf = torch.zeros(16, dtype=torch.int32)
    common = _common_metadata(
        [0, 2, 4, 6, 8],
        [8, 10, 12, 14],
        [list(range(8 * row, 8 * row + 8)) for row in range(4)],
    )
    expected = common.block_table_tensor[request_rows, :max_columns].clone()
    fixed_address = builder._prefix_block_table_buf.data_ptr()
    advanced_reads = []
    original_getitem, original_empty = torch.Tensor.__getitem__, torch.empty

    def getitem(tensor: torch.Tensor, key: Any) -> torch.Tensor:
        if tensor is common.block_table_tensor and isinstance(key[0], list):
            advanced_reads.append(tuple(key[0]))
        return original_getitem(tensor, key)

    def allocate(*args: Any, **kwargs: Any) -> torch.Tensor:
        # The private asynchronous pinned staging is covered on hardware by
        # test_npu_async_metadata_upload_survives_pinned_allocator_reuse.
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "__getitem__", getitem)
    monkeypatch.setattr(torch, "empty", allocate)
    count = len(request_rows)
    rows = 2 * count

    def build(key: Any, step: int) -> torch.Tensor:
        return builder._create_beam_tensors(
            common,
            prefix_indices=list(range(rows)),
            prefix_seqlens_q_cpu=[2] * count,
            prefix_cu_seqlens_q_cpu=list(range(2, rows + 1, 2)),
            prefix_seqlens_kv_cpu=[2 * max_columns] * count,
            suffix_seqlens_q_cpu=[1] * rows,
            suffix_cu_seqlens_q_cpu=list(range(1, rows + 1)),
            suffix_cu_seqlens_kv_cpu=list(range(step + 1, (rows + 1) * (step + 1), step + 1)),
            suffix_seqlens_kv_cpu=[step + 1] * rows,
            suffix_slot_mapping_out_list=[],
            prefix_req_idxs_cpu=request_rows,
            max_prefix_blocks_count=max_columns,
            device=torch.device("cpu"),
            max_num_splits=0,
            geom_key=key,
            decode_step=step if incremental else None,
        )[5]

    key = ("request-a", tuple(request_rows)) if incremental else None
    blocks = build(key, 0)
    assert blocks.data_ptr() == fixed_address
    assert blocks.shape == (count, 8)
    torch.testing.assert_close(blocks[:, :max_columns], expected)
    uses_view = is_npu and incremental and request_rows == list(range(count))
    assert len(advanced_reads) == (0 if uses_view else 1)

    common.block_table_tensor.add_(100)
    blocks = build(key, 1)
    # Same-request incremental geometry retains its original prefix blocks;
    # mixed/recompute geometry refreshes the current native block table.
    torch.testing.assert_close(blocks[:, :max_columns], expected if incremental else expected + 100)
    if incremental:
        blocks = build(("request-b", tuple(request_rows)), 0)
        torch.testing.assert_close(blocks[:, :max_columns], expected + 100)
    assert blocks.data_ptr() == fixed_address
    assert len(advanced_reads) == (0 if uses_view else 2)


@pytest.mark.parametrize("beam_first", [False, True])  # type: ignore[misc]
def test_npu_mixed_metadata_keeps_indexed_block_rows_and_synchronous_upload(
    monkeypatch: pytest.MonkeyPatch, beam_first: bool
) -> None:
    builder = _make_fixed_builder()
    builder._is_npu = True
    req_ids = ["beam", "ordinary"] if beam_first else ["ordinary", "beam"]
    beam_index = req_ids.index("beam")
    common = _common_metadata([0, 2, 4], [3, 3], [[1, 2, 3, 4], [5, 6, 7, 8]])
    expected = common.block_table_tensor[beam_index : beam_index + 1, :2].clone()
    reads, transfers = [], []
    original_getitem = torch.Tensor.__getitem__
    original_upload = builder._pack_int32_to_dev

    def getitem(tensor: torch.Tensor, key: Any) -> torch.Tensor:
        if tensor is common.block_table_tensor and isinstance(key[0], list):
            reads.append(tuple(key[0]))
        return original_getitem(tensor, key)

    def upload(entries: Any, *, non_blocking: bool = False) -> None:
        transfers.append(non_blocking)
        original_upload(entries, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "__getitem__", getitem)
    monkeypatch.setattr(builder, "_pack_int32_to_dev", upload)
    token = BEAM_DATA_VAR.set(
        {
            "req_ids": req_ids,
            "beam_data": {
                "beam": {
                    "is_beam_decode": True,
                    "prefix_len": 3,
                    "cache_len": 3,
                    "beam_width": 2,
                    "beam_decode_steps": 1,
                    "decode_step": 0,
                }
            },
            "beam_session": SimpleNamespace(
                is_active=True,
                context=SimpleNamespace(registry=SimpleNamespace(get_slot=lambda _: 0)),
            ),
        }
    )
    try:
        metadata = builder.build(0, common)
    finally:
        BEAM_DATA_VAR.reset(token)
    assert (beam_index,) in reads
    assert transfers and not any(transfers)
    assert metadata.std_seqlens_kv_cpu == [3]
    assert metadata.prefix_seqlens_kv_cpu == [3]
    torch.testing.assert_close(metadata.prefix_block_table[:, :2], expected)


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


def test_geom_cache_key_stable_without_decode_step() -> None:
    """Key ignores decode_step / seq_lens / q_starts (the per-step-varying
    quantities) and is identical across sessions on the same beam geometry;
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


def test_create_beam_tensors_explicit_contiguous_flag_skips_scan() -> None:
    """``prefix_indices_contiguous`` is the explicit verdict from
    _build_beam_mappings: when supplied, the per-element ``all()`` scan is
    skipped and the arange fast path is taken directly.  When absent (None),
    the conservative scan runs."""
    builder = _make_fixed_builder()
    common = _common_metadata([0, 4], [4], [[1, 2, 3, 4]])
    geom_key = ("beam", (1, 4, 8, 3), (0, 4), (4,), builder.block_size)

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
            geom_key=geom_key,
            decode_step=0,
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


def test_build_beam_mappings_incremental_returns_analytic_scalars() -> None:
    """The incremental path returns its scalar reductions analytically:
    a contiguous prefix flag of True and per-geometry max q lens; the
    recompute path returns False (scan required) and list-derived maxima."""
    builder = _make_builder()
    common = _common_metadata([0, 4], [4], [[1, 2, 3, 4]])
    geom_key = ("beam", (1, 4, 8, 8), builder.block_size)

    inc = builder._build_beam_mappings(
        beam_req_idxs=[0],
        beam_prefix_lens=[8],
        beam_widths=[4],
        beam_decode_steps=[1],
        beam_cache_lens=[8],
        beam_q_start_offsets=[0],
        q_starts=[0, 4],
        common_attn_metadata=common,
        is_incremental=True,
        beam_decode_step_list=[2],
        geom_key=geom_key,
    )
    # Tail of the tuple: (..., max_prefix_blocks_count, contiguous, pql, sql, skl)
    assert inc[-4] is True  # prefix_indices_contiguous
    assert inc[-3] == 4  # prefix_max_q_len == beam_width
    assert inc[-2] == 1  # suffix_max_q_len (one query token per beam)
    assert inc[-1] == 3  # suffix_max_seq_len == decode_step + 1 (step=2)

    rec = builder._build_beam_mappings(
        beam_req_idxs=[0],
        beam_prefix_lens=[8],
        beam_widths=[4],
        beam_decode_steps=[2],
        beam_cache_lens=[8],
        beam_q_start_offsets=[0],
        q_starts=[0, 4],
        common_attn_metadata=common,
        is_incremental=False,
        beam_decode_step_list=None,
        geom_key=None,
    )
    # The recompute path reports the SCANNED verdict (this single-group layout
    # happens to be contiguous); multi-group gather layouts scan False.
    assert rec[-4] is True
    assert rec[-3] == 4  # prefix_max_q_len derived from the group list
    assert rec[-2] == 2  # suffix_max_q_len: recompute splits chunk budget (2/beam)
    assert rec[-1] == 2  # suffix_max_seq_len derived from suffix_seqlens_kv list


def test_beam_mappings_fixed_cumsum_cached_per_geom() -> None:
    """Fixed cu_seqlens lists are cached per geom_key, not recomputed per step.

    Incremental decode only varies ``suffix_seqlens_kv`` (and its cumsum) with
    ``decode_step``; the prefix cumsums and ``suffix_cu_seqlens_q`` are fixed
    per geometry.  A geom_key hit reuses those lists instead of paying a numpy
    cumsum + tolist on every decode step, while the step-dependent suffix
    cumsum is still recomputed."""
    builder = _make_builder()
    common = _common_metadata([0, 4], [4], [[1, 2, 3, 4]])
    geom_key = ("beam", (1, 4, 8, 8), builder.block_size)

    def run(step: int) -> Any:
        return builder._build_beam_mappings(
            beam_req_idxs=[0],
            beam_prefix_lens=[8],
            beam_widths=[4],
            beam_decode_steps=[1],
            beam_cache_lens=[8],
            beam_q_start_offsets=[0],
            q_starts=[0, 4],
            common_attn_metadata=common,
            is_incremental=True,
            beam_decode_step_list=[step],
            geom_key=geom_key,
        )

    m0 = run(0)
    assert m0[7] == [4]  # prefix_cu_seqlens_q_cpu (cumsum([4]))
    assert m0[8] == [8]  # prefix_cu_seqlens_kv_cpu (cumsum([8]))
    assert m0[9] == [1, 2, 3, 4]  # suffix_cu_seqlens_q_cpu (cumsum([1]*4))
    assert m0[10] == [1, 2, 3, 4]  # suffix_cu_seqlens_kv_cpu, step 0 -> s=1

    # Same geometry, later step: fixed lists must be the SAME objects (cache
    # hit, no recompute); the step-derived suffix_cu_seqlens_kv changes.
    m1 = run(2)
    assert m1[7] is m0[7] and m1[8] is m0[8] and m1[9] is m0[9]
    assert m1[10] == [3, 6, 9, 12]  # step 2 -> s=3, recomputed

    # A different geometry must not share the cached lists.
    other_key = ("beam", (2, 8, 16, 16), builder.block_size)
    m2 = builder._build_beam_mappings(
        beam_req_idxs=[0],
        beam_prefix_lens=[16],
        beam_widths=[2],
        beam_decode_steps=[1],
        beam_cache_lens=[16],
        beam_q_start_offsets=[0],
        q_starts=[0, 2],
        common_attn_metadata=common,
        is_incremental=True,
        beam_decode_step_list=[0],
        geom_key=other_key,
    )
    assert m2[7] is not m0[7]
    assert m2[8] == [16]
    assert m2[9] == [1, 2]


def test_build_geometry_hit_fast_path_replays_layout() -> None:
    """A second decode step with the same geometry replays the cached layout.

    Frame 1 (miss) runs segmentation + std/beam mappings and registers the
    layout in the geometry cache.  Frame 2 (hit) skips those functions
    entirely: the fixed lists come from the cache, the step-derived suffix
    lists are analytic, and the decode_step is re-read from THIS frame's
    beam_data (never from the cache).
    """

    class _Registry:
        def get_slot(self, req_id: str) -> int:
            return 0

    session = SimpleNamespace(is_active=True, context=SimpleNamespace(registry=_Registry()))
    builder = _make_builder()

    def run_build(step: int) -> BeamAttentionMetadata:
        beam_data = {
            "beam": {
                "is_beam_decode": True,
                "prefix_len": 8,
                "beam_width": 4,
                "beam_decode_steps": 2,
                "cache_len": 8,
                "decode_step": step,
            }
        }
        common = _common_metadata([0, 4], [12], [[1, 2, 3, 4, 5, 6]])
        token = BEAM_DATA_VAR.set(
            {"beam_data": beam_data, "req_ids": ["beam"], "beam_session": session}
        )
        try:
            return builder.build(0, common)
        finally:
            BEAM_DATA_VAR.reset(token)

    # Frame 1 (decode_step=0): full path, registers the layout.
    m1 = run_build(0)
    assert m1.suffix_seqlens_kv_cpu == [1, 1, 1, 1]
    assert m1.suffix_cu_seqlens_kv_cpu == [1, 2, 3, 4]
    assert m1.is_beam_decode is True
    assert m1.incremental_suffix_kv_len == 1

    # Frame 2 (decode_step=2): geometry hit — the cached layout is replayed
    # with THIS frame's step; suffix lists and scalars step forward.
    m2 = run_build(2)
    assert m2.suffix_seqlens_kv_cpu == [3, 3, 3, 3]
    assert m2.suffix_cu_seqlens_kv_cpu == [3, 6, 9, 12]
    assert m2.incremental_suffix_kv_len == 3
    # Fixed geometry did not drift.
    assert m2.prefix_seqlens_kv_cpu == m1.prefix_seqlens_kv_cpu
    assert m2.suffix_cu_seqlens_q_cpu == m1.suffix_cu_seqlens_q_cpu

    # A different geometry (beam_width change) must not replay the cache.
    beam_data3 = {
        "beam": {
            "is_beam_decode": True,
            "prefix_len": 8,
            "beam_width": 2,
            "beam_decode_steps": 2,
            "cache_len": 8,
            "decode_step": 0,
        }
    }
    common3 = _common_metadata([0, 2], [10], [[1, 2, 3, 4, 5]])
    token = BEAM_DATA_VAR.set(
        {"beam_data": beam_data3, "req_ids": ["beam"], "beam_session": session}
    )
    try:
        m3 = builder.build(0, common3)
    finally:
        BEAM_DATA_VAR.reset(token)
    assert m3.suffix_seqlens_kv_cpu == [1, 1]


def test_geometry_cache_evicts_finished_session_entries() -> None:
    """Entries from a finished session are swept when a new session builds.

    Session A's geometry leaves entries in the cu_lists / layout sub-caches
    (keyed by A's request id).  When session B (a different request id)
    builds, the eviction sweep drops every key whose request ids no longer
    overlap the current batch, so a long-running server does not accumulate
    one entry per historical request.
    """

    class _Registry:
        def get_slot(self, req_id: str) -> int:
            return 0

    session = SimpleNamespace(is_active=True, context=SimpleNamespace(registry=_Registry()))
    builder = _make_builder()

    def run_build(req_id: str) -> None:
        beam_data = {
            req_id: {
                "is_beam_decode": True,
                "prefix_len": 8,
                "beam_width": 4,
                "beam_decode_steps": 2,
                "cache_len": 8,
                "decode_step": 0,
            }
        }
        common = _common_metadata([0, 4], [12], [[1, 2, 3, 4, 5, 6]])
        token = BEAM_DATA_VAR.set(
            {"beam_data": beam_data, "req_ids": [req_id], "beam_session": session}
        )
        try:
            builder.build(0, common)
        finally:
            BEAM_DATA_VAR.reset(token)

    # Session A builds: layout/cu_lists entries registered under A's key.
    run_build("req-a")
    cache = builder._geometry_cache()
    assert cache["layout"], "session A registered its layout"
    assert cache["cu_lists"], "session A registered its cu_lists"
    assert any("req-a" in str(k) for k in cache["layout"])

    # Session B (different request id) builds: A's entries are swept.
    run_build("req-b")
    assert not any("req-a" in str(k) for k in cache["layout"])
    assert not any("req-a" in str(k) for k in cache["cu_lists"])
    assert any("req-b" in str(k) for k in cache["layout"])
    # The set-valued block_table_copied sub-cache is swept too when present
    # (its entries live in a set — deletion must use discard, not del).  The
    # plain builder harness skips the fixed-buffer path, so guard for it.
    btc = cache.get("block_table_copied")
    if btc:
        assert not any("req-a" in str(k) for k in btc)


def test_block_table_copied_set_eviction_survives_fixed_buffers() -> None:
    """The eviction sweep must handle the set-valued block_table_copied.

    ``block_table_copied`` stores geom keys inside a SET (membership
    markers, not a dict).  Deleting from a set with ``del`` raises
    TypeError and killed the engine (offline beam search E2E); this test
    drives the fixed-buffer path twice with different request ids so the
    set is populated and then swept.
    """

    class _Registry:
        def get_slot(self, req_id: str) -> int:
            return 0

    session = SimpleNamespace(is_active=True, context=SimpleNamespace(registry=_Registry()))
    builder = _make_fixed_builder()

    def run_build(req_id: str) -> None:
        beam_data = {
            req_id: {
                "is_beam_decode": True,
                "prefix_len": 8,
                "beam_width": 2,
                "beam_decode_steps": 2,
                "cache_len": 8,
                "decode_step": 0,
            }
        }
        common = _common_metadata([0, 2], [10], [[1, 2, 3, 4, 5]])
        token = BEAM_DATA_VAR.set(
            {"beam_data": beam_data, "req_ids": [req_id], "beam_session": session}
        )
        try:
            builder.build(0, common)
        finally:
            BEAM_DATA_VAR.reset(token)

    # Session A registers its geometry in the fixed-buffer sub-caches
    # (including the set-valued block_table_copied).
    run_build("req-a")
    cache = builder._geometry_cache()
    btc = cache.get("block_table_copied")
    assert btc, "fixed-buffer build registered block_table_copied"
    assert any("req-a" in str(k) for k in btc)

    # Session B sweeps A's set entry without raising TypeError.
    run_build("req-b")
    assert not any("req-a" in str(k) for k in btc)


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("is_npu", [False, True])  # type: ignore[misc]
def test_incremental_identity_rows_are_proved_only_for_npu(batch_size: int, is_npu: bool) -> None:
    """The device-view opt-in is derived from the current host row layout."""
    width = 2
    rows = batch_size * width
    request_ids = [f"beam-{index}" for index in range(batch_size)]
    beam_data = {
        request_id: {
            "is_beam_decode": True,
            "prefix_len": 4,
            "cache_len": 4,
            "beam_width": width,
            "beam_decode_steps": 1,
            "decode_step": 0,
        }
        for request_id in request_ids
    }
    session = SimpleNamespace(
        is_active=True,
        beam_width=width,
        attention_width=rows,
        context=SimpleNamespace(registry=SimpleNamespace(get_slot=lambda _request: 0)),
    )
    builder = _make_builder()
    builder._is_npu = is_npu
    builder._is_cuda = not is_npu
    common = _common_metadata(
        list(range(0, rows + 1, width)),
        [4 + width] * batch_size,
        [[0, 1, 2, 3]] * batch_size,
    )
    token = BEAM_DATA_VAR.set(
        {"beam_data": beam_data, "req_ids": request_ids, "beam_session": session}
    )
    try:
        metadata = builder.build(0, common)
    finally:
        BEAM_DATA_VAR.reset(token)

    assert metadata.prefix_indices is not None
    assert metadata.prefix_indices.tolist() == list(range(rows))
    assert metadata.npu_prefix_rows_are_identity is is_npu


def test_identity_row_proof_is_not_reused_for_same_shape_changed_prefix() -> None:
    """Losing cached prefix blocks changes routing without changing B or W."""
    builder = _make_builder()
    builder._is_npu = True
    session = SimpleNamespace(
        is_active=True,
        beam_width=2,
        attention_width=2,
        context=SimpleNamespace(registry=SimpleNamespace(get_slot=lambda _request: 0)),
    )

    def build(cache_len: int) -> BeamAttentionMetadata:
        token = BEAM_DATA_VAR.set(
            {
                "req_ids": ["beam"],
                "beam_session": session,
                "beam_data": {
                    "beam": {
                        "is_beam_decode": True,
                        "prefix_len": 4,
                        "cache_len": cache_len,
                        "beam_width": 2,
                        "beam_decode_steps": 1,
                        "decode_step": 0,
                    }
                },
            }
        )
        try:
            return builder.build(0, _common_metadata([0, 2], [6], [[0, 1, 2, 3]]))
        finally:
            BEAM_DATA_VAR.reset(token)

    first = build(4)
    changed = build(0)
    restored = build(4)
    assert first.npu_prefix_rows_are_identity
    assert not changed.npu_prefix_rows_are_identity
    assert restored.npu_prefix_rows_are_identity


pytestmark = pytest.mark.cpu_test
