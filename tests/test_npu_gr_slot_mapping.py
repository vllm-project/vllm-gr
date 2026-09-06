# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the scoped NPU native-slot bypass, not NPU kernel tests."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_gr.v1.worker import npu_gr_inputs as inputs


class NativeTables:
    def __init__(self):
        self.calls = []
        self.block_tables = [
            SimpleNamespace(
                is_mamba_group=False,
                use_hybrid_blocks=False,
                pcp_world_size=1,
                dcp_world_size=1,
                slot_mapping=SimpleNamespace(gpu=torch.full((6,), 99, dtype=torch.int32)),
            )
        ]
        self.row = torch.tensor([11, 12, 13], dtype=torch.int32)

    def __getitem__(self, index):
        return self.block_tables[index]

    def commit_block_table(self, count):
        self.calls.append(("commit", count))

    def compute_slot_mapping(self, num_reqs, query_start, positions):
        self.calls.append(("native-slots", num_reqs))
        # CPU reproduction of the unchecked per-row position//block_size
        # lookup in the native kernel, not execution of that kernel.
        self.row[positions // 4]


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(inputs, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(inputs, "has_ec_transfer", lambda: False)
    output = SimpleNamespace(beam_data={})
    frame = inputs.NPUInputFrame(output, "gr", None, {}, 4, input_copy_done=True)
    tables = NativeTables()
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["gr"], block_table=tables),
        device=torch.device("cpu"),
        use_async_scheduling=True,
        _gr_npu_input_frame=frame,
    )
    query = torch.tensor([0, 4], dtype=torch.int32)
    positions = torch.arange(10, 14, dtype=torch.int64)
    return runner, output, tables, query, positions


def test_near_capacity_decode_never_launches_native_slot_lookup(case):
    runner, output, tables, query, positions = case
    with pytest.raises(IndexError):
        tables.compute_slot_mapping(1, query, positions)
    tables.calls.clear()
    slots = tables[0].slot_mapping.gpu
    address = slots.data_ptr()
    with inputs.suspend_native_slot_mapping(runner, output):
        proxy = runner.input_batch.block_table
        assert proxy is not tables
        assert proxy[0] is tables[0]
        assert proxy.block_tables is tables.block_tables
        proxy.commit_block_table(1)
        proxy.compute_slot_mapping(1, query, positions)
        assert slots.tolist() == [PAD_SLOT_ID] * 6
    assert runner.input_batch.block_table is tables
    assert slots.data_ptr() == address
    assert tables.calls == [("commit", 1)]
    with pytest.raises(RuntimeError, match="stale or repeated"):
        proxy.compute_slot_mapping(1, query, positions)


def test_ordinary_and_prefill_slot_path_delegates_original(case):
    runner, _, tables, query, _ = case
    del runner._gr_npu_input_frame
    for beam_data in ({}, {"gr": {"persistent_runner_state": True, "is_beam_decode": False}}):
        with inputs.suspend_native_slot_mapping(runner, SimpleNamespace(beam_data=beam_data)):
            assert runner.input_batch.block_table is tables
            runner.input_batch.block_table.compute_slot_mapping(1, query, torch.arange(4))
    assert tables.calls == [("native-slots", 1), ("native-slots", 1)]


def test_native_prepare_exception_restores_original_without_masking_error(case):
    runner, output, tables, _, _ = case
    error = ValueError("native prepare error")
    with (
        pytest.raises(ValueError) as caught,
        inputs.suspend_native_slot_mapping(runner, output),
    ):
        raise error
    assert caught.value is error
    assert runner.input_batch.block_table is tables
    assert tables[0].slot_mapping.gpu.tolist() == [99] * 6


def test_missing_or_duplicate_slot_boundary_fails_and_restores(case):
    runner, output, tables, query, positions = case
    with (
        pytest.raises(RuntimeError, match="skipped slot boundary"),
        inputs.suspend_native_slot_mapping(runner, output),
    ):
        pass
    assert runner.input_batch.block_table is tables
    with (
        pytest.raises(RuntimeError, match="stale or repeated"),
        inputs.suspend_native_slot_mapping(runner, output),
    ):
        runner.input_batch.block_table.compute_slot_mapping(1, query, positions)
        runner.input_batch.block_table.compute_slot_mapping(1, query, positions)
    assert runner.input_batch.block_table is tables


@pytest.mark.parametrize(
    "failure",
    [
        "multi",
        "mamba",
        "hybrid",
        "pcp",
        "dcp",
        "dtype",
        "capacity",
        "compress",
        "mrope",
        "xdrope",
        "spec",
        "identity",
        "nested",
    ],
)
def test_unsupported_native_layout_fails_before_any_slot_write(case, failure):
    runner, output, tables, _, _ = case
    group = tables[0]
    if failure == "multi":
        tables.block_tables.append(group)
    elif failure in ("mamba", "hybrid", "pcp", "dcp"):
        attribute, value = {
            "mamba": ("is_mamba_group", True),
            "hybrid": ("use_hybrid_blocks", True),
            "pcp": ("pcp_world_size", 2),
            "dcp": ("dcp_world_size", 2),
        }[failure]
        setattr(group, attribute, value)
    elif failure == "dtype":
        group.slot_mapping.gpu = group.slot_mapping.gpu.to(torch.int64)
    elif failure == "capacity":
        group.slot_mapping.gpu = group.slot_mapping.gpu[:3]
    elif failure in ("compress", "mrope", "xdrope", "spec"):
        setattr(
            runner,
            {
                "compress": "use_compress",
                "mrope": "uses_mrope",
                "xdrope": "uses_xdrope_dim",
                "spec": "use_async_spec_decode",
            }[failure],
            True,
        )
    elif failure == "identity":
        runner.input_batch.req_ids = ["other"]
    else:
        runner.input_batch.block_table = inputs._BeamOnlyBlockTable(
            tables, runner._gr_npu_input_frame, group.slot_mapping.gpu
        )
    original = runner.input_batch.block_table
    with pytest.raises(RuntimeError), inputs.suspend_native_slot_mapping(runner, output):
        pytest.fail("unsupported layout entered native preparation")
    assert runner.input_batch.block_table is original
    assert torch.all(group.slot_mapping.gpu == 99)
    assert tables.calls == []


@pytest.mark.parametrize("failure", ["copy", "bool", "query", "positions", "draft", "compressed"])
def test_slot_boundary_rejects_wrong_inputs_without_native_call(case, failure):
    runner, output, tables, query, positions = case
    with pytest.raises(RuntimeError), inputs.suspend_native_slot_mapping(runner, output):
        proxy = runner.input_batch.block_table
        if failure == "copy":
            runner._gr_npu_input_frame.input_copy_done = False
        if failure == "draft":
            proxy.compute_slot_mapping_draft(None, None)
        else:
            proxy.compute_slot_mapping(
                True if failure == "bool" else 1,
                query.to(torch.int64) if failure == "query" else query,
                positions[:3] if failure == "positions" else positions,
                positions_compressed_list=[] if failure == "compressed" else None,
            )
    assert runner.input_batch.block_table is tables
    assert tables[0].slot_mapping.gpu.tolist() == [99] * 6
    assert tables.calls == []
