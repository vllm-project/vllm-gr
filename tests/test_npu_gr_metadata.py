# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for persistent NPU prefix and Decode metadata.

The tensors in this module are CPU fakes.  These tests do not constitute NPU
attention, ACL Graph, device-residency, or performance acceptance.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadataBuilder,
    _persistent_prefix_binding,
)
from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.engine.types import GRSchedulerOutput, GRRequestStage
from vllm_gr.v1.worker import npu_model_runner_patch as npu_patch
from vllm_gr.v1.worker.npu_gr_inputs import (
    _persistent_decode_frame,
    make_patched_prepare_input_ids,
)
from vllm_gr.v1.worker.npu_gr_metadata import (
    bind_prefill_prefix,
    finish_decode_metadata,
    prepare_decode_metadata,
)


class _BlockGroup:
    def __init__(self, table: torch.Tensor, block_size: int = 16) -> None:
        self.table = table
        self.block_size = block_size
        self.is_mamba_group = False
        self.use_hybrid_blocks = False
        self.pcp_world_size = 1
        self.dcp_world_size = 1
        self.slot_mapping = SimpleNamespace(
            gpu=torch.full((table.shape[1] + 16,), 123, dtype=torch.int32)
        )

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        assert num_reqs == 1
        return self.table


class _MultiGroup:
    def __init__(self, groups: list[_BlockGroup]) -> None:
        self.block_tables = groups

    def __getitem__(self, index: int) -> _BlockGroup:
        return self.block_tables[index]


def _fixture(
    *, prefix_len: int = 31, width: int = 4, max_tokens: int = 3
) -> tuple[Any, Any, _BlockGroup]:
    blocks = (prefix_len + 15) // 16
    table = torch.arange(100, 100 + blocks + 4, dtype=torch.int32).view(1, -1)
    group = _BlockGroup(table)
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["gr"],
        req_id_to_index={"gr": 0},
        block_table=_MultiGroup([group]),
    )
    decision = SimpleNamespace(
        sequence=torch.zeros((1, width, max_tokens), dtype=torch.int32),
        beam_scores=torch.zeros((1, width), dtype=torch.float32),
        group_token_num=torch.zeros((1, width), dtype=torch.int32),
        worker_decode_steps=0,
    )
    context = SimpleNamespace(
        decision=decision,
        decode_step_buf=torch.full((1,), -7, dtype=torch.int32),
        decode_step_1based_buf=torch.full((1,), -8, dtype=torch.int32),
        xllm_block_table_buf=torch.full((1,), -9, dtype=torch.int32),
    )
    session = SimpleNamespace(
        session_id="gr",
        is_active=True,
        slot=0,
        beam_width=width,
        decode_step=0,
        context=context,
        _state=SimpleNamespace(prefix_len=prefix_len),
    )
    state = SimpleNamespace(
        request_id="gr",
        released=False,
        stage=GRRequestStage.PREFILL_RUNNING,
        step=0,
        device_step=0,
        _prepared_step=0,
        beam_width=width,
        max_tokens=max_tokens,
        session=session,
        prefix_binding=None,
        next_token_ids=torch.arange(width, dtype=torch.int32).view(1, width),
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        pcp_size=1,
        dcp_size=1,
        use_cp=False,
        input_batch=batch,
        requests={"gr": SimpleNamespace(req_id="gr", num_prompt_tokens=prefix_len)},
        _current_beam_data={"gr": {"prefix_len": prefix_len}},
        _beam_context=context,
        _beam_sessions={"gr": session},
        seq_lens=torch.full((4,), -10, dtype=torch.int32),
        optimistic_seq_lens_cpu=torch.full((4,), -11, dtype=torch.int32),
        query_start_loc=SimpleNamespace(gpu=torch.full((5,), -12, dtype=torch.int32)),
    )
    return runner, state, group


def _decode_output(runner: Any, state: Any, *, final: bool = False) -> tuple[Any, Any]:
    step = state.step
    metadata = {
        "persistent_runner_state": True,
        "persistent_request": True,
        "is_beam_decode": True,
        "is_constrained": True,
        "beam_width": state.beam_width,
        "prefix_len": state.session._state.prefix_len,
        "cache_len": state.session._state.prefix_len,
        "generation_index": step,
        "constraint_step": step,
        "beam_decode_steps": step,
        "decode_step": step - 1,
        "is_last_step": final,
        "next_token_ids": None,
        "beam_suffix": [],
        "parent_beam_ids": None,
    }
    sidecar = GRSchedulerOutput(
        request_id="gr",
        stage=GRRequestStage.DECODE_RUNNING,
        step=step,
        device_step=step,
        is_prefill=False,
        is_final=final,
        logical_token_count=1,
        physical_row_count=state.beam_width,
    )
    output = SimpleNamespace(
        beam_data={"gr": metadata},
        gr_outputs={"gr": sidecar},
        num_scheduled_tokens={"gr": state.beam_width},
        total_num_scheduled_tokens=state.beam_width,
        _gr_npu_logical_geometry=("gr", 1, state.beam_width),
    )
    runner._current_scheduler_output = output
    return output, metadata


def _bind_and_decode(
    *, prefix_len: int = 31, width: int = 4, max_tokens: int = 3, step: int = 1
) -> tuple[Any, Any, Any, Any, _BlockGroup]:
    runner, state, group = _fixture(
        prefix_len=prefix_len,
        width=width,
        max_tokens=max_tokens,
    )
    state.prefix_binding = bind_prefill_prefix(runner, state)
    state.stage = GRRequestStage.DECODE_RUNNING
    state.step = state.device_step = state._prepared_step = step
    state.session.decode_step = step - 1
    state.session.context.decision.worker_decode_steps = step
    output, metadata = _decode_output(runner, state, final=step + 1 == max_tokens)
    return runner, state, output, metadata, group


@pytest.mark.parametrize("prefix_len", [1, 15, 16, 17, 128, 1024])
def test_bind_prefill_prefix_keeps_only_ceil_prefix_blocks(prefix_len: int) -> None:
    runner, state, group = _fixture(prefix_len=prefix_len)
    binding = bind_prefill_prefix(runner, state)
    expected_blocks = (prefix_len + group.block_size - 1) // group.block_size

    assert len(binding) == 1
    assert binding[0].shape == (expected_blocks,)
    assert binding[0].data_ptr() == group.table[0].data_ptr()
    assert torch.equal(binding[0], group.table[0, :expected_blocks])
    assert binding[0].numel() < group.table.shape[1]


def test_prepared_prefix_tuple_is_consumed_by_shared_attention_binding() -> None:
    runner, state, output, metadata, _group = _bind_and_decode()
    prepare_decode_metadata(runner, state, output, metadata)
    token = BEAM_DATA_VAR.set(
        {
            "beam_data": output.beam_data,
            "beam_session": state.session,
        }
    )
    try:
        consumed = _persistent_prefix_binding()
    finally:
        BEAM_DATA_VAR.reset(token)

    assert state.session.context._gr_persistent_prefix_binding is state.prefix_binding
    assert consumed is state.prefix_binding[0]


def test_shared_builder_segments_full_bound_prefix_as_incremental_beam_only() -> None:
    prefix_len = 128
    width = 4
    current_step = 2
    runner, state, output, metadata, group = _bind_and_decode(
        prefix_len=prefix_len,
        width=width,
        max_tokens=4,
        step=current_step,
    )
    prepare_decode_metadata(runner, state, output, metadata)
    assert metadata["cache_len"] == prefix_len

    # This is the post-native-slot-boundary premise: persistent Beam suffix
    # writes do not use Native request slots. Positions/remap remain mocked in
    # the separate wrapper test and are not validated by this pure consumer.
    group.slot_mapping.gpu.fill_(PAD_SLOT_ID)
    q_starts = [0, width]
    builder = object.__new__(BeamAttentionMetadataBuilder)
    builder.block_size = group.block_size
    segmented = builder._segment_requests(
        1,
        ["gr"],
        output.beam_data,
        q_starts,
        [prefix_len + current_step],
    )
    (
        std_reqs,
        beam_req_idxs,
        beam_prefix_lens,
        beam_widths,
        beam_decode_steps,
        beam_cache_lens,
        beam_q_start_offsets,
        beam_decode_step_list,
        beam_pool_slots,
    ) = segmented

    assert std_reqs == []
    assert beam_req_idxs == [0]
    assert beam_prefix_lens == [prefix_len]
    assert beam_widths == [width]
    assert beam_decode_steps == [current_step]
    assert beam_cache_lens == [prefix_len]
    assert beam_q_start_offsets == [0]
    assert beam_decode_step_list == [current_step - 1]
    assert beam_pool_slots == [0]
    assert q_starts == [0, width]
    assert torch.all(group.slot_mapping.gpu == PAD_SLOT_ID)

    mappings = builder._build_beam_mappings(
        beam_req_idxs,
        beam_prefix_lens,
        beam_widths,
        beam_decode_steps,
        beam_cache_lens,
        beam_q_start_offsets,
        q_starts,
        SimpleNamespace(slot_mapping=group.slot_mapping.gpu),
        is_incremental=True,
        beam_decode_step_list=beam_decode_step_list,
    )
    (
        prefix_indices,
        prefix_query_lens,
        prefix_kv_lens,
        suffix_query_lens,
        suffix_kv_lens,
        prefix_req_idxs,
        suffix_slot_mapping,
        prefix_cu_query_lens,
        prefix_cu_kv_lens,
        suffix_cu_query_lens,
        suffix_cu_kv_lens,
        max_prefix_blocks,
    ) = mappings

    assert prefix_indices == list(range(width))
    assert all(0 <= index < width for index in prefix_indices)
    assert prefix_query_lens == [width]
    assert prefix_kv_lens == [prefix_len]
    assert prefix_req_idxs == [0]
    assert prefix_cu_query_lens == [width]
    assert prefix_cu_kv_lens == [prefix_len]
    assert suffix_query_lens == [1] * width
    assert suffix_kv_lens == [current_step] * width
    assert suffix_slot_mapping == []
    assert suffix_cu_query_lens == [1, 2, 3, 4]
    assert suffix_cu_kv_lens == [2, 4, 6, 8]
    assert max_prefix_blocks == prefix_len // group.block_size


def test_three_decode_steps_update_only_fixed_metadata_and_keep_addresses() -> None:
    runner, state, _output, _metadata, _group = _bind_and_decode(max_tokens=4)
    context = state.session.context
    next_tokens = state.next_token_ids
    addresses = (
        context.decode_step_buf.data_ptr(),
        context.decode_step_1based_buf.data_ptr(),
        context.xllm_block_table_buf.data_ptr(),
        runner.seq_lens.data_ptr(),
        runner.optimistic_seq_lens_cpu.data_ptr(),
        runner.query_start_loc.gpu.data_ptr(),
    )

    for step in (1, 2, 3):
        state.step = state.device_step = state._prepared_step = step
        state.session.decode_step = step - 1
        context.decision.worker_decode_steps = step
        output, metadata = _decode_output(runner, state, final=step == 3)
        prepare_decode_metadata(runner, state, output, metadata)
        # Native preparation owns query_start_loc; the persistent helper only
        # validates and preserves this already-correct physical geometry.
        runner.query_start_loc.gpu[:2] = torch.tensor([0, 4], dtype=torch.int32)
        finish_decode_metadata(runner, state, metadata)

        assert context.decode_step_buf.tolist() == [step - 1]
        assert context.decode_step_1based_buf.tolist() == [step]
        assert context.xllm_block_table_buf.tolist() == [0]
        assert runner.seq_lens[0].item() == 31 + step
        assert runner.optimistic_seq_lens_cpu[0].item() == 31 + step
        assert runner.query_start_loc.gpu[:2].tolist() == [0, 4]
        assert state.session.decode_step == step - 1
        assert state.step == step
        assert state.next_token_ids is next_tokens
        assert context._gr_persistent_prefix_binding is state.prefix_binding
        assert addresses == (
            context.decode_step_buf.data_ptr(),
            context.decode_step_1based_buf.data_ptr(),
            context.xllm_block_table_buf.data_ptr(),
            runner.seq_lens.data_ptr(),
            runner.optimistic_seq_lens_cpu.data_ptr(),
            runner.query_start_loc.gpu.data_ptr(),
        )


def test_decode_metadata_never_reextracts_native_blocks_or_queries_prefix_cache() -> None:
    runner, state, output, metadata, _group = _bind_and_decode()

    class ForbiddenBlockTable:
        @property
        def block_tables(self) -> Any:
            raise AssertionError("Decode must not re-extract Native block rows")

    runner.input_batch.block_table = ForbiddenBlockTable()
    runner.get_computed_blocks = lambda *_args: pytest.fail("Decode must not query Prefix Cache")
    prepare_decode_metadata(runner, state, output, metadata)
    finish_decode_metadata(runner, state, metadata)
    assert state.session.context._gr_persistent_prefix_binding is state.prefix_binding


@pytest.mark.parametrize(
    "failure",
    [
        "released",
        "inactive",
        "slot",
        "context-owner",
        "session-owner",
        "batch",
        "binding-group",
        "binding-dtype",
        "fixed-dtype",
        "device",
        "canonical-capacity",
        "canonical-wider",
        "native-capacity",
        "metadata-identity",
        "cache-len-zero",
        "cache-len-mismatch",
        "prepared-step",
        "decision-step",
    ],
)
def test_prepare_validation_is_transactional_without_partial_device_writes(
    failure: str,
) -> None:
    runner, state, output, metadata, _group = _bind_and_decode()
    context = state.session.context
    context.decode_step_buf.fill_(71)
    context.decode_step_1based_buf.fill_(72)
    context.xllm_block_table_buf.fill_(73)
    runner.seq_lens.fill_(74)
    runner.optimistic_seq_lens_cpu.fill_(75)
    runner.query_start_loc.gpu.fill_(76)
    next_tokens = state.next_token_ids

    if failure == "released":
        state.released = True
    elif failure == "inactive":
        state.session.is_active = False
    elif failure == "slot":
        state.session.slot = 1
    elif failure == "context-owner":
        runner._beam_context = object()
    elif failure == "session-owner":
        runner._beam_sessions["gr"] = object()
    elif failure == "batch":
        runner.input_batch.num_reqs = 2
    elif failure == "binding-group":
        state.prefix_binding = (state.prefix_binding[0], state.prefix_binding[0])
    elif failure == "binding-dtype":
        state.prefix_binding = (state.prefix_binding[0].to(torch.int64),)
    elif failure == "fixed-dtype":
        context.decode_step_1based_buf = context.decode_step_1based_buf.to(torch.int64)
    elif failure == "device":
        runner.device = torch.device("meta")
    elif failure == "canonical-capacity":
        context.decision.sequence = torch.zeros((1, 3, 3), dtype=torch.int32)
    elif failure == "canonical-wider":
        context.decision.sequence = torch.zeros((1, 5, 3), dtype=torch.int32)
    elif failure == "native-capacity":
        runner.query_start_loc.gpu = runner.query_start_loc.gpu[:1]
    elif failure == "metadata-identity":
        output.beam_data["gr"] = dict(metadata)
    elif failure == "cache-len-zero":
        metadata["cache_len"] = 0
    elif failure == "cache-len-mismatch":
        metadata["cache_len"] -= 1
    elif failure == "prepared-step":
        state._prepared_step = 2
    elif failure == "decision-step":
        context.decision.worker_decode_steps = 2

    fixed_before = (
        context.decode_step_buf.clone(),
        context.decode_step_1based_buf.clone(),
        context.xllm_block_table_buf.clone(),
    )
    native_before = (
        runner.seq_lens.clone(),
        runner.optimistic_seq_lens_cpu.clone(),
        runner.query_start_loc.gpu.clone(),
    )

    with pytest.raises(RuntimeError):
        prepare_decode_metadata(runner, state, output, metadata)
    assert torch.equal(context.decode_step_buf, fixed_before[0])
    assert torch.equal(context.decode_step_1based_buf, fixed_before[1])
    assert torch.equal(context.xllm_block_table_buf, fixed_before[2])
    assert torch.equal(runner.seq_lens, native_before[0])
    assert torch.equal(runner.optimistic_seq_lens_cpu, native_before[1])
    assert torch.equal(runner.query_start_loc.gpu, native_before[2])
    assert not hasattr(context, "_gr_persistent_prefix_binding")
    assert state.next_token_ids is next_tokens
    assert state.step == 1
    assert state.session.decode_step == 0


@pytest.mark.parametrize(
    "failure",
    [
        "prefix-owner",
        "multi-group",
        "table-dtype",
        "table-device",
        "table-capacity",
        "wrong-stage",
    ],
)
def test_prefill_binding_failures_do_not_publish_state_or_context(failure: str) -> None:
    runner, state, group = _fixture(prefix_len=31)
    if failure == "prefix-owner":
        runner.requests["gr"].num_prompt_tokens = 30
    elif failure == "multi-group":
        runner.input_batch.block_table.block_tables.append(group)
    elif failure == "table-dtype":
        group.table = group.table.to(torch.int64)
    elif failure == "table-device":
        group.table = group.table.to("meta")
    elif failure == "table-capacity":
        group.table = torch.ones((1, 1), dtype=torch.int32)
    elif failure == "wrong-stage":
        state.stage = GRRequestStage.DECODE_RUNNING

    with pytest.raises(RuntimeError):
        bind_prefill_prefix(runner, state)
    assert state.prefix_binding is None
    assert not hasattr(state.session.context, "_gr_persistent_prefix_binding")


def test_finish_validation_precedes_all_length_buffer_writes() -> None:
    runner, state, output, metadata, _group = _bind_and_decode()
    prepare_decode_metadata(runner, state, output, metadata)
    runner.seq_lens.fill_(81)
    runner.optimistic_seq_lens_cpu.fill_(82)
    runner.query_start_loc.gpu.fill_(83)
    runner.query_start_loc.gpu = runner.query_start_loc.gpu[:1]

    with pytest.raises(RuntimeError, match="capacity"):
        finish_decode_metadata(runner, state, metadata)
    assert runner.seq_lens.tolist() == [81, 81, 81, 81]
    assert runner.optimistic_seq_lens_cpu.tolist() == [82, 82, 82, 82]
    assert runner.query_start_loc.gpu.tolist() == [83]


def test_real_npu_prepare_wrapper_orders_prepare_native_and_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, state, output, metadata, group = _bind_and_decode()
    events: list[str] = []
    runner._gr_model_runner_states = {"gr": state}
    runner._current_beam_info = {"req_ids": []}
    runner.input_ids = SimpleNamespace(gpu=torch.full((6,), -1, dtype=torch.int32))
    runner.use_async_scheduling = True
    runner.pcp_size = runner.dcp_size = 1
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
    runner.use_cp = False
    runner.speculative_config = None
    runner.enable_prompt_embeds = False
    runner.input_batch.req_prompt_embeds = {}
    runner.use_compress = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.use_async_spec_decode = False
    output.scheduled_spec_decode_tokens = {}
    output.has_structured_output_requests = False
    output.scheduled_encoder_inputs = {}

    # The execute adapter validates the logical SchedulerOutput first, then
    # publishes physical W for the NPU prepare frame.
    output.num_scheduled_tokens["gr"] = 1
    output.total_num_scheduled_tokens = 1
    del output._gr_npu_logical_geometry
    frame = _persistent_decode_frame(output)
    assert frame is not None
    output.num_scheduled_tokens["gr"] = state.beam_width
    output.total_num_scheduled_tokens = state.beam_width
    output._gr_npu_logical_geometry = ("gr", 1, state.beam_width)
    runner._gr_npu_input_frame = frame
    state.prepare_decode_inputs = lambda *_args: events.append("state")
    copy_ids = make_patched_prepare_input_ids(
        lambda *_args: pytest.fail("persistent path must not copy Host control tokens")
    )

    def original(_runner: Any, scheduled: Any, _counts: Any) -> tuple[Any, Any, int]:
        events.append("native")
        assert state.session.context.decode_step_buf.tolist() == [0]
        assert state.session.context.decode_step_1based_buf.tolist() == [1]
        copy_ids(_runner, scheduled, 1, 4, np.array([4], dtype=np.int32))
        _runner.input_batch.block_table.compute_slot_mapping(
            1,
            torch.tensor([0, 4], dtype=torch.int32),
            torch.arange(4, dtype=torch.int64),
        )
        _runner.seq_lens[0] = 999
        _runner.optimistic_seq_lens_cpu[0] = 999
        _runner.query_start_loc.gpu[:2] = torch.tensor([0, 4], dtype=torch.int32)
        return object(), object(), 4

    monkeypatch.setattr(npu_patch, "_beam_remap_inputs", lambda *_args: [0, 1, 2, 3])
    monkeypatch.setattr(
        npu_patch,
        "_materialize_logits_indices",
        lambda _runner, values: torch.tensor(values, dtype=torch.int32),
    )
    monkeypatch.setattr(npu_patch, "has_kv_transfer_group", lambda: False, raising=False)
    monkeypatch.setattr(npu_patch, "has_ec_transfer", lambda: False, raising=False)
    monkeypatch.setattr(
        "vllm_gr.v1.worker.npu_gr_inputs.has_kv_transfer_group",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm_gr.v1.worker.npu_gr_inputs.has_ec_transfer",
        lambda: False,
    )

    wrapped = npu_patch.make_patched_prepare_inputs_npu(original)
    result = wrapped(runner, output, [4])

    assert events == ["state", "native"]
    assert result[0].tolist() == [0, 1, 2, 3]
    assert runner.seq_lens[0].item() == 32
    assert runner.optimistic_seq_lens_cpu[0].item() == 32
    assert runner.query_start_loc.gpu[:2].tolist() == [0, 4]
    assert state.session.context._gr_persistent_prefix_binding is state.prefix_binding
    assert torch.all(group.slot_mapping.gpu == -1)
    assert runner.input_batch.block_table.block_tables == [group]
