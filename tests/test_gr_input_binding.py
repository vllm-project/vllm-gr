# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-input contracts executable without an accelerator."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker.gr_inputs import PreparedGRInputs, bind_device_inputs


def make_binding_case(
    batch_size: int, step: int, device: str = "cpu"
) -> tuple[Any, PreparedGRInputs]:
    width = 2
    rows = batch_size * width
    bindings = tuple(SimpleNamespace(session_id="batch", item_index=i) for i in range(batch_size))
    state = SimpleNamespace(
        beam_width=width,
        is_initialized=torch.ones(batch_size, dtype=torch.bool, device=device),
        is_finished=torch.zeros(batch_size, dtype=torch.bool, device=device),
        error_code=torch.zeros(batch_size, dtype=torch.int32, device=device),
        num_generated_tokens=torch.full((batch_size,), step + 1, dtype=torch.int32, device=device),
        num_decode_steps=torch.full((batch_size,), step, dtype=torch.int32, device=device),
        active_mask=torch.ones((batch_size, width), dtype=torch.bool, device=device),
        input_ids=torch.arange(rows, dtype=torch.int32, device=device).reshape(batch_size, width),
        sequence=torch.arange(rows * 3, dtype=torch.int32, device=device).reshape(
            batch_size, width, 3
        ),
    )
    context = SimpleNamespace(decode_step_buf=torch.empty(1, dtype=torch.int32, device=device))
    owner = SimpleNamespace(
        bindings=bindings,
        beam=SimpleNamespace(state=state, session=SimpleNamespace(context=context)),
        state_rows=torch.arange(batch_size, device=device),
        execution_mask=torch.empty(rows, dtype=torch.bool, device=device),
        input_ids=torch.empty(rows, dtype=torch.int32, device=device),
        positions=torch.empty(rows, dtype=torch.int64, device=device),
        device_positions=torch.arange(rows, dtype=torch.int64, device=device) + 10,
        prefixes=torch.empty((rows, 3), dtype=torch.int32, device=device),
        metadata=SimpleNamespace(
            suffix_seqlens_kv_device=torch.empty(rows, dtype=torch.int32, device=device),
            suffix_cu_seqlens_k_device=torch.empty(rows + 1, dtype=torch.int32, device=device),
        ),
    )
    prepared = PreparedGRInputs(
        metadata=SimpleNamespace(session_id="batch", decode_step=step),
        binding=bindings,
        staging_slot=0,
        layout={},
    )
    return owner, prepared


def binding_addresses(owner: Any) -> tuple[int, ...]:
    return tuple(
        tensor.data_ptr()
        for tensor in (
            owner.input_ids,
            owner.positions,
            owner.execution_mask,
            owner.prefixes,
            owner.beam.session.context.decode_step_buf,
            owner.metadata.suffix_seqlens_kv_device,
            owner.metadata.suffix_cu_seqlens_k_device,
        )
    )


@pytest.mark.parametrize("batch_size, finished_index", [(2, 0), (2, 1), (4, 0), (4, 2)])
@pytest.mark.parametrize("step", [0, 1])
def test_early_finished_members_keep_shared_step_and_packed_suffix_extent(
    batch_size: int, finished_index: int, step: int
) -> None:
    owner, prepared = make_binding_case(batch_size, step)
    state = owner.beam.state
    rows = batch_size * state.beam_width
    addresses = binding_addresses(owner)
    bind_device_inputs(prepared, owner)
    state.is_finished[finished_index] = True
    state.active_mask[finished_index] = False
    # An early-finished member may have older counters than its live peers.
    state.num_generated_tokens[finished_index] = 1
    state.num_decode_steps[finished_index] = 0
    bind_device_inputs(prepared, owner)
    assert state.error_code.tolist() == [0] * batch_size
    assert owner.beam.session.context.decode_step_buf.item() == step
    assert owner.metadata.suffix_seqlens_kv_device.tolist() == [step + 1] * rows
    assert owner.metadata.suffix_cu_seqlens_k_device.tolist() == [
        row * (step + 1) for row in range(rows + 1)
    ]
    expected_mask = [i != finished_index for i in range(batch_size) for _ in range(2)]
    assert owner.execution_mask.tolist() == expected_mask
    assert owner.input_ids[~owner.execution_mask].count_nonzero() == 0
    assert owner.positions[~owner.execution_mask].count_nonzero() == 0
    assert owner.prefixes[~owner.execution_mask].eq(-1).all()
    assert binding_addresses(owner) == addresses

    state.is_finished.fill_(True)
    bind_device_inputs(prepared, owner)
    assert owner.beam.session.context.decode_step_buf.item() == -1
    assert owner.execution_mask.count_nonzero() == 0
    assert owner.metadata.suffix_seqlens_kv_device.tolist() == [1] * rows
    assert owner.metadata.suffix_cu_seqlens_k_device.tolist() == list(range(rows + 1))
    assert binding_addresses(owner) == addresses

    # The same fixed buffers can be rebound for a subsequent live request.
    state.is_finished.zero_()
    state.active_mask.fill_(True)
    state.num_generated_tokens.fill_(step + 1)
    state.num_decode_steps.fill_(step)
    bind_device_inputs(prepared, owner)
    assert owner.execution_mask.all()
    assert owner.beam.session.context.decode_step_buf.item() == step
    assert binding_addresses(owner) == addresses


def exercise_mixed_batch_kv(
    batch_size: int,
    finished_index: int,
    step: int,
    *,
    device: str,
    graph_mode: bool,
) -> None:
    """Real production kernels: used by CUDA tests and the CPU interpreter."""
    from vllm_gr.v1.attention.backends.beam_graph_kernels import (
        cache_unshared_kv_graph,
        copy_suffix_kv_graph,
    )

    owner, prepared = make_binding_case(batch_size, step, device)
    rows = batch_size * owner.beam.state.beam_width
    addresses = binding_addresses(owner)
    pool_key = torch.arange(rows * 3 * 8, dtype=torch.float32, device=device).reshape(
        1, rows, 1, 3, 8
    )
    pool_value = -pool_key.clone()
    current = torch.arange(rows * 8, dtype=torch.float32, device=device).reshape(rows, 1, 8) + 1000
    suffix_key = torch.full((rows * 3, 1, 8), -777.0, device=device)
    suffix_value = suffix_key.clone()

    def execute() -> None:
        bind_device_inputs(prepared, owner)
        cache_unshared_kv_graph(
            pool_key,
            pool_value,
            current,
            -current,
            0,
            owner.beam.session.context.decode_step_buf,
            owner.execution_mask,
        )
        copy_suffix_kv_graph(
            pool_key,
            pool_value,
            suffix_key,
            suffix_value,
            0,
            owner.beam.session.context.decode_step_buf,
            beam_width=rows,
            active_mask=owner.execution_mask,
        )

    replay = execute
    execute()
    if graph_mode:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute()
        replay = graph.replay
    owner.beam.state.is_finished[finished_index] = True
    owner.beam.state.active_mask[finished_index] = False
    before = pool_key.clone()
    # Distinct new KV proves live rows were written on the mixed-mask replay.
    current.add_(500)
    replay()
    live = owner.execution_mask
    assert owner.beam.session.context.decode_step_buf.item() == step
    torch.testing.assert_close(pool_key[0, ~live], before[0, ~live])
    torch.testing.assert_close(pool_key[0, live, :, step], current[live])
    expected = pool_key[0, :, :, : step + 1].permute(0, 2, 1, 3).clone()
    expected[~live] = 0
    expected = expected.reshape(rows * (step + 1), 1, 8)
    torch.testing.assert_close(suffix_key[: expected.shape[0]], expected)
    torch.testing.assert_close(suffix_value[: expected.shape[0]], -expected)
    assert owner.metadata.suffix_cu_seqlens_k_device.tolist() == [
        row * (step + 1) for row in range(rows + 1)
    ]
    assert binding_addresses(owner) == addresses

    owner.beam.state.is_finished.fill_(True)
    before = pool_key.clone()
    replay()
    torch.testing.assert_close(pool_key, before)
    assert suffix_key[:rows].count_nonzero() == 0
    assert suffix_value[:rows].count_nonzero() == 0
    assert owner.metadata.suffix_cu_seqlens_k_device.tolist() == list(range(rows + 1))
    assert binding_addresses(owner) == addresses
