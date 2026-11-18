# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compare the GR input specialization with the installed native implementation."""

from __future__ import annotations

import inspect
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_gr.v1.engine.beam_search_v1 import GRStageKind
from vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch import (
    make_patched_prepare_inputs_gpu,
)
from vllm_gr.v1.worker.cuda.v1.gpu_prefill_inputs import try_prepare_gr_prefill_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
# pytest parametrization decorators are untyped under strict mypy.
# mypy: disable-error-code="misc"

NATIVE_PREPARE = inspect.unwrap(GPUModelRunner._prepare_inputs)
BUFFER_NAMES = (
    "input_ids",
    "query_start_loc",
    "query_pos",
    "req_indices",
    "prev_positions",
    "num_scheduled_tokens",
    "discard_request_mask",
    "num_accepted_tokens",
)
TENSOR_NAMES = ("positions", "seq_lens", "num_computed_tokens", "optimistic_seq_lens_cpu")


def make_runtime(
    computed: int = 0,
    scheduled: int = 1024,
    prompt: int = 1024,
    capacity: int = 4,
    block_sizes: tuple[int, ...] = (16,),
) -> tuple[SimpleNamespace, SimpleNamespace, NDArray[np.int32]]:
    device = torch.device("cuda:0")
    max_tokens, max_model_len = 2048, 4096

    def buffer(size: int, dtype: torch.dtype) -> CpuGpuBuffer:
        b = CpuGpuBuffer(size, dtype=dtype, device=device, pin_memory=True)
        b.cpu.fill_(37)
        b.gpu.fill_(37)
        return b

    table = MultiGroupBlockTable(
        max_num_reqs=capacity,
        max_num_batched_tokens=max_tokens,
        pin_memory=True,
        device=device,
        block_sizes=list(block_sizes),
        kernel_block_sizes=list(block_sizes),
        max_num_blocks=[max_model_len // size for size in block_sizes],
    )
    for group, size in enumerate(block_sizes):
        table[group].block_table.cpu.fill_(0)
        table[group].block_table.gpu.fill_(0)
        table[group].add_row(list(range(3, 3 + (prompt + size - 1) // size)), 0)
        table[group].slot_mapping.gpu.fill_(37)
    tokens_cpu = (
        torch.arange(capacity * max_model_len, dtype=torch.int32)
        .reshape(capacity, max_model_len)
        .pin_memory()
    )
    computed_cpu = torch.full((capacity,), 37, dtype=torch.int32, pin_memory=True)
    computed_cpu[0] = computed
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["r"],
        num_computed_tokens_cpu=computed_cpu.numpy(),
        num_computed_tokens_cpu_tensor=computed_cpu,
        num_prompt_tokens=np.array([prompt]),
        token_ids_cpu=tokens_cpu.numpy(),
        token_ids_cpu_tensor=tokens_cpu,
        req_prompt_embeds={},
        prev_sampled_token_ids=None,
        prev_req_id_to_index={},
        block_table=table,
    )
    runner = SimpleNamespace(
        device=device,
        input_batch=batch,
        max_num_tokens=max_tokens,
        max_num_reqs=capacity,
        uses_mrope=False,
        mrope_num_dims=0,
        _pp_recv_work=None,
        enable_prompt_embeds=False,
        lora_config=None,
        use_async_spec_decode=False,
        use_async_scheduling=True,
        num_accepted_tokens_event=None,
        mamba_prev_last_scheduled_idx=None,
        input_ids=buffer(max_tokens, torch.int32),
        query_start_loc=buffer(capacity + 1, torch.int32),
        query_pos=buffer(max_tokens, torch.int64),
        req_indices=buffer(max_tokens, torch.int64),
        prev_positions=buffer(capacity, torch.int64),
        num_scheduled_tokens=buffer(capacity, torch.int32),
        discard_request_mask=buffer(capacity, torch.bool),
        num_accepted_tokens=buffer(capacity, torch.int32),
        positions=torch.full((max_tokens,), 37, dtype=torch.int64, device=device),
        seq_lens=torch.full((capacity,), 37, dtype=torch.int32, device=device),
        num_computed_tokens=torch.full((capacity,), 37, dtype=torch.int32, device=device),
        optimistic_seq_lens_cpu=torch.full((capacity,), 37, dtype=torch.int32, pin_memory=True),
        arange_np=np.arange(max_tokens, dtype=np.int64),
        requests={"r": SimpleNamespace(num_tokens=prompt)},
    )
    for name in ("_compute_prev_positions", "_prepare_input_ids", "_get_cumsum_and_arange"):
        setattr(runner, name, MethodType(getattr(GPUModelRunner, name), runner))
    output = SimpleNamespace(
        gr_stage_metadata={
            "r": SimpleNamespace(kind=GRStageKind.PREFILL, num_prompt_tokens=prompt)
        },
        beam_data={},
        total_num_scheduled_tokens=scheduled,
        scheduled_spec_decode_tokens={},
    )
    torch.accelerator.synchronize(device)
    return runner, output, np.array([scheduled], dtype=np.int32)


def snapshot(runner: SimpleNamespace) -> dict[str, torch.Tensor]:
    torch.accelerator.synchronize(runner.device)
    state: dict[str, torch.Tensor] = {}
    for name in BUFFER_NAMES:
        b = getattr(runner, name)
        state[name + ".cpu"] = b.cpu.clone()
        state[name + ".gpu"] = b.gpu.cpu()
    for name in TENSOR_NAMES:
        state[name] = getattr(runner, name).cpu().clone()
    for i, table in enumerate(runner.input_batch.block_table.block_tables):
        state[f"blocks.{i}"] = table.block_table.gpu.cpu()
        state[f"slots.{i}"] = table.slot_mapping.gpu.cpu()
    return state


def assert_same(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize(
    "computed,scheduled,prompt",
    [
        (0, 1024, 1024),
        (1008, 16, 1024),
        (1023, 1, 1024),
        (0, 1, 1),
        (0, 37, 1024),
        (37, 259, 1024),
        (1000, 24, 1024),
        (31, 2017, 2048),
    ],
)
@pytest.mark.parametrize("capacity,block_sizes", [(1, (16,)), (4, (16, 32)), (513, (16,))])
def test_all_native_buffers_match(
    computed: int, scheduled: int, prompt: int, capacity: int, block_sizes: tuple[int, ...]
) -> None:
    native, output, counts = make_runtime(computed, scheduled, prompt, capacity, block_sizes)
    fast, _, _ = make_runtime(computed, scheduled, prompt, capacity, block_sizes)
    # Test a surviving request's previous-position bookkeeping too.
    native.input_batch.prev_req_id_to_index = {"r": capacity - 1}
    fast.input_batch.prev_req_id_to_index = {"r": capacity - 1}
    expected_indices, spec, max_sampled = NATIVE_PREPARE(native, output, counts)

    def forbidden_fallback(*args: Any) -> None:
        raise AssertionError("supported GR Prefill must use the fast path")

    actual_indices, actual_spec, actual_max = make_patched_prepare_inputs_gpu(forbidden_fallback)(
        fast, output, counts
    )
    assert actual_indices is not None and actual_spec is spec is None
    assert actual_max == max_sampled == 1
    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    assert_same(snapshot(fast), snapshot(native))


@pytest.mark.parametrize(
    "target,attribute,value",
    [
        pytest.param("output", "gr_stage_metadata", {}, id="ordinary"),
        pytest.param("output", "beam_data", {"r": {}}, id="legacy"),
        pytest.param("stage", "kind", GRStageKind.DECODE, id="decode"),
        pytest.param("batch", "req_ids", ["other"], id="identity"),
        pytest.param("batch", "num_reqs", 2, id="multi_request"),
        pytest.param(
            "output", "gr_stage_metadata", {"r": object(), "other": object()}, id="multi_stage"
        ),
        pytest.param("runner", "uses_mrope", True, id="mrope"),
        pytest.param("runner", "_pp_recv_work", object(), id="pending_pp_broadcast"),
        pytest.param("runner", "enable_prompt_embeds", True, id="embeds"),
        pytest.param("batch", "req_prompt_embeds", {0: object()}, id="request_embeds"),
        pytest.param("runner", "lora_config", object(), id="lora"),
        pytest.param("runner", "use_async_spec_decode", True, id="async_spec"),
        pytest.param("output", "scheduled_spec_decode_tokens", {"r": [1]}, id="spec_tokens"),
        pytest.param("runner", "num_accepted_tokens_event", object(), id="accepted_event"),
        pytest.param("runner", "mamba_prev_last_scheduled_idx", object(), id="mamba"),
        pytest.param("batch", "prev_sampled_token_ids", object(), id="previous_sample"),
        pytest.param("counts", 0, 0, id="zero"),
        pytest.param("output", "total_num_scheduled_tokens", 12, id="total"),
        pytest.param("runner", "max_num_tokens", 512, id="capacity"),
        pytest.param("computed", 0, -1, id="negative_computed"),
        pytest.param("computed", 0, 1000, id="beyond_prompt"),
        pytest.param("stage", "num_prompt_tokens", 42, id="prompt_mismatch"),
    ],
)
def test_unsupported_inputs_fall_back_without_mutation(
    target: str, attribute: str | int, value: Any
) -> None:
    runner, output, counts = make_runtime()
    targets: dict[str, Any] = {
        "runner": runner,
        "batch": runner.input_batch,
        "output": output,
        "stage": output.gr_stage_metadata["r"],
        "counts": counts,
        "computed": runner.input_batch.num_computed_tokens_cpu,
    }
    if isinstance(attribute, str):
        setattr(targets[target], attribute, value)
    else:
        targets[target][attribute] = value
    before = snapshot(runner)
    sentinel = (object(), object(), 7)
    calls: list[SimpleNamespace] = []

    def original(
        self: SimpleNamespace, out: SimpleNamespace, scheduled: NDArray[np.int32]
    ) -> tuple[object, object, int]:
        assert_same(snapshot(self), before)
        calls.append(out)
        return sentinel

    # Legacy remapping is a separate contract; verify its decline directly.
    if output.beam_data:
        assert try_prepare_gr_prefill_inputs(runner, output, counts) is None
    else:
        assert make_patched_prepare_inputs_gpu(original)(runner, output, counts) == sentinel
        assert calls == [output]
    assert_same(snapshot(runner), before)


@pytest.mark.parametrize("section_name", ["mrope_section", "xdrope_section"])
@pytest.mark.parametrize("num_sections", [2, 3, 4])
def test_multidimensional_rope_uses_native_guard_without_old_runner_field(
    section_name: str, num_sections: int
) -> None:
    from transformers import PretrainedConfig
    from vllm.transformers_utils.config import mrope_num_dims, uses_mrope

    config = PretrainedConfig(rope_parameters={section_name: [16] * num_sections})
    runner, output, counts = make_runtime()
    runner.uses_mrope = uses_mrope(config)
    runner.mrope_num_dims = mrope_num_dims(config)
    assert runner.uses_mrope and runner.mrope_num_dims == max(3, num_sections)
    assert not hasattr(runner, "uses_xdrope_dim")
    before = snapshot(runner)
    assert try_prepare_gr_prefill_inputs(runner, output, counts) is None
    assert_same(snapshot(runner), before)


def test_pending_pp_receive_uses_native_wait_and_input_preparation() -> None:
    native, output, counts = make_runtime()
    fallback, _, _ = make_runtime()
    waits: list[str] = []
    native._pp_recv_work = SimpleNamespace(wait=lambda: waits.append("native"))
    fallback._pp_recv_work = SimpleNamespace(wait=lambda: waits.append("fallback"))

    expected_indices, expected_spec, expected_max = NATIVE_PREPARE(native, output, counts)
    actual_indices, actual_spec, actual_max = make_patched_prepare_inputs_gpu(NATIVE_PREPARE)(
        fallback, output, counts
    )

    assert waits == ["native", "fallback"]
    assert native._pp_recv_work is fallback._pp_recv_work is None
    assert actual_spec is expected_spec is None
    assert actual_max == expected_max == 1
    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    assert_same(snapshot(fallback), snapshot(native))


def test_reuse_refreshes_current_tokens_blocks_and_preserves_previous_logits_indices() -> None:
    native, output, counts = make_runtime()
    fast, _, _ = make_runtime()
    saved: list[tuple[torch.Tensor, int]] = []
    for computed, scheduled, block_start in [(0, 1024, 3), (1008, 16, 99), (11, 29, 201)]:
        # Same host-write boundary as the native input-prep fence.
        torch.accelerator.synchronize(fast.device)
        output.total_num_scheduled_tokens = scheduled
        counts[0] = scheduled
        for runner in (native, fast):
            runner.input_batch.num_computed_tokens_cpu[0] = computed
            runner.input_batch.token_ids_cpu[0] += 17
            runner.input_batch.block_table[0].add_row(list(range(block_start, block_start + 64)), 0)
        expected, _, max_sampled = NATIVE_PREPARE(native, output, counts)
        assert max_sampled == 1
        actual = try_prepare_gr_prefill_inputs(fast, output, counts)
        assert actual is not None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert_same(snapshot(fast), snapshot(native))
        saved.append((actual, scheduled - 1))
    for indices, expected in saved:
        assert indices.item() == expected
