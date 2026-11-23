# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""CPU regressions for immutable temperatures and reusable NPU prefix scratch."""

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.worker.npu.npu_beam_stage_runner import NPUEagerBeamStageRunner


def _runner(worker: SimpleNamespace | None = None) -> NPUEagerBeamStageRunner:
    runner = object.__new__(NPUEagerBeamStageRunner)
    runner.worker = worker or SimpleNamespace()
    runner.logits = torch.zeros(3, 6)
    runner.worker._current_scheduler_output = SimpleNamespace(
        gr_stage_metadata={"session": SimpleNamespace(beam_params=SimpleNamespace(temperature=0.0))}
    )
    runner.dynamic_state = None
    runner.attention_batch = None
    return runner


def _plan(kind: str, rows: int, step: int = 0, width: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        execution=SimpleNamespace(
            kind=kind,
            num_execution_rows=rows,
            stage_index=step,
            input_width=width,
            batch_size=1,
        )
    )


def test_shared_temperature_is_reused_across_rows_stages_and_sessions() -> None:
    runner = _runner()
    first = runner._constraint_temperatures()  # noqa: SLF001
    assert first.tolist() == [0.0] and first.dtype == torch.float32
    runner.logits = torch.zeros(12, 6)
    assert runner._constraint_temperatures() is first  # noqa: SLF001
    another = _runner(runner.worker)
    assert another._constraint_temperatures() is first  # noqa: SLF001
    params = next(
        iter(another.worker._current_scheduler_output.gr_stage_metadata.values())
    ).beam_params
    params.temperature = 0.5
    changed = another._constraint_temperatures()  # noqa: SLF001
    assert changed.tolist() == [0.5] and changed is not first
    assert first.tolist() == [0.0]  # Replacing the cache never mutates an in-flight scalar.


def test_prefix_scratch_keeps_prefill_zeros_separate_from_decode_updates() -> None:
    runner = _runner()
    prefill = runner._constraint_prefixes(_plan("prefill", 3))  # noqa: SLF001
    smaller = runner._constraint_prefixes(_plan("prefill", 1))  # noqa: SLF001
    assert prefill.data_ptr() == smaller.data_ptr()
    runner.attention_batch = SimpleNamespace(
        context=SimpleNamespace(
            decision=SimpleNamespace(
                sequence=torch.tensor([[[1, 2, 9], [3, 5, 9], [1, 4, 9]]], dtype=torch.int32)
            )
        )
    )
    first_decode = runner._constraint_prefixes(_plan("decode", 3, step=1))  # noqa: SLF001
    assert first_decode.tolist() == [[1, 0, 0], [3, 0, 0], [1, 0, 0]]
    second_decode = runner._constraint_prefixes(_plan("decode", 3, step=2))  # noqa: SLF001
    assert first_decode.data_ptr() == second_decode.data_ptr()
    assert second_decode.tolist() == [[1, 2, 0], [3, 5, 0], [1, 4, 0]]
    assert prefill.tolist() == [[0, 0, 0]] * 3


def test_prefix_scratch_handles_dynamic_beam_narrowing_and_growth() -> None:
    runner = _runner()
    # Dynamic decisions still belong to an admitted persistent attention batch.
    runner.attention_batch = SimpleNamespace()
    runner.dynamic_state = SimpleNamespace(
        sequence=torch.tensor(
            [[[7, 8, 0], [9, 10, 0], [11, 12, 0], [13, 14, 0]]], dtype=torch.int32
        )
    )
    narrow = runner._constraint_prefixes(_plan("decode", 2, step=2, width=2))  # noqa: SLF001
    assert narrow.tolist() == [[7, 8, 0], [9, 10, 0]]
    wider = runner._constraint_prefixes(_plan("decode", 4, step=2, width=4))  # noqa: SLF001
    assert wider.tolist() == [[7, 8, 0], [9, 10, 0], [11, 12, 0], [13, 14, 0]]
    assert narrow.tolist() == [[7, 8, 0], [9, 10, 0]]
    again = runner._constraint_prefixes(_plan("decode", 2, step=2, width=2))  # noqa: SLF001
    assert again.data_ptr() == wider.data_ptr()


pytestmark = [pytest.mark.cpu_test, pytest.mark.l2]
