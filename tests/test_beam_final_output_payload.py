# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production batch output ABI and independence from reused D2H storage."""

from itertools import product
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from vllm_gr.ops.cuda.beam_state import FINISH_LENGTH, FINISH_STOP
from vllm_gr.v1.beam.batch_contracts import BeamOutputOptions
from vllm_gr.v1.worker.beam_final_output import _build_batch_result, _PayloadLayout

if TYPE_CHECKING:
    from vllm_gr.v1.worker.gpu_beam_stage_runner import BeamRequestBinding


def bindings(batch_size: int) -> tuple["BeamRequestBinding", ...]:
    # Mirror the scalar binding without importing a CUDA runner on NPU hosts.
    return tuple(
        cast(
            "BeamRequestBinding",
            SimpleNamespace(
                session_id="session",
                item_index=i,
                native_request_id=f"native-{i}",
                state_row=i,
                kv_slot=i,
                workspace_row=i,
            ),
        )
        for i in range(batch_size)
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])  # type: ignore[misc]
@pytest.mark.parametrize("steps", [1, 2, 3])  # type: ignore[misc]
@pytest.mark.parametrize("as_tensor", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("flags", list(product([False, True], repeat=3)))  # type: ignore[misc]
def test_compact_payload_fields(
    batch_size: int, steps: int, as_tensor: bool, flags: tuple[bool, ...]
) -> None:
    include_stop, include_score, include_parents = flags
    layout = _PayloadLayout.create(4, steps, batch_size)
    values: npt.NDArray[np.int32] = np.zeros(layout.size, dtype=np.int32)
    for i in range(batch_size):
        start = 3 + i * layout.member_stride
        values[start] = 2  # Ignore the unused two rows of each item.
        rows = values[start + 1 : start + layout.member_stride].reshape(4, layout.row_stride)
        rows[0, :steps] = np.arange(10, 10 + steps) + 100 * i
        rows[1, :steps] = np.arange(20, 20 + steps) + 100 * i
        rows[:, steps : 2 * steps] = np.arange(steps)
        rows.view(np.float32)[:2, 2 * steps] = [-1.25 - i, -2.5 - i]
        stop_length = i % steps + 1
        rows[:2, 2 * steps + 1] = [steps, stop_length]
        rows[:2, 2 * steps + 2] = [FINISH_LENGTH, FINISH_STOP]
        rows[:2, 2 * steps + 3] = [-1, 20 + 100 * i + stop_length - 1]
    payload = torch.from_numpy(values) if as_tensor else values
    result = _build_batch_result(
        payload,
        layout,
        session_id="session",
        bindings=bindings(batch_size),
        options=BeamOutputOptions(2, include_stop, include_score, include_parents),
    )
    values.fill(0)  # Reusing D2H storage cannot mutate an already-published result.
    assert result.failure is None
    assert result.batch_id == "session"
    assert result.output is not None
    assert len(result.output.items) == batch_size
    for i, item in enumerate(result.output.items):
        assert (item.item_index, item.request_id) == (i, f"native-{i}")
        length, stop = item.sequences
        stop_length = i % steps + 1
        count = stop_length if include_stop else stop_length - 1
        assert length.token_ids == tuple(range(10 + 100 * i, 10 + 100 * i + steps))
        assert stop.token_ids == tuple(range(20 + 100 * i, 20 + 100 * i + count))
        assert length.cumulative_logprob == (-1.25 - i if include_score else None)
        assert stop.cumulative_logprob == (-2.5 - i if include_score else None)
        assert length.parent_beam_indices == (tuple(range(steps)) if include_parents else ())
        assert stop.parent_beam_indices == (tuple(range(stop_length)) if include_parents else ())
        assert (length.finish_reason, length.stop_token_id) == ("length", None)
        assert (stop.finish_reason, stop.stop_token_id) == ("stop", 20 + 100 * i + stop_length - 1)


@pytest.mark.parametrize(  # type: ignore[misc]
    "batch_size,empty_item", [(b, i) for b in (1, 2, 4) for i in range(b)]
)
def test_empty_compact_payload_failure(batch_size: int, empty_item: int) -> None:
    layout = _PayloadLayout.create(4, 3, batch_size)
    values: npt.NDArray[np.int32] = np.zeros(layout.size, dtype=np.int32)
    for i in range(batch_size):
        values[3 + i * layout.member_stride] = int(i != empty_item)
    result = _build_batch_result(
        values,
        layout,
        session_id="session",
        bindings=bindings(batch_size),
        options=BeamOutputOptions(4, True, True, True),
    )
    assert result.output is None  # No partial success if a later item fails.
    assert result.failure is not None
    assert result.failure.code == "no_valid_beams"
    assert result.failure.item_index == empty_item
