# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Upload retirement failures must never make in-flight storage reusable."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.v1.worker import gpu_prefill_v2 as module

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_event_record_failure_prevents_upload_slot_reuse(monkeypatch: Any) -> None:
    slot = object.__new__(module._PrefillUploadSlot)
    slot.gpu = SimpleNamespace(device="cuda")
    slot.done = Mock(record=Mock(side_effect=RuntimeError("record failed")))
    slot.submitted = False
    monkeypatch.setattr(module.torch.accelerator, "current_stream", lambda device: None)
    with pytest.raises(RuntimeError, match="record failed"):
        slot.record()
    with pytest.raises(RuntimeError, match="completion event"):
        slot.wait()
    slot.done.query.assert_not_called()


@pytest.mark.parametrize("submission_fails", [False, True])
def test_fused_record_failure_preserves_submission_error(
    monkeypatch: Any, submission_fails: bool
) -> None:
    from vllm_gr.ops.cuda import gr_prefill_inputs

    submission_error = RuntimeError("submission failed")
    record_error = RuntimeError("record failed")
    slot = object.__new__(module._PrefillUploadSlot)
    slot.gpu = SimpleNamespace(device="cuda")
    slot.done = Mock(record=Mock(side_effect=record_error))
    slot.submitted = False
    slot.array = module.np.empty(6, dtype=module.np.int32)
    monkeypatch.setattr(slot, "copy", Mock(return_value=object()))
    monkeypatch.setattr(module.torch.accelerator, "current_stream", lambda device: None)
    kernel = Mock()
    kernel.__getitem__ = Mock(
        return_value=Mock(side_effect=submission_error if submission_fails else None)
    )
    monkeypatch.setattr(gr_prefill_inputs, "apply_gr_prefill_state_writes_v2", kernel)

    def staged(contents: list[int]) -> Any:
        return SimpleNamespace(
            gpu=Mock(stride=Mock(return_value=3)),
            _staged_write_indices=[0],
            _staged_write_contents=contents,
        )

    states = SimpleNamespace(
        total_len=staged([3]), all_token_ids=staged([1, 2, 3]), num_computed_tokens=staged([0])
    )
    blocks = staged([1])
    worker = SimpleNamespace(
        req_states=states,
        block_tables=SimpleNamespace(block_tables=[blocks]),
        _gr_prefill_upload_buffers=SimpleNamespace(acquire=Mock(return_value=slot)),
    )
    with pytest.raises(RuntimeError) as raised:
        module._apply_fused_prefill_state_writes(worker)
    assert raised.value is (submission_error if submission_fails else record_error)
    if submission_fails:
        assert any("record failed" in note for note in submission_error.__notes__)
    assert states.all_token_ids._staged_write_indices == [0]
    assert blocks._staged_write_indices == [0]
    with pytest.raises(RuntimeError, match="completion event"):
        slot.wait()
