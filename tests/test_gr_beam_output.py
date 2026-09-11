# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.gr_persistent_beam_helpers import (
    _backend,
    _context,
    _output,
    _prefill_inputs,
    _spec,
    _stage,
)
from vllm_gr.v1.beam.batch_contracts import BeamOutputOptions, validate_beam_batch_result
from vllm_gr.v1.worker.gr_beam_output import GRBeamFinalOutput
from vllm_gr.v1.worker.gr_persistent_beam import GRDecodeDeviceInputs


def _options(count=2, scores=True, parents=True):
    return BeamOutputOptions(count, False, scores, parents)


def test_final_only_output_preserves_precision_and_survives_slot_reuse():
    backend, _ = _backend(
        [
            _output([[20, 21], [22, 23]], [[-0.3, -0.4], [-0.05, -0.06]]),
            _output([[30, 31], [32, 33]], [[-0.1, -0.3], [-0.4, -0.05]]),
        ]
    )
    output = GRBeamFinalOutput(backend)
    output.create_session(_spec(), _options())
    ack = backend.bootstrap_from_prefill(_stage("session-0", 0, 3), _prefill_inputs())
    assert ack.enqueued and not ack.terminal
    with pytest.raises(RuntimeError, match="final view is unavailable"):
        output.enqueue_final("session-0")
    inputs = torch.empty(2, dtype=torch.int32)
    for step in (1, 2):
        stage = _stage("session-0", step, 3)
        backend.prepare_decode_inputs(stage, GRDecodeDeviceInputs(inputs))
        ack = backend.advance_beam(stage, torch.zeros((2, 8)))
    assert ack.terminal
    output.enqueue_final("session-0")
    output.enqueue_final("session-0")
    result = output.get_result("session-0")
    assert output.get_result("session-0") is result
    validate_beam_batch_result(result)
    sequences = result.output.items[0].sequences
    assert [seq.token_ids for seq in sequences] == [(11, 23, 33), (11, 22, 30)]
    assert [seq.parent_beam_indices for seq in sequences] == [(0, 1, 1), (0, 1, 0)]
    assert [seq.cumulative_logprob for seq in sequences] == pytest.approx([-0.31, -0.35])
    assert all(seq.finish_reason == "length" and seq.stop_token_id is None for seq in sequences)
    pointers = (output._device.data_ptr(), output._host.data_ptr())
    output.cleanup_session("session-0")
    output.cleanup_session("session-0", abort=True)
    assert backend.metrics.state_released == 1
    assert not output._sessions
    output.create_session(_spec(session_id="next", max_tokens=1), _options())
    backend.bootstrap_from_prefill(_stage("next", 0, 1), _prefill_inputs())
    output.enqueue_final("next")
    output.get_result("next")
    output.cleanup_session("next")
    assert pointers == (output._device.data_ptr(), output._host.data_ptr())
    assert sequences[0].token_ids == (11, 23, 33)


@pytest.mark.parametrize("valid", [True, False])
def test_output_options_and_empty_beams(valid):
    backend, _ = _backend([])
    output = GRBeamFinalOutput(backend)
    output.create_session(_spec(max_tokens=1), _options(count=1, scores=False, parents=False))
    inputs = _prefill_inputs()
    inputs.valid_mask.fill_(valid)
    backend.bootstrap_from_prefill(_stage("session-0", 0, 1), inputs)
    output.enqueue_final("session-0")
    result = output.get_result("session-0")
    if valid:
        assert len(result.output.items[0].sequences) == 1
        sequence = result.output.items[0].sequences[0]
        assert sequence.cumulative_logprob is None
        assert sequence.parent_beam_indices == ()
    else:
        assert result.output is None
        assert result.failure.code == "no_valid_beams"
    output.cleanup_session("session-0")


class _PendingEvent:
    ready = False

    def query(self):
        return self.ready

    def synchronize(self):
        raise AssertionError("cleanup/poll must not synchronize")


def test_pending_copy_blocks_output_and_slot_reuse_without_host_wait():
    backend, _ = _backend([])
    output = GRBeamFinalOutput(backend)
    output.create_session(_spec(max_tokens=1), _options())
    backend.bootstrap_from_prefill(_stage("session-0", 0, 1), _prefill_inputs())
    output.enqueue_final("session-0")
    event = _PendingEvent()
    output._sessions["session-0"].event = event
    state = backend.registry.get("session-0")
    state.terminal_event = event
    state.terminal_event_recorded = True
    assert output.get_result("session-0") is None
    output.cleanup_session("session-0", abort=True)
    assert backend.context.registry.has_session("session-0")
    with pytest.raises(RuntimeError):
        output.create_session(_spec(session_id="next"), _options())
    event.ready = True
    output.reap_completed()
    assert not output._sessions
    output.create_session(_spec(session_id="next"), _options())
    output.cleanup_session("next", abort=True)


@pytest.mark.parametrize("bootstrapped", [False, True])
def test_unified_abort_and_failure_cleanup(bootstrapped, monkeypatch):
    backend, _ = _backend([])
    output = GRBeamFinalOutput(backend)
    output.create_session(_spec(max_tokens=1), _options())
    if bootstrapped:
        backend.bootstrap_from_prefill(_stage("session-0", 0, 1), _prefill_inputs())
        output.enqueue_final("session-0")

        def fail(*args):
            raise ValueError("materialization failed")

        monkeypatch.setattr(output, "_build_result", fail)
        with pytest.raises(ValueError, match="materialization failed"):
            output.get_result("session-0")
    output.cleanup_session("session-0", abort=True)
    output.cleanup_session("session-0", abort=True)
    assert backend.metrics.state_released == 1
    assert not output._sessions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [2, 128])
def test_cuda_final_single_d2h_and_nonblocking_cleanup(tmp_path, monkeypatch, width):
    device = torch.device("cuda:0")
    backend, _ = _backend([], context=_context(device=device, width=width))
    output = GRBeamFinalOutput(backend)
    output.create_session(_spec(max_tokens=1, width=width), _options(count=width))
    from vllm_gr.v1.worker.gr_persistent_beam import GRPrefillDeviceInputs

    inputs = GRPrefillDeviceInputs(
        torch.arange(10, 10 + width, dtype=torch.int32, device=device).view(1, width),
        -torch.arange(width, dtype=torch.float32, device=device).view(1, width),
        torch.ones((1, width), dtype=torch.bool, device=device),
        7,
    )
    backend.bootstrap_from_prefill(_stage("session-0", 0, 1), inputs)
    copies = []
    original_copy = torch.Tensor.copy_

    def copy_spy(dst, src, *args, **kwargs):
        if dst.device.type == "cpu" and src.device.type == "cuda":
            copies.append((src.numel() * src.element_size(), kwargs.get("non_blocking")))
        return original_copy(dst, src, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("enqueue/cleanup attempted Host materialization or synchronization")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "copy_", copy_spy)
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "tolist", forbidden)
            patch.setattr(torch.Tensor, "item", forbidden)
            patch.setattr(torch.cuda.Event, "synchronize", forbidden)
            patch.setattr(torch.cuda, "synchronize", forbidden)
            with torch.profiler.record_function("a5_enqueue"):
                output.enqueue_final("session-0")
                output.enqueue_final("session-0")
        result = output.get_result("session-0", wait=True)
        output.cleanup_session("session-0")
    expected_bytes = width * 8 * 4
    assert copies == [(expected_bytes, True)]
    assert output.final_d2h_count == 1 and output.final_d2h_bytes == expected_bytes
    assert result.output.items[0].sequences[0].token_ids == (10,)
    import json

    trace = tmp_path / "final-output.json"
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    transfers = [e for e in events if e.get("cat") == "gpu_memcpy" and "DtoH" in e["name"]]
    assert len(transfers) == 1
    assert transfers[0]["args"]["bytes"] == expected_bytes
    enqueue = next(
        e for e in events if e.get("cat") == "user_annotation" and e["name"] == "a5_enqueue"
    )
    assert not any(
        e.get("cat") == "cuda_runtime"
        and "Synchronize" in e["name"]
        and enqueue["ts"] <= e["ts"] < enqueue["ts"] + enqueue["dur"]
        for e in events
    )
    assert not output._sessions


def test_pack_failure_uses_unified_cleanup(monkeypatch):
    backend, _ = _backend([])
    output = GRBeamFinalOutput(backend)
    output.create_session(_spec(max_tokens=1), _options())
    backend.bootstrap_from_prefill(_stage("session-0", 0, 1), _prefill_inputs())
    original = torch.Tensor.copy_

    def fail(dst, src, *args, **kwargs):
        if dst.data_ptr() == output._host.data_ptr():
            raise RuntimeError("copy submission failed")
        return original(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail)
    with pytest.raises(RuntimeError, match="copy submission failed"):
        output.enqueue_final("session-0")
    output.cleanup_session("session-0", abort=True)
    assert backend.metrics.state_released == 1
    assert not output._sessions


def test_two_final_slots_remain_isolated():
    backend, _ = _backend([], context=_context(max_batch=2))
    output = GRBeamFinalOutput(backend)
    for session_id, offset in (("first", 0), ("second", 100)):
        output.create_session(_spec(session_id=session_id, max_tokens=1), _options())
        inputs = _prefill_inputs()
        inputs.candidate_token_ids.add_(offset)
        backend.bootstrap_from_prefill(_stage(session_id, 0, 1), inputs)
        output.enqueue_final(session_id)
    first = output.get_result("first")
    output.cleanup_session("first")
    second = output.get_result("second")
    output.cleanup_session("second")
    assert first.output.items[0].sequences[0].token_ids == (10,)
    assert second.output.items[0].sequences[0].token_ids == (110,)
    assert not output._sessions
