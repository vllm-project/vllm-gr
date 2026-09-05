# SPDX-License-Identifier: Apache-2.0
"""CPU integration contracts for the composite persistent NPU backend."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.config.models import ConstraintTableBackendEnum
from vllm_gr.constraints.backend import ConstraintMetadata, ConstraintOutput
from vllm_gr.v1.beam.beam_attention_context import BeamDecisionBuffers
from vllm_gr.v1.worker.gr_model_runner_state import (
    GRModelRunnerState,
    unpack_gr_final_output,
)
from vllm_gr.v1.worker.npu_gr_backend import NPUGRDeviceBackend
from vllm_gr.v1.worker.npu_gr_workspace import NPUFinalPack


class _ConstraintBackend:
    def __init__(self, outputs: list[ConstraintOutput], width: int, vocab: int) -> None:
        self.outputs = outputs
        self.calls: list[tuple[torch.Tensor, ConstraintMetadata]] = []
        self.info = SimpleNamespace(
            max_rows=width,
            max_top_k=width,
            backend=ConstraintTableBackendEnum.NPU,
            table_metadata=SimpleNamespace(vocab_size=vocab),
        )

    def select(self, logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        self.calls.append((logits, metadata))
        return self.outputs.pop(0)


class _Session:
    def __init__(self, width: int, steps: int) -> None:
        self.session_id = "gr"
        self.slot = 0
        self._state = SimpleNamespace(
            is_active=True,
            beam_width=width,
            decode_step=0,
        )
        self.context = SimpleNamespace(
            decision=BeamDecisionBuffers(
                max_batch=1,
                max_beams=width,
                max_decode_steps=steps,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )
        self.ends = 0

    @property
    def is_active(self) -> bool:
        return self._state.is_active

    @property
    def beam_width(self) -> int:
        return self._state.beam_width

    @property
    def decode_step(self) -> int:
        return self._state.decode_step

    def end_session(self) -> None:
        self.ends += 1
        self._state.is_active = False


class _KV:
    def __init__(self, session: _Session) -> None:
        self.session = session
        self.steps: list[tuple[int, bool]] = []
        self.releases = 0

    def reorder(self, session, decision, *, expected_device_step: int, is_final: bool) -> None:
        assert session is self.session
        assert decision.device_step == expected_device_step + 1
        self.steps.append((expected_device_step, is_final))
        if not is_final:
            session._state.decode_step += 1

    def release(self, session) -> bool:
        assert session is self.session
        self.releases += 1
        return self.releases == 1


def _output(
    token_ids: torch.Tensor,
    logprobs: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> ConstraintOutput:
    return ConstraintOutput(
        token_ids.to(torch.int32),
        logprobs.to(torch.float32),
        torch.ones_like(token_ids, dtype=torch.bool) if valid_mask is None else valid_mask,
    )


def test_three_stage_composite_lifecycle_and_final_only_payload() -> None:
    width = 4
    steps = 3
    vocab = 32
    prefill_tokens = torch.tensor([[10, 11, 12, 13]])
    prefill_scores = torch.tensor([[-0.1, -0.2, -0.3, -0.4]])
    middle_tokens = torch.arange(100, 116).view(width, width)
    middle_scores = -torch.arange(1, 17).view(width, width) / 10
    final_tokens = torch.arange(200, 216).view(width, width)
    final_scores = -torch.arange(1, 17).view(width, width) / 20
    constraint = _ConstraintBackend(
        [
            _output(prefill_tokens, prefill_scores),
            _output(middle_tokens, middle_scores),
            _output(final_tokens, final_scores),
        ],
        width,
        vocab,
    )
    session = _Session(width, steps)
    updates: list[str] = []
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        constraint_backend=constraint,
        requests={"gr": SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.25))},
        input_batch=SimpleNamespace(update_async_output_token_ids=lambda: updates.append("update")),
        _gr_state_peak_bytes=0,
    )
    kv_objects: list[_KV] = []
    drains: list[torch.device] = []

    def kv_factory(current_session: _Session) -> _KV:
        value = _KV(current_session)
        kv_objects.append(value)
        return value

    def final_copy(pack: NPUFinalPack, _backend_state) -> bytes:
        assert pack.status.tolist() == [0]
        return pack.payload.numpy().tobytes()

    backend = NPUGRDeviceBackend(
        cpu_test_backend=True,
        prefix_binder=lambda _runner, _state: (torch.tensor([7], dtype=torch.int32),),
        kv_factory=kv_factory,
        final_copy=final_copy,
        drain=drains.append,
    )
    state = GRModelRunnerState(
        request_id="gr",
        session=session,
        backend=backend,
        beam_width=width,
        max_tokens=steps,
        continue_token_id=29,
        finish_token_id=30,
    )
    state.start_prefill()

    prefill = backend.sample(
        runner,
        state,
        torch.zeros((1, vocab), dtype=torch.float32),
        {"result_width": 2},
    )
    state.bootstrap_from_prefill(runner, prefill, {"result_width": 2})
    first_control = state.control_token_ids
    assert first_control.tolist() == [[29]]
    assert state.next_token_ids.tolist() == prefill_tokens.tolist()

    for generation_index, is_final in ((1, False), (2, True)):
        metadata = {"generation_index": generation_index, "result_width": 2}
        state.prepare_decode_inputs(runner, object(), [width], metadata)
        sampled = backend.sample(
            runner,
            state,
            torch.zeros((width, vocab), dtype=torch.float32),
            metadata,
        )
        assert state.advance_beam(runner, sampled, metadata) is is_final
        if generation_index == 1:
            second_control = state.control_token_ids
            assert second_control.tolist() == [[29]]
            assert second_control.data_ptr() != first_control.data_ptr()

    assert state.control_token_ids.tolist() == [[30]]
    assert state.control_token_ids.data_ptr() == first_control.data_ptr()
    assert updates == ["update", "update", "update"]
    assert [call[1].current_step for call in constraint.calls] == [0, 1, 2]
    assert len(kv_objects) == 1
    assert kv_objects[0].steps == [(1, False), (2, True)]
    assert session.decode_step == 1

    payload = state.finalize_output(runner)
    decoded = unpack_gr_final_output(payload)
    assert decoded is not None
    sequences, parents, logprobs, scores = decoded
    assert len(sequences) == len(parents) == len(logprobs) == len(scores) == 2
    assert all(len(row) == steps for row in sequences)
    assert payload["is_last_step"] is True
    assert payload["step"] == payload["device_step"] == steps

    assert state.release()
    assert drains == [torch.device("cpu")]
    assert kv_objects[0].releases == 1
    assert session.ends == 1
    assert state.released


def test_reused_common_state_allocates_fresh_backend_ownership() -> None:
    width = 2
    output = _output(
        torch.tensor([[10, 11]]),
        torch.tensor([[-0.1, -0.2]]),
    )
    constraint = _ConstraintBackend([output, output], width, 32)
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        constraint_backend=constraint,
        requests={
            "first": SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.5)),
            "second": SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.5)),
        },
        input_batch=SimpleNamespace(update_async_output_token_ids=lambda: None),
        _gr_state_peak_bytes=0,
    )
    backend = NPUGRDeviceBackend(
        cpu_test_backend=True,
        prefix_binder=lambda _runner, _state: (),
        final_copy=lambda pack, _state: pack.payload.numpy().tobytes(),
        drain=lambda _device: None,
    )
    first_session = _Session(width, 1)
    state = GRModelRunnerState(
        request_id="first",
        session=first_session,
        backend=backend,
        beam_width=width,
        max_tokens=1,
        continue_token_id=29,
        finish_token_id=30,
    )
    state.start_prefill()
    sampled = backend.sample(runner, state, torch.zeros((1, 32)), {"result_width": 2})
    state.bootstrap_from_prefill(runner, sampled, {"result_width": 2})
    old_backend_state = state.backend_state
    assert state.release()
    assert old_backend_state.released

    state.reset_for_request(
        request_id="second",
        session=_Session(width, 1),
        continue_token_id=29,
        finish_token_id=30,
    )
    backend.sample(runner, state, torch.zeros((1, 32)), {"result_width": 2})

    assert state.backend_state is not old_backend_state
    assert not state.backend_state.released
    assert state.backend_state.lease.owner_id == "second"


def test_drain_failure_retains_all_backend_ownership_for_retry() -> None:
    width = 2
    constraint = _ConstraintBackend(
        [_output(torch.tensor([[10, 11]]), torch.tensor([[-0.1, -0.2]]))],
        width,
        32,
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        constraint_backend=constraint,
        requests={"gr": SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.5))},
        input_batch=SimpleNamespace(update_async_output_token_ids=lambda: None),
        _gr_state_peak_bytes=0,
    )
    attempts = 0

    def drain(_device: torch.device) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("device drain failed")

    backend = NPUGRDeviceBackend(
        cpu_test_backend=True,
        prefix_binder=lambda _runner, _state: (),
        final_copy=lambda pack, _state: pack.payload.numpy().tobytes(),
        drain=drain,
    )
    session = _Session(width, 1)
    state = GRModelRunnerState(
        request_id="gr",
        session=session,
        backend=backend,
        beam_width=width,
        max_tokens=1,
        continue_token_id=29,
        finish_token_id=30,
    )
    state.start_prefill()
    sampled = backend.sample(runner, state, torch.zeros((1, 32)), {"result_width": 2})
    state.bootstrap_from_prefill(runner, sampled, {"result_width": 2})
    backend_state = state.backend_state

    with pytest.raises(RuntimeError, match="device drain failed"):
        state.release()
    assert not backend_state.released
    assert backend_state.workspace._lease is backend_state.lease
    assert session.ends == 0

    assert state.release()
    assert backend_state.released
    assert backend_state.workspace._lease is None
    assert session.ends == 1
