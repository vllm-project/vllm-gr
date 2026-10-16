# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the NPU full-vocabulary adapter on CPU without vendor operators."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.worker import model_runner_common
from vllm_gr.v1.worker import npu_beam_stage_runner as npu_runner
from vllm_gr.v1.worker.npu_beam_stage_adapter import NPUEagerStagePlan


def _scheduled(stage_index: int, request: Any) -> SimpleNamespace:
    ids = tuple(item.request.request_id for item in request.items)
    lengths = tuple(len(item.request.prompt_token_ids) for item in request.items)
    metadata = SimpleNamespace(
        session_id=request.batch_id,
        dispatch_id=stage_index,
        stage_index=stage_index,
        kind="decode" if stage_index else "prefill",
        decode_step=stage_index - 1 if stage_index else None,
        owner_generation=1,
        request_ids=ids,
        num_prompt_tokens_by_request=lengths,
        num_prompt_tokens=sum(lengths),
        beam_params=request.beam_params,
        produces_output=True,
        execution_capacity=SimpleNamespace(beam_width=request.beam_params.beam_width),
        is_last_stage=stage_index == request.beam_params.max_tokens - 1,
        output_device_state=object(),
    )
    counts = (1,) * len(ids) if stage_index else lengths
    return SimpleNamespace(
        gr_stage_metadata={request.batch_id: metadata},
        num_scheduled_tokens=dict(zip(ids, counts)),
        total_num_scheduled_tokens=sum(counts),
        beam_data={},
    )


@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize("use_graph_context", [False, True])
def test_full_vocab_npu_adapter_preserves_early_eos_and_batch_identity(
    monkeypatch: pytest.MonkeyPatch, steps: int, use_graph_context: bool
) -> None:
    monkeypatch.setattr(npu_runner, "validate_npu_eager_capabilities", lambda worker: None)
    monkeypatch.setattr(model_runner_common, "_get_gr_config", lambda worker: object())
    context = BeamAttentionContext(2, 1, 1, 3, 2, 1, torch.device("cpu"), torch.float16)
    monkeypatch.setattr(
        model_runner_common, "_create_beam_context", lambda *args, **kwargs: context
    )
    request = SimpleNamespace(
        protocol_version=2,
        batch_id="batch-full",
        items=tuple(
            SimpleNamespace(
                item_index=index,
                request=SimpleNamespace(
                    request_id=f"item-{index}",
                    prompt_token_ids=[7] * (index + 2),
                    sampling_params=SimpleNamespace(eos_token_id=0),
                ),
            )
            for index in range(2)
        ),
        beam_params=SimpleNamespace(
            candidate_mode="full_vocab",
            beam_width=1,
            max_tokens=steps,
            stop_token_ids=(0,),
            ignore_stop_tokens=False,
            length_penalty=1.0,
            temperature=0.0,
        ),
        output_options=SimpleNamespace(
            num_return_sequences=1,
            include_stop_token=False,
            include_cumulative_logprob=True,
            include_parent_beam_indices=True,
        ),
    )

    class ForbiddenBackend:
        def select(self, *args: object) -> None:
            raise AssertionError("full-vocabulary inference must not consult the constraint table")

    worker = SimpleNamespace(
        device=torch.device("cpu"),
        model_config=SimpleNamespace(get_vocab_size=lambda: 4),
        constraint_backend=ForbiddenBackend(),
        _gr_npu_beam_graph_shapes={(2, 1)} if use_graph_context else set(),
        _gr_npu_graph_context=context if use_graph_context else None,
    )
    runner = npu_runner.NPUEagerBeamStageRunner(worker)
    runner.session_request = request
    rows = [
        torch.tensor([[0.0, -torch.inf, -torch.inf, -torch.inf], [-5.0, 5.0, 4.0, 3.0]]),
        torch.tensor([[1.0, 9.0, 1.0, 0.0], [-5.0, -2.0, 5.0, 4.0]]),
        torch.tensor([[1.0, 9.0, 1.0, 0.0], [-5.0, -2.0, 1.0, 5.0]]),
    ]
    addresses = (
        context.pools[0].unshared_key.data_ptr(),
        context.pools[0].unshared_value.data_ptr(),
    )
    for index in range(steps):
        scheduled = _scheduled(index, request)
        worker._current_scheduler_output = scheduled
        plan = NPUEagerStagePlan.from_scheduler_output(scheduled)
        runner.pending_plan = plan
        if index:
            runner._ensure_attention_batch(plan)
            # Emulate forward KV writes into the request-local physical rows.
            context.pools[0].unshared_key[:, :, :, index - 1].fill_(index)
        runner.logits = rows[index]
        output = runner.sample()
        receipt = output.gr_worker_results[request.batch_id]
        assert receipt.finished == (index == steps - 1)
        if index < steps - 1:
            assert not runner.full_vocab_state.valid[0].any()
            assert receipt.terminal_result is None
    result = receipt.terminal_result
    assert result.protocol_version == 2 and result.failure is None
    assert [item.request_id for item in result.output.items] == ["item-0", "item-1"]
    first = result.output.items[0].sequences[0]
    assert first.token_ids == ()
    # Parents describe every generation, even when EOS is hidden from tokens.
    assert first.parent_beam_indices == (0,)
    assert first.cumulative_logprob == pytest.approx(0.0)
    assert (first.finish_reason, first.stop_token_id) == ("stop", 0)
    second = result.output.items[1].sequences[0]
    assert second.token_ids == tuple(range(1, steps + 1))
    expected_score = sum(rows[index][1].log_softmax(0)[index + 1].item() for index in range(steps))
    assert second.cumulative_logprob == pytest.approx(expected_score, abs=1e-6)
    assert second.parent_beam_indices == (0,) * steps
    assert second.finish_reason == "length"
    assert runner.closed and receipt.resources_released
    assert context.registry.num_active_sessions == 0
    assert addresses == (
        context.pools[0].unshared_key.data_ptr(),
        context.pools[0].unshared_value.data_ptr(),
    )


def test_full_vocab_npu_adapter_rejects_a_sliced_projection() -> None:
    runner = object.__new__(npu_runner.NPUEagerBeamStageRunner)
    runner.worker = SimpleNamespace(model_config=SimpleNamespace(get_vocab_size=lambda: 8192))
    runner.logits = torch.zeros(1, 512)
    with pytest.raises(RuntimeError, match="received 512 columns, expected 8192"):
        runner._sample_full_vocab(
            SimpleNamespace(execution=SimpleNamespace(beam_width=1)),
            SimpleNamespace(beam_params=object()),
        )


@pytest.mark.parametrize(
    "engine_mode,request_mode",
    [("full_vocab", "constraint_table"), ("constraint_table", "full_vocab")],
)
def test_npu_candidate_mode_mismatch_is_rejected_before_owner_creation(
    monkeypatch: pytest.MonkeyPatch, engine_mode: str, request_mode: str
) -> None:
    from vllm_gr.config import BeamExecutionMode, GRConfig

    monkeypatch.setattr(
        GRConfig,
        "from_vllm_config",
        lambda config: SimpleNamespace(
            beam_candidate_mode=engine_mode, beam_execution_mode=BeamExecutionMode.V1
        ),
    )
    worker = SimpleNamespace(vllm_config=object())
    scheduled = SimpleNamespace(
        gr_stage_metadata={
            "session": SimpleNamespace(beam_params=SimpleNamespace(candidate_mode=request_mode))
        }
    )
    with pytest.raises(ValueError, match="does not match engine candidate_mode"):
        npu_runner.get_npu_stage_runner(worker, scheduled)
    assert not hasattr(worker, "_gr_npu_stage_runner")


@pytest.mark.parametrize("mismatch", ["params", "protocol"])
def test_npu_session_init_must_agree_with_dispatch(mismatch: str) -> None:
    params = SimpleNamespace(candidate_mode="full_vocab")
    session_init = SimpleNamespace(
        beam_params=SimpleNamespace(candidate_mode="constraint_table")
        if mismatch == "params"
        else params,
        protocol_version=1,
    )
    metadata = SimpleNamespace(beam_params=params, session_init=session_init)
    with pytest.raises(ValueError, match="session_init.*disagree"):
        npu_runner._validate_candidate_mode(SimpleNamespace(), metadata)


def test_npu_dispatch_cannot_change_an_existing_sessions_beam_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(npu_runner, "validate_npu_eager_capabilities", lambda worker: None)
    request = SimpleNamespace(
        batch_id="batch",
        items=(SimpleNamespace(request=SimpleNamespace(request_id="batch", prompt_token_ids=[1])),),
        beam_params=SimpleNamespace(candidate_mode="full_vocab", beam_width=1, max_tokens=3),
    )
    runner = npu_runner.NPUEagerBeamStageRunner(SimpleNamespace())
    runner.session_request = SimpleNamespace(
        beam_params=SimpleNamespace(candidate_mode="full_vocab", beam_width=1, max_tokens=2)
    )
    with pytest.raises(ValueError, match="parameters changed during a Batch Session"):
        runner.execute(_scheduled(0, request), lambda *args: pytest.fail("model executed"))
    assert runner.pending_plan is None
