# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,type-arg"

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from vllm.logprobs import FlatLogprobs
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_gr.constraints import BeamConstraintCoordinator
from vllm_gr.constraints.strategies import ConstraintTableStrategy
from vllm_gr.entrypoints import gr as offline_gr
from vllm_gr.entrypoints.openai import serving_engine
from vllm_gr.v1.engine.types import (
    GR_REQUEST_METADATA_KEY,
    BeamRequestStepUpdate,
    GRRequestMetadata,
)

_DIGEST = "sha256:" + "a" * 64


class _OutputQueue:
    def __init__(self, output: Any) -> None:
        self.output = output

    async def get(self) -> Any:
        return self.output


class _Tokenizer:
    eos_token_id = 2

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<begin>": 1, "<end>": 3}.get(token, -1)

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def _raw_block(
    token_ids: list[int],
    logprobs: list[float],
) -> tuple[list[int], list[float], list[int | None], list[str | None]]:
    raw_token_ids = [token_ids[0], *token_ids]
    raw_logprobs = [logprobs[0], *logprobs]
    size = len(raw_token_ids)
    return (
        raw_token_ids,
        raw_logprobs,
        list(range(1, size + 1)),
        [str(token_id) for token_id in raw_token_ids],
    )


def _flat(
    token_ids: list[int],
    logprobs: list[float],
) -> FlatLogprobs:
    raw_token_ids, raw_logprobs, ranks, decoded_tokens = _raw_block(
        token_ids,
        logprobs,
    )
    return FlatLogprobs(
        start_indices=[0],
        end_indices=[len(raw_token_ids)],
        token_ids=raw_token_ids,
        logprobs=raw_logprobs,
        ranks=ranks,
        decoded_tokens=decoded_tokens,
    )


def _combined_flat(
    candidates: list[tuple[list[int], list[float]]],
) -> FlatLogprobs:
    raw_token_ids: list[int] = []
    raw_logprobs: list[float] = []
    ranks: list[int | None] = []
    decoded_tokens: list[str | None] = []
    for token_ids, logprobs in candidates:
        block_token_ids, block_logprobs, block_ranks, block_decoded = _raw_block(
            token_ids,
            logprobs,
        )
        raw_token_ids.extend(block_token_ids)
        raw_logprobs.extend(block_logprobs)
        ranks.extend(block_ranks)
        decoded_tokens.extend(block_decoded)
    return FlatLogprobs(
        start_indices=[0],
        end_indices=[len(raw_token_ids)],
        token_ids=raw_token_ids,
        logprobs=raw_logprobs,
        ranks=ranks,
        decoded_tokens=decoded_tokens,
    )


def _engine_output(
    flat: FlatLogprobs | None,
    *,
    finish_reason: str | None = None,
    kv_transfer_params: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                finish_reason=finish_reason,
                logprobs=flat,
            )
        ],
        finished=True,
        kv_transfer_params=kv_transfer_params,
    )


class _OrdinaryEngineClient:
    def __init__(self, prepare_outputs: list[Any]) -> None:
        self.prepare_outputs = list(prepare_outputs)
        self.prepare_calls: list[SimpleNamespace] = []
        self.add_batch_calls: list[SimpleNamespace] = []

    async def prepare_request(
        self,
        request_id: str,
        prompt: Any,
        sampling_params: Any,
        *,
        lora_request: Any,
        trace_headers: Any,
        priority: int,
        data_parallel_rank: int | None,
    ) -> tuple[_OutputQueue, SimpleNamespace]:
        call = SimpleNamespace(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )
        self.prepare_calls.append(call)
        return _OutputQueue(self.prepare_outputs.pop(0)), SimpleNamespace(
            request_id=request_id,
            sampling_params=sampling_params,
        )

    async def _add_requests_batch(
        self,
        requests: list[Any],
        *,
        use_batch_message: bool,
    ) -> None:
        self.add_batch_calls.append(
            SimpleNamespace(
                request_ids=[request.request_id for request in requests],
                use_batch_message=use_batch_message,
            )
        )


class _GroupedEngineClient(_OrdinaryEngineClient):
    def __init__(
        self,
        prepare_outputs: list[Any],
        grouped_outputs: list[Any],
        *,
        data_parallel_size: int | None = None,
        fail_cleanup: bool = False,
    ) -> None:
        super().__init__(prepare_outputs)
        self.grouped_outputs = list(grouped_outputs)
        self.register_calls: list[SimpleNamespace] = []
        self.beam_updates: list[BeamRequestStepUpdate] = []
        self.fail_cleanup = fail_cleanup
        if data_parallel_size is not None:
            self.vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    data_parallel_size=data_parallel_size,
                )
            )

    def register_beam_output(
        self,
        session_id: str,
        tokens: list[int],
        sampling_params: Any,
        **kwargs: Any,
    ) -> _OutputQueue:
        self.register_calls.append(
            SimpleNamespace(
                session_id=session_id,
                tokens=list(tokens),
                sampling_params=sampling_params,
                kwargs=kwargs,
            )
        )
        return _OutputQueue(self.grouped_outputs.pop(0))

    async def beam_request_step_update(
        self,
        update: BeamRequestStepUpdate,
    ) -> None:
        self.beam_updates.append(update)
        if self.fail_cleanup and update.beam_width == 0:
            raise RuntimeError("cleanup failed")


def _serving(engine_client: Any) -> SimpleNamespace:
    return SimpleNamespace(
        engine_client=engine_client,
        input_processor=SimpleNamespace(tokenizer=_Tokenizer()),
        models=SimpleNamespace(catalog=None, model_name="test-model"),
    )


def _params(
    *,
    max_tokens: int,
    ignore_eos: bool = False,
    end_token: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        beam_width=2,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
        temperature=0.0,
        begin_token=None,
        end_token=end_token,
        include_stop_str_in_output=False,
        catalog_strategy="fill",
    )


async def _collect_outputs(
    generator: AsyncGenerator[RequestOutput, None],
) -> list[RequestOutput]:
    return [output async for output in generator]


def _run_search(
    engine_client: Any,
    *,
    max_tokens: int,
    ignore_eos: bool = False,
    request_id: str = "request-0",
    end_token: str | None = None,
) -> list[RequestOutput]:
    return asyncio.run(
        _collect_outputs(
            serving_engine.beam_search(
                _serving(engine_client),
                {
                    "prompt": "prompt text",
                    "prompt_token_ids": [10],
                },
                request_id,
                _params(
                    max_tokens=max_tokens,
                    ignore_eos=ignore_eos,
                    end_token=end_token,
                ),
                trace_headers={"trace": "value"},
            )
        )
    )


def _install_deterministic_request_ids(monkeypatch: Any) -> None:
    ids = iter(("step-0", "step-1", "step-2", "step-3"))
    monkeypatch.setattr(serving_engine, "random_uuid", lambda: next(ids))


def _assert_normal_output(output: RequestOutput, request_id: str) -> None:
    assert output.request_id == request_id
    assert output.prompt == "prompt text"
    assert output.prompt_token_ids == [10]
    assert output.prompt_logprobs is None
    assert output.finished is True
    assert len(output.outputs) == 2
    assert [choice.index for choice in output.outputs] == [0, 1]
    assert all(choice.text for choice in output.outputs)
    assert all(choice.token_ids for choice in output.outputs)
    assert all(choice.cumulative_logprob is not None for choice in output.outputs)
    assert all(isinstance(choice.logprobs, FlatLogprobs) for choice in output.outputs)
    assert [choice.finish_reason for choice in output.outputs] == ["length", "length"]
    assert [choice.stop_reason for choice in output.outputs] == [None, None]
    assert output.metrics is not None
    assert output.metrics.num_beams == 2
    assert output.metrics.num_generation_tokens >= 0
    assert output.metrics.beam_search_prefill_time >= 0
    assert output.metrics.beam_search_decode_time >= 0
    assert output.metrics.beam_search_sort_time >= 0


def _persistent_final_decision(*, width: int = 2, steps: int = 3) -> dict[str, Any]:
    sequences = np.asarray([[11, 21, 31], [12, 22, 32]], dtype=np.int32)
    parents = np.asarray([[0, 0, 0], [0, 1, 1]], dtype=np.int32)
    logprobs = np.asarray(
        [[-0.1, -0.2, -0.3], [-0.2, -0.2, -0.3]], dtype=np.float32
    )
    scores = np.asarray([-0.6, -0.7], dtype=np.float32)
    packed = np.concatenate(
        (
            sequences.reshape(-1),
            parents.reshape(-1),
            logprobs.reshape(-1).view(np.int32),
            scores.view(np.int32),
        )
    ).tobytes()
    return {
        "buffer_format": "gr_final_i32_f32_v1",
        "shape": [width, steps],
        "buffer": packed,
        "is_last_step": True,
        "counters": {"native_request_created_per_session": 1},
    }


def test_offline_persistent_prefill_builds_one_final_only_request() -> None:
    actual_request_id = "offline-beam-0-0-native"
    request_states: dict[str, SimpleNamespace] = {}

    def add_output_request(request, *_args: Any) -> None:
        request_states[request.request_id] = SimpleNamespace()

    output_processor = SimpleNamespace(
        add_request=Mock(side_effect=add_output_request),
        request_states=request_states,
    )

    def process_inputs(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            request_id=kwargs["request_id"],
            sampling_params=kwargs["params"],
        )

    def assign_request_id(request: SimpleNamespace) -> None:
        request.request_id = actual_request_id

    llm_engine = SimpleNamespace(
        input_processor=SimpleNamespace(
            process_inputs=process_inputs,
            assign_request_id=assign_request_id,
        ),
        output_processor=output_processor,
        get_supported_tasks=lambda: ("generate",),
    )
    beam = SimpleNamespace(
        tokens=[10],
        lora_request=None,
        get_prompt=lambda: {"prompt_token_ids": [10]},
    )
    params = SamplingParams(logprobs=2, max_tokens=1)
    instance_prefill_id: dict[int, str] = {}
    instance_prev_beam_internal_ids: dict[int, list[str]] = {}
    instance_fork_info: dict[int, list[tuple[int, int]]] = {}
    instance_prefix_len: dict[int, int] = {}
    instance_completed_sessions: set[int] = set()

    requests = offline_gr._prepare_beam_step_requests(
        llm_engine=llm_engine,
        instances_batch=[SimpleNamespace(beams=[beam])],
        token=0,
        prompt_start=0,
        tokenizer=_Tokenizer(),
        beam_search_params=params,
        sid_end_token_id=3,
        instance_prefill_id=instance_prefill_id,
        instance_prev_beam_internal_ids=instance_prev_beam_internal_ids,
        instance_fork_info=instance_fork_info,
        instance_prefix_len=instance_prefix_len,
        instance_completed_sessions=instance_completed_sessions,
        max_decode_steps=3,
        is_constrained=True,
        constraint_digest="a" * 64,
        persistent_request=True,
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.request_id == actual_request_id
    assert request.sampling_params.output_kind is RequestOutputKind.FINAL_ONLY
    assert request.sampling_params.max_tokens == 3
    assert request.sampling_params.logprobs is None
    assert request.sampling_params.ignore_eos is True
    metadata = request.sampling_params.extra_args[GR_REQUEST_METADATA_KEY]
    assert metadata["request_id"] == actual_request_id
    assert metadata["constraint_digest"] == _DIGEST
    assert set(metadata) == {
        "request_id",
        "beam_width",
        "generation_steps",
        "prefix_len",
        "result_width",
        "constraint_digest",
        "protocol_version",
    }
    output_processor.add_request.assert_called_once_with(request, None, None, 0)
    assert request_states[actual_request_id].gr_persistent_request is True

    decode_requests = offline_gr._prepare_beam_step_requests(
        llm_engine=llm_engine,
        instances_batch=[SimpleNamespace(beams=[beam])],
        token=1,
        prompt_start=0,
        tokenizer=_Tokenizer(),
        beam_search_params=params,
        sid_end_token_id=3,
        instance_prefill_id=instance_prefill_id,
        instance_prev_beam_internal_ids=instance_prev_beam_internal_ids,
        instance_fork_info=instance_fork_info,
        instance_prefix_len=instance_prefix_len,
        instance_completed_sessions=instance_completed_sessions,
        max_decode_steps=3,
        is_constrained=True,
        constraint_digest="a" * 64,
        persistent_request=True,
    )

    assert decode_requests == []
    output_processor.add_request.assert_called_once_with(request, None, None, 0)


def test_offline_persistent_terminated_target_fails_without_empty_step_loop() -> None:
    internal_id = "native-internal-0"
    calls = 0

    def bounded_step() -> list[Any]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("empty step loop exceeded its bound")
        return []

    llm_engine = SimpleNamespace(
        step=bounded_step,
        output_processor=SimpleNamespace(
            # An unrelated ordinary request must not keep this GR wait alive.
            request_states={"ordinary-request": object()}
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "persistent GR request terminated before final output.*"
            "native-internal-0.*offline-beam-4-0"
        ),
    ):
        offline_gr._step_engine_and_collect_outputs(
            llm_engine=llm_engine,
            instances_batch=[SimpleNamespace(beams=[object()])],
            token=0,
            prompt_start=4,
            instance_prefill_id={0: internal_id},
            persistent_request=True,
        )

    assert calls == 1


def test_offline_persistent_final_wins_after_output_registry_removal() -> None:
    final = SimpleNamespace(request_id="offline-beam-4-0")
    llm_engine = SimpleNamespace(
        step=Mock(return_value=[final]),
        # FINAL_ONLY processing may remove the state before step() returns.
        output_processor=SimpleNamespace(request_states={}),
    )

    outputs = offline_gr._step_engine_and_collect_outputs(
        llm_engine=llm_engine,
        instances_batch=[SimpleNamespace(beams=[object()])],
        token=0,
        prompt_start=4,
        instance_prefill_id={0: "native-internal-0"},
        persistent_request=True,
    )

    assert outputs == {final.request_id: final}
    llm_engine.step.assert_called_once_with()


def test_offline_persistent_live_target_keeps_waiting_for_final() -> None:
    internal_id = "native-internal-0"
    request_states = {internal_id: object()}
    final = SimpleNamespace(request_id="offline-beam-4-0")
    calls = 0

    def step_until_final() -> list[Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return []
        if calls == 2:
            request_states.pop(internal_id)
            return [final]
        raise AssertionError("persistent wait exceeded its bound")

    llm_engine = SimpleNamespace(
        step=step_until_final,
        output_processor=SimpleNamespace(request_states=request_states),
    )

    outputs = offline_gr._step_engine_and_collect_outputs(
        llm_engine=llm_engine,
        instances_batch=[SimpleNamespace(beams=[object()])],
        token=0,
        prompt_start=4,
        instance_prefill_id={0: internal_id},
        persistent_request=True,
    )

    assert outputs == {final.request_id: final}
    assert calls == 2


def test_offline_persistent_step_failure_aborts_live_native_request() -> None:
    internal_id = "native-internal-0"
    request_states = {internal_id: object()}
    abort_calls: list[tuple[list[str], bool]] = []

    def abort_request(request_ids: list[str], internal: bool = False) -> None:
        abort_calls.append((list(request_ids), internal))
        for request_id in request_ids:
            request_states.pop(request_id)

    llm_engine = SimpleNamespace(
        step=Mock(side_effect=RuntimeError("step failed")),
        abort_request=abort_request,
        output_processor=SimpleNamespace(request_states=request_states),
    )

    with pytest.raises(RuntimeError, match="step failed"):
        offline_gr._step_engine_and_collect_outputs(
            llm_engine=llm_engine,
            instances_batch=[SimpleNamespace(beams=[object()])],
            token=0,
            prompt_start=0,
            instance_prefill_id={0: internal_id},
            persistent_request=True,
        )

    assert abort_calls == [([internal_id], True)]
    assert request_states == {}


def test_offline_ordinary_output_collection_does_not_inspect_request_registry() -> None:
    class OrdinaryOutputProcessor:
        @property
        def request_states(self):
            raise AssertionError("ordinary path inspected persistent registry")

    final = SimpleNamespace(request_id="offline-beam-4-0")
    steps = iter(([], [final]))
    llm_engine = SimpleNamespace(
        step=lambda: next(steps),
        output_processor=OrdinaryOutputProcessor(),
    )

    outputs = offline_gr._step_engine_and_collect_outputs(
        llm_engine=llm_engine,
        instances_batch=[SimpleNamespace(beams=[object()])],
        token=0,
        prompt_start=4,
        instance_prefill_id={0: "ordinary-internal-0"},
        persistent_request=False,
    )

    assert outputs == {final.request_id: final}


def test_offline_persistent_rejects_multiple_prompts_before_batch_split() -> None:
    fake = SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                additional_config={
                    "vllm_gr_persistent_request": True,
                    "vllm_gr_persistent_runner_state": True,
                    "vllm_gr_scheduling_mode": "async",
                }
            )
        )
    )

    with pytest.raises(ValueError, match="require B=1"):
        offline_gr.GRLLM.beam_search(
            fake,
            [{"prompt_token_ids": [10]}, {"prompt_token_ids": [20]}],
            _params(max_tokens=4),
            concurrency_limit=1,
        )


def test_grouped_beam_request_lifecycle(
    monkeypatch: Any,
) -> None:
    _install_deterministic_request_ids(monkeypatch)
    engine = _GroupedEngineClient(
        prepare_outputs=[_engine_output(_flat([11, 12], [-0.1, -0.2]))],
        grouped_outputs=[
            _engine_output(
                _combined_flat(
                    [
                        ([21, 22], [-0.1, -0.11]),
                        ([31, 32], [-5.0, -6.0]),
                    ]
                )
            ),
            _engine_output(
                _combined_flat(
                    [
                        ([41, 42], [-0.1, -0.4]),
                        ([51, 52], [-0.1, -0.4]),
                    ]
                )
            ),
        ],
        data_parallel_size=2,
    )
    add_step = AsyncMock(wraps=serving_engine._add_batch_step)
    beam_step = AsyncMock(wraps=serving_engine._beam_request_step)
    cleanup = AsyncMock(wraps=serving_engine._beam_request_cleanup)
    rank_picker = AsyncMock(return_value=1)

    monkeypatch.setattr(serving_engine, "_add_batch_step", add_step)
    monkeypatch.setattr(serving_engine, "_beam_request_step", beam_step)
    monkeypatch.setattr(serving_engine, "_beam_request_cleanup", cleanup)
    monkeypatch.setattr(serving_engine, "_next_data_parallel_rank", rank_picker)

    outputs = _run_search(engine, max_tokens=3)

    assert add_step.await_count == 1
    assert beam_step.await_count == 2
    assert cleanup.await_count == 1

    decode_step_1, decode_step_2, cleanup_update = engine.beam_updates
    assert decode_step_1.session_id == "request-0-step-0-beam-0"
    assert decode_step_1.parent_beam_ids == [
        "request-0-step-0-beam-0",
        "request-0-step-0-beam-0",
    ]
    assert decode_step_1.child_beam_ids == [
        "request-0-step-1-beam-0",
        "request-0-step-1-beam-1",
    ]
    assert decode_step_1.beam_tokens == [[11], [12]]
    assert decode_step_1.pruned_ids == []

    assert decode_step_2.parent_beam_ids == [
        "request-0-step-1-beam-0",
        "request-0-step-1-beam-0",
    ]
    assert decode_step_2.child_beam_ids == [
        "request-0-step-2-beam-0",
        "request-0-step-2-beam-1",
    ]
    assert decode_step_2.beam_tokens == [[11, 21], [11, 22]]
    assert decode_step_2.pruned_ids == ["request-0-step-1-beam-1"]

    assert cleanup_update.session_id == "request-0-step-0-beam-0"
    assert cleanup_update.beam_width == 0
    assert cleanup_update.pruned_ids == decode_step_2.child_beam_ids
    assert all(update.data_parallel_rank == 1 for update in engine.beam_updates)
    assert engine.prepare_calls[0].data_parallel_rank == 1
    assert all(call.kwargs["data_parallel_rank"] == 1 for call in engine.register_calls)
    rank_picker.assert_awaited_once_with(2)

    assert len(outputs) == 1
    _assert_normal_output(outputs[0], "request-0")
    assert outputs[0].metrics.num_generation_tokens == 4


def test_grouped_engine_error_yields_error_and_cleans_up(
    monkeypatch: Any,
) -> None:
    _install_deterministic_request_ids(monkeypatch)
    engine = _GroupedEngineClient(
        prepare_outputs=[_engine_output(_flat([11, 12], [-0.1, -0.2]))],
        grouped_outputs=[_engine_output(None, finish_reason="error")],
    )
    outputs = _run_search(engine, max_tokens=3, request_id="error-request")
    assert len(engine.beam_updates) == 2
    decode_update, cleanup_update = engine.beam_updates
    assert decode_update.beam_width == 2
    assert cleanup_update.beam_width == 0
    assert cleanup_update.session_id == "error-request-step-0-beam-0"
    assert cleanup_update.pruned_ids == decode_update.child_beam_ids

    assert len(outputs) == 1
    output = outputs[0]
    assert output.request_id == "error-request"
    assert output.prompt == "prompt text"
    assert output.prompt_token_ids == [10]
    assert output.finished is True
    assert len(output.outputs) == 1
    assert output.outputs[0].finish_reason == "error"
    assert output.outputs[0].text == ""
    assert output.outputs[0].token_ids == []
    assert output.outputs[0].cumulative_logprob is None
    assert output.outputs[0].logprobs is None


def test_eos_and_distinct_sid_end_token_both_complete() -> None:
    engine = _OrdinaryEngineClient(
        prepare_outputs=[_engine_output(_flat([3, 2], [-0.1, -0.2]))],
    )

    outputs = _run_search(
        engine,
        max_tokens=2,
        request_id="multiple-stop-tokens",
        end_token="<end>",
    )

    choices = outputs[0].outputs
    assert [choice.finish_reason for choice in choices] == ["stop", "stop"]
    assert [choice.stop_reason for choice in choices] == [3, 2]
    assert [choice.token_ids for choice in choices] == [[], []]


def test_self_advance_chained_step_skips_frontend_update(
    monkeypatch: Any,
) -> None:
    """EngineCore engine-driven coordination.

    When a decode step's worker decision carries
    ``beam_self_advance_chained=True``, EngineCore already chained the next
    decode request under the same session id. The next loop iteration must
    NOT send its own beam_request_step_update (the session request is still
    running — a second ADD would trip the scheduler's "duplicate request
    id" assert), while still registering the output queue and awaiting the
    chained step's result."""
    _install_deterministic_request_ids(monkeypatch)
    decision = {
        "out_token_ids": [[21, 31]],
        "out_token_index": [[0, 1]],
        "out_log_probs": [[-0.3, -0.7]],
        "beam_self_advance_chained": True,
    }
    engine = _GroupedEngineClient(
        prepare_outputs=[_engine_output(_flat([11, 12], [-0.1, -0.2]))],
        grouped_outputs=[
            # Step 1 (frontend-driven): worker decision chained the next
            # step in-process.
            _engine_output(None, kv_transfer_params={"beam_search_result": decision}),
            # Step 2 (chained by EngineCore): plain flat logprobs, no
            # worker decision — the flag must reset for the final step.
            _engine_output(
                _combined_flat(
                    [
                        ([41, 42], [-0.1, -0.4]),
                        ([51, 52], [-0.1, -0.4]),
                    ]
                )
            ),
        ],
    )

    outputs = _run_search(engine, max_tokens=3)

    # Only the step-1 decode update and the final cleanup were sent; the
    # step-2 update was suppressed because that step was engine-chained.
    assert len(engine.beam_updates) == 2
    decode_update, cleanup_update = engine.beam_updates
    assert decode_update.session_id == "request-0-step-0-beam-0"
    assert decode_update.beam_width == 2
    assert decode_update.beam_tokens == [[11], [12]]
    assert cleanup_update.session_id == "request-0-step-0-beam-0"
    assert cleanup_update.beam_width == 0
    assert cleanup_update.pruned_ids == [
        "request-0-step-2-beam-0",
        "request-0-step-2-beam-1",
    ]

    # Both decode iterations still registered their output queue and
    # awaited the engine result.
    assert len(engine.register_calls) == 2
    assert all(call.session_id == "request-0-step-0-beam-0" for call in engine.register_calls)

    assert len(outputs) == 1
    _assert_normal_output(outputs[0], "request-0")


def test_self_advance_unflagged_decision_keeps_frontend_driven(
    monkeypatch: Any,
) -> None:
    """Without the chained flag (engine-driven disabled or a skip path in
    EngineCore), the frontend keeps driving every step with its own
    beam_request_step_update."""
    _install_deterministic_request_ids(monkeypatch)
    decision = {
        "out_token_ids": [[21, 31]],
        "out_token_index": [[0, 1]],
        "out_log_probs": [[-0.3, -0.7]],
    }
    engine = _GroupedEngineClient(
        prepare_outputs=[_engine_output(_flat([11, 12], [-0.1, -0.2]))],
        grouped_outputs=[
            _engine_output(None, kv_transfer_params={"beam_search_result": decision}),
            _engine_output(
                _combined_flat(
                    [
                        ([41, 42], [-0.1, -0.4]),
                        ([51, 52], [-0.1, -0.4]),
                    ]
                )
            ),
        ],
    )

    outputs = _run_search(engine, max_tokens=3)

    # Both decode steps sent their frontend-driven updates.
    decode_step_1, decode_step_2, _cleanup = engine.beam_updates
    assert decode_step_1.child_beam_ids == [
        "request-0-step-1-beam-0",
        "request-0-step-1-beam-1",
    ]
    assert decode_step_2.child_beam_ids == [
        "request-0-step-2-beam-0",
        "request-0-step-2-beam-1",
    ]
    assert len(engine.register_calls) == 2

    assert len(outputs) == 1
    _assert_normal_output(outputs[0], "request-0")


def test_persistent_runner_state_uses_one_native_request(
    monkeypatch: Any,
) -> None:
    _install_deterministic_request_ids(monkeypatch)
    decision = _persistent_final_decision()
    engine = _GroupedEngineClient(
        prepare_outputs=[
            _engine_output(
                None,
                kv_transfer_params={"beam_search_result": decision},
            )
        ],
        grouped_outputs=[],
    )
    engine.vllm_config = SimpleNamespace(
        additional_config={
            "vllm_gr_persistent_request": True,
            "vllm_gr_persistent_runner_state": True,
        },
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    serving = _serving(engine)
    strategy = ConstraintTableStrategy("test-triples.json")
    strategy.digest = "a" * 64
    serving.models.constraint_coordinator = BeamConstraintCoordinator(strategy)
    params = _params(max_tokens=4, ignore_eos=True, end_token="<end>")
    params.catalog_strategy = None

    outputs = asyncio.run(
        _collect_outputs(
            serving_engine.beam_search(
                serving,
                {"prompt": "prompt text", "prompt_token_ids": [10]},
                "request-0",
                params,
            )
        )
    )

    hard_counters = {
        "native_request_created_per_session": len(engine.prepare_calls),
        "scheduler_add_request_per_session": sum(
            len(call.request_ids) for call in engine.add_batch_calls
        ),
        "beam_request_step_update_calls": len(engine.beam_updates),
        "decode_request_objects_created": max(0, len(engine.prepare_calls) - 1),
    }
    assert hard_counters == {
        "native_request_created_per_session": 1,
        "scheduler_add_request_per_session": 1,
        "beam_request_step_update_calls": 0,
        "decode_request_objects_created": 0,
    }
    sampling_params = engine.prepare_calls[0].sampling_params
    assert sampling_params.max_tokens == 3
    assert sampling_params.logprobs is None
    assert sampling_params.output_kind is RequestOutputKind.FINAL_ONLY
    assert sampling_params.ignore_eos is True
    metadata = sampling_params.extra_args[GR_REQUEST_METADATA_KEY]
    assert metadata["request_id"] == (
        "request-0-step-0-beam-0"
    )
    assert metadata["constraint_digest"] == _DIGEST
    assert [choice.token_ids for choice in outputs[0].outputs] == [
        [11, 21, 31, 3],
        [12, 22, 32, 3],
    ]
    assert [choice.cumulative_logprob for choice in outputs[0].outputs] == [
        pytest.approx(-0.6),
        pytest.approx(-0.7),
    ]


@pytest.mark.asyncio
async def test_persistent_online_collection_failure_aborts_and_marks_frontend_state() -> None:
    class FailingQueue:
        async def get(self):
            raise RuntimeError("collector failed")

    request_state = SimpleNamespace()
    abort_calls: list[tuple[list[str], bool]] = []

    class Client:
        output_processor = SimpleNamespace(request_states={"native": request_state})

        async def prepare_request(self, *_args, **_kwargs):
            params = _args[2]
            return FailingQueue(), SimpleNamespace(
                request_id="native",
                sampling_params=params,
            )

        async def _add_requests_batch(self, requests, *, use_batch_message):
            assert [request.request_id for request in requests] == ["native"]
            assert use_batch_message is False

        async def abort(self, request_ids, internal=False):
            abort_calls.append((list(request_ids), internal))
            self.output_processor.request_states.pop("native")

    params = SamplingParams(logprobs=2, max_tokens=1)
    metadata = GRRequestMetadata(
        request_id="external",
        beam_width=2,
        generation_steps=3,
        prefix_len=1,
        result_width=2,
        constraint_digest=_DIGEST,
    )

    with pytest.raises(RuntimeError, match="collector failed"):
        await serving_engine._add_batch_step(
            Client(),
            ({"prompt_token_ids": [10]},),
            (None,),
            "external-step",
            params,
            False,
            None,
            gr_metadata=metadata,
        )

    assert request_state.gr_persistent_request is True
    assert abort_calls == [(["native"], True)]


@pytest.mark.asyncio
async def test_persistent_online_metadata_failure_aborts_prepared_request() -> None:
    class BrokenSamplingParams:
        @property
        def extra_args(self):
            return None

        @extra_args.setter
        def extra_args(self, _value):
            raise RuntimeError("metadata failed")

    abort_calls: list[tuple[list[str], bool]] = []

    class Client:
        async def prepare_request(self, *_args, **_kwargs):
            return _OutputQueue(None), SimpleNamespace(
                request_id="native",
                sampling_params=BrokenSamplingParams(),
            )

        async def abort(self, request_ids, internal=False):
            abort_calls.append((list(request_ids), internal))

    metadata = GRRequestMetadata(
        request_id="external",
        beam_width=2,
        generation_steps=3,
        prefix_len=1,
        result_width=2,
        constraint_digest=_DIGEST,
    )

    with pytest.raises(RuntimeError, match="metadata failed"):
        await serving_engine._add_batch_step(
            Client(),
            ({"prompt_token_ids": [10]},),
            (None,),
            "external-step",
            SamplingParams(logprobs=2, max_tokens=1),
            False,
            None,
            gr_metadata=metadata,
        )

    assert abort_calls == [(["native"], True)]


@pytest.mark.asyncio
async def test_persistent_online_abort_failure_preserves_original_context() -> None:
    class Client:
        async def abort(self, request_ids, internal=False):
            assert request_ids == ["native"]
            assert internal is True
            raise RuntimeError("abort failed")

    original = ValueError("collection failed")
    with pytest.raises(RuntimeError, match="abort failed") as caught:
        await serving_engine._abort_persistent_requests(
            Client(), ["native"], original
        )

    assert caught.value.__cause__ is original
    assert caught.value.__notes__ == [
        "Persistent GR abort failed after ValueError: collection failed"
    ]


def test_persistent_openai_does_not_require_legacy_step_update_capability(
    monkeypatch: Any,
) -> None:
    _install_deterministic_request_ids(monkeypatch)
    engine = _OrdinaryEngineClient(
        prepare_outputs=[
            _engine_output(
                None,
                kv_transfer_params={
                    "beam_search_result": _persistent_final_decision()
                },
            )
        ]
    )
    engine.vllm_config = SimpleNamespace(
        additional_config={
            "vllm_gr_persistent_request": True,
            "vllm_gr_persistent_runner_state": True,
        },
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    serving = _serving(engine)
    strategy = ConstraintTableStrategy("test-triples.json")
    strategy.digest = "a" * 64
    serving.models.constraint_coordinator = BeamConstraintCoordinator(strategy)
    params = _params(max_tokens=4, ignore_eos=True, end_token="<end>")
    params.catalog_strategy = None

    outputs = asyncio.run(
        _collect_outputs(
            serving_engine.beam_search(
                serving,
                {"prompt": "prompt text", "prompt_token_ids": [10]},
                "request-0",
                params,
            )
        )
    )

    assert not hasattr(engine, "beam_request_step_update")
    assert len(engine.prepare_calls) == 1
    assert sum(len(call.request_ids) for call in engine.add_batch_calls) == 1
    sampling_params = engine.prepare_calls[0].sampling_params
    assert sampling_params.output_kind is RequestOutputKind.FINAL_ONLY
    assert sampling_params.logprobs is None
    assert len(outputs) == 1
