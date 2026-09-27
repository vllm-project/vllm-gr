# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,type-arg"

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from vllm.logprobs import FlatLogprobs
from vllm.outputs import RequestOutput

from vllm_gr.entrypoints.openai import serving_engine
from vllm_gr.v1.engine.types import BeamRequestStepUpdate


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
        return _OutputQueue(self.prepare_outputs.pop(0)), SimpleNamespace(request_id=request_id)

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
    step_beam_widths: list[int] | None = None,
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
        step_beam_widths=step_beam_widths,
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
    step_beam_widths: list[int] | None = None,
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
                    step_beam_widths=step_beam_widths,
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
    # Cleanup is piggybacked on the final decode step via is_last_step=True
    assert cleanup.await_count == 0

    decode_step_1, decode_step_2 = engine.beam_updates
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
    assert decode_step_1.is_last_step is False

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
    assert decode_step_2.is_last_step is True
    assert all(update.data_parallel_rank == 1 for update in engine.beam_updates)
    assert engine.prepare_calls[0].data_parallel_rank == 1
    assert all(call.kwargs["data_parallel_rank"] == 1 for call in engine.register_calls)
    rank_picker.assert_awaited_once_with(engine)

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


def test_eos_completion_does_not_reenter_active_beams_with_fabricated_score() -> None:
    engine = _OrdinaryEngineClient(
        prepare_outputs=[_engine_output(_flat([11, 2], [-0.2, -2.0]))],
    )

    outputs = _run_search(
        engine,
        max_tokens=1,
        request_id="mixed-active-and-eos",
    )

    choices = outputs[0].outputs
    assert [choice.token_ids for choice in choices] == [[11], []]
    assert [choice.cumulative_logprob for choice in choices] == [-0.2, -2.0]
    assert [choice.finish_reason for choice in choices] == ["length", "stop"]
    assert [choice.stop_reason for choice in choices] == [None, 2]


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
    assert len(engine.beam_updates) == 2
    assert engine.beam_updates[-1].is_last_step is True


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

    _run_search(engine, max_tokens=3)

    # Both decode steps sent their frontend-driven updates; cleanup is
    # piggybacked on the final decode step via is_last_step=True.
    decode_step_1, decode_step_2 = engine.beam_updates
    assert decode_step_1.child_beam_ids == [
        "request-0-step-1-beam-0",
        "request-0-step-1-beam-1",
    ]
    assert decode_step_2.child_beam_ids == [
        "request-0-step-2-beam-0",
        "request-0-step-2-beam-1",
    ]
    assert len(engine.register_calls) == 2
    assert decode_step_2.is_last_step is True
    assert len(engine.beam_updates) == 2


def test_pruned_worker_beam_lifecycle(
    monkeypatch: Any,
) -> None:
    """Verify pruned worker outputs (top-W candidates with parent indices in ranks)."""
    _install_deterministic_request_ids(monkeypatch)

    def _pruned_flat(
        token_ids: list[int],
        logprobs: list[float],
        parent_ranks: list[int],
    ) -> FlatLogprobs:
        return FlatLogprobs(
            start_indices=[0],
            end_indices=[len(token_ids)],
            token_ids=token_ids,
            logprobs=logprobs,
            ranks=parent_ranks,
            decoded_tokens=[str(t) for t in token_ids],
        )

    # Beam width 2. Step 0 (prepare) outputs tokens 11, 12.
    # Step 1: Worker prunes to top 2: token 21 (parent 0, lp -0.1), token 22 (parent 1, lp -0.2)
    # Step 2: Worker prunes to top 2: token 31 (parent 0, lp -0.1), token 32 (parent 0, lp -0.3)
    engine = _GroupedEngineClient(
        prepare_outputs=[_engine_output(_flat([11, 12], [-0.1, -0.2]))],
        grouped_outputs=[
            _engine_output(_pruned_flat([21, 22], [-0.1, -0.2], [0, 1])),
            _engine_output(_pruned_flat([31, 32], [-0.1, -0.3], [0, 0])),
        ],
        data_parallel_size=1,
    )
    outputs = _run_search(engine, max_tokens=3)
    assert len(outputs) == 1
    output = outputs[0]
    assert output.finished is True
    assert len(output.outputs) == 2
    decode_step_1, decode_step_2 = engine.beam_updates
    assert decode_step_1.is_last_step is False
    assert decode_step_2.is_last_step is True


def test_p2d_dead_end_beam_fallback() -> None:
    """Catalog dead-end fallback emits one finished completion, never empty."""
    initial_logprobs = FlatLogprobs(
        start_indices=[0],
        end_indices=[1],
        token_ids=[7],
        logprobs=[-0.1],
        ranks=[1],
        decoded_tokens=["<s_a_1>"],
    )
    beam = serving_engine._p2d_dead_end_beam(
        orig_prompt=None,
        initial_tokens=[7],
        initial_logprobs=initial_logprobs,
        lora_request=None,
        eos_token_id=2,
    )
    assert beam is not None
    assert beam.tokens == [7, 2]
    assert beam.cum_logprob == float("-inf")
    assert beam.logprobs.token_ids == [7, 2]

    # No EOS token configured: falls back to 0.
    beam_zero = serving_engine._p2d_dead_end_beam(
        orig_prompt=None,
        initial_tokens=[7],
        initial_logprobs=initial_logprobs,
        lora_request=None,
        eos_token_id=None,
    )
    assert beam_zero.tokens == [7, 0]


def test_p2d_sentinel_exit_purges_beam_session(
    monkeypatch: Any,
) -> None:
    """The P2D fast-path return must send the beam_width=0 teardown."""
    from vllm_gr.v1.worker.prefill_graph_runner import P2D_SENTINEL_RANK

    _install_deterministic_request_ids(monkeypatch)

    def _p2d_flat() -> FlatLogprobs:
        # Two captured beams, three tokens each, keyed by the P2D sentinel.
        token_ids = [101, 201, 301, 102, 202, 302]
        return FlatLogprobs(
            start_indices=[0],
            end_indices=[len(token_ids)],
            token_ids=token_ids,
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
            ranks=[P2D_SENTINEL_RANK] * len(token_ids),
            decoded_tokens=[str(t) for t in token_ids],
        )

    engine = _GroupedEngineClient(
        prepare_outputs=[_engine_output(_p2d_flat())],
        grouped_outputs=[],
        data_parallel_size=1,
    )
    p2d_num_steps = 1 + serving_engine.DEFAULT_DECODE_STEPS
    outputs = _run_search(engine, max_tokens=p2d_num_steps)
    assert len(outputs) == 1
    output = outputs[0]
    assert output.finished is True
    assert len(output.outputs) == 2
    assert all(choice.token_ids for choice in output.outputs)
    # The finally-block teardown must purge the session exactly once.
    teardown_updates = [u for u in engine.beam_updates if u.beam_width == 0]
    assert len(teardown_updates) == 1
    assert teardown_updates[0].session_id == "request-0-step-0-beam-0"


def test_p2d_prefill_uses_w2_effective_with_step_widths(
    monkeypatch: Any,
) -> None:
    """Per-step widths: the prefill branch must return w2 (not beam_width) beams."""
    from vllm_gr.v1.worker.prefill_graph_runner import P2D_SENTINEL_RANK

    _install_deterministic_request_ids(monkeypatch)

    def _p2d_flat_w4() -> FlatLogprobs:
        # Four captured beams (w2), three tokens each, sentinel-keyed.
        token_ids = [
            101,
            201,
            301,
            102,
            202,
            302,
            103,
            203,
            303,
            104,
            204,
            304,
        ]
        return FlatLogprobs(
            start_indices=[0],
            end_indices=[len(token_ids)],
            token_ids=token_ids,
            logprobs=[-0.1 * i for i in range(1, len(token_ids) + 1)],
            ranks=[P2D_SENTINEL_RANK] * len(token_ids),
            decoded_tokens=[str(t) for t in token_ids],
        )

    engine = _GroupedEngineClient(
        prepare_outputs=[_engine_output(_p2d_flat_w4())],
        grouped_outputs=[],
        data_parallel_size=1,
    )
    p2d_num_steps = 1 + serving_engine.DEFAULT_DECODE_STEPS
    outputs = _run_search(
        engine,
        max_tokens=p2d_num_steps,
        step_beam_widths=[2, 3, 4],
    )
    assert len(outputs) == 1
    output = outputs[0]
    assert output.finished is True
    # beam_width=2 must NOT truncate the four w2=4 beams.
    assert len(output.outputs) == 4
    assert output.metrics.num_beams == 4
    assert [choice.token_ids[:3] for choice in output.outputs] == [
        [101, 201, 301],
        [102, 202, 302],
        [103, 203, 303],
        [104, 204, 304],
    ]


def test_build_p2d_request_output_direct() -> None:
    """Direct helper: beam count, metrics formula, and unwrap of nested flat."""
    initial_logprobs = FlatLogprobs(
        start_indices=[0],
        end_indices=[1],
        token_ids=[10],
        logprobs=[-0.1],
        ranks=[1],
        decoded_tokens=["prompt"],
    )
    token_ids = [101, 201, 301, 102, 202, 302, 103, 203, 303, 104, 204, 304]
    raw_logprobs = [-0.01 * i for i in range(1, len(token_ids) + 1)]

    # Uniform: max_step_w == w2_effective == 4.
    out = serving_engine._build_p2d_request_output(
        request_id="r1",
        prompt_text="prompt text",
        prompt_token_ids=[10],
        tokenized_length=1,
        raw_token_ids=[token_ids],  # nested list exercises the unwrap
        raw_logprobs=[raw_logprobs],
        initial_tokens=[10],
        initial_logprobs=initial_logprobs,
        orig_prompt="prompt text",
        tokenizer=_Tokenizer(),
        model_name="test-model",
        w2_effective=4,
        max_step_w=4,
    )
    assert len(out.outputs) == 4
    assert out.metrics.num_beams == 4
    assert out.metrics.is_p2d is True
    assert [c.token_ids[:3] for c in out.outputs] == [
        [101, 201, 301],
        [102, 202, 302],
        [103, 203, 303],
        [104, 204, 304],
    ]
    # The flat carries true per-token scores; the cumulative is their sum.
    assert out.outputs[0].cumulative_logprob == pytest.approx(-0.01 - 0.02 - 0.03)
    assert out.outputs[0].logprobs.logprobs[-3:] == pytest.approx([-0.01, -0.02, -0.03])


def test_build_p2d_request_output_dead_end_and_model_name(
    monkeypatch: Any,
) -> None:
    """All-invalid flat falls back to one finished beam; model name passes through."""
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        serving_engine,
        "record_beam_search_metrics",
        lambda metrics, engine, model_name: recorded.append((engine, model_name)),
    )
    initial_logprobs = FlatLogprobs(
        start_indices=[0],
        end_indices=[1],
        token_ids=[10],
        logprobs=[-0.1],
        ranks=[1],
        decoded_tokens=["prompt"],
    )
    out = serving_engine._build_p2d_request_output(
        request_id="r1",
        prompt_text="prompt text",
        prompt_token_ids=[10],
        tokenized_length=1,
        raw_token_ids=[-1] * 12,
        raw_logprobs=[-1.0] * 12,
        initial_tokens=[10],
        initial_logprobs=initial_logprobs,
        orig_prompt="prompt text",
        tokenizer=_Tokenizer(),
        model_name="test-model",
        w2_effective=4,
        max_step_w=4,
        eos_token_id=2,
    )
    assert len(out.outputs) == 1
    assert out.outputs[0].token_ids == [2]
    assert recorded == [("0", "test-model")]


def test_build_p2d_request_output_early_eos_logprob_pruning() -> None:
    """EOS at step 1: cumulative logprob sums only the surviving step scores."""
    initial_logprobs = FlatLogprobs(
        start_indices=[0],
        end_indices=[1],
        token_ids=[10],
        logprobs=[-0.1],
        ranks=[1],
        decoded_tokens=["prompt"],
    )
    token_ids = [101, 2, 301, 102, 202, 302, 103, 203, 303, 104, 204, 304]
    raw_logprobs = [-0.01 * i for i in range(1, len(token_ids) + 1)]
    out = serving_engine._build_p2d_request_output(
        request_id="r1",
        prompt_text="prompt text",
        prompt_token_ids=[10],
        tokenized_length=1,
        raw_token_ids=[token_ids],
        raw_logprobs=[raw_logprobs],
        initial_tokens=[10],
        initial_logprobs=initial_logprobs,
        orig_prompt="prompt text",
        tokenizer=_Tokenizer(),
        model_name="test-model",
        w2_effective=4,
        max_step_w=4,
        eos_token_id=2,
    )
    # Beam 0 hits EOS at step 1: [101, 2] survives, -0.03 is pruned.
    first = out.outputs[0]
    assert first.token_ids == [101, 2]
    assert first.cumulative_logprob == pytest.approx(-0.01 + -0.02)
    assert first.finish_reason == "stop"
    assert first.stop_reason == 2
    assert len(first.logprobs.token_ids) == 3  # base [10] + [101, 2]
    # Beam 1 has no EOS: full three-step sum, no stop reason.
    second = out.outputs[1]
    assert second.cumulative_logprob == pytest.approx(-0.04 + -0.05 + -0.06)
    assert second.finish_reason == "length"


def test_build_p2d_request_output_dead_end_decodes_empty_text(
    monkeypatch: Any,
) -> None:
    """Dead-end fallback beam decodes to "" — never the leading sid token."""
    monkeypatch.setattr(
        serving_engine,
        "record_beam_search_metrics",
        lambda metrics, engine, model_name: None,
    )
    initial_logprobs = FlatLogprobs(
        start_indices=[0],
        end_indices=[2],
        token_ids=[10, 1],
        logprobs=[-0.1, -0.2],
        ranks=[1, 1],
        decoded_tokens=["prompt", "<s_a_1>"],
    )
    out = serving_engine._build_p2d_request_output(
        request_id="r1",
        prompt_text="prompt text",
        prompt_token_ids=[10],
        tokenized_length=1,
        raw_token_ids=[-1] * 12,
        raw_logprobs=[-1.0] * 12,
        initial_tokens=[10, 1],
        initial_logprobs=initial_logprobs,
        orig_prompt="prompt text",
        tokenizer=_Tokenizer(),
        model_name="test-model",
        w2_effective=4,
        max_step_w=4,
        eos_token_id=2,
    )
    assert len(out.outputs) == 1
    assert out.outputs[0].text == ""
    assert out.outputs[0].cumulative_logprob == float("-inf")


def _p2d_sentinel_flat(
    token_ids: list[int],
    logprobs: list[float],
    *,
    nested: bool = False,
) -> FlatLogprobs:
    """Build a P2D worker flat keyed by the sentinel rank.

    The decode (beam-request) transport delivers one flat list holding
    ``num_beams * p2d_num_steps`` tokens, while the prefill/batch transport
    delivers one nested row per request; ``nested`` selects the latter.
    """
    if nested:
        return FlatLogprobs(
            start_indices=[0],
            end_indices=[len(token_ids)],
            token_ids=[list(token_ids)],
            logprobs=[list(logprobs)],
            ranks=[[serving_engine.P2D_SENTINEL_RANK] * len(token_ids)],
            decoded_tokens=[None] * len(token_ids),
        )
    return FlatLogprobs(
        start_indices=[0],
        end_indices=[len(token_ids)],
        token_ids=token_ids,
        logprobs=logprobs,
        ranks=[serving_engine.P2D_SENTINEL_RANK] + [1] * (len(token_ids) - 1),
        decoded_tokens=[str(token_id) for token_id in token_ids],
    )


def test_p2d_decode_prunes_post_eos_tokens(
    monkeypatch: Any,
) -> None:
    """P2D fast path: an EOS at an early or final decode pass terminates the beam.

    The P2D graph always executes both decode passes, so a mid-sequence EOS is
    followed by post-EOS tokens.  With ignore_eos=False the final beams must be
    pruned at the first EOS and reported with finish_reason="stop".
    """
    _install_deterministic_request_ids(monkeypatch)
    p2d_num_steps = 1 + serving_engine.DEFAULT_DECODE_STEPS
    engine = _GroupedEngineClient(
        prepare_outputs=[_engine_output(_flat([11, 12], [-0.1, -0.2]))],
        grouped_outputs=[
            _engine_output(
                _p2d_sentinel_flat(
                    # Beam 0: EOS at pass 1; beam 1: EOS at the final pass.
                    [21, 2, 23, 31, 32, 2],
                    [-0.1, -0.11, -0.12, -0.5, -0.6, -0.7],
                )
            ),
        ],
    )
    outputs = _run_search(engine, max_tokens=1 + p2d_num_steps)
    assert len(outputs) == 1
    choices = outputs[0].outputs
    assert [choice.finish_reason for choice in choices] == ["stop", "stop"]
    assert [choice.stop_reason for choice in choices] == [2, 2]
    assert [choice.token_ids for choice in choices] == [[21, 2], [31, 32, 2]]
    assert [choice.text for choice in choices] == ["21", "31 32"]


def test_p2d_decode_ignore_eos_keeps_post_eos_tokens(
    monkeypatch: Any,
) -> None:
    """ignore_eos=True keeps the full P2D pass output and reports "length"."""
    _install_deterministic_request_ids(monkeypatch)
    p2d_num_steps = 1 + serving_engine.DEFAULT_DECODE_STEPS
    engine = _GroupedEngineClient(
        prepare_outputs=[_engine_output(_flat([11, 12], [-0.1, -0.2]))],
        grouped_outputs=[
            _engine_output(
                _p2d_sentinel_flat(
                    [21, 2, 23, 31, 32, 33],
                    [-0.1, -0.11, -0.12, -0.5, -0.6, -0.7],
                )
            ),
        ],
    )
    outputs = _run_search(engine, max_tokens=1 + p2d_num_steps, ignore_eos=True)
    assert len(outputs) == 1
    choices = outputs[0].outputs
    assert [choice.finish_reason for choice in choices] == ["length", "length"]
    assert [choice.stop_reason for choice in choices] == [None, None]
    assert [choice.token_ids for choice in choices] == [[21, 2, 23], [31, 32, 33]]
    assert [choice.text for choice in choices] == ["21 2 23", "31 32 33"]


def test_p2d_decode_no_eos_reports_length(
    monkeypatch: Any,
) -> None:
    """Beams without EOS keep the previous P2D behavior (length finish)."""
    _install_deterministic_request_ids(monkeypatch)
    p2d_num_steps = 1 + serving_engine.DEFAULT_DECODE_STEPS
    engine = _GroupedEngineClient(
        prepare_outputs=[_engine_output(_flat([11, 12], [-0.1, -0.2]))],
        grouped_outputs=[
            _engine_output(
                _p2d_sentinel_flat(
                    [21, 22, 23, 31, 32, 33],
                    [-0.1, -0.11, -0.12, -0.5, -0.6, -0.7],
                )
            ),
        ],
    )
    outputs = _run_search(engine, max_tokens=1 + p2d_num_steps)
    assert len(outputs) == 1
    choices = outputs[0].outputs
    assert [choice.finish_reason for choice in choices] == ["length", "length"]
    assert [choice.stop_reason for choice in choices] == [None, None]
    assert [choice.token_ids for choice in choices] == [[21, 22, 23], [31, 32, 33]]


def test_p2d_batch_prunes_post_eos_tokens() -> None:
    """Prefill/batch P2D transport: the nested sentinel row is pruned at EOS."""
    p2d_num_steps = 1 + serving_engine.DEFAULT_DECODE_STEPS
    engine = _OrdinaryEngineClient(
        prepare_outputs=[
            _engine_output(
                _p2d_sentinel_flat(
                    [21, 2, 23],
                    [-0.1, -0.11, -0.12],
                    nested=True,
                )
            ),
        ],
    )
    outputs = _run_search(engine, max_tokens=p2d_num_steps)
    assert len(outputs) == 1
    choices = outputs[0].outputs
    assert [choice.finish_reason for choice in choices] == ["stop"]
    assert [choice.stop_reason for choice in choices] == [2]
    assert [choice.token_ids for choice in choices] == [[21, 2]]
    assert [choice.text for choice in choices] == ["21"]
