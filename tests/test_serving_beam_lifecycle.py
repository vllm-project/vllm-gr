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
) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                finish_reason=finish_reason,
                logprobs=flat,
            )
        ],
        finished=True,
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
        fail_step: int | None = None,
    ) -> None:
        super().__init__(prepare_outputs)
        self.grouped_outputs = list(grouped_outputs)
        self.register_calls: list[SimpleNamespace] = []
        self.beam_updates: list[BeamRequestStepUpdate] = []
        self.fail_cleanup = fail_cleanup
        self.fail_step = fail_step
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
        if (
            self.fail_step is not None
            and update.beam_width != 0
            and len(self.beam_updates) == self.fail_step
        ):
            raise RuntimeError("injected failure")
        self.beam_updates.append(update)
        if self.fail_cleanup and update.beam_width == 0:
            raise RuntimeError("cleanup failed")


class _FakeCatalog:
    """Minimal Catalog stand-in for beam_search().

    Each entry is a full token sequence; ``valid`` returns the trie
    continuation set and ``is_terminal`` reports exact matches.
    """

    def __init__(self, entries: list[list[int]]) -> None:
        self.entries = [tuple(entry) for entry in entries]

    def valid(self, token_ids: list[int]) -> set[int]:
        prefix = tuple(token_ids)
        return {
            entry[len(prefix)]
            for entry in self.entries
            if entry[: len(prefix)] == prefix and len(entry) > len(prefix)
        }

    def is_terminal(self, token_ids: list[int]) -> bool:
        return tuple(token_ids) in self.entries

    def apply_catalog_filter(
        self,
        token_ids: list[int],
        logprobs: list[float],
        valid_tokens_set: set[int],
    ) -> list[float]:
        return [
            lp if tid in valid_tokens_set else -float("inf") for tid, lp in zip(token_ids, logprobs)
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def sequence_at(self, i: int) -> list[int]:
        return list(self.entries[i])


def _serving(engine_client: Any, catalog: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        engine_client=engine_client,
        input_processor=SimpleNamespace(tokenizer=_Tokenizer()),
        models=SimpleNamespace(catalog=catalog, model_name="test-model"),
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
    assert output.metrics.num_processed_tokens >= 0
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
    # BW=2, 3 steps: 2 (step 0) + 4 (step 1) + 4 (step 2)
    assert outputs[0].metrics.num_processed_tokens == 10


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


def test_terminal_catalog_beams_are_completed_not_lost(
    monkeypatch: Any,
) -> None:
    """A beam whose generated tokens complete a catalog item must be
    finished and kept (regression for terminal beams being masked to -inf
    and dropped by the finite-score filter)."""
    _install_deterministic_request_ids(monkeypatch)
    catalog = _FakeCatalog([[11, 21], [12, 31]])
    engine = _GroupedEngineClient(
        prepare_outputs=[_engine_output(_flat([11, 12], [-0.1, -0.2]))],
        grouped_outputs=[
            _engine_output(
                _combined_flat(
                    [
                        ([21, 41], [-0.5, -3.0]),
                        ([31, 42], [-0.6, -3.0]),
                    ]
                )
            ),
            # Step 2: both prefixes [11, 21] and [12, 31] are terminal, so
            # every candidate gets masked away; the beams must complete.
            _engine_output(
                _combined_flat(
                    [
                        ([99, 98], [-0.01, -0.02]),
                        ([97, 96], [-0.03, -0.04]),
                    ]
                )
            ),
        ],
    )

    outputs = asyncio.run(
        _collect_outputs(
            serving_engine.beam_search(
                _serving(engine, catalog=catalog),
                {
                    "prompt": "prompt text",
                    "prompt_token_ids": [10],
                },
                "request-0",
                _params(max_tokens=3),
                trace_headers={"trace": "value"},
            )
        )
    )

    assert len(outputs) == 1
    choices = outputs[0].outputs
    assert [choice.finish_reason for choice in choices] == ["stop", "stop"]
    assert [choice.token_ids for choice in choices] == [[11, 21], [12, 31]]
    assert [choice.cumulative_logprob for choice in choices] == pytest.approx([-0.6, -0.8])
    assert len(engine.beam_updates) == 3
    assert engine.beam_updates[-1].beam_width == 0


def test_grouped_step_exception_still_cleans_up_session(
    monkeypatch: Any,
) -> None:
    """An exception mid-decode must still purge the persistent beam session
    (regression for stranded KV-cache/session state on cancellation)."""
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
            # Never consumed: the decode-step update raises before the
            # grouped result is awaited, but register_beam_output already
            # popped one entry.
            _engine_output(
                _combined_flat(
                    [
                        ([41, 42], [-0.1, -0.2]),
                        ([51, 52], [-0.1, -0.2]),
                    ]
                )
            ),
        ],
        fail_step=1,  # raise on the second beam_request_step_update
    )

    with pytest.raises(RuntimeError, match="injected failure"):
        _run_search(engine, max_tokens=3)

    # The finally block must still have signalled a cleanup update.
    assert engine.beam_updates[-1].beam_width == 0
    assert engine.beam_updates[-1].session_id == "request-0-step-0-beam-0"
    assert engine.beam_updates[-1].pruned_ids == engine.beam_updates[0].child_beam_ids
