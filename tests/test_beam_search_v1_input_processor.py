# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import torch
from vllm.inputs import embeds_input, tokens_input
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor

from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchRequestError
from vllm_gr.entrypoints.beam_search_v1.input_processor import BeamSearchInputProcessor
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    AsyncBeamSearchFrontend,
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.sampling_params import BeamSearchParams
from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


def _make_engine(parallel_config: Any = None) -> Any:
    config = SimpleNamespace(
        additional_config={"vllm_gr_beam_max_width": 2},
        model_config=SimpleNamespace(max_model_len=32),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        parallel_config=parallel_config
        or SimpleNamespace(
            data_parallel_size=1, data_parallel_size_local=1, local_engines_only=False
        ),
    )
    # Use native request construction/rank validation without loading a model.
    native: Any = object.__new__(InputProcessor)
    native.vllm_config = config
    native.model_config = config.model_config
    native.generation_config_fields = {}
    native.renderer = SimpleNamespace(tokenizer=None, get_eos_token_id=lambda: 2)
    native.input_preprocessor = SimpleNamespace(
        preprocess=lambda prompt, **kwargs: tokens_input(prompt)
    )
    native._validate_params = lambda *args: None
    native._validate_lora = lambda *args: None
    native._validate_model_inputs = lambda *args: None
    return SimpleNamespace(
        input_processor=native,
        get_supported_tasks=AsyncMock(return_value=("generate",)),
        vllm_config=config,
        submit_gr_session=AsyncMock(return_value=object()),
        abort_gr_session=AsyncMock(),
        release_gr_session=AsyncMock(),
    )


@pytest.mark.parametrize("disable_randomization", ["0", "1"])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "external_id", ["client-request-123", " client-request-123 "]
)
def test_online_request_preserves_native_external_id_routing(
    monkeypatch: pytest.MonkeyPatch, disable_randomization: str, external_id: str
) -> None:
    monkeypatch.setenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", disable_randomization)

    engine = _make_engine()
    serving = SimpleNamespace(
        input_processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2)),
    )
    processor = BeamSearchInputProcessor(engine)

    async def prepare() -> list[BeamBatchRequest]:
        return [
            await processor.process_online_inputs(
                serving,
                OnlineBeamSearchRequest(
                    prompt=[1, 2],
                    request_id=request_id,
                    params=BeamSearchParams(beam_width=2, max_tokens=3),
                ),
            )
            for request_id in (external_id, external_id, "unrelated-request")
        ]

    batches = asyncio.run(prepare())
    internal_ids = [batch.batch_id for batch in batches]
    assert len(set(internal_ids)) == 3
    output_processor = OutputProcessor(tokenizer=None, log_stats=False)
    for batch in batches:
        request = batch.items[0].request
        assert request.request_id == batch.batch_id
        output_processor.add_request(request, prompt=None)

    # Reused external IDs must route to every corresponding internal session.
    assert output_processor.abort_requests([external_id], internal=False) == internal_ids[:2]
    assert set(output_processor.request_states) == {internal_ids[2]}
    assert batches[0].items[0].request.external_req_id == external_id
    assert batches[1].items[0].request.external_req_id == external_id
    assert output_processor.abort_requests([internal_ids[2]], internal=True) == internal_ids[2:]
    assert not output_processor.has_unfinished_requests()
    assert not output_processor.external_req_ids


@pytest.mark.parametrize("temperature", [0.0, 0.001, 0.0099, 0.01, 0.5])  # type: ignore[misc]
def test_beam_metadata_uses_native_normalized_temperature(temperature: float) -> None:
    engine = _make_engine()
    serving = SimpleNamespace(
        input_processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2))
    )
    params = BeamSearchParams(beam_width=2, max_tokens=3, temperature=temperature)
    request = asyncio.run(
        BeamSearchInputProcessor(engine).process_online_inputs(
            serving, OnlineBeamSearchRequest([1, 2], "request-1", params)
        )
    )
    native_params = request.items[0].request.sampling_params
    assert native_params is not None
    assert request.beam_params.temperature == native_params.temperature
    assert native_params.temperature == (0.01 if 0 < temperature < 0.01 else temperature)
    assert params.temperature == temperature


def _prepare_request(
    engine: Any, prompt: Any, params: BeamSearchParams, *, online: bool
) -> BeamBatchRequest:
    tokenizer = SimpleNamespace(
        eos_token_id=2,
        convert_tokens_to_ids=lambda token: {"<begin>": 7, "<end>": 8}[token],
    )
    processor = BeamSearchInputProcessor(engine)
    if online:
        serving = SimpleNamespace(input_processor=SimpleNamespace(tokenizer=tokenizer))
        return asyncio.run(
            processor.process_online_inputs(
                serving, OnlineBeamSearchRequest(prompt, "request-1", params)
            )
        )
    engine.get_supported_tasks = lambda: ("generate",)
    llm = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        _preprocess_cmpl=lambda prompts: prompts,
        _lora_request_to_seq=lambda lora, size: [lora],
    )
    return processor.process_offline_inputs(llm, OfflineBeamSearchRequest([prompt], params))


@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "model_eos,end_token,ignore_eos,expected_stops",
    [
        ([2, 99], None, False, {2, 99}),
        ([2, 99], "<end>", False, {2, 8, 99}),
        ([2, 99], None, True, set()),
        ([2, 99], "<end>", True, set()),
        (99, "<end>", False, {2, 8, 99}),
    ],
)
def test_beam_metadata_uses_native_effective_stop_tokens(
    online: bool,
    model_eos: int | list[int],
    end_token: str | None,
    ignore_eos: bool,
    expected_stops: set[int],
) -> None:
    engine = _make_engine()
    engine.input_processor.generation_config_fields = {"eos_token_id": model_eos}
    request = _prepare_request(
        engine,
        [10, 11],
        BeamSearchParams(beam_width=2, max_tokens=3, end_token=end_token, ignore_eos=ignore_eos),
        online=online,
    )
    native_params = request.items[0].request.sampling_params
    assert native_params is not None
    effective_stops = set(native_params.stop_token_ids or ())
    if native_params.eos_token_id is not None:
        effective_stops.add(native_params.eos_token_id)
    assert effective_stops == expected_stops
    assert set(request.beam_params.stop_token_ids) == expected_stops
    assert request.beam_params.ignore_stop_tokens == native_params.ignore_eos == ignore_eos


@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("mixed", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("begin_token", [None, "<begin>"])  # type: ignore[misc]
def test_embedding_inputs_are_rejected_before_prompt_mutation(
    online: bool,
    mixed: bool,
    begin_token: str | None,
) -> None:
    engine = _make_engine()
    embeddings = torch.zeros((2, 4))
    token_ids = [10, 0] if mixed else None
    mask = [True, False] if mixed else None
    prompt = embeds_input(embeddings, prompt_token_ids=token_ids, is_token_ids=mask)
    native_process_inputs = engine.input_processor.process_inputs
    processed_requests = []

    def process_inputs(*args: Any, **kwargs: Any) -> Any:
        request = native_process_inputs(*args, **kwargs)
        processed_requests.append(request)
        return request

    engine.input_processor.process_inputs = process_inputs
    with pytest.raises(BeamSearchRequestError, match="embedding prompts are not supported"):
        _prepare_request(
            engine,
            prompt,
            BeamSearchParams(beam_width=2, max_tokens=3, begin_token=begin_token),
            online=online,
        )
    assert len(processed_requests) == 1
    request = processed_requests[0]
    assert request.prompt_token_ids == token_ids
    assert request.prompt_is_token_ids == mask
    assert request.prompt_embeds is embeddings


@pytest.mark.parametrize(  # type: ignore[misc]
    "local_only,local_size,expected_ranks",
    [(True, 2, [0, 1, 0, 1]), (True, 1, [0, 0, 0, 0]), (False, 2, [0, 1, 2, 3])],
)
def test_online_entrypoint_uses_only_visible_dp_ranks(
    monkeypatch: pytest.MonkeyPatch, local_only: bool, local_size: int, expected_ranks: list[int]
) -> None:
    import vllm_gr.entrypoints.beam_search_v1 as entrypoint
    from vllm_gr.entrypoints.openai import serving_engine

    engine = _make_engine(
        SimpleNamespace(
            data_parallel_size=4, data_parallel_size_local=local_size, local_engines_only=local_only
        )
    )
    serving = SimpleNamespace(
        engine_client=engine,
        input_processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2)),
    )
    ranks = []

    async def prepare_and_submit(
        frontend: AsyncBeamSearchFrontend, call: OnlineBeamSearchRequest
    ) -> list[Any]:
        request = await frontend.prepare_request(call)
        session = await frontend.submit_once(request)
        rank = request.items[0].request.data_parallel_rank
        assert session.data_parallel_rank == call.data_parallel_rank == rank
        ranks.append(rank)
        await frontend.close(session)
        assert session.data_parallel_rank == rank
        return []

    from vllm_gr.entrypoints.beam_search_v1.output_processor import (
        BeamSearchOutputProcessor,
    )

    # This routing-only fixture supplies an explicit test final consumer.
    monkeypatch.setattr(BeamSearchOutputProcessor, "ensure_available", lambda: None)
    monkeypatch.setattr(entrypoint, "run_beam_search_v1_async", prepare_and_submit)
    monkeypatch.setattr(serving_engine, "_dp_rank_counter", 0)

    async def run() -> None:
        for index in range(4):
            outputs = [
                output
                async for output in serving_engine.beam_search_v1(
                    serving,
                    [1, 2],
                    f"request-{index}",
                    BeamSearchParams(beam_width=2, max_tokens=3),
                )
            ]
            assert outputs == []

    asyncio.run(run())
    assert ranks == expected_ranks
