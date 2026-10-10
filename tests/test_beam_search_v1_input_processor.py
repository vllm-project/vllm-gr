# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import torch
from vllm.exceptions import VLLMValidationError
from vllm.inputs import embeds_input, tokens_input
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor

from vllm_gr.entrypoints.beam_search_v1.client import (
    AsyncBeamSearchClient,
    BeamSearchClient,
)
from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchRequestError
from vllm_gr.entrypoints.beam_search_v1.input_processor import BeamSearchInputProcessor
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.sampling_params import BeamSearchParams
from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchOutput,
    BeamBatchRequest,
    BeamBatchResult,
    BeamOutputSequence,
    BeamRequestOutput,
)


def _canonical_gr_config(*, enabled: bool = True, backend: str = "cuda") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "beam": {
            "execution_mode": "v1",
            "graph_enabled": True,
            "max_width": 2,
            "max_decode_steps": 3,
        },
        "constraint_table": {
            "enabled": enabled,
            "backend": backend,
            "path": "/tmp/test-constraint-table.safetensors",
            "format": "constraint_table_v1",
            "artifact_digest": "sha256:" + "a" * 64,
            "tokenizer_digest": "sha256:" + "b" * 64,
            "max_top_k": 4,
        },
        "attention_backend": None,
    }


def _make_engine(
    parallel_config: Any = None, *, additional_config: dict[str, Any] | None = None
) -> Any:
    config = SimpleNamespace(
        additional_config=(
            {"vllm_gr": _canonical_gr_config()} if additional_config is None else additional_config
        ),
        model_config=SimpleNamespace(max_model_len=32),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32, async_scheduling=True),
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
                    params=BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True),
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
    params = BeamSearchParams(beam_width=2, max_tokens=3, temperature=temperature, ignore_eos=True)
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


def test_online_request_without_a_tokenizer_names_the_startup_flag() -> None:
    """`skip_tokenizer_init` is a configuration answer, not an internal fault.

    A token-less engine is the only way to reach this guard, and the parameter
    reported is the one the legacy interface reports for the same cause.
    """
    engine = _make_engine()
    processor = BeamSearchInputProcessor(engine)
    serving = SimpleNamespace(input_processor=SimpleNamespace(tokenizer=None))

    with pytest.raises(VLLMValidationError) as caught:
        asyncio.run(
            processor.process_online_inputs(
                serving,
                OnlineBeamSearchRequest(
                    [1, 2],
                    "request-1",
                    BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True),
                ),
            )
        )

    assert (caught.value.parameter, caught.value.value) == ("skip_tokenizer_init", True)


@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(
    "begin,end", [(None, None), ("<begin>", None), (None, "<end>"), ("<begin>", "<end>")]
)  # type: ignore[misc]
def test_public_result_keeps_the_accepted_boundary_ids(
    online: bool, begin: str | None, end: str | None
) -> None:
    engine = _make_engine()
    lookups: list[str] = []

    def resolve(token: str) -> int:
        lookups.append(token)
        return {"<begin>": 7, "<end>": 8}[token]

    tokenizer = SimpleNamespace(
        eos_token_id=2,
        convert_tokens_to_ids=resolve,
        decode=lambda ids, **kwargs: "|".join(map(str, ids)),
        batch_decode=lambda rows, **kwargs: ["|".join(map(str, ids)) for ids in rows],
    )
    params = BeamSearchParams(
        beam_width=2, max_tokens=3, ignore_eos=True, begin_token=begin, end_token=end
    )
    if online:
        serving = SimpleNamespace(
            engine_client=engine, input_processor=SimpleNamespace(tokenizer=tokenizer)
        )
        client: Any = AsyncBeamSearchClient(serving)
        call: Any = OnlineBeamSearchRequest([1, 2], "external-id", params)
        request = asyncio.run(client.prepare_request(call))
    else:
        engine.get_supported_tasks = lambda: ("generate",)
        llm = SimpleNamespace(
            llm_engine=engine,
            get_tokenizer=lambda: tokenizer,
            _preprocess_cmpl=lambda prompts: prompts,
            _lora_request_to_seq=lambda lora, size: [lora],
        )
        client = BeamSearchClient(llm)
        call = OfflineBeamSearchRequest([[1, 2]], params)
        request = client.prepare_request(call)
    assert lookups == [token for token in (begin, end) if token is not None]
    assert not request.output_options.include_parent_beam_indices
    suffix = (9,) * (3 - int(begin is not None) - int(end is not None))
    result = BeamBatchResult(
        request.protocol_version,
        request.batch_id,
        BeamBatchOutput(
            (
                BeamRequestOutput(
                    0, request.batch_id, (BeamOutputSequence(suffix, -1.25, (), "length", None),)
                ),
            )
        ),
        None,
    )
    # A caller mutation after admission must not alter the accepted boundaries
    # or trigger another tokenizer lookup when the response is constructed.
    params.begin_token = params.end_token = "changed-after-admission"
    output = client.build_output(call, request, result)[0]
    boundaries = ([7] if begin else []) + list(suffix) + ([8] if end else [])
    if online:
        assert output.prompt_token_ids == [1, 2]
        assert output.outputs[0].token_ids == boundaries
        assert output.outputs[0].cumulative_logprob == -1.25
    else:
        assert output.sequences[0].tokens == [1, 2] + boundaries
        assert output.sequences[0].cum_logprob == -1.25
    assert lookups == [token for token in (begin, end) if token is not None]


def test_offline_inputs_preserve_caller_formed_batch_order() -> None:
    engine = _make_engine()
    engine.get_supported_tasks = lambda: ("generate",)
    tokenizer = SimpleNamespace(
        eos_token_id=2,
        convert_tokens_to_ids=lambda token: {"<begin>": 7, "<end>": 8}[token],
    )
    llm = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        _preprocess_cmpl=lambda prompts: prompts,
        _lora_request_to_seq=lambda lora, size: [lora] * size,
    )

    request = BeamSearchInputProcessor(engine).process_offline_inputs(
        llm,
        OfflineBeamSearchRequest(
            prompts=[[10, 11], [20, 21, 22]],
            params=BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True),
        ),
    )

    assert len(request.items) == 2
    assert [item.item_index for item in request.items] == [0, 1]
    assert [item.request.prompt_token_ids for item in request.items] == [
        [10, 11],
        [20, 21, 22],
    ]
    assert request.batch_id == request.items[0].request.request_id
    assert request.items[1].request.request_id != request.batch_id


def test_offline_inputs_reject_unsupported_batch_size_before_admission() -> None:
    engine = _make_engine()
    engine.get_supported_tasks = lambda: ("generate",)
    tokenizer = SimpleNamespace(eos_token_id=2, convert_tokens_to_ids=lambda token: None)
    llm = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        _preprocess_cmpl=lambda prompts: prompts,
        _lora_request_to_seq=lambda lora, size: [lora] * size,
    )

    with pytest.raises(BeamSearchRequestError, match="supports batch sizes"):
        BeamSearchInputProcessor(engine).process_offline_inputs(
            llm,
            OfflineBeamSearchRequest(
                prompts=[[10], [20], [30]],
                params=BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True),
            ),
        )


@pytest.mark.parametrize(  # type: ignore[misc]
    "additional_config",
    [
        {"vllm_gr_beam_execution_mode": "v1", "vllm_gr_beam_max_width": 2},
        {"vllm_gr": _canonical_gr_config(enabled=False)},
        {"vllm_gr": _canonical_gr_config(backend="cpu")},
    ],
    ids=["missing", "disabled", "non-cuda"],
)
def test_request_requires_enabled_canonical_cuda_constraint_table(
    additional_config: dict[str, Any],
) -> None:
    engine = _make_engine(additional_config=additional_config)
    with pytest.raises(
        BeamSearchRequestError,
        match="requires an enabled canonical CUDA constraint table",
    ):
        _prepare_request(
            engine,
            [1, 2],
            BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True),
            online=True,
        )


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
    if not ignore_eos:
        with pytest.raises(BeamSearchRequestError, match="ignore_eos=True"):
            _prepare_request(
                engine,
                [10, 11],
                BeamSearchParams(beam_width=2, max_tokens=3, end_token=end_token, ignore_eos=False),
                online=online,
            )
        return
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
            BeamSearchParams(beam_width=2, max_tokens=3, begin_token=begin_token, ignore_eos=True),
            online=online,
        )
    assert len(processed_requests) == 1
    request = processed_requests[0]
    assert request.prompt_token_ids == token_ids
    assert request.prompt_is_token_ids == mask
    assert request.prompt_embeds is embeddings


@pytest.mark.parametrize("local_only,local_size", [(True, 2), (True, 1), (False, 2)])  # type: ignore[misc]
def test_online_entrypoint_rejects_dp_before_submission(local_only: bool, local_size: int) -> None:
    from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchRequestError

    engine = _make_engine(
        SimpleNamespace(
            data_parallel_size=4, data_parallel_size_local=local_size, local_engines_only=local_only
        )
    )
    with pytest.raises(BeamSearchRequestError, match="TP/PP/DP/DCP/PCP=1"):
        _prepare_request(
            engine,
            [1, 2],
            BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True),
            online=True,
        )
    engine.submit_gr_session.assert_not_called()
