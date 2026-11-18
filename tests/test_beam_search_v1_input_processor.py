# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest
import torch
from vllm.exceptions import VLLMValidationError
from vllm.inputs import TextPrompt, TokensPrompt, embeds_input, tokens_input
from vllm.renderers.params import TokenizeParams
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
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView


def _select_backend(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    """Select the backend that serves ``device`` for this request.

    The frontend asks the selected backend for its Constraint Table placement;
    a test therefore selects a backend rather than relying on a device name in
    shared code.
    """
    from vllm_gr.platforms.cuda_services import CudaConfigService

    import vllm_gr.platforms.loader as loader

    # Bound services are cached per process; drop them before pointing
    # selection at this test's backend.
    loader._reset_backend_services_for_tests()
    service = CudaConfigService()
    placement = {"cuda": "cuda", "npu": "npu"}.get(device)
    monkeypatch.setattr(service, "default_constraint_table_backend", lambda: placement)
    provider = SimpleNamespace(
        backend_id="cuda" if device == "cuda" else "ascend",
        create_config_service=lambda: service,
    )
    monkeypatch.setattr("vllm_gr.platforms.loader.activate_current_backend", lambda: provider)


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
        model_config=SimpleNamespace(max_model_len=32, is_encoder_decoder=False),
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
    # 0.30 renders raw prompts through Renderer, not InputPreprocessor.
    # Keep native prompt parsing and request construction; only token rendering
    # is stubbed so these metadata/routing tests need no model or tokenizer.
    native.renderer = SimpleNamespace(
        tokenizer=None,
        get_eos_token_id=lambda: 2,
        default_cmpl_tok_params=TokenizeParams(max_total_tokens=32),
        render_cmpl=lambda prompts, params: [tokens_input(p["prompt_token_ids"]) for p in prompts],
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
                    prompts=([1, 2],),
                    request_id=request_id,
                    params=BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True),
                    session_id="caller-session",
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
        assert request.session_id == "caller-session"
        assert request.request_id != request.session_id
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
            serving, OnlineBeamSearchRequest(([1, 2],), "request-1", params)
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
                serving, OnlineBeamSearchRequest((prompt,), "request-1", params)
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
                    ([1, 2],),
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
        beam_width=2,
        max_tokens=3 + int(begin is not None) + int(end is not None),
        ignore_eos=True,
        begin_token=begin,
        end_token=end,
    )
    if online:
        serving = SimpleNamespace(
            engine_client=engine, input_processor=SimpleNamespace(tokenizer=tokenizer)
        )
        client: Any = AsyncBeamSearchClient(serving)
        call: Any = OnlineBeamSearchRequest(([1, 2],), "external-id", params)
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
    suffix = (9,) * 3
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
        assert output.prompt_token_ids == (1, 2)
        assert output.sequences[0].token_ids == tuple(boundaries)
        assert output.sequences[0].cumulative_logprob == -1.25
    assert lookups == [token for token in (begin, end) if token is not None]


@pytest.mark.parametrize("batch_size", range(1, 9))  # type: ignore[misc]
def test_offline_inputs_preserve_caller_formed_batch_order(batch_size: int) -> None:
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

    prompts = [[10 * (i + 1), 10 * (i + 1) + 1] for i in range(batch_size)]
    request = BeamSearchInputProcessor(engine).process_offline_inputs(
        llm,
        OfflineBeamSearchRequest(
            prompts=prompts,
            params=BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True),
        ),
    )

    assert len(request.items) == batch_size
    assert [item.item_index for item in request.items] == list(range(batch_size))
    assert [item.request.prompt_token_ids.tolist() for item in request.items] == prompts
    assert request.batch_id == request.items[0].request.request_id
    assert len({item.request.request_id for item in request.items}) == batch_size


@pytest.mark.parametrize("text_input", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("begin", [None, "<begin>"])  # type: ignore[misc]
def test_offline_compact_output_preserves_accepted_prompt_snapshot(
    text_input: bool, begin: str | None
) -> None:
    engine = _make_engine()
    engine.get_supported_tasks = lambda: ("generate",)
    tokenizer = SimpleNamespace(
        eos_token_id=2,
        convert_tokens_to_ids=lambda token: {"<begin>": 7, "<end>": 8}[token],
    )
    accepted = [1, 2]
    llm = SimpleNamespace(
        llm_engine=engine,
        get_tokenizer=lambda: tokenizer,
        _preprocess_cmpl=lambda prompts: [{"prompt_token_ids": accepted}],
        _lora_request_to_seq=lambda lora, size: [lora],
    )
    params = BeamSearchParams(beam_width=2, max_tokens=4 + int(begin is not None), ignore_eos=True)
    params.begin_token = begin
    params.end_token = "<end>"
    raw_tokens = [9, 9]
    source_prompt: TextPrompt | TokensPrompt = (
        {"prompt": "rendered text"} if text_input else {"prompt_token_ids": raw_tokens}
    )
    call = OfflineBeamSearchRequest([source_prompt], params)
    client = BeamSearchClient(llm)
    request = client.prepare_request(call)
    assert request.items[0].request.prompt_token_ids.tolist() == [1, 2] + (
        [] if begin is None else [7]
    )
    accepted[0] = 99
    raw_tokens[0] = 88
    # The engine owns a snapshot; later caller mutations do not leak in.
    assert request.items[0].request.prompt_token_ids.tolist()[0] == 1
    request.items[0].request.prompt_token_ids[0] = 77
    result = BeamBatchResult(
        request.protocol_version,
        request.batch_id,
        BeamBatchOutput((BeamRequestOutput(0, request.batch_id, ()),)),
        None,
    )
    first = client.build_output(call, request, result)[0]
    second = client.build_output(call, request, result)[0]
    assert first.prompt_token_ids == (1, 2)
    assert first.prompt_token_ids is second.prompt_token_ids


@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "device,independent_k", [("cuda", False), ("cuda", True), ("npu", False)]
)
@pytest.mark.parametrize(  # type: ignore[misc]
    "trie_ids", [None, [0, 1], [1, 1], (1, 0), [True, 0], [-1, 0], [None, 1], [0]]
)
def test_per_prompt_trie_selection(
    trie_ids: Any,
    online: bool,
    independent_k: bool,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _canonical_gr_config(backend=device)
    canonical["constraint_table"]["format"] = "constraint_table_v2"
    engine = _make_engine(additional_config={"vllm_gr": canonical})
    _select_backend(monkeypatch, device)
    params = BeamSearchParams(
        beam_width=2,
        max_tokens=3,
        ignore_eos=True,
        per_parent_beam_widths=[2, 1, 1] if independent_k else None,
    )

    def prepare() -> BeamBatchRequest:
        processor = BeamSearchInputProcessor(engine)
        if online:
            serving = SimpleNamespace(
                input_processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2))
            )
            call = OnlineBeamSearchRequest(
                prompts=([1, 2], [3, 4]),
                request_id="external-id",
                params=params,
                trie_ids=trie_ids,
            )
            return asyncio.run(processor.process_online_inputs(serving, call))
        engine.get_supported_tasks = lambda: ("generate",)
        llm = SimpleNamespace(
            get_tokenizer=lambda: SimpleNamespace(eos_token_id=2),
            _preprocess_cmpl=lambda prompts: prompts,
            _lora_request_to_seq=lambda lora, size: [lora] * size,
        )
        offline_call = OfflineBeamSearchRequest(
            prompts=[[1, 2], [3, 4]], params=params, trie_ids=trie_ids
        )
        return processor.process_offline_inputs(llm, offline_call)

    if trie_ids is not None and (
        len(trie_ids) != 2 or any(type(trie_id) is not int or trie_id < 0 for trie_id in trie_ids)
    ):
        with pytest.raises(BeamSearchRequestError, match="trie_ids"):
            prepare()
    else:
        request = prepare()
        assert [item.trie_id for item in request.items] == (
            [0, 0] if trie_ids is None else list(trie_ids)
        )
        assert request.beam_params.per_parent_beam_widths == ((2, 1, 1) if independent_k else None)


def test_configured_candidate_width_reaches_the_request_under_dynamic_schedules() -> None:
    """A configured K is carried through, whatever the dynamic schedule.

    Each stage selects its own ``output_width`` survivors out of the K
    candidates its parents supply, so a schedule needs only
    ``K >= max(stage widths)``.  The #525 review found a uniform but narrower
    schedule silently ran TopK at the stage width; the public frontend must now
    hand the worker K itself and ask the sampler for K logprobs per row.

    An unset K must not be derived onto the wire: it stays ``None`` so every
    stage keeps its pre-decoupling top-k (its own output width), while the
    sampler still returns ``W`` logprobs per row as before.
    """

    def _config(candidate_width: int | None) -> dict[str, Any]:
        additional = {"vllm_gr": _canonical_gr_config()}
        if candidate_width is not None:
            additional["vllm_gr"]["constraint_table"]["candidate_width"] = candidate_width
        return additional

    def _submit(candidate_width: int | None, dynamic_widths: list[int] | None) -> BeamBatchRequest:
        engine = _make_engine(additional_config=_config(candidate_width))
        # Offline construction is synchronous; the shared fixture's AsyncMock
        # would hand a coroutine to the engine request.
        engine.get_supported_tasks = lambda: ("generate",)
        engine.vllm_config.device_config = SimpleNamespace(device="cuda")
        tokenizer = SimpleNamespace(eos_token_id=2, convert_tokens_to_ids=lambda token: 7)
        llm = SimpleNamespace(
            get_tokenizer=lambda: tokenizer,
            _preprocess_cmpl=lambda prompts: prompts,
            _lora_request_to_seq=lambda lora, size: [lora] * size,
        )
        request = OfflineBeamSearchRequest(
            prompts=[[1, 2]],
            params=BeamSearchParams(
                beam_width=2,
                max_tokens=3,
                ignore_eos=True,
                dynamic_beam_widths=dynamic_widths,
            ),
        )
        return BeamSearchInputProcessor(engine).process_offline_inputs(llm, request)

    def _assert_k(request: Any, wire: int | None, logprobs: int) -> None:
        # The worker reads K from the wire params, and the sampler is asked for
        # K logprobs per row -- the two places K has to arrive intact.
        assert request.beam_params.candidate_width == wire
        for item in request.items:
            assert item.request.sampling_params.logprobs == logprobs

    # Configured K plus a uniform but narrower schedule: K, not the stage width.
    # The #525 review observed K=2 / (1, 1, 1) running TopK=1; it must carry 2.
    _assert_k(_submit(2, [1, 1, 1]), 2, 2)
    # Configured K plus a shrinking schedule: K covers every stage.
    _assert_k(_submit(2, [2, 2, 1]), 2, 2)
    # Configured K plus a schedule that retains exactly K: unchanged.
    _assert_k(_submit(2, [2, 2, 2]), 2, 2)
    # No configured K: nothing on the wire, so each stage takes its own output
    # width as top-k; the sampler still returns W logprobs, as before.
    for widths in ([2, 2, 1], [1, 1, 1], None):
        request = _submit(None, widths)
        _assert_k(request, None, 2)
        if widths is not None:
            params = request.beam_params
            assert params.dynamic_beam_widths == tuple(widths)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("max_logprobs", "accepted"),
    [
        (-1, True),  # vLLM's default sentinel means "the whole vocabulary"
        (0, False),
        (1, False),  # K == 2 does not fit
        (2, True),
        (16, True),
    ],
)
def test_request_logprob_budget_matches_vllm_sentinel_semantics(
    max_logprobs: int, accepted: bool
) -> None:
    """``max_logprobs=-1`` is vLLM's sentinel, not a negative limit.

    ``SamplingParams._validate_logprobs`` normalizes it to the vocabulary size
    before comparing; comparing a positive candidate width against the raw
    sentinel rejected every default deployment, K == W included.
    """
    engine = _make_engine()
    # Offline construction is synchronous; the shared fixture's AsyncMock would
    # hand a coroutine to the engine request.
    engine.get_supported_tasks = lambda: ("generate",)
    engine.vllm_config.model_config.max_logprobs = max_logprobs
    engine.vllm_config.model_config.get_vocab_size = lambda: 128
    tokenizer = SimpleNamespace(eos_token_id=2, convert_tokens_to_ids=lambda token: 7)
    llm = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        _preprocess_cmpl=lambda prompts: prompts,
        _lora_request_to_seq=lambda lora, size: [lora] * size,
    )
    request = OfflineBeamSearchRequest(
        prompts=[[1, 2]],
        params=BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True),
    )
    processor = BeamSearchInputProcessor(engine)

    if accepted:
        processor.process_offline_inputs(llm, request)
        return
    with pytest.raises(BeamSearchRequestError, match="exceeds max_logprobs"):
        processor.process_offline_inputs(llm, request)


@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("widths", [None, [4, 4, 4], [4, 2, 1]])  # type: ignore[misc]
def test_request_k_schedule_overrides_startup_candidate_width(
    online: bool, widths: list[int] | None
) -> None:
    config = _canonical_gr_config()
    config["beam"]["max_width"] = 4
    config["constraint_table"]["candidate_width"] = 4
    engine = _make_engine(additional_config={"vllm_gr": config})
    engine.vllm_config.device_config = SimpleNamespace(device="cuda")
    request = _prepare_request(
        engine,
        [1, 2],
        BeamSearchParams(
            beam_width=4,
            max_tokens=3,
            ignore_eos=True,
            dynamic_beam_widths=widths,
            per_parent_beam_widths=[4, 2, 1],
        ),
        online=online,
    )
    for stage in range(3):
        view = BeamBatchExecutionView(
            session_id=request.batch_id,
            dispatch_id=stage,
            stage_index=stage,
            kind="prefill" if stage == 0 else "decode",
            decode_step=None if stage == 0 else stage - 1,
            owner_generation=0,
            request_ids=(request.batch_id,),
            num_prompt_tokens_by_request=(2,),
            beam_width=request.beam_params.beam_width,
            produces_output=True,
            dynamic_beam_widths=request.beam_params.dynamic_beam_widths,
            per_parent_beam_widths=request.beam_params.per_parent_beam_widths,
        )
        assert view.candidate_top_k == (4, 2, 1)[stage]
        assert view.output_width == (widths or [4, 4, 4])[stage]


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
                prompts=[[10 * i] for i in range(1, 10)],
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


@pytest.mark.parametrize("device", ["cuda", "npu"])  # type: ignore[misc]
def test_request_rejects_width_below_fixed_runtime_before_submission(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = _canonical_gr_config(backend=device)
    canonical["beam"]["max_width"] = 128
    engine = _make_engine(additional_config={"vllm_gr": canonical})
    _select_backend(monkeypatch, device)
    with pytest.raises(BeamSearchRequestError, match="fixed runtime width 128"):
        _prepare_request(
            engine,
            [1, 2],
            BeamSearchParams(beam_width=4, max_tokens=3, ignore_eos=True),
            online=True,
        )
    engine.submit_gr_session.assert_not_called()


def test_npu_request_accepts_matching_width_and_canonical_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _make_engine(additional_config={"vllm_gr": _canonical_gr_config(backend="npu")})
    _select_backend(monkeypatch, "npu")
    request = _prepare_request(
        engine, [1, 2], BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True), online=True
    )
    assert request.beam_params.beam_width == 2


@pytest.mark.parametrize("device", ["cuda", "npu"])  # type: ignore[misc]
@pytest.mark.parametrize("widths", [None, [2, 2, 2], [2, 1, 2]])  # type: ignore[misc]
def test_stage_widths_and_return_limit_reach_worker_contract(
    device: str, widths: list[int] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _make_engine(additional_config={"vllm_gr": _canonical_gr_config(backend=device)})
    _select_backend(monkeypatch, device)
    request = _prepare_request(
        engine,
        [1, 2],
        BeamSearchParams(
            beam_width=2,
            dynamic_beam_widths=widths,
            result_width=1,
            max_tokens=3 if widths is None else len(widths),
            ignore_eos=True,
        ),
        online=True,
    )
    assert request.beam_params.dynamic_beam_widths == (None if widths is None else tuple(widths))
    assert request.output_options.num_return_sequences == 1


def test_cuda_per_parent_beam_widths_reaches_worker_contract() -> None:
    config = _canonical_gr_config(backend="cuda")
    config["beam"]["max_width"] = 4
    engine = _make_engine(additional_config={"vllm_gr": config})
    engine.vllm_config.device_config = SimpleNamespace(device="cuda")
    request = _prepare_request(
        engine,
        [1, 2],
        BeamSearchParams(
            beam_width=4,
            dynamic_beam_widths=[2, 4, 4],
            per_parent_beam_widths=[2, 2, 1],
            result_width=2,
            max_tokens=3,
            ignore_eos=True,
        ),
        online=True,
    )
    assert request.beam_params.dynamic_beam_widths == (2, 4, 4)
    assert request.beam_params.per_parent_beam_widths == (2, 2, 1)


@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
def test_npu_per_parent_beam_widths_is_rejected_before_submission(online: bool) -> None:
    engine = _make_engine(additional_config={"vllm_gr": _canonical_gr_config(backend="npu")})
    engine.vllm_config.device_config = SimpleNamespace(device="npu")
    with pytest.raises(BeamSearchRequestError, match="per_parent_beam_widths requires CUDA"):
        _prepare_request(
            engine,
            [1, 2],
            BeamSearchParams(
                beam_width=2,
                per_parent_beam_widths=[2, 2, 2],
                max_tokens=3,
                ignore_eos=True,
            ),
            online=online,
        )
    engine.submit_gr_session.assert_not_called()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("capacity", [1, 6, 7, 9])  # type: ignore[misc]
def test_cuda_unconstrained_requests_vary_budget_with_one_runtime(
    online: bool, capacity: int
) -> None:
    canonical = _canonical_gr_config(enabled=False)
    canonical["constrained_generation"] = False
    canonical["beam"]["max_decode_steps"] = capacity
    engine = _make_engine(additional_config={"vllm_gr": canonical})
    engine.vllm_config.device_config = SimpleNamespace(device="cuda")

    def prepare(budget: int) -> BeamBatchRequest:
        return _prepare_request(
            engine,
            [1, 2],
            BeamSearchParams(beam_width=2, max_tokens=budget, ignore_eos=True),
            online=online,
        )

    for budget in (*range(1, capacity + 1), 1, capacity):
        request = prepare(budget)
        assert request.beam_params.max_tokens == budget
        native = request.items[0].request.sampling_params
        assert native is not None and native.max_tokens == budget and native.ignore_eos
        assert canonical["beam"]["max_decode_steps"] == capacity

    with pytest.raises(
        BeamSearchRequestError,
        match=f"Decode steps {capacity + 1} exceed runtime limit {capacity}",
    ):
        prepare(capacity + 1)
    # A rejected over-capacity request must not poison the existing engine config.
    assert prepare(1).beam_params.max_tokens == 1
    assert prepare(capacity).beam_params.max_tokens == capacity


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("backend", ["cuda", "npu"])  # type: ignore[misc]
@pytest.mark.parametrize("constrained", [None, True])  # type: ignore[misc]
@pytest.mark.parametrize("table_format", ["constraint_table_v1", "constraint_table_v2"])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "begin,end", [(None, None), ("<begin>", None), (None, "<end>"), ("<begin>", "<end>")]
)
def test_constrained_length_is_checked_before_native_input_processing(
    online: bool,
    backend: str,
    constrained: bool | None,
    table_format: str,
    begin: str | None,
    end: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _canonical_gr_config(backend=backend)
    config["constrained_generation"] = constrained
    config["constraint_table"]["format"] = table_format
    config["beam"]["max_decode_steps"] = 7
    engine = _make_engine(additional_config={"vllm_gr": config})
    _select_backend(monkeypatch, backend)
    process_inputs = Mock(wraps=engine.input_processor.process_inputs)
    engine.input_processor.process_inputs = process_inputs

    def prepare(length: int) -> BeamBatchRequest:
        return _prepare_request(
            engine,
            [1, 2],
            BeamSearchParams(
                beam_width=2,
                max_tokens=length + int(begin is not None) + int(end is not None),
                ignore_eos=True,
                begin_token=begin,
                end_token=end,
            ),
            online=online,
        )

    for length in (1, 2, 4, 7):
        with pytest.raises(
            BeamSearchRequestError,
            match=f"effective generation length {length} must equal {table_format} depth",
        ):
            prepare(length)
        process_inputs.assert_not_called()
        engine.submit_gr_session.assert_not_called()
    # Rejections must not alter the startup config or poison subsequent requests.
    request = prepare(3)
    assert request.beam_params.max_tokens == 3
    assert config["beam"]["max_decode_steps"] == 7
    process_inputs.assert_called_once()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("missing", [False, True])  # type: ignore[misc]
def test_missing_constraint_table_is_not_reported_as_a_length_mismatch(
    online: bool, missing: bool
) -> None:
    canonical = _canonical_gr_config(enabled=False)
    canonical["beam"]["max_decode_steps"] = 7
    additional = (
        {
            "vllm_gr_beam_execution_mode": "v1",
            "vllm_gr_beam_max_width": 2,
            "vllm_gr_beam_max_decode_steps": 7,
        }
        if missing
        else {"vllm_gr": canonical}
    )
    engine = _make_engine(additional_config=additional)
    # Unconstrained mode must be explicitly selected; missing artifacts are
    # deployment errors, not evidence of a fixed three-token generation budget.
    for length in (1, 2, 3, 7):
        with pytest.raises(
            BeamSearchRequestError, match="requires an enabled canonical CUDA constraint table"
        ):
            _prepare_request(
                engine,
                [1, 2],
                BeamSearchParams(beam_width=2, max_tokens=length, ignore_eos=True),
                online=online,
            )


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("length", [1, 2, 4, 7])  # type: ignore[misc]
def test_core_rechecks_constrained_length_without_frontend_validation(length: int) -> None:
    from vllm_gr.v1.engine.beam_search_v1 import validate_gr_request

    config = _canonical_gr_config()
    config["beam"]["max_decode_steps"] = 7
    engine = _make_engine(additional_config={"vllm_gr": config})
    request = _prepare_request(
        engine, [1, 2], BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True), online=True
    )
    native_params = request.items[0].request.sampling_params
    assert native_params is not None
    native_params.max_tokens = length
    request = msgspec.structs.replace(
        request, beam_params=msgspec.structs.replace(request.beam_params, max_tokens=length)
    )
    with pytest.raises(
        ValueError,
        match=f"effective generation length {length} must equal constraint_table_v1 depth",
    ):
        validate_gr_request(request, config=engine.vllm_config)


@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("steps", [1, 2, 3])  # type: ignore[misc]
@pytest.mark.parametrize("widths", [None, [1, 2, 1]])  # type: ignore[misc]
def test_npu_unconstrained_request_needs_no_constraint_artifact(
    online: bool, steps: int, widths: list[int] | None
) -> None:
    canonical = _canonical_gr_config(enabled=False, backend="npu")
    canonical["constrained_generation"] = False
    # No stale path/digests should be needed in a catalog-free deployment.
    canonical["constraint_table"] = {"enabled": False, "backend": "npu", "max_top_k": 4}
    engine = _make_engine(additional_config={"vllm_gr": canonical})
    engine.vllm_config.device_config = SimpleNamespace(device="npu")

    request = _prepare_request(
        engine,
        [1, 2],
        BeamSearchParams(
            beam_width=2,
            max_tokens=steps,
            ignore_eos=True,
            dynamic_beam_widths=None if widths is None else widths[:steps],
            result_width=1,
        ),
        online=online,
    )

    assert request.beam_params.max_tokens == steps
    assert request.beam_params.dynamic_beam_widths == (
        None if widths is None else tuple(widths[:steps])
    )
    assert request.output_options.num_return_sequences == 1
    assert request.items[0].request.sampling_params.logprobs == 2


@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "steps,accepted", [(4, True), (5, True), (6, True), (7, False)]
)
def test_npu_unconstrained_request_enforces_supported_generation_length(
    online: bool, steps: int, accepted: bool
) -> None:
    canonical = _canonical_gr_config(enabled=False, backend="npu")
    canonical["constrained_generation"] = False
    canonical["beam"]["max_decode_steps"] = 6
    engine = _make_engine(additional_config={"vllm_gr": canonical})
    engine.vllm_config.device_config = SimpleNamespace(device="npu")
    engine.vllm_config.model_config.get_num_kv_heads = lambda parallel: 8
    engine.vllm_config.model_config.get_head_size = lambda: 128
    process_inputs = Mock(wraps=engine.input_processor.process_inputs)
    engine.input_processor.process_inputs = process_inputs

    def prepare() -> BeamBatchRequest:
        return _prepare_request(
            engine,
            [1, 2],
            BeamSearchParams(beam_width=2, max_tokens=steps, ignore_eos=True),
            online=online,
        )

    if accepted:
        assert prepare().beam_params.max_tokens == steps
    else:
        with pytest.raises(
            BeamSearchRequestError, match=r"^Decode steps 7 exceed runtime limit 6$"
        ):
            prepare()
        process_inputs.assert_not_called()

    engine.submit_gr_session.assert_not_called()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
def test_npu_extended_capacity_rejects_unsafe_kv_geometry_before_submission(online: bool) -> None:
    canonical = _canonical_gr_config(enabled=False, backend="npu")
    canonical["constrained_generation"] = False
    canonical["beam"]["max_decode_steps"] = 6
    engine = _make_engine(additional_config={"vllm_gr": canonical})
    engine.vllm_config.device_config = SimpleNamespace(device="npu")
    engine.vllm_config.model_config.get_num_kv_heads = lambda parallel: 64
    engine.vllm_config.model_config.get_head_size = lambda: 128

    with pytest.raises(BeamSearchRequestError, match="SelectUnsharedKV 95 KiB buffer"):
        _prepare_request(
            engine,
            [1, 2],
            BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True),
            online=online,
        )
    engine.submit_gr_session.assert_not_called()


@pytest.mark.parametrize("online", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "candidate_width,widths,accepted",
    [(2, None, True), (2, [2, 2, 2], True), (4, None, False), (2, [1, 2, 1], False)],
)
def test_npu_unconstrained_candidate_capacity_matches_retained_stage_widths(
    online: bool, candidate_width: int, widths: list[int] | None, accepted: bool
) -> None:
    canonical = _canonical_gr_config(enabled=False, backend="npu")
    canonical["constrained_generation"] = False
    canonical["constraint_table"]["candidate_width"] = candidate_width
    engine = _make_engine(additional_config={"vllm_gr": canonical})
    engine.vllm_config.device_config = SimpleNamespace(device="npu")
    params = BeamSearchParams(
        beam_width=2, max_tokens=3, ignore_eos=True, dynamic_beam_widths=widths
    )
    if accepted:
        request = _prepare_request(engine, [1, 2], params, online=online)
        assert request.beam_params.candidate_width == candidate_width
    else:
        with pytest.raises(BeamSearchRequestError, match="candidate_width.*beam width"):
            _prepare_request(engine, [1, 2], params, online=online)
        engine.submit_gr_session.assert_not_called()


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
    max_tokens = 3 + int(end_token is not None)
    if not ignore_eos:
        with pytest.raises(BeamSearchRequestError, match="ignore_eos=True"):
            _prepare_request(
                engine,
                [10, 11],
                BeamSearchParams(
                    beam_width=2, max_tokens=max_tokens, end_token=end_token, ignore_eos=False
                ),
                online=online,
            )
        return
    request = _prepare_request(
        engine,
        [10, 11],
        BeamSearchParams(
            beam_width=2, max_tokens=max_tokens, end_token=end_token, ignore_eos=ignore_eos
        ),
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
            BeamSearchParams(
                beam_width=2,
                max_tokens=3 + int(begin_token is not None),
                begin_token=begin_token,
                ignore_eos=True,
            ),
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


pytestmark = pytest.mark.cpu_test
