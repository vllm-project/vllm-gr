# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert offline and online inputs into validated native beam session requests."""

from __future__ import annotations

import time
from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.config import BeamCandidateMode, GRConfig
from vllm_gr.config.models import ConstraintTableBackendEnum
from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchRequestError
from vllm_gr.entrypoints.beam_search_v1.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.entrypoints.beam_search_v1.token_utils import boundary_token_id
from vllm_gr.v1.beam.batch_contracts import (
    DEFAULT_CONSTRAINT_RESOURCE_ID,
    BeamBatchItem,
    BeamBatchParams,
    BeamBatchRequest,
    BeamOutputOptions,
    beam_batch_protocol_version,
)


def _generate_request_id(external_request_id: str | None = None) -> str:
    """Create a unique internal request/session ID for one run."""
    prefix = external_request_id.strip() if external_request_id else "offline"
    if not prefix:
        raise BeamSearchRequestError("external request_id must not be empty")
    return f"{prefix}-gr-beam-v1-{random_uuid()}"


def _get_token_id(tokenizer: Any, token: str | None, name: str) -> int | None:
    try:
        return boundary_token_id(tokenizer, token, name)
    except ValueError as exc:
        raise BeamSearchRequestError(str(exc)) from exc


def _create_request_params(
    params: Any,
    tokenizer: Any,
    candidate_mode: BeamCandidateMode = BeamCandidateMode.CONSTRAINT_TABLE,
) -> tuple[SamplingParams, int | None, int | None]:
    if candidate_mode == BeamCandidateMode.FULL_VOCAB:
        if (
            params.temperature != 0
            or params.ignore_eos is not False
            or params.length_penalty != 1.0
        ):
            raise BeamSearchRequestError(
                "full_vocab beam_search_v1 requires temperature=0, "
                "ignore_eos=False and length_penalty=1"
            )
        if (
            getattr(params, "begin_token", None) is not None
            or getattr(params, "end_token", None) is not None
        ):
            raise BeamSearchRequestError(
                "full_vocab requires begin_token/end_token to be omitted; "
                "put the SID begin token in the prompt and use max_tokens for new tokens"
            )
    elif params.ignore_eos is not True:
        raise BeamSearchRequestError("beam_search_v1 requires ignore_eos=True")
    begin_token_id = _get_token_id(tokenizer, getattr(params, "begin_token", None), "begin_token")
    end_token_id = _get_token_id(tokenizer, getattr(params, "end_token", None), "end_token")
    reserved_tokens = int(begin_token_id is not None) + int(end_token_id is not None)
    decode_steps = params.max_tokens - reserved_tokens
    if decode_steps <= 0:
        raise BeamSearchRequestError(
            "max_tokens must leave at least one Decode step after begin/end tokens"
        )

    stop_token_id = end_token_id
    if stop_token_id is None:
        stop_token_id = getattr(tokenizer, "eos_token_id", None)
    stop_token_ids = () if stop_token_id is None else (int(stop_token_id),)
    if params.ignore_eos or candidate_mode == BeamCandidateMode.FULL_VOCAB:
        stop_token_ids = ()
    sampling_params = SamplingParams(
        logprobs=params.beam_width,
        max_tokens=decode_steps,
        temperature=params.temperature,
        # Native requests advance on accounting placeholders, not the selected
        # Beam token. Only Worker Beam state may terminate full-vocabulary EOS.
        ignore_eos=(candidate_mode == BeamCandidateMode.FULL_VOCAB or params.ignore_eos),
        stop_token_ids=list(stop_token_ids),
        detokenize=False,
        flat_logprobs=True,
        skip_clone=True,
    )
    return sampling_params, begin_token_id, end_token_id


def _prepare_prompt(request: EngineCoreRequest, begin_token_id: int | None) -> None:
    # Mixed inputs also carry token IDs. Reject embeddings before changing IDs
    # so their per-position mask and embedding rows cannot become misaligned.
    if request.prompt_embeds is not None or request.prompt_token_ids is None:
        raise BeamSearchRequestError("embedding prompts are not supported by beam_search_v1")
    if begin_token_id is not None:
        request.prompt_token_ids = [*request.prompt_token_ids, begin_token_id]


def _validate_request(request: BeamBatchRequest, vllm_config: Any) -> None:
    from vllm_gr.v1.engine.beam_search_v1 import (
        validate_gr_device_config,
        validate_gr_execution_config,
        validate_gr_request,
    )

    try:
        validate_gr_execution_config(vllm_config)
        validate_gr_device_config(vllm_config)
        validate_gr_request(request, vllm_config)
    except ValueError as error:
        raise BeamSearchRequestError(str(error)) from error
    # Shared validation above owns request/config semantics. Only frontend
    # limits and the configured Constraint Table remain to check here.
    gr_config = GRConfig.from_vllm_config(vllm_config)
    table = gr_config.constraint_table
    device = getattr(getattr(vllm_config, "device_config", None), "device", "cuda")
    device_type = str(device).split(":")[0]
    required_backend = (
        ConstraintTableBackendEnum.NPU if device_type == "npu" else ConstraintTableBackendEnum.CUDA
    )
    if gr_config.beam_candidate_mode == BeamCandidateMode.CONSTRAINT_TABLE and (
        table is None or not table.enabled or table.backend is not required_backend
    ):
        raise BeamSearchRequestError(
            "beam_search_v1 requires an enabled canonical "
            f"{required_backend.value.upper()} constraint table"
        )
    max_width = gr_config.beam_max_width
    if max_width is None:
        raise BeamSearchRequestError(
            "beam_search_v1 requires vllm-gr beam_max_width to be configured"
        )
    if request.beam_params.beam_width != max_width:
        raise BeamSearchRequestError(
            f"Asynchronous scheduling beam_width must equal the fixed runtime width {max_width}"
        )
    if request.beam_params.max_tokens > gr_config.beam_max_decode_steps:
        raise BeamSearchRequestError(
            f"Decode steps {request.beam_params.max_tokens} exceed runtime limit "
            f"{gr_config.beam_max_decode_steps}"
        )

    model_config = vllm_config.model_config
    prompt_lengths = []
    for item in request.items:
        prompt_tokens = item.request.prompt_token_ids
        if prompt_tokens is None:
            raise BeamSearchRequestError("embedding prompts are not supported by beam_search_v1")
        prompt_lengths.append(len(prompt_tokens))
        total_length = len(prompt_tokens) + request.beam_params.max_tokens
        if total_length > model_config.max_model_len:
            raise BeamSearchRequestError(
                f"prompt plus Decode length {total_length} exceeds max_model_len "
                f"{model_config.max_model_len}"
            )
    scheduler_config = vllm_config.scheduler_config
    if len(request.items) > getattr(scheduler_config, "max_num_seqs", len(request.items)):
        raise BeamSearchRequestError("Beam batch size exceeds max_num_seqs")
    total_prompt_tokens = sum(prompt_lengths)
    if total_prompt_tokens > scheduler_config.max_num_batched_tokens:
        raise BeamSearchRequestError(
            f"Prefill length {total_prompt_tokens} exceeds max_num_batched_tokens "
            f"{scheduler_config.max_num_batched_tokens}"
        )


def _create_batch_request(
    *,
    session_id: str,
    engine_requests: tuple[EngineCoreRequest, ...],
    params: Any,
    vllm_config: Any,
    tokenizer_eos_token_id: int | None = None,
) -> BeamBatchRequest:
    if not engine_requests or engine_requests[0].request_id != session_id:
        raise BeamSearchRequestError(
            "the first EngineCoreRequest.request_id must equal the GR session ID"
        )
    engine_request = engine_requests[0]
    sampling_params = engine_request.sampling_params
    if sampling_params is None or sampling_params.max_tokens is None:
        raise BeamSearchRequestError("beam_search_v1 requires sampling parameters with max_tokens")
    candidate_mode = GRConfig.from_vllm_config(vllm_config).beam_candidate_mode
    if candidate_mode == BeamCandidateMode.FULL_VOCAB:
        # The official offline Beam path stops only at the tokenizer EOS.
        # Native bookkeeping consumes placeholder tokens, so it must ignore all
        # stops (including EOS=0). Worker Beam state owns logical termination.
        for item in engine_requests:
            native_params = item.sampling_params
            assert native_params is not None
            native_params.stop_token_ids = []
            native_params.all_stop_token_ids.clear()
        stop_token_ids = set(() if tokenizer_eos_token_id is None else (tokenizer_eos_token_id,))
    else:
        # Table mode retains the native generation_config merge.
        stop_token_ids = set(sampling_params.stop_token_ids or ())
        if sampling_params.eos_token_id is not None:
            stop_token_ids.add(sampling_params.eos_token_id)
    batch_params = BeamBatchParams(
        beam_width=params.beam_width,
        max_tokens=sampling_params.max_tokens,
        temperature=sampling_params.temperature,
        stop_token_ids=tuple(sorted(stop_token_ids)),
        ignore_stop_tokens=params.ignore_eos,
        constraint_resource_id=(
            None
            if candidate_mode == BeamCandidateMode.FULL_VOCAB
            else DEFAULT_CONSTRAINT_RESOURCE_ID
        ),
        length_penalty=params.length_penalty,
        catalog_strategy=getattr(params, "catalog_strategy", None),
        candidate_mode=candidate_mode,
    )
    output_options = BeamOutputOptions(
        num_return_sequences=batch_params.beam_width,
        # Native offline Beam returns generated token IDs including EOS; the
        # include_stop_str_in_output flag concerns text, not this raw suffix.
        include_stop_token=(
            candidate_mode == BeamCandidateMode.FULL_VOCAB
            or bool(getattr(params, "include_stop_str_in_output", False))
        ),
        include_cumulative_logprob=True,
        # Public offline/online outputs expose sequences and scores only.
        # Beam execution still retains its complete parent/KV history.
        include_parent_beam_indices=False,
    )
    request = BeamBatchRequest(
        protocol_version=beam_batch_protocol_version(batch_params),
        batch_id=session_id,
        items=tuple(
            BeamBatchItem(item_index=index, request=request)
            for index, request in enumerate(engine_requests)
        ),
        beam_params=batch_params,
        output_options=output_options,
    )
    _validate_request(request, vllm_config)
    return request


class BeamSearchInputProcessor:
    """Prepare requests using the engine's native InputProcessor and limits."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.boundary_token_ids: tuple[int | None, int | None] = (None, None)

    def process_offline_inputs(self, llm: Any, call: OfflineBeamSearchRequest) -> BeamBatchRequest:
        if not call.prompts:
            raise BeamSearchRequestError("beam_search_v1 requires at least one prompt")
        if call.concurrency_limit is not None and call.concurrency_limit <= 0:
            raise BeamSearchRequestError("concurrency_limit must be positive or None")
        tokenizer = llm.get_tokenizer()
        sampling_params, begin_token_id, end_token_id = _create_request_params(
            call.params,
            tokenizer,
            GRConfig.from_vllm_config(self.engine.vllm_config).beam_candidate_mode,
        )
        engine_inputs = llm._preprocess_cmpl(call.prompts)
        lora_requests = llm._lora_request_to_seq(call.lora_request, len(engine_inputs))
        engine_requests = []
        arrival_time = time.time()
        supported_tasks = self.engine.get_supported_tasks()
        for engine_input, lora_request in zip(engine_inputs, lora_requests, strict=True):
            engine_request = self.engine.input_processor.process_inputs(
                _generate_request_id(),
                engine_input,
                sampling_params,
                supported_tasks=supported_tasks,
                arrival_time=arrival_time,
                lora_request=lora_request,
            )
            _prepare_prompt(engine_request, begin_token_id)
            self.engine.input_processor.assign_request_id(engine_request)
            engine_requests.append(engine_request)
        session_id = engine_requests[0].request_id
        request = _create_batch_request(
            session_id=session_id,
            engine_requests=tuple(engine_requests),
            params=call.params,
            vllm_config=self.engine.vllm_config,
            tokenizer_eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        self.boundary_token_ids = (begin_token_id, end_token_id)
        return request

    async def process_online_inputs(
        self, serving: Any, call: OnlineBeamSearchRequest
    ) -> BeamBatchRequest:
        tokenizer = serving.input_processor.tokenizer
        if tokenizer is None:
            raise VLLMValidationError(
                "beam_search_v1 requires tokenizer initialization",
                parameter="skip_tokenizer_init",
                value=True,
            )
        sampling_params, begin_token_id, end_token_id = _create_request_params(
            call.params,
            tokenizer,
            GRConfig.from_vllm_config(self.engine.vllm_config).beam_candidate_mode,
        )
        request_id = _generate_request_id(call.request_id)
        engine_request = self.engine.input_processor.process_inputs(
            request_id,
            call.prompt,
            sampling_params,
            supported_tasks=await self.engine.get_supported_tasks(),
            arrival_time=time.time(),
            lora_request=call.lora_request,
            trace_headers=call.trace_headers,
            priority=call.priority,
            data_parallel_rank=call.data_parallel_rank,
        )
        _prepare_prompt(engine_request, begin_token_id)
        self.engine.input_processor.assign_request_id(engine_request)
        # Native assignment copies our generated ID into external_req_id.
        # Preserve the caller's original identity for output and abort routing.
        engine_request.external_req_id = call.request_id
        session_id = engine_request.request_id
        request = _create_batch_request(
            session_id=session_id,
            engine_requests=(engine_request,),
            params=call.params,
            vllm_config=self.engine.vllm_config,
            tokenizer_eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        self.boundary_token_ids = (begin_token_id, end_token_id)
        return request
