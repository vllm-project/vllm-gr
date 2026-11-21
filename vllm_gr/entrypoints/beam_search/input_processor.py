# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Convert offline and online inputs into validated native beam session requests."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.config import GRConfig, resolve_candidate_width
from vllm_gr.config.models import ConstraintTableBackendEnum, configured_candidate_width
from vllm_gr.entrypoints.beam_search.boundary_tokens import boundary_token_id
from vllm_gr.entrypoints.beam_search.exceptions import BeamSearchRequestError
from vllm_gr.entrypoints.beam_search.protocol import (
    OfflineBeamSearchRequest,
    OnlineBeamSearchRequest,
)
from vllm_gr.entrypoints.beam_search.request_params import (
    apply_beam_request_defaults,
)
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    DEFAULT_CONSTRAINT_RESOURCE_ID,
    BeamBatchItem,
    BeamBatchParams,
    BeamBatchRequest,
    BeamOutputOptions,
    to_int64_tokens,
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


def _resolve_candidate_width(vllm_config: Any, params: Any) -> int:
    """Resolve K for one request, matching the engine's startup derivation.

    ``constraint_table.candidate_width`` is the single source of truth; when it
    is unset K follows the beam width (the pre-decoupling behaviour).  Sampling
    ``logprobs`` must equal K so each beam row yields exactly K candidates.
    """

    config = GRConfig.from_vllm_config(vllm_config)
    if config.beam_max_width is None:
        raise BeamSearchRequestError("beam_search requires beam.max_width to be configured")
    candidate_width = resolve_candidate_width(config)
    max_logprobs = _effective_max_logprobs(vllm_config)
    if max_logprobs is not None and candidate_width > max_logprobs:
        raise BeamSearchRequestError(
            f"candidate width {candidate_width} exceeds max_logprobs {max_logprobs}; "
            "raise --max-logprobs to at least the candidate width"
        )
    return candidate_width


def _configured_candidate_width(vllm_config: Any) -> int | None:
    """Return the explicitly configured K, or ``None`` when K follows W."""

    config = GRConfig.from_vllm_config(vllm_config)
    device = getattr(getattr(vllm_config, "device_config", None), "device", None)
    if (
        str(device).split(":")[0] == "npu"
        and config.deployment.candidate_width is not None
        and config.deployment.candidate_width == config.deployment.beam_width
    ):
        # NPU derives K from each stage's retained width. Do not serialize
        # the workspace capacity as a fixed request override for dynamic W.
        return None
    return configured_candidate_width(config)


def _effective_max_logprobs(vllm_config: Any) -> int | None:
    """Return the served logprob budget, matching vLLM's own normalization.

    ``max_logprobs`` is ``-1`` by default, which vLLM's ``SamplingParams``
    validation treats as "the whole vocabulary", not as a negative limit.
    Comparing a positive candidate width against the raw sentinel would reject
    every default deployment, so the sentinel is normalized the same way here.
    """

    model_config = getattr(vllm_config, "model_config", None)
    max_logprobs = getattr(model_config, "max_logprobs", None)
    if not isinstance(max_logprobs, int) or isinstance(max_logprobs, bool):
        return None
    if max_logprobs == -1:
        get_vocab_size = getattr(model_config, "get_vocab_size", None)
        return int(get_vocab_size()) if callable(get_vocab_size) else None
    return max_logprobs


def _create_request_params(
    params: Any,
    tokenizer: Any,
    *,
    vllm_config: Any,
) -> tuple[SamplingParams, int | None, int | None]:
    if params.ignore_eos is not True:
        raise BeamSearchRequestError("beam_search requires ignore_eos=True")
    begin_token_id = _get_token_id(tokenizer, getattr(params, "begin_token", None), "begin_token")
    end_token_id = _get_token_id(tokenizer, getattr(params, "end_token", None), "end_token")
    reserved_tokens = int(begin_token_id is not None) + int(end_token_id is not None)
    decode_steps = params.max_tokens - reserved_tokens
    if decode_steps <= 0:
        raise BeamSearchRequestError(
            "max_tokens must leave at least one Decode step after begin/end tokens"
        )
    from vllm_gr.v1.engine.beam_admission import validate_gr_generation_budget

    try:
        validate_gr_generation_budget(GRConfig.from_vllm_config(vllm_config), decode_steps)
    except ValueError as error:
        raise BeamSearchRequestError(str(error)) from error

    stop_token_id = end_token_id
    if stop_token_id is None:
        stop_token_id = getattr(tokenizer, "eos_token_id", None)
    stop_token_ids = () if stop_token_id is None else (int(stop_token_id),)
    if params.ignore_eos:
        stop_token_ids = ()
    sampling_params = SamplingParams(
        logprobs=_resolve_candidate_width(vllm_config, params),
        max_tokens=decode_steps,
        temperature=params.temperature,
        ignore_eos=params.ignore_eos,
        stop_token_ids=list(stop_token_ids),
        detokenize=False,
        flat_logprobs=True,
        skip_clone=True,
    )
    return sampling_params, begin_token_id, end_token_id


def _prepare_prompt(request: EngineCoreRequest, begin_token_id: int | None) -> list[int]:
    # Mixed inputs also carry token IDs. Reject embeddings before changing IDs
    # so their per-position mask and embedding rows cannot become misaligned.
    if request.prompt_embeds is not None or request.prompt_token_ids is None:
        raise BeamSearchRequestError("embedding prompts are not supported by beam_search")
    original = request.prompt_token_ids
    if begin_token_id is None:
        # The batch contract normalizes the sequence exactly once; copying it
        # here only to convert it later would unbox every token twice.
        return original
    tokens = to_int64_tokens(original)
    owned = np.empty(tokens.size + 1, dtype=np.int64)
    owned[:-1] = tokens
    owned[-1] = begin_token_id
    request.prompt_token_ids = owned
    return original


def _validate_request(request: BeamBatchRequest, vllm_config: Any) -> None:
    from vllm_gr.v1.engine.beam_admission import (
        validate_gr_device_config,
        validate_gr_execution_config,
        validate_gr_request,
    )

    try:
        validate_gr_execution_config(vllm_config)
        validate_gr_device_config(vllm_config, request)
        validate_gr_request(request, config=vllm_config)
    except ValueError as error:
        raise BeamSearchRequestError(str(error)) from error
    # Shared validation above owns request/config semantics. Only frontend
    # limits and the configured Constraint Table remain to check here.
    from vllm_gr.config.models import ConstraintMode, configured_constraint_mode

    gr_config = GRConfig.from_vllm_config(vllm_config)
    table = gr_config.constraint_table
    unconstrained = configured_constraint_mode(gr_config) is ConstraintMode.NONE
    device = getattr(getattr(vllm_config, "device_config", None), "device", "cuda")
    device_type = str(device).split(":")[0]
    if unconstrained and device_type not in ("cuda", "npu"):
        raise BeamSearchRequestError("unconstrained beam_search requires CUDA or NPU")
    required_backend = (
        ConstraintTableBackendEnum.NPU if device_type == "npu" else ConstraintTableBackendEnum.CUDA
    )
    if not unconstrained and (
        table is None or not table.enabled or table.backend is not required_backend
    ):
        raise BeamSearchRequestError(
            "beam_search requires an enabled canonical "
            f"{required_backend.value.upper()} constraint table"
        )
    if not unconstrained and request.beam_params.catalog_strategy is not None:
        raise BeamSearchRequestError(
            "catalog_strategy is not applicable to the constraint_table backend"
        )
    max_width = gr_config.beam_max_width
    if max_width is None:
        raise BeamSearchRequestError("beam_search requires vllm-gr beam_max_width to be configured")
    if request.beam_params.beam_width != max_width:
        raise BeamSearchRequestError(
            f"Asynchronous scheduling beam_width must equal the fixed runtime width {max_width}"
        )
    # K is also an engine-lifetime constant: the Worker prewarms exactly one
    # postprocess workspace and looks it up by key, so a request carrying a
    # different candidate width would surface later as a "geometry changed"
    # error instead of a request-level rejection.
    candidate_width = resolve_candidate_width(gr_config)
    carried_width = request.beam_params.candidate_width
    if carried_width is not None and carried_width != candidate_width:
        raise BeamSearchRequestError(
            f"Beam batch candidate_width {carried_width} must equal the configured "
            f"candidate width {candidate_width}"
        )

    model_config = vllm_config.model_config
    prompt_lengths = []
    for item in request.items:
        prompt_tokens = item.request.prompt_token_ids
        if prompt_tokens is None:
            raise BeamSearchRequestError("embedding prompts are not supported by beam_search")
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
    trie_ids: list[int] | tuple[int, ...] | None = None,
) -> BeamBatchRequest:
    if trie_ids is None:
        trie_ids = (0,) * len(engine_requests)
    if (
        not isinstance(trie_ids, (list, tuple))
        or len(trie_ids) != len(engine_requests)
        or any(type(trie_id) is not int or trie_id < 0 for trie_id in trie_ids)
    ):
        raise BeamSearchRequestError("trie_ids must contain one non-negative integer per prompt")
    if not engine_requests or engine_requests[0].request_id != session_id:
        raise BeamSearchRequestError(
            "the first EngineCoreRequest.request_id must equal the GR session ID"
        )
    engine_request = engine_requests[0]
    sampling_params = engine_request.sampling_params
    if sampling_params is None or sampling_params.max_tokens is None:
        raise BeamSearchRequestError("beam_search requires sampling parameters with max_tokens")
    dynamic_widths = getattr(params, "dynamic_beam_widths", None)
    per_parent_beam_widths = getattr(params, "per_parent_beam_widths", None)
    result_width = getattr(params, "result_width", None)
    device = getattr(getattr(vllm_config, "device_config", None), "device", "cuda")
    if str(device).split(":")[0] not in ("cuda", "npu") and (
        dynamic_widths is not None or result_width is not None
    ):
        raise BeamSearchRequestError(
            "dynamic_beam_widths and result_width require CUDA or NPU beam_search"
        )
    if dynamic_widths is not None and (
        not isinstance(dynamic_widths, list)
        or len(dynamic_widths) != sampling_params.max_tokens
        or any(
            type(width) is not int or not 1 <= width <= params.beam_width
            for width in dynamic_widths
        )
    ):
        raise BeamSearchRequestError(
            "dynamic_beam_widths must contain one integer in [1, beam_width] per effective step"
        )
    # Native processing merges generation_config EOS IDs and stores the primary
    # EOS separately. Beam metadata must describe the same effective stop set.
    stop_token_ids = set(sampling_params.stop_token_ids or ())
    if sampling_params.eos_token_id is not None:
        stop_token_ids.add(sampling_params.eos_token_id)
    # K (and its logprob budget) was validated when the sampling parameters
    # were built.  Put it on the wire only when it is *configured*: an unset K
    # must stay ``None`` so the execution view keeps taking each stage's top-k
    # from its output width, as it did before K existed.  Writing the derived
    # ``K == W`` here would make every stage of a shrinking dynamic schedule
    # select ``W`` candidates.
    candidate_width = _configured_candidate_width(vllm_config)
    batch_params = BeamBatchParams(
        beam_width=params.beam_width,
        max_tokens=sampling_params.max_tokens,
        temperature=sampling_params.temperature,
        stop_token_ids=tuple(sorted(stop_token_ids)),
        ignore_stop_tokens=sampling_params.ignore_eos,
        constraint_resource_id=DEFAULT_CONSTRAINT_RESOURCE_ID,
        length_penalty=params.length_penalty,
        catalog_strategy=getattr(params, "catalog_strategy", None),
        dynamic_beam_widths=None if dynamic_widths is None else tuple(dynamic_widths),
        per_parent_beam_widths=None
        if per_parent_beam_widths is None
        else tuple(per_parent_beam_widths),
        candidate_width=candidate_width,
    )
    final_width = (
        batch_params.beam_width
        if batch_params.dynamic_beam_widths is None
        else batch_params.dynamic_beam_widths[-1]
    )
    output_options = BeamOutputOptions(
        num_return_sequences=final_width if result_width is None else result_width,
        include_stop_token=bool(getattr(params, "include_stop_str_in_output", False)),
        include_cumulative_logprob=True,
        # Public offline/online outputs expose sequences and scores only.
        # Optional output parent lineage is separate from the required
        # Beam decision and KV state, which execution still retains.
        include_parent_beam_indices=False,
    )
    request = BeamBatchRequest(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=session_id,
        items=tuple(
            BeamBatchItem(item_index=index, request=request, trie_id=trie_ids[index])
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
        self.original_prompt_token_ids: tuple[tuple[int, ...], ...] | None = None

    def process_offline_inputs(self, llm: Any, call: OfflineBeamSearchRequest) -> BeamBatchRequest:
        if not call.prompts:
            raise BeamSearchRequestError("beam_search requires at least one prompt")
        if call.concurrency_limit is not None and call.concurrency_limit <= 0:
            raise BeamSearchRequestError("concurrency_limit must be positive or None")
        tokenizer = llm.get_tokenizer()
        params = apply_beam_request_defaults(self.engine.vllm_config, call.params)
        sampling_params, begin_token_id, end_token_id = _create_request_params(
            params, tokenizer, vllm_config=self.engine.vllm_config
        )
        engine_inputs = llm._preprocess_cmpl(call.prompts)
        lora_requests = llm._lora_request_to_seq(call.lora_request, len(engine_inputs))
        engine_requests = []
        original_prompts = []
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
            original = _prepare_prompt(engine_request, begin_token_id)
            if call.output_mode == "compact":
                original_prompts.append(tuple(original))
            self.engine.input_processor.assign_request_id(engine_request)
            engine_requests.append(engine_request)
        session_id = engine_requests[0].request_id
        request = _create_batch_request(
            session_id=session_id,
            engine_requests=tuple(engine_requests),
            params=params,
            vllm_config=self.engine.vllm_config,
            trie_ids=call.trie_ids,
        )
        self.boundary_token_ids = (begin_token_id, end_token_id)
        self.original_prompt_token_ids = (
            tuple(original_prompts) if call.output_mode == "compact" else None
        )
        return request

    async def process_online_inputs(
        self, serving: Any, call: OnlineBeamSearchRequest
    ) -> BeamBatchRequest:
        tokenizer = serving.input_processor.tokenizer
        if tokenizer is None:
            raise VLLMValidationError(
                "beam_search requires tokenizer initialization",
                parameter="skip_tokenizer_init",
                value=True,
            )
        if not call.prompts:
            raise VLLMValidationError("beam_search requires at least one prompt")
        params = apply_beam_request_defaults(self.engine.vllm_config, call.params)
        sampling_params, begin_token_id, end_token_id = _create_request_params(
            params, tokenizer, vllm_config=self.engine.vllm_config
        )
        # One arrival time and one task set describe the batch, not its
        # members; asking per prompt would spread a single submission over the
        # arrival clock and repeat an engine round trip k times.
        supported_tasks = await self.engine.get_supported_tasks()
        arrival_time = time.time()
        engine_requests = []
        for prompt in call.prompts:
            engine_request = self.engine.input_processor.process_inputs(
                _generate_request_id(call.request_id),
                prompt,
                sampling_params,
                supported_tasks=supported_tasks,
                arrival_time=arrival_time,
                lora_request=call.lora_request,
                trace_headers=call.trace_headers,
                priority=call.priority,
                data_parallel_rank=call.data_parallel_rank,
                session_id=call.session_id,
            )
            _prepare_prompt(engine_request, begin_token_id)
            self.engine.input_processor.assign_request_id(engine_request)
            # Native assignment copies our generated ID into external_req_id.
            # Preserve the caller's original identity for output and abort routing.
            engine_request.external_req_id = call.request_id
            engine_requests.append(engine_request)
        session_id = engine_requests[0].request_id
        request = _create_batch_request(
            session_id=session_id,
            engine_requests=tuple(engine_requests),
            params=params,
            vllm_config=self.engine.vllm_config,
            trie_ids=call.trie_ids,
        )
        self.boundary_token_ids = (begin_token_id, end_token_id)
        return request
