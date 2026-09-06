# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from collections.abc import AsyncGenerator, Mapping

from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.exceptions import VLLMValidationError
from vllm.inputs import EngineInput, tokens_input
from vllm.logger import init_logger
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.utils import random_uuid

from vllm_gr.beam_search_decision_utils import (
    complete_eos_candidates,
    configure_persistent_sampling_params,
    extract_and_dedup_flat_logprobs,
    materialize_final_beams,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.config import MAX_BEAM_WIDTH, GRConfig
from vllm_gr.constraints import BeamConstraintCoordinator, ParentCandidateBuilder
from vllm_gr.constraints.strategies import (
    ConstraintTableStrategy,
    validate_constraint_generation_limits,
    validate_constraint_table_protocol,
)
from vllm_gr.logprobs import reconstruct_beam_logprobs
from vllm_gr.v1.engine.types import (
    GR_REQUEST_METADATA_KEY,
    GRRequestMetadata,
    canonicalize_constraint_digest,
)
from vllm_gr.v1.metrics.beam_search_metrics import record_beam_search_metrics
from vllm_gr.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)


def _worker_decision_from_result(beam_result) -> Mapping[str, object] | None:
    """Extract the worker beam-decision dict from a step's RequestOutput."""
    kv_params = getattr(beam_result, "kv_transfer_params", None)
    if not kv_params:
        return None
    decision = kv_params.get("beam_search_result")
    if not isinstance(decision, Mapping):
        return None
    return decision


_dp_rank_counter = 0
_dp_rank_lock = asyncio.Lock()


# return rank for next beam search request in a round-robin manner across the data parallel ranks
async def _next_data_parallel_rank(data_parallel_size: int) -> int:
    global _dp_rank_counter

    async with _dp_rank_lock:
        rank = _dp_rank_counter % data_parallel_size
        _dp_rank_counter += 1
        return rank


# Persistent execution emits only the terminal frontend item. A lost output
# would otherwise leave the request waiting forever.
_BEAM_COLLECT_TIMEOUT_S = 120.0


async def _collect_beam_result(q):
    """Drain output queue until finished (bounded wait)."""
    import asyncio

    while True:
        try:
            out = await asyncio.wait_for(q.get(), timeout=_BEAM_COLLECT_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.error(
                "vllm-gr: beam result collection timed out after %.0fs "
                "(final output never reached the collector)",
                _BEAM_COLLECT_TIMEOUT_S,
            )
            raise
        if out.finished:
            return out


async def _gather_beam_results(queues):
    """Gather results from multiple output queues."""
    tasks = [asyncio.create_task(_collect_beam_result(q)) for q in queues]
    return list(await asyncio.gather(*tasks))


async def _abort_persistent_requests(engine_client, request_ids, cause) -> None:
    """Abort prepared persistent requests and preserve cleanup failures."""
    if not request_ids:
        return
    abort = getattr(engine_client, "abort", None)
    if not callable(abort):
        error = RuntimeError("persistent GR engine client has no abort boundary")
        error.add_note(f"Original persistent GR failure: {cause!r}")
        raise error from cause
    try:
        await abort(request_ids, internal=True)
    except BaseException as cleanup_error:
        cleanup_error.add_note(
            f"Persistent GR abort failed after {cause.__class__.__name__}: {cause}"
        )
        raise cleanup_error from cause


async def _add_batch_step(
    engine_client,
    prompts_batch,
    lora_req_batch,
    request_id_batch,
    beam_search_params,
    trace_headers,
    priority=0,
    data_parallel_rank: int | None = None,
    gr_metadata: GRRequestMetadata | None = None,
):
    """Prepare requests and submit them through vLLM's native ADD protocol."""
    queues = []
    engine_core_requests = []
    persistent_ids = []
    try:
        for i, (individual_prompt, lora_req) in enumerate(
            zip(prompts_batch, lora_req_batch)
        ):
            request_id_item = f"{request_id_batch}-beam-{i}"
            queue, request = await engine_client.prepare_request(
                request_id_item,
                individual_prompt,
                beam_search_params,
                lora_request=lora_req,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
            )
            if gr_metadata is not None:
                persistent_ids.append(request.request_id)
                metadata = {
                    field: getattr(gr_metadata, field)
                    for field in gr_metadata.__struct_fields__
                }
                metadata["request_id"] = request.request_id
                request.sampling_params.extra_args = dict(
                    request.sampling_params.extra_args or {}
                )
                request.sampling_params.extra_args[GR_REQUEST_METADATA_KEY] = metadata
            queues.append(queue)
            engine_core_requests.append(request)
    except BaseException as error:
        await _abort_persistent_requests(engine_client, persistent_ids, error)
        raise

    if persistent_ids:
        request_states = getattr(
            getattr(engine_client, "output_processor", None), "request_states", None
        )
        if request_states is not None:
            for request_id in persistent_ids:
                state = request_states.get(request_id)
                if state is not None:
                    state.gr_persistent_request = True

    try:
        await engine_client._add_requests_batch(engine_core_requests)
        output = await _gather_beam_results(queues)
    except BaseException as error:
        await _abort_persistent_requests(engine_client, persistent_ids, error)
        raise
    return output, persistent_ids


async def beam_search(
    self,
    prompt: EngineInput,
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    generation_time: float = 0.0
    num_generation_tokens: int = 0
    beam_width = params.beam_width
    max_tokens = params.max_tokens
    ignore_eos = params.ignore_eos
    temperature = params.temperature
    begin_token = params.begin_token
    end_token = params.end_token
    catalog_strategy = getattr(params, "catalog_strategy", None)
    constraint_coordinator = getattr(self.models, "constraint_coordinator", None)
    if constraint_coordinator is None:
        constraint_coordinator = BeamConstraintCoordinator.from_catalog(
            getattr(self.models, "catalog", None)
        )

    # Resolve the frontend catalog-strategy per backend (Trie: None → "fill";
    # Constraint Tables: an explicit value is a contradiction and is rejected).
    try:
        catalog_strategy = constraint_coordinator.resolve_catalog_strategy(catalog_strategy)
    except ValueError as exc:
        raise VLLMValidationError(
            str(exc),
            parameter="catalog_strategy",
            value=catalog_strategy,
        ) from exc

    # Constraint Tables generate candidates in the Worker; the decode-step
    # update must carry the constraint_step so the engine can route the raw
    # logits through the constrained top-k operator.
    _is_constrained = constraint_coordinator is not None and isinstance(
        constraint_coordinator.strategy, ConstraintTableStrategy
    )
    gr_config = GRConfig.from_vllm_config(
        getattr(self.engine_client, "vllm_config", None)
    )
    persistent_request = bool(gr_config.persistent_request)
    persistent_runner_state = bool(gr_config.persistent_runner_state)
    if persistent_request and not _is_constrained:
        raise VLLMValidationError(
            "persistent GR requests require constraint_backend='constraint_table'"
        )

    # Constraint Tables defer the collective-RPC broadcast to the first
    # beam_search; ensure workers hold the tables before any constrained step.
    # The method is mounted globally on OpenAIServingModels but only
    # constraint-table instances initialize the constraint fields, so the
    # ``_is_constrained`` gate keeps Trie / no-catalog requests on the fast path.
    ensure_injected = getattr(self.models, "_ensure_constraint_tables_injected", None)
    if _is_constrained and ensure_injected is not None:
        await ensure_injected()

    # All beams from the same logical beam search will share the same priority value
    # (the prefill timestamp), so the scheduler will treat them as a group with equal priority
    MICROSECONDS = 1000000
    priority = int(time.perf_counter() * MICROSECONDS)  # us granularity
    rank = None
    if (
        hasattr(self.engine_client, "vllm_config")
        and self.engine_client.vllm_config.parallel_config is not None
    ):
        data_parallel_size = self.engine_client.vllm_config.parallel_config.data_parallel_size
        if data_parallel_size is not None and data_parallel_size > 1:
            # In DP mode, we assign ranks to beam search requests in a round-robin manner.
            rank = await _next_data_parallel_rank(data_parallel_size)
            logger.debug(
                f"rank for beam search: {rank} out of data_parallel_size: {data_parallel_size}"
            )

    include_stop_str_in_output = params.include_stop_str_in_output
    if (
        isinstance(beam_width, bool)
        or not isinstance(beam_width, int)
        or not 1 <= beam_width <= MAX_BEAM_WIDTH
    ):
        raise VLLMValidationError(
            f"Beam width must be an integer in [1, {MAX_BEAM_WIDTH}]",
            parameter="beam_width",
            value=beam_width,
        )
    input_processor = self.input_processor
    tokenizer = input_processor.tokenizer
    if tokenizer is None:
        raise VLLMValidationError(
            "You cannot use beam search when `skip_tokenizer_init=True`",
            parameter="skip_tokenizer_init",
            value=True,
        )

    eos_token_id: int = tokenizer.eos_token_id
    sid_begin_token_id: int | None = None
    if begin_token is not None:
        sid_begin_token_id = tokenizer.convert_tokens_to_ids(begin_token)
        if sid_begin_token_id is None or sid_begin_token_id == -1:
            raise VLLMValidationError(
                "begin_token must be a valid token in the tokenizer vocabulary",
                parameter="begin_token",
                value=begin_token,
            )
    sid_end_token_id: int | None = None
    if end_token is not None:
        sid_end_token_id = tokenizer.convert_tokens_to_ids(end_token)
        if sid_end_token_id is None or sid_end_token_id == -1:
            raise VLLMValidationError(
                "end_token must be a valid token in the tokenizer vocabulary",
                parameter="end_token",
                value=end_token,
            )

    if not isinstance(prompt, str) and prompt.get("type") == "enc_dec":
        raise NotImplementedError

    prompt_text = prompt if isinstance(prompt, str) else prompt.get("prompt")
    prompt_token_ids = [] if isinstance(prompt, str) else prompt.get("prompt_token_ids", [])
    tokenized_length = len(prompt_token_ids)
    logprobs_num = beam_width

    beam_search_params = SamplingParams(
        logprobs=logprobs_num,
        max_tokens=1,
        temperature=temperature,
        detokenize=False,
        flat_logprobs=True,
    )
    initial_tokens = list(prompt_token_ids)
    # NOTE: FlatLogprobs intentionally passed where BeamSearchSequence
    # expects list[dict[int, Logprob]].  Intermediate logprobs are never
    # read as dicts; the final beam.logprobs is replaced during
    # reconstruction at the end.
    # This reference is shared read-only across all beams — do not mutate
    # after the initial append below.
    initial_logprobs = FlatLogprobs()
    if sid_begin_token_id is not None:
        initial_tokens.append(sid_begin_token_id)
        initial_logprobs.append_fast([sid_begin_token_id], [0.0], [None], [None])

    orig_prompt = prompt
    if isinstance(prompt, str) or "type" not in prompt:
        orig_prompt = tokens_input(initial_tokens, prompt=prompt_text)

    all_beams = [
        BeamSearchSequence(
            orig_prompt=orig_prompt,
            tokens=initial_tokens,
            cum_logprob=0,
            logprobs=initial_logprobs,
            lora_request=lora_request,
        )
    ]
    # Deferred logprobs: instead of copying 6 FlatLogprobs lists per beam
    # per step (O(beams * steps^2)), store parent pointers and reconstruct
    # only for the final best beams at the end.
    # These dynamic attrs live on BeamSearchSequence (a plain @dataclass
    # without __slots__).  If upstream adds __slots__, these will break —
    # upstream the fields if that happens.
    all_beams[0]._lp_parent = None
    all_beams[0]._lp_step_data = None
    completed = []

    beam_search_start: int = time.perf_counter()
    prefill_start = beam_search_start  # Track initial start for prefill measurement
    pre_calc = 0 if sid_begin_token_id is None else 1
    if sid_end_token_id is not None:
        pre_calc += 1
    if max_tokens - pre_calc < 0:
        raise VLLMValidationError(
            f"max tokens should be not lower than {pre_calc} to accommodate reserved begin/end tokens.",
            parameter="max_tokens",
            value=max_tokens,
        )

    if _is_constrained:
        # Constraint Tables are a fixed 3-level FSM; requesting more steps than
        # the Worker can run is a hard error, never a silent fallback.
        try:
            validate_constraint_generation_limits(
                beam_width=beam_width,
                max_generate_steps=max_tokens - pre_calc,
            )
        except ValueError as exc:
            raise VLLMValidationError(str(exc)) from exc

    gr_metadata = None
    if persistent_request:
        decode_steps = max_tokens - pre_calc
        if decode_steps < 1:
            raise VLLMValidationError(
                "persistent GR requests require at least one generated token",
                parameter="max_tokens",
                value=max_tokens,
            )
        configure_persistent_sampling_params(
            beam_search_params,
            generation_steps=decode_steps,
        )
        constraint_digest = getattr(
            getattr(constraint_coordinator, "strategy", None),
            "digest",
            None,
        )
        gr_metadata = GRRequestMetadata(
            request_id=request_id,
            beam_width=beam_width,
            generation_steps=decode_steps,
            prefix_len=len(initial_tokens),
            result_width=beam_width,
            constraint_digest=canonicalize_constraint_digest(constraint_digest),
        )

    validate_constraint_table_protocol(
        is_constraint_table=_is_constrained,
        beam_protocol_available=persistent_request and persistent_runner_state,
        required="the persistent GR ASYNC scheduler",
        hint=(
            "Constraint Tables now use one persistent native request; enable "
            "persistent_request and persistent_runner_state."
        ),
    )

    for token in range(max_tokens - pre_calc):
        if not all_beams:
            break
        if token == 1:
            beam_search_start = time.perf_counter()

        prompts_batch, lora_req_batch = zip(
            *[
                (
                    beam.get_prompt(),
                    beam.lora_request,
                )
                for beam in all_beams
            ]
        )
        if token > 0:
            num_generation_tokens += len(all_beams)
        request_id_batch = f"{request_id}-{random_uuid()}"

        # Launch catalog filtering in parallel with the engine step.
        # Constraint Tables generate candidates in the Worker, so the Trie
        # lookup must not run (ConstraintTableStrategy.valid_next_token_ids
        # raises by contract).
        catalog_task = None
        if constraint_coordinator.enabled and not _is_constrained:

            def get_valid_tokens_set(beam) -> set[int]:
                generated_tokens = beam.tokens[len(prompt_token_ids) :]
                valid_token_ids = constraint_coordinator.valid_next_token_ids(generated_tokens)
                assert valid_token_ids is not None
                return valid_token_ids

            async def _run_catalog(coros):
                return list(await asyncio.gather(*coros))

            catalog_task = asyncio.create_task(
                _run_catalog([asyncio.to_thread(get_valid_tokens_set, beam) for beam in all_beams])
            )

        gen_start = time.perf_counter()

        output, _ = await _add_batch_step(
            self.engine_client,
            prompts_batch,
            lora_req_batch,
            request_id_batch,
            beam_search_params,
            trace_headers,
            priority=priority,
            data_parallel_rank=rank,
            gr_metadata=gr_metadata,
        )

        if persistent_runner_state:
            final_decision = (
                _worker_decision_from_result(output[0])
                if len(output) == 1
                else None
            )
            if final_decision is None:
                raise RuntimeError("persistent GR final output is missing")
            from vllm_gr.v1.worker.gr_model_runner_state import (
                unpack_gr_final_output,
            )

            final_output = unpack_gr_final_output(final_decision)
            if final_output is None:
                raise RuntimeError("persistent GR final output is invalid")
            sequences, _, selected_logprobs, scores = final_output
            all_beams = materialize_final_beams(
                all_beams[0], sequences, selected_logprobs, scores
            )
            break

        valid_tokens_sets = None
        if catalog_task is not None:
            valid_tokens_sets = await catalog_task
        if token > 0 or persistent_runner_state:
            generation_time += time.perf_counter() - gen_start
        candidate_builder = ParentCandidateBuilder()
        for i, result in enumerate(output):
            current_beam = all_beams[i]
            if result.outputs[0].finish_reason == "error":
                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt_text,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="",
                            token_ids=[],
                            cumulative_logprob=None,
                            logprobs=None,
                            finish_reason="error",
                        )
                    ],
                    finished=True,
                    prompt_token_ids=prompt_token_ids,
                    prompt_logprobs=None,
                )
                return

            flat = result.outputs[0].logprobs
            if flat is None:
                constraint_coordinator.add_parent_candidates(
                    candidate_builder,
                    parent_index=i,
                    parent_cumulative_logprob=current_beam.cum_logprob,
                    token_ids=[],
                    logprobs=[],
                    ranks=None,
                    decoded_tokens=None,
                    valid_token_ids=None,
                    candidate_strategy=catalog_strategy,
                )
                continue
            token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                extract_and_dedup_flat_logprobs(flat, logprobs_num)
            )
            constraint_coordinator.add_parent_candidates(
                candidate_builder,
                parent_index=i,
                parent_cumulative_logprob=current_beam.cum_logprob,
                token_ids=token_ids_pos,
                logprobs=logprobs_pos,
                ranks=ranks_pos,
                decoded_tokens=decoded_pos,
                valid_token_ids=(
                    valid_tokens_sets[i] if valid_tokens_sets is not None else None
                ),
                candidate_strategy=catalog_strategy,
            )

        candidate_batch = candidate_builder.build()
        all_beams_token_id = candidate_batch.token_ids
        all_beams_logprob = candidate_batch.cumulative_logprobs
        all_beams_beam_idx_arr = candidate_batch.parent_indices
        beam_flat_cache = candidate_batch.parent_cache

        if not ignore_eos:
            stop_token_ids = {eos_token_id}
            if sid_end_token_id is not None:
                stop_token_ids.add(sid_end_token_id)
            eos_beams, all_beams_logprob = complete_eos_candidates(
                parents=all_beams,
                token_ids=all_beams_token_id,
                scores=all_beams_logprob,
                beam_idx=all_beams_beam_idx_arr,
                beam_flat_cache=beam_flat_cache,
                stop_token_ids=stop_token_ids,
                initial_logprobs=initial_logprobs,
                include_eos_token=include_stop_str_in_output,
                inherit_lora_request=False,
            )
            completed.extend(eos_beams)

        # Processing non-EOS tokens
        topn_idx = select_top_indices(
            all_beams_logprob,
            beam_width,
            catalog_strategy,
        )
        new_beams, _ = materialize_selected_beams(
            parents=all_beams,
            selected_indices=topn_idx,
            token_ids=all_beams_token_id,
            scores=all_beams_logprob,
            beam_idx=all_beams_beam_idx_arr,
            beam_flat_cache=beam_flat_cache,
            initial_logprobs=initial_logprobs,
        )

        if not new_beams:
            all_beams = []
            break

        all_beams = new_beams

    if sid_end_token_id is not None:
        for beam in all_beams:
            beam.tokens.append(sid_end_token_id)
    completed.extend(all_beams)

    # Measure beam sort time
    beam_search_sort_start = time.perf_counter()
    sorted_completed = sorted(completed, key=lambda x: x.cum_logprob, reverse=True)
    beam_search_sort_time = time.perf_counter() - beam_search_sort_start
    best_beams = sorted_completed[:beam_width]

    # Reconstruct full logprobs only for the final best beams by walking
    # the parent-pointer tree.  This replaces the O(beams * steps^2)
    # per-step copying with a single O(beams * steps) pass at the end.
    for beam in best_beams:
        reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)

    for i, beam in enumerate(best_beams):
        tokens = (
            beam.tokens[tokenized_length:-1]
            if beam.tokens[-1] == eos_token_id and not ignore_eos
            else beam.tokens[tokenized_length:]
        )
        beam.text = tokenizer.decode(tokens)

    beam_search_decode_time = time.perf_counter() - beam_search_start
    beam_search_prefill_time = beam_search_start - prefill_start
    beam_search_overhead = beam_search_decode_time - generation_time
    metrics = RequestStateStats(
        beam_search_overhead=beam_search_overhead,
        beam_search_prefill_time=beam_search_prefill_time,
        beam_search_decode_time=beam_search_decode_time,
        beam_search_sort_time=beam_search_sort_time,
        num_generation_tokens=num_generation_tokens,
        num_beams=beam_width,
    )

    out = RequestOutput(
        request_id=request_id,
        prompt=prompt_text,
        outputs=[
            CompletionOutput(
                text=beam.text,
                cumulative_logprob=beam.cum_logprob,
                token_ids=beam.tokens[tokenized_length:],
                index=i,
                logprobs=beam.logprobs,
                finish_reason=beam.finish_reason if beam.finish_reason is not None else "length",
                stop_reason=beam.stop_reason,
            )
            for i, beam in enumerate(best_beams)
        ],
        finished=True,
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        metrics=metrics,
    )

    # Record beam search metrics for Prometheus
    # Get the model name - it could be a property or method
    try:
        model_name = self.models.model_name
        if callable(model_name):
            model_name = model_name()
        model_name = str(model_name)
    except Exception:
        model_name = "unknown"
    record_beam_search_metrics(metrics, engine="0", model_name=model_name)

    yield out
