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
    extract_and_dedup_flat_logprobs,
    extract_and_dedup_segment,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.beam_output import BeamOutputSpec
from vllm_gr.config import BeamControlMode, GRConfig
from vllm_gr.constraints import BeamConstraintCoordinator, ParentCandidateBuilder
from vllm_gr.logprobs import project_flat_logprobs, reconstruct_beam_logprobs
from vllm_gr.sampling_params import clone_beam_decode_sampling_params
from vllm_gr.v1.beam.controller_types import (
    BeamControllerFinalResult,
    BeamControllerFinalResultLite,
    BeamControllerRequest,
)
from vllm_gr.v1.engine.types import BeamRequestStepUpdate
from vllm_gr.v1.metrics.beam_search_metrics import (
    record_beam_output_metrics,
    record_beam_search_metrics,
)
from vllm_gr.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)

_dp_rank_counter = 0
_dp_rank_lock = asyncio.Lock()


def _get_output_spec(params) -> BeamOutputSpec:
    spec = getattr(params, "output_spec", None)
    if spec is not None:
        return spec
    # Compatibility for callers constructing upstream BeamSearchParams-like
    # objects without the new contract.
    return BeamOutputSpec(
        result_width=params.beam_width,
        logprobs_mode="top_k",
        top_logprobs=params.beam_width,
    )


# return rank for next beam search request in a round-robin manner across the data parallel ranks
async def _next_data_parallel_rank(data_parallel_size: int) -> int:
    global _dp_rank_counter

    async with _dp_rank_lock:
        rank = _dp_rank_counter % data_parallel_size
        _dp_rank_counter += 1
        return rank


async def _collect_beam_result(q):
    """Drain output queue until finished."""
    while True:
        out = await q.get()
        if out.finished:
            return out


async def _gather_beam_results(queues):
    """Gather results from multiple output queues."""
    tasks = [asyncio.create_task(_collect_beam_result(q)) for q in queues]
    return list(await asyncio.gather(*tasks))


async def _add_batch_step(
    engine_client,
    prompts_batch,
    lora_req_batch,
    request_id_batch,
    beam_search_params,
    use_beam_req,
    trace_headers,
    priority=0,
    data_parallel_rank: int | None = None,
):
    """ADD_BATCH path: prepare and batch-send requests (step 0)."""
    queues = []
    engine_core_requests = []
    for i, (individual_prompt, lora_req) in enumerate(zip(prompts_batch, lora_req_batch)):
        request_id_item = f"{request_id_batch}-beam-{i}"
        q, ec_req = await engine_client.prepare_request(
            request_id_item,
            individual_prompt,
            beam_search_params,
            lora_request=lora_req,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )
        queues.append(q)
        engine_core_requests.append(ec_req)

    await engine_client._add_requests_batch(engine_core_requests, use_batch_message=use_beam_req)

    internal_ids = [r.request_id for r in engine_core_requests] if use_beam_req else []
    output = await _gather_beam_results(queues)
    return output, internal_ids


async def _beam_request_step(
    engine_client,
    session_id: str,
    fork_info,
    prev_beam_internal_ids,
    all_beams,
    request_id_batch,
    beam_search_params,
    eos_token_id,
    lora_request,
    trace_headers,
    prefix_len: int,
    priority=0,
    data_parallel_rank: int | None = None,
    max_decode_steps: int = 0,
):
    """BEAM_REQUEST path: single logical message update for all active branches."""
    parent_beam_ids = []
    child_beam_ids = []
    beam_tokens = []
    parent_beam_indices = []
    beam_width = len(fork_info)

    # Map the current generation step's routing array
    for new_idx, (parent_beam_idx, tok) in enumerate(fork_info):
        parent_id = prev_beam_internal_ids[parent_beam_idx]
        child_id = f"{request_id_batch}-beam-{new_idx}"
        parent_beam_ids.append(parent_id)
        child_beam_ids.append(child_id)
        parent_beam_indices.append(parent_beam_idx)

        full_tokens = all_beams[new_idx].tokens
        beam_tokens.append(full_tokens[prefix_len:])

    # SamplingParams.n retains its upstream meaning: this is one logical
    # request. The active beam count is carried explicitly by beam_width.
    decode_sampling_params = clone_beam_decode_sampling_params(beam_search_params)

    # Listen exclusively to the parent session_id stream queue
    q = engine_client.register_beam_output(
        session_id,
        all_beams[0].tokens,
        decode_sampling_params,
        eos_token_id=eos_token_id,
        lora_request=lora_request,
        trace_headers=trace_headers,
        priority=priority,
        data_parallel_rank=data_parallel_rank,
    )

    # Detect branches that were pruned during top-K filtering
    used_parents = set(parent_beam_ids)
    pruned_ids = [pid for pid in prev_beam_internal_ids if pid not in used_parents]

    # Dispatch the consolidated persistent update to the client
    await engine_client.beam_request_step_update(
        BeamRequestStepUpdate(
            session_id=session_id,
            parent_beam_ids=parent_beam_ids,
            child_beam_ids=child_beam_ids,
            beam_tokens=beam_tokens,
            pruned_ids=pruned_ids,
            prefix_len=prefix_len,
            beam_width=beam_width,
            sampling_params=decode_sampling_params,
            eos_token_id=eos_token_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            data_parallel_rank=data_parallel_rank,
            priority=priority,
            parent_beam_indices=parent_beam_indices,
            max_decode_steps=max_decode_steps,
        )
    )

    # Await the singular unified engine execution response block
    single_output = await _collect_beam_result(q)

    # Return the singular raw output instance wrapped as a single-element list
    # to maintain backward-compatible pipeline execution with the loop below
    return [single_output], child_beam_ids


async def _beam_request_cleanup(engine_client, session_id, final_ids, rank):
    """Signals the core engine to fully purge all remaining session cache allocations."""
    if not final_ids:
        return
    try:
        await engine_client.beam_request_step_update(
            BeamRequestStepUpdate(
                session_id=session_id,
                parent_beam_ids=[],
                child_beam_ids=[],
                beam_tokens=[],
                pruned_ids=final_ids,
                prefix_len=0,
                beam_width=0,
                sampling_params=SamplingParams(),
                data_parallel_rank=rank,
            )
        )
    except Exception as e:
        logger.error("Failed to clear beam_request cache for session %s: %s", session_id, e)


async def _serving_controlled_beam_search(
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
    catalog_strategy = getattr(params, "catalog_strategy", "fill")
    constraint_coordinator = getattr(self.models, "constraint_coordinator", None)
    if constraint_coordinator is None:
        constraint_coordinator = BeamConstraintCoordinator.from_catalog(
            getattr(self.models, "catalog", None)
        )

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
    if beam_width == 0:
        raise VLLMValidationError(
            "Beam width must be greater than 0", parameter="beam_width", value=0
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

    # Check once if the engine supports batch submission / beam fork.
    use_beam_request = hasattr(self.engine_client, "beam_request_step_update")

    prev_beam_internal_ids: list[str] = []
    fork_info: list[tuple[int, int]] | None = None
    prefill_id: str | None = None

    for token in range(max_tokens - pre_calc):
        if not all_beams:
            # Catalog filtering eliminated every active beam (no valid continuation).
            # This is not an engine error — fall through to the normal post-loop path
            # so any EOS beams already in `completed` are returned.
            if use_beam_request and prev_beam_internal_ids:
                await _beam_request_cleanup(
                    self.engine_client, prefill_id, prev_beam_internal_ids, rank
                )
                prev_beam_internal_ids = []
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
        catalog_task = None
        if constraint_coordinator.enabled:

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

        # Execute Engine Step Pass
        if use_beam_request and fork_info is not None:
            output, prev_beam_internal_ids = await _beam_request_step(
                self.engine_client,
                session_id=prefill_id,
                fork_info=fork_info,
                prev_beam_internal_ids=prev_beam_internal_ids,
                all_beams=all_beams,
                request_id_batch=request_id_batch,
                beam_search_params=beam_search_params,
                eos_token_id=eos_token_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prefix_len=len(initial_tokens),
                priority=priority,
                data_parallel_rank=rank,
                max_decode_steps=max_tokens - pre_calc,
            )
        else:
            output, new_ids = await _add_batch_step(
                self.engine_client,
                prompts_batch,
                lora_req_batch,
                request_id_batch,
                beam_search_params,
                use_beam_request,
                trace_headers,
                priority=priority,
                data_parallel_rank=rank,
            )
            if new_ids:
                prev_beam_internal_ids = new_ids
                if prefill_id is None:
                    prefill_id = new_ids[0]

        valid_tokens_sets = None
        if catalog_task is not None:
            valid_tokens_sets = await catalog_task
        if token > 0:
            generation_time += time.perf_counter() - gen_start
        candidate_builder = ParentCandidateBuilder()

        # -------------------------------------------------------------------------
        # STRIDE-AWARE METADATA EXTRACTION FOR GENERATED BEAMS
        # -------------------------------------------------------------------------
        if use_beam_request and fork_info is not None:
            # Under the optimized mode, 'output' contains exactly 1 combined RequestOutput instance
            beam_result = output[0]
            if beam_result.outputs[0].finish_reason == "error":
                await _beam_request_cleanup(
                    self.engine_client, prefill_id, prev_beam_internal_ids, rank
                )
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

            flat = beam_result.outputs[0].logprobs
            if flat is not None:
                # Extract tracking attributes cleanly
                raw_token_ids = flat.token_ids
                raw_logprobs = flat.logprobs
                raw_ranks = getattr(flat, "ranks", None)
                raw_decoded = getattr(flat, "decoded_tokens", None)
                total_logprobs = len(raw_token_ids)

                stride_k = total_logprobs // logprobs_num

                # Each logical beam block should extract its top choices natively
                for b_idx in range(len(fork_info)):
                    current_beam = all_beams[b_idx]

                    # Compute the true exact allocation footprint boundaries
                    # from the flat remuxed vector block for this specific beam
                    start_offset = b_idx * stride_k
                    end_offset = start_offset + stride_k

                    token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_and_dedup_segment(
                        raw_token_ids,
                        raw_logprobs,
                        raw_ranks,
                        raw_decoded,
                        start_offset,
                        end_offset,
                        logprobs_num,
                    )

                    constraint_coordinator.add_parent_candidates(
                        candidate_builder,
                        parent_index=b_idx,
                        parent_cumulative_logprob=current_beam.cum_logprob,
                        token_ids=token_ids_pos,
                        logprobs=logprobs_pos,
                        ranks=ranks_pos,
                        decoded_tokens=decoded_pos,
                        valid_token_ids=(
                            valid_tokens_sets[b_idx] if valid_tokens_sets is not None else None
                        ),
                        candidate_strategy=catalog_strategy,
                    )
            else:
                for b_idx, current_beam in enumerate(all_beams):
                    constraint_coordinator.add_parent_candidates(
                        candidate_builder,
                        parent_index=b_idx,
                        parent_cumulative_logprob=current_beam.cum_logprob,
                        token_ids=[],
                        logprobs=[],
                        ranks=None,
                        decoded_tokens=None,
                        valid_token_ids=None,
                        candidate_strategy=catalog_strategy,
                    )
        else:
            # Baseline execution path for prefill / batch loops
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

                if result.outputs[0].logprobs is not None:
                    flat = result.outputs[0].logprobs

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
                else:
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
        new_beams, selected_fork_info = materialize_selected_beams(
            parents=all_beams,
            selected_indices=topn_idx,
            token_ids=all_beams_token_id,
            scores=all_beams_logprob,
            beam_idx=all_beams_beam_idx_arr,
            beam_flat_cache=beam_flat_cache,
            initial_logprobs=initial_logprobs,
        )

        # Build fork_info for next iteration's BEAM_FORK.
        if use_beam_request:
            fork_info = selected_fork_info

        if not new_beams:
            all_beams = []
            if use_beam_request and prev_beam_internal_ids:
                await _beam_request_cleanup(
                    self.engine_client, prefill_id, prev_beam_internal_ids, rank
                )
                prev_beam_internal_ids = []
            break

        all_beams = new_beams

    if use_beam_request and prev_beam_internal_ids:
        await _beam_request_cleanup(self.engine_client, prefill_id, prev_beam_internal_ids, rank)

    if sid_end_token_id is not None:
        for beam in all_beams:
            beam.tokens.append(sid_end_token_id)
    completed.extend(all_beams)

    # Measure beam sort time
    beam_search_sort_start = time.perf_counter()
    sorted_completed = sorted(completed, key=lambda x: x.cum_logprob, reverse=True)
    beam_search_sort_time = time.perf_counter() - beam_search_sort_start
    output_spec = _get_output_spec(params)
    best_beams = sorted_completed[: output_spec.result_width]

    # Reconstruct full logprobs only for the final best beams by walking
    # the parent-pointer tree.  This replaces the O(beams * steps^2)
    # per-step copying with a single O(beams * steps) pass at the end.
    if output_spec.logprobs_mode != "none":
        for beam in best_beams:
            reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)
            completion_tokens = list(beam.tokens[tokenized_length:])
            beam.logprobs = project_flat_logprobs(
                beam.logprobs,
                completion_tokens,
                output_spec.logprobs_mode,
                output_spec.top_logprobs,
            )

    decode_batches = []
    for beam in best_beams:
        tokens = (
            beam.tokens[tokenized_length:-1]
            if beam.tokens[-1] == eos_token_id and not ignore_eos
            else beam.tokens[tokenized_length:]
        )
        decode_batches.append(tokens)
    for beam, text in zip(best_beams, _batch_decode_equivalent(tokenizer, decode_batches)):
        beam.text = text

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
                logprobs=(None if output_spec.logprobs_mode == "none" else beam.logprobs),
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
    record_beam_output_metrics(
        output_spec,
        len(best_beams),
        beam_width=params.beam_width,
        engine="0",
        model_name=model_name,
    )

    yield out


def _flat_logprobs_from_wire(payload: dict) -> FlatLogprobs:
    return FlatLogprobs(
        start_indices=list(payload["start_indices"]),
        end_indices=list(payload["end_indices"]),
        token_ids=list(payload["token_ids"]),
        logprobs=list(payload["logprobs"]),
        ranks=list(payload["ranks"]),
        decoded_tokens=list(payload["decoded_tokens"]),
    )


def _batch_decode_equivalent(tokenizer, token_batches: list[list[int]]) -> list[str]:
    """Use batch decoding after a one-time tokenizer equivalence check."""
    if not token_batches:
        return []
    cached = getattr(tokenizer, "_vllm_gr_batch_decode_equivalent", None)
    if cached is False or not hasattr(tokenizer, "batch_decode"):
        return [tokenizer.decode(tokens) for tokens in token_batches]
    batch_texts = list(tokenizer.batch_decode(token_batches))
    if cached is True:
        return batch_texts
    individual = [tokenizer.decode(tokens) for tokens in token_batches]
    equivalent = batch_texts == individual
    try:
        setattr(tokenizer, "_vllm_gr_batch_decode_equivalent", equivalent)
    except Exception:
        pass
    return batch_texts if equivalent else individual


async def _engine_controlled_beam_search(
    self,
    prompt: EngineInput,
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    """Thin Serving adapter for either EngineCore-owned beam control mode."""
    if not hasattr(self.engine_client, "run_beam_controller"):
        raise RuntimeError(
            "EngineCore beam control requires the EngineCore beam controller client patch"
        )
    if params.beam_width <= 0:
        raise VLLMValidationError(
            "Beam width must be greater than 0",
            parameter="beam_width",
            value=params.beam_width,
        )

    tokenizer = self.input_processor.tokenizer
    if tokenizer is None:
        raise VLLMValidationError(
            "You cannot use beam search when `skip_tokenizer_init=True`",
            parameter="skip_tokenizer_init",
            value=True,
        )
    if not isinstance(prompt, str) and prompt.get("type") == "enc_dec":
        raise NotImplementedError

    prompt_text = prompt if isinstance(prompt, str) else prompt.get("prompt")
    if isinstance(prompt, str):
        prompt_token_ids = list(tokenizer.encode(prompt))
    else:
        prompt_token_ids = list(prompt.get("prompt_token_ids", []))
    prompt_length = len(prompt_token_ids)

    eos_token_id = tokenizer.eos_token_id
    begin_token_id = None
    if params.begin_token is not None:
        begin_token_id = tokenizer.convert_tokens_to_ids(params.begin_token)
        if begin_token_id is None or begin_token_id == -1:
            raise VLLMValidationError(
                "begin_token must be a valid token in the tokenizer vocabulary",
                parameter="begin_token",
                value=params.begin_token,
            )
    end_token_id = None
    if params.end_token is not None:
        end_token_id = tokenizer.convert_tokens_to_ids(params.end_token)
        if end_token_id is None or end_token_id == -1:
            raise VLLMValidationError(
                "end_token must be a valid token in the tokenizer vocabulary",
                parameter="end_token",
                value=params.end_token,
            )

    initial_tokens = list(prompt_token_ids)
    if begin_token_id is not None:
        initial_tokens.append(begin_token_id)
    reserved = int(begin_token_id is not None) + int(end_token_id is not None)
    if params.max_tokens < reserved:
        raise VLLMValidationError(
            f"max tokens should be not lower than {reserved} to accommodate "
            "reserved begin/end tokens.",
            parameter="max_tokens",
            value=params.max_tokens,
        )

    priority = int(time.perf_counter() * 1_000_000)
    rank = None
    vllm_config = getattr(self.engine_client, "vllm_config", None)
    if vllm_config is not None and vllm_config.parallel_config is not None:
        data_parallel_size = vllm_config.parallel_config.data_parallel_size
        if data_parallel_size is not None and data_parallel_size > 1:
            rank = await _next_data_parallel_rank(data_parallel_size)

    gr_config = GRConfig.from_vllm_config(vllm_config)
    resource = getattr(self.models, "beam_constraint_resource", None)
    resource_wire = getattr(self.models, "beam_constraint_resource_wire", None)
    resource_id = getattr(resource, "digest", None)
    beam_session_id = f"{request_id}-beam-session-{random_uuid()}"
    controller_request = BeamControllerRequest(
        beam_session_id=beam_session_id,
        external_request_id=request_id,
        beam_width=params.beam_width,
        max_tokens=params.max_tokens,
        prompt_token_ids=prompt_token_ids,
        initial_tokens=initial_tokens,
        prompt_length=prompt_length,
        eos_token_id=eos_token_id,
        temperature=params.temperature,
        ignore_eos=params.ignore_eos,
        include_stop_token=params.include_stop_str_in_output,
        begin_token_id=begin_token_id,
        end_token_id=end_token_id,
        catalog_strategy=getattr(params, "catalog_strategy", "fill"),
        constraint_backend=gr_config.constraint_backend.value,
        constraint_resource_id=resource_id,
        beam_control_mode=gr_config.beam_control_mode.value,
        output_spec=_get_output_spec(params),
    )
    sampling_params = SamplingParams(
        logprobs=params.beam_width,
        max_tokens=1,
        temperature=params.temperature,
        detokenize=False,
        flat_logprobs=True,
    )
    engine_prompt = tokens_input(initial_tokens, prompt=prompt_text)

    try:
        result_wire = await self.engine_client.run_beam_controller(
            engine_prompt,
            sampling_params,
            controller_request,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=rank,
            constraint_resource=resource_wire,
        )
    except asyncio.CancelledError:
        try:
            await asyncio.shield(
                self.engine_client.cancel_beam_controller(
                    beam_session_id,
                    data_parallel_rank=rank,
                )
            )
        finally:
            raise

    result = BeamControllerFinalResult.from_wire(result_wire)
    if result.beam_session_id != beam_session_id or result.external_request_id != request_id:
        raise RuntimeError("EngineCore returned a beam controller result for a different session")
    if result.error is not None:
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

    metrics = None
    if result.metrics is not None:
        metrics = RequestStateStats(
            beam_search_overhead=result.metrics.beam_search_overhead,
            beam_search_prefill_time=result.metrics.beam_search_prefill_time,
            beam_search_decode_time=result.metrics.beam_search_decode_time,
            beam_search_sort_time=result.metrics.beam_search_sort_time,
            num_generation_tokens=result.metrics.num_generation_tokens,
            num_beams=result.metrics.num_beams,
        )

    token_id_batches = []
    for beam in result.beams:
        decode_tokens = getattr(beam, "decode_token_ids", None)
        if decode_tokens is None:
            decode_tokens = list(beam.token_ids)
            if decode_tokens and decode_tokens[-1] == eos_token_id and not params.ignore_eos:
                decode_tokens = decode_tokens[:-1]
        token_id_batches.append(decode_tokens)
    decoded_texts = _batch_decode_equivalent(tokenizer, token_id_batches)
    lite_result = isinstance(result, BeamControllerFinalResultLite)
    output = RequestOutput(
        request_id=request_id,
        prompt=prompt_text,
        outputs=[
            CompletionOutput(
                index=index,
                text=decoded_texts[index],
                token_ids=beam.token_ids,
                cumulative_logprob=beam.cumulative_logprob,
                logprobs=(None if lite_result else _flat_logprobs_from_wire(beam.logprobs)),
                finish_reason=beam.finish_reason,
                stop_reason=beam.stop_reason,
            )
            for index, beam in enumerate(result.beams)
        ],
        finished=True,
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        metrics=metrics,
    )
    try:
        model_name = self.models.model_name
        if callable(model_name):
            model_name = model_name()
        model_name = str(model_name)
    except Exception:
        model_name = "unknown"
    record_beam_search_metrics(metrics, engine="0", model_name=model_name)
    record_beam_output_metrics(
        _get_output_spec(params),
        len(result.beams),
        beam_width=params.beam_width,
        engine="0",
        model_name=model_name,
    )
    yield output


async def beam_search(
    self,
    prompt: EngineInput,
    request_id: str,
    params: BeamSearchParams,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    """Dispatch to the staged Serving- or EngineCore-owned control loop."""
    mode = GRConfig.from_vllm_config(
        getattr(self.engine_client, "vllm_config", None)
    ).beam_control_mode
    implementation = (
        _engine_controlled_beam_search
        if mode
        in (
            BeamControlMode.ENGINE_PER_STEP_REQUEST,
            BeamControlMode.ENGINE_PERSISTENT_REQUEST,
        )
        else _serving_controlled_beam_search
    )
    async for output in implementation(
        self,
        prompt,
        request_id,
        params,
        lora_request=lora_request,
        trace_headers=trace_headers,
    ):
        yield output
