# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import math
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
    extract_constrained_flat_logprobs,
    extract_constrained_segment,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.config import (
    CONSTRAINT_TABLE_MAX_GENERATION_STEPS,
    GRConfig,
)
from vllm_gr.constraints import ParentCandidateBuilder
from vllm_gr.constraints.models import CandidateLayout
from vllm_gr.logprobs import reconstruct_beam_logprobs
from vllm_gr.sampling_params import clone_beam_decode_sampling_params
from vllm_gr.v1.engine.types import BeamRequestStepUpdate
from vllm_gr.v1.metrics.beam_search_metrics import record_beam_search_metrics
from vllm_gr.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)

_dp_rank_counter = 0
_dp_rank_lock = asyncio.Lock()


def _validate_serving_constraint_options(
    gr_config: GRConfig,
    *,
    temperature: object,
    catalog_strategy: str | None,
) -> tuple[float, str | None]:
    if not gr_config.constraint_table.enabled:
        return float(temperature), catalog_strategy
    if catalog_strategy is not None:
        raise ValueError(
            f"catalog_strategy is not applicable to constraint_table; got {catalog_strategy!r}"
        )
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or temperature < 0
    ):
        raise ValueError(
            "constraint_table temperature must be a finite non-negative number, "
            f"got {temperature!r}"
        )
    return float(temperature), None


def _validate_serving_constraint_capacity(
    gr_config: GRConfig,
    *,
    beam_width: int,
    max_generate_steps: int,
) -> None:
    if not gr_config.constraint_table.enabled:
        return
    max_width = gr_config.beam.max_width
    if max_width is None:
        raise RuntimeError("enabled constraint_table is missing configured beam.max_width")
    if beam_width > max_width:
        raise ValueError(
            "constraint_table beam_width must be <= configured beam.max_width "
            f"{max_width}, got {beam_width}"
        )
    if max_generate_steps > gr_config.beam.max_decode_steps:
        raise ValueError(
            "constraint_table generated steps must be <= configured "
            f"beam.max_decode_steps {gr_config.beam.max_decode_steps}, "
            f"got {max_generate_steps}"
        )
    if max_generate_steps > CONSTRAINT_TABLE_MAX_GENERATION_STEPS:
        raise ValueError(
            "constraint_table protocol supports at most "
            f"{CONSTRAINT_TABLE_MAX_GENERATION_STEPS} generated steps, "
            f"got {max_generate_steps}"
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
    is_constrained: bool = False,
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
            # Constraint Tables: the Worker's raw-logits hook needs the step to
            # run (1/2 for decode; 0 reserved for the constrained prefill).
            constraint_step=len(beam_tokens[0]) if is_constrained else 0,
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
    gr_config = getattr(self.models, "_gr_config", None)
    if isinstance(gr_config, GRConfig):
        constraints_enabled = gr_config.constraint_table.enabled
    else:
        constraints_enabled = bool(getattr(self.models, "constraints_enabled", False))
        if constraints_enabled:
            raise RuntimeError(
                "constraint_table serving frontend is missing its validated GRConfig"
            )
        gr_config = GRConfig()

    # Frontend filtering is gone; an explicit legacy strategy is rejected when
    # Worker constraints are enabled.
    try:
        temperature, catalog_strategy = _validate_serving_constraint_options(
            gr_config,
            temperature=temperature,
            catalog_strategy=catalog_strategy,
        )
    except ValueError as exc:
        parameter = "catalog_strategy" if catalog_strategy is not None else "temperature"
        value = catalog_strategy if catalog_strategy is not None else temperature
        raise VLLMValidationError(str(exc), parameter=parameter, value=value) from exc

    _is_constrained = constraints_enabled

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

    if _is_constrained:
        try:
            _validate_serving_constraint_capacity(
                gr_config,
                beam_width=beam_width,
                max_generate_steps=max_tokens - pre_calc,
            )
        except ValueError as exc:
            raise VLLMValidationError(str(exc)) from exc

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
                is_constrained=_is_constrained,
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

        if token > 0:
            generation_time += time.perf_counter() - gen_start
        candidate_builder = ParentCandidateBuilder(
            layout=(
                CandidateLayout.CONSTRAINED_TOPK
                if _is_constrained
                else CandidateLayout.STANDARD_SAMPLER
            )
        )

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

                    if _is_constrained:
                        token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                            extract_constrained_segment(
                                raw_token_ids,
                                raw_logprobs,
                                raw_ranks,
                                raw_decoded,
                                start_offset,
                                end_offset,
                                logprobs_num,
                            )
                        )
                    else:
                        token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                            extract_and_dedup_segment(
                                raw_token_ids,
                                raw_logprobs,
                                raw_ranks,
                                raw_decoded,
                                start_offset,
                                end_offset,
                                logprobs_num,
                            )
                        )

                    candidate_builder.add_parent(
                        parent_index=b_idx,
                        parent_cumulative_logprob=current_beam.cum_logprob,
                        token_ids=token_ids_pos,
                        local_logprobs=logprobs_pos,
                        ranks=ranks_pos,
                        decoded_tokens=decoded_pos,
                    )
            else:
                for b_idx, current_beam in enumerate(all_beams):
                    candidate_builder.add_parent(
                        parent_index=b_idx,
                        parent_cumulative_logprob=current_beam.cum_logprob,
                        token_ids=[],
                        local_logprobs=[],
                        ranks=None,
                        decoded_tokens=None,
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

                    if _is_constrained:
                        token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                            extract_constrained_flat_logprobs(flat, logprobs_num)
                        )
                    else:
                        token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                            extract_and_dedup_flat_logprobs(flat, logprobs_num)
                        )
                    candidate_builder.add_parent(
                        parent_index=i,
                        parent_cumulative_logprob=current_beam.cum_logprob,
                        token_ids=token_ids_pos,
                        local_logprobs=logprobs_pos,
                        ranks=ranks_pos,
                        decoded_tokens=decoded_pos,
                    )
                else:
                    candidate_builder.add_parent(
                        parent_index=i,
                        parent_cumulative_logprob=current_beam.cum_logprob,
                        token_ids=[],
                        local_logprobs=[],
                        ranks=None,
                        decoded_tokens=None,
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
