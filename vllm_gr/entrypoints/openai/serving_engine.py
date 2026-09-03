# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from collections.abc import AsyncGenerator, Mapping

import numpy as np
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.exceptions import VLLMValidationError
from vllm.inputs import EngineInput, tokens_input
from vllm.logger import init_logger
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import BeamSearchParams, RequestOutputKind, SamplingParams
from vllm.utils import random_uuid

from vllm_gr.beam_search_decision_utils import (
    complete_eos_candidates,
    extract_and_dedup_flat_logprobs,
    extract_and_dedup_segment,
    extract_constrained_flat_logprobs,
    extract_constrained_segment,
    materialize_final_beams,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.config import MAX_BEAM_WIDTH, GRConfig
from vllm_gr.constraints import BeamConstraintCoordinator, ParentCandidateBuilder
from vllm_gr.constraints.models import CandidateLayout
from vllm_gr.constraints.strategies import (
    ConstraintTableStrategy,
    TrieConstraintStrategy,
    validate_constraint_generation_limits,
    validate_constraint_table_protocol,
)
from vllm_gr.logprobs import reconstruct_beam_logprobs
from vllm_gr.sampling_params import clone_beam_decode_sampling_params
from vllm_gr.v1.engine.types import (
    GR_REQUEST_METADATA_KEY,
    BeamRequestStepUpdate,
    GRRequestMetadata,
)
from vllm_gr.v1.metrics.beam_search_metrics import record_beam_search_metrics
from vllm_gr.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)


def _worker_decision_flatten(
    decision: Mapping[str, object],
    num_beams: int,
    parents: list,
) -> tuple[list[int], list[float], list[int], list] | None:
    """Flatten a worker beam decision into per-beam candidate lists.

    The worker has already selected the surviving beams, so ``out_token_ids`` /
    ``out_log_probs`` / ``out_token_index`` each hold exactly ``num_beams``
    entries.  The per-parent cache entries ("local" logprob = cumulative minus
    the parent's cumulative) let ``materialize_selected_beams`` reconstruct
    ``_lp_step_data`` identically to the flat-logprobs path.

    The payload arrives via ``kv_transfer_params['beam_search_result']`` and
    may be host tensors (in-process) or nested Python lists (post-IPC; the
    untyped mapping is the only container that msgpack round-trips
    natively); ``flatten_decision_value`` handles both.  A serialized-tensor
    triple (an unnormalized payload) returns ``None`` and the caller falls
    back to the legacy flat-logprobs path.

    Returns ``(token_ids, cumulative_logprobs, parent_indices, parent_cache)``
    or ``None`` when the payload cannot be mapped to ``num_beams`` winners.
    """
    from vllm_gr.v1.beam.beam_decision import (
        flatten_decision_value,
        unpack_beam_result_buffer,
    )

    compact = unpack_beam_result_buffer(decision)
    if compact is not None:
        ids_flat, lps_flat, parent_flat = compact
    else:
        ids_flat = flatten_decision_value(decision.get("out_token_ids"))
        lps_flat = flatten_decision_value(decision.get("out_log_probs"))
        parent_flat = flatten_decision_value(decision.get("out_token_index"))
    if ids_flat is None or lps_flat is None or parent_flat is None:
        return None
    if not ids_flat:
        return None
    try:
        ids = [int(v) for v in ids_flat]
        lps = [float(v) for v in lps_flat]
        parent = [int(v) for v in parent_flat]
    except (TypeError, ValueError):
        return None
    if len(ids) != num_beams or len(ids) != len(lps) or len(ids) != len(parent):
        return None
    # out_token_index keeps the request offset; strip it (batch == 1).
    parent = [p - num_beams if p >= num_beams else p for p in parent]
    cache: list = []
    for i, (pid, tid, lp) in enumerate(zip(parent, ids, lps)):
        parent_cum = parents[pid].cum_logprob if pid < len(parents) else 0.0
        cache.append(([tid], [lp - parent_cum], [0], [None]))
    return ids, lps, parent, cache


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


# Collector timeout: EngineCore engine-driven chaining can run steps ahead
# of the frontend's collector registration (buffered by register_beam_output's
# replay).  A lost output would otherwise hang the request forever.
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
                "(chained step output never reached the collector)",
                _BEAM_COLLECT_TIMEOUT_S,
            )
            raise
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
    session_template=None,
    gr_metadata: GRRequestMetadata | None = None,
):
    """ADD_BATCH path: prepare and batch-send requests (step 0).

    ``session_template`` (engine-driven prefill chain) is sent fire-and-forget
    right after the ADD_BATCH and BEFORE awaiting the prefill outputs: the
    EngineCore consumes it in the same busy-loop iteration that processes
    the prefill output, so any later send would always be too late.
    When the client supports SUBMIT_SESSION, the ADD_BATCH and the template
    travel as ONE wire message instead (same ordering guarantees, one fewer
    message per session); the two-message path stays available for clients
    without the merged send.
    """
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
        if gr_metadata is not None:
            metadata = {
                field: getattr(gr_metadata, field)
                for field in gr_metadata.__struct_fields__
            }
            metadata["request_id"] = ec_req.request_id
            ec_req.sampling_params.extra_args = dict(
                ec_req.sampling_params.extra_args or {}
            )
            ec_req.sampling_params.extra_args[GR_REQUEST_METADATA_KEY] = metadata
        queues.append(q)
        engine_core_requests.append(ec_req)

    # Track the session before EngineCore dispatch: engine-driven chaining
    # can emit the first decode output while this coroutine still awaits
    # prefill, and process_outputs only buffers tracked session ids.
    if use_beam_req and engine_core_requests:
        track = getattr(engine_client, "track_beam_session", None)
        if track is not None:
            track(engine_core_requests[0].request_id)

    # The template must land before the prefill output iteration runs its
    # engine-driven hook — either merged into the SUBMIT_SESSION send below
    # or as the fire-and-forget message right after ADD_BATCH; both precede
    # the _gather_beam_results await, and anything later would be too late.
    submit_session = getattr(engine_client, "submit_session", None)
    if (
        session_template is not None
        and use_beam_req
        and engine_core_requests
        and submit_session is not None
    ):
        session_template.session_id = engine_core_requests[0].request_id
        await submit_session(engine_core_requests, session_template)
    else:
        await engine_client._add_requests_batch(
            engine_core_requests, use_batch_message=use_beam_req
        )

        # Cache the session's step template now (same socket as ADD_BATCH,
        # FIFO) so it is in beam_cache before the prefill output iteration
        # runs its engine-driven hook.  The session id is stamped from the
        # actual request ids instead of assuming the batch layout.
        if session_template is not None and use_beam_req and engine_core_requests:
            session_template.session_id = engine_core_requests[0].request_id
            await engine_client.beam_session_template(session_template)

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
    self_advance_chained: bool = False,
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

    # Dispatch the consolidated persistent update to the client, unless
    # EngineCore already chained this decode step from the worker's beam
    # decision (engine-driven). Re-sending the update would re-ADD the
    # session's request while it is still running under the same
    # session_id, crashing scheduler.add_request with "duplicate request
    # id"; only the queue registration + await below are needed to receive
    # the chained step's output.
    if not self_advance_chained:
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
        beam_search_params.max_tokens = decode_steps
        beam_search_params.output_kind = RequestOutputKind.DELTA
        if persistent_runner_state:
            beam_search_params.logprobs = None
        finish_token_id = sid_end_token_id if sid_end_token_id is not None else eos_token_id
        if finish_token_id is None:
            finish_token_id = 0
        continue_token_id = 0 if finish_token_id != 0 else 1
        gr_metadata = GRRequestMetadata(
            request_id="",
            beam_width=beam_width,
            max_tokens=decode_steps,
            prefix_len=len(initial_tokens),
            continue_token_id=continue_token_id,
            finish_token_id=finish_token_id,
            num_return_sequences=beam_width,
        )

    # Check once if the engine supports batch submission / beam fork.
    use_beam_request = hasattr(self.engine_client, "beam_request_step_update")

    # Trie constraints are enforced in the frontend (catalog trie); the
    # baseline _add_batch_step loop constrains candidates correctly and
    # returns per-beam FlatLogprobs with precise boundaries.  The BEAM_REQUEST
    # protocol's merged-flat / fixed-stride extraction deviates from that path
    # (regression: CI GPU online pass@1 0.03 vs 0.09 threshold), so route Trie
    # through the per-beam batch loop.  Constraint Tables keep the BEAM_REQUEST
    # path: their candidates are produced by the Worker raw-logits hook, which
    # is only installed on the beam-request session (see
    # validate_constraint_table_protocol below).
    if constraint_coordinator is not None and isinstance(
        constraint_coordinator.strategy, TrieConstraintStrategy
    ):
        use_beam_request = False

    # Constraint Tables generate candidates in the Worker from a beam-request
    # session (step0 prefill via EngineCore ADD_BATCH cache hook, step1/2 decode
    # via beam_request_step_update).  The _add_batch_step fallback cannot express
    # constraint_step + beam_suffix, so a client without the beam protocol must
    # fail fast — never silently degrade.
    validate_constraint_table_protocol(
        is_constraint_table=_is_constrained,
        beam_protocol_available=use_beam_request,
        required=(
            "an engine client implementing beam_request_step_update (EngineCore beam sessions)"
        ),
    )

    prev_beam_internal_ids: list[str] = []
    fork_info: list[tuple[int, int]] | None = None
    prefill_id: str | None = None
    # Set when the previous decode step's worker decision was chained
    # in-process by EngineCore (engine-driven); the next loop iteration
    # must then skip its beam_request_step_update send (see
    # _beam_request_step) and only await the chained step's output.
    prev_step_self_advance_chained = False

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
                # EngineCore chained the previous step from the worker's
                # beam decision (engine-driven); re-sending the step
                # update would duplicate the session request id.
                self_advance_chained=prev_step_self_advance_chained,
            )
        else:
            # Prefill session template: fields EngineCore cannot derive
            # itself when it constructs the step-0 BeamRequestStepUpdate
            # in-process from the worker's constrained-prefill decision.
            # Sent on the non-persistent constrained beam path;
            # EngineCore-side gating (beam_engine_driven) decides whether
            # to consume it. Persistent requests are advanced by
            # GRAsyncScheduler and must not install the parallel
            # engine-driven session chain.
            prefill_session_template = None
            if use_beam_request and _is_constrained and not persistent_request:
                prefill_session_template = BeamRequestStepUpdate(
                    session_id="",  # stamped by _add_batch_step from the request ids
                    parent_beam_ids=[],
                    child_beam_ids=[],
                    beam_tokens=[],
                    pruned_ids=[],
                    prefix_len=len(initial_tokens),
                    beam_width=beam_width,
                    sampling_params=clone_beam_decode_sampling_params(beam_search_params),
                    eos_token_id=eos_token_id,
                    trace_headers=trace_headers,
                    data_parallel_rank=rank,
                    priority=priority,
                    max_decode_steps=max_tokens - pre_calc,
                    # Constraint Tables: 1 = constrained decode step 0 (the
                    # value the frontend itself puts on the step-0 update).
                    constraint_step=1,
                )
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
                session_template=prefill_session_template,
                gr_metadata=gr_metadata,
            )
            if new_ids:
                prev_beam_internal_ids = new_ids
                if prefill_id is None:
                    prefill_id = new_ids[0]

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
            prev_beam_internal_ids = []
            if final_decision.get("counters"):
                logger.info(
                    "vllm-gr request counters: %s",
                    final_decision["counters"],
                )
            break

        valid_tokens_sets = None
        if catalog_task is not None:
            valid_tokens_sets = await catalog_task
        if token > 0 or persistent_runner_state:
            generation_time += time.perf_counter() - gen_start
        candidate_builder = ParentCandidateBuilder(
            layout=(
                CandidateLayout.CONSTRAINED_TOPK
                if _is_constrained
                else CandidateLayout.STANDARD_SAMPLER
            )
        )
        # Worker-side beam decision result. Initialised here so
        # the per-beam extraction below can overwrite it and the bottom-of-loop
        # selection always has a defined value even on the prefill/batch path
        # (where the use_beam_request branch never runs).
        worker_candidates = None

        # Constraint prefill also has a canonical CUDA worker decision.  It
        # arrives through the same result channel as decode, but this first
        # iteration uses the ADD_BATCH branch (``fork_info is None``).
        if use_beam_request and _is_constrained and fork_info is None and len(output) == 1:
            prefill_decision = _worker_decision_from_result(output[0])
            if prefill_decision is not None:
                worker_candidates = _worker_decision_flatten(
                    prefill_decision, beam_width, all_beams
                )
                # EngineCore chained step-0 in-process from this prefill
                # decision (engine-driven).  Iteration 1 must then use the
                # existing skip path (only collector registration + await;
                # re-sending the step update would re-ADD the running session
                # request and trip scheduler.add_request's duplicate-id assert).
                prev_step_self_advance_chained = bool(
                    prefill_decision.get("beam_self_advance_chained")
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

            # Worker-side beam decision (accelerator path): the worker
            # already selected the surviving beams, so skip the frontend flat
            # logprobs extraction + NumPy top-k entirely and feed the worker's
            # winners into the existing EOS / materialisation pipeline.
            worker_decision = _worker_decision_from_result(beam_result)
            worker_candidates = None
            if worker_decision is not None:
                worker_candidates = _worker_decision_flatten(worker_decision, beam_width, all_beams)
            # EngineCore chained this step's successor from the worker's
            # decision (engine-driven): the next iteration skips its
            # beam_request_step_update and only awaits the chained output.
            # Missing flag (terminal decision / skip / fallback) leaves the
            # frontend-driven path in charge.
            prev_step_self_advance_chained = bool(
                worker_decision is not None and worker_decision.get("beam_self_advance_chained")
            )

            flat = beam_result.outputs[0].logprobs
            if worker_candidates is not None:
                # Workers already decided: no per-beam flat segment to parse.
                pass
            elif flat is not None:
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

                    constraint_coordinator.add_parent_candidates(
                        candidate_builder,
                        parent_index=b_idx,
                        parent_cumulative_logprob=current_beam.cum_logprob,
                        token_ids=token_ids_pos,
                        logprobs=logprobs_pos,
                        ranks=ranks_pos,
                        decoded_tokens=decoded_pos,
                        valid_token_ids=(
                            None
                            if _is_constrained
                            else valid_tokens_sets[b_idx]
                            if valid_tokens_sets is not None
                            else None
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
                if worker_candidates is not None:
                    break
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
                    constraint_coordinator.add_parent_candidates(
                        candidate_builder,
                        parent_index=i,
                        parent_cumulative_logprob=current_beam.cum_logprob,
                        token_ids=token_ids_pos,
                        logprobs=logprobs_pos,
                        ranks=ranks_pos,
                        decoded_tokens=decoded_pos,
                        valid_token_ids=(
                            None
                            if _is_constrained
                            else valid_tokens_sets[i]
                            if valid_tokens_sets is not None
                            else None
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
        if worker_candidates is not None:
            # Worker side already selected the surviving beams; the candidate
            # batch built from the (skipped) flat segments is meaningless, so
            # swap in the worker's winners directly.
            (
                all_beams_token_id,
                all_beams_logprob,
                all_beams_beam_idx_arr,
                beam_flat_cache,
            ) = worker_candidates
            all_beams_token_id = np.asarray(all_beams_token_id, dtype=np.int64)
            all_beams_logprob = np.asarray(all_beams_logprob, dtype=np.float64)
            all_beams_beam_idx_arr = np.asarray(all_beams_beam_idx_arr, dtype=np.intp)
        else:
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
        topn_idx = (
            range(len(all_beams_token_id))
            if worker_candidates is not None
            else select_top_indices(
                all_beams_logprob,
                beam_width,
                catalog_strategy,
            )
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
