# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GRLLM — user-facing entry point for offline inference with beam search.

A thin subclass of vLLM's :class:`~vllm.entrypoints.llm.LLM` that adds:

- optional catalog-constrained generation (``catalog_path``)
- a catalog-aware :meth:`beam_search` implementation
- an isolated :meth:`beam_search_v1` entry point for persistent execution work

The previous GR adapter / recommend runtime has been removed; this class now
always delegates construction to the standard vLLM ``LLM`` initializer so any
HuggingFace checkpoint (e.g. OneRec, Pangu) is supported via the regular vLLM
LLM path.
"""

import gc
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Union, cast, overload

import numpy as np
from tqdm.auto import tqdm
from vllm.entrypoints.generate.beam_search.utils import (
    BeamSearchInstance,
    BeamSearchOutput,
    BeamSearchSequence,
)
from vllm.entrypoints.llm import LLM
from vllm.inputs import TokensPrompt
from vllm.logprobs import FlatLogprobs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.beam_search_decision_utils import (
    complete_eos_candidates,
    extract_and_dedup_flat_logprobs,
    extract_and_dedup_segment,
    extract_constrained_flat_logprobs,
    extract_constrained_segment,
    materialize_selected_beams,
    select_top_beams,
    select_top_indices,
)
from vllm_gr.config import ConstraintBackend, GRConfig
from vllm_gr.config.validation import (
    validate_beam_max_batch_size,
    validate_beam_max_width,
    validate_beam_width,
)
from vllm_gr.constraints import (
    BeamConstraintCoordinator,
    ParentCandidateBuilder,
    create_constraint_strategy,
)
from vllm_gr.constraints.models import BeamCandidateCache, CandidateLayout
from vllm_gr.constraints.startup import install_constraint_table
from vllm_gr.constraints.strategies import (
    ConstraintTableStrategy,
    TrieConstraintStrategy,
    validate_beam_request_prefix_cache,
    validate_constraint_generation_limits,
    validate_constraint_table_protocol,
)
from vllm_gr.engine.arg_utils_gr import engine_args_accepts_gr_values
from vllm_gr.logger import init_logger
from vllm_gr.logprobs import (
    reconstruct_beam_logprobs_batch,
)
from vllm_gr.outputs import CompactBeamSearchOutput, CompactBeamSearchSequence
from vllm_gr.sampling_params import clone_beam_decode_sampling_params
from vllm_gr.tokenizer_fast import fast_decode
from vllm_gr.utils.engine_lifecycle import frontend_request_scope
from vllm_gr.utils.npu_adapter import release_accelerator_memory
from vllm_gr.v1.engine.types import BeamRequestStepUpdate

if TYPE_CHECKING:
    from vllm.entrypoints.generate.beam_search.utils import BeamSearchOutput
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import BeamSearchParams

    from vllm_gr.entrypoints.beam_search_v1.protocol import BeamSearchFrontend

logger = init_logger(__name__)

ForkInfo = list[tuple[int, int]]


def _beam_full_tokens(beam: "BeamSearchSequence") -> list[int]:
    """Return the beam's full token sequence.

    In shared-prefix mode the full prompt lives in the read-only
    ``_prompt_token_ids`` field and ``tokens`` holds only the generated tail;
    in the legacy path ``tokens`` is already the full sequence.
    """
    prompt = getattr(beam, "_prompt_token_ids", None)
    return prompt + beam.tokens if prompt is not None else beam.tokens


class _LazyBeamField:
    """Data descriptor deferring ``tokens``/``logprobs``/``text`` materialization.

    Assignments land directly in the instance ``__dict__`` (the same place a
    plain dataclass field lives), so construction and in-place updates are
    unchanged.  A read first hands the beam to the per-request materializer
    exactly once; afterwards the field behaves like an ordinary attribute.
    """

    __slots__ = ("name",)

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        materializer = obj.__dict__.get("_lazy_materializer")
        if materializer is not None:
            obj.__dict__["_lazy_materializer"] = None
            materializer(obj)
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        obj.__dict__[self.name] = value


@dataclass
class _LazyMaterializedBeamSequence(BeamSearchSequence):
    """Beam whose final fields materialize on first access.

    The class body rebinds the three expensive fields to descriptors without
    redeclaring them as dataclass fields (no annotations), so ``fields()``,
    ``replace()`` and the generated ``__init__`` keep working unchanged.
    """

    tokens = _LazyBeamField()
    logprobs = _LazyBeamField()
    text = _LazyBeamField()


class _BeamTailMaterializer:
    """Per-request context shared by every lazy beam of one beam_search call.

    ``__call__(beam)`` runs the tail pipeline for a single beam — logprob
    reconstruction, full-token restore and tail detokenization — using the
    request-shared ``initial_logprobs`` and prompt text.  Reading k of 128
    beams therefore costs k (not 128) reconstructions/decodes/copies, and the
    1024-token init-segment list copies drop from 128 per request to k.
    """

    __slots__ = (
        "_initial_logprobs",
        "_sid_end_token_id",
        "_tokenizer",
        "_prompt_text",
        "_prompt_len",
    )

    def __init__(
        self,
        initial_logprobs,
        sid_end_token_id,
        tokenizer,
        prompt_text,
        prompt_len,
    ):
        self._initial_logprobs = initial_logprobs
        self._sid_end_token_id = sid_end_token_id
        self._tokenizer = tokenizer
        self._prompt_text = prompt_text
        self._prompt_len = prompt_len

    def __call__(self, beam: "BeamSearchSequence") -> None:
        reconstruct_beam_logprobs_batch([beam], self._initial_logprobs, self._sid_end_token_id)
        full = _beam_full_tokens(beam)
        tail = full[self._prompt_len :]
        for attr in ("_prompt_token_ids", "_lp_parent", "_lp_step_data"):
            if hasattr(beam, attr):
                delattr(beam, attr)
        beam.__dict__["tokens"] = full
        beam.text = self._prompt_text + fast_decode(self._tokenizer, tail)
        # Back to the plain dataclass: the descriptors have served their
        # purpose and later reads take the normal attribute fast path.
        beam.__class__ = BeamSearchSequence


def _prompt_text_cached(tokenizer, prompt_tokens, prompt_len: int) -> str:
    """Decode the prompt prefix through a small per-tokenizer LRU.

    Benchmark replays an identical prompt per request, and production
    prefixes (user profiles) repeat at a high rate, so the ~1ms 1023-token
    decode is nearly always avoidable.  The cache hangs off the tokenizer
    instance, so it dies with it and distinct vocabularies can never share
    entries.  Falls back to an uncached decode for exotics (e.g. slotted
    tokenizers without a ``__dict__``).
    """
    try:
        cache = tokenizer.__dict__.get("_gr_prompt_text_cache")
        if cache is None:
            cache = tokenizer.__dict__["_gr_prompt_text_cache"] = OrderedDict()
    except AttributeError:
        return fast_decode(tokenizer, prompt_tokens[:prompt_len])

    key = tuple(prompt_tokens[:prompt_len])
    text = cache.get(key)
    if text is not None:
        cache.move_to_end(key)
        return text
    text = fast_decode(tokenizer, prompt_tokens[:prompt_len])
    cache[key] = text
    if len(cache) > 64:
        cache.popitem(last=False)
    return text


def _create_tokens_prompt_from_beam(beam: "BeamSearchSequence") -> "TokensPrompt":
    """Create a TokensPrompt payload from a BeamSearchSequence."""
    # get_prompt() expects the full sequence, so hand it a temporary view with
    # the shared prompt prefix prepended when in shared-prefix mode.
    return replace(beam, tokens=_beam_full_tokens(beam)).get_prompt()


def _beam_request_cleanup(llm_engine, session_id: str, final_ids: list[str]) -> None:
    """Clear beam_request cache for a finished/failed session."""
    if not final_ids:
        return
    try:
        llm_engine.engine_core.beam_request_step_update(
            BeamRequestStepUpdate(
                session_id=session_id,
                parent_beam_ids=[],
                child_beam_ids=[],
                beam_tokens=[],
                pruned_ids=final_ids,
                prefix_len=0,
                beam_width=0,
                sampling_params=SamplingParams(),
            )
        )
    except Exception as e:
        logger.error("Failed to clear beam_request cache for session %s: %s", session_id, e)


def _prepare_beam_step_requests(
    llm_engine,
    instances_batch: list["BeamSearchInstance"],
    token: int,
    prompt_start: int,
    tokenizer,
    beam_search_params: "SamplingParams",
    sid_end_token_id: int | None,
    instance_prefill_id: dict[int, str],
    instance_prev_beam_internal_ids: dict[int, list[str]],
    instance_fork_info: dict[int, ForkInfo],
    instance_prefix_len: dict[int, int],
    instance_completed_sessions: set[int],
    max_decode_steps: int = 0,
    is_constrained: bool = False,
    instance_self_advance_chained: set[int] | None = None,
) -> list["EngineCoreRequest"]:
    """Prepare EngineCoreRequest items for active instances at this step."""
    engine_core_requests = []
    beam_step_updates = []
    for j, instance in enumerate(instances_batch):
        if len(instance.beams) == 0 or j in instance_completed_sessions:
            # This instance has no active beams (finished) — the terminal
            # short-circuit also lands here: it keeps the rebuilt final
            # beams on instance.beams but marks the session completed, so
            # the loop must stop stepping this instance while its results
            # stay intact for the caller's final assembly.
            if j not in instance_completed_sessions and j in instance_prefill_id:
                _beam_request_cleanup(
                    llm_engine,
                    instance_prefill_id[j],
                    instance_prev_beam_internal_ids[j],
                )
                instance_completed_sessions.add(j)
            continue

        session_id = instance_prefill_id.get(j)
        fork_info_j = instance_fork_info.get(j)
        if token > 0 and fork_info_j is not None:
            # BEAM_REQUEST path
            parent_beam_ids = []
            child_beam_ids = []
            beam_tokens = []
            W = len(fork_info_j)

            for new_idx, (parent_beam_idx, tok) in enumerate(fork_info_j):
                parent_id = instance_prev_beam_internal_ids[j][parent_beam_idx]
                child_id = f"offline-beam-{prompt_start + j}-{token}-{new_idx}"
                parent_beam_ids.append(parent_id)
                child_beam_ids.append(child_id)

                prefix_len = instance_prefix_len[j]
                # `tokens` is the generated tail under shared-prefix mode.
                beam_tokens.append(instance.beams[new_idx].tokens)

            # One grouped-beam execution is one logical upstream request.
            # Beam width travels separately in BeamRequestStepUpdate.
            decode_sampling_params = clone_beam_decode_sampling_params(beam_search_params)
            eos_token_id = (
                sid_end_token_id if sid_end_token_id is not None else tokenizer.eos_token_id
            )
            decode_sampling_params.update_from_generation_config({}, eos_token_id)

            # Register beam output
            ec_req = EngineCoreRequest(
                request_id=session_id,
                prompt_token_ids=instance.beams[0]._prompt_token_ids + instance.beams[0].tokens,
                mm_features=None,
                sampling_params=decode_sampling_params,
                pooling_params=None,
                arrival_time=time.time(),
                lora_request=instance.beams[0].lora_request,
                cache_salt=None,
                data_parallel_rank=None,
            )
            ec_req.external_req_id = session_id
            llm_engine.output_processor.add_request(ec_req, None, None, 0)
            used_parents = set(parent_beam_ids)
            pruned_ids = [
                pid for pid in instance_prev_beam_internal_ids[j] if pid not in used_parents
            ]

            # Collect numeric parent beam indices for incremental decode
            parent_beam_indices = []
            for new_idx, (parent_beam_idx, tok) in enumerate(fork_info_j):
                parent_beam_indices.append(parent_beam_idx)

            parent_cumulative_logprobs = (
                [float(beam.cum_logprob) for beam in instance.beams]
                if all(hasattr(beam, "cum_logprob") for beam in instance.beams)
                else None
            )

            update = BeamRequestStepUpdate(
                session_id=session_id,
                parent_beam_ids=parent_beam_ids,
                child_beam_ids=child_beam_ids,
                beam_tokens=beam_tokens,
                pruned_ids=pruned_ids,
                prefix_len=prefix_len,
                beam_width=W,
                sampling_params=decode_sampling_params,
                eos_token_id=sid_end_token_id
                if sid_end_token_id is not None
                else tokenizer.eos_token_id,
                lora_request=instance.beams[0].lora_request,
                parent_beam_indices=parent_beam_indices,
                parent_cumulative_logprobs=parent_cumulative_logprobs,
                max_decode_steps=max_decode_steps,
                # Constraint Tables: the Worker's raw-logits hook needs the
                # step to run (1/2 for decode; 0 reserved for the prefill).
                constraint_step=len(beam_tokens[0]) if is_constrained else 0,
            )
            # EngineCore already chained this decode step from the worker's
            # beam decision (engine-driven). Re-sending the update would
            # re-ADD the session's request while it is still running under
            # the same session_id, crashing scheduler.add_request with
            # "duplicate request id"; the output registration above already
            # wired this instance to the chained step's output.
            self_advance_chained = (
                instance_self_advance_chained is not None and j in instance_self_advance_chained
            )
            if self_advance_chained:
                # The chained step consumed this tracking entry; step 0-style
                # re-registration must not run again for this step.
                instance_self_advance_chained.discard(j)
            else:
                beam_step_updates.append(update)

            # Update tracking variables
            instance_prev_beam_internal_ids[j] = child_beam_ids

        else:
            # ADD_BATCH path (step 0)
            request_id_batch = f"offline-beam-{prompt_start + j}"
            instance_prefix_len[j] = len(instance.beams[0]._prompt_token_ids)

            request_id_item = f"{request_id_batch}-0"
            tokens_prompt = _create_tokens_prompt_from_beam(instance.beams[0])
            ec_req = llm_engine.input_processor.process_inputs(
                request_id=request_id_item,
                prompt=tokens_prompt,
                params=beam_search_params,
                supported_tasks=llm_engine.get_supported_tasks(),
                arrival_time=time.time(),
                lora_request=instance.beams[0].lora_request,
            )
            llm_engine.input_processor.assign_request_id(ec_req)
            llm_engine.output_processor.add_request(ec_req, None, None, 0)
            engine_core_requests.append(ec_req)

    if len(beam_step_updates) == 1:
        llm_engine.engine_core.beam_request_step_update(beam_step_updates[0])
    elif beam_step_updates:
        cohort_id = (int(prompt_start) << 32) | int(token)
        for update in beam_step_updates:
            update.cohort_id = cohort_id
        llm_engine.engine_core.beam_request_step_update_batch(beam_step_updates)
    return engine_core_requests


def _step_engine_and_collect_outputs(
    llm_engine,
    instances_batch: list["BeamSearchInstance"],
    token: int,
    prompt_start: int,
    instance_prefill_id: dict[int, str],
) -> dict[str, Any]:
    """Execute LLM engine step until outputs for all active instances are gathered."""
    active_indices = [j for j, instance in enumerate(instances_batch) if len(instance.beams) > 0]
    step_outputs_accumulated = []
    while active_indices:
        step_outputs = llm_engine.step()
        step_outputs_accumulated.extend(step_outputs)

        outputs_map = {out.request_id: out for out in step_outputs}
        new_active_indices = []
        for j in active_indices:
            session_id = instance_prefill_id.get(j)
            search_id = f"offline-beam-{prompt_start + j}-0" if token == 0 else session_id
            if search_id not in outputs_map:
                new_active_indices.append(j)
        active_indices = new_active_indices

    return {out.request_id: out for out in step_outputs_accumulated}


def _parse_step_logprobs(
    result,
    instance: "BeamSearchInstance",
    token: int,
    logprobs_num: int,
    fork_info_j: ForkInfo | None,
    catalog: Any | None = None,
    catalog_strategy: str | None = None,
    constraint_coordinator: BeamConstraintCoordinator | None = None,
    return_arrays: bool = False,
) -> tuple[bool, Any, Any, Any, list[BeamCandidateCache]]:
    """Parse flat logprobs from engine step results.

    Returns:
        is_error: bool indicating if the engine returned an error
        all_beams_token_id: candidate token ids across candidate beams
        all_beams_logprob: cumulative logprobs
        all_beams_beam_idx: parent index for each entry
        beam_flat_cache: list of parsed segments for each parent beam
    """
    if result.outputs[0].finish_reason == "error":
        if return_arrays:
            return (
                True,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.intp),
                [],
            )
        return True, [], [], [], []

    coordinator = constraint_coordinator or BeamConstraintCoordinator.from_catalog(catalog)
    # Constraint Tables generate candidates in the Worker; the flat segments
    # are the operator's fixed-width rows (no sampled-token slot, no dedup).
    is_constrained = isinstance(coordinator.strategy, ConstraintTableStrategy)
    candidate_builder = ParentCandidateBuilder(
        layout=(
            CandidateLayout.CONSTRAINED_TOPK if is_constrained else CandidateLayout.STANDARD_SAMPLER
        )
    )

    flat = result.outputs[0].logprobs
    if flat is None:
        raise ValueError(
            "Beam search generation expected non-empty logprobs for each beam, but received 'None'."
        )

    raw_token_ids = flat.token_ids
    raw_logprobs = flat.logprobs
    raw_ranks = getattr(flat, "ranks", None)
    raw_decoded = getattr(flat, "decoded_tokens", None)
    total_logprobs = len(raw_token_ids)

    if token > 0 and fork_info_j is not None:
        stride_k = total_logprobs // len(fork_info_j)

        for b_idx in range(len(fork_info_j)):
            current_beam = instance.beams[b_idx]
            start_offset = b_idx * stride_k
            end_offset = start_offset + stride_k

            if is_constrained:
                token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_constrained_segment(
                    raw_token_ids,
                    raw_logprobs,
                    raw_ranks,
                    raw_decoded,
                    start_offset,
                    end_offset,
                    logprobs_num,
                )
            else:
                token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_and_dedup_segment(
                    raw_token_ids,
                    raw_logprobs,
                    raw_ranks,
                    raw_decoded,
                    start_offset,
                    end_offset,
                    logprobs_num,
                )
            valid_token_ids = None
            if coordinator.enabled and not is_constrained:
                generated_tokens = _beam_full_tokens(current_beam)[instance.prompt_len :]
                valid_token_ids = coordinator.valid_next_token_ids(generated_tokens)

            coordinator.add_parent_candidates(
                candidate_builder,
                parent_index=b_idx,
                parent_cumulative_logprob=current_beam.cum_logprob,
                token_ids=token_ids_pos,
                logprobs=logprobs_pos,
                ranks=ranks_pos,
                decoded_tokens=decoded_pos,
                valid_token_ids=valid_token_ids,
                candidate_strategy=catalog_strategy,
            )
    else:
        if is_constrained:
            token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_constrained_flat_logprobs(
                flat, logprobs_num
            )
        else:
            token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_and_dedup_flat_logprobs(
                flat, logprobs_num
            )
        valid_token_ids = None
        if coordinator.enabled and not is_constrained:
            current_beam = instance.beams[0]
            generated_tokens = _beam_full_tokens(current_beam)[instance.prompt_len :]
            valid_token_ids = coordinator.valid_next_token_ids(generated_tokens)

        coordinator.add_parent_candidates(
            candidate_builder,
            parent_index=0,
            parent_cumulative_logprob=instance.beams[0].cum_logprob,
            token_ids=token_ids_pos,
            logprobs=logprobs_pos,
            ranks=ranks_pos,
            decoded_tokens=decoded_pos,
            valid_token_ids=valid_token_ids,
            candidate_strategy=catalog_strategy,
        )

    candidate_batch = candidate_builder.build()
    if not return_arrays:
        return (
            False,
            candidate_batch.token_ids.tolist(),
            candidate_batch.cumulative_logprobs.tolist(),
            candidate_batch.parent_indices.tolist(),
            candidate_batch.parent_cache,
        )
    # Keep the arrays produced by the builder for the internal hot path.
    return (
        False,
        candidate_batch.token_ids,
        candidate_batch.cumulative_logprobs,
        candidate_batch.parent_indices,
        candidate_batch.parent_cache,
    )


def _custom_beam_search_batch(
    llm_engine,
    instances_batch: list[BeamSearchInstance],
    token_iter,
    prompt_start: int,
    tokenizer,
    beam_width: int,
    beam_search_params: SamplingParams,
    sid_end_token_id: int | None,
    ignore_eos: bool,
    initial_logprobs: "FlatLogprobs",
    logprobs_num: int,
    catalog: Any | None = None,
    catalog_strategy: str | None = None,
    constraint_coordinator: BeamConstraintCoordinator | None = None,
    max_decode_steps: int = 0,
) -> None:
    """Run beam search in batch mode for custom attention backend."""
    # Stage 0 form guard: with async scheduling the engine runs the
    # two-phase step_with_batch_queue pipeline, but this loop's per-step
    # Request lifecycle (max_tokens=1, finishes same-step, engine-driven
    # chaining) keeps exactly one batch in flight — the pipeline cannot
    # overlap and the M2 inline-sample fast path is disabled on top
    # (_should_inline_sample_tokens requires use_async_scheduling=False).
    # Net effect is a regression, and the combination is unvalidated;
    # fail fast instead.  The Stage 1 resident-Request form (see
    # ``GRLLM.async_beam_search``) is the async-scheduling
    # entry point.
    scheduler_config = getattr(
        getattr(llm_engine, "vllm_config", None),
        "scheduler_config",
        None,
    )
    if getattr(scheduler_config, "async_scheduling", False):
        raise ValueError(
            "vllm-gr beam_search (CUSTOM attention) does not support "
            "--async-scheduling in the Stage 0 per-step form; it regresses "
            "both the batch pipeline and the inline-sample fast path. "
            "Disable async scheduling, or use the Stage 1 resident-Request "
            "async_beam_search form."
        )
    instance_prefill_id: dict[int, str] = {}
    instance_prev_beam_internal_ids: dict[int, list[str]] = {}
    instance_fork_info: dict[int, ForkInfo] = {}
    instance_prefix_len: dict[int, int] = {}
    instance_completed_sessions: set[int] = set()
    # Per-instance: EngineCore chained this step's successor from the worker's
    # beam decision (engine-driven); the next iteration must skip its own
    # beam_request_step_update send (the session request is still running
    # under the same session id — a second ADD trips the scheduler's
    # "duplicate request id" assert).
    instance_self_advance_chained: set[int] = set()

    # Resolve the frontend catalog-strategy per backend (Trie: None → "fill";
    # Constraint Tables: an explicit value is a contradiction and is rejected).
    if constraint_coordinator is not None:
        catalog_strategy = constraint_coordinator.resolve_catalog_strategy(catalog_strategy)

    # Constraint Tables generate candidates in the Worker; the decode-step
    # update must carry the constraint_step so the engine can route the raw
    # logits through the constrained top-k operator.
    _is_constrained = constraint_coordinator is not None and isinstance(
        constraint_coordinator.strategy, ConstraintTableStrategy
    )

    for token in token_iter:
        # Check if all instances are done.  A terminal-short-circuit
        # instance keeps its rebuilt final beams but is in
        # instance_completed_sessions — count it as done.
        all_done = True
        for j, instance in enumerate(instances_batch):
            if len(instance.beams) > 0 and j not in instance_completed_sessions:
                all_done = False
                break
        if all_done:
            break

        # Step 1: Prepare and update requests
        engine_core_requests = _prepare_beam_step_requests(
            llm_engine=llm_engine,
            instances_batch=instances_batch,
            token=token,
            prompt_start=prompt_start,
            tokenizer=tokenizer,
            beam_search_params=beam_search_params,
            sid_end_token_id=sid_end_token_id,
            instance_prefill_id=instance_prefill_id,
            instance_prev_beam_internal_ids=instance_prev_beam_internal_ids,
            instance_fork_info=instance_fork_info,
            instance_prefix_len=instance_prefix_len,
            instance_completed_sessions=instance_completed_sessions,
            max_decode_steps=max_decode_steps,
            is_constrained=_is_constrained,
            instance_self_advance_chained=instance_self_advance_chained,
        )

        if engine_core_requests:
            llm_engine.engine_core.add_requests(engine_core_requests, force_batch=True)
            for ec_req in engine_core_requests:
                parts = ec_req.external_req_id.split("-")
                j = int(parts[2]) - prompt_start
                instance_prefill_id[j] = ec_req.request_id
                instance_prev_beam_internal_ids[j] = [ec_req.request_id]

        # Step 2: Step engine and collect outputs
        outputs_map = _step_engine_and_collect_outputs(
            llm_engine=llm_engine,
            instances_batch=instances_batch,
            token=token,
            prompt_start=prompt_start,
            instance_prefill_id=instance_prefill_id,
        )

        # Step 3: Process outputs for each instance
        for j, instance in enumerate(instances_batch):
            if len(instance.beams) == 0:
                continue

            session_id = instance_prefill_id[j]
            search_id = f"offline-beam-{prompt_start + j}-0" if token == 0 else session_id
            result = outputs_map.get(search_id)

            if result is None:
                # Engine-driven chaining can run the whole session ahead of
                # the frontend (chained steps produce outputs under the
                # session id only).  With no result for this instance this
                # iteration there is nothing to consume; a later iteration
                # (or the terminal short-circuit there) picks it up.
                continue

            # Worker-side beam decision: when the accelerator
            # already selected the surviving beams, use its winners directly
            # and skip the frontend flat-logprobs parsing + NumPy top-k.  The
            # downstream EOS handling / beam materialisation is unchanged.
            from vllm_gr.entrypoints.openai.serving_engine import (
                _worker_decision_flatten,
                _worker_decision_from_result,
                _worker_final_sequences,
            )

            worker_decision = _worker_decision_from_result(result)

            # Terminal short-circuit (worker-sinking Stage 0): a terminal
            # decision carries the full cumulative state (out_sequence
            # [1, W, steps] + out_log_probs [1, W]) accumulated on the
            # device across every decode step.  Rebuild the instance's
            # beams from it in one shot and retire the session — the
            # per-step materialize / EOS-ledger / parent-pointer chain and
            # all further step iterations for this instance are skipped.
            # EOS is constrained to ignore_eos (the worker decision's step
            # count is authoritative).
            final_state = _worker_final_sequences(worker_decision or {})
            if final_state is not None:
                sequences, scores = final_state
                seed = instance.beams[0]
                new_beams = []
                for beam_tokens, beam_score in zip(sequences, scores):
                    beam = BeamSearchSequence(
                        orig_prompt=seed.orig_prompt,
                        tokens=list(seed.tokens[: instance.prompt_len]) + list(beam_tokens),
                        # Per the settled design: per-step logprobs are not
                        # reconstructible without parent pointers; the
                        # terminal path degrades to cum_logprob only.
                        logprobs=None,
                        cum_logprob=beam_score,
                        lora_request=seed.lora_request,
                    )
                    new_beams.append(beam)
                instance.beams = new_beams
                _beam_request_cleanup(
                    llm_engine,
                    session_id,
                    instance_prev_beam_internal_ids[j],
                )
                instance_completed_sessions.add(j)
                # A chained flag on a terminal decision is meaningless (no
                # successor exists); make sure no stale chained marker
                # survives into any later iteration.
                instance_self_advance_chained.discard(j)
                continue

            worker_candidates = None
            if worker_decision is not None:
                worker_candidates = _worker_decision_flatten(
                    worker_decision, beam_width, instance.beams
                )
            if (
                worker_candidates is None
                and worker_decision is not None
                and bool(worker_decision.get("is_last_step", False))
            ):
                # Terminal decision WITHOUT a usable out_sequence (CUDA
                # compact-buffer transport, malformed payload): fall back
                # to the per-step path for this step so the frontend
                # finishes the session from the per-candidate logprobs.
                logger.warning(
                    "vllm-gr: terminal worker decision lacks out_sequence; "
                    "falling back to the per-step materialize path"
                )
            # EngineCore chained this step's successor from the worker's
            # decision (engine-driven): the next iteration skips its
            # beam_request_step_update send. A missing flag (terminal
            # decision / skip path / engine-driven disabled) keeps the
            # frontend-driven path in charge.
            if worker_decision is not None and worker_decision.get("beam_self_advance_chained"):
                instance_self_advance_chained.add(j)
            else:
                instance_self_advance_chained.discard(j)

            if worker_candidates is not None:
                (
                    all_beams_token_id,
                    all_beams_logprob,
                    all_beams_beam_idx_arr,
                    beam_flat_cache,
                ) = worker_candidates
                if not all_beams_token_id:
                    is_error = True
                else:
                    is_error = False
                    all_beams_token_id = np.asarray(all_beams_token_id, dtype=np.int64)
                    all_beams_logprob = np.asarray(all_beams_logprob, dtype=np.float64)
                    all_beams_beam_idx_arr = np.asarray(all_beams_beam_idx_arr, dtype=np.intp)
            else:
                (
                    is_error,
                    all_beams_token_id,
                    all_beams_logprob,
                    all_beams_beam_idx,
                    beam_flat_cache,
                ) = _parse_step_logprobs(
                    result=result,
                    instance=instance,
                    token=token,
                    logprobs_num=logprobs_num,
                    fork_info_j=instance_fork_info.get(j),
                    catalog=catalog,
                    catalog_strategy=catalog_strategy,
                    constraint_coordinator=constraint_coordinator,
                    return_arrays=True,
                )
                all_beams_beam_idx_arr = all_beams_beam_idx

            if is_error:
                _beam_request_cleanup(
                    llm_engine,
                    session_id,
                    instance_prev_beam_internal_ids[j],
                )
                instance_completed_sessions.add(j)
                instance.beams = []
                continue

            # _parse_step_logprobs returns the builder's NumPy arrays for the
            # internal path; do not convert them through Python lists again.
            eos_beams: list[BeamSearchSequence] = []
            if not ignore_eos:
                eos_beams, all_beams_logprob = complete_eos_candidates(
                    parents=instance.beams,
                    token_ids=all_beams_token_id,
                    scores=all_beams_logprob,
                    beam_idx=all_beams_beam_idx_arr,
                    beam_flat_cache=beam_flat_cache,
                    stop_token_ids={tokenizer.eos_token_id},
                    initial_logprobs=initial_logprobs,
                    include_eos_token=True,
                    inherit_lora_request=True,
                    share_prompt_prefix=True,
                )
                instance.completed.extend(eos_beams)

            # A worker decision already contains exactly W survivors in the
            # order used by its device-resident KV state. Keep that order.
            topn_idx = (
                np.arange(all_beams_token_id.size, dtype=np.intp)
                if worker_candidates is not None and not eos_beams
                else select_top_indices(
                    all_beams_logprob,
                    beam_width,
                    catalog_strategy,
                    exclude_nonfinite=bool(eos_beams),
                )
            )
            new_beams, fork_info = materialize_selected_beams(
                parents=instance.beams,
                selected_indices=topn_idx,
                token_ids=all_beams_token_id,
                scores=all_beams_logprob,
                beam_idx=all_beams_beam_idx_arr,
                beam_flat_cache=beam_flat_cache,
                initial_logprobs=initial_logprobs,
                share_prompt_prefix=True,
            )
            instance.beams = new_beams
            instance_fork_info[j] = fork_info

    # Step 4: Cleanup completed sessions
    for j, instance in enumerate(instances_batch):
        if j not in instance_completed_sessions and j in instance_prefill_id:
            _beam_request_cleanup(
                llm_engine,
                instance_prefill_id[j],
                instance_prev_beam_internal_ids[j],
            )
            instance_completed_sessions.add(j)


def _install_renderer_eos_token_id_cache() -> None:
    """Cache ``Renderer.get_eos_token_id`` on the tokenizer object.

    vLLM's renderer re-reads ``tokenizer.eos_token_id`` on every request;
    for HF fast tokenizers that property re-runs
    ``convert_tokens_to_ids`` through the token-pool borrow (~0.07ms per
    request).  The value is constant for a tokenizer's lifetime, so the
    first read is cached directly on the tokenizer instance (no global
    state, no leak, lifetime tied to the tokenizer).  Idempotent; a failed
    attribute write (exotic tokenizer hosts) silently falls back to the
    uncached path.
    """
    from vllm.renderers.base import BaseRenderer

    if getattr(BaseRenderer.get_eos_token_id, "_gr_eos_cache", False):
        return

    original = BaseRenderer.get_eos_token_id

    def get_eos_token_id_cached(self):
        tok = self.tokenizer
        if tok is None:
            return original(self)
        cached = getattr(tok, "_gr_eos_token_id", None)
        if cached is not None:
            return cached
        value = original(self)
        if value is not None:
            try:
                object.__setattr__(tok, "_gr_eos_token_id", value)
            except (AttributeError, TypeError):
                pass
        return value

    get_eos_token_id_cached._gr_eos_cache = True  # type: ignore[attr-defined]
    BaseRenderer.get_eos_token_id = get_eos_token_id_cached


def _custom_async_beam_search_batch(
    llm_engine,
    instances_batch: list["BeamSearchInstance"],
    prompt_start: int,
    tokenizer,
    beam_width: int,
    beam_search_params: "SamplingParams",
    sid_end_token_id: int | None,
    max_decode_steps: int,
    is_constrained: bool = False,
) -> None:
    """Stage 1 single-Request form: submit once, consume only the terminal.

    Per instance:
    1. ADD_BATCH the prefill and cache a ``single_request=True`` session
       template (BEAM_SESSION_TEMPLATE).  The prefill ADD carries the WHOLE
       session lifecycle (``max_tokens = max_decode_steps``,
       ``ignore_eos=True``), so it terminates naturally via ``check_stop``
       (FINISHED_LENGTH_CAPPED) after ``loop_len`` output tokens.  EngineCore
       suppresses the engine-driven prefill→step0 ADD chain — the upstream
       AsyncScheduler schedules every decode step as an ordinary cached-reqs
       step (1 token/step) on that SAME Request, while the worker overwrites
       input_ids/positions from the device beam decision (the 9.diff
       worker-overwrite model) and seeds its decision buffers at prefill
       sample time.
    2. Drain ``llm_engine.step()`` outputs.  Intermediate steps carry only
       the per-step decision — nothing to consume.  The terminal decision
       (``is_last_step=True`` + ``out_sequence``) rebuilds the instance's
       beams in one shot (Stage 0 terminal short-circuit, reused verbatim).
    3. Cleanup via the standard ``beam_width=0`` update.

    The Request never finishes early, so its OutputProcessor state is never
    popped mid-session — no frontend re-registration is needed (the prefill
    and the decode steps all run under the same id).  This loop never sends a
    per-step step update; the session's decisions are marked
    ``beam_self_advance_chained`` by EngineCore, so the frontend per-step
    protocol is bypassed entirely.
    """
    from vllm_gr.entrypoints.openai.serving_engine import (
        _worker_decision_from_result,
        _worker_final_sequences,
    )

    instance_prefill_id: dict[int, str] = {}
    # Frontend RequestOutputs arrive under the EXTERNAL request id; map them
    # back to the batch index (the internal session id never leaves the
    # EngineCore for the sync offline path).
    instance_external_id: dict[str, int] = {}
    instance_completed: set[int] = set()
    # Internal session id -> batch index: decode-step outputs and the terminal
    # decision ride the same Request, but a RequestOutput may surface under
    # either the external or the internal id depending on the async-scheduling
    # iteration; match both.
    instance_session_id: dict[str, int] = {}

    # ---- Step 1: one submission per instance ----
    engine_core_requests = []
    templates: dict[str, "BeamRequestStepUpdate"] = {}
    for j, instance in enumerate(instances_batch):
        if len(instance.beams) == 0:
            instance_completed.add(j)
            continue
        request_id_item = f"offline-beam-{prompt_start + j}-0"
        tokens_prompt = _create_tokens_prompt_from_beam(instance.beams[0])
        ec_req = llm_engine.input_processor.process_inputs(
            request_id=request_id_item,
            prompt=tokens_prompt,
            params=beam_search_params,
            supported_tasks=llm_engine.get_supported_tasks(),
            arrival_time=time.time(),
            lora_request=instance.beams[0].lora_request,
        )
        llm_engine.input_processor.assign_request_id(ec_req)
        llm_engine.output_processor.add_request(ec_req, None, None, 0)
        engine_core_requests.append(ec_req)
        # instance.beams[0].tokens is the full prefill prompt (incl. optional
        # sid_begin); prefix_len must mirror the per-step form's step-0 value.
        prefix_len = len(instance.beams[0].tokens)
        # assign_request_id swapped in the internal id (external-uuid suffix);
        # the EngineCore session (beam_cache key, Request.request_id) runs
        # under that internal id, so the template / cleanup / output matching
        # must all use it.  The frontend RequestOutput arrives under the
        # EXTERNAL id, tracked separately below.
        session_id = ec_req.request_id
        instance_prefill_id[j] = session_id
        instance_session_id[session_id] = j
        instance_external_id[ec_req.external_req_id] = j
        # Resident template: the fields EngineCore cannot derive itself when
        # chaining step-0 from the worker's prefill decision, plus the
        # Stage-1 ``resident`` marker that makes the chained step-0 Request
        # long-lived (core.py _apply_beam_step resident branch).
        decode_sampling_params = clone_beam_decode_sampling_params(beam_search_params)
        eos_token_id = (
            sid_end_token_id if sid_end_token_id is not None else tokenizer.eos_token_id
        )
        templates[session_id] = BeamRequestStepUpdate(
            session_id=session_id,
            parent_beam_ids=[],
            child_beam_ids=[],
            beam_tokens=[],
            pruned_ids=[],
            prefix_len=prefix_len,
            beam_width=beam_width,
            sampling_params=decode_sampling_params,
            eos_token_id=eos_token_id,
            lora_request=instance.beams[0].lora_request,
            max_decode_steps=max_decode_steps,
            # Constraint Tables: 1 = constrained decode step 0 (the value the
            # per-step frontend puts on the step-0 update).
            constraint_step=1 if is_constrained else 0,
            resident=True,
            # Single-Request form: the prefill ADD carries the whole session
            # (max_tokens=max_decode_steps); EngineCore suppresses the
            # prefill→step0 ADD chain and the frontend never re-registers.
            single_request=True,
        )

    if not engine_core_requests:
        return

    # Send the session template BEFORE the ADD_BATCH: an unconstrained NPU
    # prefill must be recognisable as a beam-session prefill at
    # _cache_beam_request time so the worker prefill-decision gate
    # (is_beam_prefill metadata) can fire.  The EngineCore's out-of-order
    # template handling (stub entry preserved by _cache_beam_request) makes
    # either order correct; template-first keeps the cache hit path.
    for session_id, template in templates.items():
        llm_engine.engine_core.beam_session_template(template)
    llm_engine.engine_core.add_requests(engine_core_requests, force_batch=True)

    # ---- Step 2: drain outputs until every session reports its terminal ----
    while len(instance_completed) < len(instances_batch):
        step_outputs = llm_engine.step()
        if not step_outputs:
            # An async-scheduling step can legitimately produce no frontend
            # output (schedule-only iteration of step_with_batch_queue);
            # keep draining until the terminal decision surfaces.
            continue
        for out in step_outputs:
            j = instance_external_id.get(out.request_id)
            if j is None:
                # Not the external prefill id: decode-step outputs and the
                # terminal decision arrive under the internal session id.
                j = instance_session_id.get(out.request_id)
            if j is None or j in instance_completed:
                # Not one of this batch's sessions (or an already-retired
                # session's late duplicate output) — drop it.
                continue
            worker_decision = _worker_decision_from_result(out)
            final_state = _worker_final_sequences(worker_decision or {})
            if final_state is None:
                # Non-terminal per-step output: the single-Request form has
                # nothing to consume (no materialize / EOS ledger on this path).
                continue
            sequences, scores = final_state
            seed = instances_batch[j].beams[0]
            new_beams = []
            for beam_tokens, beam_score in zip(sequences, scores):
                new_beams.append(
                    BeamSearchSequence(
                        orig_prompt=seed.orig_prompt,
                        tokens=list(seed.tokens) + list(beam_tokens),
                        # Settled design §7 ②: the terminal path degrades to
                        # cum_logprob-only; per-step logprobs are not
                        # reconstructible without the parent-pointer chain.
                        logprobs=None,
                        cum_logprob=beam_score,
                        lora_request=seed.lora_request,
                    )
                )
            instances_batch[j].beams = new_beams
            _beam_request_cleanup(llm_engine, instance_prefill_id[j], [instance_prefill_id[j]])
            instance_completed.add(j)

    # ---- Step 3: safety-net cleanup (error / fallback paths) ----
    for j, session_id in instance_prefill_id.items():
        if j not in instance_completed:
            _beam_request_cleanup(llm_engine, session_id, [session_id])
            instance_completed.add(j)


class GRLLM(LLM):
    """LLM subclass with catalog-aware beam search.

    Construction is fully delegated to :class:`vllm.entrypoints.llm.LLM`; the
    only added state is the optional ``catalog`` used to constrain beam-search
    token sequences.
    """

    def __init__(
        self,
        model: str = "",
        *,
        catalog_path: str | None = None,
        constraint_backend: ConstraintBackend | str | None = None,
        beam_graph_enabled: bool | None = None,
        beam_max_width: int | None = None,
        beam_max_decode_steps: int | None = None,
        beam_max_batch_size: int | None = None,
        beam_async: bool = False,
        **kwargs,
    ):
        beam_max_width = validate_beam_max_width(beam_max_width)
        if beam_max_batch_size is not None:
            beam_max_batch_size = validate_beam_max_batch_size(beam_max_batch_size)
        from vllm_gr.config.validation import validate_beam_async

        beam_async = validate_beam_async(beam_async)
        # vLLM 0.22 enables ModelRunner V2 automatically for Qwen3. Apply the
        # vllm-gr default before LLM.__init__ constructs VllmConfig so the V1
        # beam-search worker patches remain active.
        from vllm_gr._env_check import configure_model_runner_default

        configure_model_runner_default()

        # Ensure the vllm-gr runtime patches are installed (EngineArgs
        # arg-threading, worker hooks, EngineCore entrypoint) before building
        # EngineArgs.  The vLLM general-plugin loader normally does this at
        # process start, but offline GRLLM construction can precede that;
        # calling it explicitly is idempotent and self-contained.
        from vllm_gr.plugin import initialize_runtime

        initialize_runtime()
        # Force the EngineCore process entrypoint swap: re_apply_patches may
        # skip an already-marked batch_and_fork patch (e.g. when the general
        # plugin ran first), and without the swap a spawn child re-imports the
        # original run_engine_core and loses every worker hook.
        from vllm_gr.entrypoints.openai.beam_search_patch import (
            patch_engine_core_process_entrypoint,
        )

        patch_engine_core_process_entrypoint()

        self._shutdown_complete = False

        # Thread the GR values into the EngineArgs that LLM.__init__ builds
        # internally (EngineArgs(model=..., **kwargs)); the arg_utils_gr patch
        # stashes them and create_engine_config writes them into VllmConfig so
        # EngineCore can learn the constraint backend before any request runs.
        # Without this, the offline Python API compiles/injects the tables but
        # EngineCore never marks requests constraint_prefill/constrained and the
        # Worker raw-logits hook would never run.
        #
        # Only constraint_table strictly requires the patch: the tables must
        # reach VllmConfig so every worker can run the raw-logits hook.  A
        # Trie catalog is handled entirely driver-side (TrieConstraintStrategy
        # below), so ``catalog_path`` with the default backend must not demand
        # the plugin — vLLM 0.22 loads general plugins lazily from inside
        # EngineArgs.__init__, which has not run yet at this point.
        # Resolve the effective constraint backend here (before EngineArgs is
        # built): on NPU with no explicit backend choice and a catalog
        # available the platform default is Constraint Tables.  A user
        # catalog_path is kept as the constraint data source (a Triples
        # document for the promoted backend).  Without a catalog there is no
        # constraint data to promote against — a GRLLM constructed with no
        # catalog must stay unconstrained (legacy behavior) rather than force
        # the tokenizer-based Constraint Table builder, which fails fast on
        # tokenizers whose s_a/s_b/s_c groups exceed max_valid_items (e.g.
        # OneRec 8192 > 500) and regresses catalog-free single-prompt tests.
        # Explicit user choices (any backend) pass through unchanged.
        from vllm_gr.engine.arg_utils_gr import (
            _catalog_is_triples_document,
            _is_npu_platform,
        )

        if (
            constraint_backend is None
            and catalog_path is not None
            and _catalog_is_triples_document(catalog_path)
            and beam_max_width is not None
            and _is_npu_platform()
        ):
            constraint_backend = ConstraintBackend.CONSTRAINT_TABLE

        gr_engine_args: dict[str, object] = {}
        if engine_args_accepts_gr_values():
            gr_engine_args["catalog_path"] = catalog_path
            gr_engine_args["constraint_backend"] = constraint_backend
        elif constraint_backend not in (None, ConstraintBackend.TRIE):
            raise RuntimeError(
                "vllm-gr runtime patches are not active: EngineArgs cannot "
                "carry constraint_backend into VllmConfig. Ensure the vllm-gr "
                "plugin is registered (vllm general_plugins) or call "
                "vllm_gr.plugin.initialize_runtime() before constructing GRLLM "
                "with constraint_backend='constraint_table'."
            )

        # Beam-graph settings ride vLLM's native additional_config so they
        # survive into the spawned engine-core process (same channel as the
        # serve CLI flags; see vllm_gr/engine/arg_utils_gr.py).
        beam_updates: dict[str, object] = {}
        if beam_graph_enabled is not None:
            beam_updates["vllm_gr_beam_graph_enabled"] = beam_graph_enabled
        if beam_max_width is not None:
            beam_updates["vllm_gr_beam_max_width"] = beam_max_width
        if beam_max_decode_steps is not None:
            beam_updates["vllm_gr_beam_max_decode_steps"] = beam_max_decode_steps
        if beam_max_batch_size is not None:
            beam_updates["vllm_gr_beam_max_batch_size"] = beam_max_batch_size
        if beam_async:
            beam_updates["vllm_gr_beam_async"] = True
        if beam_updates:
            additional_config = dict(kwargs.pop("additional_config", None) or {})
            additional_config.update(beam_updates)
            kwargs["additional_config"] = additional_config

        super().__init__(
            model=model,
            **gr_engine_args,
            **kwargs,
        )

        _install_renderer_eos_token_id_cache()

        self.catalog = None
        self.constraint_strategy = None
        self.constraint_coordinator = BeamConstraintCoordinator(None)
        # Resolve the effective constraint backend from the engine config
        # (create_engine_config already applied the NPU platform default: a
        # sentinel catalog_path when Constraint Tables are auto-selected).
        # Explicit user choices pass through unchanged.
        resolved_gr = GRConfig.from_vllm_config(getattr(self.llm_engine, "vllm_config", None))
        effective_backend = resolved_gr.constraint_backend
        effective_catalog_path = resolved_gr.catalog_path
        canonical_table = resolved_gr.constraint_table
        if canonical_table is not None and canonical_table.enabled:
            # Canonical artifacts are loaded independently by every worker.
            # The frontend strategy is only the request-routing marker that
            # carries constraint_step/beam_suffix through the beam protocol.
            self.constraint_strategy = ConstraintTableStrategy(canonical_table.path or "")
            self.constraint_strategy.digest = canonical_table.artifact_digest
            self.constraint_coordinator = BeamConstraintCoordinator(self.constraint_strategy)
        elif effective_catalog_path is not None:
            self.constraint_strategy = create_constraint_strategy(
                backend=effective_backend,
                tokenizer=self.get_tokenizer(),
                catalog_path=effective_catalog_path,
            )
            self.constraint_coordinator = BeamConstraintCoordinator(self.constraint_strategy)
            if isinstance(self.constraint_strategy, TrieConstraintStrategy):
                self.catalog = self.constraint_strategy.catalog
            elif isinstance(self.constraint_strategy, ConstraintTableStrategy):
                # Constraint Tables compile on the driver and are broadcast to
                # every worker before any beam-search step runs.  Failing here
                # is a hard startup error — never a silent fall back to Trie.
                model_config = self.llm_engine.model_config
                get_vocab_size = getattr(model_config, "get_vocab_size", None)
                vocab_size = (
                    get_vocab_size() if callable(get_vocab_size) else model_config.vocab_size
                )
                resource = install_constraint_table(
                    self.llm_engine,
                    catalog_path=effective_catalog_path,
                    tokenizer=self.get_tokenizer(),
                    vocab_size=vocab_size,
                )
                self.constraint_strategy.digest = resource.digest

    def beam_search_v1(
        self,
        prompts: list[Union["TokensPrompt", "TextPrompt"]],
        params: "BeamSearchParams",
        lora_request: Union[list["LoRARequest"], "LoRARequest", None] = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
    ) -> list["BeamSearchOutput"]:
        """Run Beam Search through the new EngineCore-owned V1 pipeline."""
        from vllm_gr.config import BeamExecutionMode
        from vllm_gr.config.execution import require_beam_execution_mode

        require_beam_execution_mode(
            getattr(self.llm_engine, "vllm_config", None), BeamExecutionMode.V1
        )
        from vllm_gr.entrypoints.beam_search_v1 import run_beam_search_v1
        from vllm_gr.entrypoints.beam_search_v1.protocol import OfflineBeamSearchRequest

        call = OfflineBeamSearchRequest(
            prompts=prompts,
            params=params,
            lora_request=lora_request,
            use_tqdm=use_tqdm,
            concurrency_limit=concurrency_limit,
        )
        return cast(
            list["BeamSearchOutput"],
            run_beam_search_v1(self._create_beam_search_v1_client(), call),
        )

    def _create_beam_search_v1_client(self) -> "BeamSearchFrontend":
        """Create the offline A2 request/session adapter."""
        from vllm_gr.entrypoints.beam_search_v1.client import (
            BeamSearchClient,
        )

        return BeamSearchClient(self)

    @overload
    def beam_search(
        self,
        prompts: list[Union["TokensPrompt", "TextPrompt"]],
        params: "BeamSearchParams",
        lora_request: Union[list["LoRARequest"], "LoRARequest", None] = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
        decode_prompt_in_text: bool = True,
        *,
        output_mode: Literal["full"] = "full",
    ) -> list["BeamSearchOutput"]: ...

    @overload
    def beam_search(
        self,
        prompts: list[Union["TokensPrompt", "TextPrompt"]],
        params: "BeamSearchParams",
        lora_request: Union[list["LoRARequest"], "LoRARequest", None] = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
        decode_prompt_in_text: bool = True,
        *,
        output_mode: Literal["compact"],
    ) -> list[CompactBeamSearchOutput]: ...

    @frontend_request_scope
    def beam_search(
        self,
        prompts: list[Union["TokensPrompt", "TextPrompt"]],
        params: "BeamSearchParams",
        lora_request: Union[list["LoRARequest"], "LoRARequest", None] = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
        decode_prompt_in_text: bool = True,
        *,
        output_mode: Literal["full", "compact"] = "full",
    ) -> list["BeamSearchOutput"] | list[CompactBeamSearchOutput]:
        """
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            use_tqdm: Whether to use tqdm to display the progress bar.
            concurrency_limit: The maximum number of concurrent requests.
                If None, the number of concurrent requests is unlimited.
            decode_prompt_in_text: Include the original prompt in sequence text.
                When False, return only generated text, including SID markers.
                Ignored in compact mode, which does not produce text.
            output_mode: "full" preserves the native, lazily materialized output.
                "compact" returns CPU token_ids and cumulative_logprob
                without reconstructing full tokens, detailed logprobs, or text.
                Generated tokens include configured begin/end markers and match
                the suffix after the original prompt in full mode.
        """
        # Lazy imports for beam search dependencies
        import itertools

        from vllm_gr.config import BeamExecutionMode
        from vllm_gr.config.execution import require_beam_execution_mode

        if output_mode not in ("full", "compact"):
            raise ValueError(f"output_mode must be 'full' or 'compact', got {output_mode!r}")

        require_beam_execution_mode(
            getattr(self.llm_engine, "vllm_config", None), BeamExecutionMode.LEGACY
        )

        # TODO: how does beam search work together with length penalty,
        # frequency, penalty, and stopping criteria, etc.?
        logger.debug("Called beam_search")
        # Defer the full gen2 scan: it re-walks ~900K process-static
        # objects (torch/engine funcs, tuples, dicts) and costs ~480ms
        # while reclaiming almost nothing (collected=70 over 90 reqs).
        # Raising only gen2 (gen0/gen1 keep current values) makes that
        # scan ~100x rarer and moves it out of the decode request window.
        _gc0, _gc1, _ = gc.get_threshold()
        gc.set_threshold(_gc0, _gc1, 1000)
        beam_width = validate_beam_width(params.beam_width)
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        begin_token = params.begin_token if hasattr(params, "begin_token") else None
        end_token = params.end_token if hasattr(params, "end_token") else None
        catalog_strategy = getattr(params, "catalog_strategy", None)

        tokenizer = self.get_tokenizer()
        # Token-id conversion cache on the GRLLM instance: convert_tokens_to_ids
        # goes through the HF token-pool borrow (~0.06ms per call) and the
        # begin/end special tokens are identical across every beam_search call
        # on the same tokenizer, so convert once and reuse.
        token_id_cache = getattr(self, "_gr_token_id_cache", None)
        if token_id_cache is None or token_id_cache[0] is not tokenizer:
            token_id_cache = (tokenizer, {})
            self._gr_token_id_cache = token_id_cache

        def _cached_convert_token(token: str) -> int | None:
            cache = token_id_cache[1]
            hit = cache.get(token, -2)  # -2 sentinel: not attempted yet
            if hit != -2:
                return hit
            converted = tokenizer.convert_tokens_to_ids(token)
            if converted is None or converted == -1:
                # Do not cache failures for tokens that a future call may
                # legitimately retry; only cache resolved ids.
                return None
            cache[token] = converted
            return converted

        sid_begin_token_id: int | None = None
        if begin_token is not None:
            sid_begin_token_id = _cached_convert_token(begin_token)
            if sid_begin_token_id is None:
                logger.warning(
                    f"begin_token must be a valid token in the tokenizer vocabulary, got token {begin_token}"
                )
                sid_begin_token_id = None
        sid_end_token_id: int | None = None
        if end_token is not None:
            sid_end_token_id = _cached_convert_token(end_token)
            if sid_end_token_id is None:
                logger.warning(
                    f"end_token must be a valid token in the tokenizer vocabulary, got token {end_token}"
                )
                sid_end_token_id = None

        if use_tqdm and concurrency_limit is not None:
            logger.warning(
                "Progress bar is not supported when using concurrency_limit. Disabling progress bar."
            )
            use_tqdm = False

        if concurrency_limit is None:
            concurrency_limit = len(prompts)

        logprobs_num = beam_width
        # generate beam_width candidates at each step
        beam_search_params = SamplingParams(
            logprobs=logprobs_num,
            max_tokens=1,
            temperature=temperature,
            skip_clone=True,  # Internal beam search, safe to skip clone
            detokenize=False,
            flat_logprobs=True,
        )
        # NOTE: FlatLogprobs intentionally passed where BeamSearchSequence
        # expects list[dict[int, Logprob]].  Intermediate logprobs are never
        # read as dicts; the final beam.logprobs is replaced during
        # reconstruction at the end.
        # This reference is shared read-only across all beams — do not mutate
        # after the initial append below.
        initial_logprobs = FlatLogprobs()
        if sid_begin_token_id is not None:
            initial_logprobs.append_fast([sid_begin_token_id], [0.0], [None], [None])

        instances: list[BeamSearchInstance] = []
        engine_inputs = self._preprocess_cmpl(prompts)
        lora_requests = self._lora_request_to_seq(lora_request, len(engine_inputs))

        for lora_req, prompt in zip(lora_requests, engine_inputs):
            if prompt["type"] == "embeds":
                raise NotImplementedError("Embedding prompt not supported for beam search")

            # Structural shallow copy instead of copy.deepcopy: the beam
            # pipeline only REBINDS keys on this dict (prompt_token_ids is
            # rebuilt via list(...) before append) and never mutates the
            # original list in place, so deep-copying the 1024-token list
            # paid ~6ms per request on the pre-prefill critical path for
            # no semantic benefit.  Nested enc_dec dicts are copied one
            # level for the same reason.
            engine_prompt = dict(prompt)
            if engine_prompt.get("type") == "enc_dec":
                engine_prompt["decoder_prompt"] = dict(engine_prompt["decoder_prompt"])
            decoder_prompt = (
                engine_prompt
                if engine_prompt["type"] != "enc_dec"
                else engine_prompt["decoder_prompt"]
            )
            prompt_tokens = decoder_prompt["prompt_token_ids"]

            if sid_begin_token_id is not None or sid_end_token_id is not None:
                prompt_tokens = list(prompt_tokens)
                if sid_begin_token_id is not None:
                    prompt_tokens.append(sid_begin_token_id)
                decoder_prompt["prompt_token_ids"] = prompt_tokens

            instance = BeamSearchInstance(
                engine_prompt,
                lora_request=lora_req,
                logprobs=None,
            )
            instance.beams[0].logprobs = initial_logprobs
            # Deferred logprobs: parent pointers on BeamSearchSequence (a plain
            # @dataclass without __slots__).  If upstream adds __slots__, these
            # dynamic attrs will break — upstream the fields if that happens.
            instance.beams[0]._lp_parent = None
            instance.beams[0]._lp_step_data = None
            # Shared-prefix representation: keep the full prompt in a read-only
            # _prompt_token_ids (shared by every descendant beam) and reset
            # `tokens` to the generated tail, so materialize only copies the
            # incremental tokens instead of the whole prompt each step.
            instance.beams[0]._prompt_token_ids = instance.beams[0].tokens
            instance.beams[0].tokens = []
            instance.prompt_len = (
                len(prompt_tokens) - 1 if sid_begin_token_id is not None else len(prompt_tokens)
            )
            instances.append(instance)

        # Cache prompt token references and lengths per instance.  The prompt
        # TEXT is detokenized lazily at output assembly (below, after
        # generation completes): decoding here would spend ~2.5ms of
        # tokenizer time on the pre-prefill critical path, and the text is
        # only ever consumed when concatenating beam.text at the very end.
        cached_prompts = []
        for instance in instances:
            prompt_tokens = instance.beams[0]._prompt_token_ids
            # SID-begin seeds the engine prompt but belongs to generated output.
            cached_prompts.append({"tokens": prompt_tokens, "len": instance.prompt_len})

        pre_calc = 0 if sid_begin_token_id is None else 1
        if sid_end_token_id is not None:
            pre_calc += 1
        if max_tokens - pre_calc < 0:
            logger.warning(
                "max_tokens is less than the number of tokens to be pre-calculated. "
                "Setting max_tokens to the number of pre-calculated tokens."
            )
            max_tokens = pre_calc
        if isinstance(self.constraint_strategy, ConstraintTableStrategy):
            # Constraint Tables are a fixed 3-level FSM; requesting more steps
            # than the Worker can run is a hard error, never a silent fallback.
            validate_constraint_generation_limits(
                beam_width=beam_width,
                max_generate_steps=max_tokens - pre_calc,
            )
        vllm_config = getattr(self.llm_engine, "vllm_config", None)
        attention_backend = None
        if vllm_config is not None and hasattr(vllm_config, "attention_config"):
            attention_backend = getattr(vllm_config.attention_config, "backend", None)
        use_custom_attn = attention_backend is not None and (
            "CUSTOM" in str(attention_backend).upper()
            or getattr(attention_backend, "name", "").upper() == "CUSTOM"
        )
        # Trie constraints are enforced in the frontend (catalog trie); the
        # baseline generate() loop already constrains candidates correctly and
        # returns per-beam FlatLogprobs with precise boundaries.  The CUSTOM
        # beam-request protocol's merged-flat / fixed-stride extraction deviates
        # from that path (regression: CI GPU pass@1 0.04), so route Trie
        # through the native loop.  Constraint Tables keep the CUSTOM path:
        # their candidates are produced by the Worker raw-logits hook, which is
        # only installed on the beam-request path (see validate_constraint_table_protocol).
        if self.constraint_coordinator is not None and isinstance(
            self.constraint_coordinator.strategy, TrieConstraintStrategy
        ):
            use_custom_attn = False
        cache_config = getattr(vllm_config, "cache_config", None)
        validate_beam_request_prefix_cache(
            beam_request_active=use_custom_attn,
            prefix_caching_enabled=getattr(cache_config, "enable_prefix_caching", None),
        )
        # Constraint Tables run in the Worker raw-logits hook, which is only
        # installed on the CUSTOM-attention beam-request path.  The baseline
        # generate() loop below cannot express constraint_step + beam_suffix,
        # so fail fast rather than silently producing uncorrected samples.
        validate_constraint_table_protocol(
            is_constraint_table=isinstance(self.constraint_strategy, ConstraintTableStrategy),
            beam_protocol_available=use_custom_attn,
            required="the CUSTOM attention backend",
            hint="Configure attention_config={'backend': 'CUSTOM'}.",
        )
        for prompt_start in range(0, len(prompts), concurrency_limit):
            instances_batch = instances[prompt_start : prompt_start + concurrency_limit]

            token_iter = range(max_tokens - pre_calc)
            if use_tqdm:
                token_iter = tqdm(token_iter, desc="Beam search", unit="token", unit_scale=False)
                logger.warning(
                    "The progress bar shows the upper bound on token steps and "
                    "may finish early due to stopping conditions. It does not "
                    "reflect instance-level progress."
                )
            if use_custom_attn:
                _custom_beam_search_batch(
                    llm_engine=self.llm_engine,
                    instances_batch=instances_batch,
                    token_iter=token_iter,
                    prompt_start=prompt_start,
                    tokenizer=tokenizer,
                    beam_width=beam_width,
                    beam_search_params=beam_search_params,
                    sid_end_token_id=sid_end_token_id,
                    ignore_eos=ignore_eos,
                    initial_logprobs=initial_logprobs,
                    logprobs_num=logprobs_num,
                    catalog=self.catalog,
                    catalog_strategy=catalog_strategy,
                    constraint_coordinator=self.constraint_coordinator,
                    max_decode_steps=max_tokens - pre_calc,
                )
            else:
                for _ in token_iter:
                    all_beams: list[BeamSearchSequence] = list(
                        sum((instance.beams for instance in instances_batch), [])
                    )
                    pos = [0] + list(
                        itertools.accumulate(len(instance.beams) for instance in instances_batch)
                    )
                    instance_start_and_end: list[tuple[int, int]] = list(zip(pos[:-1], pos[1:]))

                    if len(all_beams) == 0:
                        break

                    # create corresponding batch entries for prompt & optional lora
                    prompts_batch, lora_req_batch = zip(
                        *[
                            (_create_tokens_prompt_from_beam(beam), beam.lora_request)
                            for beam in all_beams
                        ]
                    )

                    # only runs for one step
                    # we don't need to use tqdm here
                    output = self.generate(
                        prompts_batch,
                        sampling_params=beam_search_params,
                        use_tqdm=False,
                        lora_request=lora_req_batch,
                    )

                    # Validate generated tokens using catalog if available
                    valid_tokens_sets: list[set[int]] = []
                    if self.constraint_coordinator.enabled:
                        for instance in instances_batch:
                            prompt_len = instance.prompt_len
                            for beam in instance.beams:
                                generated_tokens = _beam_full_tokens(beam)[prompt_len:]
                                valid_token_ids = self.constraint_coordinator.valid_next_token_ids(
                                    generated_tokens
                                )
                                assert valid_token_ids is not None
                                valid_tokens_sets.append(valid_token_ids)

                    for (start, end), instance in zip(instance_start_and_end, instances_batch):
                        # Gather all logprobs and cumulative scores for this instance
                        candidate_builder = ParentCandidateBuilder()
                        beam_outputs = output[start:end]  # outputs for this instance's beams

                        for i, (current_beam, result) in enumerate(
                            zip(all_beams[start:end], beam_outputs)
                        ):
                            if result.outputs[0].logprobs is not None:
                                flat = result.outputs[0].logprobs
                                token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                                    extract_and_dedup_flat_logprobs(flat, logprobs_num)
                                )
                                self.constraint_coordinator.add_parent_candidates(
                                    candidate_builder,
                                    parent_index=i,
                                    parent_cumulative_logprob=current_beam.cum_logprob,
                                    token_ids=token_ids_pos,
                                    logprobs=logprobs_pos,
                                    ranks=ranks_pos,
                                    decoded_tokens=decoded_pos,
                                    valid_token_ids=(
                                        valid_tokens_sets[start + i]
                                        if self.constraint_coordinator.enabled
                                        else None
                                    ),
                                    candidate_strategy=catalog_strategy,
                                )
                            else:
                                raise ValueError(
                                    "Beam search generation expected non-empty logprobs "
                                    "for each beam, but received 'None'. Please ensure "
                                    "that sampling parameters are configured to return "
                                    "logprobs when using beam search."
                                )

                        candidate_batch = candidate_builder.build()
                        all_beams_token_id = candidate_batch.token_ids
                        all_beams_logprob = candidate_batch.cumulative_logprobs
                        all_beams_beam_idx_arr = candidate_batch.parent_indices
                        beam_flat_cache = candidate_batch.parent_cache
                        # Handle EOS tokens
                        eos_beams: list[BeamSearchSequence] = []
                        if not ignore_eos:
                            eos_beams, all_beams_logprob = complete_eos_candidates(
                                parents=all_beams[start:end],
                                token_ids=all_beams_token_id,
                                scores=all_beams_logprob,
                                beam_idx=all_beams_beam_idx_arr,
                                beam_flat_cache=beam_flat_cache,
                                stop_token_ids={tokenizer.eos_token_id},
                                initial_logprobs=initial_logprobs,
                                include_eos_token=True,
                                inherit_lora_request=True,
                                share_prompt_prefix=True,
                            )
                            instance.completed.extend(eos_beams)

                        # Select top beams
                        topn_idx = select_top_indices(
                            all_beams_logprob,
                            beam_width,
                            catalog_strategy,
                            exclude_nonfinite=bool(eos_beams),
                        )
                        new_beams, _ = materialize_selected_beams(
                            parents=all_beams[start:end],
                            selected_indices=topn_idx,
                            token_ids=all_beams_token_id,
                            scores=all_beams_logprob,
                            beam_idx=all_beams_beam_idx_arr,
                            beam_flat_cache=beam_flat_cache,
                            initial_logprobs=initial_logprobs,
                            share_prompt_prefix=True,
                        )
                        instance.beams = new_beams

        outputs: list[BeamSearchOutput] = []
        compact_outputs: list[CompactBeamSearchOutput] = []
        for idx, instance in enumerate(instances):
            # Append sid_end_token_id only to active beams (not EOS-completed
            # beams that already terminated naturally).
            if sid_end_token_id is not None:
                for beam in instance.beams:
                    beam.tokens.append(sid_end_token_id)
            # Move remaining active beams into the completed list.
            instance.completed.extend(instance.beams)
            best_beams = select_top_beams(instance.completed, beam_width)

            if output_mode == "compact":
                # The begin marker seeds the shared prompt; beam.tokens already
                # holds only the generated tail. Copy this small suffix before
                # installing lazy fields, without retaining the search history.
                begin_ids = (sid_begin_token_id,) if sid_begin_token_id is not None else ()
                compact_outputs.append(
                    CompactBeamSearchOutput(
                        sequences=tuple(
                            CompactBeamSearchSequence(
                                token_ids=begin_ids + tuple(beam.tokens),
                                cumulative_logprob=float(beam.cum_logprob),
                            )
                            for beam in best_beams
                        )
                    )
                )
                continue

            prompt_cache = cached_prompts[idx]
            prompt_len = prompt_cache["len"]
            # Prompt detokenization runs through a per-tokenizer LRU: the
            # benchmark replays one prompt and production prefixes repeat, so
            # the ~1ms 1023-token decode is nearly always a cache hit.
            prompt_text = (
                _prompt_text_cached(tokenizer, prompt_cache["tokens"], prompt_len)
                if decode_prompt_in_text
                else ""
            )
            # Lazy beam materialization.  The request returns as soon as
            # select_top_beams has ranked the survivors; tokens/text/logprobs
            # are only reconstructed / detokenized / restored for beams the
            # caller actually reads (property descriptors route the first
            # access through a per-request materializer).  A caller
            # consuming top-k of 128 beams therefore skips (128-k) logprob
            # rebuilds + decodes + full-token copies, and the 1024-token
            # init-segment copies drop from 128/request to k.  Validation
            # harnesses that read every beam do so outside the timed
            # region, so results stay byte-identical either way.
            materializer = _BeamTailMaterializer(
                initial_logprobs=initial_logprobs,
                sid_end_token_id=sid_end_token_id,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                prompt_len=prompt_len,
            )
            for beam in best_beams:
                beam.__class__ = _LazyMaterializedBeamSequence
                beam._lazy_materializer = materializer
            outputs.append(BeamSearchOutput(sequences=best_beams))

        # The worker-side freeze only covers the EngineCore/worker process; in
        # the default multi-process deployment this frontend driver is a
        # separate process and its own gen-1 GC (heavy per-request beam
        # bookkeeping) would spike beam_search() every few requests.  Freeze
        # the frontend heap once after the first full beam request completes.
        from vllm_gr.utils.engine_lifecycle import schedule_frontend_gc_freeze

        schedule_frontend_gc_freeze()
        return compact_outputs if output_mode == "compact" else outputs

    def async_beam_search(
        self,
        prompts: list[Union["TokensPrompt", "TextPrompt"]],
        params: "BeamSearchParams",
        lora_request: Union[list["LoRARequest"], "LoRARequest", None] = None,
        concurrency_limit: int | None = None,
    ) -> list["BeamSearchOutput"]:
        """Stage 1 resident-Request beam search (async scheduling entry point).

        One submission per prompt (prefill ADD + resident session template);
        EngineCore chains step-0 from the worker's prefill decision and builds
        ONE long-lived Request per session (``max_tokens = decode steps``,
        ``ignore_eos=True``), the upstream AsyncScheduler drives every further
        decode step as a cached-reqs step, and the worker overwrites
        input_ids/positions from the device beam decision.  The frontend only
        consumes the terminal decision (``out_sequence`` + cumulative scores)
        — no per-step materialize, EOS ledger, or step-update RPC.

        Requires the ``beam_async`` deployment (``GRLLM(..., beam_async=True)``),
        which keeps async scheduling ON and is refused by the Stage 0
        ``beam_search`` per-step form.  Per the settled design (§7): EOS is
        constrained to ``ignore_eos`` (the worker decision's step count is
        authoritative) and ``logprobs`` degrade to ``cum_logprob`` only.

        Args:
            prompts: A list of prompts (text or token IDs).
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            concurrency_limit: Max concurrent resident sessions. If None,
                uses min(number of prompts, beam_max_batch_size).
        """
        logger.debug("Called async_beam_search")
        beam_width = validate_beam_width(params.beam_width)
        max_tokens = params.max_tokens
        temperature = params.temperature
        begin_token = params.begin_token if hasattr(params, "begin_token") else None
        end_token = params.end_token if hasattr(params, "end_token") else None

        # ---- Deployment guards ----
        vllm_config = getattr(self.llm_engine, "vllm_config", None)
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        gr_cfg = GRConfig.from_vllm_config(vllm_config)
        if not gr_cfg.beam_async:
            raise ValueError(
                "vllm-gr async_beam_search requires the Stage 1 deployment: "
                "construct GRLLM(..., beam_async=True) so async scheduling "
                "stays enabled for beam sessions."
            )
        if not getattr(scheduler_config, "async_scheduling", False):
            raise ValueError(
                "vllm-gr async_beam_search requires async scheduling "
                "(scheduler_config.async_scheduling); the beam_async "
                "deployment normally keeps it on — check startup warnings."
            )
        attention_backend = None
        if vllm_config is not None and hasattr(vllm_config, "attention_config"):
            attention_backend = getattr(vllm_config.attention_config, "backend", None)
        use_custom_attn = attention_backend is not None and (
            "CUSTOM" in str(attention_backend).upper()
            or getattr(attention_backend, "name", "").upper() == "CUSTOM"
        )
        if not use_custom_attn:
            raise ValueError(
                "vllm-gr async_beam_search requires the CUSTOM attention "
                "backend: attention_config={'backend': 'CUSTOM'}."
            )
        if self.constraint_coordinator is not None and isinstance(
            self.constraint_coordinator.strategy, TrieConstraintStrategy
        ):
            raise ValueError(
                "vllm-gr async_beam_search does not support Trie catalog "
                "constraints (frontend-driven candidate path); use "
                "constraint_table or the per-step beam_search."
            )
        is_constrained = isinstance(self.constraint_strategy, ConstraintTableStrategy)
        if is_constrained:
            validate_constraint_generation_limits(
                beam_width=beam_width,
                max_generate_steps=max_tokens,
            )
        if not params.ignore_eos:
            # Settled design §7 ①: EOS cannot be honoured by the resident form
            # (the worker decision's step count is authoritative); rather than
            # silently changing semantics, fail fast.
            raise ValueError(
                "vllm-gr async_beam_search requires ignore_eos=True in "
                "BeamSearchParams (the resident Request runs to the planned "
                "decode-step count; early-EOS semantics stay on beam_search)."
            )

        tokenizer = self.get_tokenizer()
        sid_begin_token_id: int | None = None
        if begin_token is not None:
            sid_begin_token_id = tokenizer.convert_tokens_to_ids(begin_token)
            if sid_begin_token_id is None or sid_begin_token_id == -1:
                logger.warning(
                    "begin_token must be a valid token in the tokenizer "
                    "vocabulary, got token %s",
                    begin_token,
                )
                sid_begin_token_id = None
        sid_end_token_id: int | None = None
        if end_token is not None:
            sid_end_token_id = tokenizer.convert_tokens_to_ids(end_token)
            if sid_end_token_id is None or sid_end_token_id == -1:
                logger.warning(
                    "end_token must be a valid token in the tokenizer "
                    "vocabulary, got token %s",
                    end_token,
                )
                sid_end_token_id = None

        resident_capacity = int(gr_cfg.beam_max_batch_size)
        if concurrency_limit is None:
            concurrency_limit = min(len(prompts), resident_capacity)
        elif concurrency_limit < 1 or concurrency_limit > resident_capacity:
            raise ValueError(
                "async_beam_search concurrency_limit must be in [1, "
                f"{resident_capacity}] for the configured beam_max_batch_size"
            )

        instances: list[BeamSearchInstance] = []
        engine_inputs = self._preprocess_cmpl(prompts)
        lora_requests = self._lora_request_to_seq(lora_request, len(engine_inputs))

        for lora_req, prompt in zip(lora_requests, engine_inputs):
            if prompt["type"] == "embeds":
                raise NotImplementedError("Embedding prompt not supported for beam search")

            # Match the synchronous entry point: only keys are rebound below,
            # and prompt_token_ids is copied before appending the begin token.
            # Avoid deep-copying the full prompt on the prefill critical path.
            engine_prompt = dict(prompt)
            if engine_prompt.get("type") == "enc_dec":
                engine_prompt["decoder_prompt"] = dict(
                    engine_prompt["decoder_prompt"]
                )
            decoder_prompt = (
                engine_prompt
                if engine_prompt["type"] != "enc_dec"
                else engine_prompt["decoder_prompt"]
            )
            prompt_tokens = decoder_prompt["prompt_token_ids"]

            if sid_begin_token_id is not None or sid_end_token_id is not None:
                prompt_tokens = list(prompt_tokens)
                if sid_begin_token_id is not None:
                    prompt_tokens.append(sid_begin_token_id)
                decoder_prompt["prompt_token_ids"] = prompt_tokens

            instance = BeamSearchInstance(engine_prompt, lora_request=lora_req, logprobs=None)
            instance.beams[0].logprobs = FlatLogprobs()
            instance.prompt_len = (
                len(prompt_tokens) - 1 if sid_begin_token_id is not None else len(prompt_tokens)
            )
            instances.append(instance)

        pre_calc = 0 if sid_begin_token_id is None else 1
        if sid_end_token_id is not None:
            pre_calc += 1
        if max_tokens - pre_calc < 0:
            logger.warning(
                "max_tokens is less than the number of tokens to be pre-calculated. "
                "Setting max_tokens to the number of pre-calculated tokens."
            )
            max_tokens = pre_calc
        # decode-loop length (== number of generated tokens): iteration 0 is the
        # prefill, the remaining ``loop_len - 1`` are decode steps.
        max_decode_steps = max_tokens - pre_calc

        # Single-Request form: ONE Request carries the whole session lifecycle.
        # ``max_tokens = max_decode_steps`` lets the Request terminate naturally
        # via check_stop (FINISHED_LENGTH_CAPPED) after exactly ``loop_len``
        # output tokens (prefill + ``loop_len - 1`` decodes), so the prefill
        # never returns to the frontend for a re-ADD — the upstream
        # AsyncScheduler keeps scheduling the same Request.  ``logprobs`` stays
        # at ``beam_width`` / ``flat_logprobs=True`` so the worker's
        # unconstrained prefill-decision gate reads the step-0 candidates from
        # the prefill's flat top-K.
        beam_search_params = SamplingParams(
            logprobs=beam_width,
            max_tokens=max_decode_steps,
            temperature=temperature,
            skip_clone=True,
            detokenize=False,
            flat_logprobs=True,
            # The single Request runs the whole session; EOS must NOT cap it
            # early (settled design §7 ①: the worker decision's step count is
            # authoritative — the beam rows' tokens are overwritten anyway).
            ignore_eos=True,
        )

        # Cache decoded prompt text and token length for the final assembly.
        cached_prompts = []
        for instance in instances:
            prompt_tokens = instance.beams[0].tokens
            prompt_text = tokenizer.decode(prompt_tokens)
            cached_prompts.append({"text": prompt_text, "len": len(prompt_tokens)})

        for prompt_start in range(0, len(prompts), concurrency_limit):
            instances_batch = instances[prompt_start : prompt_start + concurrency_limit]
            _custom_async_beam_search_batch(
                llm_engine=self.llm_engine,
                instances_batch=instances_batch,
                prompt_start=prompt_start,
                tokenizer=tokenizer,
                beam_width=beam_width,
                beam_search_params=beam_search_params,
                sid_end_token_id=sid_end_token_id,
                max_decode_steps=max_decode_steps,
                is_constrained=is_constrained,
            )

        # ---- Final assembly (mirrors beam_search, cum_logprob-only) ----
        outputs = []
        for idx, instance in enumerate(instances):
            # The worker's terminal sequence covers exactly the decode steps;
            # append the optional sid_end token to every surviving beam.
            if sid_end_token_id is not None:
                for beam in instance.beams:
                    beam.tokens.append(sid_end_token_id)
            sorted_completed = sorted(
                instance.completed + instance.beams,
                key=lambda x: x.cum_logprob,
                reverse=True,
            )
            best_beams = sorted_completed[:beam_width]

            prompt_cache = cached_prompts[idx]
            prompt_text = prompt_cache["text"]
            prompt_len = prompt_cache["len"]
            beam_token_lists = [beam.tokens[prompt_len:] for beam in best_beams]
            if hasattr(tokenizer, "batch_decode"):
                decoded_gen_texts = tokenizer.batch_decode(beam_token_lists)
            else:
                decoded_gen_texts = [tokenizer.decode(tokens) for tokens in beam_token_lists]
            for beam, gen_text in zip(best_beams, decoded_gen_texts):
                beam.text = prompt_text + gen_text
            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True

        from vllm_gr.utils.engine_lifecycle import shutdown_sync_engine

        if hasattr(self, "llm_engine") and self.llm_engine is not None:
            shutdown_sync_engine(self.llm_engine)
            self.llm_engine = None

        gc.collect()
        release_accelerator_memory()
        logger.info("GRLLM shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
