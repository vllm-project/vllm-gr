# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GRLLM — user-facing entry point for offline inference with beam search.

A thin subclass of vLLM's :class:`~vllm.entrypoints.llm.LLM` that adds:

- optional catalog-constrained generation (``catalog_path``)
- a catalog-aware :meth:`beam_search` implementation

The previous GR adapter / recommend runtime has been removed; this class now
always delegates construction to the standard vLLM ``LLM`` initializer so any
HuggingFace checkpoint (e.g. OneRec, Pangu) is supported via the regular vLLM
LLM path.
"""

import gc
import time
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
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
    select_top_indices,
)
from vllm_gr.config import ConstraintBackend, GRConfig
from vllm_gr.config.validation import validate_beam_max_width, validate_beam_width
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
    validate_constraint_generation_limits,
    validate_constraint_table_protocol,
)
from vllm_gr.engine.arg_utils_gr import engine_args_accepts_gr_values
from vllm_gr.logger import init_logger
from vllm_gr.logprobs import (
    reconstruct_beam_logprobs,
)
from vllm_gr.sampling_params import clone_beam_decode_sampling_params
from vllm_gr.utils.npu_adapter import release_accelerator_memory
from vllm_gr.v1.engine.types import BeamRequestStepUpdate

if TYPE_CHECKING:
    from vllm.entrypoints.generate.beam_search.utils import BeamSearchOutput
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import BeamSearchParams

logger = init_logger(__name__)

ForkInfo = list[tuple[int, int]]


def _create_tokens_prompt_from_beam(beam: "BeamSearchSequence") -> "TokensPrompt":
    """Create a TokensPrompt payload from a BeamSearchSequence."""
    return beam.get_prompt()


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
    for j, instance in enumerate(instances_batch):
        if len(instance.beams) == 0:
            # This instance has no active beams (finished)
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

                full_tokens = instance.beams[new_idx].tokens
                prefix_len = instance_prefix_len[j]
                beam_tokens.append(full_tokens[prefix_len:])

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
                prompt_token_ids=instance.beams[0].tokens,
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
                llm_engine.engine_core.beam_request_step_update(update)

            # Update tracking variables
            instance_prev_beam_internal_ids[j] = child_beam_ids

        else:
            # ADD_BATCH path (step 0)
            request_id_batch = f"offline-beam-{prompt_start + j}"
            instance_prefix_len[j] = len(instance.beams[0].tokens)

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
                generated_tokens = current_beam.tokens[instance.prompt_len :]
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
            generated_tokens = current_beam.tokens[instance.prompt_len :]
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
        # Check if all instances are done
        all_done = True
        for instance in instances_batch:
            if len(instance.beams) > 0:
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
                continue

            # Worker-side beam decision: when the accelerator
            # already selected the surviving beams, use its winners directly
            # and skip the frontend flat-logprobs parsing + NumPy top-k.  The
            # downstream EOS handling / beam materialisation is unchanged.
            from vllm_gr.entrypoints.openai.serving_engine import (
                _worker_decision_flatten,
                _worker_decision_from_result,
            )

            worker_decision = _worker_decision_from_result(result)
            worker_candidates = None
            if worker_decision is not None:
                worker_candidates = _worker_decision_flatten(
                    worker_decision, beam_width, instance.beams
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
                )
                instance.completed.extend(eos_beams)

            # A worker decision already contains exactly W survivors in the
            # order used by its device-resident KV state. Keep that order.
            topn_idx = (
                np.arange(all_beams_token_id.size, dtype=np.intp)
                if worker_candidates is not None
                else select_top_indices(all_beams_logprob, beam_width, catalog_strategy)
            )
            new_beams, fork_info = materialize_selected_beams(
                parents=instance.beams,
                selected_indices=topn_idx,
                token_ids=all_beams_token_id,
                scores=all_beams_logprob,
                beam_idx=all_beams_beam_idx_arr,
                beam_flat_cache=beam_flat_cache,
                initial_logprobs=initial_logprobs,
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
        **kwargs,
    ):
        beam_max_width = validate_beam_max_width(beam_max_width)
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
        if beam_updates:
            additional_config = dict(kwargs.pop("additional_config", None) or {})
            additional_config.update(beam_updates)
            kwargs["additional_config"] = additional_config

        super().__init__(
            model=model,
            **gr_engine_args,
            **kwargs,
        )

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

    def beam_search(
        self,
        prompts: list[Union["TokensPrompt", "TextPrompt"]],
        params: "BeamSearchParams",
        lora_request: Union[list["LoRARequest"], "LoRARequest", None] = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
    ) -> list["BeamSearchOutput"]:
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
        """
        # Lazy imports for beam search dependencies
        import itertools

        # TODO: how does beam search work together with length penalty,
        # frequency, penalty, and stopping criteria, etc.?
        logger.debug("Called beam_search")
        beam_width = validate_beam_width(params.beam_width)
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        begin_token = params.begin_token if hasattr(params, "begin_token") else None
        end_token = params.end_token if hasattr(params, "end_token") else None
        catalog_strategy = getattr(params, "catalog_strategy", None)

        tokenizer = self.get_tokenizer()
        sid_begin_token_id: int | None = None
        if begin_token is not None:
            sid_begin_token_id = tokenizer.convert_tokens_to_ids(begin_token)
            if sid_begin_token_id is None or sid_begin_token_id == -1:
                logger.warning(
                    f"begin_token must be a valid token in the tokenizer vocabulary, got token {begin_token}"
                )
                sid_begin_token_id = None
        sid_end_token_id: int | None = None
        if end_token is not None:
            sid_end_token_id = tokenizer.convert_tokens_to_ids(end_token)
            if sid_end_token_id is None or sid_end_token_id == -1:
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
            prompt_tokens = instance.beams[0].tokens
            cached_prompts.append({"tokens": prompt_tokens, "len": len(prompt_tokens)})

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
                                generated_tokens = beam.tokens[prompt_len:]
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
                            )
                            instance.completed.extend(eos_beams)

                        # Select top beams
                        topn_idx = select_top_indices(
                            all_beams_logprob, beam_width, catalog_strategy
                        )
                        new_beams, _ = materialize_selected_beams(
                            parents=all_beams[start:end],
                            selected_indices=topn_idx,
                            token_ids=all_beams_token_id,
                            scores=all_beams_logprob,
                            beam_idx=all_beams_beam_idx_arr,
                            beam_flat_cache=beam_flat_cache,
                            initial_logprobs=initial_logprobs,
                        )
                        instance.beams = new_beams

        outputs = []
        for idx, instance in enumerate(instances):
            # Append sid_end_token_id only to active beams (not EOS-completed
            # beams that already terminated naturally).
            if sid_end_token_id is not None:
                for beam in instance.beams:
                    beam.tokens.append(sid_end_token_id)
            # Move remaining active beams into the completed list.
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(instance.completed, key=lambda x: x.cum_logprob, reverse=True)
            best_beams = sorted_completed[:beam_width]

            prompt_cache = cached_prompts[idx]
            # Deferred prompt detokenization: the decode runs here (post
            # generation) instead of on the pre-prefill critical path; same
            # tokenizer, same tokens, identical text.
            prompt_text = tokenizer.decode(prompt_cache["tokens"])
            prompt_len = prompt_cache["len"]
            # Reconstruct FlatLogprobs from parent pointers for the best beams.
            # Batching the detokenize below removes the per-beam Rust decode that
            # used to release the GIL and spread the reconstruct allocations;
            # running 128 reconstructs back-to-back now triggers cyclic-GC pauses.
            # No reference cycles exist in this path, so suspend GC across the
            # tight loop and restore it afterwards.
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                for beam in best_beams:
                    reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)
            finally:
                if gc_was_enabled:
                    gc.enable()
            # Batch-detokenize the generated tokens of all best beams in a
            # single fast-tokenizer call.  batch_decode() is not part of the
            # TokenizerLike protocol (e.g. Grok2Tokenizer only implements
            # decode()), so fall back to per-beam decode() when it is absent.
            # Qwen2Tokenizer.batch_decode accepts a list of token-id lists and
            # returns list[str], equivalent to N individual decode() calls but
            # ~5x faster (Rust batch path vs per-call Python wrapper overhead).
            beam_token_lists = [beam.tokens[prompt_len:] for beam in best_beams]
            if hasattr(tokenizer, "batch_decode"):
                decoded_gen_texts = cast(PreTrainedTokenizerBase, tokenizer).batch_decode(
                    beam_token_lists
                )
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
