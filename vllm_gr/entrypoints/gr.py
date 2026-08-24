# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GRLLM — user-facing entry point for offline inference with beam search.

A thin subclass of vLLM's :class:`~vllm.entrypoints.llm.LLM` that adds Beam
search and a resource-free marker for Worker-owned constraint execution.

The previous GR adapter / recommend runtime has been removed; this class now
always delegates construction to the standard vLLM ``LLM`` initializer so any
HuggingFace checkpoint (e.g. OneRec, Pangu) is supported via the regular vLLM
LLM path.
"""

import copy
import gc
import math
import time
from typing import TYPE_CHECKING, Any, Union

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
    select_top_indices,
)
from vllm_gr.config import (
    CONSTRAINT_TABLE_MAX_GENERATION_STEPS,
    GRConfig,
    GRConfigInput,
)
from vllm_gr.constraints import ParentCandidateBuilder
from vllm_gr.constraints.models import BeamCandidateCache, CandidateLayout
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


def _validate_offline_constraint_options(
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


def _validate_offline_constraint_capacity(
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


def _validate_offline_constraint_protocol_available(
    vllm_config: object | None,
    gr_config: GRConfig,
) -> bool:
    attention_config = getattr(vllm_config, "attention_config", None)
    backend = getattr(attention_config, "backend", None)
    available = backend is not None and (
        "CUSTOM" in str(backend).upper() or getattr(backend, "name", "").upper() == "CUSTOM"
    )
    if gr_config.constraint_table.enabled and not available:
        raise RuntimeError(
            "constraint_table requires the beam-request protocol path "
            "(the CUSTOM attention backend); the fallback cannot carry "
            "step/prefix metadata. Configure "
            "attention_config={'backend': 'CUSTOM'}."
        )
    return available


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
    constraints_enabled: bool = False,
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

    candidate_builder = ParentCandidateBuilder(
        layout=(
            CandidateLayout.CONSTRAINED_TOPK
            if constraints_enabled
            else CandidateLayout.STANDARD_SAMPLER
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

            if constraints_enabled:
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
            candidate_builder.add_parent(
                parent_index=b_idx,
                parent_cumulative_logprob=current_beam.cum_logprob,
                token_ids=token_ids_pos,
                local_logprobs=logprobs_pos,
                ranks=ranks_pos,
                decoded_tokens=decoded_pos,
            )
    else:
        if constraints_enabled:
            token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_constrained_flat_logprobs(
                flat, logprobs_num
            )
        else:
            token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_and_dedup_flat_logprobs(
                flat, logprobs_num
            )
        candidate_builder.add_parent(
            parent_index=0,
            parent_cumulative_logprob=instance.beams[0].cum_logprob,
            token_ids=token_ids_pos,
            local_logprobs=logprobs_pos,
            ranks=ranks_pos,
            decoded_tokens=decoded_pos,
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
    catalog_strategy: str | None = None,
    constraints_enabled: bool = False,
    max_decode_steps: int = 0,
) -> None:
    """Run beam search in batch mode for custom attention backend."""
    instance_prefill_id: dict[int, str] = {}
    instance_prev_beam_internal_ids: dict[int, list[str]] = {}
    instance_fork_info: dict[int, ForkInfo] = {}
    instance_prefix_len: dict[int, int] = {}
    instance_completed_sessions: set[int] = set()

    if constraints_enabled and catalog_strategy is not None:
        raise ValueError(
            f"catalog_strategy is not applicable to constraint_table; got {catalog_strategy!r}"
        )

    # Constraint Tables generate candidates in the Worker; the decode-step
    # update must carry the constraint_step so the engine can route the raw
    # logits through the constrained top-k operator.
    _is_constrained = constraints_enabled

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

            is_error, all_beams_token_id, all_beams_logprob, all_beams_beam_idx, beam_flat_cache = (
                _parse_step_logprobs(
                    result=result,
                    instance=instance,
                    token=token,
                    logprobs_num=logprobs_num,
                    fork_info_j=instance_fork_info.get(j),
                    constraints_enabled=constraints_enabled,
                    return_arrays=True,
                )
            )

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
            all_beams_beam_idx_arr = all_beams_beam_idx
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

            topn_idx = select_top_indices(all_beams_logprob, beam_width, catalog_strategy)
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
    """LLM subclass with Beam search and Worker-owned constraint execution."""

    def __init__(
        self,
        model: str = "",
        *,
        vllm_gr_config: GRConfigInput | None = None,
        **kwargs,
    ):
        # vLLM 0.22 enables ModelRunner V2 automatically for Qwen3. Apply the
        # vllm-gr default before LLM.__init__ constructs VllmConfig so the V1
        # beam-search worker patches remain active.
        from vllm_gr._env_check import configure_model_runner_default

        configure_model_runner_default()

        self._shutdown_complete = False

        gr_engine_args: dict[str, object] = {}
        if engine_args_accepts_gr_values():
            gr_engine_args["vllm_gr_config"] = vllm_gr_config
        elif vllm_gr_config is not None:
            raise RuntimeError(
                "vllm-gr runtime patches are not active: EngineArgs cannot "
                "carry vllm_gr_config into VllmConfig. Ensure the vllm-gr "
                "plugin is registered before constructing GRLLM."
            )

        super().__init__(
            model=model,
            **gr_engine_args,
            **kwargs,
        )

        vllm_config = getattr(getattr(self, "llm_engine", None), "vllm_config", None)
        gr_config = GRConfig.from_vllm_config(vllm_config)
        self._gr_config = gr_config
        self.constraints_enabled = gr_config.constraint_table.enabled
        self._use_custom_attn = _validate_offline_constraint_protocol_available(
            vllm_config,
            gr_config,
        )

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
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        begin_token = params.begin_token if hasattr(params, "begin_token") else None
        end_token = params.end_token if hasattr(params, "end_token") else None
        catalog_strategy = getattr(params, "catalog_strategy", None)
        temperature, catalog_strategy = _validate_offline_constraint_options(
            self._gr_config,
            temperature=temperature,
            catalog_strategy=catalog_strategy,
        )

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

            engine_prompt = copy.deepcopy(prompt)
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

        # Cache decoded prompt text and token length for each instance to optimize final decoding
        cached_prompts = []
        for instance in instances:
            prompt_tokens = instance.beams[0].tokens
            prompt_text = tokenizer.decode(prompt_tokens)
            cached_prompts.append({"text": prompt_text, "len": len(prompt_tokens)})

        pre_calc = 0 if sid_begin_token_id is None else 1
        if sid_end_token_id is not None:
            pre_calc += 1
        if max_tokens - pre_calc < 0:
            logger.warning(
                "max_tokens is less than the number of tokens to be pre-calculated. "
                "Setting max_tokens to the number of pre-calculated tokens."
            )
            max_tokens = pre_calc
        if self.constraints_enabled:
            _validate_offline_constraint_capacity(
                self._gr_config,
                beam_width=beam_width,
                max_generate_steps=max_tokens - pre_calc,
            )
        use_custom_attn = self._use_custom_attn
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
                    catalog_strategy=catalog_strategy,
                    constraints_enabled=self.constraints_enabled,
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
                                candidate_builder.add_parent(
                                    parent_index=i,
                                    parent_cumulative_logprob=current_beam.cum_logprob,
                                    token_ids=token_ids_pos,
                                    local_logprobs=logprobs_pos,
                                    ranks=ranks_pos,
                                    decoded_tokens=decoded_pos,
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
            prompt_text = prompt_cache["text"]
            prompt_len = prompt_cache["len"]
            # Reconstruct FlatLogprobs from parent pointers for the best beams.
            for beam in best_beams:
                reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)
                gen_tokens = beam.tokens[prompt_len:]
                beam.text = prompt_text + tokenizer.decode(gen_tokens)
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
