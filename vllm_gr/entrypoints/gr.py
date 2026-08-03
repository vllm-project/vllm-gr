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

import copy
import gc
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from tqdm.auto import tqdm
from vllm.entrypoints.generate.beam_search.utils import (
    BeamSearchInstance,
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
    mask_candidate_logprobs,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.beam_search_step import apply_beam_search_step
from vllm_gr.entrypoints.beam_search_config import (
    BeamSearchStep,
    build_beam_search_steps,
    select_best_beams,
)
from vllm_gr.logger import init_logger
from vllm_gr.logprobs import reconstruct_beam_logprobs
from vllm_gr.sampling_params import clone_beam_decode_sampling_params
from vllm_gr.utils.catalog import Catalog
from vllm_gr.utils.npu_adapter import release_accelerator_memory
from vllm_gr.v1.engine.types import BeamRequestStepUpdate

if TYPE_CHECKING:
    from vllm.entrypoints.generate.beam_search.utils import BeamSearchOutput
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import BeamSearchParams

logger = init_logger(__name__)

BeamCandidateCache = tuple[
    Sequence[int],
    Sequence[float],
    Sequence[int | None] | None,
    Sequence[str | None] | None,
]
ForkInfo = list[tuple[int, int]]


def _create_tokens_prompt_from_beam(beam: "BeamSearchSequence") -> "TokensPrompt":
    """Create a TokensPrompt payload from a BeamSearchSequence."""
    return beam.get_prompt()


def _build_step_sampling_params(
    step: BeamSearchStep,
    temperature: float,
) -> SamplingParams:
    """Create the identical per-level sampling policy for both backends."""

    return SamplingParams(
        logprobs=step.expand_width,
        max_tokens=1,
        temperature=temperature,
        skip_clone=True,
        detokenize=False,
        flat_logprobs=True,
        allowed_token_ids=step.allowed_token_ids,
    )


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
    beam_search_params: "SamplingParams",
    eos_token_id: int | None,
    instance_prefill_id: dict[int, str],
    instance_prev_beam_internal_ids: dict[int, list[str]],
    instance_fork_info: dict[int, ForkInfo],
    instance_prefix_len: dict[int, int],
    instance_completed_sessions: set[int],
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

            update = BeamRequestStepUpdate(
                session_id=session_id,
                parent_beam_ids=parent_beam_ids,
                child_beam_ids=child_beam_ids,
                beam_tokens=beam_tokens,
                pruned_ids=pruned_ids,
                prefix_len=prefix_len,
                beam_width=W,
                sampling_params=decode_sampling_params,
                eos_token_id=eos_token_id,
                lora_request=instance.beams[0].lora_request,
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
    catalog: Optional[Any] = None,
    catalog_strategy: str = "fill",
) -> tuple[bool, list[int], list[float], list[int], list[BeamCandidateCache]]:
    """Parse flat logprobs from engine step results.

    Returns:
        is_error: bool indicating if the engine returned an error
        all_beams_token_id: list of token ids across candidate beams
        all_beams_logprob: list of cumulative logprobs
        all_beams_beam_idx: list of beam indices for each entry
        beam_flat_cache: list of parsed segments for each parent beam
    """
    if result.outputs[0].finish_reason == "error":
        return True, [], [], [], []

    all_beams_token_id = []
    all_beams_logprob = []
    all_beams_beam_idx = []
    beam_flat_cache = []

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

            token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_and_dedup_segment(
                raw_token_ids,
                raw_logprobs,
                raw_ranks,
                raw_decoded,
                start_offset,
                end_offset,
                logprobs_num,
            )
            if catalog is not None:
                generated_tokens = current_beam.tokens[instance.prompt_len :]
                valid_tokens_set = catalog.valid(generated_tokens)
                logprobs_pos = mask_candidate_logprobs(
                    token_ids_pos,
                    logprobs_pos,
                    valid_tokens_set,
                )

            beam_flat_cache.append((token_ids_pos, logprobs_pos, ranks_pos, decoded_pos))
            all_beams_token_id.extend(token_ids_pos)
            all_beams_logprob.extend(current_beam.cum_logprob + lp for lp in logprobs_pos)
            all_beams_beam_idx.extend([b_idx] * len(token_ids_pos))
    else:
        token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_and_dedup_flat_logprobs(
            flat, logprobs_num
        )
        if catalog is not None:
            current_beam = instance.beams[0]
            generated_tokens = current_beam.tokens[instance.prompt_len :]
            valid_tokens_set = catalog.valid(generated_tokens)
            logprobs_pos = mask_candidate_logprobs(
                token_ids_pos,
                logprobs_pos,
                valid_tokens_set,
            )

        beam_flat_cache.append((token_ids_pos, logprobs_pos, ranks_pos, decoded_pos))
        all_beams_token_id.extend(token_ids_pos)
        all_beams_logprob.extend(instance.beams[0].cum_logprob + lp for lp in logprobs_pos)
        all_beams_beam_idx.extend([0] * len(token_ids_pos))

    return False, all_beams_token_id, all_beams_logprob, all_beams_beam_idx, beam_flat_cache


def _custom_beam_search_batch(
    llm_engine,
    instances_batch: list[BeamSearchInstance],
    steps: Sequence[BeamSearchStep],
    prompt_start: int,
    temperature: float,
    eos_token_id: int | None,
    ignore_eos: bool,
    initial_logprobs: "FlatLogprobs",
    use_tqdm: bool,
    catalog: Optional[Any] = None,
    catalog_strategy: str = "fill",
) -> None:
    """Run beam search in batch mode for custom attention backend."""
    instance_prefill_id: dict[int, str] = {}
    instance_prev_beam_internal_ids: dict[int, list[str]] = {}
    instance_fork_info: dict[int, ForkInfo] = {}
    instance_prefix_len: dict[int, int] = {}
    instance_completed_sessions: set[int] = set()

    step_iter: Any = enumerate(steps)
    if use_tqdm:
        step_iter = enumerate(tqdm(steps, desc="Beam search", unit="token", unit_scale=False))
        logger.warning(
            "The progress bar shows the upper bound on token steps and may "
            "finish early due to stopping conditions."
        )

    for token, step in step_iter:
        # Check if all instances are done
        all_done = True
        for instance in instances_batch:
            if len(instance.beams) > 0:
                all_done = False
                break
        if all_done:
            break

        beam_search_params = _build_step_sampling_params(step, temperature)

        # Step 1: Prepare and update requests
        engine_core_requests = _prepare_beam_step_requests(
            llm_engine=llm_engine,
            instances_batch=instances_batch,
            token=token,
            prompt_start=prompt_start,
            beam_search_params=beam_search_params,
            eos_token_id=eos_token_id,
            instance_prefill_id=instance_prefill_id,
            instance_prev_beam_internal_ids=instance_prev_beam_internal_ids,
            instance_fork_info=instance_fork_info,
            instance_prefix_len=instance_prefix_len,
            instance_completed_sessions=instance_completed_sessions,
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
                    logprobs_num=step.expand_width,
                    fork_info_j=instance_fork_info.get(j),
                    catalog=catalog,
                    catalog_strategy=catalog_strategy,
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

            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)
            all_beams_beam_idx_arr = np.array(all_beams_beam_idx, dtype=np.intp)
            if not ignore_eos and eos_token_id is not None:
                eos_beams, all_beams_logprob = complete_eos_candidates(
                    parents=instance.beams,
                    token_ids=all_beams_token_id,
                    scores=all_beams_logprob,
                    beam_idx=all_beams_beam_idx_arr,
                    beam_flat_cache=beam_flat_cache,
                    stop_token_ids={eos_token_id},
                    initial_logprobs=initial_logprobs,
                    include_eos_token=True,
                    inherit_lora_request=True,
                )
                instance.completed.extend(eos_beams)

            topn_idx = select_top_indices(all_beams_logprob, step.keep_width, catalog_strategy)
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


def _run_beam_search_steps(
    llm: "GRLLM",
    instances_batch: list[BeamSearchInstance],
    steps: Sequence[BeamSearchStep],
    *,
    temperature: float,
    initial_logprobs: "FlatLogprobs",
    eos_token_id: int | None,
    ignore_eos: bool,
    catalog: Any | None,
    catalog_strategy: str,
    use_tqdm: bool,
) -> None:
    """Run the shared expand, score and prune loop through ``LLM.generate``."""

    import itertools

    step_iter: Any = steps
    if use_tqdm:
        step_iter = tqdm(steps, desc="Beam search", unit="token", unit_scale=False)
        logger.warning(
            "The progress bar shows the upper bound on token steps and may "
            "finish early due to stopping conditions."
        )

    for step in step_iter:
        all_beams: list[BeamSearchSequence] = list(
            sum((instance.beams for instance in instances_batch), [])
        )
        if not all_beams:
            break
        positions = [0] + list(
            itertools.accumulate(len(instance.beams) for instance in instances_batch)
        )
        instance_ranges = list(zip(positions[:-1], positions[1:]))

        sampling_params = _build_step_sampling_params(step, temperature)
        prompts_batch, lora_req_batch = zip(
            *[(_create_tokens_prompt_from_beam(beam), beam.lora_request) for beam in all_beams]
        )
        output = llm.generate(
            prompts_batch,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_req_batch,
        )

        for (start, end), instance in zip(instance_ranges, instances_batch):
            parents = all_beams[start:end]
            allowed = None
            if catalog is not None:
                allowed = [catalog.valid(beam.tokens[instance.prompt_len :]) for beam in parents]
            transition = apply_beam_search_step(
                parents,
                output[start:end],
                step,
                initial_logprobs,
                allowed_token_ids_by_parent=allowed,
                stop_token_ids=(
                    {eos_token_id} if eos_token_id is not None and not ignore_eos else None
                ),
                catalog_strategy=catalog_strategy,
            )
            instance.beams = transition.active
            instance.completed.extend(transition.completed)


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
        catalog_path: Optional[str] = None,
        **kwargs,
    ):
        # vLLM 0.22 enables ModelRunner V2 automatically for Qwen3. Apply the
        # vllm-gr default before LLM.__init__ constructs VllmConfig so the V1
        # beam-search worker patches remain active.
        from vllm_gr._env_check import configure_model_runner_default

        configure_model_runner_default()

        self._shutdown_complete = False

        super().__init__(
            model=model,
            **kwargs,
        )

        self.catalog = None
        if catalog_path is not None:
            self.catalog = Catalog(self.get_tokenizer())
            self.catalog.load(catalog_path)

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
            prompts: Standard vLLM text or token-ID prompts. Model-specific
                semantic inputs must be converted by the caller.
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            use_tqdm: Whether to use tqdm to display the progress bar.
            concurrency_limit: The maximum number of concurrent requests.
                If None, the number of concurrent requests is unlimited.
        """
        # TODO: how does beam search work together with length penalty,
        # frequency, penalty, and stopping criteria, etc.?
        logger.debug("Called beam_search")
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        begin_token = getattr(params, "begin_token", None)
        end_token = getattr(params, "end_token", None)
        catalog_strategy = getattr(params, "catalog_strategy", "fill")

        model_config = self.llm_engine.model_config
        tokenizer = None
        if not getattr(model_config, "skip_tokenizer_init", False):
            tokenizer = self.get_tokenizer()

        if tokenizer is None and (begin_token is not None or end_token is not None):
            raise ValueError("begin_token and end_token require an initialized tokenizer")

        sid_begin_token_id: int | None = None
        if begin_token is not None and tokenizer is not None:
            sid_begin_token_id = tokenizer.convert_tokens_to_ids(begin_token)
            if sid_begin_token_id is None or sid_begin_token_id == -1:
                logger.warning(
                    "begin_token must be a valid token in the tokenizer "
                    f"vocabulary, got token {begin_token}"
                )
                sid_begin_token_id = None

        sid_end_token_id: int | None = None
        if end_token is not None and tokenizer is not None:
            sid_end_token_id = tokenizer.convert_tokens_to_ids(end_token)
            if sid_end_token_id is None or sid_end_token_id == -1:
                logger.warning(
                    "end_token must be a valid token in the tokenizer "
                    f"vocabulary, got token {end_token}"
                )
                sid_end_token_id = None

        engine_inputs = []
        generated_starts: list[int] = []
        decode_prompt_ends: list[int] = []
        for prompt in self._preprocess_cmpl(prompts):
            if prompt["type"] == "embeds":
                raise NotImplementedError("Embedding prompt not supported for beam search")
            engine_prompt = copy.deepcopy(prompt)
            decoder_prompt = (
                engine_prompt
                if engine_prompt["type"] != "enc_dec"
                else engine_prompt["decoder_prompt"]
            )
            prompt_token_ids = decoder_prompt["prompt_token_ids"]
            generated_starts.append(len(prompt_token_ids))
            if sid_begin_token_id is not None:
                prompt_token_ids = list(prompt_token_ids)
                prompt_token_ids.append(sid_begin_token_id)
                decoder_prompt["prompt_token_ids"] = prompt_token_ids
            decode_prompt_ends.append(len(prompt_token_ids))
            engine_inputs.append(engine_prompt)

        pre_calc = int(sid_begin_token_id is not None) + int(sid_end_token_id is not None)
        if max_tokens < pre_calc:
            logger.warning(
                "max_tokens is less than the number of tokens to be "
                "pre-calculated. Setting max_tokens to the number of "
                "pre-calculated tokens."
            )
            max_tokens = pre_calc

        architectures = getattr(model_config, "architectures", None) or []
        steps = build_beam_search_steps(
            architectures,
            max_tokens - pre_calc,
            beam_width,
        )
        ignore_eos = params.ignore_eos
        catalog = self.catalog

        if use_tqdm and concurrency_limit is not None:
            logger.warning(
                "Progress bar is not supported when using concurrency_limit. "
                "Disabling progress bar."
            )
            use_tqdm = False

        if concurrency_limit is None:
            concurrency_limit = len(prompts)

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
        lora_requests = self._lora_request_to_seq(lora_request, len(engine_inputs))
        for index, (lora_req, engine_prompt) in enumerate(zip(lora_requests, engine_inputs)):
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
            instance.prompt_len = generated_starts[index]
            instances.append(instance)

        vllm_config = getattr(self.llm_engine, "vllm_config", None)
        attention_backend = None
        if vllm_config is not None and hasattr(vllm_config, "attention_config"):
            attention_backend = getattr(vllm_config.attention_config, "backend", None)
        use_custom_attn = attention_backend is not None and (
            "CUSTOM" in str(attention_backend).upper()
            or getattr(attention_backend, "name", "").upper() == "CUSTOM"
        )
        hf_text_config = getattr(model_config, "hf_text_config", None)
        eos_token_id = getattr(hf_text_config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if sid_end_token_id is not None:
            eos_token_id = sid_end_token_id
        for prompt_start in range(0, len(prompts), concurrency_limit):
            instances_batch = instances[prompt_start : prompt_start + concurrency_limit]
            if use_custom_attn:
                _custom_beam_search_batch(
                    llm_engine=self.llm_engine,
                    instances_batch=instances_batch,
                    steps=steps,
                    prompt_start=prompt_start,
                    temperature=temperature,
                    eos_token_id=eos_token_id,
                    ignore_eos=ignore_eos,
                    initial_logprobs=initial_logprobs,
                    use_tqdm=use_tqdm,
                    catalog=catalog,
                    catalog_strategy=catalog_strategy,
                )
            else:
                _run_beam_search_steps(
                    self,
                    instances_batch,
                    steps,
                    temperature=temperature,
                    initial_logprobs=initial_logprobs,
                    eos_token_id=eos_token_id,
                    ignore_eos=ignore_eos,
                    catalog=catalog,
                    catalog_strategy=catalog_strategy,
                    use_tqdm=use_tqdm,
                )

        from vllm.entrypoints.generate.beam_search.utils import BeamSearchOutput

        outputs = []
        for index, instance in enumerate(instances):
            if sid_end_token_id is not None:
                for beam in instance.beams:
                    beam.tokens.append(sid_end_token_id)
            best_beams = select_best_beams(instance, beam_width)
            prompt_text = None
            if tokenizer is not None and best_beams:
                prompt_text = tokenizer.decode(best_beams[0].tokens[: decode_prompt_ends[index]])
            for beam in best_beams:
                reconstruct_beam_logprobs(
                    beam,
                    initial_logprobs,
                    sid_end_token_id,
                )
                if tokenizer is not None and prompt_text is not None:
                    generated_text = tokenizer.decode(beam.tokens[decode_prompt_ends[index] :])
                    beam.text = prompt_text + generated_text
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
