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
    BeamSearchOutput,
    BeamSearchSequence,
)
from vllm.entrypoints.llm import LLM
from vllm.inputs import TokensPrompt
from vllm.logprobs import FlatLogprobs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.beam_search_decision_utils import (
    GlobalRunningStats,
    complete_eos_candidates,
    create_stat,
    extract_and_dedup_flat_logprobs,
    extract_and_dedup_segment,
    mask_beam_star_candidates,
    mask_candidate_logprobs,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.logger import init_logger
from vllm_gr.logprobs import (
    reconstruct_beam_logprobs,
)
from vllm_gr.sampling_params import clone_beam_decode_sampling_params
from vllm_gr.utils.catalog import Catalog, complete_catalog_beam, fill_catalog_beams
from vllm_gr.utils.npu_adapter import release_accelerator_memory
from vllm_gr.v1.engine.types import BeamRequestStepUpdate

if TYPE_CHECKING:
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
) -> tuple[
    bool,
    list[int],
    list[float],
    list[int],
    list[BeamCandidateCache | None],
    list[int],
]:
    """Parse flat logprobs from engine step results.

    Returns:
        is_error: bool indicating if the engine returned an error
        all_beams_token_id: list of token ids across candidate beams
        all_beams_logprob: list of cumulative logprobs
        all_beams_beam_idx: list of beam indices for each entry
        beam_flat_cache: list of parsed segments for each parent beam
        terminal_beam_idxs: indices of beams whose generated tokens complete
            a catalog item (no valid continuation); callers must complete
            them instead of carrying them forward with masked candidates
    """
    if result.outputs[0].finish_reason == "error":
        return True, [], [], [], [], []

    all_beams_token_id: list[int] = []
    all_beams_logprob: list[float] = []
    all_beams_beam_idx: list[int] = []
    beam_flat_cache: list[BeamCandidateCache | None] = []
    terminal_beam_idxs: list[int] = []

    flat = result.outputs[0].logprobs
    if flat is None:
        raise ValueError(
            "Beam search generation expected non-empty logprobs for each beam, but received 'None'."
        )

    raw_token_ids = flat.token_ids
    raw_logprobs = flat.logprobs
    raw_ranks = getattr(flat, "ranks", None)
    raw_decoded = getattr(flat, "decoded_tokens", None)

    if token > 0 and fork_info_j is not None:
        # Each beam contributes a fixed block of [sampled, top1..topK]
        # entries. The engine may pad the request's flat row to the
        # batch-wide maximum beam count, so the stride must come from the
        # fixed block size, not from total_logprobs // len(fork_info_j).
        stride_k = logprobs_num + 1

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
            is_terminal_beam = False
            if catalog is not None:
                generated_tokens = current_beam.tokens[instance.prompt_len :]
                valid_tokens_set = catalog.valid(generated_tokens)
                is_terminal_beam = not valid_tokens_set and catalog.is_terminal(generated_tokens)
                if not is_terminal_beam:
                    logprobs_pos = mask_candidate_logprobs(
                        token_ids_pos,
                        logprobs_pos,
                        valid_tokens_set,
                    )
            if is_terminal_beam:
                terminal_beam_idxs.append(b_idx)
                beam_flat_cache.append(None)
                continue

            beam_flat_cache.append((token_ids_pos, logprobs_pos, ranks_pos, decoded_pos))
            all_beams_token_id.extend(token_ids_pos)
            all_beams_logprob.extend(current_beam.cum_logprob + lp for lp in logprobs_pos)
            all_beams_beam_idx.extend([b_idx] * len(token_ids_pos))
    else:
        token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_and_dedup_flat_logprobs(
            flat, logprobs_num
        )
        is_terminal_beam = False
        if catalog is not None:
            current_beam = instance.beams[0]
            generated_tokens = current_beam.tokens[instance.prompt_len :]
            valid_tokens_set = catalog.valid(generated_tokens)
            is_terminal_beam = not valid_tokens_set and catalog.is_terminal(generated_tokens)
            if not is_terminal_beam:
                logprobs_pos = mask_candidate_logprobs(
                    token_ids_pos,
                    logprobs_pos,
                    valid_tokens_set,
                )
        if is_terminal_beam:
            terminal_beam_idxs.append(0)
            beam_flat_cache.append(None)
        else:
            beam_flat_cache.append((token_ids_pos, logprobs_pos, ranks_pos, decoded_pos))
            all_beams_token_id.extend(token_ids_pos)
            all_beams_logprob.extend(instance.beams[0].cum_logprob + lp for lp in logprobs_pos)
            all_beams_beam_idx.extend([0] * len(token_ids_pos))

    return (
        False,
        all_beams_token_id,
        all_beams_logprob,
        all_beams_beam_idx,
        beam_flat_cache,
        terminal_beam_idxs,
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
    catalog: Optional[Any] = None,
    beam_star: bool = False,
    beam_star_coef: float = 2.0,
    stats: Optional[list[GlobalRunningStats]] = None,
    token_idx_offset: int = 1,
) -> None:
    """Run beam search in batch mode for custom attention backend."""
    instance_prefill_id: dict[int, str] = {}
    instance_prev_beam_internal_ids: dict[int, list[str]] = {}
    instance_fork_info: dict[int, ForkInfo] = {}
    instance_prefix_len: dict[int, int] = {}
    instance_completed_sessions: set[int] = set()
    instance_processed: dict[int, int] = {}

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

            (
                is_error,
                all_beams_token_id,
                all_beams_logprob,
                all_beams_beam_idx,
                beam_flat_cache,
                terminal_beam_idxs,
            ) = _parse_step_logprobs(
                result=result,
                instance=instance,
                token=token,
                logprobs_num=logprobs_num,
                fork_info_j=instance_fork_info.get(j),
                catalog=catalog,
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

            for b_idx in terminal_beam_idxs:
                terminal_beam = complete_catalog_beam(instance.beams[b_idx])
                instance.completed.append(terminal_beam)

            instance_processed[prompt_start + j] = instance_processed.get(
                prompt_start + j, 0
            ) + len(all_beams_token_id)

            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)
            all_beams_beam_idx_arr = np.array(all_beams_beam_idx, dtype=np.intp)
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

            topn_idx = select_top_indices(all_beams_logprob, beam_width)

            if beam_star:
                topn_idx = mask_beam_star_candidates(
                    all_beams_logprob,
                    topn_idx,
                    stats[token],
                    token,
                    token_idx_offset,
                    beam_star_coef,
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

    return instance_processed


def _finalize_beam_search_instance(
    instance: "BeamSearchInstance",
    beam_width: int,
    catalog: Any,
    prompt_cache: dict[str, Any],
    tokenizer,
    sid_begin_token_id: int | None,
    sid_end_token_id: int | None,
    initial_logprobs: "FlatLogprobs",
    lora_request: Optional["LoRARequest"],
    processed_tokens: int,
    max_tokens: int,
) -> "BeamSearchOutput":
    """Assemble the final ``BeamSearchOutput`` for one instance.

    ``lora_request`` is passed in explicitly: after an engine error the
    instance's beam list may be empty, so the handle cannot be recovered from
    ``instance.beams[0]``.
    """
    # Append sid_end_token_id only to active beams (not EOS-completed
    # beams that already terminated naturally).
    if sid_end_token_id is not None:
        for beam in instance.beams:
            beam.tokens.append(sid_end_token_id)
    # Move remaining active beams into the completed list.
    instance.completed.extend(instance.beams)
    sorted_completed = sorted(instance.completed, key=lambda x: x.cum_logprob, reverse=True)
    valid_completed = [beam for beam in sorted_completed if np.isfinite(beam.cum_logprob)]
    best_beams = valid_completed[:beam_width]

    prompt_text = prompt_cache["text"]
    prompt_len = prompt_cache["len"]
    if catalog is not None:
        best_beams = fill_catalog_beams(
            best_beams,
            beam_width,
            catalog,
            prompt_token_ids=prompt_cache["tokens"],
            prompt_text=prompt_text,
            sid_begin_token_id=sid_begin_token_id,
            sid_end_token_id=sid_end_token_id,
            initial_logprobs=initial_logprobs,
            max_tokens=max_tokens,
            lora_request=lora_request,
        )
    # Reconstruct FlatLogprobs from parent pointers for the best beams.
    for beam in best_beams:
        reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)
        gen_tokens = beam.tokens[prompt_len:]
        beam.text = prompt_text + tokenizer.decode(gen_tokens)
    output = BeamSearchOutput(sequences=best_beams)
    output.processed_tokens = processed_tokens
    return output


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
        beam_star = bool(getattr(params, "beam_star", False))
        beam_star_coef = float(getattr(params, "beam_star_coef", 3.3))

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
        # Keep the LoRA handle per instance independently of the beam list:
        # an engine error empties instance.beams mid-search, and finalization
        # must not dereference beams[0] to recover the adapter.
        instance_lora_requests: list[Any] = []

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
            instance_lora_requests.append(lora_req)

        # Cache decoded prompt text and token length for each instance to optimize final decoding
        cached_prompts = []
        for instance in instances:
            prompt_tokens = instance.beams[0].tokens
            prompt_text = tokenizer.decode(prompt_tokens)
            cached_prompts.append(
                {"text": prompt_text, "len": len(prompt_tokens), "tokens": prompt_tokens}
            )

        pre_calc = 0 if sid_begin_token_id is None else 1
        if sid_end_token_id is not None:
            pre_calc += 1
        if max_tokens - pre_calc < 0:
            logger.warning(
                "max_tokens is less than the number of tokens to be pre-calculated. "
                "Setting max_tokens to the number of pre-calculated tokens."
            )
            max_tokens = pre_calc
        vllm_config = getattr(self.llm_engine, "vllm_config", None)
        attention_backend = None
        if vllm_config is not None and hasattr(vllm_config, "attention_config"):
            attention_backend = getattr(vllm_config.attention_config, "backend", None)
        use_custom_attn = attention_backend is not None and (
            "CUSTOM" in str(attention_backend).upper()
            or getattr(attention_backend, "name", "").upper() == "CUSTOM"
        )
        processed_by_global_idx: dict[int, int] = {}
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
                stats = create_stat(max_tokens)
                instance_processed = _custom_beam_search_batch(
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
                    beam_star=beam_star,
                    beam_star_coef=beam_star_coef,
                    stats=stats,
                    token_idx_offset=2 if sid_begin_token_id is not None else 1,
                )
                processed_by_global_idx.update(instance_processed)
            else:
                if beam_star:
                    logger.info("Beam star is not supported for non-custom attention backend")
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
                    valid_tokens_sets = []
                    has_catalog = getattr(self, "catalog", None) is not None
                    if has_catalog:
                        for instance in instances_batch:
                            prompt_len = instance.prompt_len
                            for beam in instance.beams:
                                generated_tokens = beam.tokens[prompt_len:]
                                valid_tokens_sets.append(self.catalog.valid(generated_tokens))

                    for local_idx, ((start, end), instance) in enumerate(
                        zip(instance_start_and_end, instances_batch)
                    ):
                        # Gather all logprobs and cumulative scores for this instance
                        all_beams_token_id = []
                        all_beams_logprob = []
                        # Explicit beam index for each entry in all_beams_token_id/logprob.
                        # Replaces idx // logprobs_num so that null-strategy beams (empty lists)
                        # don't shift the mapping for subsequent beams.
                        all_beams_beam_idx: list[int] = []
                        beam_flat_cache: list[tuple | None] = []
                        beam_outputs = output[start:end]  # outputs for this instance's beams

                        for i, (current_beam, result) in enumerate(
                            zip(all_beams[start:end], beam_outputs)
                        ):
                            if result.outputs[0].logprobs is not None:
                                flat = result.outputs[0].logprobs
                                token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = (
                                    extract_and_dedup_flat_logprobs(flat, logprobs_num)
                                )
                                is_terminal_beam = False
                                if has_catalog:
                                    valid_tokens_set = valid_tokens_sets[start + i]
                                    is_terminal_beam = (
                                        not valid_tokens_set
                                        and self.catalog.is_terminal(
                                            current_beam.tokens[instance.prompt_len :]
                                        )
                                    )
                                    if not is_terminal_beam:
                                        logprobs_pos = mask_candidate_logprobs(
                                            token_ids_pos,
                                            logprobs_pos,
                                            valid_tokens_set,
                                        )
                                if is_terminal_beam:
                                    instance.completed.append(complete_catalog_beam(current_beam))
                                    beam_flat_cache.append(None)
                                    continue
                                beam_flat_cache.append(
                                    (token_ids_pos, logprobs_pos, ranks_pos, decoded_pos)
                                )
                                all_beams_token_id.extend(token_ids_pos)
                                all_beams_logprob.extend(
                                    current_beam.cum_logprob + lp for lp in logprobs_pos
                                )
                                all_beams_beam_idx.extend([i] * len(token_ids_pos))
                            else:
                                raise ValueError(
                                    "Beam search generation expected non-empty logprobs "
                                    "for each beam, but received 'None'. Please ensure "
                                    "that sampling parameters are configured to return "
                                    "logprobs when using beam search."
                                )

                        # Convert to numpy for efficient processing
                        processed_by_global_idx[prompt_start + local_idx] = (
                            processed_by_global_idx.get(prompt_start + local_idx, 0)
                            + len(all_beams_token_id)
                        )
                        all_beams_token_id = np.array(all_beams_token_id)
                        all_beams_logprob = np.array(all_beams_logprob)
                        all_beams_beam_idx_arr = np.array(all_beams_beam_idx, dtype=np.intp)
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
                        topn_idx = select_top_indices(all_beams_logprob, beam_width)
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
            outputs.append(
                _finalize_beam_search_instance(
                    instance=instance,
                    beam_width=beam_width,
                    catalog=self.catalog,
                    prompt_cache=cached_prompts[idx],
                    tokenizer=tokenizer,
                    sid_begin_token_id=sid_begin_token_id,
                    sid_end_token_id=sid_end_token_id,
                    initial_logprobs=initial_logprobs,
                    lora_request=instance_lora_requests[idx],
                    processed_tokens=processed_by_global_idx.get(idx, 0),
                    max_tokens=max_tokens,
                )
            )

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


GRLLM.beam_search.create_stat = create_stat
