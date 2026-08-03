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
from collections.abc import Mapping, Sequence
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

from vllm_gr.entrypoints.beam_search_config import (
    BeamSearchModel,
    BeamSearchStep,
)
from vllm_gr.beam_search_decision_utils import (
    complete_eos_candidates,
    extract_and_dedup_flat_logprobs,
    extract_and_dedup_segment,
    mask_candidate_logprobs,
    materialize_selected_beams,
    select_top_indices,
)
from vllm_gr.logger import init_logger
from vllm_gr.models.model_adapter import create_model_adapter
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
    catalog_strategy: str = "fill",
) -> None:
    """Run beam search in batch mode for custom attention backend."""
    instance_prefill_id: dict[int, str] = {}
    instance_prev_beam_internal_ids: dict[int, list[str]] = {}
    instance_fork_info: dict[int, ForkInfo] = {}
    instance_prefix_len: dict[int, int] = {}
    instance_completed_sessions: set[int] = set()

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

            is_error, all_beams_token_id, all_beams_logprob, all_beams_beam_idx, beam_flat_cache = (
                _parse_step_logprobs(
                    result=result,
                    instance=instance,
                    token=token,
                    logprobs_num=logprobs_num,
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
        positions = [0] + list(
            itertools.accumulate(len(instance.beams) for instance in instances_batch)
        )
        instance_ranges = list(zip(positions[:-1], positions[1:]))
        if not all_beams:
            break

        sampling_params = SamplingParams(
            logprobs=step.expand_width,
            max_tokens=1,
            temperature=temperature,
            skip_clone=True,
            detokenize=False,
            flat_logprobs=True,
            allowed_token_ids=step.allowed_token_ids,
        )
        prompts_batch, lora_req_batch = zip(
            *[
                (_create_tokens_prompt_from_beam(beam), beam.lora_request)
                for beam in all_beams
            ]
        )
        output = llm.generate(
            prompts_batch,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_req_batch,
        )

        valid_tokens_sets = []
        if catalog is not None:
            for instance in instances_batch:
                for beam in instance.beams:
                    generated_tokens = beam.tokens[instance.prompt_len :]
                    valid_tokens_sets.append(catalog.valid(generated_tokens))

        for (start, end), instance in zip(instance_ranges, instances_batch):
            all_token_ids = []
            all_logprobs = []
            beam_flat_cache: list[tuple | None] = []
            for index, (current_beam, result) in enumerate(
                zip(all_beams[start:end], output[start:end])
            ):
                flat = result.outputs[0].logprobs
                if flat is None:
                    raise ValueError(
                        "Beam search expected non-empty logprobs for every beam"
                    )
                token_ids, logprobs, ranks, decoded = (
                    extract_and_dedup_flat_logprobs(flat, step.expand_width)
                )
                if catalog is not None:
                    valid_tokens = valid_tokens_sets[start + index]
                    logprobs = [
                        value if token_id in valid_tokens else -float("inf")
                        for token_id, value in zip(token_ids, logprobs)
                    ]
                beam_flat_cache.append((token_ids, logprobs, ranks, decoded))
                all_token_ids.extend(token_ids)
                all_logprobs.extend(
                    current_beam.cum_logprob + value for value in logprobs
                )

            token_ids_array = np.array(all_token_ids)
            logprobs_array = np.array(all_logprobs)
            if eos_token_id is not None and not ignore_eos:
                eos_indices = np.where(token_ids_array == eos_token_id)[0]
                for index in eos_indices:
                    beam_index = index // step.expand_width
                    current_beam = all_beams[start:end][beam_index]
                    cached = beam_flat_cache[beam_index]
                    assert cached is not None
                    completed_beam = BeamSearchSequence(
                        orig_prompt=current_beam.orig_prompt,
                        tokens=current_beam.tokens + [eos_token_id],
                        logprobs=initial_logprobs,
                        cum_logprob=float(logprobs_array[index]),
                        finish_reason="stop",
                        stop_reason=eos_token_id,
                        lora_request=current_beam.lora_request,
                    )
                    completed_beam._lp_parent = current_beam
                    completed_beam._lp_step_data = cached
                    instance.completed.append(completed_beam)
                logprobs_array[eos_indices] = -np.inf

            if len(logprobs_array) <= step.keep_width:
                top_indices = np.arange(len(logprobs_array))
            else:
                top_indices = np.argpartition(
                    np.negative(logprobs_array), step.keep_width
                )[: step.keep_width]

            new_beams = []
            for index in top_indices:
                beam_index = index // step.expand_width
                current_beam = all_beams[start:end][beam_index]
                cached = beam_flat_cache[beam_index]
                assert cached is not None
                new_beam = BeamSearchSequence(
                    orig_prompt=current_beam.orig_prompt,
                    tokens=current_beam.tokens + [int(token_ids_array[index])],
                    logprobs=initial_logprobs,
                    lora_request=current_beam.lora_request,
                    cum_logprob=float(logprobs_array[index]),
                )
                new_beam._lp_parent = current_beam
                new_beam._lp_step_data = cached
                new_beams.append(new_beam)
            instance.beams = new_beams


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
        beam_search_model: BeamSearchModel = BeamSearchModel.ONE_REC,
        **kwargs,
    ):
        # vLLM 0.22 enables ModelRunner V2 automatically for Qwen3. Apply the
        # vllm-gr default before LLM.__init__ constructs VllmConfig so the V1
        # beam-search worker patches remain active.
        from vllm_gr._env_check import configure_model_runner_default

        configure_model_runner_default()

        self._beam_search_model = BeamSearchModel(beam_search_model)
        self.model_adapter = create_model_adapter(self._beam_search_model)
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
        prompts: list[Union["TokensPrompt", "TextPrompt"]] | list[Mapping[str, Any]],
        params: "BeamSearchParams",
        lora_request: Union[list["LoRARequest"], "LoRARequest", None] = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
    ) -> list["BeamSearchOutput"]:
        """
        Generate sequences using beam search.

        Args:
            prompts: Text/token prompts. RECIF accepts mappings containing a
                ``history_sids`` integer sequence.
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
        temperature = params.temperature
        catalog_strategy = getattr(params, "catalog_strategy", "fill")

        adapted = self.model_adapter.adapt_beam_search_request(
            self,
            prompts,
            params,
        )
        prepared = adapted.inputs
        steps = adapted.steps
        ignore_eos = adapted.ignore_eos
        catalog = adapted.catalog

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
        if prepared.begin_token_id is not None:
            initial_logprobs.append_fast(
                [prepared.begin_token_id], [0.0], [None], [None]
            )

        instances: list[BeamSearchInstance] = []
        lora_requests = self._lora_request_to_seq(
            lora_request, len(prepared.engine_inputs)
        )
        for index, (lora_req, engine_prompt) in enumerate(
            zip(lora_requests, prepared.engine_inputs)
        ):
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
            instance.prompt_len = prepared.generated_starts[index]
            instances.append(instance)

        vllm_config = getattr(self.llm_engine, "vllm_config", None)
        attention_backend = None
        if vllm_config is not None and hasattr(vllm_config, "attention_config"):
            attention_backend = getattr(vllm_config.attention_config, "backend", None)
        use_custom_attn = attention_backend is not None and (
            "CUSTOM" in str(attention_backend).upper()
            or getattr(attention_backend, "name", "").upper() == "CUSTOM"
        )
        use_custom_attn = (
            use_custom_attn and self.model_adapter.supports_custom_attention
        )
        for prompt_start in range(0, len(prompts), concurrency_limit):
            instances_batch = instances[prompt_start : prompt_start + concurrency_limit]
            if use_custom_attn:
                tokenizer = prepared.tokenizer
                assert tokenizer is not None
                token_iter: Any = range(len(steps))
                if use_tqdm:
                    token_iter = tqdm(
                        token_iter,
                        desc="Beam search",
                        unit="token",
                        unit_scale=False,
                    )
                    logger.warning(
                        "The progress bar shows the upper bound on token "
                        "steps and may finish early due to stopping conditions."
                    )
                beam_search_params = SamplingParams(
                    logprobs=beam_width,
                    max_tokens=1,
                    temperature=temperature,
                    skip_clone=True,
                    detokenize=False,
                    flat_logprobs=True,
                )
                _custom_beam_search_batch(
                    llm_engine=self.llm_engine,
                    instances_batch=instances_batch,
                    token_iter=token_iter,
                    prompt_start=prompt_start,
                    tokenizer=tokenizer,
                    beam_width=beam_width,
                    beam_search_params=beam_search_params,
                    sid_end_token_id=prepared.end_token_id,
                    ignore_eos=ignore_eos,
                    initial_logprobs=initial_logprobs,
                    logprobs_num=beam_width,
                    catalog=catalog,
                    catalog_strategy=catalog_strategy,
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
                    valid_tokens_sets = []
                    has_catalog = getattr(self, "catalog", None) is not None
                    if has_catalog:
                        for instance in instances_batch:
                            prompt_len = instance.prompt_len
                            for beam in instance.beams:
                                generated_tokens = beam.tokens[prompt_len:]
                                valid_tokens_sets.append(self.catalog.valid(generated_tokens))

                    for (start, end), instance in zip(instance_start_and_end, instances_batch):
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
                                if has_catalog:
                                    valid_tokens_set = valid_tokens_sets[start + i]
                                    logprobs_pos = mask_candidate_logprobs(
                                        token_ids_pos,
                                        logprobs_pos,
                                        valid_tokens_set,
                                    )
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
