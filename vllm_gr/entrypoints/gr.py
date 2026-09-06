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
from typing import TYPE_CHECKING, Any, Union, cast

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

from vllm_gr.beam_search_decision_utils import (
    complete_eos_candidates,
    configure_persistent_sampling_params,
    extract_and_dedup_flat_logprobs,
    materialize_final_beams,
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
from vllm_gr.utils.npu_adapter import release_accelerator_memory
from vllm_gr.v1.engine.types import (
    GR_REQUEST_METADATA_KEY,
    GRRequestMetadata,
    canonicalize_constraint_digest,
)

if TYPE_CHECKING:
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import BeamSearchParams

logger = init_logger(__name__)

ForkInfo = list[tuple[int, int]]


def _create_tokens_prompt_from_beam(beam: "BeamSearchSequence") -> "TokensPrompt":
    """Create a TokensPrompt payload from a BeamSearchSequence."""
    return beam.get_prompt()


def _abort_persistent_requests(llm_engine, request_ids: list[str], cause: BaseException) -> None:
    """Abort admitted native requests when the synchronous frontend fails."""
    if not request_ids:
        return
    abort = getattr(llm_engine, "abort_request", None)
    if not callable(abort):
        error = RuntimeError("persistent GR LLMEngine has no abort boundary")
        error.add_note(f"Original persistent GR failure: {cause!r}")
        raise error from cause
    try:
        abort(request_ids, internal=True)
    except BaseException as cleanup_error:
        cleanup_error.add_note(
            f"Persistent GR abort failed after {cause.__class__.__name__}: {cause}"
        )
        raise cleanup_error from cause


def _prepare_persistent_request(
    llm_engine,
    instance: "BeamSearchInstance",
    prompt_index: int,
    beam_search_params: "SamplingParams",
    max_decode_steps: int,
    constraint_digest: str,
):
    """Create and register the one native request that owns a GR session."""
    request_id = f"offline-gr-{prompt_index}"
    request_params = beam_search_params.clone()
    configure_persistent_sampling_params(
        request_params,
        generation_steps=max_decode_steps,
    )
    execution_width = int(beam_search_params.logprobs or 1)
    metadata = GRRequestMetadata(
        request_id=request_id,
        beam_width=execution_width,
        generation_steps=max_decode_steps,
        prefix_len=len(instance.beams[0].tokens),
        result_width=execution_width,
        constraint_digest=canonicalize_constraint_digest(constraint_digest),
    )
    request_params.extra_args = dict(request_params.extra_args or {})
    request_params.extra_args[GR_REQUEST_METADATA_KEY] = {
        field: getattr(metadata, field) for field in metadata.__struct_fields__
    }
    request = llm_engine.input_processor.process_inputs(
        request_id=request_id,
        prompt=_create_tokens_prompt_from_beam(instance.beams[0]),
        params=request_params,
        supported_tasks=llm_engine.get_supported_tasks(),
        arrival_time=time.time(),
        lora_request=instance.beams[0].lora_request,
    )
    llm_engine.input_processor.assign_request_id(request)
    request.sampling_params.extra_args[GR_REQUEST_METADATA_KEY]["request_id"] = (
        request.request_id
    )
    llm_engine.output_processor.add_request(request, None, None, 0)
    llm_engine.output_processor.request_states[
        request.request_id
    ].gr_persistent_request = True
    return request


def _collect_persistent_output(llm_engine, request) -> Any:
    """Step the synchronous engine until FINAL_ONLY publishes the result."""
    try:
        while True:
            for output in llm_engine.step():
                if output.request_id == request.external_req_id:
                    return output
            if request.request_id not in llm_engine.output_processor.request_states:
                raise RuntimeError(
                    "persistent GR request terminated before final output "
                    f"(internal_request_id={request.request_id!r}, "
                    f"external_request_id={request.external_req_id!r})"
                )
    except BaseException as error:
        if request.request_id in llm_engine.output_processor.request_states:
            _abort_persistent_requests(llm_engine, [request.request_id], error)
        raise


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
    """Run one Constraint Table request through persistent ASYNC scheduling."""
    del token_iter, tokenizer, sid_end_token_id, ignore_eos, initial_logprobs
    del logprobs_num, catalog, catalog_strategy
    if len(instances_batch) != 1:
        raise ValueError("persistent GR requests currently require B=1")
    if not isinstance(
        getattr(constraint_coordinator, "strategy", None), ConstraintTableStrategy
    ):
        raise ValueError(
            "persistent GR requests require constraint_backend='constraint_table'"
        )
    gr_config = GRConfig.from_vllm_config(getattr(llm_engine, "vllm_config", None))
    if not (gr_config.persistent_request and gr_config.persistent_runner_state):
        raise RuntimeError(
            "constraint_table requires the persistent GR ASYNC scheduler"
        )
    constraint_digest = constraint_coordinator.strategy.digest
    request = _prepare_persistent_request(
        llm_engine,
        instances_batch[0],
        prompt_start,
        beam_search_params,
        max_decode_steps,
        constraint_digest,
    )
    try:
        llm_engine.engine_core.add_request(request)
    except BaseException as error:
        _abort_persistent_requests(llm_engine, [request.request_id], error)
        raise

    result = _collect_persistent_output(llm_engine, request)
    from vllm_gr.entrypoints.openai.serving_engine import _worker_decision_from_result
    from vllm_gr.v1.worker.gr_model_runner_state import unpack_gr_final_output

    decision = _worker_decision_from_result(result)
    final_output = unpack_gr_final_output(decision) if decision is not None else None
    if final_output is None:
        raise RuntimeError("persistent GR final output is missing or invalid")
    sequences, _, selected_logprobs, scores = final_output
    instance = instances_batch[0]
    instance.beams = materialize_final_beams(
        instance.beams[0], sequences, selected_logprobs, scores
    )
    if decision.get("counters"):
        logger.info("vllm-gr request counters: %s", decision["counters"])


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
        # skip an already-installed persistent runtime patch (e.g. when the general
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
        decode_prompt_in_text: bool = True,
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
            decode_prompt_in_text: Include the decoded prompt before the
                generated tail in each returned sequence's ``text``.
        """
        # Lazy imports for beam search dependencies
        import itertools

        # TODO: how does beam search work together with length penalty,
        # frequency, penalty, and stopping criteria, etc.?
        logger.debug("Called beam_search")
        beam_width = validate_beam_width(params.beam_width)
        gr_config = GRConfig.from_vllm_config(
            getattr(self.llm_engine, "vllm_config", None)
        )
        if gr_config.persistent_request and len(prompts) != 1:
            raise ValueError(
                "persistent GR requests require B=1; multiple prompts cannot "
                "be split into independent persistent sessions"
            )
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
            prompt_text = tokenizer.decode(prompt_tokens) if decode_prompt_in_text else None
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
            prompt_text = prompt_cache["text"]
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
                beam.text = gen_text if prompt_text is None else prompt_text + gen_text
            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return

        from vllm_gr.utils.engine_lifecycle import shutdown_sync_engine

        engine = getattr(self, "llm_engine", None)
        if engine is not None:
            processor = getattr(engine, "output_processor", None)
            persistent_ids = (
                [request_id for request_id, state in processor.request_states.items()
                 if getattr(state, "gr_persistent_request", False) is True]
                if processor is not None else []
            )
            if not getattr(self, "_engine_shutdown_complete", False):
                # Worker/device drain precedes Scheduler Native KV release.
                # Failure preserves the engine and all frontend records.
                shutdown_sync_engine(engine)
                self._engine_shutdown_complete = True
            if persistent_ids:
                # The core is already closed. Use only the local native API,
                # never enqueue another Worker release RPC during teardown.
                processor.abort_requests(persistent_ids, internal=True)
            self.llm_engine = None
        del engine

        gc.collect()
        release_accelerator_memory()
        self._shutdown_complete = True
        logger.info("GRLLM shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
