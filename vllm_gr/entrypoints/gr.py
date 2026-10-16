# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline vLLM entry point with persistent GR Beam Search V1 execution."""

import gc
from typing import TYPE_CHECKING, Any, NoReturn, Union, cast

from vllm.entrypoints.generate.beam_search.utils import (
    BeamSearchOutput,
)
from vllm.entrypoints.llm import LLM
from vllm.inputs import TokensPrompt

from vllm_gr.config import BeamExecutionMode, ConstraintBackend, GRConfig
from vllm_gr.config.validation import validate_beam_max_width
from vllm_gr.constraints import (
    BeamConstraintCoordinator,
    create_constraint_strategy,
)
from vllm_gr.constraints.startup import install_constraint_table
from vllm_gr.constraints.strategies import (
    ConstraintTableStrategy,
    TrieConstraintStrategy,
)
from vllm_gr.engine.arg_utils_gr import engine_args_accepts_gr_values
from vllm_gr.logger import init_logger
from vllm_gr.utils.npu_adapter import release_accelerator_memory

if TYPE_CHECKING:
    from vllm.entrypoints.generate.beam_search.utils import BeamSearchOutput
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import BeamSearchParams

    from vllm_gr.entrypoints.beam_search_v1.protocol import BeamSearchFrontend

logger = init_logger(__name__)


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


class GRLLM(LLM):
    """LLM subclass exposing persistent Beam Search through beam_search_v1."""

    def __init__(
        self,
        model: str = "",
        *,
        catalog_path: str | None = None,
        constraint_backend: ConstraintBackend | str | None = None,
        constrained_generation: bool | None = None,
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
        self._frontend_gc_registered = False

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
            constrained_generation is not False
            and constraint_backend is None
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
            gr_engine_args["constrained_generation"] = constrained_generation
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

        _install_renderer_eos_token_id_cache()

        self.catalog = None
        self.constraint_strategy = None
        self.constraint_coordinator = BeamConstraintCoordinator(None)
        # Resolve the effective constraint backend from the engine config
        # (create_engine_config already applied the NPU platform default: a
        # sentinel catalog_path when Constraint Tables are auto-selected).
        # Explicit user choices pass through unchanged.
        resolved_gr = GRConfig.from_vllm_config(getattr(self.llm_engine, "vllm_config", None))
        if resolved_gr.constrained_generation is not False:
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

        if resolved_gr.beam_execution_mode is BeamExecutionMode.V1:
            from vllm_gr.utils.engine_lifecycle import acquire_frontend_startup_heap

            try:
                acquire_frontend_startup_heap()
            except Exception:
                self.shutdown()
                raise
            self._frontend_gc_registered = True

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

    def beam_search(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Reject the retired API instead of inheriting vLLM's implementation."""
        raise NotImplementedError(
            "GRLLM.beam_search() has been removed. Use beam_search_v1() with "
            "beam.execution_mode='v1' and a canonical constraint artifact. "
            "Legacy arguments and stopping rules are not translated automatically."
        )

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True

        from vllm_gr.utils.engine_lifecycle import shutdown_sync_engine

        try:
            if hasattr(self, "llm_engine") and self.llm_engine is not None:
                shutdown_sync_engine(self.llm_engine)
                self.llm_engine = None
        finally:
            if getattr(self, "_frontend_gc_registered", False):
                from vllm_gr.utils.engine_lifecycle import release_frontend_startup_heap

                self._frontend_gc_registered = False
                release_frontend_startup_heap()

        gc.collect()
        release_accelerator_memory()
        logger.info("GRLLM shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
