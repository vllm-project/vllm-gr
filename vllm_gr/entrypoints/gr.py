# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline vLLM entry point with persistent GR Beam Search V1 execution."""

import gc
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Union, cast, overload

from vllm.entrypoints.generate.beam_search.utils import (
    BeamSearchOutput,
)
from vllm.entrypoints.llm import LLM
from vllm.inputs import TokensPrompt

from vllm_gr.config import BeamExecutionMode, ConstraintBackend, GRConfig
from vllm_gr.config.validation import validate_beam_max_width
from vllm_gr.logger import init_logger
from vllm_gr.utils.npu_adapter import release_accelerator_memory

if TYPE_CHECKING:
    from vllm.entrypoints.generate.beam_search.utils import BeamSearchOutput
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import BeamSearchParams

    from vllm_gr.entrypoints.beam_search_v1.protocol import BeamSearchFrontend
    from vllm_gr.outputs import CompactBeamSearchOutput

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
        if catalog_path is not None or constraint_backend is not None:
            raise ValueError(
                "Legacy catalog_path/constraint_backend arguments have been removed. "
                "Use vllm_gr_config with beam.execution_mode='v1' and a canonical "
                "constraint_table artifact."
            )
        beam_max_width = validate_beam_max_width(beam_max_width)
        # Apply the vllm-gr ModelRunner policy before LLM.__init__ constructs
        # VllmConfig. This keeps V1 as the default and preserves an explicit
        # V2 opt-in for the adapted beam_search_v1 path.
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

        if constrained_generation is not None:
            kwargs["constrained_generation"] = constrained_generation

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
            **kwargs,
        )

        _install_renderer_eos_token_id_cache()

        resolved_gr = GRConfig.from_vllm_config(getattr(self.llm_engine, "vllm_config", None))
        if resolved_gr.beam_execution_mode is BeamExecutionMode.V1:
            from vllm_gr.utils.engine_lifecycle import acquire_frontend_startup_heap

            try:
                acquire_frontend_startup_heap()
            except Exception:
                self.shutdown()
                raise
            self._frontend_gc_registered = True

    @overload
    def beam_search_v1(
        self,
        prompts: list[Union["TokensPrompt", "TextPrompt"]],
        params: "BeamSearchParams",
        lora_request: Union[list["LoRARequest"], "LoRARequest", None] = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
        trie_ids: list[int] | tuple[int, ...] | None = None,
        *,
        output_mode: Literal["full"],
    ) -> list["BeamSearchOutput"]: ...

    @overload
    def beam_search_v1(
        self,
        prompts: list[Union["TokensPrompt", "TextPrompt"]],
        params: "BeamSearchParams",
        lora_request: Union[list["LoRARequest"], "LoRARequest", None] = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
        trie_ids: list[int] | tuple[int, ...] | None = None,
        *,
        output_mode: Literal["compact"] = "compact",
    ) -> list["CompactBeamSearchOutput"]: ...

    def beam_search_v1(
        self,
        prompts: list[Union["TokensPrompt", "TextPrompt"]],
        params: "BeamSearchParams",
        lora_request: Union[list["LoRARequest"], "LoRARequest", None] = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
        trie_ids: list[int] | tuple[int, ...] | None = None,
        *,
        output_mode: Literal["full", "compact"] = "compact",
    ) -> list["BeamSearchOutput"] | list["CompactBeamSearchOutput"]:
        """Run persistent Beam Search with eager generated-only output by default.

        Explicit ``full`` preserves native lazy BeamSearchOutput objects. ``compact``
        returns short token IDs, cumulative scores and stop metadata, with the
        accepted original prompt stored once per request. It never constructs
        full prompt-plus-tail tokens or text; callers decode the tail if needed.
        Search, ranking and configured begin/end/stop boundaries are identical.
        """
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
            output_mode=output_mode,
            trie_ids=trie_ids,
        )
        return cast(
            list["BeamSearchOutput"] | list["CompactBeamSearchOutput"],
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
