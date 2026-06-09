# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GRLLMEngine — extends vLLM's LLMEngine for recommendation inference.

After standard initialization, replaces the default input/output processors
with GR-specific versions (GRInputProcessor, GROutputProcessor) so the
existing add_request() / step() pipeline handles GRPrompt transparently.
"""

import gc
import logging

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory

from vllm_gr.config.gr_config import GRConfig
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_input_processor import GRInputProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor
from vllm_gr.utils.npu_adapter import release_accelerator_memory

logger = logging.getLogger("vllm_gr.engine")


class GRLLMEngine(LLMEngine):
    """LLMEngine with GR-specific input/output processors.

    After calling super().__init__(), replaces self.input_processor and
    self.output_processor so the parent's add_request() and step() work
    with GRPrompt and produce GRRequestOutput without further overrides.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
        gr_config: GRConfig | None = None,
    ) -> None:
        self.gr_config = gr_config or GRConfig()
        # Attach gr_config to vllm_config so workers can access it
        vllm_config.gr_config = self.gr_config

        super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            mm_registry=mm_registry,
            use_cached_outputs=use_cached_outputs,
            multiprocess_mode=multiprocess_mode,
        )

        self._replace_processors()

        logger.info("GRLLMEngine initialized")

    def get_prefix_cache_counters(self) -> dict[str, int]:
        getter = getattr(self.output_processor, "get_prefix_cache_counters", None)
        if callable(getter):
            return getter()
        return {"requests": 0, "queries": 0, "hits": 0}

    def shutdown(self) -> None:
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True

        for attr_name in ("executor", "model_executor"):
            executor = getattr(self, attr_name, None)
            if executor is None:
                continue

            shutdown = getattr(executor, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                close = getattr(executor, "close", None)
                if callable(close):
                    close()

            setattr(self, attr_name, None)

        engine_core = getattr(self, "engine_core", None)
        if engine_core is not None:
            engine_core.shutdown()
            self.engine_core = None

        for attr_name in (
            "input_processor",
            "output_processor",
            "tokenizer",
            "detokenizer",
            "logger_manager",
        ):
            if hasattr(self, attr_name):
                setattr(self, attr_name, None)

        vllm_config = getattr(self, "vllm_config", None)
        compilation_config = getattr(vllm_config, "compilation_config", None)
        static_forward_context = getattr(
            compilation_config,
            "static_forward_context",
            None,
        )
        if isinstance(static_forward_context, dict):
            static_forward_context.clear()
        try:
            from vllm.config import get_cached_compilation_config
        except ImportError:
            get_cached_compilation_config = None
        if get_cached_compilation_config is not None:
            get_cached_compilation_config.cache_clear()

        gc.collect()
        release_accelerator_memory()
        logger.info("GRLLMEngine shutdown complete")

    def _replace_processors(self) -> None:
        """Replace default processors with GR-specific versions.

        Looks up the adapter class from the registry and injects it into both
        input and output processors so they delegate model-specific logic
        correctly.

        Only replaces processors if model_type is detected and a GR adapter
        is registered for it. Falls back to standard vLLM processors otherwise.
        """
        from vllm_gr.adapters.registry import (
            create_adapter,
            is_adapter_registered,
            resolve_model_type,
        )

        model_type = resolve_model_type(self.vllm_config)

        # If no model_type detected, keep standard vLLM processors
        if model_type is None:
            logger.info("No GR model_type detected; using standard vLLM processors")
            return

        # Check if adapter is registered; fall back to standard processors if not
        if not is_adapter_registered(model_type):
            logger.info(
                "No GR adapter registered for model_type=%s; using standard vLLM processors",
                model_type,
            )
            return

        adapter = create_adapter(model_type, self.vllm_config, self.gr_config)
        logger.info("Created adapter for model_type=%s: %s", model_type, type(adapter).__name__)

        self.input_processor = GRInputProcessor(
            vllm_config=self.vllm_config,
            tokenizer=self.tokenizer,
            adapter=adapter,
        )

        stream_interval = self.vllm_config.scheduler_config.stream_interval
        self.output_processor = GROutputProcessor(
            tokenizer=self.tokenizer,
            log_stats=self.log_stats,
            stream_interval=stream_interval,
            adapter=adapter,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: GREngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_multiprocessing: bool = False,
    ) -> "GRLLMEngine":
        """Create GRLLMEngine from engine arguments."""
        vllm_config = engine_args.create_engine_config(usage_context)

        executor_class = Executor.get_class(vllm_config)

        gr_config = GRConfig(
            max_seq_len=engine_args.max_seq_len,
            max_candidates=engine_args.max_candidates,
        )
        for policy_name in (
            "enable_npu_fused_prefill",
            "force_hammer_npu",
        ):
            if hasattr(gr_config, policy_name):
                setattr(gr_config, policy_name, getattr(engine_args, policy_name))

        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
            gr_config=gr_config,
        )
