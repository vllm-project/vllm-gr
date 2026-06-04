# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Engine factory: GR recommendation path vs standard vLLM LLM passthrough."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm_gr.config.gr_config import GRConfig
from vllm_gr.config.model_taxonomy import ModelCategory, is_llm_category

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine

    from vllm_gr.engine.arg_utils import GRAsyncEngineArgs, GREngineArgs


def attach_model_category_to_vllm_config(
    vllm_config: "VllmConfig",
    engine_args: Any,
) -> None:
    """Copy ``model_category`` from engine args onto ``vllm_config``."""
    category = getattr(engine_args, "model_category", None)
    if category is not None:
        vllm_config.model_category = category


def create_async_llm_from_engine_args(
    engine_args: "GRAsyncEngineArgs",
    *,
    usage_context: "UsageContext",
    gr_config: GRConfig | None = None,
    **kwargs: Any,
) -> "AsyncLLM":
    """Build ``AsyncLLM`` (LLM) or ``GRAsyncLLM`` (GR) from async engine args."""
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    resolved_gr_config = gr_config or GRConfig(
        max_seq_len=engine_args.max_seq_len,
        max_candidates=engine_args.max_candidates,
    )
    for policy_name in (
        "enable_npu_fused_prefill",
        "force_hammer_npu",
    ):
        if hasattr(resolved_gr_config, policy_name):
            setattr(
                resolved_gr_config,
                policy_name,
                getattr(engine_args, policy_name),
            )

    return create_async_llm_from_vllm_config(
        vllm_config=vllm_config,
        gr_config=resolved_gr_config,
        usage_context=usage_context,
        **kwargs,
    )


def create_async_llm_from_vllm_config(
    vllm_config: "VllmConfig",
    *,
    gr_config: GRConfig,
    usage_context: "UsageContext",
    **kwargs: Any,
) -> "AsyncLLM":
    """Instantiate the correct async engine for ``vllm_config.model_category``."""
    if is_llm_category(getattr(vllm_config, "model_category", None)):
        from vllm.v1.engine.async_llm import AsyncLLM

        return AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            **kwargs,
        )

    from vllm_gr.engine.async_gr_engine import GRAsyncLLM

    return GRAsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        gr_config=gr_config,
        usage_context=usage_context,
        **kwargs,
    )


def create_sync_engine_from_engine_args(
    engine_args: "GREngineArgs",
    *,
    usage_context: "UsageContext",
    stat_loggers: list | None = None,
    enable_multiprocessing: bool = False,
) -> "LLMEngine":
    """Build ``LLMEngine`` (LLM) or ``GRLLMEngine`` (GR) from sync engine args."""
    from vllm.v1.executor import Executor

    vllm_config = engine_args.create_engine_config(usage_context)
    executor_class = Executor.get_class(vllm_config)

    if is_llm_category(getattr(engine_args, "model_category", None)):
        from vllm.v1.engine.llm_engine import LLMEngine

        return LLMEngine.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            disable_log_stats=engine_args.disable_log_stats,
        )

    from vllm_gr.engine.gr_engine import GRLLMEngine

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

    return GRLLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=not engine_args.disable_log_stats,
        usage_context=usage_context,
        stat_loggers=stat_loggers,
        multiprocess_mode=enable_multiprocessing,
        gr_config=gr_config,
    )
