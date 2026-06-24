# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Engine factory: GR recommendation path vs standard vLLM LLM passthrough."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm_gr.adapters.registry import get_model_type_from_config, is_adapter_registered
from vllm_gr.config.gr_config import GRConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine

    from vllm_gr.engine.arg_utils import GRAsyncEngineArgs, GREngineArgs

# Engine classes are imported inside factory helpers to avoid import cycles
# and defer heavy optional-runtime imports until an engine is constructed.


def create_async_llm(
    *,
    engine_args: "GRAsyncEngineArgs | None" = None,
    vllm_config: "VllmConfig | None" = None,
    usage_context: "UsageContext",
    gr_config: GRConfig | None = None,
    **kwargs: Any,
) -> "AsyncLLM":
    """Build ``AsyncLLM`` (LLM) or ``GRAsyncLLM`` (GR).

    Exactly one of ``engine_args`` or ``vllm_config`` must be provided.
    """
    if engine_args is None and vllm_config is None:
        raise ValueError("Either engine_args or vllm_config must be provided")
    if engine_args is not None and vllm_config is not None:
        raise ValueError("Only one of engine_args or vllm_config may be provided")

    if engine_args is not None:
        return _create_async_llm_from_engine_args(
            engine_args,
            usage_context=usage_context,
            gr_config=gr_config,
            **kwargs,
        )

    resolved_gr_config = gr_config or GRConfig()
    return _create_async_llm_from_vllm_config(
        vllm_config,
        gr_config=resolved_gr_config,
        usage_context=usage_context,
        **kwargs,
    )


def _create_async_llm_from_engine_args(
    engine_args: "GRAsyncEngineArgs",
    *,
    usage_context: "UsageContext",
    gr_config: GRConfig | None = None,
    **kwargs: Any,
) -> "AsyncLLM":
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

    return _create_async_llm_from_vllm_config(
        vllm_config=vllm_config,
        gr_config=resolved_gr_config,
        usage_context=usage_context,
        **kwargs,
    )


def _create_async_llm_from_vllm_config(
    vllm_config: "VllmConfig",
    *,
    gr_config: GRConfig,
    usage_context: "UsageContext",
    **kwargs: Any,
) -> "AsyncLLM":
    """Instantiate the correct async engine for the resolved ``model_type``."""
    model_type = get_model_type_from_config(vllm_config)
    if not is_adapter_registered(model_type):
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


def create_sync_engine(
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

    if not is_adapter_registered(engine_args.model_type):
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
