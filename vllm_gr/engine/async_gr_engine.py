# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import cast

from vllm.config import VllmConfig
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerFactory

from vllm_gr.config.gr_config import GRConfig
from vllm_gr.engine.gr_input_processor import GRInputProcessor
from vllm_gr.engine.gr_output_processor import GROutputProcessor
from vllm_gr.inputs.data import GRPrompt
from vllm_gr.outputs import GRRequestOutput


class GRAsyncLLM(AsyncLLM):
    """GR async engine client for online recommendation serving."""

    gr_config: GRConfig

    def _replace_processors(self) -> None:
        """Replace default processors with GR-specific versions.

        Only replaces processors when an adapter is registered for the model.
        Falls back to standard vLLM processors otherwise.
        """
        import logging

        from vllm_gr.adapters.registry import (
            create_adapter,
            get_model_type_from_config,
            is_adapter_registered,
        )

        model_type = get_model_type_from_config(self.vllm_config)

        if not is_adapter_registered(model_type):
            logging.getLogger("vllm_gr.engine").info(
                "No adapter registered for model_type=%r; using standard vLLM processors",
                model_type,
            )
            return

        adapter = create_adapter(model_type, self.vllm_config, self.gr_config)
        logging.getLogger("vllm_gr.engine").info(
            "Created adapter for model_type=%s: %s", model_type, type(adapter).__name__
        )

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

    def _restart_output_handler_with_gr_processors(self) -> None:
        """Restart the background output handler after replacing processors."""
        task = getattr(self, "output_handler", None)
        if task is not None:
            task.cancel()
            self.output_handler = None

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return

        self._run_output_handler()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        gr_config: GRConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "GRAsyncLLM":
        vllm_config.gr_config = gr_config
        base = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            enable_log_requests=enable_log_requests,
            aggregate_engine_logging=aggregate_engine_logging,
            disable_log_stats=disable_log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )
        base.__class__ = cls
        gr_async_llm = cast(GRAsyncLLM, base)
        gr_async_llm.gr_config = gr_config
        gr_async_llm._replace_processors()
        gr_async_llm._restart_output_handler_with_gr_processors()
        return gr_async_llm

    async def encode(
        self,
        prompt: GRPrompt,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: dict[str, str] | None = None,
        priority: int = 0,
        truncate_prompt_tokens: int | None = None,
        tokenization_kwargs: dict | None = None,
    ) -> AsyncGenerator[GRRequestOutput, None]:
        if not isinstance(prompt, dict):
            raise TypeError(f"GRAsyncLLM.encode expects dict input; got {type(prompt).__name__}")
        if pooling_params.task != "embed":
            raise ValueError(
                "GRAsyncLLM.encode requires PoolingParams(task='embed'); "
                f"got task={pooling_params.task!r}"
            )

        async for out in super().encode(
            prompt=prompt,
            pooling_params=pooling_params,
            request_id=request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
            truncate_prompt_tokens=truncate_prompt_tokens,
            tokenization_kwargs=tokenization_kwargs,
        ):
            if not isinstance(out, GRRequestOutput):
                raise TypeError(
                    "GRAsyncLLM expected GRRequestOutput from GROutputProcessor; "
                    f"got {type(out).__name__}"
                )
            yield out


class GRAsyncLLMEngine(GRAsyncLLM):
    """Alias class to align with AsyncLLMEngine naming."""
