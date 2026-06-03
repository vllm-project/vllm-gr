# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import time

from fastapi import Request
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ErrorResponse
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels

from vllm_gr.entrypoints.pooling.recommend.protocol import (
    GRRecommendRequest,
    GRRecommendResponse,
    GRRecommendResult,
)
from vllm_gr.outputs import GRRequestOutput


class GRRecommendResponseMixin:
    """Pure adapter from GRRequestOutput to GRRecommendResponse."""

    def to_response(
        self,
        output: GRRequestOutput,
        request_id: str,
        created: int,
        model_name: str,
    ) -> GRRecommendResponse:
        recommendations = [
            GRRecommendResult(
                index=idx,
                item_id=item.item_id,
                score=float(item.score),
                task_scores=item.task_scores,
            )
            for idx, item in enumerate(output.recommendations)
        ]

        return GRRecommendResponse(
            id=request_id,
            created=created,
            model=model_name,
            user_id=output.user_id,
            recommendations=recommendations,
        )


class GRServingRecommendation(OpenAIServing):
    request_id_prefix = "recommend"

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        response_mixin: GRRecommendResponseMixin | None = None,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )
        self.response_mixin = response_mixin or GRRecommendResponseMixin()

    async def create_recommend(
        self,
        request: GRRecommendRequest,
        raw_request: Request | None = None,
    ) -> GRRecommendResponse | ErrorResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = self._base_request_id(raw_request, request.request_id)
        if request_id is None:
            return self.create_error_response("Failed to resolve request id")
        if request_id != request.request_id:
            request = request.model_copy(update={"request_id": request_id})

        prompt = request.prompt
        pooling_params = request.to_pooling_params()

        try:
            result_stream = self.engine_client.encode(
                prompt=prompt,
                pooling_params=pooling_params,
                request_id=request_id,
                priority=request.priority,
            )

            final_output: GRRequestOutput | None = None
            async for output in result_stream:
                if raw_request is not None and await raw_request.is_disconnected():
                    await self.engine_client.abort(request_id)
                    return self.create_error_response("Client disconnected")

                if output.finished:
                    if not isinstance(output, GRRequestOutput):
                        return self.create_error_response(
                            "Recommend output type mismatch from engine client"
                        )
                    final_output = output

            if final_output is None:
                return self.create_error_response(
                    "No finished recommendation output received from engine"
                )

            model_name = request.model or self.models.model_name()
            return self.response_mixin.to_response(
                output=final_output,
                request_id=request.request_id,
                created=int(time.time()),
                model_name=model_name,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except Exception as exc:
            return self.create_error_response(exc)
