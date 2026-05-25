# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import load_aware_call, with_cancellation

from vllm_gr.entrypoints.pooling.recommend.protocol import (
    GRRecommendRequest,
    GRRecommendResponse,
)
from vllm_gr.entrypoints.pooling.recommend.serving import GRServingRecommendation

router = APIRouter()


def recommend(request: Request) -> GRServingRecommendation | None:
    return request.app.state.gr_serving_recommendation


@router.post(
    "/recommend",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_recommend(request: GRRecommendRequest, raw_request: Request):
    handler = recommend(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Recommend API"
        )

    try:
        result = await handler.create_recommend(request, raw_request)
    except Exception as exc:
        return handler.create_error_response(exc)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    if isinstance(result, GRRecommendResponse):
        return JSONResponse(content=result.model_dump())

    assert_never(result)
