# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FastAPI router registration for RECIF online inference."""

from functools import wraps
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from vllm_gr.entrypoints.recif.protocol import (
    RecifBeamSearchRequest,
    RecifBeamSearchResponse,
)
from vllm_gr.entrypoints.recif.serving_engine import recif_beam_search

router = APIRouter()


@router.post("/v1/recif/beam_search", response_model=RecifBeamSearchResponse)
async def create_recif_beam_search(
    request: RecifBeamSearchRequest,
    raw_request: Request,
) -> RecifBeamSearchResponse:
    try:
        return await recif_beam_search(
            raw_request.app.state.engine_client,
            request,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def patch_recif_api_router() -> bool:
    """Attach the RECIF router to the upstream OpenAI server app."""

    from vllm.entrypoints.openai import api_server

    current = api_server.build_app
    if getattr(current, "_vllm_gr_recif_router", False):
        return False

    @wraps(current)
    def build_app(*args: Any, **kwargs: Any):
        app = current(*args, **kwargs)
        app.include_router(router)
        return app

    build_app._vllm_gr_recif_router = True  # type: ignore[attr-defined]
    api_server.build_app = build_app
    return True


__all__ = ["patch_recif_api_router", "router"]
