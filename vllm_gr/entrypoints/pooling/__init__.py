# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI


def register_gr_pooling_api_routers(app: FastAPI) -> None:
    from vllm_gr.entrypoints.pooling.recommend.api_router import (
        router as recommend_router,
    )

    app.include_router(recommend_router)
