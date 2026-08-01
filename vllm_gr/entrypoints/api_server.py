# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OpenAI-compatible API server entry point for vllm-gr.

After removing the GR adapter / recommend runtime, this is a thin wrapper
around vLLM's standard ``vllm.entrypoints.openai.api_server``. The beam-search
and catalog-constrained generation patches needed by the OneRec serving path
are applied via the vLLM general plugin (``vllm_gr.plugin:register``) and do
not require any custom engine wiring here.
"""

from __future__ import annotations

from vllm.entrypoints.openai import api_server as openai_api_server

from vllm_gr.logger import init_logger

logger = init_logger(__name__)


async def run_server(args, **uvicorn_kwargs) -> None:
    """Launch the OpenAI-compatible API server.

    Defers entirely to ``vllm.entrypoints.openai.api_server.run_server``; the
    vLLM general plugin loads the vllm-gr beam-search patches before this runs.
    """
    await openai_api_server.run_server(args, **uvicorn_kwargs)
