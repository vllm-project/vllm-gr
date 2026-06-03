# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai import api_server as openai_api_server
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.usage.usage_lib import UsageContext
from vllm.utils.system_utils import decorate_logs

from vllm_gr.config.gr_config import GRConfig
from vllm_gr.engine.arg_utils import GRAsyncEngineArgs
from vllm_gr.engine.async_gr_engine import GRAsyncLLM
from vllm_gr.entrypoints.pooling import register_gr_pooling_api_routers
from vllm_gr.entrypoints.pooling.recommend.serving import GRServingRecommendation
from vllm_gr.logger import init_logger

logger = init_logger(__name__)


def _apply_client_process_config(
    engine_args: GRAsyncEngineArgs,
    client_config: dict[str, Any] | None,
) -> None:
    if client_config:
        engine_args._api_process_count = client_config.get("client_count", 1)
        engine_args._api_process_rank = client_config.get("client_index", 0)


def _log_disabled_recommend_api(
    model_family: str | None,
    supported_tasks: Any,
) -> None:
    supported_task_names = sorted(str(task) for task in supported_tasks)
    supported_tasks_text = ", ".join(supported_task_names) or "<none>"

    reasons: list[str] = []
    if model_family is None:
        reasons.append(
            "model_family could not be resolved to 'hstu'; when serving from a "
            "local model path, Recommend API requires config.json to declare "
            '"model_type": "hstu" or the server to be started with '
            "--model-family hstu."
        )
    elif model_family != "hstu":
        reasons.append("Recommend API is mounted only for model_family='hstu'.")

    if "embed" not in supported_tasks:
        reasons.append("The model does not advertise the required 'embed' task.")

    logger.warning(
        "Recommend API disabled: model_family=%r, supported_tasks=%s. %s",
        model_family,
        supported_tasks_text,
        " ".join(reasons),
    )


@asynccontextmanager
async def build_async_gr_engine_client(
    args: Namespace,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: bool | None = None,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    engine_args = GRAsyncEngineArgs.from_cli_args(args)
    _apply_client_process_config(engine_args, client_config)

    if disable_frontend_multiprocessing is None:
        disable_frontend_multiprocessing = bool(
            getattr(args, "disable_frontend_multiprocessing", False)
        )

    async with build_async_gr_engine_client_from_engine_args(
        engine_args,
        usage_context=usage_context,
        disable_frontend_multiprocessing=disable_frontend_multiprocessing,
        client_config=client_config,
    ) as engine:
        yield engine


@asynccontextmanager
async def build_async_gr_engine_client_from_engine_args(
    engine_args: GRAsyncEngineArgs,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: bool = False,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
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

    if disable_frontend_multiprocessing:
        logger.warning("V1 is enabled, but got --disable-frontend-multiprocessing.")

    async_llm: GRAsyncLLM | None = None

    client_config = dict(client_config) if client_config else {}
    client_count = client_config.pop("client_count", 1)
    client_index = client_config.pop("client_index", 0)

    try:
        async_llm = GRAsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            gr_config=gr_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
            client_addresses=client_config,
            client_count=client_count,
            client_index=client_index,
        )

        await async_llm.reset_mm_cache()
        yield async_llm
    finally:
        if async_llm is not None:
            async_llm.shutdown()


def build_app(args: Namespace):
    app = openai_api_server.build_app(args)
    register_gr_pooling_api_routers(app)
    return app


async def init_gr_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    model_family: str | None,
) -> None:
    await openai_api_server.init_app_state(engine_client, state, args)

    supported_tasks = await engine_client.get_supported_tasks()

    request_logger = None
    embedding_handler = getattr(state, "openai_serving_embedding", None)
    if embedding_handler is not None:
        request_logger = embedding_handler.request_logger

    if "embed" in supported_tasks and model_family == "hstu":
        state.gr_serving_recommendation = GRServingRecommendation(
            engine_client=engine_client,
            models=state.openai_serving_models,
            request_logger=request_logger,
            log_error_stack=getattr(args, "log_error_stack", False),
        )
    else:
        state.gr_serving_recommendation = None
        _log_disabled_recommend_api(
            model_family=model_family,
            supported_tasks=supported_tasks,
        )


async def run_server(args, **uvicorn_kwargs) -> None:
    decorate_logs("APIServer")
    listen_address, sock = openai_api_server.setup_server(args)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def run_server_worker(
    listen_address,
    sock,
    args,
    client_config=None,
    **uvicorn_kwargs,
) -> None:
    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    log_config = openai_api_server.load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    engine_args = GRAsyncEngineArgs.from_cli_args(args)
    _apply_client_process_config(engine_args, client_config)

    async with build_async_gr_engine_client_from_engine_args(
        engine_args,
        client_config=client_config,
    ) as engine_client:
        app = build_app(args)
        await init_gr_app_state(
            engine_client,
            app.state,
            args,
            model_family=engine_args.model_family,
        )

        logger.info(
            "Starting vLLM-GR API server %d on %s",
            engine_client.vllm_config.parallel_config._api_process_rank,
            listen_address,
        )

        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=openai_api_server.envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )

    try:
        await shutdown_task
    finally:
        sock.close()
