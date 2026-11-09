# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"
"""Hot-update endpoints backed by the authoritative Engine Core.

gRPC only: set VLLM_GR_ADMIN_PORT for a separate loopback HTTP control listener.
Sidecar owns update sequencing. Weight loading uses the upstream collective RPC.
"""

from __future__ import annotations

import asyncio
import inspect
import socket
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from vllm_gr.update_status import ResourceType, ServiceState, ServiceStatus

router = APIRouter()
_STATUS_ADAPTER = TypeAdapter(ServiceStatus)


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource_type: ResourceType = "checkpoint"
    version: str = ""


class CheckpointReload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource_type: Literal["checkpoint"] = "checkpoint"
    version: str = Field(min_length=1)
    weights_path: str = Field(min_length=1)


class TrieReload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource_type: Literal["trie"] = "trie"
    version: str = Field(min_length=1)
    trie_path: str = Field(min_length=1)


async def _call(request: Request, method: str, *args: Any) -> Any:
    engine = getattr(request.app.state, "gr_update_engine", None)
    if engine is None:
        raise HTTPException(503, "No inference engine is attached to the control API")
    try:
        result = engine.engine_core.call_utility_async(method, *args)
        if inspect.isawaitable(result):
            result = await result
    except Exception as error:
        raise HTTPException(503, f"Engine control request failed: {error}") from error
    if isinstance(result, dict) and result.get("ok") is False:
        code = {"busy": 409, "invalid": 400}.get(result.get("error_type", ""), 500)
        raise HTTPException(code, result.get("error", "Engine update failed"))
    return result


@router.get("/service/status")
async def service_status(request: Request) -> ServiceStatus:
    result = _STATUS_ADAPTER.validate_python(await _call(request, "service_status"))
    loaded = getattr(request.app.state, "gr_checkpoint", None)
    if loaded is not None:
        result = replace(result, checkpoint_version=loaded[0], checkpoint_path=loaded[1])
    return replace(result, checkpoint_valid=request.app.state.gr_checkpoint_valid)


@router.post("/pause")
async def pause(
    request: Request, target: Target | None = None, mode: Literal["wait"] = "wait"
) -> Any:
    del mode
    result = await _call(request, "pause", target.resource_type if target else "checkpoint")
    # Native drain futures resolve to None; the Sidecar expects a JSON object.
    return {"ok": True} if result is None else result


@router.post("/resume")
async def resume(request: Request, target: Target | None = None) -> Any:
    del target
    if (
        not request.app.state.gr_checkpoint_valid
        or request.app.state.gr_checkpoint_reload_lock.locked()
    ):
        raise HTTPException(409, "Checkpoint is not valid; manual service recovery is required")
    return await _call(request, "resume")


@router.post("/reload_weights")
async def reload_weights(request: Request, target: CheckpointReload) -> Any:
    app = request.app
    if not app.state.gr_checkpoint_valid:
        raise HTTPException(409, "Checkpoint is not valid; restart the model service to recover")
    lock = app.state.gr_checkpoint_reload_lock
    if lock.locked():
        raise HTTPException(409, "Checkpoint reload is already in progress")
    async with lock:
        status = await service_status(request)
        if status.service_state is not ServiceState.PAUSED:
            raise HTTPException(409, "Pause and drain the service before loading weights")
        # Native reload is in-place and cannot roll back a partially loaded model.
        # Keep this latch false on exceptions/cancellation. A fresh model service,
        # not a new Sidecar process, is required to recover after failure.
        app.state.gr_checkpoint_valid = False
        try:
            await app.state.gr_update_engine.collective_rpc(
                "reload_weights", kwargs={"weights_path": target.weights_path}
            )
        except Exception as error:
            raise HTTPException(500, f"Weight reload failed: {error}") from error
        app.state.gr_checkpoint = (target.version, target.weights_path)
        app.state.gr_checkpoint_valid = True
        return {"ok": True, "version": target.version}


@router.post("/reload_trie")
async def reload_trie(request: Request, target: TrieReload) -> Any:
    return await _call(request, "reload_trie", target.trie_path, target.version)


def control_app(engine_client: Any) -> FastAPI:
    app = FastAPI()
    app.state.gr_update_engine = engine_client
    app.state.gr_checkpoint_valid = True
    app.state.gr_checkpoint_reload_lock = asyncio.Lock()
    app.include_router(router)
    return app


class AdminServer:
    """Optional gRPC-side HTTP control listener, stopped before engine teardown."""

    def __init__(self, engine_client: Any, port: int, host: str = "127.0.0.1") -> None:
        import uvicorn

        if not 1 <= port <= 65535:
            raise ValueError("VLLM_GR_ADMIN_PORT must be in [1, 65535]")

        class Server(uvicorn.Server):
            @contextmanager
            def capture_signals(self) -> Iterator[None]:
                # The upstream inference server owns process signals.
                yield

        self.server = Server(
            uvicorn.Config(control_app(engine_client), host=host, port=port, log_level="warning")
        )
        self.host, self.port = host, port
        self.task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        if self.task is not None:
            raise RuntimeError("Control listener has already been started")
        # Bind here so address errors remain ordinary startup exceptions;
        # Uvicorn's own bind failure raises SystemExit in its serving task.
        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        listener = socket.socket(family, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen(128)
            listener.setblocking(False)
        except BaseException:
            listener.close()
            raise

        async def serve() -> None:
            try:
                await self.server.serve(sockets=[listener])
            finally:
                listener.close()

        self.task = asyncio.create_task(serve())
        while not self.server.started:
            if self.task.done():
                await self.task
                raise RuntimeError("Control listener did not start")
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        if self.task is None:
            return
        self.server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.shield(self.task), timeout=10)
        except TimeoutError:
            self.server.force_exit = True
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
