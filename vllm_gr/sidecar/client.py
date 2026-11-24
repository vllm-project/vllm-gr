# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""HTTP client for the proposed server maintenance contract.

These routes are not added to the inference server by this module. Servers must
serialize maintenance and implement safe reload, before acknowledging success.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import TypeAdapter, ValidationError

from vllm_gr.update_status import ServiceStatus

from .contracts import (
    PAUSE_PATH,
    RELOAD_TRIE_PATH,
    RELOAD_WEIGHTS_PATH,
    RESUME_PATH,
    STATUS_PATH,
    Instance,
    UpdateContext,
    UpdateError,
)

_STATUS_ADAPTER: TypeAdapter[ServiceStatus] = TypeAdapter(ServiceStatus)


class ControlClient:
    def __init__(self, timeout_seconds: float = 300) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Control timeout must be positive")
        self.timeout_seconds = timeout_seconds

    async def _request(
        self, instance: Instance, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        def invoke() -> dict[str, Any]:
            request = Request(
                instance.control_url.rstrip("/") + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={"Content-Type": "application/json"},
                method="POST" if body is not None else "GET",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    if response.status not in {200, 204}:
                        raise UpdateError("Control operation is not synchronously complete")
                    raw = response.read()
            except (HTTPError, URLError, OSError) as exc:
                # A timed-out call may still run; the next poll checks live status.
                raise UpdateError(
                    f"Control request failed for {instance.id} {path}: {exc}"
                ) from exc
            try:
                value = json.loads(raw) if raw else {}
            except (ValueError, UnicodeError) as exc:
                raise UpdateError("Control response is not valid JSON") from exc
            if not isinstance(value, dict):
                raise UpdateError("Control response must be a JSON object")
            if value.get("ok") is False:
                raise UpdateError(f"Control operation failed: {value}")
            return value

        return await asyncio.to_thread(invoke)

    @staticmethod
    def _body(context: UpdateContext) -> dict[str, Any]:
        return {
            "version": context.version,
            "resource_type": context.resource_type,
        }

    async def pause_and_drain(self, instance: Instance, context: UpdateContext) -> None:
        await self._request(instance, f"{PAUSE_PATH}?mode=wait", self._body(context))

    async def reload(self, instance: Instance, context: UpdateContext) -> None:
        """The server completes loading before replying successfully."""
        body = self._body(context)
        if context.resource_type == "checkpoint":
            body["weights_path"] = str(context.local_path)
            route = RELOAD_WEIGHTS_PATH
        else:
            body["trie_path"] = str(context.local_path)
            route = RELOAD_TRIE_PATH
        await self._request(instance, route, body)

    async def resume(self, instance: Instance, context: UpdateContext) -> None:
        await self._request(instance, RESUME_PATH, self._body(context))

    async def status(self, instance: Instance) -> ServiceStatus:
        value = await self._request(instance, STATUS_PATH)
        try:
            return _STATUS_ADAPTER.validate_python(value)
        except ValidationError as exc:
            raise UpdateError(f"Invalid service status: {exc}") from exc
