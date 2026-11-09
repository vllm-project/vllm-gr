# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stateless updates based on service state and committed resource versions."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path

from vllm_gr.update_status import ServiceState, ServiceStatus

from .client import ControlClient
from .contracts import Checker, Control, Instance, UpdateContext, UpdateError
from .validation import default_ckpt_checker, default_trie_checker


async def _check_resource(
    status: ServiceStatus,
    context: UpdateContext,
    checker: Checker | None,
) -> None:
    selected = (
        checker
        if checker is not None
        else (
            default_ckpt_checker if context.resource_type == "checkpoint" else default_trie_checker
        )
    )
    resource_path = status.path_for(context.resource_type)
    old_path = Path(resource_path) if resource_path is not None else None
    try:
        # Sync checkers may read large directories; keep other Tasks responsive.
        if inspect.iscoroutinefunction(selected):
            result = await selected(old_path, context.local_path)
        else:
            result = await asyncio.to_thread(selected, old_path, context.local_path)
            if inspect.isawaitable(result):
                result = await result
        if result is not True:
            raise UpdateError("Resource checker rejected the update")
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError(f"Resource checker failed: {exc}") from exc


@dataclass
class PausedUpdateProcess:
    control: Control = field(default_factory=ControlClient)

    async def run(
        self, instance: Instance, context: UpdateContext, *, checker: Checker | None = None
    ) -> bool:
        status = await self.control.status(instance)
        if not status.checkpoint_valid:
            raise UpdateError(
                f"Instance {instance.id} has invalid checkpoint weights; manual recovery is required"
            )
        if status.version_for(context.resource_type) == context.version:
            if status.service_state is ServiceState.PAUSED:
                await self.control.resume(instance, context)
                await _verify_running(self.control, instance, context)
                return True
            return False
        await _check_resource(status, context, checker)
        await self.control.pause_and_drain(instance, context)
        await self.control.reload(instance, context)
        await self.control.resume(instance, context)
        await _verify_running(self.control, instance, context)
        return True


@dataclass
class OnlineUpdateProcess:
    control: Control = field(default_factory=ControlClient)

    async def run(
        self, instance: Instance, context: UpdateContext, *, checker: Checker | None = None
    ) -> bool:
        status = await self.control.status(instance)
        if not status.checkpoint_valid:
            raise UpdateError(
                f"Instance {instance.id} has invalid checkpoint weights; manual recovery is required"
            )
        if status.service_state is ServiceState.PAUSED:
            raise UpdateError(f"Instance {instance.id} is paused; wait for the service")
        if status.version_for(context.resource_type) == context.version:
            return False
        await _check_resource(status, context, checker)
        await self.control.reload(instance, context)
        await _verify_running(self.control, instance, context)
        return True


async def _verify_running(control: Control, instance: Instance, context: UpdateContext) -> None:
    status = await control.status(instance)
    if (
        not status.checkpoint_valid
        or status.version_for(context.resource_type) != context.version
        or status.service_state is not ServiceState.RUNNING
    ):
        raise UpdateError("Instance has not verified the requested version as running")
