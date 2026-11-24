# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""The sidecar client and the production control router share one contract."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import TypeAdapter

from vllm_gr.entrypoints.management import CheckpointReload, Target, TrieReload, router
from vllm_gr.sidecar import ControlClient, Instance, UpdateContext
from vllm_gr.sidecar.contracts import (
    CONTROL_PATHS,
    PAUSE_PATH,
    RELOAD_TRIE_PATH,
    RELOAD_WEIGHTS_PATH,
    RESUME_PATH,
    STATUS_PATH,
)
from vllm_gr.update_status import ServiceState, ServiceStatus


def test_client_paths_match_production_router() -> None:
    assert {route.path for route in router.routes} == CONTROL_PATHS
    assert CONTROL_PATHS == {
        STATUS_PATH,
        PAUSE_PATH,
        RESUME_PATH,
        RELOAD_WEIGHTS_PATH,
        RELOAD_TRIE_PATH,
    }


def test_client_request_bodies_validate_against_production_models(tmp_path: Path) -> None:
    async def run() -> list[tuple[str, dict[str, Any]]]:
        client = ControlClient()
        request = AsyncMock(return_value={})
        instance = Instance("node", "http://127.0.0.1:1")
        checkpoint = UpdateContext("ckpt", "checkpoint", "v2", tmp_path / "ckpt")
        trie = UpdateContext("trie", "trie", "v2", tmp_path / "trie.safetensors")
        with patch.object(client, "_request", request):
            await client.pause_and_drain(instance, checkpoint)
            await client.reload(instance, checkpoint)
            await client.reload(instance, trie)
            await client.resume(instance, trie)
        return [(call.args[1], call.args[2]) for call in request.await_args_list]

    calls = asyncio.run(run())
    assert [path for path, _ in calls] == [
        f"{PAUSE_PATH}?mode=wait",
        RELOAD_WEIGHTS_PATH,
        RELOAD_TRIE_PATH,
        RESUME_PATH,
    ]
    Target.model_validate(calls[0][1])
    CheckpointReload.model_validate(calls[1][1])
    TrieReload.model_validate(calls[2][1])
    Target.model_validate(calls[3][1])


def test_service_status_wire_fields_are_the_shared_contract() -> None:
    status = ServiceStatus(ServiceState.RUNNING, "v1", "t1", "/models/v1", "/tries/t1", True)
    assert (
        TypeAdapter(ServiceStatus).validate_python(
            {
                "service_state": "running",
                "checkpoint_version": "v1",
                "trie_version": "t1",
                "checkpoint_path": "/models/v1",
                "trie_path": "/tries/t1",
                "checkpoint_valid": True,
            }
        )
        == status
    )
    assert {state.value for state in ServiceState} == {"running", "paused"}


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
