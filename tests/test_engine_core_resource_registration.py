# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.beam.controller_types import BeamControllerRequest
from vllm_gr.v1.engine.async_llm import run_beam_controller_fn
from vllm_gr.v1.engine.core_client import ensure_beam_constraint_resource_async
from vllm_gr.v1.engine.wire import register_engine_core_request_types

register_engine_core_request_types()


class _MPClient:
    def __init__(self, *, dp: bool = False) -> None:
        self.client_index = 4
        self.current_wave = 2
        self.utility_results: dict[int, asyncio.Future] = {}
        self.engines_running = True
        self.sent: list[tuple[Any, Any, Any]] = []
        if dp:
            self.core_engines = ["engine-0", "engine-1"]

    def _send_input(self, request_type: Any, request: Any, engine: Any = None):
        async def send() -> None:
            self.sent.append((request_type, request, engine))
            self.utility_results[request.call_id].set_result(
                {
                    "resource_id": request.resource_id,
                    "status": "registered",
                    "error_code": None,
                    "error": None,
                }
            )

        return send()

    def _ensure_output_queue_task(self) -> None:
        pass


def test_registration_is_single_flight_and_cached_per_wave() -> None:
    async def run() -> None:
        client = _MPClient()
        calls = [
            ensure_beam_constraint_resource_async(
                client,
                "a" * 64,
                {"payload": b"large"},
            )
            for _ in range(16)
        ]
        results = await asyncio.gather(*calls)
        assert len(client.sent) == 1
        assert all(result["resource_id"] == "a" * 64 for result in results)
        assert client.sent[0][0] is EngineCoreRequestType.BEAM_CONSTRAINT_RESOURCE_REGISTER

        cached = await ensure_beam_constraint_resource_async(
            client, "a" * 64, {"payload": b"must-not-send"}
        )
        assert cached["status"] == "already_registered"
        assert len(client.sent) == 1

        client.current_wave += 1
        await ensure_beam_constraint_resource_async(
            client, "a" * 64, {"payload": b"send-after-restart"}
        )
        assert len(client.sent) == 2

    asyncio.run(run())


def test_registration_is_scoped_and_routed_by_dp_rank() -> None:
    async def run() -> None:
        client = _MPClient(dp=True)
        await ensure_beam_constraint_resource_async(
            client,
            "b" * 64,
            {"payload": b"rank-0"},
            data_parallel_rank=0,
        )
        await ensure_beam_constraint_resource_async(
            client,
            "b" * 64,
            {"payload": b"rank-1"},
            data_parallel_rank=1,
        )
        assert [entry[2] for entry in client.sent] == ["engine-0", "engine-1"]

        with pytest.raises(ValueError, match="data_parallel_rank is required"):
            await ensure_beam_constraint_resource_async(
                client,
                "c" * 64,
                {"payload": b"missing-rank"},
            )

    asyncio.run(run())


def test_single_engine_does_not_require_explicit_dp_rank() -> None:
    async def run() -> None:
        client = _MPClient()
        client.core_engines = ["engine-0"]
        result = await ensure_beam_constraint_resource_async(
            client,
            "c" * 64,
            {"payload": b"large"},
        )

        assert result["status"] == "registered"
        assert client.sent[0][2] is None

    asyncio.run(run())


def test_async_llm_registers_before_start_and_strips_payload() -> None:
    events: list[tuple[str, Any]] = []

    class _EngineCore:
        async def ensure_beam_constraint_resource_async(
            self, resource_id: str, resource: dict, *, data_parallel_rank: int | None
        ) -> None:
            events.append(("register", (resource_id, resource, data_parallel_rank)))

        async def run_beam_controller_async(self, start: Any) -> dict:
            events.append(("start", start))
            return {"ok": True}

    class _InputProcessor:
        def process_inputs(self, request_id: str, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(request_id=request_id, external_req_id=None)

    async def run() -> None:
        llm = SimpleNamespace(
            errored=False,
            engine_core=_EngineCore(),
            input_processor=_InputProcessor(),
        )

        async def supported_tasks() -> set[str]:
            return {"generate"}

        llm.get_supported_tasks = supported_tasks
        controller = BeamControllerRequest(
            beam_session_id="session-0",
            external_request_id="request-0",
            beam_width=2,
            max_tokens=2,
            prompt_token_ids=[1],
            initial_tokens=[1],
            prompt_length=1,
            eos_token_id=2,
            constraint_resource_id="d" * 64,
        )
        result = await run_beam_controller_fn(
            llm,
            {"prompt_token_ids": [1]},
            SamplingParams(max_tokens=1),
            controller,
            data_parallel_rank=1,
            constraint_resource={"payload": b"large"},
        )
        assert result == {"ok": True}

    asyncio.run(run())
    assert [event[0] for event in events] == ["register", "start"]
    assert events[0][1] == ("d" * 64, {"payload": b"large"}, 1)
    assert events[1][1].constraint_resource is None
    assert events[1][1].data_parallel_rank == 1


def test_async_llm_reregisters_and_retries_unknown_resource_once() -> None:
    events: list[tuple[str, Any]] = []

    class _EngineCore:
        starts = 0

        async def ensure_beam_constraint_resource_async(
            self,
            resource_id: str,
            resource: dict,
            *,
            data_parallel_rank: int | None,
            force: bool = False,
        ) -> None:
            events.append(("register", force))

        async def run_beam_controller_async(self, start: Any) -> dict:
            self.starts += 1
            events.append(("start", start.constraint_resource))
            if self.starts == 1:
                return {
                    "error": "unknown resource",
                    "error_code": "UNKNOWN_RESOURCE_ID",
                }
            return {"error": None, "ok": True}

    class _InputProcessor:
        def process_inputs(self, request_id: str, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(request_id=request_id, external_req_id=None)

    async def run() -> None:
        llm = SimpleNamespace(
            errored=False,
            engine_core=_EngineCore(),
            input_processor=_InputProcessor(),
        )

        async def supported_tasks() -> set[str]:
            return {"generate"}

        llm.get_supported_tasks = supported_tasks
        controller = BeamControllerRequest(
            beam_session_id="session-retry",
            external_request_id="request-retry",
            beam_width=1,
            max_tokens=1,
            prompt_token_ids=[1],
            initial_tokens=[1],
            prompt_length=1,
            eos_token_id=2,
            constraint_resource_id="e" * 64,
        )
        result = await run_beam_controller_fn(
            llm,
            {"prompt_token_ids": [1]},
            SamplingParams(max_tokens=1),
            controller,
            data_parallel_rank=0,
            constraint_resource={"payload": b"large"},
        )
        assert result == {"error": None, "ok": True}

    asyncio.run(run())
    assert events == [
        ("register", False),
        ("start", None),
        ("register", True),
        ("start", None),
    ]
