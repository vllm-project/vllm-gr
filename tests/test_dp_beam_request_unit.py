# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Data Parallelism support in beam-request beam search (no GPU required).

Tests cover the following DP beam search features:
  - data_parallel_rank threading through beam_search → _beam_request_step
  - DP-aware engine routing in add_requests_async / beam_request_step_update_async
  - FIRST_REQ coordinator notification
  - reqs_in_flight tracking for batch and beam-request step
  - BeamRequestStepUpdate.data_parallel_rank serialization
  - register_beam_output rank passthrough
  - current_wave propagation via beam_request_step_update_async
  - Cleanup (_beam_request_cleanup) passes data_parallel_rank
"""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgspec
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.entrypoints.openai.serving_engine import (
    _beam_request_cleanup,
    _beam_request_step,
)
from vllm_gr.v1.engine.core_client import (
    add_requests_async,
    beam_request_step_update_async,
)
from vllm_gr.v1.engine.core_client_patch import register_beam_output_fn
from vllm_gr.v1.engine.engine_core_patch import (
    ADD_BATCH_WIRE_BYTE,
    BEAM_REQUEST_STEP_UPDATE_WIRE_BYTE,
    _add_enum_member,
)
from vllm_gr.v1.engine.types import BeamRequestStepUpdate

# Ensure custom enum members exist in this process.
_add_enum_member("ADD_BATCH", ADD_BATCH_WIRE_BYTE)
_add_enum_member("BEAM_REQUEST_STEP_UPDATE", BEAM_REQUEST_STEP_UPDATE_WIRE_BYTE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sampling_params(**overrides: Any) -> SamplingParams:
    defaults = dict(max_tokens=1, logprobs=3, temperature=0.0, detokenize=False)
    defaults.update(overrides)
    return SamplingParams(**defaults)


def _make_dp_client_mock(
    *,
    client_index: int = 0,
    engines_running: bool = True,
    num_engines: int = 2,
    has_current_wave: bool = False,
    current_wave: int = 0,
) -> MagicMock:
    """Create a mock that looks like DPLBAsyncMPClient."""
    mock_self = MagicMock()
    mock_self.client_index = client_index
    mock_self.engines_running = engines_running
    mock_self._send_input = AsyncMock()
    mock_self._ensure_output_queue_task = MagicMock()
    mock_self._ensure_stats_update_task = MagicMock()
    mock_self.first_req_send_socket = MagicMock()
    mock_self.first_req_send_socket.send = AsyncMock()
    mock_self.reqs_in_flight = {}

    engines = [f"engine-{i}" for i in range(num_engines)]
    mock_self.core_engines = engines

    def _get_engine(request: Any) -> str:
        rank: int | None = getattr(request, "data_parallel_rank", None)
        idx = rank if rank is not None else 0
        chosen = engines[idx]
        mock_self.reqs_in_flight[request.request_id] = chosen
        return chosen

    mock_self.get_core_engine_for_request = MagicMock(side_effect=_get_engine)

    if has_current_wave:
        mock_self.current_wave = current_wave
    else:
        del mock_self.current_wave

    return mock_self


def _make_non_dp_client_mock(*, client_index: int = 0) -> MagicMock:
    """Create a mock that looks like plain AsyncMPClient (no DP)."""
    mock_self = MagicMock()
    mock_self.client_index = client_index
    mock_self._send_input = AsyncMock()
    mock_self._ensure_output_queue_task = MagicMock()

    del mock_self.get_core_engine_for_request
    del mock_self.reqs_in_flight
    del mock_self.first_req_send_socket
    del mock_self.core_engines
    del mock_self.engines_running
    del mock_self.current_wave

    return mock_self


def _make_request_mock(
    request_id: str = "req-0",
    data_parallel_rank: int | None = None,
) -> MagicMock:
    m = MagicMock()
    m.request_id = request_id
    m.data_parallel_rank = data_parallel_rank
    m.client_index = 0
    m.current_wave = 0
    return m


def _make_beam_update(
    session_id: str = "session-0",
    parent_beam_ids: list[str] | None = None,
    child_beam_ids: list[str] | None = None,
    beam_tokens: list[list[int]] | None = None,
    pruned_ids: list[str] | None = None,
    prefix_len: int = 2,
    beam_width: int = 1,
    data_parallel_rank: int | None = None,
) -> BeamRequestStepUpdate:
    return BeamRequestStepUpdate(
        session_id=session_id,
        parent_beam_ids=parent_beam_ids if parent_beam_ids is not None else ["p0"],
        child_beam_ids=child_beam_ids if child_beam_ids is not None else ["c0"],
        beam_tokens=beam_tokens if beam_tokens is not None else [[50]],
        pruned_ids=pruned_ids if pruned_ids is not None else [],
        prefix_len=prefix_len,
        beam_width=beam_width,
        sampling_params=_sampling_params(),
        data_parallel_rank=data_parallel_rank,
    )


def _finished_queue() -> MagicMock:
    out = MagicMock()
    out.finished = True
    q = MagicMock()
    q.get = AsyncMock(return_value=out)
    return q


# ===========================================================================
# Section 1 — DP add_requests_async
# ===========================================================================


class TestAddRequestsAsyncDP:
    def test_dp_add_batch_chooses_engine_by_rank(self) -> None:
        mock_self = _make_dp_client_mock(num_engines=3)
        reqs = [
            _make_request_mock("r0", data_parallel_rank=2),
            _make_request_mock("r1", data_parallel_rank=2),
        ]
        asyncio.run(add_requests_async(mock_self, reqs, force_batch=True))

        mock_self._send_input.assert_awaited_once()
        called_engine = mock_self._send_input.call_args[0][2]
        assert called_engine == "engine-2"

    def test_dp_add_batch_registers_all_reqs_in_flight(self) -> None:
        mock_self = _make_dp_client_mock()
        reqs = [
            _make_request_mock("r0", data_parallel_rank=0),
            _make_request_mock("r1", data_parallel_rank=0),
        ]
        asyncio.run(add_requests_async(mock_self, reqs, force_batch=True))

        assert "r0" in mock_self.reqs_in_flight
        assert "r1" in mock_self.reqs_in_flight
        assert mock_self.reqs_in_flight["r0"] == mock_self.reqs_in_flight["r1"]

    def test_dp_add_batch_sends_first_req_when_engines_idle(self) -> None:
        mock_self = _make_dp_client_mock(engines_running=False)
        reqs = [_make_request_mock("r0"), _make_request_mock("r1")]
        asyncio.run(add_requests_async(mock_self, reqs, force_batch=True))

        mock_self.first_req_send_socket.send.assert_awaited_once()

    def test_dp_add_batch_no_first_req_when_engines_running(self) -> None:
        mock_self = _make_dp_client_mock(engines_running=True)
        reqs = [_make_request_mock("r0"), _make_request_mock("r1")]
        asyncio.run(add_requests_async(mock_self, reqs, force_batch=True))

        mock_self.first_req_send_socket.send.assert_not_awaited()

    def test_dp_add_batch_calls_ensure_stats_update_task(self) -> None:
        mock_self = _make_dp_client_mock()
        reqs = [_make_request_mock("r0"), _make_request_mock("r1")]
        asyncio.run(add_requests_async(mock_self, reqs, force_batch=True))

        mock_self._ensure_stats_update_task.assert_called()


# ===========================================================================
# Section 2 — DP beam_request_step_update_async
# ===========================================================================


class TestBeamRequestStepUpdateAsyncDP:
    def test_dp_beam_update_routes_by_rank(self) -> None:
        mock_self = _make_dp_client_mock(num_engines=3)
        update = _make_beam_update(data_parallel_rank=2, child_beam_ids=["c0"], pruned_ids=[])

        asyncio.run(beam_request_step_update_async(mock_self, update))

        mock_self._send_input.assert_awaited_once()
        engine_arg = mock_self._send_input.call_args[0][2]
        assert engine_arg == "engine-2"

    def test_dp_beam_update_no_routing_when_rank_is_none(self) -> None:
        mock_self = _make_dp_client_mock()
        update = _make_beam_update(data_parallel_rank=None)

        asyncio.run(beam_request_step_update_async(mock_self, update))

        # Engine argument to _send_input should be None
        engine_arg = mock_self._send_input.call_args[0][2]
        assert engine_arg is None

    def test_dp_beam_update_tracks_child_beam_ids_in_flight(self) -> None:
        mock_self = _make_dp_client_mock()
        update = _make_beam_update(
            data_parallel_rank=0,
            child_beam_ids=["new-0", "new-1"],
            pruned_ids=[],
            parent_beam_ids=["p0", "p0"],
            beam_tokens=[[50], [60]],
            beam_width=2,
        )

        asyncio.run(beam_request_step_update_async(mock_self, update))

        assert "new-0" in mock_self.reqs_in_flight
        assert "new-1" in mock_self.reqs_in_flight
        assert mock_self.reqs_in_flight["new-0"] == "engine-0"
        assert mock_self.reqs_in_flight["new-1"] == "engine-0"

    def test_dp_beam_update_removes_pruned_ids_from_flight(self) -> None:
        mock_self = _make_dp_client_mock()
        mock_self.reqs_in_flight["old-0"] = "engine-0"
        mock_self.reqs_in_flight["old-1"] = "engine-0"

        update = _make_beam_update(
            data_parallel_rank=0,
            child_beam_ids=["new-0"],
            pruned_ids=["old-0"],  # pruned; old-1 survives
            parent_beam_ids=["p0"],
            beam_tokens=[[50]],
            beam_width=1,
        )

        asyncio.run(beam_request_step_update_async(mock_self, update))

        assert "old-0" not in mock_self.reqs_in_flight
        assert "old-1" in mock_self.reqs_in_flight
        assert "new-0" in mock_self.reqs_in_flight

    def test_dp_beam_update_sends_first_req_when_engines_idle(self) -> None:
        mock_self = _make_dp_client_mock(engines_running=False)
        update = _make_beam_update(data_parallel_rank=0)

        asyncio.run(beam_request_step_update_async(mock_self, update))

        mock_self.first_req_send_socket.send.assert_awaited_once()

    def test_dp_beam_update_no_first_req_when_engines_running(self) -> None:
        mock_self = _make_dp_client_mock(engines_running=True)
        update = _make_beam_update(data_parallel_rank=0)

        asyncio.run(beam_request_step_update_async(mock_self, update))

        mock_self.first_req_send_socket.send.assert_not_awaited()

    def test_dp_beam_update_sets_client_index(self) -> None:
        mock_self = _make_dp_client_mock(client_index=5)
        update = _make_beam_update(data_parallel_rank=0)

        asyncio.run(beam_request_step_update_async(mock_self, update))

        assert update.client_index == 5


# ===========================================================================
# Section 3 — register_beam_output rank passthrough
# ===========================================================================


class TestRegisterBeamOutputDP:
    def test_register_beam_output_passes_rank_to_ec_request(self) -> None:
        mock_self = MagicMock()
        mock_self.errored = False
        mock_self.log_requests = False

        register_beam_output_fn(
            mock_self,
            request_id="child-1",
            prompt_token_ids=[10, 20],
            sampling_params=_sampling_params(),
            data_parallel_rank=2,
        )

        ec_req = mock_self.output_processor.add_request.call_args[0][0]
        assert ec_req.data_parallel_rank == 2

    def test_register_beam_output_none_rank(self) -> None:
        mock_self = MagicMock()
        mock_self.errored = False
        mock_self.log_requests = False

        register_beam_output_fn(
            mock_self,
            request_id="child-0",
            prompt_token_ids=[1],
            sampling_params=_sampling_params(),
            data_parallel_rank=None,
        )

        ec_req = mock_self.output_processor.add_request.call_args[0][0]
        assert ec_req.data_parallel_rank is None


# ===========================================================================
# Section 4 — BeamRequestStepUpdate DP rank serialization
# ===========================================================================


class TestBeamRequestStepUpdateDPRank:
    def test_data_parallel_rank_survives_msgpack(self) -> None:
        update = _make_beam_update(data_parallel_rank=3)
        encoded = msgspec.msgpack.encode(update)
        decoded = msgspec.msgpack.decode(encoded, type=BeamRequestStepUpdate)
        assert decoded.data_parallel_rank == 3

    def test_data_parallel_rank_none_survives_msgpack(self) -> None:
        update = _make_beam_update(data_parallel_rank=None)
        encoded = msgspec.msgpack.encode(update)
        decoded = msgspec.msgpack.decode(encoded, type=BeamRequestStepUpdate)
        assert decoded.data_parallel_rank is None

    def test_data_parallel_rank_zero_survives_msgpack(self) -> None:
        update = _make_beam_update(data_parallel_rank=0)
        encoded = msgspec.msgpack.encode(update)
        decoded = msgspec.msgpack.decode(encoded, type=BeamRequestStepUpdate)
        assert decoded.data_parallel_rank == 0


# ===========================================================================
# Section 5 — Cleanup passes rank
# ===========================================================================


class TestCleanupPassesRank:
    def test_cleanup_passes_data_parallel_rank(self) -> None:
        ec = MagicMock()
        ec.beam_request_step_update = AsyncMock()

        asyncio.run(_beam_request_cleanup(ec, "sess-0", ["x"], rank=1))

        update: BeamRequestStepUpdate = ec.beam_request_step_update.call_args[0][0]
        assert update.data_parallel_rank == 1

    def test_cleanup_passes_none_rank(self) -> None:
        ec = MagicMock()
        ec.beam_request_step_update = AsyncMock()

        asyncio.run(_beam_request_cleanup(ec, "sess-0", ["x"], rank=None))

        update = ec.beam_request_step_update.call_args[0][0]
        assert update.data_parallel_rank is None

    def test_cleanup_sends_b0_message(self) -> None:
        ec = MagicMock()
        ec.beam_request_step_update = AsyncMock()

        asyncio.run(_beam_request_cleanup(ec, "sess-0", ["a", "b"], rank=0))

        update = ec.beam_request_step_update.call_args[0][0]
        assert update.beam_width == 0
        assert set(update.pruned_ids) == {"a", "b"}
        assert update.session_id == "sess-0"


# ===========================================================================
# Section 6 — current_wave propagation
# ===========================================================================


class TestCurrentWave:
    def test_beam_update_sets_current_wave_from_client(self) -> None:
        mock_self = _make_dp_client_mock(has_current_wave=True, current_wave=9)
        update = _make_beam_update(data_parallel_rank=0)
        update.current_wave = 0  # will be overwritten

        asyncio.run(beam_request_step_update_async(mock_self, update))

        assert update.current_wave == 9

    def test_beam_update_leaves_wave_unchanged_when_client_missing_attr(self) -> None:
        mock_self = _make_dp_client_mock(has_current_wave=False)
        update = _make_beam_update(data_parallel_rank=0)
        update.current_wave = 42

        asyncio.run(beam_request_step_update_async(mock_self, update))

        assert update.current_wave == 42

    def test_add_batch_sets_current_wave_from_client(self) -> None:
        mock_self = _make_dp_client_mock(has_current_wave=True, current_wave=3)
        reqs = [_make_request_mock("r0"), _make_request_mock("r1")]
        asyncio.run(add_requests_async(mock_self, reqs, force_batch=True))

        for req in reqs:
            assert req.current_wave == 3

    def test_add_batch_leaves_wave_unchanged_when_client_missing_attr(self) -> None:
        mock_self = _make_dp_client_mock(has_current_wave=False)
        reqs = [_make_request_mock("r0"), _make_request_mock("r1")]
        for r in reqs:
            r.current_wave = 77

        asyncio.run(add_requests_async(mock_self, reqs, force_batch=True))

        for req in reqs:
            assert req.current_wave == 77


# ===========================================================================
# Section 7 — Non-DP add_requests_async
# ===========================================================================


class TestAddRequestsAsyncNonDP:
    def test_non_dp_add_batch_sends_without_engine(self) -> None:
        mock_self = _make_non_dp_client_mock()
        reqs = [_make_request_mock("r0"), _make_request_mock("r1")]
        asyncio.run(add_requests_async(mock_self, reqs, force_batch=True))

        mock_self._send_input.assert_awaited_once()
        call_args = mock_self._send_input.call_args[0]
        assert call_args[0] is EngineCoreRequestType.ADD_BATCH

    def test_non_dp_single_no_force_delegates(self) -> None:
        mock_self = _make_non_dp_client_mock()
        mock_self.add_request_async = AsyncMock()
        req = _make_request_mock("r0")
        asyncio.run(add_requests_async(mock_self, [req], force_batch=False))

        mock_self.add_request_async.assert_awaited_once_with(req)

    def test_non_dp_beam_update_no_engine_arg(self) -> None:
        mock_self = _make_non_dp_client_mock()
        update = _make_beam_update(data_parallel_rank=None)

        asyncio.run(beam_request_step_update_async(mock_self, update))

        call_args = mock_self._send_input.call_args[0]
        assert call_args[0] is EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE
        assert call_args[2] is None  # no engine


# ===========================================================================
# Section 8 — _beam_request_step passes rank
# ===========================================================================


class TestStepFunctionsPassRank:
    def test_beam_request_step_passes_rank_in_update(self) -> None:
        ec = MagicMock()
        ec.register_beam_output = MagicMock(return_value=_finished_queue())
        ec.beam_request_step_update = AsyncMock()

        all_beams = [SimpleNamespace(tokens=[10, 50])]
        asyncio.run(
            _beam_request_step(
                ec,
                session_id="s",
                fork_info=[(0, 50)],
                prev_beam_internal_ids=["p0"],
                all_beams=all_beams,
                request_id_batch="r",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                prefix_len=1,
                priority=0,
                data_parallel_rank=1,
            )
        )

        update: BeamRequestStepUpdate = ec.beam_request_step_update.call_args[0][0]
        assert update.data_parallel_rank == 1

    def test_beam_request_step_none_rank_in_update(self) -> None:
        ec = MagicMock()
        ec.register_beam_output = MagicMock(return_value=_finished_queue())
        ec.beam_request_step_update = AsyncMock()

        all_beams = [SimpleNamespace(tokens=[10, 50])]
        asyncio.run(
            _beam_request_step(
                ec,
                session_id="s",
                fork_info=[(0, 50)],
                prev_beam_internal_ids=["p0"],
                all_beams=all_beams,
                request_id_batch="r",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                prefix_len=1,
                priority=0,
                data_parallel_rank=None,
            )
        )

        update = ec.beam_request_step_update.call_args[0][0]
        assert update.data_parallel_rank is None

    def test_beam_request_step_register_beam_output_passes_rank(self) -> None:
        ec = MagicMock()
        ec.register_beam_output = MagicMock(return_value=_finished_queue())
        ec.beam_request_step_update = AsyncMock()

        all_beams = [SimpleNamespace(tokens=[10, 50])]
        asyncio.run(
            _beam_request_step(
                ec,
                session_id="sess-2",
                fork_info=[(0, 50)],
                prev_beam_internal_ids=["p0"],
                all_beams=all_beams,
                request_id_batch="r",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                prefix_len=1,
                priority=0,
                data_parallel_rank=2,
            )
        )

        call_kwargs = ec.register_beam_output.call_args[1]
        assert call_kwargs.get("data_parallel_rank") == 2


# ===========================================================================
# Section 9 — Stats update task
# ===========================================================================


class TestStatsUpdateTask:
    def test_stats_update_task_called_on_multi_add_batch(self) -> None:
        mock_self = _make_dp_client_mock()
        reqs = [_make_request_mock("r0"), _make_request_mock("r1")]
        asyncio.run(add_requests_async(mock_self, reqs, force_batch=True))

        mock_self._ensure_stats_update_task.assert_called()

    def test_stats_update_task_not_called_on_single_no_force(self) -> None:
        mock_self = _make_dp_client_mock()
        mock_self.add_request_async = AsyncMock()
        req = _make_request_mock("r0")
        asyncio.run(add_requests_async(mock_self, [req], force_batch=False))

        mock_self._ensure_stats_update_task.assert_not_called()
