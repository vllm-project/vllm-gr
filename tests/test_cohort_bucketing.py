# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for atomic EngineCore cohort release."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.engine.core import _ensure_beam_state
from vllm_gr.v1.engine.engine_core_patch import (
    _patch_scheduler_update_from_output,
    make_patched_handle_client_request,
)
from vllm_gr.v1.engine.wire import register_engine_core_request_types

register_engine_core_request_types()


def _engine():
    engine = SimpleNamespace()
    _ensure_beam_state(engine)
    return engine


def _add(request_id: str, cohort_key=None):
    request = SimpleNamespace(
        request_id=request_id,
        _vllm_gr_beam_cohort_key=cohort_key,
    )
    return (request, 0)


def test_partial_cohort_is_held_until_all_members_arrive() -> None:
    engine = _engine()
    key = (0, 0, 17)
    engine._beam_cohort_expected[key] = 2
    engine._beam_cohort_pending[key] = []
    original = MagicMock()
    handler = make_patched_handle_client_request(original)

    handler(engine, EngineCoreRequestType.ADD, _add("a", key))
    original.assert_not_called()

    handler(engine, EngineCoreRequestType.ADD, _add("b", key))
    assert [call.args[2][0].request_id for call in original.call_args_list] == ["a", "b"]
    assert key not in engine._beam_cohort_expected
    assert key not in engine._beam_cohort_pending


def test_unregistered_cohort_fails_closed() -> None:
    engine = _engine()
    handler = make_patched_handle_client_request(MagicMock())

    with pytest.raises(RuntimeError, match="without registration"):
        handler(engine, EngineCoreRequestType.ADD, _add("a", (0, 0, 9)))


def test_plain_add_bypasses_cohort_gate() -> None:
    engine = _engine()
    original = MagicMock(return_value="ok")
    handler = make_patched_handle_client_request(original)
    request = _add("plain")

    assert handler(engine, EngineCoreRequestType.ADD, request) == "ok"
    original.assert_called_once_with(engine, EngineCoreRequestType.ADD, request)


def test_batched_worker_decisions_are_fanned_out_by_request_id() -> None:
    class Scheduler:
        def update_from_output(self, scheduler_output, model_runner_output):
            return self.outputs

    _patch_scheduler_update_from_output(Scheduler)
    out_a = SimpleNamespace(request_id="a", kv_transfer_params=None)
    out_b = SimpleNamespace(request_id="b", kv_transfer_params=None)
    result = {0: SimpleNamespace(outputs=[out_a, out_b])}
    scheduler = Scheduler()
    scheduler.outputs = result
    scheduler_output = SimpleNamespace(
        beam_data={
            "a": {"is_beam_decode": True},
            "b": {"is_beam_decode": True},
        }
    )
    payload_a = {"session_id": "a", "out_token_ids": [[1]]}
    payload_b = {"session_id": "b", "out_token_ids": [[2]]}
    model_output = SimpleNamespace(
        beam_search_result={
            "by_request": {
                "a": payload_a,
                "b": payload_b,
            }
        }
    )

    assert scheduler.update_from_output(scheduler_output, model_output) is result
    assert out_a.kv_transfer_params["beam_search_result"] == payload_a
    assert out_b.kv_transfer_params["beam_search_result"] == payload_b
