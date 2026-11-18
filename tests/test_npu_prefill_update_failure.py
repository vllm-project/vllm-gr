# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Exercise Prefill update failure isolation through the real NPU forward wrapper.

Only device APIs and the forward context are mocked. CPU tensors and complete
production modules cover wrapper routing, task updates, and retained ownership.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from tests.test_npu_fia_update_cache import _CacheHarness
from tests.test_npu_fia_update_cache import backend as backend


class _ForwardCase:
    def __init__(self, backend_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        self.cache = _CacheHarness(backend_module, monkeypatch)
        monkeypatch.setitem(
            sys.modules, "vllm_gr.v1.attention.backends.beam_attn_npu", backend_module
        )
        runner_module = importlib.import_module("vllm_gr.v1.worker.npu_prefill_graph_runner")
        self.patch = importlib.import_module("vllm_gr.v1.worker.npu_model_runner_patch")
        monkeypatch.delenv("VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS", raising=False)
        self.owner = SimpleNamespace(
            device=torch.device("cpu"),
            vllm_config=None,
            input_batch=SimpleNamespace(
                req_id_to_index={"a": 0, "b": 1},
                num_computed_tokens_cpu=[0, 0],
                num_prompt_tokens=[2, 2],
            ),
            _current_scheduler_output=SimpleNamespace(
                num_scheduled_tokens={"a": 2, "b": 2}, total_num_scheduled_tokens=4
            ),
        )
        self.runner: Any = runner_module.NPUPrefillGraphRunner(self.owner, buckets=[4, 8])
        self.owner._prefill_graph_runner = self.runner
        self.runner._update_stream = self.cache.update_stream
        self.cache.api.current = SimpleNamespace(
            wait_stream=lambda stream: self.cache.api.record("serve_wait", stream)
        )
        self.graph = Mock()
        self.other_graph = Mock()
        self.bsnd_graph = Mock()
        self.output = torch.zeros((4, 4), device="cpu")
        buf = self.runner.buffers[4]
        buf.block_table_buf = torch.zeros((8, 4), dtype=torch.int32, device="cpu")
        buf.captured_graphs[1] = self.graph
        buf.output_bufs[1] = self.output
        buf.bsnd_captured_graphs = {1: self.bsnd_graph}
        self.runner.entries[4] = self.cache.entry
        self.runner.entries[8] = object()
        self.runner.buffers[8].captured_graphs[1] = self.other_graph
        self.runner.bsnd_entries[(4, 1)] = {"retained": object()}
        self.metadata: Any = SimpleNamespace(
            slot_mapping=torch.arange(4, dtype=torch.int64, device="cpu"),
            block_table=self.cache.block_table,
            query_start_loc_cpu=[0, 2, 4],
            seq_lens_cpu=[8, 12],
            num_reqs=2,
            num_actual_tokens=4,
        )
        context = importlib.import_module("vllm.forward_context")
        monkeypatch.setattr(
            context,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"layer": self.metadata}),
        )
        monkeypatch.setattr(self.patch, "is_prefill_graph_enabled", lambda: True)
        self.native = Mock(return_value=object())

        def native(*args: Any, **kwargs: Any) -> Any:
            return self.native(*args, **kwargs)

        self.forward = self.patch.make_patched_model_forward_npu(native)
        self.input_ids = torch.arange(4, dtype=torch.int32, device="cpu")
        self.positions = self.input_ids.clone()

    def run(self, **kwargs: Any) -> Any:
        inputs = dict(input_ids=self.input_ids, positions=self.positions)
        inputs.update(kwargs)
        return self.forward(self.owner, 4, **inputs)

    def fail(self) -> Exception:
        self.cache.api.failure = ("fia", self.cache.entry.layers[1].handle)
        with pytest.raises(RuntimeError, match="injected task update failure") as caught:
            self.run()
        return cast(Exception, caught.value)

    def assert_no_execution(self) -> None:
        self.native.assert_not_called()
        self.graph.replay.assert_not_called()
        self.other_graph.replay.assert_not_called()
        self.bsnd_graph.replay.assert_not_called()


@pytest.fixture
def case(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> _ForwardCase:
    return _ForwardCase(request.getfixturevalue("backend"), monkeypatch)


def test_success_and_cache_hit_still_replay(case: _ForwardCase) -> None:
    assert case.runner._task_update_error is None
    assert case.run() is case.output
    assert len(case.cache.api.calls) == 2
    assert case.graph.replay.call_count == 1
    assert case.runner._replays_by_bucket == {4: 1}
    case.cache.api.clear()
    assert case.run() is case.output
    assert case.cache.api.calls == []
    assert case.graph.replay.call_count == 2
    assert case.runner._replays_by_bucket == {4: 2}
    case.native.assert_not_called()


@pytest.mark.parametrize("phase", ["begin", "fia", "end"])
@pytest.mark.parametrize("layer_index", [0, 1])
def test_update_failure_preserves_original_error_and_graph_ownership(
    case: _ForwardCase, phase: str, layer_index: int
) -> None:
    entry = case.cache.entry
    buffers = dict(case.runner.buffers)
    workspace = entry.workspace
    bsnd_entry = case.runner.bsnd_entries[(4, 1)]
    task = entry.layers[layer_index]
    case.cache.api.failure = (phase, task.handle)
    with pytest.raises(RuntimeError, match="injected task update failure") as caught:
        case.run()
    assert case.runner._task_update_error is caught.value
    assert entry._last_update_signature is None
    assert entry.workspace is workspace
    assert case.runner.entries[4] is entry
    assert case.runner.buffers == buffers
    assert case.runner.buffers[4].captured_graphs[1] is case.graph
    assert case.runner.bsnd_entries[(4, 1)] is bsnd_entry
    assert case.runner._replays_by_bucket == {}
    case.assert_no_execution()
    # Ending a failed begin/FIA from a finally block would hide or compound
    # the CANN failure. Only operations up to the failing phase may be issued.
    expected = [("enter", case.cache.update_stream), ("wait", case.cache.api.current)]
    for completed in entry.layers[:layer_index]:
        expected.extend((step, completed.handle) for step in ("begin", "fia", "end"))
    phases = ("begin", "fia", "end")
    expected.extend((step, task.handle) for step in phases[: phases.index(phase) + 1])
    expected.append(("exit", case.cache.update_stream))
    assert case.cache.api.events == expected


@pytest.mark.parametrize("layer_index", [0, 1])
def test_binding_lost_after_preflight_cannot_request_native_fallback(
    case: _ForwardCase, monkeypatch: pytest.MonkeyPatch, layer_index: int
) -> None:
    record = case.cache.api.record
    trigger = (
        ("enter", case.cache.update_stream)
        if layer_index == 0
        else ("end", case.cache.entry.layers[0].handle)
    )

    def drop_binding(*event: Any) -> None:
        record(*event)
        if event == trigger:
            case.cache.entry.layers[layer_index].query = None

    monkeypatch.setattr(case.cache.api, "record", drop_binding)
    with pytest.raises(RuntimeError, match="after preflight") as caught:
        case.run()
    assert case.runner._task_update_error is caught.value
    assert case.cache.entry._last_update_signature is None
    assert len(case.cache.api.calls) == layer_index
    case.assert_no_execution()


@pytest.mark.parametrize(
    "route",
    ["prefill", "decode", "disabled", "missing_bucket", "bsnd", "intermediate", "embeds", "extra"],
)
def test_latched_failure_blocks_every_later_forward_route(
    case: _ForwardCase, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    original = case.fail()
    events = list(case.cache.api.events)
    kwargs: dict[str, Any] = {}
    if route == "decode":
        case.owner.input_batch.num_computed_tokens_cpu = [2, 2]
    elif route == "disabled":
        monkeypatch.setattr(case.patch, "is_prefill_graph_enabled", lambda: False)
    elif route == "missing_bucket":
        case.owner._current_scheduler_output.num_scheduled_tokens = {"a": 99}
        case.owner._current_scheduler_output.total_num_scheduled_tokens = 99
    elif route == "bsnd":
        case.runner._BSND_CAPTURE_SPECS = ((4, 1),)
        case.owner._current_scheduler_output.num_scheduled_tokens = {"a": 4}
        case.owner.input_batch.num_prompt_tokens[0] = 4
    elif route == "intermediate":
        kwargs["intermediate_tensors"] = object()
    elif route == "embeds":
        kwargs["inputs_embeds"] = torch.empty((4, 4), device="cpu")
    elif route == "extra":
        kwargs["extra_payload"] = object()
    with pytest.raises(RuntimeError, match="restart the worker") as caught:
        case.run(**kwargs)
    assert caught.value.__cause__ is original
    assert case.runner._task_update_error is original
    assert case.cache.api.events == events
    case.assert_no_execution()


@pytest.mark.parametrize("route", ["tnd_lookup", "other_lookup", "bsnd_lookup", "tnd", "bsnd"])
def test_direct_runner_calls_cannot_reuse_any_graph_after_failure(
    case: _ForwardCase, route: str
) -> None:
    original = case.fail()
    events = list(case.cache.api.events)
    calls: dict[str, Callable[[], Any]] = {
        "tnd_lookup": lambda: case.runner.has_captured_graph(4, 2),
        "other_lookup": lambda: case.runner.has_captured_graph(8, 1),
        "bsnd_lookup": lambda: case.runner.has_bsnd_captured_graph(4, 1),
        "tnd": lambda: case.runner.execute_model_prefill(99, None, None, None, []),
        "bsnd": lambda: case.runner.execute_model_prefill_bsnd(99, 1, None, None, None, []),
    }
    with pytest.raises(RuntimeError, match="restart the worker") as caught:
        calls[route]()
    assert caught.value.__cause__ is original
    assert case.cache.api.events == events
    case.assert_no_execution()


@pytest.mark.parametrize("missing", ["workspace", "query", "input_ids", "positions", "slots"])
def test_safe_preflight_rejection_keeps_native_fallback(case: _ForwardCase, missing: str) -> None:
    kwargs: dict[str, Any] = {}
    if missing == "workspace":
        case.cache.entry.workspace = None
    elif missing == "query":
        case.cache.entry.layers[1].query = None
    elif missing == "slots":
        case.metadata.slot_mapping = None
    else:
        kwargs[missing] = None
    assert case.run(**kwargs) is case.native.return_value
    assert case.runner._task_update_error is None
    assert case.cache.api.events == []
    assert case.cache.api.calls == []
    case.native.assert_called_once()
    case.graph.replay.assert_not_called()
    assert case.runner.has_captured_graph(4, 2)


def test_failure_state_belongs_to_the_runner_instance(case: _ForwardCase) -> None:
    case.fail()
    module = importlib.import_module("vllm_gr.v1.worker.npu_prefill_graph_runner")
    fresh = module.NPUPrefillGraphRunner(SimpleNamespace(device=torch.device("cpu")), buckets=[4])
    fresh.raise_if_failed()
    assert fresh._task_update_error is None
    assert fresh.has_captured_graph(4) is False


pytestmark = pytest.mark.cpu_test
