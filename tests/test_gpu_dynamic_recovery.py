# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_gr.v1.worker.beam_decode_graph import (
    BeamDecodeCaptureUnsupported,
    BeamDecodeGraph,
)
from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import V2StageExecutionHooks
from vllm_gr.v1.worker.gpu_dynamic_beam import uses_dynamic_beam


@pytest.mark.parametrize(
    "widths,expected",
    [
        (None, False),
        ((128, 128, 128), False),
        ((64, 64, 64), True),
        ((128, 64, 32), True),
        ((32, 64, 128), True),
    ],
)
def test_dynamic_route_depends_only_on_search_geometry(widths, expected):
    assert (
        uses_dynamic_beam(SimpleNamespace(beam_width=128, dynamic_beam_widths=widths)) is expected
    )


def test_discard_dynamic_graphs_preserves_startup_capture_decisions():
    """Discard buffer-specific refusals, but preserve startup eager decisions.

    ``discard_dynamic_graphs`` clears every captured graph because the adapter's
    forward buffers are replaced.  A refusal recorded against those buffers
    describes a graph that no longer exists, so keeping it would veto the
    replacement this session is about to capture. Startup decisions remain in
    force until the runtime closes, independent of the discarded buffers.
    """
    from vllm_gr.v1.worker.gpu_dynamic_beam import discard_dynamic_graphs

    worker, graph = _graph_runtime()
    graph.refuse_capture("capture refused")
    worker._gr_decode_capture_failures = {1: "startup refused"}
    worker._gr_async_decode_graph_capture_failed = True
    worker._gr_async_decode_graph_capture_error = "startup refused"
    context = SimpleNamespace(registry=SimpleNamespace(num_active_sessions=0))
    discard_dynamic_graphs(worker, context)
    assert worker._gr_dynamic_graphs_discarded
    assert not graph.capture_failures
    assert worker._gr_decode_capture_failures == {1: "startup refused"}
    assert worker._gr_async_decode_graph_capture_failed
    assert worker._gr_async_decode_graph_capture_error == "startup refused"


@pytest.mark.parametrize("global_failure", [False, True])
def test_discard_keeps_startup_failure_on_eager_path(monkeypatch, global_failure):
    from vllm_gr.v1.worker.gpu_dynamic_beam import discard_dynamic_graphs

    worker, graph = _graph_runtime()
    worker.model_config = SimpleNamespace(enforce_eager=False)
    worker._vllm_gr_config = SimpleNamespace(beam_graph_enabled=True, beam_online_batch_sizes=(1,))
    worker._gr_async_decode_graph_capture_failed = global_failure
    worker._gr_decode_capture_failures = {} if global_failure else {1: "startup refused"}
    discard_dynamic_graphs(worker, SimpleNamespace(registry=SimpleNamespace(num_active_sessions=0)))
    hooks = V2StageExecutionHooks(worker)
    decode_graph = Mock(side_effect=graph.select)
    monkeypatch.setattr(hooks, "decode_graph", decode_graph)
    owner = SimpleNamespace(bindings=(None,), inputs=SimpleNamespace(width=4), _forward=Mock())

    assert hooks.decode_logits(owner) is owner._forward.return_value
    decode_graph.assert_not_called()
    assert owner.graph is None


def _graph_runtime():
    worker = SimpleNamespace(model=object(), _beam_context=object())
    graph = object.__new__(BeamDecodeGraph)
    graph.worker = worker
    graph.original_model = worker.model
    graph.context = worker._beam_context
    graph.wrapper = SimpleNamespace(concrete_cudagraph_entries={}, clear_graphs=Mock())
    graph._registries = {}
    graph._bindings = {}
    graph.capture_failures = {}
    graph.select(1, 4)
    worker._beam_decode_graph = graph
    return worker, graph


def test_dynamic_owner_released_while_gc_frozen():
    # A separate process keeps the suite's pre-existing frozen generation intact.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gc
import weakref
from types import SimpleNamespace
import torch
from vllm_gr.v1.worker.gpu_dynamic_beam import GPUDynamicBeam
class Owner:
    pass
owner = Owner()
owner.beam = SimpleNamespace(
    state=SimpleNamespace(max_new_tokens=3),
    session=SimpleNamespace(context=SimpleNamespace(
        max_batch=1, max_decode_steps=3,
        pools=[SimpleNamespace(unshared_key=torch.empty(1))],
    )),
)
owner.bindings = (None,)
owner.dynamic = GPUDynamicBeam(owner)
ref = weakref.ref(owner)
gc.freeze()
del owner
gc.collect()
assert ref() is None
""",
        ],
        check=True,
        timeout=90,
    )


@pytest.mark.parametrize("failure", ["refused", "fatal", "unhealthy", "capturing"])
def test_runtime_capture_recovery(monkeypatch, failure):
    cfg = SimpleNamespace(beam_graph_enabled=True, beam_online_batch_sizes=(1,))
    worker, graph = _graph_runtime()
    worker.model_config = SimpleNamespace(enforce_eager=False)
    worker._vllm_gr_config = cfg
    worker.device = "cuda"
    error = (
        RuntimeError("fatal")
        if failure == "fatal"
        else BeamDecodeCaptureUnsupported("capture refused")
    )
    graph.capture = Mock(side_effect=error)
    hooks = V2StageExecutionHooks(worker)
    monkeypatch.setattr(hooks, "decode_graph", Mock(side_effect=graph.select))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: failure == "capturing")
    sync = Mock(side_effect=RuntimeError("unhealthy") if failure == "unhealthy" else None)
    monkeypatch.setattr(torch.accelerator, "synchronize", sync)
    owner = SimpleNamespace(
        bindings=(None,),
        inputs=SimpleNamespace(width=4),
        dynamic=object(),
        _forward=Mock(return_value="eager"),
        _forward_hidden=Mock(),
    )
    if failure != "refused":
        with pytest.raises(RuntimeError):
            hooks.decode_logits(owner)
        owner._forward.assert_not_called()
        assert not graph.capture_failures
        return
    assert hooks.decode_logits(owner) == "eager"
    assert owner.graph is None
    assert not graph.ready
    assert graph.capture_error == "capture refused"
    sync.assert_called_once_with("cuda")
    assert hooks.decode_logits(owner) == "eager"
    assert graph.capture.call_count == 1
    # Another width must still attempt capture rather than inheriting failure.
    owner.inputs.width = 2
    assert hooks.decode_logits(owner) == "eager"
    assert graph.capture.call_count == 2
    assert {(s.batch_size, s.beam_width) for s in graph.capture_failures} == {(1, 4), (1, 2)}
