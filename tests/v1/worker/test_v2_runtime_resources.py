# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""V2 runtime generations and lifecycle cleanup preconditions."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from vllm_gr.v1.worker.gpu.beam_stage_runner import GPUBeamStageRunner
from vllm_gr.v1.worker.gpu.v2.model_runner_v2_resources import (
    close_v2_runtime,
    get_v2_runtime,
)
from vllm_gr.v1.worker.model_runner_patch import (
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


class _StageOwner(SimpleNamespace):
    pass


def _owner(**changes: Any) -> SimpleNamespace:
    return _StageOwner(
        **dict(
            closed=True,
            pending=None,
            control_busy=[False],
            control_lock=Lock(),
            terminal_cleanup_pending=False,
        )
        | changes
    )


def _worker() -> SimpleNamespace:
    graph = SimpleNamespace(uninstall=Mock(), clear=Mock())
    worker = SimpleNamespace(
        _beam_decode_graph=graph,
        _gr_gpu_stage_runner=_owner(),
        model=SimpleNamespace(modules=Mock(return_value=[])),
        constraint_backend=SimpleNamespace(close=Mock()),
    )
    runtime = get_v2_runtime(worker)
    runtime.startup_failures = {2: "startup refused"}
    runtime.runtime_failures[(1, 4)] = "runtime refused"
    runtime.capture_failed = True
    return worker


@pytest.mark.parametrize(  # type: ignore[misc]
    "borrower",
    ["active", "pending", "control", "cached", "retirement", "context", "registry", "workspace"],
)
@pytest.mark.parametrize("operation", ["close", "load", "shutdown"])  # type: ignore[misc]
def test_live_borrower_prevents_all_teardown(borrower: str, operation: str) -> None:
    worker = _worker()
    runtime, graph, backend = (
        worker._gr_v2_runtime,
        worker._beam_decode_graph,
        worker.constraint_backend,
    )
    if borrower == "active":
        worker._gr_gpu_stage_runner.closed = False
    elif borrower == "pending":
        worker._gr_gpu_stage_runner.pending = object()
    elif borrower == "control":
        worker._gr_gpu_stage_runner.control_busy = [True]
    elif borrower == "cached":
        worker._gr_gpu_stage_runner_cache = {2: _owner(closed=False)}
    elif borrower == "retirement":
        worker._gr_retirement_output = SimpleNamespace(owner=_owner(), _output=None)
    elif borrower == "context":
        worker._beam_context = SimpleNamespace(_gr_execution_owner=object())
    elif borrower == "registry":
        worker._beam_context = SimpleNamespace(registry=SimpleNamespace(num_active_sessions=1))
    elif borrower == "workspace":
        worker._beam_postprocess_workspaces = SimpleNamespace(
            keys=(1,), require=lambda key: SimpleNamespace(execution_owner=object())
        )
    native = Mock()

    def original(runner: Any) -> None:
        native(runner)

    run = (
        close_v2_runtime
        if operation == "close"
        else make_load_model_with_constraint_backend(original, close_runtime=close_v2_runtime)
        if operation == "load"
        else make_shutdown_with_constraint_backend(original, close_runtime=close_v2_runtime)
    )
    with pytest.raises(RuntimeError, match="V2 runtime cleanup requires"):
        run(worker)
    assert worker._gr_v2_runtime is runtime
    assert worker._beam_decode_graph is graph
    assert worker.constraint_backend is backend
    graph.uninstall.assert_not_called()
    graph.clear.assert_not_called()
    backend.close.assert_not_called()
    worker.model.modules.assert_not_called()
    native.assert_not_called()


@pytest.mark.parametrize("native_fails", [False, True])  # type: ignore[misc]
def test_terminal_native_cleanup_must_finish_before_teardown(
    monkeypatch: pytest.MonkeyPatch, native_fails: bool
) -> None:
    worker = _worker()
    graph = worker._beam_decode_graph
    owner = object.__new__(GPUBeamStageRunner)
    binding = SimpleNamespace(native_request_id="request-0")
    context = SimpleNamespace(_gr_execution_owner=owner)
    workspace = SimpleNamespace(execution_owner=owner)
    owner.worker = worker
    owner.bindings = (binding,)
    owner.binding = binding
    owner.beam = SimpleNamespace(
        session=SimpleNamespace(context=context),
        state=SimpleNamespace(device="cuda"),
        workspace=workspace,
        end_session=Mock(),
    )
    owner.pending = None
    owner.control_busy = [False]
    owner.control_lock = Lock()
    owner.closed = False
    owner.terminal_release_requested = True
    owner.terminal_cleanup_pending = False
    owner.final_output = Mock()
    owner.last_work = Mock()
    worker._gr_gpu_stage_runner = owner
    worker._beam_sessions = {binding.native_request_id: object()}
    worker._beam_context = context
    worker._beam_postprocess_workspaces = SimpleNamespace(
        keys=(1,), require=Mock(return_value=workspace)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", Mock())

    entered, resume = Event(), Event()

    def consume_native_state(scheduled: Any) -> None:
        assert scheduled.finished_req_ids == {binding.native_request_id}
        entered.set()
        assert resume.wait(10), "test did not resume native cleanup"
        if native_fails:
            raise RuntimeError("native terminal cleanup failed")

    owner.hooks = SimpleNamespace(consume_native_state=consume_native_state)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(owner.release_terminal_if_drained)
        try:
            assert entered.wait(10), "terminal consumer did not enter native cleanup"
            assert owner.closed
            assert context._gr_execution_owner is None
            assert workspace.execution_owner is None
            with pytest.raises(RuntimeError, match="terminal native cleanup"):
                close_v2_runtime(worker)
            graph.clear.assert_not_called()
            assert worker._gr_gpu_stage_runner is owner
        finally:
            resume.set()
        if native_fails:
            with pytest.raises(RuntimeError, match="native terminal cleanup failed"):
                result.result(timeout=10)
        else:
            assert result.result(timeout=10)

    if native_fails:
        with pytest.raises(RuntimeError, match="terminal native cleanup"):
            close_v2_runtime(worker)
        graph.clear.assert_not_called()
    else:
        close_v2_runtime(worker)
        graph.clear.assert_called_once_with()


def test_cleanup_releases_generation_and_receipt_without_releasing_model_resources() -> None:
    import weakref

    worker = _worker()
    owner = worker._gr_gpu_stage_runner
    runtime = worker._gr_v2_runtime
    owner_ref, runtime_ref = weakref.ref(owner), weakref.ref(runtime)
    output = object()
    worker._gr_retirement_output = SimpleNamespace(owner=owner, _output=output)
    worker._gr_gpu_stage_runner_cache = {1: owner}
    worker._gr_gpu_stage_runner_cache_key = object()
    head_cache = object()
    worker.model.lm_head = SimpleNamespace(_gr_unconstrained_head_layout=head_cache)
    context = object()
    worker._beam_context = context
    worker._beam_postprocess_workspaces = SimpleNamespace(keys=())
    pool = worker._beam_postprocess_workspaces
    graph = worker._beam_decode_graph
    del owner, runtime
    close_v2_runtime(worker)
    close_v2_runtime(worker)
    assert owner_ref() is None and runtime_ref() is None
    graph.uninstall.assert_called_once()
    graph.clear.assert_called_once()
    assert worker.model.lm_head._gr_unconstrained_head_layout is head_cache
    assert worker._beam_context is context and worker._beam_postprocess_workspaces is pool
    assert not hasattr(worker, "_gr_retirement_output")
    replacement = get_v2_runtime(worker)
    assert replacement.startup_failures is None
    assert not replacement.runtime_failures and not replacement.capture_failed
    assert replacement.inputs is None and replacement.capture_error is None


@pytest.mark.parametrize("native_fails", [False, True])  # type: ignore[misc]
def test_shutdown_invalidates_before_native_call_even_on_failure(native_fails: bool) -> None:
    worker = _worker()
    graph = worker._beam_decode_graph

    def native(runner: Any) -> None:
        assert not hasattr(runner, "_gr_v2_runtime")
        assert runner._beam_decode_graph is None
        assert runner.constraint_backend is None
        if native_fails:
            raise RuntimeError("native shutdown failed")

    wrapped = make_shutdown_with_constraint_backend(native, close_runtime=close_v2_runtime)
    if native_fails:
        with pytest.raises(RuntimeError, match="native shutdown failed"):
            wrapped(worker)
    else:
        wrapped(worker)
    graph.clear.assert_called_once()


@pytest.mark.parametrize("replacement", ["generation", "inputs", "model"])  # type: ignore[misc]
def test_new_shape_capture_cannot_rebind_a_stale_generation(
    monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    from vllm_gr.v1.worker.gpu.beam_decode_graph import BeamDecodeGraph
    from vllm_gr.v1.worker.gpu.v2.model_runner_v2_resources import (
        V2BeamDecodeGraph,
        V2RuntimeState,
    )

    worker = SimpleNamespace(model=object())
    runtime = get_v2_runtime(worker)
    runtime.inputs = cast(Any, object())
    graph = object.__new__(V2BeamDecodeGraph)
    graph.worker = worker
    graph.runtime = runtime
    graph.input_buffers = runtime.inputs
    graph.original_model = worker.model
    graph.wrapper = SimpleNamespace(concrete_cudagraph_entries={})
    graph.descriptor = object()
    if replacement == "generation":
        worker._gr_v2_runtime = V2RuntimeState(inputs=runtime.inputs)
    elif replacement == "inputs":
        runtime.inputs = None
    else:
        worker.model = object()
    native_capture = Mock()
    monkeypatch.setattr(BeamDecodeGraph, "capture", native_capture)
    with pytest.raises(RuntimeError, match="changed after capture"):
        graph.capture(Mock(), persistent_install=False)
    assert graph.runtime is runtime
    native_capture.assert_not_called()
