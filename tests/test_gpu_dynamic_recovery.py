# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeCaptureUnsupported
from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import CudaStageHooksV2
from vllm_gr.v1.worker.gpu_dynamic_beam import uses_dynamic_beam
from vllm_gr.v1.worker.gpu_model_runner_v2_resources import (
    V2RuntimeState,
    get_v2_runtime,
)


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


def test_per_parent_beam_widths_uses_dynamic_selection_path():
    assert uses_dynamic_beam(
        SimpleNamespace(
            beam_width=128,
            dynamic_beam_widths=(128, 128, 128),
            per_parent_beam_widths=(64, 32, 16),
        )
    )


def test_discard_dynamic_graphs_drops_runtime_capture_refusals():
    """A discarded graph takes its refusal record with it.

    ``discard_dynamic_graphs`` clears every captured graph because the adapter's
    forward buffers are replaced.  A refusal recorded against those buffers
    describes a graph that no longer exists, so keeping it would veto the
    replacement this session is about to capture.
    """
    from vllm_gr.v1.worker.gpu_dynamic_beam import discard_dynamic_graphs

    graph = Mock()
    worker = SimpleNamespace(
        _beam_decode_graph=graph,
        _gr_v2_runtime=V2RuntimeState(runtime_failures={(1, 4): "capture refused"}),
    )
    context = SimpleNamespace(registry=SimpleNamespace(num_active_sessions=0))
    discard_dynamic_graphs(worker, context)
    graph.clear.assert_called_once()
    assert get_v2_runtime(worker).graphs_discarded
    assert not get_v2_runtime(worker).runtime_failures


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
    worker = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        _vllm_gr_config=cfg,
        device="cuda",
    )
    graph = Mock(ready=False)
    error = (
        RuntimeError("fatal")
        if failure == "fatal"
        else BeamDecodeCaptureUnsupported("capture refused")
    )
    graph.capture.side_effect = error
    hooks = CudaStageHooksV2(worker)
    monkeypatch.setattr(hooks, "decode_graph", Mock(return_value=graph))
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
        assert not get_v2_runtime(worker).runtime_failures
        return
    assert hooks.decode_logits(owner) == "eager"
    assert owner.graph is None
    graph.discard_selected.assert_called_once()
    sync.assert_called_once_with("cuda")
    assert hooks.decode_logits(owner) == "eager"
    assert graph.capture.call_count == 1
    # Another width must still attempt capture rather than inheriting failure.
    owner.inputs.width = 2
    assert hooks.decode_logits(owner) == "eager"
    assert graph.capture.call_count == 2
    assert set(get_v2_runtime(worker).runtime_failures) == {(1, 4), (1, 2)}


pytestmark = pytest.mark.cpu_test
