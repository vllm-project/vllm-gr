# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.config import CUDAGraphMode

from vllm_gr.v1.worker import beam_decode_graph as shared


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.NONE])  # type: ignore[misc]
@pytest.mark.parametrize("is_beam", [False, True])  # type: ignore[misc]
def test_only_beam_full_uses_beam_registry(
    monkeypatch: pytest.MonkeyPatch, mode: CUDAGraphMode, is_beam: bool
) -> None:
    graph = SimpleNamespace(
        original_model=Mock(return_value="native"), model_forward=Mock(return_value="beam")
    )
    context = SimpleNamespace(
        cudagraph_runtime_mode=mode,
        attn_metadata={"layer": SimpleNamespace(is_beam_decode=is_beam)},
    )
    monkeypatch.setattr(shared, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(shared, "get_forward_context", lambda: context)
    model = object.__new__(shared._BeamDecodeModel)
    model.graph = graph
    expected_beam = is_beam and mode is CUDAGraphMode.FULL
    assert model(input_ids="fixed") == ("beam" if expected_beam else "native")
    assert graph.model_forward.call_count == int(expected_beam)
    assert graph.original_model.call_count == int(not expected_beam)


def test_context_replacement_fails_before_graph_launch() -> None:
    graph = object.__new__(shared.BeamDecodeGraph)
    graph.context = object()
    graph.worker = SimpleNamespace(_beam_context=object())
    with pytest.raises(RuntimeError, match="context changed"):
        graph.replay()


def test_native_model_discovery_unwraps_beam_wrapper() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    model = object.__new__(shared._BeamDecodeModel)
    model.runnable = object()
    assert GPUModelRunner.get_model(SimpleNamespace(model=model)) is model.runnable
