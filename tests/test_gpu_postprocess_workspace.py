# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared CUDA workspace initialization, storage identity and load accounting."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker.cuda import gpu_postprocess_workspace as workspace_support

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


# ---------------------------------------------------------------------------
# Persistent postprocess workspace load hook
# ---------------------------------------------------------------------------


def test_postprocess_workspace_load_wrapper_initializes_after_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def original(runner: Any, marker: Any) -> str:
        calls.append(("load", marker))
        return "loaded"

    def initialize(runner: Any) -> None:
        calls.append(("workspace", runner))

    monkeypatch.setattr(workspace_support, "initialize_beam_postprocess_workspaces", initialize)
    wrapped = workspace_support.make_load_model_with_beam_postprocess_workspace(original)
    runner = SimpleNamespace()

    assert wrapped(runner, "weights") == "loaded"
    assert calls == [("load", "weights"), ("workspace", runner)]
    assert workspace_support.make_load_model_with_beam_postprocess_workspace(wrapped) is wrapped


def test_postprocess_workspace_load_wrapper_accounts_device_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_pool = SimpleNamespace(device_memory_bytes=37)

    def original(runner: Any) -> str:
        # Native load_model recomputes this value on every model load.
        runner.model_memory_usage = 100
        return "loaded"

    monkeypatch.setattr(
        workspace_support,
        "initialize_beam_postprocess_workspaces",
        lambda runner: workspace_pool,
    )

    wrapped = workspace_support.make_load_model_with_beam_postprocess_workspace(original)
    runner = SimpleNamespace()

    assert wrapped(runner) == "loaded"
    assert runner.model_memory_usage == 137
    # Reusing an existing pool after reload must be accounted once relative
    # to the freshly measured model memory, rather than accumulated twice.
    assert wrapped(runner) == "loaded"
    assert runner.model_memory_usage == 137


def test_postprocess_workspace_initializer_is_noop_off_cuda() -> None:
    runner = SimpleNamespace(device=torch.device("cpu"))
    assert workspace_support.initialize_beam_postprocess_workspaces(runner) is None
    assert not hasattr(runner, "_beam_postprocess_workspaces")


@pytest.mark.parametrize("failure", ["load", "workspace"])  # type: ignore[misc]
def test_load_wrapper_propagates_failure_without_accounting(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from unittest.mock import Mock

    error = RuntimeError(failure)
    initialize = Mock(side_effect=error if failure == "workspace" else None)
    monkeypatch.setattr(workspace_support, "initialize_beam_postprocess_workspaces", initialize)

    def original(runner: Any) -> None:
        if failure == "load":
            raise error

    runner = SimpleNamespace(model_memory_usage=100)
    wrapped = workspace_support.make_load_model_with_beam_postprocess_workspace(original)
    with pytest.raises(RuntimeError) as caught:
        wrapped(runner)
    assert caught.value is error
    assert runner.model_memory_usage == 100
    assert initialize.call_count == (failure == "workspace")


def test_postprocess_workspace_initializer_matches_v1_capacity_and_reuses_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    from vllm_gr.config import SUPPORTED_BEAM_BATCH_SIZES, BeamExecutionMode, GRConfig
    from vllm_gr.v1.beam.gpu_beam_postprocess_workspace import (
        BeamPostprocessWorkspace,
        BeamPostprocessWorkspaceKey,
    )

    capacity = max(SUPPORTED_BEAM_BATCH_SIZES)
    key = BeamPostprocessWorkspaceKey(4, 4, torch.float16)
    workspace = SimpleNamespace(
        key=key,
        max_batch=capacity,
        max_decode_steps=3,
        device_memory_bytes=0,
        pinned_memory_bytes=0,
    )
    allocate = Mock(return_value=workspace)
    monkeypatch.setattr(BeamPostprocessWorkspace, "allocate", allocate)
    runner = SimpleNamespace(
        device=torch.device("cuda"),
        model_config=SimpleNamespace(dtype=torch.float16),
        _vllm_gr_config=GRConfig(
            beam_execution_mode=BeamExecutionMode.V1,
            beam_worker_decision=True,
            beam_max_width=4,
            beam_max_decode_steps=3,
        ),
    )

    pool = workspace_support.initialize_beam_postprocess_workspaces(runner)
    assert pool.require(key) is workspace
    allocate.assert_called_once_with(
        max_batch=capacity,
        beam_width=4,
        top_k=4,
        max_decode_steps=3,
        device=runner.device,
        logits_dtype=torch.float16,
        pin_host=True,
    )
    runner._beam_context = SimpleNamespace(max_batch=capacity)
    assert workspace_support.initialize_beam_postprocess_workspaces(runner) is pool
    assert allocate.call_count == 1

    runner._beam_context.max_batch = capacity - 1
    with pytest.raises(RuntimeError, match="geometry changed"):
        workspace_support.initialize_beam_postprocess_workspaces(runner)
    assert allocate.call_count == 1
