# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared CUDA workspace initialization, storage identity and load accounting."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tests.helpers.gr_config import table_constraint, worker_config
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


@pytest.mark.parametrize("enable_radix_topk", [False, True])  # type: ignore[misc]
def test_postprocess_workspace_initializer_matches_v1_capacity_and_reuses_storage(
    monkeypatch: pytest.MonkeyPatch, enable_radix_topk: bool
) -> None:
    from unittest.mock import Mock

    from vllm_gr.config import SUPPORTED_BEAM_BATCH_SIZES, GRConfig
    from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import (
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
        radix_scratch=object() if enable_radix_topk else None,
    )
    allocate = Mock(return_value=workspace)
    monkeypatch.setattr(BeamPostprocessWorkspace, "allocate", allocate)
    runner = SimpleNamespace(
        device=torch.device("cuda"),
        model_config=SimpleNamespace(dtype=torch.float16),
        vllm_config=worker_config(
            GRConfig(
                beam_worker_decision=True,
                beam_max_width=4,
                beam_max_decode_steps=3,
                enable_radix_topk=enable_radix_topk,
            )
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
        flash_logits_rows=0,
        flash_logits_vocab_size=0,
        enable_radix_topk=enable_radix_topk,
    )
    runner._beam_context = SimpleNamespace(max_batch=capacity)
    assert workspace_support.initialize_beam_postprocess_workspaces(runner) is pool
    assert allocate.call_count == 1

    runner._beam_context.max_batch = capacity - 1
    with pytest.raises(RuntimeError, match="geometry changed"):
        workspace_support.initialize_beam_postprocess_workspaces(runner)
    assert allocate.call_count == 1


def _flash_runner(*, constrained: bool = False, beam_width: int = 64, **deployment: Any) -> Any:
    from vllm_gr.config import GRConfig
    from vllm_gr.config.models import (
        ConstraintConfig,
        ConstraintMode,
        DeploymentOptions,
    )

    config = GRConfig(
        beam_worker_decision=True,
        beam_max_width=beam_width,
        beam_max_decode_steps=3,
        constraint=(
            table_constraint(max_top_k=beam_width)
            if constrained
            else ConstraintConfig(mode=ConstraintMode.NONE)
        ),
        deployment=DeploymentOptions(**{"topk_select": "flash", **deployment}),
    )
    return SimpleNamespace(
        device=torch.device("cuda"),
        model_config=SimpleNamespace(dtype=torch.bfloat16, get_vocab_size=lambda: 176256),
        vllm_config=worker_config(config),
    )


@pytest.mark.parametrize(
    ("deployment", "rows"),
    [
        ({}, 256),
        ({"topk_select": "i64"}, 0),
        ({"head_slice": 16384}, 0),
        ({"head_slice": 0}, 256),
    ],
)  # type: ignore[misc]
def test_workspace_reserves_shared_flash_logits_before_kv_cache_sizing(
    monkeypatch: pytest.MonkeyPatch, deployment: dict[str, Any], rows: int
) -> None:
    from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import (
        BeamPostprocessWorkspace,
    )

    seen: dict[str, Any] = {}

    def allocate(**kwargs: Any) -> Any:
        seen.update(kwargs)
        buffer = torch.empty((kwargs["flash_logits_rows"], 4), dtype=torch.bfloat16)
        return SimpleNamespace(
            key=workspace_support.BeamPostprocessWorkspaceKey(64, 64, torch.bfloat16),
            max_batch=kwargs["max_batch"],
            max_decode_steps=3,
            device_memory_bytes=buffer.numel() * buffer.element_size(),
            flash_logits=buffer if kwargs["flash_logits_rows"] else None,
        )

    monkeypatch.setattr(BeamPostprocessWorkspace, "allocate", allocate)
    runner = _flash_runner(**deployment)
    runner.model_memory_usage = 0
    wrapped = workspace_support.make_load_model_with_beam_postprocess_workspace(
        lambda self: setattr(self, "model_memory_usage", 1000)
    )
    wrapped(runner)

    # Eight prompts at width 64 would need 512 rows, so the buffer covers the
    # largest eligible batch (four prompts, 256 rows) and is sized from the
    # model vocabulary, not from the first request.
    assert seen["flash_logits_rows"] == rows
    assert seen["flash_logits_vocab_size"] == (176256 if rows else 0)
    assert runner.model_memory_usage == 1000 + rows * 4 * 2


@pytest.mark.parametrize(
    ("beam_width", "rows"),
    [
        # Decode of four requests (4 x 64 rows) already covers Prefill.
        (64, 256),
        # No Decode batch fits in 256 rows from width 257 on, but Prefill
        # still selects one row per request, so eight requests need 8 rows.
        (512, 8),
        (1024, 8),
    ],
)  # type: ignore[misc]
def test_workspace_reserves_flash_logits_for_multi_request_prefill(
    monkeypatch: pytest.MonkeyPatch, beam_width: int, rows: int
) -> None:
    from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import (
        BeamPostprocessWorkspace,
    )

    allocate = Mock(
        return_value=SimpleNamespace(
            key=workspace_support.BeamPostprocessWorkspaceKey(
                beam_width, beam_width, torch.bfloat16
            ),
            max_batch=8,
            max_decode_steps=3,
            device_memory_bytes=0,
        )
    )
    monkeypatch.setattr(BeamPostprocessWorkspace, "allocate", allocate)
    workspace_support.initialize_beam_postprocess_workspaces(_flash_runner(beam_width=beam_width))

    assert allocate.call_args.kwargs["flash_logits_rows"] == rows
    assert allocate.call_args.kwargs["flash_logits_vocab_size"] == 176256


def test_workspace_does_not_reserve_flash_logits_for_constrained_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import (
        BeamPostprocessWorkspace,
    )

    allocate = Mock(
        return_value=SimpleNamespace(
            key=workspace_support.BeamPostprocessWorkspaceKey(64, 64, torch.bfloat16),
            max_batch=8,
            max_decode_steps=3,
            device_memory_bytes=0,
        )
    )
    monkeypatch.setattr(BeamPostprocessWorkspace, "allocate", allocate)
    runner = _flash_runner(constrained=True)
    workspace_support.initialize_beam_postprocess_workspaces(runner)
    assert allocate.call_args.kwargs["flash_logits_rows"] == 0


@pytest.mark.l1  # type: ignore[misc]
def test_cuda_admission_fixture_supports_unconstrained_head_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from tests.kernels import beam_stage_test_utils as builders
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
        _get_unconstrained_head_layout,
    )
    from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import (
        BeamPostprocessWorkspace,
    )

    # Keep the real worker/model metadata while replacing device allocations.
    backend = Mock()
    backend.info.table_metadata.vocab_size = 32
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(builders, "_upload_cuda_backend", lambda *args, **kwargs: (backend, None))
    monkeypatch.setattr(BeamPostprocessWorkspace, "allocate", Mock())
    worker = builders.admission_worker(sliding_windows=())
    worker.model = SimpleNamespace()
    assert _get_unconstrained_head_layout(worker) is None
