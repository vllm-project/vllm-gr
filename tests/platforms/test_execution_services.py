# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Backend execution services answer selection that used to read a device name."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.platforms import ascend_services, cuda_services, legacy
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.npu import NPU_BACKEND
from vllm_gr.platforms.services import StageExecutionService

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_builtin_providers_expose_a_stage_execution_service() -> None:
    """Both shipped backends answer stage selection through the same contract."""
    for provider in (CUDA_BACKEND, NPU_BACKEND):
        service = provider.create_stage_execution_service()
        assert isinstance(service, StageExecutionService), provider.backend_id
        assert service.backend_id == provider.backend_id


def test_each_call_returns_a_service_for_the_same_backend() -> None:
    """The service is bound to its provider, not to process-global state."""
    assert CUDA_BACKEND.create_stage_execution_service().backend_id == "cuda"
    assert NPU_BACKEND.create_stage_execution_service().backend_id == "ascend"


# --------------------------------------------------------------------------- #
# CUDA service
# --------------------------------------------------------------------------- #


def _cuda_execute(monkeypatch: pytest.MonkeyPatch, **call_kwargs: Any) -> tuple[Any, Any, Mock]:
    from vllm_gr.v1.worker import gpu_beam_stage_runner

    runner = SimpleNamespace(execute=Mock(return_value="output"), execution_error=None)
    build = Mock(return_value=runner)
    monkeypatch.setattr(gpu_beam_stage_runner, "get_stage_runner", build)
    service = cuda_services.CudaStageExecutionService()
    result = service.execute(SimpleNamespace(), "scheduled", "original", **call_kwargs)
    return result, runner, build


def test_cuda_execution_without_hooks_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.worker import gpu_beam_stage_runner

    build = Mock()
    monkeypatch.setattr(gpu_beam_stage_runner, "get_stage_runner", build)
    service = cuda_services.CudaStageExecutionService()

    with pytest.raises(ValueError, match="requires explicit stage hooks"):
        service.execute(SimpleNamespace(), "scheduled", "original")
    build.assert_not_called()


def test_cuda_execution_binds_the_hooks_and_returns_the_stage_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hooks = object()
    result, runner, build = _cuda_execute(monkeypatch, hooks_type=hooks)

    assert result == "output"
    assert build.call_args.kwargs["hooks_type"] is hooks
    runner.execute.assert_called_once()


def test_cuda_execution_failure_is_recorded_for_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale logits must never hide a failed execute from the sample Future."""
    from vllm_gr.v1.worker import gpu_beam_stage_runner

    error = RuntimeError("native execution failed")
    runner = SimpleNamespace(execute=Mock(side_effect=error), execution_error=None)
    monkeypatch.setattr(gpu_beam_stage_runner, "get_stage_runner", Mock(return_value=runner))

    with pytest.raises(RuntimeError) as raised:
        cuda_services.CudaStageExecutionService().execute(
            SimpleNamespace(), "scheduled", "original", hooks_type=object()
        )

    assert raised.value is error
    assert runner.execution_error is error


def test_cuda_retirement_delegates_to_the_shared_cuda_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker import gpu_beam_stage_runner

    retire = Mock(return_value="retired")
    monkeypatch.setattr(gpu_beam_stage_runner, "execute_gr_retirement", retire)

    result = cuda_services.CudaStageExecutionService().retire("worker", "scheduled")

    assert result == "retired"
    retire.assert_called_once_with("worker", "scheduled")


# --------------------------------------------------------------------------- #
# Ascend service
# --------------------------------------------------------------------------- #


def test_ascend_execution_ignores_the_cuda_hooks_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ascend owns its own stage state, so hooks are accepted and not required."""
    from vllm_gr.v1.worker import npu_beam_stage_runner

    runner = SimpleNamespace(execute=Mock(return_value="output"), execution_error=None)
    build = Mock(return_value=runner)
    monkeypatch.setattr(npu_beam_stage_runner, "get_npu_stage_runner", build)

    service = ascend_services.AscendStageExecutionService()
    result = service.execute(SimpleNamespace(), "scheduled", "original")

    assert result == "output"
    build.assert_called_once()


def test_ascend_execution_failure_is_recorded_for_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker import npu_beam_stage_runner

    error = RuntimeError("npu execution failed")
    runner = SimpleNamespace(execute=Mock(side_effect=error), execution_error=None)
    monkeypatch.setattr(npu_beam_stage_runner, "get_npu_stage_runner", Mock(return_value=runner))

    with pytest.raises(RuntimeError) as raised:
        ascend_services.AscendStageExecutionService().execute(
            SimpleNamespace(), "scheduled", "original"
        )

    assert raised.value is error
    assert runner.execution_error is error


def test_ascend_retirement_delegates_to_the_eager_owner_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker import npu_beam_stage_runner

    retire = Mock(return_value="retired")
    monkeypatch.setattr(npu_beam_stage_runner, "execute_npu_gr_retirement", retire)

    result = ascend_services.AscendStageExecutionService().retire("worker", "scheduled")

    assert result == "retired"
    retire.assert_called_once_with("worker", "scheduled")


def test_ascend_unconstrained_admission_applies_the_device_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ascend keeps its own admission rules instead of a shared device test."""
    from vllm_gr.v1.engine import beam_search_v1

    validate = Mock()
    monkeypatch.setattr(beam_search_v1, "validate_gr_device_config", validate)
    worker = SimpleNamespace(vllm_config="config")

    ascend_services.AscendStageExecutionService().admit_unconstrained_beam(worker)

    validate.assert_called_once_with("config")


def test_cuda_unconstrained_admission_accepts_every_admitted_configuration() -> None:
    """CUDA serves the unconstrained configurations this release admits."""
    cuda_services.CudaStageExecutionService().admit_unconstrained_beam(
        SimpleNamespace(vllm_config="config")
    )


# --------------------------------------------------------------------------- #
# Runner patch module selection
# --------------------------------------------------------------------------- #


def test_cuda_selects_its_v2_patch_module_only_on_request() -> None:
    assert CUDA_BACKEND.select_runner_patch_module_name() == (
        "vllm_gr.v1.worker.gpu_model_runner_patch"
    )
    assert CUDA_BACKEND.select_runner_patch_module_name(prefer_v2=True) == (
        "vllm_gr.v1.worker.gpu_model_runner_v2_patch"
    )


def test_ascend_ignores_the_v2_opt_in_instead_of_borrowing_cuda() -> None:
    """Ascend has no V2 runner; the flag must not select another backend's module."""
    assert NPU_BACKEND.select_runner_patch_module_name() == (
        "vllm_gr.v1.worker.npu_model_runner_patch"
    )
    assert NPU_BACKEND.select_runner_patch_module_name(prefer_v2=True) == (
        "vllm_gr.v1.worker.npu_model_runner_patch"
    )


# --------------------------------------------------------------------------- #
# Legacy adapter
# --------------------------------------------------------------------------- #


class _LegacyProvider:
    device_type = "cuda"

    def check_environment(self) -> dict[str, str]:
        return {}

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return 0

    def get_model_runner_hooks(self) -> Any:
        return object()

    def get_attention_execution(self) -> Any:
        return object()

    def create_constrained_topk_execution(self, *, device: Any = None) -> None:
        del device
        return None


@pytest.mark.parametrize(  # type: ignore[misc]
    ("device_type", "backend_id", "module_name"),
    [
        ("cuda", "cuda", "vllm_gr.v1.worker.gpu_model_runner_patch"),
        ("npu", "ascend", "vllm_gr.v1.worker.npu_model_runner_patch"),
    ],
)
def test_legacy_adapter_supplies_the_new_selection_methods(
    device_type: str, backend_id: str, module_name: str
) -> None:
    """A version-1 provider keeps working without declaring the new methods."""
    provider = _LegacyProvider()
    provider.device_type = device_type
    adapted = legacy.adapt_legacy_provider(provider, source=f"entrypoint:{device_type}")

    assert adapted is not None
    assert adapted.backend_id == backend_id
    assert adapted.select_runner_patch_module_name() == module_name
    assert adapted.create_stage_execution_service().backend_id == backend_id


# --------------------------------------------------------------------------- #
# Import isolation
# --------------------------------------------------------------------------- #


def test_building_services_loads_no_hardware_module() -> None:
    """Service construction must not defeat the provider import isolation."""
    project_root = Path(__file__).resolve().parents[2]
    code = """
import sys

from vllm_gr.platforms.builtins import register_builtin_backends
from vllm_gr.platforms.registry import get_active_backend

register_builtin_backends()
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.npu import NPU_BACKEND

for provider in (CUDA_BACKEND, NPU_BACKEND):
    provider.create_stage_execution_service()
    provider.select_runner_patch_module_name()
    provider.select_runner_patch_module_name(prefer_v2=True)

for module_name in (
    "torch",
    "torch_npu",
    "vllm",
    "vllm_ascend",
    "vllm_gr.v1.worker.gpu_beam_stage_runner",
    "vllm_gr.v1.worker.npu_beam_stage_runner",
    "vllm_gr.v1.attention.backends.beam_attn_gpu",
    "vllm_gr.v1.attention.backends.beam_attn_npu",
):
    assert module_name not in sys.modules, module_name
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
