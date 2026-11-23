# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Subprocess checks for backend infrastructure import isolation."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.helpers.paths import REPO_ROOT


@pytest.mark.l1  # type: ignore[misc]
def test_package_import_defers_runtime_checks_and_runs_opencv_before_grllm() -> None:
    code = """
import importlib.metadata
import sys
from types import ModuleType
from unittest.mock import patch


def unexpected_version(distribution):
    raise AssertionError(f"unexpected distribution lookup for {distribution}")


with patch.object(importlib.metadata, "version", unexpected_version):
    import vllm_gr

assert "vllm" not in sys.modules
assert "torch" not in sys.modules
assert "GRLLM" in dir(vllm_gr)
assert "GRLLM" not in vars(vllm_gr)

from vllm_gr import _env_check

with patch.object(_env_check, "check_opencv_env", side_effect=RuntimeError("bad ABI")):
    try:
        vllm_gr.GRLLM
    except RuntimeError as exc:
        assert str(exc) == "bad ABI"
    else:
        raise AssertionError("preflight failure should stop the lazy import")
assert "vllm_gr.entrypoints.llm" not in sys.modules
assert "GRLLM" not in vars(vllm_gr)

frontend = ModuleType("vllm_gr.entrypoints.llm")
frontend.GRLLM = object()
sys.modules[frontend.__name__] = frontend
with patch.object(_env_check, "check_opencv_env") as preflight:
    assert vllm_gr.GRLLM is frontend.GRLLM
    assert vllm_gr.GRLLM is frontend.GRLLM
    preflight.assert_called_once_with()
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )


def test_platform_package_and_builtins_do_not_load_hardware_modules() -> None:
    project_root = REPO_ROOT
    code = """
import sys

import vllm_gr.platforms

assert "vllm_gr.config.parser" not in sys.modules
assert "vllm_gr.config.models" not in sys.modules

for module_name in (
    "torch",
    "torch_npu",
    "vllm",
    "vllm_ascend",
    "vllm_gr.v1.worker.gpu.v1.model_runner_patch",
    "vllm_gr.v1.worker.npu.npu_model_runner_patch",
    "vllm_gr.v1.attention.backends.beam_attn_gpu",
    "vllm_gr.v1.attention.backends.beam_attn_npu",
    "vllm_gr.v1.ascend_compat",
    "vllm_gr.platforms.npu.config_patch",
    "vllm_gr.v1.worker.npu.startup_patch",
    "vllm_gr.v1.worker.npu.kv_layout_patch",
):
    assert module_name not in sys.modules, module_name

from vllm_gr.platforms.loader import load_backend_plugins

load_backend_plugins()

for module_name in (
    "torch",
    "torch_npu",
    "vllm",
    "vllm_ascend",
    "vllm_gr.v1.worker.gpu.v1.model_runner_patch",
    "vllm_gr.v1.worker.npu.npu_model_runner_patch",
    "vllm_gr.v1.attention.backends.beam_attn_gpu",
    "vllm_gr.v1.attention.backends.beam_attn_npu",
    "vllm_gr.v1.ascend_compat",
    "vllm_gr.platforms.npu.config_patch",
    "vllm_gr.v1.worker.npu.startup_patch",
    "vllm_gr.v1.worker.npu.kv_layout_patch",
):
    assert module_name not in sys.modules, module_name
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.l1  # type: ignore[misc]
def test_cuda_plugin_initialization_does_not_import_ascend() -> None:
    project_root = REPO_ROOT
    code = """
import sys

# On Ascend images, importing torch autoloads the torch_npu device backend
# (torch >= 2.10 entry-point auto-loading) and importing vllm activates the
# vendor ascend platform plugin during platform detection. Those vendor
# imports are properties of the machine image, not of vllm-gr's backend
# selection, so they cannot be forbidden here. The enforceable isolation
# contract is that vllm-gr's own Ascend integration stays unloaded while the
# CUDA provider initializes.
blocked = (
    "vllm_gr.v1.ascend_compat",
    "vllm_gr.platforms.npu.config_patch",
    "vllm_gr.platforms.npu.spawn_patch",
    "vllm_gr.v1.worker.npu.startup_patch",
    "vllm_gr.v1.worker.npu.kv_layout_patch",
)

def reject_ascend(event, args):
    # Metadata probes (find_spec) are allowed; importing the GR Ascend
    # integration is not.
    if event == "import" and args[0] in blocked:
        raise AssertionError(f"unselected backend imported: {args[0]}")

sys.addaudithook(reject_ascend)
import vllm.platforms
from vllm.platforms.interface import Platform

# Synthetic device identity: exercise host initialization, not CUDA execution.
# The identity carries the upstream enum and class lineage, so provider
# selection never has to infer CUDA from tensor placement alone.
platform = Platform()
platform._enum = "CUDA"
platform.device_type = "cuda"
platform.device_name = "cuda"
vllm.platforms.current_platform = platform
from vllm_gr.platforms import loader
loader.entry_points = lambda **kwargs: ()

from vllm_gr.plugin import initialize_runtime
first = initialize_runtime()
assert first["patch_count"] > 0, first
assert set(first["patches"].values()) == {"applied"}, first
second = initialize_runtime()
assert second["patch_count"] == 0, second
assert set(second["patches"].values()) == {"already_applied"}, second
assert not any(name in sys.modules for name in blocked)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "module_name",
    [
        "vllm_gr.v1.worker.gpu.postprocess_workspace",
        "vllm_gr.v1.worker.gpu.v2.model_runner_v2_patch",
        "vllm_gr.v1.worker.model_runner_patch",
    ],
)
def test_shared_workspace_and_v2_do_not_import_v1_integration(module_name: str) -> None:
    project_root = REPO_ROOT
    code = """
import importlib
import importlib.abc
import os
import sys
from unittest.mock import Mock

blocked_packages = ("vllm_gr.v1.worker.gpu.v1", "vllm_gr.v1.worker.npu")

def forbidden(name):
    return name == "vllm_gr.v1.worker.model_runner_common" or any(
        name == package or name.startswith(package + ".") for package in blocked_packages
    )

class RejectInactiveIntegration(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise AssertionError(f"unexpected integration dependency: {fullname}")

sys.meta_path.insert(0, RejectInactiveIntegration())
module_name = sys.argv[1]
if module_name == "vllm_gr.v1.worker.model_runner_patch":
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import vllm.platforms
    from vllm.platforms.interface import Platform

    platform = Platform()
    platform._enum = "CUDA"
    platform.device_type = "cuda"
    platform.device_name = "cuda"
    vllm.platforms.current_platform = platform

    from vllm_gr.platforms import loader
    from vllm_gr.v1.worker.gpu import runner_installer as runner
    from vllm_gr.v1.worker import model_runner_patch
    class Worker:
        vllm_config = None
        use_v2_model_runner = True
        def init_device(self):
            pass

    runner.discover_worker_runners = lambda: (("CUDA Worker.init_device", Worker),)
    from vllm_gr.v1.worker.gpu.v2 import model_runner_v2_patch as v2
    native_class = type("V2", (), {})
    v2.discover_worker_runners = lambda: (("V2", native_class),)
    v2.is_patched = lambda cls: bool(getattr(cls, "_installed", False))
    install_v2 = Mock(side_effect=lambda: setattr(native_class, "_installed", True))
    v2._apply_v2_runner_patches = install_v2
    loader.entry_points = lambda **kwargs: ()
    runner.install_worker_hooks = Mock()
    model_runner_patch.apply_worker_patches()
    install_v2.assert_not_called()
    Worker().init_device()
    install_v2.assert_called_once_with()
else:
    importlib.import_module(module_name)
assert not any(forbidden(name) for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code, module_name],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
def test_runner_contracts_import_stays_hardware_free() -> None:
    code = """
import sys
from vllm_gr.v1.worker import gpu, model_runner_contracts, npu
from vllm_gr.v1.worker.gpu import v1, v2

for package in (gpu, v1, v2, npu):
    assert not any(
        name.startswith(package.__name__ + ".")
        and (package is not gpu or name not in (v1.__name__, v2.__name__))
        for name in sys.modules
    )
assert callable(model_runner_contracts.is_worker_runner_patched)
for name in ("torch", "vllm", "torch_npu", "vllm_ascend",
             "vllm_gr.v1.worker.model_runner_common",
             "vllm_gr.v1.worker.gpu.v2.model_runner_v2_patch"):
    assert name not in sys.modules, name
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
