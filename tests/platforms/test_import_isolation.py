# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Subprocess checks for backend infrastructure import isolation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
    "torch_npu",
    "vllm_ascend",
    "vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch",
    "vllm_gr.v1.worker.npu.npu_model_runner_patch",
    "vllm_gr.v1.attention.backends.beam_attn_gpu",
    "vllm_gr.v1.attention.backends.beam_attn_npu",
    "vllm_gr.platforms.ascend.config_patch",
    "vllm_gr.v1.worker.npu.startup_patch",
    "vllm_gr.v1.worker.npu.kv_layout_patch",
):
    assert module_name not in sys.modules, module_name

from vllm_gr.platforms import BUILTIN_PROBES
assert set(BUILTIN_PROBES) == {"cuda", "ascend"}
assert "torch_npu" not in sys.modules

from tests.helpers.native_platform import native_platform
import vllm.platforms
vllm.platforms.current_platform = native_platform("CUDA", "cuda")
from vllm_gr.platforms import get_current_gr_platform
assert get_current_gr_platform().backend_id == "cuda"
assert "vllm_gr.platforms.ascend.platform" not in sys.modules
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.l1  # type: ignore[misc]
def test_obsolete_installed_entry_point_cannot_replace_native_cuda(tmp_path: Path) -> None:
    # Reproduce an environment that still contains a distribution from the
    # retired plugin system. Its broken import must be irrelevant to selection.
    dist = tmp_path / "stale_gr_plugin-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: stale-gr-plugin\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text(
        "[vllm_gr.platform_backends]\ncuda = missing_plugin:replacement\n"
    )
    code = """
import os
import sys
from importlib import metadata
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, sys.argv[1])
assert any(entry.value == "missing_plugin:replacement" for entry in
           metadata.entry_points(group="vllm_gr.platform_backends"))
from tests.helpers.native_platform import native_platform
import vllm.platforms as native
native.current_platform = native_platform("CUDA","cuda")
os.environ["VLLM_GR_BACKEND"] = "old-replacement"
os.environ["VLLM_GR_DISABLE_BUILTIN_BACKENDS"] = "cuda"

installed_entry_points = metadata.entry_points
def current_group_only(*, group):
    assert group == "vllm_gr.platform_plugins", group
    return installed_entry_points(group=group)
with patch.object(metadata, "entry_points", side_effect=current_group_only):
    from vllm_gr.platforms import get_current_gr_platform

    assert type(get_current_gr_platform()).__module__ == "vllm_gr.platforms.cuda.platform"
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
# CUDA integration initializes.
blocked = (
    "vllm_gr.platforms.ascend.config_patch",
    "vllm_gr.platforms.ascend.spawn_patch",
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
from tests.helpers.cuda_platform import CudaGRPlatform
from vllm.platforms.cuda import CudaPlatform

# The real CUDA class provides concrete methods needed during vLLM startup.
platform = CudaPlatform()
assert isinstance(platform, CudaGRPlatform.native_platform_type)
if not __import__("torch").cuda.is_available():
    # No device is visible on host-only builds. Stub the capability on the
    # class: initialize_runtime installs the GR platform instance as
    # current_platform, and that instance inherits the stub through the MRO.
    CudaPlatform.get_device_capability = classmethod(
        lambda *args, **kwargs: None
    )

# vllm's fused-MoE import chain resolves CUTLASS support at module import
# through current_platform. Import it while the native platform is still
# selected: CUDA hosts provide the compiled op for the probe, and every
# other host short-circuits on is_cuda() == False. After the CUDA platform
# switch below, the cached modules need no re-execution, so a host whose
# vllm build lacks CUDA ops (an Ascend image) cannot crash in the probe.
import vllm.v1.core.sched.scheduler  # noqa: F401

vllm.platforms.current_platform = platform

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
        "vllm_gr.v1.worker.cuda.gpu_postprocess_workspace",
        "vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_patch",
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

blocked_packages = ("vllm_gr.v1.worker.cuda.v1", "vllm_gr.v1.worker.npu")

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
from vllm_gr.v1.worker import cuda, model_runner_contracts, npu, runner
from vllm_gr.v1.worker.cuda import v1, v2

for package in (cuda, v1, v2, npu):
    assert not any(
        name.startswith(package.__name__ + ".")
        and (package is not cuda or name not in (v1.__name__, v2.__name__))
        for name in sys.modules
    )
assert callable(model_runner_contracts.is_worker_runner_patched)
assert callable(runner.require_gr_runner)
for name in ("torch", "vllm", "torch_npu", "vllm_ascend",
             "vllm_gr.v1.worker.model_runner_common",
             "vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_patch"):
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


@pytest.mark.l1  # type: ignore[misc]
def test_shared_execution_imports_no_platform_implementation() -> None:
    code = """
import importlib
import os
import sys

# Exercise GR imports independently of third-party plugin autoload on the host.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["VLLM_PLUGINS"] = ""

# Ascend images install torch_npu, and this transformers release eagerly
# imports it from its NPU flash-attention integration whenever its package
# probe finds it. The shared modules legitimately reach vllm.config ->
# transformers, so hide the package from that upstream probe: the
# integration stays inert and the contract under test - the shared worker
# modules load no device backend - stays observable on every host.
import importlib.util

_orig_find_spec = importlib.util.find_spec

def _hide_torch_npu(name, *args, **kwargs):
    if name == "torch_npu":
        return None
    return _orig_find_spec(name, *args, **kwargs)

importlib.util.find_spec = _hide_torch_npu

for suffix in (
    "worker.gr_execution",
    "worker.model_runner_common",
    "worker.model_runner_patch",
    "worker.gr_startup",
    "worker.runner",
    "engine.worker_readiness",
):
    importlib.import_module("vllm_gr.v1." + suffix)

unexpected = [
    name for name in sys.modules
    if name.startswith(("vllm_gr.v1.worker.cuda.", "vllm_gr.v1.worker.npu."))
    or name == "torch_npu" or name.startswith("torch_npu.")
]
assert not unexpected, unexpected
"""
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True, timeout=120)
