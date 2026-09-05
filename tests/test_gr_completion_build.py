# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-target packaging and actual host-only native completion compilation."""

import ctypes
import importlib.util
from pathlib import Path
import runpy
import subprocess
from unittest.mock import Mock

import pytest
import setuptools_scm
from setuptools import Distribution
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vllm_gr/ops/cuda/native/gr_completion_ticket.cc"


def _load_setup(monkeypatch):
    # Real build entry points import SCM submodules, so preserve its package
    # identity and stub only the version-file writer used by setup.py.
    monkeypatch.setattr(setuptools_scm, "get_version", Mock(return_value="0.22.1"))
    spec = importlib.util.spec_from_file_location("gr_completion_setup_test", ROOT / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("device", ["cuda", "cpu", "npu"])
def test_native_extension_platform_routing(monkeypatch, device):
    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", device)
    setup = _load_setup(monkeypatch)
    extensions = setup.get_ext_modules()
    if device != "cuda":
        assert extensions == []
        return
    assert len(extensions) == 1
    extension = extensions[0]
    assert extension.name == "vllm_gr.ops.cuda._completion"
    assert extension.sources == [str(SOURCE.relative_to(ROOT))]
    assert extension.language == "c++"
    assert extension.extra_compile_args == ["-std=c++17", "-pthread"]
    assert extension.extra_link_args == ["-pthread"]
    assert not extension.libraries and not extension.include_dirs


@pytest.mark.parametrize("device", ["cuda", "cpu", "npu"])
def test_setup_main_passes_platform_extensions(monkeypatch, device):
    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", device)
    _load_setup(monkeypatch)  # Stubs only SCM writing; no _version.py mutation.
    import setuptools

    setup_call = Mock()
    monkeypatch.setattr(setuptools, "setup", setup_call)
    runpy.run_path(str(ROOT / "setup.py"), run_name="__main__")
    setup_call.assert_called_once()
    assert len(setup_call.call_args.kwargs["ext_modules"]) == int(device == "cuda")


@pytest.fixture(scope="module")
def built_completion(tmp_path_factory):
    # Unique temporary output: never build --inplace or replace an installed
    # extension. No torch import/device is needed even for the CUDA target.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_GR_TARGET_DEVICE", "cuda")
        patch.chdir(ROOT)
        with pytest.MonkeyPatch.context() as setup_patch:
            setup = _load_setup(setup_patch)
        # Distribution loads real setuptools entry points, including SCM's
        # package submodules. The setup-import-only stub must not escape here.
        directory = tmp_path_factory.mktemp("gr_completion_build")
        distribution = Distribution({"name": "gr-completion-build-test",
                                     "ext_modules": setup.get_ext_modules()})
        distribution.get_command_obj("build").build_base = str(directory / "build")
        command = build_ext(distribution)
        command.build_lib = str(directory / "lib")
        command.build_temp = str(directory / "objects")
        command.inplace = False
        command.parallel = 1
        command.ensure_finalized()
        command.run()
        artifact = Path(command.get_ext_fullpath("vllm_gr.ops.cuda._completion"))
        assert artifact.is_file() and artifact.is_relative_to(directory)
        print(f"native completion build artifact: {artifact}")
        return artifact


def test_actual_build_imports_python_module_and_versioned_c_abi(built_completion):
    spec = importlib.util.spec_from_file_location("_completion", built_completion)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.API_VERSION == 2
    library = ctypes.CDLL(str(built_completion))
    library.gr_ticket_api_version.argtypes = []
    library.gr_ticket_api_version.restype = ctypes.c_int
    assert library.gr_ticket_api_version() == module.API_VERSION

    library.gr_ticket_live.argtypes = []
    library.gr_ticket_live.restype = ctypes.c_int
    library.gr_ticket_create.argtypes = [ctypes.c_uint64]
    library.gr_ticket_create.restype = ctypes.c_void_p
    library.gr_ticket_release.argtypes = [ctypes.c_void_p]
    library.gr_ticket_release.restype = None
    library.gr_ticket_register.argtypes = [ctypes.c_void_p] * 3
    library.gr_ticket_register.restype = ctypes.c_int
    library.gr_ticket_wait.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.gr_ticket_wait.restype = ctypes.c_int
    library.gr_ticket_snapshot.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
                                          ctypes.POINTER(ctypes.c_int)]
    library.gr_ticket_snapshot.restype = ctypes.c_int
    assert library.gr_ticket_live() == 0
    ticket = library.gr_ticket_create(73)
    assert ticket
    try:
        assert library.gr_ticket_register(ticket, None, None) == 801
        assert library.gr_ticket_wait(ticket, 0, 1) == 3
        generation, status = ctypes.c_uint64(), ctypes.c_int()
        flags = library.gr_ticket_snapshot(ticket, ctypes.byref(generation), ctypes.byref(status))
        assert (generation.value, status.value, flags) == (73, 801, 4)
    finally:
        library.gr_ticket_release(ticket)
    assert library.gr_ticket_live() == 0


def test_standalone_test_main_uses_production_primitive(tmp_path):
    # Preserve the original no-Python-headers sanitizer/diagnostic build route.
    source = ROOT / "tests/profiling/native/gr_completion_ticket.cc"
    executable = tmp_path / "gr_completion_ticket_test"
    subprocess.run(["c++", "-std=c++17", "-pthread", "-DGR_COMPLETION_TEST_MAIN",
                    str(source), "-o", str(executable)], check=True, timeout=60,
                   capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, timeout=60,
                            capture_output=True, text=True)
    assert "1000 waiter/owner/callback races; live=0" in result.stdout
