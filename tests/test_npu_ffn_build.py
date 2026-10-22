# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Header-only FFN changes must invalidate the Python extension build."""

from __future__ import annotations

import os
import runpy
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import setuptools
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


@pytest.mark.parametrize("header", ["pytorch.h", "pytorch_npu_helper.hpp"])  # type: ignore[misc]
def test_header_only_change_rebuilds_ffn_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, header: str
) -> None:
    source = Path(__file__).parents[1] / "csrc/npu/fused_ffn"
    root = tmp_path / "fused_ffn"
    binding_dir = root / "ops/gr_gate_up_swi_glu/pytorch"
    binding_dir.mkdir(parents=True)
    shutil.copy2(source / "setup.py", root / "setup.py")
    for name in ("bindings.cpp", "pytorch.h", "pytorch_npu_helper.hpp"):
        shutil.copy2(source / "ops/gr_gate_up_swi_glu/pytorch" / name, binding_dir / name)

    # Execute the real setup with only hardware-specific extension factories
    # replaced. The dependency list still comes from the production setup.py.
    npu = ModuleType("torch_npu")
    npu.__file__ = str(tmp_path / "torch_npu/__init__.py")
    torch_extension = ModuleType("torch.utils.cpp_extension")
    torch_extension.BuildExtension = SimpleNamespace(  # type: ignore[attr-defined]
        with_options=lambda **kwargs: build_ext
    )
    npu_extension = ModuleType("torch_npu.utils.cpp_extension")
    npu_extension.NpuExtension = Extension  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", torch_extension)
    monkeypatch.setitem(sys.modules, "torch_npu.utils.cpp_extension", npu_extension)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "cann"))
    setup = Mock()
    monkeypatch.setattr(setuptools, "setup", setup)
    runpy.run_path(str(root / "setup.py"), run_name="__main__")
    setup.assert_called_once()
    extension = setup.call_args.kwargs["ext_modules"][0]

    # Use setuptools' real timestamp/dependency decision. A fake compiler
    # prevents hardware/toolchain requirements without replacing that logic.
    command = build_ext(Distribution({"ext_modules": [extension]}))
    command.ensure_finalized()
    command.build_lib = str(tmp_path / "lib")
    command.build_temp = str(tmp_path / "objects")
    compiler = Mock()
    compiler.compile.return_value = [str(tmp_path / "bindings.o")]
    compiler.detect_language.return_value = "c++"
    command.compiler = compiler
    built = Path(command.get_ext_fullpath(extension.name))
    built.parent.mkdir(parents=True)
    built.touch()
    timestamp = time.time() - 120
    for path in binding_dir.iterdir():
        os.utime(path, (timestamp, timestamp))
    os.utime(built, (timestamp + 10, timestamp + 10))
    command.build_extension(extension)
    compiler.compile.assert_not_called()
    compiler.link_shared_object.assert_not_called()

    changed = binding_dir / header
    changed.write_bytes(changed.read_bytes() + b"\n// header-only build regression\n")
    os.utime(changed, (timestamp + 20, timestamp + 20))
    command.build_extension(extension)
    compiler.compile.assert_called_once()
    compiler.link_shared_object.assert_called_once()
    assert str(changed.resolve()) in compiler.compile.call_args.kwargs["depends"]
