# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise GPU test GC hygiene around real CUDA graph capture."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tools.hardware_marks import hardware_marks

pytestmark = [
    *hardware_marks(res={"cuda": "A100"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.l2  # type: ignore[misc]
def test_gpu_gc_isolation_and_restoration(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text(
        "from tests.kernels.conftest import isolate_cuda_graph_pool\n"
        "\n"
        "def pytest_configure(config):\n"
        "    config.addinivalue_line('markers', 'gpu_test: CUDA test')\n",
        encoding="utf-8",
    )
    (tmp_path / "test_gc.py").write_text(
        """
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from tests.kernels.conftest import isolate_cuda_graph_pool
from vllm.platforms import current_platform


@pytest.mark.gpu_test
def test_capture_keeps_unreachable_graph_until_capture_ends():
    # Arrange for automatic GC to become eligible only after the first graph
    # has become cyclic garbage inside the second graph's capture.
    thresholds = gc.get_threshold()
    gc.set_threshold(1000000, 1000000, 1000000)
    inputs = torch.ones(4, device="cuda")
    old_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(old_graph):
        old_output = inputs + 1
    old_ref = weakref.ref(old_graph)
    new_graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(new_graph):
            cycle = [old_graph]
            cycle.append(cycle)
            del old_graph, cycle
            gc.set_threshold(1, 1, 1)
            for _ in range(200):
                _ = [object() for _ in range(8)]
            assert old_ref() is not None
            output = inputs + 2
    finally:
        gc.set_threshold(*thresholds)
    inputs.fill_(5)
    new_graph.replay()
    torch.testing.assert_close(output, torch.full_like(inputs, 7))
    assert old_output.numel() == 4


def test_automatic_gc_state_restored_after_gpu_case():
    assert gc.isenabled()


def test_cpu_and_npu_requests_leave_gc_unchanged(monkeypatch):
    original = gc.isenabled()
    request = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda _: None))
    fixture = isolate_cuda_graph_pool.__wrapped__(request)
    next(fixture)
    assert gc.isenabled() is original
    fixture.close()
    assert gc.isenabled() is original
    monkeypatch.setattr(type(current_platform), "device_type", "npu")
    request = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda _: object()))
    fixture = isolate_cuda_graph_pool.__wrapped__(request)
    next(fixture)
    assert gc.isenabled() is original
    fixture.close()
    assert gc.isenabled() is original


def test_previously_disabled_gc_remains_disabled():
    gc.disable()
    try:
        request = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda _: object()))
        fixture = isolate_cuda_graph_pool.__wrapped__(request)
        next(fixture)
        assert not gc.isenabled()
        fixture.close()
        assert not gc.isenabled()
    finally:
        gc.enable()
""",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")])
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(tmp_path / "test_gc.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
