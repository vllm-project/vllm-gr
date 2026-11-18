# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise graph-pool isolation across real pytest test lifetimes."""

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
def test_separate_kernel_tests_do_not_reuse_retired_graph_pool(tmp_path: Path) -> None:
    # A child pytest run gives each function the real autouse fixture lifecycle.
    (tmp_path / "conftest.py").write_text(
        "from tests.kernels.conftest import isolate_cuda_graph_pool\n"
        "\n"
        "def pytest_configure(config):\n"
        "    config.addinivalue_line('markers', 'gpu_test: CUDA test')\n",
        encoding="utf-8",
    )
    (tmp_path / "test_pool.py").write_text(
        """
import gc
import weakref
import pytest
import torch
from vllm.platforms import current_platform

outputs = []
pools = []

@pytest.mark.gpu_test
def test_retire_previous_graph():
    pool = current_platform.get_global_graph_pool()
    inputs = torch.ones(4, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        output = inputs + 2
    ref = weakref.ref(graph)
    outputs.append(output)  # Keep allocations from the old pool alive.
    pools.append(pool)
    del graph
    gc.collect()
    assert ref() is None

@pytest.mark.gpu_test
def test_new_graph_uses_live_pool():
    pool = current_platform.get_global_graph_pool()
    assert pool != pools[0]
    inputs = torch.ones(4, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        output = inputs + 2
    inputs.fill_(5)
    graph.replay()
    torch.testing.assert_close(output, torch.full_like(inputs, 7))

def test_unmarked_successor_does_not_restore_stale_handle():
    assert type(current_platform)._global_graph_pool is None
""",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")])
    # The child checks only this fixture. Parent-specific selection plugins or
    # command-line options must not alter its independent test run.
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(tmp_path / "test_pool.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
