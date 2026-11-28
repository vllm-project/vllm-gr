# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise the real GR Worker subclass against a synthetic native NPU Worker."""

import subprocess
import sys

import pytest

from tests.helpers.paths import REPO_ROOT

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_ascend_subclass_preserves_initialization_order_and_native_class() -> None:
    code = """
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

events = []
class NativeWorker:
    def __init__(self):
        events.append("native_init")
        self.vllm_config = object()
        self.rank = 1
        self.use_v2_model_runner = False
    def _init_device(self):
        events.append("native_device")
        return "cpu"
    def init_device(self):
        raise AssertionError("the native hard-coded Runner must not be constructed")
native = ModuleType("vllm_ascend.worker.worker")
native.NPUWorker = NativeWorker
sys.modules[native.__name__] = native
native_method = NativeWorker.init_device
fake_runner = ModuleType("vllm_gr.v1.worker.npu.model_runner")
def make_runner(config, device):
    events.append("gr_runner")
    assert device == "cpu"
    return SimpleNamespace(vllm_config=config)
fake_runner.AscendGRModelRunner = make_runner
sys.modules[fake_runner.__name__] = fake_runner
worker_module = importlib.import_module("vllm_gr.v1.worker.npu.worker")
with patch("vllm_gr.platforms.ascend.config_patch.install_ascend_config_filter",
           side_effect=lambda: events.append("filter")), \
     patch("vllm_gr.v1.worker.npu.kv_layout_patch.apply_npu_contiguous_kv_cache_patch",
           side_effect=lambda: events.append("kv_layout")), \
     patch("vllm_gr.v1.worker.npu.startup_patch.apply_penalties_warmup_oom_guard",
           side_effect=lambda: events.append("penalties")), \
     patch("vllm_gr.v1.worker.npu.stage_projection.install_npu_stage_weight_update_hooks",
           side_effect=lambda cls: events.append("weight_updates")), \
     patch.object(worker_module, "init_workspace_manager",
                  side_effect=lambda device, count: events.append("workspace")):
    worker = worker_module.AscendGRWorker()
    worker.init_device()
    assert events == ["filter", "native_init", "kv_layout", "penalties", "weight_updates",
                      "native_device", "workspace", "gr_runner"], events
    assert worker.model_runner.vllm_config is worker.vllm_config
    assert NativeWorker.init_device is native_method
    events.clear()
    worker.use_v2_model_runner = True
    try:
        worker.init_device()
    except ValueError as exc:
        assert "requires ModelRunner V1" in str(exc)
    else:
        raise AssertionError("unsupported NPU V2 was accepted")
    assert not events
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
