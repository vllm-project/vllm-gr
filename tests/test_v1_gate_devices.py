# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device routing for the Beam V1 decode benchmark gates."""

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from benchmarks.beam_search import v1_performance_gate, v1_recall_gate


def test_recall_uses_npu_memory_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_cuda_probe(_device: int) -> tuple[int, int]:
        raise AssertionError("CUDA memory must not be queried on NPU")

    npu = SimpleNamespace(
        mem_get_info=lambda _device: (60 * 2**30, 64 * 2**30),
        get_device_name=lambda _device: "Ascend",
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(mem_get_info=unexpected_cuda_probe),
            npu=npu,
        ),
    )

    assert v1_recall_gate.resolve_memory_utilization(0.5, "npu") == 0.5


@pytest.mark.parametrize("device,backend", [("gpu", "cuda"), ("npu", "npu")])  # type: ignore[misc]
def test_performance_prepare_selects_constraint_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, backend: str
) -> None:
    module = ModuleType("tools.build_constraint_table")
    setattr(module, "build_constraint_table", lambda **_kwargs: ("artifact", "tokenizer", None))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    triples = tmp_path / "triples.json"
    prompt = tmp_path / "prompt.txt"
    triples.write_text("{}", encoding="utf-8")
    prompt.write_text("prompt", encoding="utf-8")
    output_dir = tmp_path / "results"
    args = Namespace(
        device=device,
        output_dir=output_dir,
        triples=triples,
        prompt=prompt,
        model="OpenOneRec/OneRec-1.7B",
        revision="revision",
        input_len=1024,
        beam_width=128,
        generated_steps=3,
        batch_sizes=[1, 2, 4],
        warmup_runs=2,
        measured_runs=5,
        max_model_len=8192,
        gpu_memory_utilization=0.5,
        request_timeout=180,
    )

    v1_performance_gate.prepare(args)

    config = yaml.safe_load((output_dir / "vllm_gr_config.yaml").read_text(encoding="utf-8"))
    setup = json.loads((output_dir / "setup.json").read_text(encoding="utf-8"))
    assert config["constraint_table"]["backend"] == backend
    assert config["beam"].get("graph_batch_size", 1) == (4 if device == "npu" else 1)
    assert setup["device"] == device
