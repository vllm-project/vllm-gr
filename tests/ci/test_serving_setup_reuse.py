# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute orchestration with fake engines; real engine coverage stays on the device."""

import ast
import copy
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("fail_lifecycle", [0, 2])  # type: ignore[misc]
def test_reuse_inputs_retains_fresh_lifecycles_and_isolated_configs(
    monkeypatch: pytest.MonkeyPatch, fail_lifecycle: int
) -> None:
    source = Path(__file__).resolve().parents[1] / "test_serving_beam_search.py"
    module = ast.parse(source.read_text())
    module.body = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"beam_search_test", "_run_gr_beam_search_target"}
    ]
    builds: list[Path] = []
    configs: list[dict[str, Any]] = []
    patch = ModuleType("vllm_gr.patch")
    patch.re_apply_patches = lambda: None  # type: ignore[attr-defined]
    fixtures = ModuleType("tests.st.test_onerec_v1_scheduler_st")

    def inputs(path: Path) -> tuple[Any, ...]:
        builds.append(path)
        return ("model", None, {"beam": {"max_width": 128}}, [1, 2, 3], None, None)

    fixtures._inputs = inputs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm_gr.patch", patch)
    monkeypatch.setitem(sys.modules, "tests.st.test_onerec_v1_scheduler_st", fixtures)
    monkeypatch.setenv("VLLM_GR_TEST_GPU_MEMORY_UTILIZATION", "0.7")

    async def engine(
        loops: int, model: str, config: dict[str, Any], memory: float, prompt: list[int]
    ) -> None:
        assert loops == 2 and model == "model" and memory == 0.7 and prompt == [1, 2, 3]
        assert config["beam"]["max_width"] == 128
        assert builds[0].is_dir()  # artifacts outlive all engine lifecycles
        configs.append(config)
        config["beam"]["max_width"] = 1
        if len(configs) == fail_lifecycle:
            raise RuntimeError("engine failed")

    namespace: dict[str, Any] = {
        "Any": Any,
        "os": os,
        "copy": copy,
        "Path": Path,
        "TemporaryDirectory": TemporaryDirectory,
        "_run_serving_v1": engine,
        "BeamSearchParams": SimpleNamespace(begin_token=True),
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: None, memory_allocated=lambda: 0)
        ),
        "psutil": SimpleNamespace(
            Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0))
        ),
    }
    exec(compile(module, str(source), "exec"), namespace)  # noqa: S102 - repository-owned test code
    if fail_lifecycle:
        with pytest.raises(RuntimeError, match="engine failed"):
            namespace["_run_gr_beam_search_target"](2)
    else:
        namespace["_run_gr_beam_search_target"](2)
    assert len(builds) == 1
    assert len(configs) == (fail_lifecycle or 3)
    assert len({id(config) for config in configs}) == len(configs)
    assert not builds[0].exists()
