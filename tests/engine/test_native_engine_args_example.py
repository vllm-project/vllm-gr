# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise the configuration guide's native Beam startup example."""

import json
from pathlib import Path
from typing import Any

import pytest

from vllm_gr.config import GRConfig
from vllm_gr.config.backend_plan import read_backend_plan

pytestmark = [pytest.mark.gpu_test, pytest.mark.l2]


def test_native_engine_args_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        pytest.skip("requires CUDA for native engine configuration")

    guide = Path(__file__).resolve().parents[2] / "docs/configuration/README.md"
    markdown = guide.read_text()
    scenario = markdown.split("Save this example as `scenario.yaml`", 1)[1]
    (tmp_path / "scenario.yaml").write_text(scenario.split("```yaml\n", 1)[1].split("```", 1)[0])
    example = (
        markdown.split("### Native EngineArgs", 1)[1].split("```python\n", 1)[1].split("```", 1)[0]
    )

    # Configuration creation reads model metadata and needs no weights.
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "max_position_embeddings": 40960,
                "torch_dtype": "bfloat16",
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    scope: dict[str, Any] = {"model_path": str(model)}
    exec(compile(example, str(guide), "exec"), scope)
    config = scope["vllm_config"]
    assert config.model_config.max_model_len == 4096
    assert config.scheduler_config.max_num_batched_tokens == 4096
    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.async_scheduling is True
    assert GRConfig.from_vllm_config(config).deployment.beam_width == (8, 4, 2)
    assert read_backend_plan(config) is not None
