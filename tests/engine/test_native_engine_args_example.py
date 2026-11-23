# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise the configuration guide's native Beam startup example."""

import json
from pathlib import Path
from typing import Any

import pytest

from vllm_gr.config import GRConfig, gr_config_to_mapping, load_gr_config
from vllm_gr.config.backend_plan import read_backend_plan

pytestmark = [pytest.mark.gpu_test, pytest.mark.l2]


@pytest.fixture  # type: ignore[misc]
def model_path(tmp_path: Path) -> str:
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        pytest.skip("requires CUDA for native engine configuration")

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
    return str(model)


def test_native_engine_args_example(
    model_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guide = Path(__file__).resolve().parents[2] / "docs/configuration/README.md"
    markdown = guide.read_text()
    scenario = markdown.split("Save this example as `scenario.yaml`", 1)[1]
    (tmp_path / "scenario.yaml").write_text(scenario.split("```yaml\n", 1)[1].split("```", 1)[0])
    example = (
        markdown.split("### Native EngineArgs", 1)[1].split("```python\n", 1)[1].split("```", 1)[0]
    )

    monkeypatch.chdir(tmp_path)
    scope: dict[str, Any] = {"model_path": model_path}
    exec(compile(example, str(guide), "exec"), scope)
    config = scope["vllm_config"]
    assert config.model_config.max_model_len == 4096
    assert config.scheduler_config.max_num_batched_tokens == 4096
    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.async_scheduling is True
    assert GRConfig.from_vllm_config(config).deployment.beam_width == (8, 4, 2)
    assert read_backend_plan(config) is not None


@pytest.mark.parametrize("options", [{}, {"tile": 4}])  # type: ignore[misc]
def test_backend_options_require_explicit_gr_config(
    model_path: str, options: dict[str, int]
) -> None:
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(model=model_path, additional_config={"vllm_gr_backend_options": options})
    with pytest.raises(
        ValueError, match=r'vllm_gr_backend_options.*additional_config\["vllm_gr"\]'
    ):
        args.create_engine_config()


def test_backend_options_with_explicit_gr_config(model_path: str) -> None:
    from vllm.engine.arg_utils import EngineArgs

    gr = load_gr_config(overrides={"beam_width": [4], "candidate_width": [4]})
    args = EngineArgs(
        model=model_path,
        async_scheduling=True,
        enable_chunked_prefill=False,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        additional_config={"vllm_gr_backend_options": {}, "vllm_gr": gr_config_to_mapping(gr)},
    )
    config = args.create_engine_config()
    assert GRConfig.from_vllm_config(config).beam_max_width == 4
    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.max_num_batched_tokens == 4096
    assert config.additional_config["vllm_gr_backend_options"] == {}
    plan = read_backend_plan(config)
    assert plan is not None and plan.request.options.to_dict() == {}
