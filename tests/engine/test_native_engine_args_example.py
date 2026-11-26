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


@pytest.mark.parametrize("packaged_profile", [False, True])  # type: ignore[misc]
def test_native_engine_args_example(
    model_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packaged_profile: bool
) -> None:
    from vllm_gr.config.parser import COMMON_ENGINE_CONFIG
    from vllm_gr.entrypoints.beam_search.input_processor import _create_request_params
    from vllm_gr.sampling_params import BeamSearchParams

    guide = Path(__file__).resolve().parents[2] / "docs/configuration/README.md"
    markdown = guide.read_text()
    scenario = markdown.split("Save this example as `scenario.yaml`", 1)[1]
    yaml = scenario.split("```yaml\n", 1)[1].split("```", 1)[0]
    (tmp_path / "scenario.yaml").write_text(
        COMMON_ENGINE_CONFIG.read_text() if packaged_profile else yaml
    )
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
    expected_widths = (128, 128, 128) if packaged_profile else (8, 4, 2)
    assert GRConfig.from_vllm_config(config).deployment.beam_width == expected_widths
    assert read_backend_plan(config) is not None
    candidate_width = max(expected_widths)
    assert config.model_config.max_logprobs == candidate_width
    request = BeamSearchParams(beam_width=candidate_width, max_tokens=3, ignore_eos=True)
    sampling, _, _ = _create_request_params(request, None, vllm_config=config)
    assert sampling.logprobs == candidate_width


@pytest.mark.parametrize("options", [{}, {"tile": 4}])  # type: ignore[misc]
def test_backend_options_require_explicit_gr_config(
    model_path: str, options: dict[str, int]
) -> None:
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(model=model_path, additional_config={"vllm_gr_backend_options": options})
    with pytest.raises(
        ValueError, match=r'vllm_gr_backend_options.*additional_config\["vllm_gr"\].*max_logprobs'
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


def test_grllm_passes_native_plugin_config_to_engine(
    model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    from vllm_gr.entrypoints.llm import GRLLM

    class PluginConfig:
        def compute_hash(self) -> str:
            return "native-plugin"

    class ConfigCreated(Exception):
        pass

    plugin_config = PluginConfig()

    def capture_config(*, engine_args: EngineArgs, usage_context: Any) -> None:
        assert engine_args.additional_config is plugin_config
        config = engine_args.create_engine_config(usage_context=usage_context)
        assert config.additional_config is plugin_config
        assert read_backend_plan(config) is None
        raise ConfigCreated

    monkeypatch.setattr(LLMEngine, "from_engine_args", staticmethod(capture_config))

    with pytest.raises(ConfigCreated):
        GRLLM(model=model_path, additional_config=plugin_config)

    with pytest.raises(ValueError, match="GR startup requires mapping additional_config"):
        GRLLM(model=model_path, gr={}, additional_config=plugin_config)
