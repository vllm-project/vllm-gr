# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-free regression coverage for the offline NPU graph example."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from examples.offline_inference.beam_search import offline_beam_search_v1 as example


def _cli_args(tmp_path: Path) -> list[str]:
    return [
        "example",
        "--model",
        str(tmp_path),
        "--catalog",
        str(tmp_path / "catalog.json"),
        "--prompt",
        str(tmp_path / "prompt.txt"),
        "--constraint-backend",
        "npu",
        "--beam-width",
        "8",
    ]


@pytest.mark.parametrize(  # type: ignore[misc]
    "options,message",
    [
        (["--dynamic-beam-widths", "8,4"], "one width per generated step"),
        (["--dynamic-beam-widths", "8,4,0"], "each in [1, beam-width]"),
        (["--dynamic-beam-widths", "8,4,9"], "each in [1, beam-width]"),
        (["--dynamic-beam-widths", "8,no,2"], "comma-separated integers"),
        (["--dynamic-beam-widths", "8,4,2", "--result-width", "3"], "last-stage-width"),
        (["--result-width", "0"], "last-stage-width"),
        (["--constraint-backend", "cuda", "--dynamic-beam-widths", "8,4,2"], "NPU backend"),
        (["--constraint-backend", "cuda", "--result-width", "2"], "NPU backend"),
    ],
)
def test_dynamic_beam_cli_rejects_invalid_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    options: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path) + options)
    with pytest.raises(SystemExit) as exc:
        example.parse_args()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("backend", ["cuda", "npu"])  # type: ignore[misc]
@pytest.mark.parametrize("steps", [1, 2])  # type: ignore[misc]
@pytest.mark.parametrize("mode", [[], ["--cot"]])  # type: ignore[misc]
def test_constrained_example_rejects_incomplete_sid_at_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    backend: str,
    steps: int,
    mode: list[str],
) -> None:
    options = ["--constraint-backend", backend, "--generated-steps", str(steps)]
    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path) + options + mode)
    with pytest.raises(SystemExit) as exc:
        example.parse_args()
    assert exc.value.code == 2
    assert "requires --generated-steps 3" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path) + options + ["--enable-thinking"])
    assert example.parse_args().generated_steps == steps


@pytest.mark.parametrize(  # type: ignore[misc]
    "widths,graph_widths", [([8, 4, 2], [4, 8]), ([2, 4, 8], [2, 4])]
)
def test_dynamic_beam_example_wires_graphs_and_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    widths: list[int],
    graph_widths: list[int],
) -> None:
    output_path = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        sys,
        "argv",
        _cli_args(tmp_path)
        + [
            "--dynamic-beam-widths",
            ",".join(str(width) for width in widths),
            "--result-width",
            "2",
            "--execution",
            "graph",
            "--batch-size",
            "2",
            "--repeat",
            "1",
            "--warmup-runs",
            "0",
            "--output-json",
            str(output_path),
        ],
    )
    args = example.parse_args()
    engine_kwargs: dict[str, Any] = {}
    params_kwargs: dict[str, Any] = {}
    sequence = SimpleNamespace(
        tokens=[1, 2, 3, 4, 5], cum_logprob=-1.0, finish_reason="length", stop_reason=None
    )
    outputs = [SimpleNamespace(sequences=[sequence, sequence]) for _ in range(2)]
    stats = {"captured_shapes": [[2, width] for width in graph_widths], "full_dispatches": 2}

    def beam_search(prompts: list[Any], params: Any, *, output_mode: str) -> list[Any]:
        assert output_mode == "full"
        assert len(prompts) == 2
        assert params.dynamic_beam_widths == widths and params.result_width == 2
        return outputs

    llm = SimpleNamespace(
        get_tokenizer=lambda: SimpleNamespace(convert_ids_to_tokens=lambda ids: ids),
        beam_search_v1=beam_search,
        collective_rpc=lambda name: [stats],
    )

    def make_llm(**kwargs: Any) -> Any:
        engine_kwargs.update(kwargs)
        return nullcontext(llm)

    def make_params(**kwargs: Any) -> Any:
        params_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    modules: dict[str, dict[str, Any]] = {
        "vllm.inputs": {"tokens_input": lambda ids, **kw: dict(prompt_token_ids=ids, **kw)},
        "vllm_gr.entrypoints.gr": {"GRLLM": make_llm},
        "vllm_gr.sampling_params": {"BeamSearchParams": make_params},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(example, "build_artifact", lambda *args: ("artifact", "tokenizer"))
    monkeypatch.setattr(example, "make_prompt", lambda *args: [11, 12])

    example.run(args, tmp_path / "unused.safetensors")

    assert params_kwargs["beam_width"] == 8
    assert params_kwargs["dynamic_beam_widths"] == widths
    assert params_kwargs["result_width"] == 2
    beam_config = engine_kwargs["vllm_gr_config"]["beam"]
    assert beam_config["max_width"] == 8
    assert beam_config["graph_widths"] == graph_widths
    assert beam_config["graph_batch_size"] == 2
    report = json.loads(output_path.read_text())
    assert report["dynamic_beam_widths"] == widths and report["result_width"] == 2
    assert report["runs"][0]["returned_beams"] == 4
    assert report["npu_beam_graph"] == stats


@pytest.mark.parametrize(  # type: ignore[misc]
    "steps,captured,dispatches,fails",
    [
        (1, True, 0, False),
        (1, False, 0, False),
        (2, True, 1, False),
        (3, True, 2, False),
        (2, True, 0, True),
        (2, False, 1, True),
    ],
)
def test_npu_graph_example_requires_replay_only_for_decode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    steps: int,
    captured: bool,
    dispatches: int,
    fails: bool,
) -> None:
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "example",
            "--model",
            str(tmp_path),
            "--catalog",
            str(tmp_path / "catalog.json"),
            "--prompt",
            str(tmp_path / "prompt.txt"),
            "--constraint-backend",
            "npu",
            "--execution",
            "graph",
            "--batch-size",
            "2",
        ],
    )
    args = example.parse_args()
    # Test replay accounting with synthetic outputs independently of SID CLI admission.
    args.generated_steps = steps
    sequence = SimpleNamespace(
        tokens=[1, 2, 3, 4, 5], cum_logprob=-1.0, finish_reason="length", stop_reason=None
    )
    outputs = [SimpleNamespace(sequences=[sequence]) for _ in range(args.batch_size)]
    stats = {"captured_shapes": [[2, 128]] if captured else [], "full_dispatches": dispatches}
    llm = SimpleNamespace(
        get_tokenizer=lambda: SimpleNamespace(convert_ids_to_tokens=lambda ids: ids),
        beam_search_v1=lambda prompts, params, *, output_mode: outputs,
        collective_rpc=lambda name: [stats],
    )
    modules: dict[str, dict[str, Any]] = {
        "vllm.inputs": {"tokens_input": lambda ids, **kw: dict(prompt_token_ids=ids, **kw)},
        "vllm_gr.entrypoints.gr": {"GRLLM": lambda **kw: nullcontext(llm)},
        "vllm_gr.sampling_params": {"BeamSearchParams": lambda **kw: SimpleNamespace(**kw)},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(example, "build_artifact", lambda *args: ("artifact", "tokenizer"))
    monkeypatch.setattr(example, "make_prompt", lambda *args: [11, 12])
    if fails:
        with pytest.raises(RuntimeError, match="capture/replay was not observed"):
            example.run(args, tmp_path / "unused.safetensors")
    else:
        example.run(args, tmp_path / "unused.safetensors")
        output = capsys.readouterr().out
        assert '"npu_beam_graph"' in output
        assert "request=0 returned_beams=1" in output and "request=1 returned_beams=1" in output


pytestmark = pytest.mark.cpu_test
