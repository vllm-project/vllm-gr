# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-free coverage for the offline shared-prefix cache workload."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from examples.offline_inference.beam_search import offline_beam_search_v1 as example


def _cli_args(
    tmp_path: Path,
    workload: Path | None = None,
    *,
    promptless: bool = False,
) -> list[str]:
    args = [
        "example",
        "--model",
        str(tmp_path / "model"),
        "--catalog",
        str(tmp_path / "catalog.json"),
    ]
    if not promptless:
        args.extend(("--prompt", str(tmp_path / "prompt.txt")))
    if workload is not None:
        args.extend(("--workload-jsonl", str(workload)))
    return args


def _fake_tokenizer() -> Any:
    return SimpleNamespace(
        encode=lambda text, *, add_special_tokens: [ord(char) for char in text]
    )


def test_prefix_workload_preserves_final_tokens_and_reports_common_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workload.jsonl"
    path.write_text(
        json.dumps({"prompt_token_ids": [1, 2, 3, 4]})
        + "\n"
        + json.dumps({"text": "abx"})
        + "\n",
        encoding="utf-8",
    )

    workload = example.load_prefix_workload(path, _fake_tokenizer())

    assert workload.prompt_token_ids == ((1, 2, 3, 4), (97, 98, 120))
    assert workload.common_prefix_tokens == 0
    assert workload.prompt_lengths == (4, 3)


def test_prefix_workload_rejects_duplicate_complete_prompts(tmp_path: Path) -> None:
    path = tmp_path / "workload.jsonl"
    row = json.dumps({"prompt_token_ids": [1, 2, 3]})
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate complete prompts"):
        example.load_prefix_workload(path, _fake_tokenizer())


@pytest.mark.parametrize(
    "options,message",
    [
        (["--cache-mode", "shared-prefix"], "shared-prefix requires --workload-jsonl"),
        (
            ["--workload-jsonl", "workload.jsonl", "--enable-thinking"],
            "only supports direct constrained SID Beam mode",
        ),
        (
            ["--workload-jsonl", "workload.jsonl", "--cot"],
            "only supports direct constrained SID Beam mode",
        ),
        (
            ["--workload-jsonl", "workload.jsonl", "--profile-dir", "profile"],
            "requires --repeat, not --profile-dir",
        ),
        (
            ["--workload-jsonl", "workload.jsonl", "--prompt", "prompt.txt"],
            "mutually exclusive",
        ),
        (["--shared-cache-salt", ""], "nonempty"),
    ],
)
def test_prefix_cache_cli_rejects_invalid_modes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    options: list[str],
    message: str,
) -> None:
    if "--workload-jsonl" in options and "--repeat" not in options:
        options += ["--repeat", "1"]
    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path) + options)
    with pytest.raises(SystemExit) as exc:
        example.parse_args()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_prefix_cache_cli_accepts_promptless_workload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workload = tmp_path / "workload.jsonl"
    workload.write_text(json.dumps({"prompt_token_ids": [1, 2]}) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", _cli_args(tmp_path, workload, promptless=True) + ["--repeat", "1"]
    )

    args = example.parse_args()

    assert args.cache_mode == "cold"
    assert args.workload_jsonl == workload
    assert args.shared_cache_salt == "offline-beam-v1-shared-prefix"


@pytest.mark.parametrize("cache_mode", ["off", "cold", "shared-prefix"])  # type: ignore[misc]
def test_prefix_workload_benchmark_consumes_unique_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache_mode: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _cli_args(tmp_path, tmp_path / "workload.jsonl", promptless=True)
        + ["--repeat", "2", "--warmup-runs", "1", "--cache-mode", cache_mode],
    )
    args = example.parse_args()
    token_groups = [[10, 20, index], [30, 40, index], [50, 60, index], [70, 80, index]]
    workload_requests = [
        {
            "prompt_token_ids": ids,
            "cache_salt": (
                f"offline-beam-v1-workload-{index}"
                if cache_mode == "cold"
                else args.shared_cache_salt
            ),
        }
        for index, ids in enumerate(token_groups)
    ]
    workload = example.PrefixWorkload(
        prompt_token_ids=tuple(tuple(ids) for ids in token_groups),
        common_prefix_tokens=0,
    )
    calls: list[list[dict[str, Any]]] = []

    def beam_search(prompts: list[Any], params: Any, *, output_mode: str) -> list[Any]:
        assert output_mode == "full"
        calls.append(prompts)
        return [SimpleNamespace(sequences=[]) for _ in prompts]

    llm = SimpleNamespace(beam_search_v1=beam_search)
    outputs, report = example.benchmark_beam_search(
        llm,
        args,
        workload_requests[:1],
        object(),
        workload_requests=workload_requests,
        workload=workload,
    )

    assert len(outputs) == 1
    assert len(calls) == 4
    assert [[ids["prompt_token_ids"] for ids in call] for call in calls] == token_groups
    if cache_mode == "cold":
        assert [call[0]["cache_salt"] for call in calls] == [
            "offline-beam-v1-workload-0-seed",
            "offline-beam-v1-workload-1-warmup-0",
            "offline-beam-v1-workload-2-benchmark-0",
            "offline-beam-v1-workload-3-benchmark-1",
        ]
    else:
        assert {call[0]["cache_salt"] for call in calls} == {args.shared_cache_salt}
    assert report["input_len"] is None
    assert report["workload"]["common_prefix_tokens"] == 0
    assert report["workload"]["required_requests"] == 4
    assert "seed" in report["timing_scope"]


@pytest.mark.parametrize("cache_mode,expected_caching", [("off", False), ("shared-prefix", True)])
def test_prefix_cache_run_controls_engine_caching_and_workload_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache_mode: str, expected_caching: bool
) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        json.dumps({"prompt_token_ids": [10, 11]}) + "\n"
        + json.dumps({"prompt_token_ids": [10, 12]}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _cli_args(tmp_path, workload_path, promptless=True)
        + [
            "--repeat",
            "1",
            "--warmup-runs",
            "0",
            "--cache-mode",
            cache_mode,
            "--max-model-len",
            "7",
        ],
    )
    args = example.parse_args()
    engine_kwargs: dict[str, Any] = {}
    beam_calls: list[list[dict[str, Any]]] = []
    outputs = [SimpleNamespace(sequences=[])]

    def beam_search(prompts: list[Any], params: Any, *, output_mode: str) -> list[Any]:
        assert output_mode == "full"
        beam_calls.append(prompts)
        return outputs

    llm = SimpleNamespace(
        get_tokenizer=lambda: object(),
        beam_search_v1=beam_search,
    )

    def make_llm(**kwargs: Any) -> Any:
        engine_kwargs.update(kwargs)
        return nullcontext(llm)

    modules: dict[str, dict[str, Any]] = {
        "transformers": {
            "AutoTokenizer": SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object())
        },
        "vllm.inputs": {
            "tokens_input": lambda ids, **kwargs: dict(prompt_token_ids=ids, **kwargs)
        },
        "vllm_gr.entrypoints.gr": {"GRLLM": make_llm},
        "vllm_gr.sampling_params": {"BeamSearchParams": lambda **kwargs: SimpleNamespace(**kwargs)},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(example, "build_artifact", lambda *_args: ("artifact", "tokenizer"))

    example.run(args, tmp_path / "unused.safetensors")

    assert engine_kwargs["enable_prefix_caching"] is expected_caching
    assert [[ids["prompt_token_ids"] for ids in call] for call in beam_calls] == [
        [[10, 11]],
        [[10, 12]],
    ]
    assert beam_calls[0][0]["cache_salt"] == beam_calls[1][0]["cache_salt"]
    assert engine_kwargs["max_num_batched_tokens"] == 7


pytestmark = pytest.mark.cpu_test
