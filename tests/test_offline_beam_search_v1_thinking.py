# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Device-free coverage of ordinary thinking generation in the V1 example."""

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from examples.offline_inference.beam_search import offline_beam_search_v1 as example


def _args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *options: str) -> Any:
    prompt = tmp_path / "question.txt"
    prompt.write_text(
        "\u8bf7\u8ba1\u7b97 12 + 8 \xd7 3\uff0c\u5e76\u7ed9\u51fa\u63a8\u7406\u548c\u7b54\u6848\u3002",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["example", "--model", str(tmp_path), "--prompt", str(prompt), *options],
    )
    return example.parse_args()


def _runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    state = SimpleNamespace(
        ids=[101, 102, 103, 104],
        template_calls=[],
        tokenizer_calls=[],
        engine_calls=[],
        sampling_calls=[],
        generate_calls=[],
        events=[],
        fail_on_call=None,
    )

    class Tokenizer:
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
            state.template_calls.append((messages, kwargs))
            return list(state.ids)

        def encode(self, text: str, **kwargs: Any) -> list[int]:
            specials = {"<think>": 10, "</think>": 11}
            return [specials[text]] if text in specials else [ord(char) + 100 for char in text]

        def convert_ids_to_tokens(self, token: int) -> str:
            return {10: "<think>", 11: "</think>"}[token]

        def decode(self, ids: list[int], **kwargs: Any) -> str:
            return "".join(
                {10: "<think>", 11: "</think>", 20: "\n"}.get(
                    token, chr(token - 100) if token >= 100 else "x"
                )
                for token in ids
            )

    tokenizer = Tokenizer()

    def from_pretrained(model: str, **kwargs: Any) -> Any:
        state.tokenizer_calls.append((model, kwargs))
        return tokenizer

    def sampling_params(**kwargs: Any) -> Any:
        state.sampling_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    def generate(prompts: list[Any], sampling_params: Any, **kwargs: Any) -> list[Any]:
        index = len(state.generate_calls)
        state.generate_calls.append((prompts, sampling_params, kwargs))
        state.events.append("generate")
        if state.fail_on_call == index:
            raise RuntimeError("simulated generation failure")
        completion = SimpleNamespace(
            index=0,
            text="<think>\u5148\u4e58\u540e\u52a0\u3002</think>\n\u7b54\u6848\u662f36\u3002",
            token_ids=[11, 12, 13],
            finish_reason="stop",
            stop_reason=13,
        )
        return [SimpleNamespace(outputs=[completion]) for _ in prompts]

    def start_profile(*args: Any, **kwargs: Any) -> None:
        state.events.append("start_profile")

    def stop_profile(*args: Any, **kwargs: Any) -> None:
        state.events.append("stop_profile")

    llm = SimpleNamespace(
        generate=generate,
        start_profile=start_profile,
        stop_profile=stop_profile,
        get_tokenizer=lambda: tokenizer,
    )

    @contextmanager
    def make_llm(**kwargs: Any) -> Any:
        state.engine_calls.append(kwargs)
        try:
            yield llm
        finally:
            state.events.append("engine_closed")

    modules: dict[str, dict[str, Any]] = {
        "transformers": {"AutoTokenizer": SimpleNamespace(from_pretrained=from_pretrained)},
        "vllm": {"SamplingParams": sampling_params},
        "vllm.inputs": {"tokens_input": lambda ids, **kw: dict(prompt_token_ids=ids, **kw)},
        "vllm_gr": {"GRLLM": make_llm},
        "vllm_gr.entrypoints": {},
        "vllm_gr.entrypoints.gr": {"GRLLM": make_llm},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        module.__dict__["__path__"] = []
        monkeypatch.setitem(sys.modules, name, module)

    def unexpected_beam_call(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Thinking generation must not use Beam/catalog helpers")

    for name in ("build_artifact", "load_triples", "make_prompt"):
        monkeypatch.setattr(example, name, unexpected_beam_call)
    state.tokenizer = tokenizer
    return state


def test_default_cli_still_requires_catalog_for_beam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as exc:
        _args(monkeypatch, tmp_path)
    assert exc.value.code == 2
    args = _args(monkeypatch, tmp_path, "--catalog", str(tmp_path / "catalog.json"))
    assert not args.enable_thinking
    assert args.beam_width == 128
    assert args.generated_steps == 3


def test_thinking_cli_allows_standalone_json_and_preserves_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(
        monkeypatch, tmp_path, "--enable-thinking", "--output-json", str(tmp_path / "result.json")
    )
    assert args.enable_thinking
    assert args.catalog is None
    assert args.repeat is None
    assert args.max_tokens == 1024
    assert args.temperature == 0.6
    assert args.top_p == 0.95
    assert args.top_k == 20
    assert args.seed == 0


@pytest.mark.parametrize("budget", [1, 1536])  # type: ignore[misc]
def test_thinking_budget_is_not_limited_to_beam_stage_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, budget: int
) -> None:
    args = _args(monkeypatch, tmp_path, "--enable-thinking", "--max-tokens", str(budget))
    assert args.max_tokens == budget


@pytest.mark.parametrize(  # type: ignore[misc]
    "options",
    [
        ("--dynamic-beam-widths", "128,64,32"),
        ("--result-width", "32"),
        ("--max-tokens", "0"),
        ("--max-tokens", "-1"),
    ],
)
def test_thinking_cli_rejects_beam_schedule_and_nonpositive_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, options: tuple[str, str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        _args(monkeypatch, tmp_path, "--enable-thinking", *options)
    assert exc.value.code == 2


def test_thinking_prompt_uses_real_tokens_and_one_thinking_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _runtime(monkeypatch)
    prompt = tmp_path / "question.txt"
    prompt.write_text("\u8bf7\u8ba1\u7b971\u52a02\u3002", encoding="utf-8")
    result = example.make_thinking_prompt(state.tokenizer, prompt)
    assert result == [101, 102, 103, 104]
    assert state.template_calls == [
        (
            [{"role": "user", "content": "\u8bf7\u8ba1\u7b971\u52a02\u3002"}],
            {
                "tokenize": True,
                "return_dict": False,
                "enable_thinking": True,
                "add_generation_prompt": True,
            },
        )
    ]


@pytest.mark.parametrize(  # type: ignore[misc]
    "tokens,error", [([], ValueError), ([[101, 102]], TypeError)]
)
def test_thinking_prompt_requires_nonempty_flat_token_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokens: list[Any],
    error: type[Exception],
) -> None:
    state = _runtime(monkeypatch)
    state.ids = tokens
    prompt = tmp_path / "question.txt"
    prompt.write_text("\u95ee\u9898", encoding="utf-8")
    with pytest.raises(error):
        example.make_thinking_prompt(state.tokenizer, prompt)


@pytest.mark.parametrize(  # type: ignore[misc]
    "batch_size,execution", [(1, "eager"), (4, "graph")]
)
def test_thinking_uses_unconstrained_generate_and_exports_utf8_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    batch_size: int,
    execution: str,
) -> None:
    state = _runtime(monkeypatch)
    output = tmp_path / "nested" / "thinking.json"
    args = _args(
        monkeypatch,
        tmp_path,
        "--enable-thinking",
        "--constraint-backend",
        "npu",
        "--catalog",
        str(tmp_path / "missing-catalog.json"),
        "--batch-size",
        str(batch_size),
        "--execution",
        execution,
        "--input-len",
        "999999",
        "--max-model-len",
        "12",
        "--max-tokens",
        "8",
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--top-k",
        "-1",
        "--seed",
        "7",
        "--output-json",
        str(output),
    )
    assert example.run_thinking(args) is None
    assert len(state.engine_calls) == 1
    kwargs = state.engine_calls[0]
    config = kwargs["vllm_gr_config"]
    assert config["attention_backend"] == "CUSTOM"
    assert config["constrained_generation"] is False
    assert config["constraint_table"]["enabled"] is False
    assert config["constraint_table"]["max_top_k"] == 1
    assert "beam" not in config
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["enforce_eager"] is (execution == "eager")
    assert kwargs["max_num_seqs"] == max(4, batch_size)
    assert len(state.generate_calls) == 1
    prompts, sampling, _ = state.generate_calls[0]
    assert len(prompts) == batch_size
    assert all(prompt["prompt_token_ids"] == state.ids for prompt in prompts)
    if batch_size > 1:
        assert prompts[0]["prompt_token_ids"] is not prompts[1]["prompt_token_ids"]
    assert sampling.max_tokens == 8
    assert sampling.temperature == 0
    assert sampling.top_p == 1
    assert sampling.top_k == -1
    assert sampling.seed == 7
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "thinking"
    assert report["input_tokens"] == len(state.ids)
    assert report["max_tokens"] == 8
    assert len(report["results"]) == batch_size
    for row in report["results"]:
        assert (
            row["text"]
            == "<think>\u5148\u4e58\u540e\u52a0\u3002</think>\n\u7b54\u6848\u662f36\u3002"
        )
        assert row["token_ids"] == [11, 12, 13]
        assert row["generated_tokens"] == 3
        assert row["finish_reason"] == "stop"
        assert row["stop_reason"] == 13
    assert "\u7b54\u6848\u662f36" in output.read_text(encoding="utf-8")
    assert "\u7b54\u6848\u662f36" in capsys.readouterr().out


@pytest.mark.parametrize(  # type: ignore[misc]
    "batch_size,execution", [(1, "eager"), (4, "graph")]
)
@pytest.mark.parametrize("encoding", ["raw", "json"])  # type: ignore[misc]
def test_thinking_submits_normalized_serialized_chat_without_cot_stopping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    batch_size: int,
    execution: str,
    encoding: str,
) -> None:
    state = _runtime(monkeypatch)
    args = _args(
        monkeypatch,
        tmp_path,
        "--enable-thinking",
        "--constraint-backend",
        "npu",
        "--batch-size",
        str(batch_size),
        "--execution",
        execution,
        "--input-len",
        "1",
        "--max-tokens",
        "8",
    )
    text = (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n"
        "History <s_a_1> C:\\new literal \\n/no_think<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    source = json.dumps(text) if encoding == "json" else text
    args.prompt.write_text(source, encoding="utf-8")
    expected = (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n"
        "History <s_a_1> C:\\new literal \\n/think<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    expected_ids = state.tokenizer.encode(expected)

    example.run_thinking(args)

    assert state.template_calls == []
    assert len(state.generate_calls) == 1
    prompts, sampling, _ = state.generate_calls[0]
    assert len(prompts) == batch_size
    assert all(prompt["prompt_token_ids"] == expected_ids for prompt in prompts)
    if batch_size > 1:
        assert prompts[0]["prompt_token_ids"] is not prompts[1]["prompt_token_ids"]
    assert sampling.max_tokens == 8
    assert "stop_token_ids" not in state.sampling_calls[0]
    assert args.prompt.read_text(encoding="utf-8") == source


@pytest.mark.parametrize("encoding", ["raw", "json"])  # type: ignore[misc]
def test_thinking_checks_normalized_serialized_length_before_engine_allocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, encoding: str
) -> None:
    state = _runtime(monkeypatch)
    args = _args(
        monkeypatch,
        tmp_path,
        "--enable-thinking",
        "--max-model-len",
        "12",
        "--max-tokens",
        "8",
    )
    text = "<|im_start|>user\nHistory/no_think<|im_end|>\n<|im_start|>assistant\n"
    args.prompt.write_text(json.dumps(text) if encoding == "json" else text, encoding="utf-8")

    with pytest.raises(ValueError, match="max-model-len"):
        example.run_thinking(args)

    assert state.template_calls == []
    assert state.engine_calls == []
    assert state.generate_calls == []


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
def test_thinking_submits_reopened_template_span_without_cot_stopping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, batch_size: int
) -> None:
    state = _runtime(monkeypatch)
    state.ids = [41, 42, 10, 20, 11, 20]
    args = _args(
        monkeypatch,
        tmp_path,
        "--enable-thinking",
        "--batch-size",
        str(batch_size),
        "--max-tokens",
        "8",
    )

    example.run_thinking(args)

    assert len(state.template_calls) == 1
    prompts, sampling, _ = state.generate_calls[0]
    assert len(prompts) == batch_size
    expected = [41, 42, 10, *state.tokenizer.encode("\n")]
    assert all(prompt["prompt_token_ids"] == expected for prompt in prompts)
    assert sampling.max_tokens == 8
    assert "stop_token_ids" not in state.sampling_calls[0]


def test_thinking_rejects_real_context_overflow_before_engine_allocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _runtime(monkeypatch)
    args = _args(
        monkeypatch,
        tmp_path,
        "--enable-thinking",
        "--max-model-len",
        "11",
        "--max-tokens",
        "8",
        "--input-len",
        "1",
    )
    with pytest.raises(ValueError, match="max-model-len"):
        example.run_thinking(args)
    assert len(state.template_calls) == 1
    assert state.engine_calls == []
    assert state.generate_calls == []


def test_thinking_repeat_counts_generation_warmups_and_exports_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _runtime(monkeypatch)
    output = tmp_path / "benchmark.json"
    args = _args(
        monkeypatch,
        tmp_path,
        "--enable-thinking",
        "--max-tokens",
        "8",
        "--warmup-runs",
        "3",
        "--repeat",
        "2",
        "--output-json",
        str(output),
    )
    example.run_thinking(args)
    assert len(state.generate_calls) == 5
    assert "start_profile" not in state.events
    report = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(report["benchmark"], dict)
    assert "returned_beams" not in json.dumps(report["benchmark"])


def test_thinking_profile_excludes_warmup_and_stops_after_collection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _runtime(monkeypatch)
    args = _args(
        monkeypatch,
        tmp_path,
        "--enable-thinking",
        "--max-tokens",
        "8",
        "--profile-dir",
        str(tmp_path / "profile"),
        "--profile-warmup-runs",
        "2",
        "--profile-runs",
        "3",
    )
    example.run_thinking(args)
    assert state.events == [
        "generate",
        "generate",
        "start_profile",
        "generate",
        "generate",
        "generate",
        "stop_profile",
        "engine_closed",
    ]
    assert "profiler_config" in state.engine_calls[0]


def test_thinking_profile_stops_when_generation_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _runtime(monkeypatch)
    state.fail_on_call = 0
    args = _args(
        monkeypatch,
        tmp_path,
        "--enable-thinking",
        "--max-tokens",
        "8",
        "--profile-dir",
        str(tmp_path / "profile"),
        "--profile-warmup-runs",
        "0",
    )
    with pytest.raises(RuntimeError, match="simulated generation failure"):
        example.run_thinking(args)
    assert state.events == ["start_profile", "generate", "stop_profile", "engine_closed"]


def test_main_dispatches_thinking_without_catalog_check_or_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(
        monkeypatch,
        tmp_path,
        "--enable-thinking",
        "--catalog",
        str(tmp_path / "not-a-catalog.json"),
        "--input-len",
        "999999",
        "--output-json",
        str(tmp_path / "result.json"),
    )
    received = []

    def unexpected_beam_call(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Thinking dispatch must not allocate a Beam artifact")

    monkeypatch.setattr(example, "parse_args", lambda: args)
    monkeypatch.setattr(example, "run_thinking", lambda options: received.append(options))
    monkeypatch.setattr(example, "run", unexpected_beam_call)
    monkeypatch.setattr(example.tempfile, "TemporaryDirectory", unexpected_beam_call)
    example.main()
    assert received == [args]


pytestmark = pytest.mark.cpu_test
