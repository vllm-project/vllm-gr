# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CoT-to-SID handoff must preserve token IDs and fail before incomplete Beam input."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from examples.offline_inference.beam_search import offline_beam_search as example
from tests.helpers.paths import TESTS_DIR


class Tokenizer:
    specials = {"<think>": 10, "</think>": 11}

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.template_calls: list[Any] = []

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        self.texts.append(text)
        if text in self.specials:
            return [self.specials[text]]
        if text == "\n\n":
            return [20, 20]
        return [ord(char) + 100 for char in text]

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return "".join(
            {10: "<think>", 11: "</think>", 20: "\n"}.get(i, chr(i - 100) if i >= 100 else "x")
            for i in ids
        )

    def convert_ids_to_tokens(self, value: int) -> str:
        return {10: "<think>", 11: "</think>"}[value]

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
        self.template_calls.append((messages, kwargs))
        return self.encode(
            "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"
        )


def args() -> Any:
    return SimpleNamespace(
        cot=True,
        cot_max_tokens=100,
        temperature=0,
        top_p=1,
        top_k=-1,
        seed=0,
        generated_steps=3,
        max_model_len=1000,
    )


def runtime(monkeypatch: pytest.MonkeyPatch, completions: list[list[int]]) -> Any:
    state = SimpleNamespace(generated=[], beam_calls=[], tokenizer=Tokenizer(), sampling=[])

    def sampling(**kwargs: Any) -> Any:
        state.sampling.append(kwargs)
        return kwargs

    module = ModuleType("vllm")
    module.SamplingParams = sampling  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)

    def generate(prompts: Any, params: Any, **kwargs: Any) -> list[Any]:
        state.generated.append(prompts)
        return [
            SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[SimpleNamespace(token_ids=ids, finish_reason="stop")],
            )
            for prompt, ids in zip(prompts, completions)
        ]

    def beam(prompts: Any, params: Any, *, output_mode: str) -> list[Any]:
        assert output_mode == "full"
        state.beam_calls.append((prompts, params))
        return [SimpleNamespace(sequences=[]) for _ in prompts]

    state.llm = SimpleNamespace(
        generate=generate, beam_search=beam, get_tokenizer=lambda: state.tokenizer
    )
    return state


@pytest.mark.parametrize("batch", [1, 4])  # type: ignore[misc]
def test_exact_token_handoff_and_per_row_ownership(
    monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    completions = [[300 + row, 11] for row in range(batch)]
    state = runtime(monkeypatch, completions)
    prompts: list[dict[str, Any]] = [
        {"prompt_token_ids": [100, 200 + row], "cache_salt": f"row-{row}"} for row in range(batch)
    ]
    before = json.dumps(prompts)
    options = args()
    params = object()
    outputs = example.execute_beam_batch(state.llm, options, prompts, params)
    assert len(outputs) == batch
    assert state.generated == [prompts]
    assert state.sampling[0]["stop_token_ids"] == [11]
    assert state.sampling[0]["max_tokens"] == 100
    assert state.sampling[0]["detokenize"] is False
    transferred, received_params = state.beam_calls[0]
    assert received_params is params
    for index, transferred_row in enumerate(transferred):
        assert transferred_row["prompt_token_ids"] == [
            100,
            200 + index,
            *completions[index],
            20,
            20,
        ]
        assert transferred_row["cache_salt"] == prompts[index]["cache_salt"] + f"-sid-row-{index}"
    assert json.dumps(prompts) == before
    assert len(options.cot_last_report["requests"]) == batch


@pytest.mark.parametrize("completion", [[], [300], [300, 12]])  # type: ignore[misc]
def test_incomplete_thinking_never_starts_beam(
    monkeypatch: pytest.MonkeyPatch, completion: list[int]
) -> None:
    state = runtime(monkeypatch, [completion])
    with pytest.raises(RuntimeError, match="did not reach"):
        example.execute_beam_batch(state.llm, args(), [{"prompt_token_ids": [100]}], object())
    assert not state.beam_calls


def test_bad_second_row_does_not_partially_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    state = runtime(monkeypatch, [[300, 11], [300]])
    with pytest.raises(RuntimeError):
        example.execute_beam_batch(state.llm, args(), [{"prompt_token_ids": [100]}] * 2, object())
    assert not state.beam_calls


def test_reordered_thinking_results_do_not_mix_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    state = runtime(monkeypatch, [[300 + row, 11] for row in range(4)])
    generate = state.llm.generate
    state.llm.generate = lambda *pos, **kw: list(reversed(generate(*pos, **kw)))
    prompts: list[dict[str, Any]] = [
        {"prompt_token_ids": [100, 200 + row], "cache_salt": f"row-{row}"} for row in range(4)
    ]
    with pytest.raises(RuntimeError, match="prompt/order mismatch"):
        example.execute_beam_batch(state.llm, args(), prompts, object())
    assert not state.beam_calls


@pytest.mark.parametrize("count", [0, 3])  # type: ignore[misc]
def test_missing_thinking_result_stops_entire_batch(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    state = runtime(monkeypatch, [[300, 11]] * count)
    with pytest.raises(RuntimeError, match="different request count"):
        example.execute_beam_batch(state.llm, args(), [{"prompt_token_ids": [100]}] * 4, object())
    assert not state.beam_calls


def test_multiple_continuations_do_not_start_beam(monkeypatch: pytest.MonkeyPatch) -> None:
    state = runtime(monkeypatch, [[300, 11]])
    generate = state.llm.generate

    def duplicated(*pos: Any, **kw: Any) -> Any:
        outputs = generate(*pos, **kw)
        outputs[0].outputs *= 2
        return outputs

    state.llm.generate = duplicated
    with pytest.raises(RuntimeError, match="one continuation"):
        example.execute_beam_batch(state.llm, args(), [{"prompt_token_ids": [100]}], object())
    assert not state.beam_calls


def test_missing_beam_results_are_not_reported_as_a_complete_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = runtime(monkeypatch, [[300, 11]] * 4)
    state.llm.beam_search = lambda *args, **kwargs: []
    with pytest.raises(RuntimeError, match="Beam returned a different request count"):
        example.execute_beam_batch(state.llm, args(), [{"prompt_token_ids": [100]}] * 4, object())


@pytest.mark.parametrize("batch", [1, 4])  # type: ignore[misc]
def test_disabled_mode_does_not_generate_or_rewrite(
    monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    state = runtime(monkeypatch, [])
    prompts = [{"prompt_token_ids": [100 + row]} for row in range(batch)]
    example.execute_beam_batch(state.llm, SimpleNamespace(cot=False), prompts, object())
    assert not state.generated
    assert state.beam_calls[0][0] is prompts


@pytest.mark.parametrize(  # type: ignore[misc]
    "builder", [example.make_thinking_prompt, example.make_cot_prompt]
)
@pytest.mark.parametrize("encoding", ["raw", "json"])  # type: ignore[misc]
def test_existing_serialized_prompt_reopens_empty_think_only(
    tmp_path: Path, encoding: str, builder: Any
) -> None:
    tokenizer = Tokenizer()
    text = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nHistory <s_a_1>/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    path = tmp_path / "prompt.txt"
    path.write_text(json.dumps(text) if encoding == "json" else text, encoding="utf-8")
    before = path.read_bytes()
    result = builder(tokenizer, path)
    expected = (
        text[: text.index("<|im_start|>assistant")].replace("/no_think", "/think")
        + "<|im_start|>assistant\n<think>\n"
    )
    assert result == tokenizer.encode(expected)
    assert path.read_bytes() == before
    assert "History <s_a_1>" in tokenizer.texts[-1]
    assert tokenizer.template_calls == []


@pytest.mark.parametrize(  # type: ignore[misc]
    "builder", [example.make_thinking_prompt, example.make_cot_prompt]
)
def test_legacy_single_line_recommendation_prompt_requires_explicit_encoding(builder: Any) -> None:
    tokenizer = Tokenizer()
    path = TESTS_DIR / "resources" / "single_one_rec_prompt.txt"
    before = path.read_bytes()
    source = path.read_text(encoding="utf-8")
    assert "\\n" in source
    with pytest.raises(ValueError, match="ambiguous single-line ChatML"):
        builder(tokenizer, path)
    assert tokenizer.template_calls == []
    assert path.read_bytes() == before


@pytest.mark.parametrize(  # type: ignore[misc]
    "builder", [example.make_thinking_prompt, example.make_cot_prompt]
)
@pytest.mark.parametrize("encoding", ["raw", "json"])  # type: ignore[misc]
def test_explicitly_encoded_recommendation_prompt_keeps_original_conversation(
    tmp_path: Path, builder: Any, encoding: str
) -> None:
    tokenizer = Tokenizer()
    fixture = TESTS_DIR / "resources" / "single_one_rec_prompt.txt"
    before = fixture.read_bytes()
    source = fixture.read_text(encoding="utf-8")
    # This fixture is known to use escaped structural separators. Convert a
    # test-owned copy explicitly; the prompt reader must never guess this rule.
    normalized = source.replace("\\n", "\n")
    path = tmp_path / "explicit-prompt.txt"
    path.write_text(json.dumps(normalized) if encoding == "json" else normalized, encoding="utf-8")
    prefix, assistant, _ = normalized.rpartition("<|im_start|>assistant")
    assert assistant
    assert prefix.count("/no_think<|im_end|>") == 1
    expected = prefix.replace("/no_think<|im_end|>", "/think<|im_end|>")
    expected += "<|im_start|>assistant\n<think>\n"

    result = builder(tokenizer, path)

    assert result == tokenizer.encode(expected)
    assert tokenizer.template_calls == []
    assert expected.count("<|im_start|>system") == 1
    assert expected.count("<|im_start|>user") == 1
    assert expected.count("<|im_start|>assistant") == 1
    assert fixture.read_bytes() == before


def test_plain_prompt_gets_one_chat_template_and_open_think(tmp_path: Path) -> None:
    tokenizer = Tokenizer()
    path = tmp_path / "prompt.txt"
    path.write_text("user history", encoding="utf-8")
    result = example.make_cot_prompt(tokenizer, path)
    rendered = tokenizer.decode(result)
    assert rendered.count("<|im_start|>user") == 1
    assert rendered.endswith("<think>\n")


@pytest.mark.parametrize("tail", [[], [10, 20], [10, 20, 20, 11, 20, 20]])  # type: ignore[misc]
def test_plain_prompt_reopens_only_terminal_empty_template_span(
    tmp_path: Path, tail: list[int]
) -> None:
    tokenizer = Tokenizer()
    prefix = tokenizer.encode("<|im_start|>assistant\n")
    tokenizer.apply_chat_template = lambda *args, **kwargs: [*prefix, *tail]  # type: ignore[method-assign]
    path = tmp_path / "prompt.txt"
    path.write_text("a complete user question", encoding="utf-8")
    result = example.make_cot_prompt(tokenizer, path)
    rendered = tokenizer.decode(result)
    assert rendered.count("<think>") == 1
    assert "</think>" not in rendered
    assert rendered.endswith("<think>\n")
    assert result[: len(prefix)] == prefix


@pytest.mark.parametrize(  # type: ignore[misc]
    "builder", [example.make_thinking_prompt, example.make_cot_prompt]
)
@pytest.mark.parametrize("whitespace", ["", "\n\n", " \t\r\n"])  # type: ignore[misc]
def test_both_modes_reopen_closed_empty_template_span_without_reencoding_prefix(
    tmp_path: Path, builder: Any, whitespace: str
) -> None:
    tokenizer = Tokenizer()
    # The first two IDs deliberately have no round-trip text encoding. Earlier
    # nonempty thinking belongs to the conversation and must remain untouched.
    prefix = [41, 42, 10, 300, 11, 20, 10]
    original = [*prefix, *tokenizer.encode(whitespace), 11, *tokenizer.encode(whitespace)]
    tokenizer.apply_chat_template = lambda *args, **kwargs: list(original)  # type: ignore[method-assign]
    path = tmp_path / "question.txt"
    path.write_text("a complete user question", encoding="utf-8")

    result = builder(tokenizer, path)

    assert result == [*prefix, *tokenizer.encode("\n")]
    assert original[: len(prefix)] == prefix


def test_plain_template_nonempty_thinking_is_not_removed(tmp_path: Path) -> None:
    tokenizer = Tokenizer()
    tokenizer.apply_chat_template = lambda *args, **kwargs: [100, 10, 300, 11, 20]  # type: ignore[method-assign]
    path = tmp_path / "prompt.txt"
    path.write_text("question", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty thinking span"):
        example.make_cot_prompt(tokenizer, path)


def test_thinking_only_keeps_nonempty_template_thinking(tmp_path: Path) -> None:
    tokenizer = Tokenizer()
    original = [41, 42, 10, 300, 11, 20]
    tokenizer.apply_chat_template = lambda *args, **kwargs: list(original)  # type: ignore[method-assign]
    path = tmp_path / "question.txt"
    path.write_text("question", encoding="utf-8")
    assert example.make_thinking_prompt(tokenizer, path) == original


@pytest.mark.parametrize(  # type: ignore[misc]
    "builder", [example.make_thinking_prompt, example.make_cot_prompt]
)
def test_terminal_empty_span_cannot_reuse_an_earlier_opening_token(
    tmp_path: Path, builder: Any
) -> None:
    tokenizer = Tokenizer()
    # The last rendered tag is ordinary text in this malformed template, not
    # a dedicated token. An earlier conversation tag cannot authorize slicing.
    original = [10, 300, 11, 20, *tokenizer.encode("<think>\n</think>\n")]
    tokenizer.apply_chat_template = lambda *args, **kwargs: list(original)  # type: ignore[method-assign]
    path = tmp_path / "question.txt"
    path.write_text("question", encoding="utf-8")
    with pytest.raises(ValueError, match="terminal <think> token boundary"):
        builder(tokenizer, path)


@pytest.mark.parametrize(  # type: ignore[misc]
    "builder", [example.make_thinking_prompt, example.make_cot_prompt]
)
@pytest.mark.parametrize("encoding", ["raw", "json"])  # type: ignore[misc]
def test_prompt_keeps_literal_escapes_and_earlier_thinking_content(
    tmp_path: Path, builder: Any, encoding: str
) -> None:
    tokenizer = Tokenizer()
    path = tmp_path / "prompt.txt"
    text = (
        "<|im_start|>user\nA literal \\n marker, C:\\new, and /no_think text<|im_end|>\n"
        "<|im_start|>assistant\n<think>previous reasoning</think>answer<|im_end|>\n"
        "<|im_start|>user\nNext question /no_think<|im_end|>\n<|im_start|>assistant\n"
    )
    source = json.dumps(text) if encoding == "json" else text
    path.write_text(source, encoding="utf-8")
    expected = text.replace("Next question /no_think", "Next question /think") + "<think>\n"
    assert builder(tokenizer, path) == tokenizer.encode(expected)
    assert tokenizer.template_calls == []
    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize(  # type: ignore[misc]
    "builder", [example.make_thinking_prompt, example.make_cot_prompt]
)
def test_ambiguous_single_line_chat_is_rejected_without_decoding_body(
    tmp_path: Path, builder: Any
) -> None:
    tokenizer = Tokenizer()
    path = tmp_path / "ambiguous.txt"
    text = (
        r"<|im_start|>user\nKeep C:\new and a literal \n marker/no_think"
        r"<|im_end|>\n<|im_start|>assistant\n"
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous single-line ChatML"):
        builder(tokenizer, path)
    assert not tokenizer.template_calls
    assert text not in tokenizer.texts
    assert path.read_text(encoding="utf-8") == text


@pytest.mark.parametrize(  # type: ignore[misc]
    "builder", [example.make_thinking_prompt, example.make_cot_prompt]
)
def test_json_chat_is_decoded_once_and_preserves_body_backslashes(
    tmp_path: Path, builder: Any
) -> None:
    tokenizer = Tokenizer()
    path = tmp_path / "prompt.json"
    # One real structural newline and two literal backslashes before n in the
    # user body remain distinct after exactly one JSON decoding operation.
    text = (
        "<|im_start|>user\n"
        r'Keep C:\new, one \n, two \\n, and "quoted text"/no_think'
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    path.write_text(json.dumps(text, ensure_ascii=False), encoding="utf-8")
    expected = text.replace('"quoted text"/no_think', '"quoted text"/think') + "<think>\n"
    assert builder(tokenizer, path) == tokenizer.encode(expected)
    assert not tokenizer.template_calls
    assert r"C:\new" in tokenizer.texts[-1]
    assert r"one \n, two \\n" in tokenizer.texts[-1]


@pytest.mark.parametrize(  # type: ignore[misc]
    "builder", [example.make_thinking_prompt, example.make_cot_prompt]
)
def test_incomplete_json_chat_does_not_fall_back_to_plain_user_text(
    tmp_path: Path, builder: Any
) -> None:
    path = tmp_path / "incomplete.json"
    path.write_text('"<|im_start|>user\\nQ<|im_end|>', encoding="utf-8")
    tokenizer = Tokenizer()
    with pytest.raises(ValueError, match="JSON-encoded ChatML"):
        builder(tokenizer, path)
    assert not tokenizer.template_calls


def test_quoted_plain_text_is_not_treated_as_serialized_chat(tmp_path: Path) -> None:
    tokenizer = Tokenizer()
    path = tmp_path / "question.txt"
    text = json.dumps(r'Explain "quoted" and literal \n text')
    path.write_text(text, encoding="utf-8")
    example.make_thinking_prompt(tokenizer, path)
    assert tokenizer.template_calls[0][0] == [{"role": "user", "content": text}]


@pytest.mark.parametrize(  # type: ignore[misc]
    "builder", [example.make_thinking_prompt, example.make_cot_prompt]
)
def test_nonempty_assistant_content_is_not_silently_removed(tmp_path: Path, builder: Any) -> None:
    path = tmp_path / "prompt.txt"
    path.write_text(
        "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>already generated",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty assistant"):
        builder(Tokenizer(), path)


@pytest.mark.parametrize(  # type: ignore[misc]
    "options", [["--cot", "--enable-thinking"], ["--cot", "--cot-max-tokens", "0"]]
)
def test_invalid_mode_or_budget_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, options: list[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["example", "--model", str(tmp_path), "--prompt", "p", "--catalog", "c", *options],
    )
    with pytest.raises(SystemExit):
        example.parse_args()


def test_cot_requires_same_catalog_but_allows_single_output_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["example", "--model", str(tmp_path), "--prompt", "p", "--cot"]
    )
    with pytest.raises(SystemExit):
        example.parse_args()
    monkeypatch.setattr(sys, "argv", sys.argv + ["--catalog", "c", "--output-json", "out.json"])
    options = example.parse_args()
    assert options.cot and not options.enable_thinking
    assert options.cot_max_tokens == 1024
    assert options.cot_on_limit == "error"


def test_explicit_budget_truncation_is_marked_and_keeps_every_generated_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = runtime(monkeypatch, [[300, 301]])
    # This opt-in applies only to length exhaustion, never to other stop reasons.
    original_generate = state.llm.generate

    def length_stop(*pos: Any, **kw: Any) -> Any:
        outputs = original_generate(*pos, **kw)
        outputs[0].outputs[0].finish_reason = "length"
        return outputs

    state.llm.generate = length_stop
    options = args()
    options.cot_max_tokens = 2
    options.cot_on_limit = "truncate"
    example.execute_beam_batch(state.llm, options, [{"prompt_token_ids": [100]}], object())
    assert state.beam_calls[0][0][0]["prompt_token_ids"] == [100, 300, 301, 11, 20, 20]
    report = options.cot_last_report["requests"][0]
    assert report["forced_close"] and report["thinking_truncated"]
    assert report["generated_thinking_tokens"] == 2


def test_truncate_opt_in_does_not_hide_other_early_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    state = runtime(monkeypatch, [[300]])
    options = args()
    options.cot_max_tokens = 1
    options.cot_on_limit = "truncate"
    with pytest.raises(RuntimeError, match="did not reach"):
        example.execute_beam_batch(state.llm, options, [{"prompt_token_ids": [100]}], object())
    assert not state.beam_calls


def test_actual_context_overflow_never_starts_beam(monkeypatch: pytest.MonkeyPatch) -> None:
    state = runtime(monkeypatch, [[300, 11]])
    options = args()
    options.max_model_len = 5
    with pytest.raises(ValueError, match="max-model-len"):
        example.execute_beam_batch(state.llm, options, [{"prompt_token_ids": [100]}], object())
    assert not state.beam_calls


def test_cot_benchmark_repeats_both_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = runtime(monkeypatch, [[300, 11]])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "example",
            "--model",
            str(tmp_path),
            "--prompt",
            "p",
            "--catalog",
            "c",
            "--cot",
            "--repeat",
            "2",
            "--warmup-runs",
            "3",
        ],
    )
    options = example.parse_args()
    _, report = example.benchmark_beam_search(
        state.llm, options, [{"prompt_token_ids": [100], "cache_salt": "row"}], object()
    )
    assert len(state.generated) == len(state.beam_calls) == 5
    assert report["mode"] == "cot_beam"
    assert report["input_len"] is None
    assert "CoT generation" in report["timing_scope"]


@pytest.mark.parametrize("failed_stage", ["generate", "beam_search"])  # type: ignore[misc]
def test_profile_stops_after_either_stage_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failed_stage: str
) -> None:
    state = runtime(monkeypatch, [[300, 11]])
    events: list[str] = []
    state.llm.start_profile = lambda *args: events.append("start")
    state.llm.stop_profile = lambda: events.append("stop")

    def fail(*pos: Any, **kw: Any) -> Any:
        raise RuntimeError("stage failed")

    setattr(state.llm, failed_stage, fail)
    options = args()
    options.profile_dir = tmp_path
    options.profile_warmup_runs = 0
    options.profile_runs = 1
    options.execution = "graph"
    options.batch_size = 1
    options.beam_width = 128
    with pytest.raises(RuntimeError, match="stage failed"):
        example.run_beam_search(
            state.llm, options, [{"prompt_token_ids": [100], "cache_salt": "row"}], object()
        )
    assert events == ["start", "stop"]
    if failed_stage == "generate":
        assert not state.beam_calls


pytestmark = pytest.mark.cpu_test
