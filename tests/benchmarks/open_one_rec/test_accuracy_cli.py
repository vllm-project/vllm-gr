# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Check OpenOneRec accuracy requests and failure reports."""

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import requests

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]

_SID_A = "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"
_SID_B = "<|sid_begin|><s_a_4><s_b_5><s_c_6><|sid_end|>"


@pytest.mark.parametrize(  # type: ignore[misc]
    ("beam_widths", "result_width", "returned_count", "expected_error"),
    [
        pytest.param([128, 128, 128], 32, 32, None, id="explicit-result-width"),
        pytest.param([128, 64, 32], None, 32, None, id="final-retained-width"),
        pytest.param([128, 128, 128], 32, 31, "31 of 32", id="too-few-candidates"),
        pytest.param([128, 128, 128], None, 128, None, id="default-full-width"),
    ],
)
def test_offline_accuracy_checks_resolved_result_width(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    beam_widths: list[int],
    result_width: int | None,
    returned_count: int,
    expected_error: str | None,
) -> None:
    from transformers import AutoTokenizer

    import benchmarks.open_one_rec.one_rec_acc_test as accuracy

    def beam_search(
        prompts: list[dict[str, str]], params: Any, *, output_mode: str
    ) -> list[SimpleNamespace]:
        assert prompts == [{"prompt": "question"}]
        assert params.beam_width == 128
        assert output_mode == "compact"
        sequence = SimpleNamespace(token_ids=(1, 2, 3), cumulative_logprob=0.0)
        return [SimpleNamespace(sequences=[sequence] * returned_count)]

    @contextmanager
    def engine(*, gr: dict[str, Any], **kwargs: Any) -> Iterator[SimpleNamespace]:
        vllm_config = SimpleNamespace(additional_config={"vllm_gr": {"gr": gr}})
        yield SimpleNamespace(
            llm_engine=SimpleNamespace(vllm_config=vllm_config),
            beam_search=beam_search,
            get_tokenizer=lambda: SimpleNamespace(decode=lambda *a, **kw: _SID_A),
        )

    samples = {"video-42": {"prompt": "question", "ground_truth": _SID_A}}
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(
        accuracy, "DataLoaderWrapper", lambda **kw: SimpleNamespace(load_data=lambda **kw: samples)
    )
    monkeypatch.setattr(accuracy, "GRLLM", engine)
    monkeypatch.setitem(
        accuracy.TASK_REGISTRY["video"].config["evaluation_config"], "evaluation_mode", "sid"
    )
    settings = {
        "beam_width": beam_widths,
        "candidate_width": beam_widths,
        "result_width": result_width,
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accuracy",
            "--task-type",
            "video",
            "--n",
            "128",
            "--gr",
            json.dumps(settings),
            "--result-dir",
            str(tmp_path),
        ],
    )

    status = accuracy.main()
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    if expected_error is not None:
        assert status == 1
        assert expected_error in report["sample_errors"]["video-42"]
    else:
        assert status == 0, report
        assert report["total_samples"] == 1
        assert report["recall@32"] == 1.0


@pytest.mark.parametrize(  # type: ignore[misc]
    ("failure", "expected"),
    [
        ("http", "HTTPError"),
        ("timeout", "Timeout"),
        ("no choices", "no choices"),
        ("too few choices", "expected 2 choices"),
        ("bad choice", "no text message"),
        ("null content", "no text message"),
        ("empty content", "no text message"),
        ("whitespace only", "no text message"),
    ],
)
def test_accuracy_cli_reports_failed_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str, expected: str
) -> None:
    from transformers import AutoTokenizer

    import benchmarks.open_one_rec.one_rec_acc_test as accuracy

    def post(_url: str, **kwargs: Any) -> Any:
        assert kwargs["timeout"] == 12
        if failure == "timeout":
            raise requests.Timeout("read timed out")
        response = requests.Response()
        response.status_code = 500 if failure == "http" else 200
        choices = {
            "no choices": [],
            "too few choices": [{"message": {"content": "short"}}],
            "bad choice": [{}, {}],
            "null content": [
                {"message": {"content": "valid answer"}},
                {"message": {"content": None}},
            ],
            "empty content": [
                {"message": {"content": "", "reasoning_content": None}},
                {"message": {"content": "valid answer"}},
            ],
            "whitespace only": [
                {"message": {"content": " ", "reasoning_content": "\n"}},
                {"message": {"content": "valid answer"}},
            ],
        }
        response._content = json.dumps({"choices": choices.get(failure, [])}).encode()
        return response

    def load(**kwargs: Any) -> dict[str, dict[str, str]]:
        assert kwargs["sample_size"] is None
        return {"video-42": {"prompt": "question", "ground_truth": "answer"}}

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    loader = Mock(return_value=SimpleNamespace(load_data=load))
    monkeypatch.setattr(accuracy, "DataLoaderWrapper", loader)
    monkeypatch.setattr(accuracy.requests, "post", post)
    monkeypatch.setitem(
        accuracy.TASK_REGISTRY["video"].config["evaluation_config"], "evaluation_mode", "sid"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accuracy",
            "--task-type",
            "video",
            "--endpoint",
            "/v1/chat/completions",
            "--n",
            "2",
            "--request-timeout",
            "12",
            "--result-dir",
            str(tmp_path),
        ],
    )
    assert accuracy.main() == 1
    assert loader.call_args.kwargs["data_dir"] == "./data/benchmark_data"
    report = json.loads((tmp_path / "video-failed.json").read_text())
    assert report["status"] == "failed"
    assert expected in report["sample_errors"]["video-42"]
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize(  # type: ignore[misc]
    ("ground_truth", "content", "expected_pass_at_1"),
    [
        pytest.param(_SID_B, _SID_B, 1, id="final-answer-matches"),
        pytest.param(_SID_A, _SID_B, 0, id="only-reasoning-matches"),
        pytest.param(_SID_A, None, 0, id="missing-final-answer"),
    ],
)
def test_accuracy_scores_final_sid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ground_truth: str,
    content: str | None,
    expected_pass_at_1: int,
) -> None:
    from transformers import AutoTokenizer

    import benchmarks.open_one_rec.one_rec_acc_test as accuracy

    messages = [
        {"role": "system", "content": "Preserve whitespace."},
        {"role": "user", "content": "\nquestion\n"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Next question"},
    ]
    data = {
        "video-42": {
            "prompt": "locally rendered prompt",
            "chat_messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "ground_truth": ground_truth,
        }
    }

    def post(_url: str, **kwargs: Any) -> Any:
        assert kwargs["json"]["messages"] == messages
        assert kwargs["json"]["chat_template_kwargs"] == {"enable_thinking": True}
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"reasoning": _SID_A, "content": content}}]},
        )

    reg = accuracy.TASK_REGISTRY["video"]
    config = {
        **reg.config,
        "evaluation_config": {"metrics": ["pass@1"], "k_values": [1], "evaluation_mode": "sid"},
    }
    monkeypatch.setitem(
        accuracy.TASK_REGISTRY,
        "video",
        SimpleNamespace(config=config, evaluator_class=reg.evaluator_class),
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    load = Mock(return_value=data)
    monkeypatch.setattr(accuracy, "DataLoaderWrapper", lambda **kw: SimpleNamespace(load_data=load))
    monkeypatch.setattr(accuracy.requests, "post", post)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accuracy",
            "--task-type",
            "video",
            "--endpoint",
            "/v1/chat/completions",
            "--n",
            "1",
            "--num-prompts",
            "1",
            "--chat-template-kwargs",
            '{"enable_thinking": true}',
            "--result-dir",
            str(tmp_path),
        ],
    )
    assert accuracy.main() == 0
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert report["total_samples"] == 1
    assert report["pass@1"] == expected_pass_at_1
    load.assert_called_once_with(task_name="video", split="test", sample_size=1)


def test_accuracy_keeps_truncated_inline_thinking_at_its_pid_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from transformers import AutoTokenizer

    import benchmarks.open_one_rec.one_rec_acc_test as accuracy
    from benchmarks.open_one_rec.tasks.recommendation.utils_by_pid import encode_sid

    samples = {
        "video-42": {"prompt": "question", "ground_truth": _SID_A, "metadata": {"answer_pid": [42]}}
    }
    mapping = {encode_sid(1, 2, 3): [{"pid": 42, "count_after_downsample": 1}]}
    (tmp_path / "sid2pid.json").write_text(json.dumps(mapping), encoding="utf-8")
    choices = [
        {"message": {"content": f"<think>maybe {_SID_A}"}},
        {"message": {"content": _SID_A}},
    ]
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"choices": choices})
    config = accuracy.TASK_REGISTRY["video"].config["evaluation_config"]
    monkeypatch.setitem(config, "evaluation_mode", "pid")
    monkeypatch.setitem(config, "k_values", [1, 2])
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(
        accuracy, "DataLoaderWrapper", lambda **kw: SimpleNamespace(load_data=lambda **kw: samples)
    )
    monkeypatch.setattr(accuracy.requests, "post", lambda *a, **kw: response)
    results = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accuracy",
            "--task-type",
            "video",
            "--endpoint",
            "/v1/chat/completions",
            "--n",
            "2",
            "--dataset-path",
            str(tmp_path),
            "--result-dir",
            str(results),
        ],
    )

    assert accuracy.main() == 0
    report = json.loads(next(results.glob("*.json")).read_text())
    assert report["total_samples"] == 1
    assert report["pid_pass@1"] == 0
    assert report["pid_pass@2"] == 1


@pytest.mark.parametrize(  # type: ignore[misc]
    ("task", "judge_method"),
    [("rec_reason", "_evaluate_reasoning"), ("item_understand", "_evaluate_wip")],
)
def test_accuracy_judges_receive_final_answers(
    monkeypatch: pytest.MonkeyPatch, task: str, judge_method: str
) -> None:
    from transformers import AutoTokenizer

    import benchmarks.open_one_rec.one_rec_acc_test as accuracy

    messages = {
        "both": {"reasoning": "working notes", "content": "final answer"},
        "reasoning-only": {"reasoning_content": "working notes", "content": None},
        "content-only": {"content": "plain answer"},
        "inline": {"content": "<think>the object could be red</think>It is blue"},
    }
    samples = {key: {"prompt": key, "ground_truth": "reference"} for key in messages}

    def post(_url: str, **kwargs: Any) -> Any:
        message = messages[kwargs["json"]["messages"][0]["content"]]
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"choices": [{"message": message}]}
        )

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(
        accuracy, "DataLoaderWrapper", lambda **kw: SimpleNamespace(load_data=lambda **kw: samples)
    )
    monkeypatch.setattr(accuracy.requests, "post", post)
    judge = Mock(return_value=({}, {}))
    monkeypatch.setattr(accuracy.TASK_REGISTRY[task].evaluator_class, judge_method, judge)
    monkeypatch.setattr(
        sys, "argv", ["accuracy", "--task-type", task, "--endpoint", "/v1/chat/completions"]
    )

    assert accuracy.main() == 0
    judge.assert_called_once()
    assert judge.call_args.kwargs["predictions"] == [
        "final answer",
        "",
        "plain answer",
        "It is blue",
    ]


@pytest.mark.parametrize(  # type: ignore[misc]
    "message,expected",
    [
        ({"reasoning": "a clue", "content": None}, ("a clue", "")),
        ({"reasoning_content": "old clue", "content": None}, ("old clue", "")),
        ({"reasoning": "A", "reasoning_content": "A", "content": "B"}, ("A", "B")),
        (
            {"reasoning": "new", "reasoning_content": "old", "content": " answer"},
            ("new", " answer"),
        ),
        ({"reasoning": None, "reasoning_content": "old", "content": " answer"}, ("old", " answer")),
        ({"reasoning": "", "reasoning_content": "old", "content": None}, ("old", "")),
        ({"content": "answer"}, ("", "answer")),
        ({"content": "\n  <think>notes</think> answer"}, ("notes", " answer")),
        ({"content": "<think>notes"}, ("notes", "")),
        ({"content": "Use <think>notes</think> here."}, ("", "Use <think>notes</think> here.")),
        (
            {
                "reasoning": " ",
                "reasoning_content": "separate",
                "content": "<think>example</think>",
            },
            ("separate", "<think>example</think>"),
        ),
        ({"content": None, "reasoning": None, "reasoning_content": ""}, None),
        ({"reasoning": 3, "content": "answer"}, None),
        ({"reasoning_content": [], "content": "answer"}, None),
        ({"content": {"text": "answer"}}, None),
    ],
)
def test_accuracy_http_preserves_answer_and_reasoning(
    monkeypatch: pytest.MonkeyPatch, message: dict[str, Any], expected: tuple[str, str] | None
) -> None:
    import benchmarks.open_one_rec.one_rec_acc_test as accuracy

    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"choices": [{"message": message}]},
    )
    monkeypatch.setattr(accuracy.requests, "post", lambda *a, **kw: response)
    if expected is None:
        with pytest.raises(accuracy.InferenceError, match="no text message|non-text"):
            accuracy.run_api_inference(["prompt"], "model", "http://localhost/unused", "EMPTY", {})
    else:
        outputs = accuracy.run_api_inference(
            ["prompt"], "model", "http://localhost/unused", "EMPTY", {}
        )
        completion = outputs[0].outputs[0]
        assert (completion.reasoning, completion.content) == expected


@pytest.mark.parametrize("num_prompts", ["0", "-1"])  # type: ignore[misc]
def test_accuracy_rejects_invalid_num_prompts_before_loading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], num_prompts: str
) -> None:
    from transformers import AutoTokenizer

    import benchmarks.open_one_rec.one_rec_acc_test as accuracy

    tokenizer = Mock()
    model = Mock()
    loader = Mock()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(accuracy, "GRLLM", model)
    monkeypatch.setattr(accuracy, "DataLoaderWrapper", loader)
    monkeypatch.setattr(
        sys,
        "argv",
        ["accuracy", "--task-type", "video", "--gr", "{}", "--num-prompts", num_prompts],
    )

    with pytest.raises(SystemExit) as exc:
        accuracy.main()
    assert exc.value.code == 2
    assert "--num-prompts" in capsys.readouterr().err
    tokenizer.assert_not_called()
    model.assert_not_called()
    loader.assert_not_called()


def test_accuracy_classification_requires_logprobs_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import benchmarks.open_one_rec.one_rec_acc_test as accuracy

    def unexpected_load(**kwargs: Any) -> Any:
        pytest.fail("classification attempted to load input without logprob support")

    monkeypatch.setattr(accuracy, "DataLoaderWrapper", unexpected_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accuracy",
            "--task-type",
            "label_pred",
            "--endpoint",
            "/v1/chat/completions",
            "--result-dir",
            str(tmp_path),
        ],
    )
    assert accuracy.main() == 1
    report = json.loads((tmp_path / "label_pred-failed.json").read_text())
    assert report["auc"] is None
    assert report["valid_samples"] == report["invalid_samples"] == 0
    assert report["auc_unavailable_reason"] == "classification_logprobs_unavailable"
