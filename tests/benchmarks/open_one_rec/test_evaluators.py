# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Check OpenOneRec probability and Judge evaluation."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.open_one_rec.base_evaluator import EvaluationError
from benchmarks.open_one_rec.tasks.item_understand import evaluator as item_evaluator
from benchmarks.open_one_rec.tasks.item_understand import utils as wip_utils
from benchmarks.open_one_rec.tasks.item_understand.utils import evaluate_wip
from benchmarks.open_one_rec.tasks.label_pred.evaluator import LabelPredEvaluator
from benchmarks.open_one_rec.tasks.rec_reason import evaluator as reason_evaluator
from benchmarks.open_one_rec.tasks.rec_reason import utils as reason_utils
from benchmarks.open_one_rec.tasks.rec_reason.utils import evaluate_reasoning

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("preds", "valid", "auc", "reason"),
    [
        (["\u662f", "\u5426"], 0, None, "no_valid_probabilities"),
        (
            ['{"\u662f": 0.9, "\u5426": 0.1}', '{"\u662f": NaN, "\u5426": 1}'],
            1,
            None,
            "single_class",
        ),
        (
            ['{"\u662f": 0.9, "\u5426": 0.1}', '{"\u662f": -0.1, "\u5426": 1}'],
            1,
            None,
            "single_class",
        ),
        (['{"\u662f": 0.9, "\u5426": 0.1}', '{"\u662f": 0.1, "\u5426": 0.9}'], 2, 1.0, None),
    ],
)
def test_auc_uses_usable_probabilities(
    preds: list[str], valid: int, auc: float | None, reason: str | None, tmp_path: Path
) -> None:
    samples = {
        str(i): {"ground_truth": label, "generations": [pred]}
        for i, (label, pred) in enumerate(zip(["\u662f", "\u5426"], preds, strict=True))
    }
    for debug in (False, True):
        metrics, details = LabelPredEvaluator(
            samples, debug=debug, predictions_dir=str(tmp_path), cached_metrics={"auc": 0.5}
        ).evaluate()
        assert metrics["auc"] == auc
        assert metrics["total_samples"] == 2
        assert metrics["valid_samples"] == len(details) == valid
        assert metrics["invalid_samples"] == 2 - valid
        assert metrics.get("auc_unavailable_reason") == reason


def test_reason_judge_recomputes_a_changed_prediction(tmp_path: Path) -> None:
    calls: list[str] = []

    def score(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps({"llm_score": 5 if "prediction first" in prompt else 1})

    for prediction, score_expected in [("prediction first", 5), ("prediction second", 1)]:
        metrics, _ = evaluate_reasoning(
            {"42": prediction},
            {"42": "reference"},
            SimpleNamespace(generate=score),
            model_name="judge",
            save_dir=str(tmp_path),
        )
        assert metrics["llm_score"] == score_expected
    assert len(calls) == 2


@pytest.mark.parametrize("empty", ["", " \n ", "<think>private reasoning</think>"])  # type: ignore[misc]
def test_reason_judge_fails_when_a_selected_prediction_is_empty(empty: str) -> None:
    predictions = {"01-scored": "prediction", "02-empty": empty}
    references = {key: "reference" for key in predictions}
    judge = SimpleNamespace(generate=lambda _: '{"llm_score": 5}')

    with pytest.raises(EvaluationError, match="02-empty.*empty prediction"):
        evaluate_reasoning(predictions, references, judge)

    metrics, per_sample = evaluate_reasoning(predictions, references, judge, max_samples=1)
    assert metrics["llm_eval_num_samples"] == 1
    assert set(per_sample) == {"01-scored"}


def test_reason_judge_rejects_incomplete_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    predictions = {"01-scored": "first", "02-unscored": "second"}
    references = {key: "reference" for key in predictions}
    monkeypatch.setattr(
        reason_utils, "evaluate_batch", lambda *a, **kw: ({"01-scored": {"llm_score": 5}}, {})
    )
    with pytest.raises(EvaluationError, match="02-unscored: missing Judge score"):
        evaluate_reasoning(predictions, references, object())


@pytest.mark.parametrize("invalid", [True, 0, 6, 3.5, float("nan"), "5"])  # type: ignore[misc]
def test_reason_judge_rejects_invalid_scores(invalid: Any) -> None:
    judge = SimpleNamespace(generate=lambda _: json.dumps({"llm_score": invalid}))
    with pytest.raises(EvaluationError, match="42: Invalid Judge score"):
        evaluate_reasoning({"42": "prediction"}, {"42": "reference"}, judge)


@pytest.mark.parametrize("invalid", [True, 0, 6, 3.5, float("nan"), "4", None])  # type: ignore[misc]
def test_reason_metrics_reject_invalid_sample_scores(invalid: Any) -> None:
    with pytest.raises(EvaluationError, match="42: Invalid Judge score"):
        reason_utils.calculate_metrics({"41": {"llm_score": 4}, "42": {"llm_score": invalid}})


def test_reason_judge_averages_integer_sample_scores() -> None:
    scores = iter((3, 4))
    judge = SimpleNamespace(generate=lambda _: json.dumps({"llm_score": next(scores)}))
    metrics, samples = evaluate_reasoning(
        {"41": "prediction", "42": "prediction"},
        {"41": "reference", "42": "reference"},
        judge,
        max_workers=1,
    )
    assert metrics["llm_score"] == 3.5
    assert metrics["llm_eval_num_samples"] == len(samples) == 2


@pytest.mark.parametrize("use_async", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "response_kind,expected_error",
    [
        ("score", None),
        ("no choices", "Judge response contains no choices"),
        ("empty content", "Judge response contains no text content"),
        ("http error", "Error code: 400"),
    ],
)
def test_reason_judge_factory_uses_configured_http_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_async: bool,
    response_kind: str,
    expected_error: str | None,
) -> None:
    monkeypatch.setenv("no_proxy", "*")
    requests: list[dict[str, Any]] = []

    class JudgeServer(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "payload": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                }
            )
            if response_kind == "http error":
                self.send_response(400)
                response: dict[str, Any] = {"error": {"message": "rejected"}}
            else:
                self.send_response(200)
                choices = (
                    []
                    if response_kind == "no choices"
                    else [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None
                                if response_kind == "empty content"
                                else '{"llm_score": 4}',
                            },
                            "finish_reason": "stop",
                        }
                    ]
                )
                response = {
                    "id": "judge-completion",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "Qwen/Qwen3-8B",
                    "choices": choices,
                }
            body = json.dumps(response).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: Any) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), JudgeServer) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            evaluator = reason_evaluator.RecoReasonEvaluator(
                {"42": {"ground_truth": "reference", "generations": ["prediction"]}},
                task_name="rec_reason",
                predictions_dir=str(tmp_path),
                task_config={
                    "evaluation_config": {
                        "llm_eval_enabled": True,
                        "llm_judge_model": "Qwen/Qwen3-8B",
                        "vllm_endpoint": f"http://127.0.0.1:{server.server_port}/v1",
                        "vllm_api_key": "test-judge-key",
                        "vllm_async": use_async,
                    }
                },
            )
            if expected_error is not None:
                with pytest.raises(EvaluationError, match=expected_error):
                    evaluator.evaluate()
                assert len(requests) == 1
            else:
                # Repeated evaluations also exercise a fresh SDK client and async
                # event loop in each scoring worker.
                for _ in range(2):
                    metrics, samples = evaluator.evaluate()
                    assert metrics["llm_score"] == 4
                    assert metrics["llm_eval_num_samples"] == 1
                    assert samples["42"]["llm_score"] == 4
                    diagnostics = tmp_path / "llm_eval_Qwen%2FQwen3-8B.json"
                    saved = json.loads(diagnostics.read_text(encoding="utf-8"))
                    assert saved["42"]["eval_result"]["llm_score"] == 4
                assert len(requests) == 2
        finally:
            server.shutdown()
            thread.join()

    for request in requests:
        assert request["path"] == "/v1/chat/completions"
        assert request["authorization"] == "Bearer test-judge-key"
        payload = request["payload"]
        assert payload["model"] == "Qwen/Qwen3-8B"
        assert payload["messages"][0]["role"] == "user"
        assert "prediction" in payload["messages"][0]["content"]


def test_reason_judge_rejects_client_without_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = reason_evaluator.RecoReasonEvaluator(
        {},
        task_name="rec_reason",
        task_config={"evaluation_config": {"llm_eval_enabled": True}},
        cached_metrics={"llm_score": 5},
    )
    monkeypatch.setattr(evaluator, "_get_vllm_client", lambda _: object())
    with pytest.raises(EvaluationError, match="must provide generate"):
        evaluator.evaluate()


def test_wip_judge_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "api", None)
    evaluator = item_evaluator.ItemUnderstandEvaluator(
        {},
        task_name="item_understand",
        task_config={"evaluation_config": {"wip_enabled": True}},
        cached_metrics={"macro_wip_double_weighted_f1": 1.0},
    )
    with pytest.raises(EvaluationError, match="WIP Judge dependencies"):
        evaluator.evaluate()


def test_wip_judge_recomputes_reference_and_prediction(tmp_path: Path) -> None:
    prompts: list[str] = []

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        for text in ("reference first", "reference second"):
            if text in prompt:
                return json.dumps({"wips": [{"info_point": text, "importance_score": 3}]})
        return '{"wips": []}'

    for suffix in ("first", "second"):
        metrics, _ = evaluate_wip(
            {"42": f"prediction {suffix}"},
            {"42": f"reference {suffix}"},
            SimpleNamespace(generate=generate),
            save_dir=str(tmp_path),
            model_name="judge",
        )
        assert metrics["wip_num_samples"] == 1
        assert metrics["macro_wip_double_weighted_f1"] == 0
    assert len(prompts) == 4
    for prompt, expected in zip(
        prompts,
        ["reference first", "prediction first", "reference second", "prediction second"],
        strict=True,
    ):
        assert expected in prompt


def test_wip_judge_rejects_malformed_reference() -> None:
    judge = SimpleNamespace(generate=lambda _: '{"wips": [{"importance_score": 99}]}')
    with pytest.raises(EvaluationError, match="Reference WIP extraction.*42"):
        evaluate_wip({"42": "prediction"}, {"42": "reference"}, judge)


@pytest.mark.parametrize(  # type: ignore[misc]
    "reference,matched,unmatched_reference,unmatched_model,error",
    [
        ("AB", "A", "", "", "reference WIP coverage mismatch: 1 missing"),
        ("AB", "A", "B", "C", "model WIP coverage mismatch: 0 missing, 1 unexpected"),
        ("AB", "AA", "B", "", "reference WIP coverage mismatch: 0 missing, 1 unexpected"),
        ("AA", "AA", "", "", "model WIP coverage mismatch: 0 missing, 1 unexpected"),
        ("AB", "A", "B", "", None),
        ("AA", "A", "A", "", None),
    ],
)
def test_wip_judge_matching_covers_each_occurrence(
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
    matched: str,
    unmatched_reference: str,
    unmatched_model: str,
    error: str | None,
) -> None:
    def point(text: str) -> dict[str, Any]:
        return {"info_point": text, "importance_score": 5}

    responses = iter(
        [
            {"wips": [point(item) for item in reference]},
            {"wips": [point("A")]},
            {
                "matches": [{"gt_wip": point(text), "model_wip": point("A")} for text in matched],
                "unmatched_gt_wips": [point(item) for item in unmatched_reference],
                "unmatched_model_wips": [point(item) for item in unmatched_model],
            },
        ]
    )

    def score_matches(results: dict[str, Any], *_: Any) -> None:
        for match in results["42"]["matches"]:
            match["match_quality"] = 1.0

    monkeypatch.setattr(wip_utils, "_compute_bertscore", score_matches)
    judge = SimpleNamespace(generate=lambda _: json.dumps(next(responses)))
    if error is not None:
        with pytest.raises(EvaluationError, match=f"42: {error}"):
            evaluate_wip({"42": "prediction"}, {"42": "reference"}, judge, max_workers=1)
    else:
        metrics, per_sample = evaluate_wip(
            {"42": "prediction"}, {"42": "reference"}, judge, max_workers=1
        )
        assert metrics["macro_wip_unweighted_f1"] == pytest.approx(2 / 3)
        assert per_sample["42"]["wip_unweighted_f1"] == pytest.approx(2 / 3)
