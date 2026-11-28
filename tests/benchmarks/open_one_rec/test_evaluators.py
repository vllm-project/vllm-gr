# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Check OpenOneRec label-probability evaluation."""

from pathlib import Path

import pytest

from benchmarks.open_one_rec.tasks.label_pred.evaluator import LabelPredEvaluator

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
