# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-untyped-def"

from copy import deepcopy

import pytest

from benchmarks.open_one_rec.gate_constrained_golden import (
    evaluate_constrained_golden,
)


def _fixtures():
    protocol = {
        "format_version": 1,
        "model": "model",
        "revision": "revision",
        "beam_width": 32,
        "temperature": 0.0,
        "evaluation_mode": "both",
        "execution": "online",
        "execution_variant": "graph-worker",
        "seed": 123,
        "selected_samples": 100,
        "sample_manifest_sha256": "samples",
        "prepared_manifest_sha256": "constraints",
    }
    constraint = {"mode": "triples", "domain": "video", "digest": "digest"}
    golden = {
        "version": 1,
        "description": "test golden",
        "policy": {
            "comparison": "candidate_greater_than_or_equal",
            "blocking_metrics": ["pass@32"],
            "diagnostic_metrics": ["position1_pass@32"],
        },
        "protocol": protocol,
        "tasks": {
            "video": {
                "source_sha256": "source",
                "constraint": constraint,
                "metrics": {"pass@32": 0.2, "position1_pass@32": 0.08},
            }
        },
    }
    candidate = {
        **{key: value for key, value in protocol.items() if key != "evaluation_mode"},
        "source_sha256": "source",
        "constraint": constraint,
        "invalid_samples": 0,
        "metrics": {
            "evaluation_mode": "both",
            "total_samples": 100,
            "pass@32": 0.2,
            "position1_pass@32": 0.07,
        },
    }
    return golden, {"video": candidate}


def test_constrained_golden_allows_diagnostic_metric_difference() -> None:
    golden, candidates = _fixtures()

    report = evaluate_constrained_golden(golden, candidates)

    assert report["passed"]
    metric = report["tasks"]["video"]["metrics"]["position1_pass@32"]
    assert metric["regressed"]
    assert not metric["blocking"]


def test_constrained_golden_blocks_primary_metric_regression() -> None:
    golden, candidates = _fixtures()
    candidates["video"]["metrics"]["pass@32"] = 0.19

    report = evaluate_constrained_golden(golden, candidates)

    assert not report["passed"]
    assert "blocking metric pass@32 regressed" in report["errors"][0]


def test_constrained_golden_allows_regression_within_blocking_tolerance() -> None:
    golden, candidates = _fixtures()
    golden["policy"]["blocking_absolute_tolerance"] = 0.02
    candidates["video"]["metrics"]["pass@32"] = 0.18

    report = evaluate_constrained_golden(golden, candidates)

    assert report["passed"]
    metric = report["tasks"]["video"]["metrics"]["pass@32"]
    assert metric["absolute_tolerance"] == 0.02
    assert not metric["regressed"]


def test_constrained_golden_uses_metric_specific_blocking_tolerance() -> None:
    golden, candidates = _fixtures()
    golden["policy"]["blocking_absolute_tolerance"] = 0.02
    golden["policy"]["blocking_metric_absolute_tolerances"] = {"pass@32": 0.03}
    candidates["video"]["metrics"]["pass@32"] = 0.17

    report = evaluate_constrained_golden(golden, candidates)

    assert report["passed"]
    assert report["blocking_absolute_tolerance"] == 0.02
    assert report["blocking_metric_absolute_tolerances"] == {"pass@32": 0.03}
    metric = report["tasks"]["video"]["metrics"]["pass@32"]
    assert metric["absolute_tolerance"] == 0.03
    assert not metric["regressed"]


def test_constrained_golden_rejects_tolerance_for_nonblocking_metric() -> None:
    golden, candidates = _fixtures()
    golden["policy"]["blocking_metric_absolute_tolerances"] = {"position1_pass@32": 0.03}

    with pytest.raises(ValueError, match="contains non-blocking metrics"):
        evaluate_constrained_golden(golden, candidates)


def test_constrained_golden_ignores_runtime_identity_but_blocks_contract_changes() -> None:
    golden, candidates = _fixtures()
    candidates = deepcopy(candidates)
    candidates["video"]["model"] = "/local/cache/OneRec-1.7B"
    candidates["video"]["revision"] = None
    candidates["video"]["prepared_manifest_sha256"] = "local-constraints"
    candidates["video"]["constraint"] = {"mode": "none"}
    candidates["video"]["invalid_samples"] = 1

    report = evaluate_constrained_golden(golden, candidates)

    assert not report["passed"]
    assert not any("model" in error for error in report["errors"])
    assert not any("revision" in error for error in report["errors"])
    assert not any("prepared_manifest" in error for error in report["errors"])
    assert any("constraint" in error for error in report["errors"])
    assert any("invalid_samples" in error for error in report["errors"])


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf"), -0.1, 1.1, True],
)
def test_constrained_golden_blocks_invalid_candidate_metrics(
    invalid_value: float | bool,
) -> None:
    golden, candidates = _fixtures()
    candidates["video"]["metrics"]["pass@32"] = invalid_value

    report = evaluate_constrained_golden(golden, candidates)

    assert not report["passed"]
    assert "must be a finite number in [0, 1]" in report["errors"][0]


def test_constrained_golden_rejects_invalid_baseline_metrics() -> None:
    golden, candidates = _fixtures()
    golden["tasks"]["video"]["metrics"]["pass@32"] = float("nan")

    with pytest.raises(ValueError, match="invalid constrained golden metric"):
        evaluate_constrained_golden(golden, candidates)


def test_constrained_golden_rejects_boolean_invalid_sample_count() -> None:
    golden, candidates = _fixtures()
    candidates["video"]["invalid_samples"] = False

    report = evaluate_constrained_golden(golden, candidates)

    assert not report["passed"]
    assert "expected integer 0" in report["errors"][0]


def test_constrained_golden_filters_metrics_for_sid_only_mode() -> None:
    golden, candidates = _fixtures()
    golden["policy"]["blocking_metrics"].append("pid_pass@32")
    golden["tasks"]["video"]["metrics"]["pid_pass@32"] = 0.2
    candidates["video"]["metrics"]["evaluation_mode"] = "sid"

    report = evaluate_constrained_golden(
        golden,
        candidates,
        evaluation_mode="sid",
    )

    assert report["passed"]
    assert report["evaluation_mode"] == "sid"
    assert report["blocking_metrics"] == ["pass@32"]
