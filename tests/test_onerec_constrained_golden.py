# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-untyped-def"

from copy import deepcopy
from typing import Any

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


def _summary(task: str, **overrides) -> dict[str, Any]:
    digest = "sha256:" + "a" * 64
    summary = {
        "format_version": 1,
        "task": task,
        "model": "/cache/OneRec-1.7B",
        "revision": None,
        "beam_width": 32,
        "temperature": 0.0,
        "seed": 123,
        "execution": "offline",
        "execution_variant": "offline-v1-graph",
        "max_model_len": 32768,
        "config_env": {
            "VLLM_USE_V2_MODEL_RUNNER": None,
            "VLLM_ENABLE_PREFILL_CUDAGRAPH": None,
            "VLLM_GR_PREFILL_GRAPH_KV_BOUND": None,
            "VLLM_PLUGINS": None,
            "PYTHONHASHSEED": "123",
        },
        "selected_samples": 100,
        "invalid_samples": 0,
        "sample_manifest_sha256": "samples",
        "prepared_manifest_sha256": "prepared",
        "source_sha256": "source",
        "constraint": {"domain": "video", "mode": "artifact", "digest": digest},
        "metrics": {
            "evaluation_mode": "both",
            "total_samples": 100,
            "pass@1": 0.2,
            "pass@32": 0.4,
            "recall@32": 0.5,
            "pid_pass@1": 0.2,
            "pid_pass@32": 0.4,
            "pid_recall@32": 0.5,
            "position1_pass@1": 0.1,
            "position1_pass@32": 0.2,
            "recall@1": 0.2,
            "pid_position1_pass@1": 0.1,
            "pid_position1_pass@32": 0.2,
            "pid_recall@1": 0.2,
        },
    }
    summary.update(overrides)
    return summary


def test_capture_builds_golden_from_consistent_summaries() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
        evaluate_constrained_golden,
    )

    manifest = capture_constrained_golden(
        {"video": _summary("video"), "ad": _summary("ad")},
        description="test capture",
        model_label="OpenOneRec/OneRec-1.7B",
    )

    assert manifest["version"] == 1
    assert manifest["protocol"]["execution"] == "offline"
    assert manifest["protocol"]["execution_variant"] == "offline-v1-graph"
    assert manifest["protocol"]["model"] == "OpenOneRec/OneRec-1.7B"
    assert set(manifest["tasks"]) == {"video", "ad"}
    assert manifest["tasks"]["video"]["metrics"]["pass@32"] == 0.4
    assert manifest["policy"]["comparison"] == "candidate_greater_than_or_equal"

    # Round trip: gating a fresh identical run against the captured golden
    # must pass with zero regressions.
    candidates = {task: _summary(task) for task in ("video", "ad")}
    report = evaluate_constrained_golden(manifest, candidates)
    assert report["passed"], report["errors"]
    assert report["aggregate_metrics"]["pass@32"]["delta"] == 0.0


def _summary_with_mode(task: str, mode: str) -> dict[str, Any]:
    summary = _summary(task)
    metrics = {
        key: value
        for key, value in summary["metrics"].items()
        if "@" not in key or key.startswith("pid_") == (mode == "pid")
    }
    metrics["evaluation_mode"] = mode
    return {**summary, "metrics": metrics}


def test_capture_and_gate_round_trip_sid_only() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
    )

    summary = _summary_with_mode("video", "sid")
    manifest = capture_constrained_golden({"video": summary}, description="sid only")
    assert manifest["protocol"]["evaluation_mode"] == "sid"
    assert "pass@32" in manifest["tasks"]["video"]["metrics"]
    assert "pid_pass@32" not in manifest["tasks"]["video"]["metrics"]

    report = evaluate_constrained_golden(manifest, {"video": summary})
    assert report["passed"], report["errors"]


def test_capture_and_gate_round_trip_pid_only() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
    )

    summary = _summary_with_mode("video", "pid")
    manifest = capture_constrained_golden({"video": summary}, description="pid only")
    assert manifest["protocol"]["evaluation_mode"] == "pid"
    assert "pid_pass@32" in manifest["tasks"]["video"]["metrics"]
    assert "pass@32" not in manifest["tasks"]["video"]["metrics"]

    report = evaluate_constrained_golden(manifest, {"video": summary})
    assert report["passed"], report["errors"]


def test_gate_projects_both_golden_for_sid_candidate() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
    )

    both_samples = {str(i): {"pass@1": i % 2 == 0, "pid_pass@1": i % 2 == 0} for i in range(100)}
    manifest = capture_constrained_golden(
        {"video": _summary("video")},
        description="both-mode golden",
        samples={"video": both_samples},
    )
    sid_samples = {str(i): {"pass@1": i % 2 == 0} for i in range(100)}

    report = evaluate_constrained_golden(
        manifest,
        {"video": _summary_with_mode("video", "sid")},
        evaluation_mode="sid",
        candidate_samples={"video": sid_samples},
    )
    assert report["passed"], report["errors"]


def test_capture_rejects_mixed_evaluation_modes() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
    )

    with pytest.raises(ValueError, match="disagree on evaluation_mode"):
        capture_constrained_golden(
            {
                "video": _summary_with_mode("video", "sid"),
                "ad": _summary_with_mode("ad", "pid"),
            },
            description="mixed modes",
        )


def test_capture_and_gate_allow_per_task_sample_counts() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
    )

    video = _summary("video")
    ad_metrics = dict(video["metrics"])
    ad_metrics["total_samples"] = 25
    ad = _summary("ad", selected_samples=25, metrics=ad_metrics)
    samples = {
        "video": {str(i): {"pass@1": i % 2 == 0} for i in range(100)},
        "ad": {str(i): {"pass@1": i % 2 == 0} for i in range(25)},
    }
    manifest = capture_constrained_golden(
        {"video": video, "ad": ad},
        description="per-task counts",
        samples=samples,
    )
    assert manifest["tasks"]["video"]["selected_samples"] == 100
    assert manifest["tasks"]["ad"]["selected_samples"] == 25

    report = evaluate_constrained_golden(
        manifest, {"video": video, "ad": ad}, candidate_samples=samples
    )
    assert report["passed"], report["errors"]


def test_capture_rejects_mixed_protocol_runs() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
    )

    with pytest.raises(ValueError, match="execution_variant.*differs"):
        capture_constrained_golden(
            {
                "video": _summary("video"),
                "ad": _summary("ad", execution_variant="graph-worker"),
            },
            description="mixed",
        )


def test_capture_records_runtime_config_but_does_not_gate_it() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
        evaluate_constrained_golden,
    )

    manifest = capture_constrained_golden(
        {"video": _summary("video")},
        description="test capture",
    )
    # Runtime parameters are recorded for provenance ...
    assert manifest["protocol"]["max_model_len"] == 32768
    assert manifest["protocol"]["config_env"]["PYTHONHASHSEED"] == "123"

    # ... yet one golden stays reusable across a wide range of them.
    changed = _summary("video", max_model_len=16384)
    changed["config_env"] = {**changed["config_env"], "VLLM_ENABLE_PREFILL_CUDAGRAPH": "0"}
    report = evaluate_constrained_golden(manifest, {"video": changed})
    assert report["passed"], report["errors"]


def test_constrained_golden_blocks_measurement_semantics_change() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
        evaluate_constrained_golden,
    )

    manifest = capture_constrained_golden(
        {"video": _summary("video")},
        description="test capture",
    )
    # Only the core measurement semantics are hard-gated.
    for overrides, field in (
        ({"beam_width": 16}, "beam_width"),
        ({"temperature": 0.7}, "temperature"),
        ({"sample_manifest_sha256": "other"}, "sample_manifest_sha256"),
        ({"source_sha256": "other"}, "source_sha256"),
    ):
        report = evaluate_constrained_golden(manifest, {"video": _summary("video", **overrides)})
        assert not report["passed"], field
        assert any(field in error for error in report["errors"]), (field, report["errors"])


def _golden_with_samples():
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
    )

    # 50 samples pass@1=True, 50 pass@1=False.
    golden_samples = {str(i): {"pass@1": i % 2 == 0} for i in range(100)}
    manifest = capture_constrained_golden(
        {"video": _summary("video")},
        description="test capture",
        samples={"video": golden_samples},
    )
    return manifest, golden_samples


def _candidate_samples(golden_samples, *, true_to_false=0, false_to_true=0):
    candidate = {
        sample_id: {"pass@1": value["pass@1"]} for sample_id, value in golden_samples.items()
    }
    true_ids = [i for i in range(100) if i % 2 == 0]
    false_ids = [i for i in range(100) if i % 2 == 1]
    for index in true_ids[:true_to_false]:
        candidate[str(index)]["pass@1"] = False
    for index in false_ids[:false_to_true]:
        candidate[str(index)]["pass@1"] = True
    return candidate


def test_capture_and_gate_freeze_per_sample_metrics() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        evaluate_constrained_golden,
    )

    manifest, golden_samples = _golden_with_samples()
    assert manifest["tasks"]["video"]["samples"] == golden_samples

    report = evaluate_constrained_golden(
        manifest,
        {"video": _summary("video")},
        candidate_samples={"video": dict(golden_samples)},
    )

    assert report["passed"], report["errors"]
    assert report["tasks"]["video"]["per_sample"]["common_samples"] == 100


def test_per_sample_stats_tolerate_benign_symmetric_flips() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        evaluate_constrained_golden,
    )

    manifest, golden_samples = _golden_with_samples()
    candidate = _candidate_samples(golden_samples, true_to_false=5, false_to_true=5)

    report = evaluate_constrained_golden(
        manifest,
        {"video": _summary("video")},
        candidate_samples={"video": candidate},
    )

    assert report["passed"], report["errors"]
    per_sample = report["tasks"]["video"]["per_sample"]
    assert per_sample["significant_regressions"] == []


def test_per_sample_stats_flag_one_sided_regression() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        evaluate_constrained_golden,
    )

    manifest, golden_samples = _golden_with_samples()
    candidate = _candidate_samples(golden_samples, true_to_false=40)

    report = evaluate_constrained_golden(
        manifest,
        {"video": _summary("video")},
        candidate_samples={"video": candidate},
    )

    assert not report["passed"]
    assert any("per-sample metric family pass regressed" in error for error in report["errors"])
    assert report["tasks"]["video"]["per_sample"]["significant_regressions"] == ["pass"]


def test_per_sample_family_catches_mid_range_depth_shift() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
        evaluate_constrained_golden,
    )

    def depth(rank: int) -> dict[str, bool]:
        return {f"pass@{k}": k >= rank for k in range(1, 33)}

    # Even ids: the correct answer moves from rank 10 to rank 20 in the
    # candidate. pass@1 stays False and pass@32 stays True, so a gate that only
    # looked at the {1,32} endpoints would pass; the family aggregate must catch
    # the mid-range regression.
    golden_samples = {str(i): depth(10 if i % 2 == 0 else 5) for i in range(100)}
    candidate = {str(i): depth(20 if i % 2 == 0 else 5) for i in range(100)}
    manifest = capture_constrained_golden(
        {"video": _summary("video")},
        description="mid-range depth shift",
        samples={"video": golden_samples},
    )

    report = evaluate_constrained_golden(
        manifest,
        {"video": _summary("video")},
        candidate_samples={"video": candidate},
    )

    assert not report["passed"], report["errors"]
    assert report["tasks"]["video"]["per_sample"]["significant_regressions"] == ["pass"]


def test_per_sample_reports_per_case_three_way() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
        evaluate_constrained_golden,
    )

    def depth(rank: int) -> dict[str, bool]:
        return {f"pass@{k}": k >= rank for k in range(1, 33)}

    # Even ids move from depth 10 to 20 (worse); odd ids are untouched (same).
    golden_samples = {str(i): depth(10 if i % 2 == 0 else 5) for i in range(100)}
    candidate = {str(i): depth(20 if i % 2 == 0 else 5) for i in range(100)}
    manifest = capture_constrained_golden(
        {"video": _summary("video")},
        description="per-case three-way",
        samples={"video": golden_samples},
    )

    report = evaluate_constrained_golden(
        manifest,
        {"video": _summary("video")},
        candidate_samples={"video": candidate},
    )

    per_case = report["tasks"]["video"]["per_sample"]["per_case"]
    assert per_case["0"]["pass"] == "worse"
    assert per_case["1"]["pass"] == "same"
    assert per_case["2"]["pass"] == "worse"


def test_per_sample_rejects_empty_candidate_samples() -> None:
    manifest, _ = _golden_with_samples()
    with pytest.raises(ValueError, match="per-sample metrics are empty"):
        evaluate_constrained_golden(
            manifest,
            {"video": _summary("video")},
            candidate_samples={"video": {}},
        )


def test_per_sample_rejects_mismatched_sample_ids() -> None:
    manifest, golden_samples = _golden_with_samples()
    candidate = _candidate_samples(golden_samples)
    candidate.pop("0")
    with pytest.raises(ValueError, match="sample ids differ"):
        evaluate_constrained_golden(
            manifest,
            {"video": _summary("video")},
            candidate_samples={"video": candidate},
        )


def test_per_sample_requires_candidate_samples_when_golden_has_them() -> None:
    manifest, _ = _golden_with_samples()
    report = evaluate_constrained_golden(manifest, {"video": _summary("video")})
    assert not report["passed"]
    assert any("candidate per-sample metrics are missing" in error for error in report["errors"])


def test_load_sample_metrics_rejects_duplicates(tmp_path) -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import _load_sample_metrics

    path = tmp_path / "video.jsonl"
    path.write_text(
        '{"sample_id": "0", "metrics": {"pass@1": true}}\n'
        '{"sample_id": "0", "metrics": {"pass@1": true}}\n'
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        _load_sample_metrics(path)


def test_capture_rejects_invalid_samples() -> None:
    from benchmarks.open_one_rec.gate_constrained_golden import (
        capture_constrained_golden,
    )

    with pytest.raises(ValueError, match="zero invalid samples"):
        capture_constrained_golden(
            {"video": _summary("video", invalid_samples=3)},
            description="dirty",
        )
