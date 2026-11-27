# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise the Recall diagnostic comparison with public synthetic records."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from benchmarks.beam_search.compare_recall_diagnostics import compare

pytestmark = [pytest.mark.l1, pytest.mark.cpu_test]


def sample(sample_id: str, *beams: tuple[int, float | None]) -> dict[str, Any]:
    prompt = json.dumps([ord(sample_id[0])], separators=(",", ":")).encode()
    return {
        "sample_id": sample_id,
        "prompt_token_count": 1,
        "prompt_token_sha256": hashlib.sha256(prompt).hexdigest(),
        "beams": [{"token_ids": [sid, 10, 20], "score": score} for sid, score in beams],
    }


@pytest.fixture
def diagnostic_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = {
        "schema_version": 1,
        "task": "video",
        "token_hash_format": "sha256 of compact JSON integer array",
        "artifact_digest": "sha256:catalog",
        "tokenizer_digest": "sha256:tokenizer",
    }
    reference = {
        **metadata,
        "commit": "reference",
        "batches": [["a", "b"], ["c", "d"]],
        "samples": [
            sample("a", (1, -1.0), (2, -2.0)),
            sample("b", (3, -1.0), (4, -2.0)),
            sample("c", (5, None), (6, -3.0)),
            sample("d"),
        ],
    }
    candidate = {
        **metadata,
        "commit": "candidate",
        "batches": [["a", "b"], ["d"], ["c"]],
        "samples": [
            sample("d"),
            sample("c", (5, -7.0), (7, -4.0)),
            sample("b", (4, -1.0), (3, -2.0)),
            sample("a", (1, -1.0), (2, -2.0)),
        ],
    }
    return reference, candidate


def test_separates_reordering_and_replacements_with_matched_scores_and_batches(
    diagnostic_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    result = compare(*diagnostic_pair, details=True)
    assert result["rankings"] == {
        "samples": 4,
        "exact_order": 2,
        "exact_order_rate": 0.5,
        "same_set": 3,
        "same_set_rate": 0.75,
        "reordered_only": 1,
        "changed_set": 1,
        "top1_equal": 2,
        "top1_compared": 3,
        "mean_shared_candidates": 1.25,
        "removed_candidates": 1,
        "added_candidates": 1,
    }
    assert result["batches"]["submission_order_equal"] is False
    assert result["batches"]["same"]["reordered_only"] == 1
    assert result["batches"]["changed"]["changed_set"] == 1
    assert result["scores"] == {"pairs": 4, "p95_absolute_delta": 1.0, "max_absolute_delta": 1.0}
    assert result["changed_metadata"] == ["commit"]
    assert [row["sample_id"] for row in result["samples"]] == ["a", "b", "c", "d"]
    changed, empty = result["samples"][2:]
    assert changed["reference_only"] == [(6, 10, 20)]
    assert changed["candidate_only"] == [(7, 10, 20)]
    assert changed["score_pairs"] == 0
    assert changed["max_abs_score_delta"] is None
    assert empty["top1_equal"] is None


def test_top_k_truncates_both_lists(
    diagnostic_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    result = compare(*diagnostic_pair, top_k=1)
    assert result["top_k"] == 1
    assert result["rankings"]["exact_order"] == 3
    assert result["rankings"]["reordered_only"] == 0
    assert result["rankings"]["changed_set"] == 1
    assert "samples" not in result


def test_legacy_files_report_unknown_batch_context_and_missing_scores(
    diagnostic_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    reference, candidate = diagnostic_pair
    del reference["batches"]
    for row in reference["samples"]:
        for beam in row["beams"]:
            beam["score"] = None
    result = compare(reference, candidate)
    assert result["batches"]["submission_order_equal"] is None
    assert result["batches"]["unrecorded"]["samples"] == 4
    assert result["beam_search_params_status"] == "unrecorded"
    assert result["rankings"]["exact_order"] == 2
    assert result["scores"] == {"pairs": 0, "p95_absolute_delta": None, "max_absolute_delta": None}


@pytest.mark.parametrize(
    ("ref_temperature", "cand_temperature", "status"),
    [
        (0.6, 0.6, "same"),
        (0.6, 1.0, "changed"),
        (None, 0.6, "unrecorded"),
        (0.6, None, "unrecorded"),
    ],
)
def test_reports_search_parameter_changes_only_when_both_are_recorded(
    diagnostic_pair: tuple[dict[str, Any], dict[str, Any]],
    ref_temperature: float | None,
    cand_temperature: float | None,
    status: str,
) -> None:
    reference, _ = diagnostic_pair
    candidate = deepcopy(reference)
    for run, temperature in ((reference, ref_temperature), (candidate, cand_temperature)):
        if temperature is not None:
            run["beam_search_params"] = {"beam_width": 2, "temperature": temperature}
    candidate["samples"][0]["beams"][0]["score"] = -0.5

    result = compare(reference, candidate)
    assert result["beam_search_params_status"] == status
    assert result["changed_metadata"] == (["beam_search_params"] if status == "changed" else [])
    assert result["rankings"]["exact_order"] == 4
    assert result["scores"]["max_absolute_delta"] == 0.5


def test_distinguishes_batch_submission_order_and_member_order(
    diagnostic_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    reference, _ = diagnostic_pair
    candidate = deepcopy(reference)
    candidate["batches"] = [["c", "d"], ["a", "b"]]
    result = compare(reference, candidate)
    assert result["batches"]["submission_order_equal"] is False
    assert result["batches"]["same"]["samples"] == 4
    candidate["batches"][0].reverse()
    result = compare(reference, candidate)
    assert result["batches"]["same"]["samples"] == 2
    assert result["batches"]["changed"]["samples"] == 2


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", 2, "schema_version 1"),
        ("schema_version", True, "schema_version 1"),
        ("task", "product", "task mismatch"),
        ("tokenizer_digest", "sha256:other", "tokenizer_digest mismatch"),
        ("token_hash_format", "other", "token_hash_format mismatch"),
        ("batches", [["a", "b"], ["c", "c"]], "each sample exactly once"),
        ("batches", [["a", "b"], ["c", "missing"]], "each sample exactly once"),
        ("batches", [["a", "b"], [], ["c", "d"]], "non-empty lists"),
        ("batches", ["a", "b", "c", "d"], "non-empty lists"),
        ("samples", [], "non-empty samples"),
    ],
)
def test_rejects_invalid_artifact_or_incompatible_metadata(
    diagnostic_pair: tuple[dict[str, Any], dict[str, Any]],
    field: str,
    value: Any,
    error: str,
) -> None:
    reference, candidate = diagnostic_pair
    candidate[field] = value
    with pytest.raises(ValueError, match=error):
        compare(reference, candidate)


def test_rejects_mismatched_sample_ids_and_prompt_hashes(
    diagnostic_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    reference, candidate = diagnostic_pair
    del candidate["batches"]
    candidate["samples"][0] = sample("e")
    with pytest.raises(ValueError, match="sample IDs differ"):
        compare(reference, candidate)
    candidate["samples"][0] = sample("d")
    candidate["samples"][0]["prompt_token_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="prompt_token_sha256 mismatch"):
        compare(reference, candidate)


def test_rejects_duplicate_samples_and_sids(
    diagnostic_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    reference, candidate = diagnostic_pair
    candidate["samples"].append(candidate["samples"][0])
    with pytest.raises(ValueError, match="duplicate sample_id"):
        compare(reference, candidate)
    candidate["samples"].pop()
    beams = candidate["samples"][-1]["beams"]
    beams.append(beams[0])
    with pytest.raises(ValueError, match="duplicate beam token_ids"):
        compare(reference, candidate)


@pytest.mark.parametrize("score", [True, "1", float("inf"), float("nan")])
def test_rejects_non_numeric_or_non_finite_scores(
    diagnostic_pair: tuple[dict[str, Any], dict[str, Any]], score: object
) -> None:
    reference, candidate = diagnostic_pair
    candidate["samples"][-1]["beams"][0]["score"] = score
    with pytest.raises(ValueError, match="finite or null"):
        compare(reference, candidate)


def test_module_cli(diagnostic_pair: tuple[dict[str, Any], dict[str, Any]], tmp_path: Path) -> None:
    inputs = [tmp_path / name for name in ("reference.json", "candidate.json")]
    for path, data in zip(inputs, diagnostic_pair, strict=True):
        path.write_text(json.dumps(data), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "benchmarks.beam_search.compare_recall_diagnostics",
        *(str(path) for path in inputs),
    ]
    result = subprocess.run(command + ["--top-k", "1"], check=True, capture_output=True)
    assert json.loads(result.stdout)["rankings"]["exact_order"] == 3
    invalid = subprocess.run(command + ["--top-k", "0"], check=False, capture_output=True)
    assert invalid.returncode == 2
    assert b"top_k must be positive" in invalid.stderr
