# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compare ranked SIDs in two prompt-free Beam Recall diagnostic files."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.metrics import linear_percentile_sorted

SID = tuple[int, ...]
METADATA_FIELDS = (
    "artifact_digest",
    "model_revision",
    "model_files",
    "commit",
    "source_sha256",
    "runtime_provenance",
    "engine_config",
    "gr_config",
    "prefix_environment",
)
BATCH_CONTEXTS = ("same", "changed", "unrecorded")


@dataclass
class Sample:
    prompt_digest: str
    scores: dict[SID, float | None]


@dataclass
class DiagnosticRun:
    task: str
    samples: dict[str, Sample]
    batches: list[tuple[str, ...]] | None


def _read_samples(rows: object, label: str) -> dict[str, Sample]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label}: expected a non-empty samples list")
    samples: dict[str, Sample] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("sample_id"), str)
            or not row["sample_id"]
        ):
            raise ValueError(f"{label}: each sample must have a non-empty string sample_id")
        sample_id = row["sample_id"]
        if sample_id in samples:
            raise ValueError(f"{label}: duplicate sample_id {sample_id!r}")
        where = f"{label}: sample {sample_id!r}"
        digest = row.get("prompt_token_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(digit not in "0123456789abcdef" for digit in digest)
        ):
            raise ValueError(f"{where}: expected a SHA-256 prompt_token_sha256")
        beams = row.get("beams")
        if not isinstance(beams, list):
            raise ValueError(f"{where}: expected a beams list")
        scores: dict[SID, float | None] = {}
        for beam in beams:
            tokens = beam.get("token_ids") if isinstance(beam, dict) else None
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)
            ):
                raise ValueError(f"{where}: expected non-empty integer beam token_ids")
            sid = tuple(tokens)
            if sid in scores:
                raise ValueError(f"{where}: duplicate beam token_ids {sid}")
            score = beam.get("score")
            if score is not None:
                if type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError(f"{where}: beam score must be finite or null")
                score = float(score)
            scores[sid] = score
        samples[sample_id] = Sample(digest, scores)
    return samples


def _read_run(data: Any, label: str) -> DiagnosticRun:
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or (data["schema_version"] != 1)
    ):
        raise ValueError(f"{label}: expected a Recall diagnostic object with schema_version 1")
    task = data.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError(f"{label}: expected a task name")
    samples = _read_samples(data.get("samples"), label)
    if "batches" not in data:
        return DiagnosticRun(task, samples, None)

    groups = data["batches"]
    if not isinstance(groups, list) or any(
        not isinstance(group, list)
        or not group
        or any(not isinstance(sample_id, str) for sample_id in group)
        for group in groups
    ):
        raise ValueError(f"{label}: batches must contain non-empty lists of sample IDs")
    members = [sample_id for group in groups for sample_id in group]
    if len(members) != len(samples) or set(members) != samples.keys():
        raise ValueError(f"{label}: batches must list each sample exactly once")
    return DiagnosticRun(task, samples, [tuple(group) for group in groups])


def _compare_sample(
    reference: Sample, candidate: Sample, top_k: int | None
) -> tuple[dict[str, Any], list[float]]:
    ref_order = list(reference.scores)[:top_k]
    cand_order = list(candidate.scores)[:top_k]
    ref_set, cand_set = set(ref_order), set(cand_order)
    shared = ref_set & cand_set
    score_deltas = []
    for sid in shared:
        old, new = reference.scores[sid], candidate.scores[sid]
        if old is not None and new is not None:
            delta = abs(new - old)
            if not math.isfinite(delta):
                raise ValueError("absolute score difference exceeds the float range")
            score_deltas.append(delta)
    return {
        "exact_order": ref_order == cand_order,
        "same_set": ref_set == cand_set,
        "top1_equal": ref_order[0] == cand_order[0] if ref_order and cand_order else None,
        "shared_candidates": len(shared),
        "reference_only": [sid for sid in ref_order if sid not in cand_set],
        "candidate_only": [sid for sid in cand_order if sid not in ref_set],
    }, score_deltas


def _rank_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    exact = sum(row["exact_order"] for row in rows)
    same_set = sum(row["same_set"] for row in rows)
    return {
        "samples": len(rows),
        "exact_order": exact,
        "same_set": same_set,
        "reordered_only": same_set - exact,
        "changed_set": len(rows) - same_set,
    }


def compare(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    top_k: int | None = None,
    details: bool = False,
) -> dict[str, Any]:
    """Match samples by ID and prompt digest, then compare their ranked SIDs."""
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    base, new = _read_run(reference, "reference"), _read_run(candidate, "candidate")
    if base.task != new.task:
        raise ValueError(f"task mismatch: {base.task!r} != {new.task!r}")
    for field in ("token_hash_format", "tokenizer_digest"):
        old_value, new_value = reference.get(field), candidate.get(field)
        if old_value is not None and new_value is not None and old_value != new_value:
            raise ValueError(f"{field} mismatch")
    if base.samples.keys() != new.samples.keys():
        missing = sorted(base.samples.keys() - new.samples.keys())
        unexpected = sorted(new.samples.keys() - base.samples.keys())
        raise ValueError(f"sample IDs differ: missing {missing[:5]}, unexpected {unexpected[:5]}")

    have_batches = base.batches is not None and new.batches is not None
    base_groups = {sample_id: group for group in (base.batches or []) for sample_id in group}
    new_groups = {sample_id: group for group in (new.batches or []) for sample_id in group}
    rows, all_score_deltas = [], []
    for sample_id, ref_sample in base.samples.items():
        cand_sample = new.samples[sample_id]
        if ref_sample.prompt_digest != cand_sample.prompt_digest:
            raise ValueError(f"sample {sample_id!r}: prompt_token_sha256 mismatch")
        row, score_deltas = _compare_sample(ref_sample, cand_sample, top_k)
        context = "unrecorded"
        if have_batches:
            context = "same" if base_groups[sample_id] == new_groups[sample_id] else "changed"
        rows.append(
            {
                "sample_id": sample_id,
                "batch_context": context,
                **row,
                "score_pairs": len(score_deltas),
                "max_abs_score_delta": max(score_deltas, default=None),
            }
        )
        all_score_deltas.extend(score_deltas)
    all_score_deltas.sort()

    n = len(rows)
    counts = _rank_counts(rows)
    changed_metadata = [
        field for field in METADATA_FIELDS if reference.get(field) != candidate.get(field)
    ]
    ref_params = reference.get("beam_search_params")
    cand_params = candidate.get("beam_search_params")
    params_status = "unrecorded"
    if ref_params is not None and cand_params is not None:
        params_status = "same" if ref_params == cand_params else "changed"
        if params_status == "changed":
            changed_metadata.append("beam_search_params")

    result = {
        "task": base.task,
        "top_k": top_k,
        "rankings": {
            **counts,
            "exact_order_rate": counts["exact_order"] / n,
            "same_set_rate": counts["same_set"] / n,
            "top1_equal": sum(row["top1_equal"] is True for row in rows),
            "top1_compared": sum(row["top1_equal"] is not None for row in rows),
            "mean_shared_candidates": sum(row["shared_candidates"] for row in rows) / n,
            "removed_candidates": sum(len(row["reference_only"]) for row in rows),
            "added_candidates": sum(len(row["candidate_only"]) for row in rows),
        },
        "batches": {
            "submission_order_equal": base.batches == new.batches if have_batches else None,
            **{
                context: _rank_counts([row for row in rows if row["batch_context"] == context])
                for context in BATCH_CONTEXTS
            },
        },
        "scores": {
            "pairs": len(all_score_deltas),
            "p95_absolute_delta": (
                linear_percentile_sorted(all_score_deltas, 0.95) if all_score_deltas else None
            ),
            "max_absolute_delta": max(all_score_deltas, default=None),
        },
        "beam_search_params_status": params_status,
        "changed_metadata": changed_metadata,
    }
    if details:
        result["samples"] = rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference diagnostic_samples.json.")
    parser.add_argument("candidate", type=Path, help="Candidate diagnostic_samples.json.")
    parser.add_argument("--top-k", type=int, help="Limit both lists to their first K SIDs.")
    parser.add_argument("--details", action="store_true", help="Include per-sample differences.")
    args = parser.parse_args()
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        result = compare(reference, candidate, top_k=args.top_k, details=args.details)
    except (OSError, ValueError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
