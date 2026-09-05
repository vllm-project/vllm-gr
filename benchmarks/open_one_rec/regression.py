# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic OneRec recommendation regression helpers.

This module intentionally has no vLLM dependency.  Dataset selection,
streaming metric aggregation, result serialization, and baseline comparison
can therefore be unit-tested on CPU and reused by offline and serving runners.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.open_one_rec.tasks.recommendation import utils as utils_sid
from benchmarks.open_one_rec.tasks.recommendation import utils_by_pid as utils_pid

RECOMMENDATION_TASKS = ("video", "ad", "product", "label_cond", "interactive")
TASK_SIZES = {
    "video": 38_781,
    "ad": 27_677,
    "product": 27_910,
    "label_cond": 34_891,
    "interactive": 1_000,
}
TASK_MAPPING_FILES = {
    "video": "sid2pid.json",
    "ad": "sid2pid.json",
    "product": "sid2iid.json",
    "label_cond": "sid2pid.json",
    "interactive": "sid2pid.json",
}

_SID_RE = re.compile(r"^\s*<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><\|sid_end\|>\s*$")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a content fingerprint without loading a large dataset at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        while chunk := file_obj.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_sample_manifest(
    path: str | Path, *, dataset_path: str | Path | None = None
) -> dict[str, list[int]]:
    """Load and strictly validate a versioned task -> row-ID manifest."""
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as file_obj:
        document = json.load(file_obj)

    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("sample manifest must declare version 1")
    tasks = document.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("sample manifest must be an object or contain a 'tasks' object")

    unknown = sorted(set(tasks) - set(RECOMMENDATION_TASKS))
    if unknown:
        raise ValueError(f"sample manifest contains unknown tasks: {unknown}")

    result: dict[str, list[int]] = {}
    for task, raw_ids in tasks.items():
        if not isinstance(raw_ids, list):
            raise ValueError(f"manifest task {task!r} must contain a list of row IDs")
        ids = [int(value) for value in raw_ids]
        if len(ids) != len(set(ids)):
            raise ValueError(f"manifest task {task!r} contains duplicate row IDs")
        if any(value < 0 or value >= TASK_SIZES[task] for value in ids):
            raise ValueError(f"manifest task {task!r} contains an out-of-range row ID")
        result[task] = ids

    if dataset_path is not None:
        source_sha256 = document.get("source_sha256")
        if not isinstance(source_sha256, dict) or set(source_sha256) != set(result):
            raise ValueError("sample manifest source_sha256 must cover exactly its selected tasks")
        dataset_root = Path(dataset_path)
        for task, expected_digest in source_sha256.items():
            if not isinstance(expected_digest, str) or not re.fullmatch(
                r"[0-9a-f]{64}", expected_digest
            ):
                raise ValueError(f"sample manifest has an invalid source digest for {task!r}")
            parquet_path = dataset_root / task / f"{task}_test.parquet"
            actual_digest = sha256_file(parquet_path)
            if actual_digest != expected_digest:
                raise ValueError(
                    f"sample manifest source mismatch for {task}: "
                    f"expected {expected_digest}, got {actual_digest}"
                )
    return result


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            yield value


def validate_generations(generations: Sequence[str], beam_width: int) -> dict[str, Any]:
    """Validate the observable OneRec SID output contract."""
    normalized = [utils_sid.extract_id_from_generation(text) for text in generations]
    return {
        "beam_count": len(generations),
        "beam_count_ok": len(generations) == beam_width,
        "empty_count": sum(not value for value in normalized),
        "invalid_sid_count": sum(not _SID_RE.fullmatch(text) for text in generations),
        "duplicate_sid_count": len(normalized) - len(set(normalized)),
    }


@dataclass
class RecommendationMetricAccumulator:
    """Streaming equivalent of ``RecommendationEvaluator`` for full runs."""

    task: str
    data_dir: str | Path
    k_values: tuple[int, ...] = (1, 32)
    evaluation_mode: str = "both"
    _mapping: dict[int, list[dict[str, int]]] | None = field(init=False, default=None)
    _count: int = field(init=False, default=0)
    _sums: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in RECOMMENDATION_TASKS:
            raise ValueError(f"unsupported recommendation task: {self.task}")
        if self.evaluation_mode not in ("sid", "pid", "both"):
            raise ValueError("evaluation_mode must be 'sid', 'pid', or 'both'")
        if self.evaluation_mode in ("pid", "both"):
            mapping_path = Path(self.data_dir) / TASK_MAPPING_FILES[self.task]
            self._mapping = utils_pid.load_pid_mapping(str(mapping_path))

    def add(
        self,
        generations: Sequence[str],
        ground_truth: str,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Evaluate one sample, update totals, and return its metric record."""
        sample_metrics: dict[str, Any] = {}

        if self.evaluation_mode in ("sid", "both"):
            predicted_sids = [utils_sid.extract_id_from_generation(text) for text in generations]
            expected_sids = utils_sid.extract_ids_from_answer(ground_truth)
            first_sid = utils_sid.extract_first_id_from_answer(ground_truth)
            for k in self.k_values:
                sample_metrics[f"pass@{k}"] = utils_sid.compute_pass_at_k(
                    predicted_sids, expected_sids, k
                )
                sample_metrics[f"position1_pass@{k}"] = utils_sid.compute_position1_pass_at_k(
                    predicted_sids, first_sid, k
                )
                sample_metrics[f"recall@{k}"] = utils_sid.compute_recall_at_k(
                    predicted_sids, expected_sids, k
                )

        if self.evaluation_mode in ("pid", "both"):
            assert self._mapping is not None
            answer_key = "answer_iid" if self.task == "product" else "answer_pid"
            raw_expected = metadata.get(answer_key, [])
            if isinstance(raw_expected, str):
                raw_expected = json.loads(raw_expected)
            expected_pids = utils_pid.extract_ids_from_answer(raw_expected)
            first_pid = utils_pid.extract_first_id_from_answer(raw_expected)
            predicted_pids = [
                utils_pid.extract_id_from_generation(
                    text,
                    self._mapping,
                    "most_popular_after_downsampling",
                )
                for text in generations
            ]
            for k in self.k_values:
                sample_metrics[f"pid_pass@{k}"] = utils_pid.compute_pass_at_k(
                    predicted_pids, expected_pids, k
                )
                sample_metrics[f"pid_position1_pass@{k}"] = utils_pid.compute_position1_pass_at_k(
                    predicted_pids, first_pid, k
                )
                sample_metrics[f"pid_recall@{k}"] = utils_pid.compute_recall_at_k(
                    predicted_pids, expected_pids, k
                )

        self.add_metrics(sample_metrics)
        return sample_metrics

    def add_metrics(self, sample_metrics: Mapping[str, Any]) -> None:
        """Add stored per-sample metrics when resuming a JSONL run."""
        expected_names = self.metric_names
        missing = sorted(expected_names - set(sample_metrics))
        if missing:
            raise ValueError(f"stored sample metrics are missing keys: {missing}")
        self._count += 1
        for name in expected_names:
            self._sums[name] = self._sums.get(name, 0.0) + float(sample_metrics[name])

    @property
    def metric_names(self) -> set[str]:
        prefixes = []
        if self.evaluation_mode in ("sid", "both"):
            prefixes.append("")
        if self.evaluation_mode in ("pid", "both"):
            prefixes.append("pid_")
        return {
            f"{prefix}{metric}@{k}"
            for prefix in prefixes
            for metric in ("pass", "position1_pass", "recall")
            for k in self.k_values
        }

    def summary(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {"total_samples": self._count}
        for name in sorted(self.metric_names):
            metrics[name] = self._sums.get(name, 0.0) / self._count if self._count else 0.0
        metrics.update(
            {
                "evaluation_mode": self.evaluation_mode,
                "sid_to_pid_strategy": "most_popular_after_downsampling_stable",
            }
        )
        return metrics


def compare_jsonl(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    top_k: int = 32,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Compare stable sample IDs and ranked generations from two runs."""
    baseline, duplicate_baseline = _index_records(baseline_path)
    candidate, duplicate_candidate = _index_records(candidate_path)
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    common_keys = sorted(baseline_keys & candidate_keys)

    top1_mismatches = 0
    ranked_topk_mismatches = 0
    topk_set_mismatches = 0
    metric_mismatches = 0
    metric_schema_mismatches = 0
    prompt_mismatches = 0
    examples: list[dict[str, Any]] = []
    baseline_metric_sums: dict[str, float] = {}
    candidate_metric_sums: dict[str, float] = {}
    comparable_metric_samples: dict[str, int] = {}

    for key in common_keys:
        base = baseline[key]
        cand = candidate[key]
        prompt_diff = base.get("prompt_sha256") != cand.get("prompt_sha256")
        prompt_mismatches += int(prompt_diff)
        base_gens = list(base.get("generations", []))
        cand_gens = list(cand.get("generations", []))
        top1_diff = base_gens[:1] != cand_gens[:1]
        ranked_diff = base_gens[:top_k] != cand_gens[:top_k]
        set_diff = set(base_gens[:top_k]) != set(cand_gens[:top_k])
        metrics_diff = not _metrics_equal(base.get("metrics", {}), cand.get("metrics", {}))
        top1_mismatches += int(top1_diff)
        ranked_topk_mismatches += int(ranked_diff)
        topk_set_mismatches += int(set_diff)
        metric_mismatches += int(metrics_diff)
        base_metrics = base.get("metrics", {})
        cand_metrics = cand.get("metrics", {})
        if isinstance(base_metrics, Mapping) and isinstance(cand_metrics, Mapping):
            metric_schema_mismatches += int(set(base_metrics) != set(cand_metrics))
            for metric_name in set(base_metrics) & set(cand_metrics):
                base_value = base_metrics[metric_name]
                cand_value = cand_metrics[metric_name]
                if isinstance(base_value, (bool, int, float)) and isinstance(
                    cand_value, (bool, int, float)
                ):
                    baseline_metric_sums[metric_name] = baseline_metric_sums.get(
                        metric_name, 0.0
                    ) + float(base_value)
                    candidate_metric_sums[metric_name] = candidate_metric_sums.get(
                        metric_name, 0.0
                    ) + float(cand_value)
                    comparable_metric_samples[metric_name] = (
                        comparable_metric_samples.get(metric_name, 0) + 1
                    )
        if (prompt_diff or top1_diff or ranked_diff or metrics_diff) and len(
            examples
        ) < max_examples:
            examples.append(
                {
                    "task": key[0],
                    "sample_id": key[1],
                    "prompt_sha256_baseline": base.get("prompt_sha256"),
                    "prompt_sha256_candidate": cand.get("prompt_sha256"),
                    "top1_baseline": base_gens[:1],
                    "top1_candidate": cand_gens[:1],
                    "topk_overlap": len(set(base_gens[:top_k]) & set(cand_gens[:top_k])),
                    "baseline_metrics": base.get("metrics", {}),
                    "candidate_metrics": cand.get("metrics", {}),
                }
            )

    aggregate_metrics = {}
    aggregate_metric_regressions = []
    for metric_name in sorted(comparable_metric_samples):
        sample_count = comparable_metric_samples[metric_name]
        baseline_value = baseline_metric_sums[metric_name] / sample_count
        candidate_value = candidate_metric_sums[metric_name] / sample_count
        delta = candidate_value - baseline_value
        aggregate_metrics[metric_name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
            "samples": sample_count,
        }
        if delta < -1e-12:
            aggregate_metric_regressions.append(metric_name)

    return {
        "baseline_samples": len(baseline),
        "candidate_samples": len(candidate),
        "common_samples": len(common_keys),
        "duplicate_baseline": [list(key) for key in duplicate_baseline],
        "duplicate_candidate": [list(key) for key in duplicate_candidate],
        "missing_from_candidate": [list(key) for key in sorted(baseline_keys - candidate_keys)],
        "unexpected_in_candidate": [list(key) for key in sorted(candidate_keys - baseline_keys)],
        "top1_mismatches": top1_mismatches,
        f"ranked_top{top_k}_mismatches": ranked_topk_mismatches,
        f"top{top_k}_set_mismatches": topk_set_mismatches,
        "per_sample_metric_mismatches": metric_mismatches,
        "metric_schema_mismatches": metric_schema_mismatches,
        "prompt_mismatches": prompt_mismatches,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_metric_regressions": aggregate_metric_regressions,
        "examples": examples,
    }


GOLDEN_IDENTITY_REQUIRED_FIELDS = (
    "model",
    "revision",
    "source_sha256",
    "beam_width",
    "temperature",
    "evaluation_mode",
)

# Fields that are allowed to differ on purpose between a native golden
# baseline and a vllm-gr candidate: the paired differential intentionally
# compares an offline native run against an online vllm-gr candidate across
# implementations, execution topologies, and constraint backends.  Candidate
# bookkeeping (seed) and native batch size are likewise not shared settings.
GOLDEN_IDENTITY_IGNORED_FIELDS = (
    "execution",
    "implementation_revision",
    "execution_variant",
    "constraint",
    "batch_size",
    "seed",
)


def _read_run_identity(path: str | Path) -> dict[str, Any] | None:
    """Return the run_identity of the first record in a JSONL results file."""
    with Path(path).open(encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            identity = json.loads(line).get("run_identity")
            if isinstance(identity, dict):
                return identity
    return None


def validate_golden_identity(
    baseline_path: str | Path,
    candidate_path: str | Path,
) -> dict[str, Any]:
    """Verify a reused golden baseline shares the candidate's run settings.

    Golden records carry the settings that produced them.  A baseline
    generated under a different model, dataset digest, or beam configuration
    must never gate a candidate, so every required field must match.
    Implementation, execution-variant, and constraint differences are
    intentionally ignored.
    """
    baseline_identity = _read_run_identity(baseline_path)
    if baseline_identity is None:
        raise ValueError(f"baseline records carry no run_identity: {baseline_path}")
    candidate_identity = _read_run_identity(candidate_path)
    if candidate_identity is None:
        raise ValueError(f"candidate records carry no run_identity: {candidate_path}")
    mismatches = [
        f"{field}: baseline={baseline_identity.get(field)!r} != "
        f"candidate={candidate_identity.get(field)!r}"
        for field in GOLDEN_IDENTITY_REQUIRED_FIELDS
        if baseline_identity.get(field) != candidate_identity.get(field)
    ]
    return {
        "baseline": baseline_identity,
        "candidate": candidate_identity,
        "ignored_fields": list(GOLDEN_IDENTITY_IGNORED_FIELDS),
        "mismatches": mismatches,
    }


def comparison_passed(
    comparison: Mapping[str, Any],
    *,
    policy: str = "exact",
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    metric_names: Sequence[str] | None = None,
) -> bool:
    """Apply exact diagnostic or aggregate no-regression comparison policy."""
    structural_failure = bool(
        comparison.get("duplicate_baseline")
        or comparison.get("duplicate_candidate")
        or comparison.get("missing_from_candidate")
        or comparison.get("unexpected_in_candidate")
        or comparison.get("metric_schema_mismatches")
        or comparison.get("prompt_mismatches")
    )
    if structural_failure:
        return False
    if policy == "exact":
        return not (
            comparison.get("top1_mismatches") or comparison.get("per_sample_metric_mismatches")
        )
    if policy == "no-regression":
        if absolute_tolerance < 0 or not 0 <= relative_tolerance <= 1:
            raise ValueError("comparison tolerances must be non-negative and relative <= 1")
        aggregate_metrics = comparison.get("aggregate_metrics", {})
        selected_names = list(metric_names) if metric_names is not None else list(aggregate_metrics)
        if any(name not in aggregate_metrics for name in selected_names):
            return False
        for name in selected_names:
            values = aggregate_metrics[name]
            baseline = float(values["baseline"])
            candidate = float(values["candidate"])
            allowed_drop = max(absolute_tolerance, abs(baseline) * relative_tolerance)
            if candidate < baseline - allowed_drop - 1e-12:
                return False
        return True
    raise ValueError(f"unknown comparison policy: {policy}")


def _record_key(record: Mapping[str, Any]) -> tuple[str, str]:
    try:
        return str(record["task"]), str(record["sample_id"])
    except KeyError as error:
        raise ValueError(f"regression record is missing {error.args[0]!r}") from error


def _index_records(
    path: str | Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[tuple[str, str]]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates: list[tuple[str, str]] = []
    for record in iter_jsonl(path):
        key = _record_key(record)
        if key in records:
            duplicates.append(key)
        records[key] = record
    return records, sorted(set(duplicates))


def _metrics_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        lhs, rhs = left[key], right[key]
        if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
            if not math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=1e-12):
                return False
        elif lhs != rhs:
            return False
    return True


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]], mode: str = "a") -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open(mode, encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
