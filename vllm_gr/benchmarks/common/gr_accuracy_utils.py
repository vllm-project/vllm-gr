# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared helpers for GR accuracy benchmark records and metrics."""

from __future__ import annotations

import math
from typing import Any

import torch


def _validate_grouped_metric_inputs(
    grouped_predictions_by_task: dict[str, list[list[float]]],
    grouped_labels_by_task: dict[str, list[list[float]]],
    grouped_weights_by_task: dict[str, list[list[float]]],
    num_candidates: list[int],
) -> list[str]:
    task_names = list(grouped_predictions_by_task.keys())
    if len(task_names) == 0:
        raise ValueError("grouped_predictions_by_task must be non-empty")
    if len(num_candidates) == 0:
        raise ValueError("num_candidates must be non-empty")

    num_requests = len(num_candidates)
    for task_name in task_names:
        predictions_groups = grouped_predictions_by_task[task_name]
        labels_groups = grouped_labels_by_task[task_name]
        weights_groups = grouped_weights_by_task[task_name]
        if len(predictions_groups) != num_requests:
            raise ValueError(
                f"Prediction group count mismatch for task={task_name}: "
                f"{len(predictions_groups)} vs {num_requests}"
            )
        if len(labels_groups) != num_requests:
            raise ValueError(
                f"Label group count mismatch for task={task_name}: "
                f"{len(labels_groups)} vs {num_requests}"
            )
        if len(weights_groups) != num_requests:
            raise ValueError(
                f"Weight group count mismatch for task={task_name}: "
                f"{len(weights_groups)} vs {num_requests}"
            )

        for request_idx, expected_num_candidates in enumerate(num_candidates):
            predictions = predictions_groups[request_idx]
            labels = labels_groups[request_idx]
            weights = weights_groups[request_idx]
            if len(predictions) != expected_num_candidates:
                raise ValueError(
                    f"Prediction length mismatch for task={task_name}, request={request_idx}: "
                    f"{len(predictions)} vs {expected_num_candidates}"
                )
            if len(labels) != expected_num_candidates:
                raise ValueError(
                    f"Label length mismatch for task={task_name}, request={request_idx}: "
                    f"{len(labels)} vs {expected_num_candidates}"
                )
            if len(weights) != expected_num_candidates:
                raise ValueError(
                    f"Weight length mismatch for task={task_name}, request={request_idx}: "
                    f"{len(weights)} vs {expected_num_candidates}"
                )
    return task_names


def extract_predictions_by_task(
    all_scores: Any,
    task_names: list[str],
    num_candidates: int,
) -> dict[str, list[float]]:
    """Convert raw score tensors into a task-indexed prediction mapping."""
    scores = torch.as_tensor(all_scores, dtype=torch.float32).detach().cpu()
    if scores.ndim == 1:
        if len(task_names) != 1:
            raise ValueError(f"Received 1D scores for multi-task request: tasks={task_names}")
        values = scores.tolist()
        if len(values) != num_candidates:
            raise ValueError(
                f"Candidate count mismatch: scores={len(values)}, expected={num_candidates}"
            )
        return {task_names[0]: [float(value) for value in values]}

    if scores.ndim != 2:
        raise ValueError(f"Expected 1D or 2D all_scores, got ndim={scores.ndim}.")

    if scores.shape[0] == len(task_names):
        task_major = scores
    elif scores.shape[1] == len(task_names):
        task_major = scores.transpose(0, 1)
    else:
        raise ValueError(
            "Score tensor shape is incompatible with task_names: "
            f"shape={tuple(scores.shape)}, num_tasks={len(task_names)}"
        )

    predictions_by_task: dict[str, list[float]] = {}
    for task_idx, task_name in enumerate(task_names):
        values = task_major[task_idx].tolist()
        if len(values) != num_candidates:
            raise ValueError(
                f"Candidate count mismatch for task={task_name}: "
                f"scores={len(values)}, expected={num_candidates}"
            )
        predictions_by_task[task_name] = [float(value) for value in values]
    return predictions_by_task


def _weighted_accuracy(
    predictions: list[float],
    labels: list[float],
    weights: list[float],
) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    correct = 0.0
    for prediction, label, weight in zip(predictions, labels, weights):
        predicted_label = 1.0 if prediction >= 0.5 else 0.0
        correct += weight * float(predicted_label == label)
    return correct / total_weight


def _normalized_entropy(
    predictions: list[float],
    labels: list[float],
    weights: list[float],
) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    eps = 1e-12
    cross_entropy = 0.0
    weighted_positives = 0.0
    for prediction, label, weight in zip(predictions, labels, weights):
        probability = min(max(prediction, eps), 1.0 - eps)
        cross_entropy += -weight * (
            label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability)
        )
        weighted_positives += weight * label
    cross_entropy /= total_weight
    positive_rate = min(max(weighted_positives / total_weight, eps), 1.0 - eps)
    label_entropy = -(
        positive_rate * math.log(positive_rate)
        + (1.0 - positive_rate) * math.log(1.0 - positive_rate)
    )
    return cross_entropy / label_entropy if label_entropy > 0 else 0.0


def _group_auc(
    grouped_predictions: list[list[float]],
    grouped_labels: list[list[float]],
    grouped_weights: list[list[float]],
) -> float:
    weighted_sum = 0.0
    weight_total = 0.0
    for predictions, labels, weights in zip(
        grouped_predictions,
        grouped_labels,
        grouped_weights,
    ):
        positives = [(p, w) for p, y, w in zip(predictions, labels, weights) if y >= 0.5]
        negatives = [(p, w) for p, y, w in zip(predictions, labels, weights) if y < 0.5]
        if not positives or not negatives:
            continue
        auc_sum = 0.0
        pair_weight = 0.0
        for pos_prediction, pos_weight in positives:
            for neg_prediction, neg_weight in negatives:
                current_weight = pos_weight * neg_weight
                pair_weight += current_weight
                if pos_prediction > neg_prediction:
                    auc_sum += current_weight
                elif pos_prediction == neg_prediction:
                    auc_sum += 0.5 * current_weight
        if pair_weight <= 0:
            continue
        group_auc = auc_sum / pair_weight
        group_weight = sum(weights)
        weighted_sum += group_auc * group_weight
        weight_total += group_weight
    return weighted_sum / weight_total if weight_total > 0 else 0.0


def compute_metrics(
    grouped_predictions_by_task: dict[str, list[list[float]]],
    grouped_labels_by_task: dict[str, list[list[float]]],
    grouped_weights_by_task: dict[str, list[list[float]]],
    num_candidates: list[int],
) -> dict[str, dict[str, float]]:
    """Compute accuracy metrics with pure Python."""
    _validate_grouped_metric_inputs(
        grouped_predictions_by_task=grouped_predictions_by_task,
        grouped_labels_by_task=grouped_labels_by_task,
        grouped_weights_by_task=grouped_weights_by_task,
        num_candidates=num_candidates,
    )

    metrics_by_task: dict[str, dict[str, float]] = {}
    for task_name, grouped_predictions in grouped_predictions_by_task.items():
        grouped_labels = grouped_labels_by_task[task_name]
        grouped_weights = grouped_weights_by_task[task_name]
        flat_predictions = [value for group in grouped_predictions for value in group]
        flat_labels = [value for group in grouped_labels for value in group]
        flat_weights = [value for group in grouped_weights for value in group]
        metrics_by_task[task_name] = {
            "ne": float(_normalized_entropy(flat_predictions, flat_labels, flat_weights)),
            "accuracy": float(_weighted_accuracy(flat_predictions, flat_labels, flat_weights)),
            "gauc": float(_group_auc(grouped_predictions, grouped_labels, grouped_weights)),
        }
    return metrics_by_task


def validate_accuracy_record(record: dict[str, Any]) -> None:
    """Validate the shape and required keys of a benchmark record."""
    required_keys = {
        "request_id",
        "dataset_name",
        "task_names",
        "num_candidates",
        "predictions_by_task",
        "labels_by_task",
        "weights_by_task",
    }
    missing = required_keys - set(record.keys())
    if missing:
        raise ValueError(f"Accuracy record is missing required keys: {sorted(missing)}")

    num_candidates = int(record["num_candidates"])
    if num_candidates <= 0:
        raise ValueError("num_candidates must be > 0")

    task_names = list(record["task_names"])
    if not task_names:
        raise ValueError("task_names must be non-empty")

    for task_name in task_names:
        predictions = list(record["predictions_by_task"][task_name])
        labels = list(record["labels_by_task"][task_name])
        weights = list(record["weights_by_task"][task_name])
        if len(predictions) != num_candidates:
            raise ValueError(
                f"Prediction length mismatch for task={task_name}: "
                f"{len(predictions)} vs {num_candidates}"
            )
        if len(labels) != num_candidates:
            raise ValueError(
                f"Label length mismatch for task={task_name}: {len(labels)} vs {num_candidates}"
            )
        if len(weights) != num_candidates:
            raise ValueError(
                f"Weight length mismatch for task={task_name}: {len(weights)} vs {num_candidates}"
            )
