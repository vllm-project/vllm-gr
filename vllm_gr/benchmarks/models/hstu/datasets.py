# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HSTU dataset adapters for GR accuracy benchmarks."""

from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from vllm_gr.benchmarks.common.base import BenchmarkDataset, BenchmarkSampleRequest
from vllm_gr.config.dataset_configs import (
    get_embedding_table_overrides,
    get_hstu_config_overrides,
)
from vllm_gr.inputs.data import GRPrompt

# Match upstream handling for large sequence CSV fields.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit((1 << 31) - 1)

SUPPORTED_HSTU_DATASETS = (
    "movielens-1m",
    "movielens-20m",
    "kuairand-1k",
)

DEFAULT_MAX_UIH_LEN = 1_000_000_000
KUAIRAND_TASK_NAMES = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "is_profile_enter",
]


def _parse_sequence(value: str | int | float | Sequence[Any]) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (int, float)):
        return [value]
    stripped = str(value).strip()
    if not stripped:
        return []
    if stripped[0] == "[" and stripped[-1] == "]":
        parsed = json.loads(stripped)
        return [parsed] if isinstance(parsed, (int, float, str)) else list(parsed)
    return [part.strip() for part in stripped.split(",") if part.strip()]


def _separate_uih_candidates(
    value: str | int | float | Sequence[Any],
    candidates_max_seq_len: int,
) -> tuple[list[Any], list[Any]]:
    items = _parse_sequence(value)
    candidates = items[-candidates_max_seq_len:]
    uih = items[:-candidates_max_seq_len]
    return uih, candidates


def _maybe_truncate_seq(values: list[Any], max_seq_len: int) -> list[Any]:
    if len(values) > max_seq_len:
        return values[:max_seq_len]
    return values


def _parse_sequence_literal(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    parsed = ast.literal_eval(str(value))
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _hash_feature(value: int, hash_size: int) -> int:
    return int(value) % hash_size


def _build_hash_sizes(embedding_tables: list[dict[str, Any]]) -> dict[str, int]:
    hash_sizes: dict[str, int] = {}
    for table in embedding_tables:
        table_hash_size = int(table["num_embeddings"])
        for feature_name in table["feature_names"]:
            hash_sizes[str(feature_name)] = table_hash_size
    return hash_sizes


# -----------------------------------------------------------------------------
# Concrete Dataset Classes
# -----------------------------------------------------------------------------


class MovieLensBenchmarkDataset(BenchmarkDataset):
    """MovieLens benchmark dataset aligned with vLLM's dataset class flow."""

    def __init__(
        self,
        *,
        dataset_name: str,
        input_csv: str,
        random_seed: int = BenchmarkDataset.DEFAULT_SEED,
        disable_shuffle: bool = False,
        candidate_count: int = 5,
    ) -> None:
        overrides = get_hstu_config_overrides(dataset_name)
        self.contextual_feature_to_max_length = dict(overrides["contextual_feature_to_max_length"])
        self.action_weights = list(overrides["action_weights"])
        self.max_uih_len = DEFAULT_MAX_UIH_LEN
        self._candidate_count = int(candidate_count)
        super().__init__(
            dataset_name=dataset_name,
            input_csv=input_csv,
            random_seed=random_seed,
            disable_shuffle=disable_shuffle,
        )

    @property
    def candidate_count(self) -> int:
        return self._candidate_count

    def load_data(self) -> None:
        with Path(self.input_csv).open("r", newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))

    def is_valid_sample(self, idx: int) -> bool:
        row = self.get_row(idx)
        history_ids, _ = _separate_uih_candidates(
            row["sequence_item_ids"],
            candidates_max_seq_len=self.candidate_count,
        )
        return bool(history_ids)

    def build_sample_request(
        self,
        row: dict[str, object],
        request_id: str,
    ) -> BenchmarkSampleRequest:
        history_ids, candidate_ids = _separate_uih_candidates(
            row["sequence_item_ids"],
            candidates_max_seq_len=self.candidate_count,
        )
        history_ratings, candidate_ratings = _separate_uih_candidates(
            row["sequence_ratings"],
            candidates_max_seq_len=self.candidate_count,
        )
        timestamps_uih, _ = _separate_uih_candidates(
            row["sequence_timestamps"],
            candidates_max_seq_len=self.candidate_count,
        )

        history_ids = _maybe_truncate_seq(history_ids, self.max_uih_len)
        history_ratings = _maybe_truncate_seq(history_ratings, self.max_uih_len)
        timestamps_uih = _maybe_truncate_seq(timestamps_uih, self.max_uih_len)

        user_history: list[dict[str, int]] = []
        for movie_id, rating, timestamp in zip(history_ids, history_ratings, timestamps_uih):
            rating_float = float(rating)
            rating_int = int(rating_float)
            user_history.append(
                {
                    "movie_id": int(movie_id),
                    "movie_rating": rating_int,
                    "action_timestamp": int(timestamp),
                    "item_weights": int(self.action_weights[rating_int - 1]),
                    "dummy_watch_time": 0,
                }
            )

        query_time = max((int(timestamp) for timestamp in timestamps_uih), default=0)
        candidates: list[dict[str, int]] = []
        labels: list[float] = []
        for movie_id, rating in zip(candidate_ids, candidate_ratings):
            raw_rating = float(rating)
            rating_int = int(raw_rating)
            label = float(raw_rating >= 3.5)
            labels.append(label)
            candidates.append(
                {
                    "item_movie_id": int(movie_id),
                    "item_movie_rating": rating_int,
                    "item_query_time": int(query_time),
                    "item_action_weights": int(label),
                    "item_dummy_watchtime": 1,
                }
            )

        context = {
            key: int(row[key])
            for key in self.contextual_feature_to_max_length
            if key in row and str(row[key]).strip() != ""
        }

        return BenchmarkSampleRequest(
            prompt={
                "user_id": str(row.get("user_id", "")),
                "user_history": user_history,
                "candidates": candidates,
                "context": context,
            },
            request_id=request_id,
            dataset_name=self.dataset_name,
            task_names=["rating"],
            labels_by_task={"rating": labels},
            weights_by_task={"rating": [1.0] * len(labels)},
            num_candidates=len(candidates),
        )


class KuaiRandBenchmarkDataset(BenchmarkDataset):
    """KuaiRand benchmark dataset aligned with vLLM's dataset class flow."""

    def __init__(
        self,
        *,
        dataset_name: str,
        input_csv: str,
        random_seed: int = BenchmarkDataset.DEFAULT_SEED,
        disable_shuffle: bool = False,
        candidate_count: int = 5,
    ) -> None:
        overrides = get_hstu_config_overrides(dataset_name)
        self.contextual_feature_to_max_length = dict(overrides["contextual_feature_to_max_length"])
        self.action_weights = list(overrides["action_weights"])
        self.max_uih_len = DEFAULT_MAX_UIH_LEN
        self._candidate_count = int(candidate_count)
        embedding_tables = get_embedding_table_overrides(dataset_name)
        self.hash_sizes = _build_hash_sizes(embedding_tables)
        super().__init__(
            dataset_name=dataset_name,
            input_csv=input_csv,
            random_seed=random_seed,
            disable_shuffle=disable_shuffle,
        )

    @property
    def candidate_count(self) -> int:
        return self._candidate_count

    def load_data(self) -> None:
        with Path(self.input_csv).open("r", newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))

    def is_valid_sample(self, idx: int) -> bool:
        row = self.get_row(idx)
        history_ids, _ = _separate_uih_candidates(
            _parse_sequence_literal(row["video_id"]),
            candidates_max_seq_len=self.candidate_count,
        )
        return bool(history_ids)

    def build_sample_request(
        self,
        row: dict[str, object],
        request_id: str,
    ) -> BenchmarkSampleRequest:
        history_ids, candidate_ids = _separate_uih_candidates(
            _parse_sequence_literal(row["video_id"]),
            candidates_max_seq_len=self.candidate_count,
        )
        history_action_weights, candidate_action_weights = _separate_uih_candidates(
            _parse_sequence_literal(row["action_weights"]),
            candidates_max_seq_len=self.candidate_count,
        )
        history_timestamps, _ = _separate_uih_candidates(
            _parse_sequence_literal(row["time_ms"]),
            candidates_max_seq_len=self.candidate_count,
        )
        history_watch_times, candidate_watch_times = _separate_uih_candidates(
            _parse_sequence_literal(row["play_time_ms"]),
            candidates_max_seq_len=self.candidate_count,
        )

        history_ids = _maybe_truncate_seq(history_ids, self.max_uih_len)
        history_action_weights = _maybe_truncate_seq(history_action_weights, self.max_uih_len)
        history_timestamps = _maybe_truncate_seq(history_timestamps, self.max_uih_len)
        history_watch_times = _maybe_truncate_seq(history_watch_times, self.max_uih_len)

        video_hash_size = self.hash_sizes["video_id"]
        context: dict[str, int] = {}
        for key in self.contextual_feature_to_max_length:
            if key not in row or str(row[key]).strip() == "":
                continue
            value = int(row[key])
            hash_size = self.hash_sizes.get(key)
            context[key] = _hash_feature(value, hash_size) if hash_size is not None else value

        user_history: list[dict[str, int]] = []
        for video_id, timestamp, action_weight, watch_time in zip(
            history_ids,
            history_timestamps,
            history_action_weights,
            history_watch_times,
        ):
            user_history.append(
                {
                    "video_id": _hash_feature(int(video_id), video_hash_size),
                    "action_timestamp": int(timestamp),
                    "action_weight": int(action_weight),
                    "watch_time": int(watch_time),
                }
            )

        query_time = max((int(timestamp) for timestamp in history_timestamps), default=0)
        labels_by_task = {task_name: [] for task_name in KUAIRAND_TASK_NAMES}
        candidates: list[dict[str, int]] = []

        for video_id, action_weight, watch_time in zip(
            candidate_ids,
            candidate_action_weights,
            candidate_watch_times,
        ):
            encoded_action_weight = int(action_weight)
            candidates.append(
                {
                    "item_video_id": _hash_feature(int(video_id), video_hash_size),
                    "item_action_weight": encoded_action_weight,
                    "item_target_watchtime": int(watch_time),
                    "item_query_time": query_time,
                }
            )
            for task_name, bit in zip(KUAIRAND_TASK_NAMES, self.action_weights):
                labels_by_task[task_name].append(float(bool(encoded_action_weight & int(bit))))

        weights_by_task = {task_name: [1.0] * len(candidates) for task_name in KUAIRAND_TASK_NAMES}

        return BenchmarkSampleRequest(
            prompt={
                "user_id": str(row.get("user_id", "")),
                "user_history": user_history,
                "candidates": candidates,
                "context": context,
            },
            request_id=request_id,
            dataset_name=self.dataset_name,
            task_names=list(KUAIRAND_TASK_NAMES),
            labels_by_task=labels_by_task,
            weights_by_task=weights_by_task,
            num_candidates=len(candidates),
        )


# -----------------------------------------------------------------------------
# Factory / Sampling Helpers
# -----------------------------------------------------------------------------


def create_dataset(
    *,
    dataset_name: str,
    input_csv: str,
    dataset_candidate_count: int = 5,
    random_seed: int = 0,
    disable_shuffle: bool = False,
) -> BenchmarkDataset:
    """Instantiate an HSTU benchmark dataset."""
    if dataset_name.startswith("movielens"):
        return MovieLensBenchmarkDataset(
            dataset_name=dataset_name,
            input_csv=input_csv,
            random_seed=random_seed,
            disable_shuffle=disable_shuffle,
            candidate_count=dataset_candidate_count,
        )

    if dataset_name.startswith("kuairand"):
        return KuaiRandBenchmarkDataset(
            dataset_name=dataset_name,
            input_csv=input_csv,
            random_seed=random_seed,
            disable_shuffle=disable_shuffle,
            candidate_count=dataset_candidate_count,
        )

    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. "
        f"Supported datasets: {', '.join(SUPPORTED_HSTU_DATASETS)}."
    )


def get_samples(
    *,
    dataset_name: str,
    input_csv: str,
    num_prompts: int | None,
    dataset_candidate_count: int = 5,
    random_seed: int = 0,
    disable_shuffle: bool = False,
) -> tuple[list[BenchmarkSampleRequest], list[GRPrompt], int]:
    """Create a dataset and sample benchmark requests from it."""
    dataset = create_dataset(
        dataset_name=dataset_name,
        input_csv=input_csv,
        dataset_candidate_count=dataset_candidate_count,
        random_seed=random_seed,
        disable_shuffle=disable_shuffle,
    )
    requests = dataset.sample(
        num_prompts,
        request_id_prefix=f"{dataset_name}-",
    )
    prompts = [request.prompt for request in requests]
    return requests, prompts, dataset.candidate_count
