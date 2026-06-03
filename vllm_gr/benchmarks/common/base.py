# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common benchmark types and dataset abstractions."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkSampleRequest:
    """Represents one benchmark request emitted by a dataset adapter."""

    prompt: Any
    request_id: str
    dataset_name: str
    task_names: list[str]
    labels_by_task: dict[str, list[float]]
    weights_by_task: dict[str, list[float]]
    num_candidates: int


class BenchmarkDataset(ABC):
    """Base class for benchmark datasets."""

    DEFAULT_SEED = 0

    def __init__(
        self,
        *,
        dataset_name: str,
        input_csv: str,
        random_seed: int = DEFAULT_SEED,
        disable_shuffle: bool = False,
    ) -> None:
        csv_path = Path(input_csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {csv_path}")

        self.dataset_name = dataset_name
        self.input_csv = str(csv_path)
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.disable_shuffle = disable_shuffle
        self.rows: list[dict[str, object]] = []

        self.load_data()

    @property
    @abstractmethod
    def candidate_count(self) -> int:
        """Number of tail candidates emitted per benchmark request."""

    @abstractmethod
    def load_data(self) -> None:
        """Load raw rows from the prepared benchmark CSV."""

    def get_item_count(self) -> int:
        return len(self.rows)

    def get_row(self, idx: int) -> dict[str, object]:
        return self.rows[idx]

    @abstractmethod
    def is_valid_sample(self, idx: int) -> bool:
        """Return whether a row can produce a non-empty history."""

    @abstractmethod
    def build_sample_request(
        self,
        row: dict[str, object],
        request_id: str,
    ) -> BenchmarkSampleRequest:
        """Build one benchmark request from a loaded row."""

    def get_query_ids(self, num_requests: int | None = None) -> list[int]:
        ordered_ids = list(range(self.get_item_count()))
        if not self.disable_shuffle:
            random.Random(self.random_seed).shuffle(ordered_ids)

        valid_ids: list[int] = []
        for query_id in ordered_ids:
            if self.is_valid_sample(query_id):
                valid_ids.append(query_id)
                if num_requests is not None and len(valid_ids) >= num_requests:
                    break

        if num_requests is not None and len(valid_ids) < num_requests:
            raise ValueError(
                f"Requested {num_requests} prompts, but only {len(valid_ids)} "
                "valid benchmark samples are available."
            )
        return valid_ids

    def sample(
        self,
        num_requests: int | None,
        request_id_prefix: str = "",
    ) -> list[BenchmarkSampleRequest]:
        query_ids = self.get_query_ids(num_requests)
        if not query_ids:
            raise ValueError("No valid benchmark queries available for the selected dataset.")

        requests = [
            self.build_sample_request(self.get_row(query_id), request_id=str(query_id))
            for query_id in query_ids
        ]
        for index, request in enumerate(requests):
            request.request_id = f"{request_id_prefix}{index}"
        return requests
