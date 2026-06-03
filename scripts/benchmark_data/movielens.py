#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare upstream-aligned MovieLens benchmark data for vllm-GR."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts.benchmark_data.data_processor import DataProcessor, _ensure_dir

logging.basicConfig(level=logging.INFO)

SUPPORTED_DATASETS = [
    "movielens-1m",
    "movielens-20m",
]


def _to_seq_data(
    ratings_data: pd.DataFrame,
    user_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if user_data is not None:
        ratings_data = ratings_data.join(user_data.set_index("user_id"), on="user_id")
    ratings_data = ratings_data.copy()
    ratings_data.item_ids = ratings_data.item_ids.apply(
        lambda values: ",".join(str(value) for value in values)
    )
    ratings_data.ratings = ratings_data.ratings.apply(
        lambda values: ",".join(str(value) for value in values)
    )
    ratings_data.timestamps = ratings_data.timestamps.apply(
        lambda values: ",".join(str(value) for value in values)
    )
    ratings_data.rename(
        columns={
            "item_ids": "sequence_item_ids",
            "ratings": "sequence_ratings",
            "timestamps": "sequence_timestamps",
        },
        inplace=True,
    )
    return ratings_data


def _build_seq_ratings_data(ratings: pd.DataFrame) -> pd.DataFrame:
    ordered_ratings = ratings.sort_values(
        by=["user_id", "unix_timestamp", "movie_id"],
        kind="mergesort",
    )
    ratings_group = ordered_ratings.groupby("user_id", sort=True)
    return (
        pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "item_ids": list(ratings_group.movie_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.unix_timestamp.apply(list)),
            }
        )
        .sort_values(by=["user_id"], kind="mergesort")
        .reset_index(drop=True)
    )


class MovieLensDataProcessor(DataProcessor):
    """MovieLens preprocessor aligned with upstream public-data preprocessing."""

    def __init__(
        self,
        download_url: str,
        output_root: Path,
        archive_name: str,
        prefix: str,
        insecure: bool = False,
    ) -> None:
        super().__init__(
            download_url,
            output_root,
            archive_name,
            prefix,
            insecure=insecure,
        )
        self.output_path = output_root / "data" / prefix / "sasrec_format.csv"

    def _load_users(self) -> pd.DataFrame | None:
        if self.prefix != "ml-1m":
            return None

        users = pd.read_csv(
            self.dataset_root / "users.dat",
            sep="::",
            names=["user_id", "sex", "age_group", "occupation", "zip_code"],
            engine="python",
        )
        users.sex = pd.Categorical(users.sex)
        users["sex"] = users.sex.cat.codes
        users.age_group = pd.Categorical(users.age_group)
        users["age_group"] = users.age_group.cat.codes
        users.occupation = pd.Categorical(users.occupation)
        users["occupation"] = users.occupation.cat.codes
        users.zip_code = pd.Categorical(users.zip_code)
        users["zip_code"] = users.zip_code.cat.codes
        return users

    def _load_ratings(self) -> pd.DataFrame:
        if self.prefix == "ml-1m":
            return pd.read_csv(
                self.dataset_root / "ratings.dat",
                sep="::",
                names=["user_id", "movie_id", "rating", "unix_timestamp"],
                engine="python",
            )

        ratings = pd.read_csv(self.dataset_root / "ratings.csv", sep=",")
        ratings.rename(
            columns={
                "userId": "user_id",
                "movieId": "movie_id",
                "timestamp": "unix_timestamp",
            },
            inplace=True,
        )
        return ratings

    def preprocess(self) -> None:
        self.extract()
        ratings = self._load_ratings()
        users = self._load_users()

        seq_ratings_data = _to_seq_data(
            _build_seq_ratings_data(ratings),
            users,
        ).sort_values(by=["user_id"], kind="mergesort")
        _ensure_dir(self.output_path.parent)
        seq_ratings_data.reset_index(drop=True).to_csv(
            self.output_path,
            index=False,
            sep=",",
        )
        self.log.info("Prepared benchmark dataset at %s", self.output_path)


def build_processors(output_root: Path, insecure: bool = False) -> dict[str, DataProcessor]:
    """Build dataset preprocessors keyed by public dataset name."""
    return {
        "movielens-1m": MovieLensDataProcessor(
            download_url="http://files.grouplens.org/datasets/movielens/ml-1m.zip",
            output_root=output_root,
            archive_name="movielens1m.zip",
            prefix="ml-1m",
            insecure=insecure,
        ),
        "movielens-20m": MovieLensDataProcessor(
            download_url="http://files.grouplens.org/datasets/movielens/ml-20m.zip",
            output_root=output_root,
            archive_name="movielens20m.zip",
            prefix="ml-20m",
            insecure=insecure,
        ),
    }


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments for the MovieLens benchmark data preparation script."""
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        required=True,
        help="MovieLens dataset to prepare.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Root directory where tmp/ and data/ will be created.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification for download only. Use temporarily when certificate validation fails.",
    )
    return parser


def main() -> None:
    parser = add_cli_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    _ensure_dir(output_root)
    processors = build_processors(output_root, insecure=args.insecure)
    processors[args.dataset].preprocess()


if __name__ == "__main__":
    main()
