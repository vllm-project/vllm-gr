#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare upstream-aligned KuaiRand benchmark data for vllm-GR."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_data.data_processor import DataProcessor, _ensure_dir

logging.basicConfig(level=logging.INFO)

SUPPORTED_DATASETS = [
    "kuairand-1k",
]

EVENT_MERGE_WEIGHT = {
    "is_click": 1,
    "is_like": 2,
    "is_follow": 4,
    "is_comment": 8,
    "is_forward": 16,
    "is_hate": 32,
    "long_view": 64,
    "is_profile_enter": 128,
}


class KuaiRandDataProcessor(DataProcessor):
    """KuaiRand preprocessor aligned with upstream public-data preprocessing."""

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
        if prefix != "KuaiRand-1K":
            raise ValueError(f"Unsupported KuaiRand prefix: {prefix}")
        self.log_files = [
            self.dataset_root / "data" / "log_standard_4_08_to_4_21_1k.csv",
            self.dataset_root / "data" / "log_standard_4_22_to_5_08_1k.csv",
        ]
        self.user_features_file = self.dataset_root / "data" / "user_features_1k.csv"
        self.output_path = output_root / "data" / prefix / "processed_seqs.csv"

    def _encode_user_features(self, user_features_df: pd.DataFrame) -> pd.DataFrame:
        def _one_hot_encode(row: pd.Series) -> pd.Series:
            mapping = {category: i + 1 for i, category in enumerate(row.unique())}
            return row.map(mapping)

        for column in [
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ]:
            user_features_df[column] = _one_hot_encode(user_features_df[column])
        return user_features_df

    def preprocess(self) -> None:
        self.extract()
        seq_cols = [
            "video_id",
            "time_ms",
            "action_weights",
            "play_time_ms",
            "duration_ms",
        ]
        merged_df: pd.DataFrame | None = None
        event_columns = list(EVENT_MERGE_WEIGHT.keys())

        for index, log_file in enumerate(self.log_files):
            self.log.info("Processing %s", log_file)
            log_df = pd.read_csv(log_file, delimiter=",")
            grouped_df = log_df.groupby("user_id").agg(list).reset_index()

            for event_name, weight in EVENT_MERGE_WEIGHT.items():
                grouped_df[event_name] = grouped_df[event_name].apply(
                    lambda sequence, current_weight=weight: np.where(
                        np.array(sequence) == 0, 0, current_weight
                    )
                )

            grouped_df["action_weights"] = grouped_df.apply(
                lambda row: [
                    int(sum(values))
                    for values in zip(*[row[column] for column in event_columns])
                ],
                axis=1,
            )
            grouped_df = grouped_df[["user_id"] + seq_cols]

            if index == 0:
                merged_df = grouped_df
                continue

            assert merged_df is not None
            merged_df = merged_df.merge(grouped_df, on="user_id", suffixes=("_x", "_y"))
            for column in seq_cols:
                merged_df[column] = merged_df.apply(
                    lambda row, current_column=column: (
                        row[f"{current_column}_x"] + row[f"{current_column}_y"]
                    ),
                    axis=1,
                )
                merged_df = merged_df.drop(columns=[f"{column}_x", f"{column}_y"])

        assert merged_df is not None
        user_features_df = pd.read_csv(self.user_features_file, delimiter=",")
        user_features_df = self._encode_user_features(user_features_df)
        final_df = pd.merge(merged_df, user_features_df, on="user_id")
        _ensure_dir(self.output_path.parent)
        final_df.to_csv(self.output_path, index=False, sep=",")
        self.log.info("Prepared benchmark dataset at %s", self.output_path)


def build_processors(output_root: Path, insecure: bool = False) -> dict[str, DataProcessor]:
    """Build dataset preprocessors keyed by public dataset name."""
    return {
        "kuairand-1k": KuaiRandDataProcessor(
            download_url="https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
            output_root=output_root,
            archive_name="KuaiRand-1K.tar.gz",
            prefix="KuaiRand-1K",
            insecure=insecure,
        ),
    }


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments for the KuaiRand benchmark data preparation script."""
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        required=True,
        help="KuaiRand dataset to prepare.",
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
