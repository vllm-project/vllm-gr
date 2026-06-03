# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Batch tensor construction for FuXi-alpha model kwargs."""

from __future__ import annotations

from typing import Any

import torch


def _truncate_or_pad_seq(values: list[int], target_len: int, *, pad_value: int = 0) -> list[int]:
    """Pad or truncate a 1-D sequence to ``target_len``.

    Mirrors FuXi-alpha ``DatasetV2.load_item`` helper
    ``generative_recommenders/data/dataset.py::_truncate_or_pad_seq``:

        if y_len < target_len:
            y = y + [0] * (target_len - y_len)

    MovieLens training uses ``chronological=True``, so valid history tokens are
    left-aligned and padding id ``0`` is appended at the end. The same layout is
    assumed by:

    - ``movielens_seq_features_from_row`` when it appends zero columns for
      decode slots (``features.py``);
    - ``get_current_embeddings`` which indexes ``lengths[i] - 1``;
    - fbgemm ``dense_to_jagged`` / ``jagged_to_padded_dense``, which consume the
      first ``past_lengths[i]`` entries of each dense row.

    Batch construction only needs the ``len(values) <= target_len`` branch: each
    row is padded to ``n_max`` / ``k_max`` within the batch.
    """
    if len(values) > target_len:
        raise ValueError(f"Sequence length {len(values)} exceeds padded max length {target_len}")
    if not values:
        return [pad_value] * target_len
    if len(values) < target_len:
        return values + [pad_value] * (target_len - len(values))
    return list(values)


def build_fuxi_model_kwargs(
    batch_features: list[tuple[str, dict[str, Any]]],
    device,
    *,
    use_rating: bool,
    use_timestamp: bool,
) -> dict[str, Any]:
    """Build dense padded tensors for FuXiAlphaInferenceModel.forward."""
    if not batch_features:
        return {}

    history_lens = [int(feat["history_len"]) for _, feat in batch_features]
    candidate_lens = [int(feat["num_candidates"]) for _, feat in batch_features]
    n_max = max(history_lens) if history_lens else 0
    k_max = max(candidate_lens) if candidate_lens else 0

    past_ids_rows: list[list[int]] = []
    past_ratings_rows: list[list[int]] = []
    past_timestamps_rows: list[list[int]] = []
    candidate_id_rows: list[list[int]] = []

    for _, feat in batch_features:
        past_ids_rows.append(
            _truncate_or_pad_seq(list(feat["past_ids"]), n_max),
        )
        past_ratings_rows.append(
            _truncate_or_pad_seq(list(feat["past_ratings"]), n_max),
        )
        past_timestamps_rows.append(
            _truncate_or_pad_seq(list(feat["past_timestamps"]), n_max),
        )
        candidate_id_rows.append(
            _truncate_or_pad_seq(list(feat["candidate_ids"]), k_max),
        )

    past_payloads: dict[str, torch.Tensor] = {}
    if use_timestamp:
        past_payloads["timestamps"] = torch.tensor(
            past_timestamps_rows,
            dtype=torch.int64,
            device=device,
        )
    if use_rating:
        past_payloads["ratings"] = torch.tensor(
            past_ratings_rows,
            dtype=torch.int64,
            device=device,
        )

    return {
        "past_lengths": torch.tensor(history_lens, dtype=torch.int64, device=device),
        "past_ids": torch.tensor(past_ids_rows, dtype=torch.int64, device=device),
        "past_payloads": past_payloads,
        "candidate_ids": torch.tensor(
            candidate_id_rows,
            dtype=torch.int64,
            device=device,
        ),
        "candidate_lengths": torch.tensor(
            candidate_lens,
            dtype=torch.int64,
            device=device,
        ),
    }
