# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Feasible CUDA V1 Prefill shapes for buffer allocation and graph capture."""

from collections.abc import Iterable


def plan_prefill_capture(
    buckets: Iterable[int],
    request_counts: Iterable[int],
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int | None,
    max_model_len: int,
    total_token_buckets: bool,
) -> dict[int, tuple[int, ...]]:
    """Return only buckets with at least one feasible request count.

    A configured GPU V1 bucket counts total tokens across requests. With the
    list unset, V1 retains its per-context grid; V2's native-grid policy is
    handled by its own planner. Capacity clamping preserves existing endpoints.
    """
    limits = [value for value in (max_num_batched_tokens,) if value and value > 0]
    if not total_token_buckets and max_model_len > 0:
        limits.append(max_model_len)
    limit = min(limits) if limits else None
    grid = sorted({min(bucket, limit) if limit else bucket for bucket in buckets})
    counts = sorted({count for count in request_counts if 0 < count <= max_num_seqs})
    result = {}
    for bucket in grid:
        feasible = tuple(
            count
            for count in counts
            if count <= bucket and (max_model_len <= 0 or bucket <= count * max_model_len)
        )
        if feasible:
            result[bucket] = feasible
    return result
