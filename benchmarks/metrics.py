# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Statistics shared by benchmark runners."""

import math
import statistics
from collections.abc import Sequence


def linear_percentile_sorted(values: Sequence[float], quantile: float) -> float:
    """Interpolate a quantile in [0, 1] from non-empty, ascending values."""
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution_ms(values: list[float]) -> dict[str, float | str]:
    """Summarize millisecond samples with population standard deviation.

    Percentiles interpolate linearly at position ``(n - 1) * q`` in sorted samples.
    """
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0 for value in ordered):
        raise ValueError("canonical latency samples must be finite and positive")

    def percentile(percent: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percent / 100.0
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "unit": "ms",
    }
