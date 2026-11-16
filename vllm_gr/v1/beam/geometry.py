# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from typing import Tuple


def _compute_beam_bounds(
    b: int,
    b_prefix_len: int,
    b_cache_len: int,
    decode_steps: int,
    chunk_budget: int,
) -> Tuple[int, int, int, int]:
    """Calculates active lookahead allocations and historical trace lengths."""
    delta = max(0, b_cache_len - b_prefix_len)
    remaining_prefix = b_prefix_len - b_cache_len
    b_physical = delta // decode_steps
    b_uncached = 0
    b_suffix_len = 0

    b_past_suffix = max(0, min(decode_steps, delta - b_physical * decode_steps))
    if remaining_prefix >= 0:
        # Scenario A: Prefix Incomplete - compute remaining prefix + suffix of first beam
        b_uncached = min(chunk_budget, remaining_prefix + decode_steps)
    else:
        # Scenario B: Compute only suffix of beams
        b_uncached = min(chunk_budget, max(decode_steps - b_past_suffix, 0))
    b_suffix_len = b_uncached + b_past_suffix
    return b_uncached, b_suffix_len, b_past_suffix, b_physical
