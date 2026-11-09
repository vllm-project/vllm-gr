# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Private single-Trie views for the unchanged six-table NPU operator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from vllm_gr.constraints.table import CONSTRAINT_TABLE_TENSOR_NAMES, ConstraintTable


def npu_trie_arrays(table: ConstraintTable, trie_id: int) -> tuple[NDArray[Any], ...]:
    """Rebase one validated root without modifying the canonical artifact."""
    if not 0 <= trie_id < table.metadata.trie_count:
        raise ValueError("trie_id is outside the constraint table")
    if table.metadata.trie_count == 1:
        return tuple(getattr(table, name) for name in CONSTRAINT_TABLE_TENSOR_NAMES)
    vocab = table.metadata.vocab_size
    first_begin, first_end = table.root_offsets[trie_id : trie_id + 2]
    offsets1 = table.prefix1_offsets[trie_id * vocab : (trie_id + 1) * vocab + 1]
    begin1, end1 = int(offsets1[0]), int(offsets1[-1])
    root_key = trie_id * vocab**2
    pair_begin, pair_end = np.searchsorted(table.prefix1_pair_keys, [root_key, root_key + vocab**2])
    offsets2 = table.prefix2_value_offsets[pair_begin : pair_end + 1]
    begin2, end2 = int(offsets2[0]), int(offsets2[-1])
    return (
        table.first_token_ids[first_begin:first_end],
        offsets1 - begin1,
        table.prefix1_values[begin1:end1],
        table.prefix1_pair_keys[pair_begin:pair_end] - root_key,
        offsets2 - begin2,
        table.prefix2_values[begin2:end2],
    )
