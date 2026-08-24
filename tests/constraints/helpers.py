# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np

from vllm_gr.config import CONSTRAINT_TABLE_FORMAT
from vllm_gr.constraints.backend import ConstraintBackendInfo
from vllm_gr.constraints.table import ConstraintTable, ConstraintTableInfo

TOKENIZER_DIGEST = "sha256:" + "a" * 64
ARTIFACT_DIGEST = "sha256:" + "b" * 64


def make_table(*, artifact_digest: str | None = ARTIFACT_DIGEST) -> ConstraintTable:
    arrays = {
        "first_token_ids": np.asarray([1, 2], dtype=np.int32),
        "prefix1_offsets": np.asarray([0, 0, 2, 3, 3, 3, 3, 3, 3], dtype=np.int32),
        "prefix1_values": np.asarray([3, 4, 4], dtype=np.int32),
        "prefix1_pair_keys": np.asarray([11, 12, 20], dtype=np.int64),
        "prefix2_value_offsets": np.asarray([0, 1, 2, 3], dtype=np.int32),
        "prefix2_values": np.asarray([5, 6, 7], dtype=np.int32),
    }
    return ConstraintTable(
        **arrays,
        info=ConstraintTableInfo(
            format=CONSTRAINT_TABLE_FORMAT,
            artifact_digest=artifact_digest,
            tokenizer_digest=TOKENIZER_DIGEST,
            vocab_size=8,
            max_prefix1_degree=2,
            max_prefix2_degree=1,
            total_bytes=sum(int(array.nbytes) for array in arrays.values()),
        ),
    )


def make_backend_info(
    table: ConstraintTable,
    *,
    max_top_k: int = 4,
    device_memory_bytes: int = 0,
) -> ConstraintBackendInfo:
    return ConstraintBackendInfo(
        table=table.info,
        max_top_k=max_top_k,
        device_memory_bytes=device_memory_bytes,
    )
