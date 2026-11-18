# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Compatibility compiler for the canonical constraint-table artifact ABI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from vllm_gr.config.models import (
    CONSTRAINT_TABLE_FORMAT,
    MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
)
from vllm_gr.config.validation import validate_tokenizer_digest
from vllm_gr.constraints.compiler import ConstraintTableCompiler
from vllm_gr.constraints.table import (
    MAX_CONSTRAINT_TABLE_BYTES,
    ConstraintTable,
    ConstraintTableMetadata,
    validate_constraint_table,
)


class ConstraintTableArtifactCompiler:
    """Wrap the legacy compiler output without duplicating its triples logic."""

    def __init__(
        self,
        tokenizer: Any,
        vocab_size: int,
        *,
        tokenizer_digest: str,
    ) -> None:
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {vocab_size!r}")
        normalized_digest = validate_tokenizer_digest(tokenizer_digest)
        if normalized_digest is None:  # pragma: no cover - guarded by the type
            raise ValueError("tokenizer_digest is required")
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.tokenizer_digest = normalized_digest

    def compile(self, document: dict[str, object]) -> ConstraintTable:
        """Compile through the legacy implementation and wrap its six tensors."""

        resource = ConstraintTableCompiler(self.tokenizer, self.vocab_size).compile(document)
        table = ConstraintTable(
            first_token_ids=resource.first_token_ids,
            prefix1_offsets=resource.prefix1_offsets,
            prefix1_values=resource.prefix1_values,
            prefix1_pair_keys=resource.prefix1_pair_keys,
            prefix2_value_offsets=resource.prefix2_value_offsets,
            prefix2_values=resource.prefix2_values,
            metadata=ConstraintTableMetadata(
                format=CONSTRAINT_TABLE_FORMAT,
                artifact_digest=None,
                tokenizer_digest=self.tokenizer_digest,
                vocab_size=resource.vocab_size,
                max_prefix1_degree=resource.max_prefix1_degree,
                max_prefix2_degree=resource.max_prefix2_degree,
                total_bytes=resource.total_bytes,
            ),
        )
        validate_constraint_table(table)
        return table

    def compile_tries(self, documents: Sequence[dict[str, object]]) -> ConstraintTable:
        """Compile Tries in ID order using the existing single-Trie compiler."""
        if not documents:
            raise ValueError("at least one Trie is required")
        if len(documents) * self.vocab_size**2 - 1 > np.iinfo(np.int64).max:
            raise ValueError("multi-trie SID keys exceed int64 capacity")
        # Joining CSR rows shares the two final int32 boundaries across Tries.
        tables = []
        total_bytes = 8 + 4 * (len(documents) + 1)
        for document in documents:
            table = self.compile(document)
            total_bytes += table.total_bytes - 8
            if total_bytes > MAX_CONSTRAINT_TABLE_BYTES:
                raise ValueError("multi-trie constraint table exceeds the artifact size limit")
            tables.append(table)
        first_offsets = np.asarray(
            [0, *np.cumsum([t.first_token_ids.size for t in tables], dtype=np.int64)],
            dtype=np.int32,
        )
        pair_base = value_base = 0
        offsets1 = []
        offsets2 = []
        keys = []
        for trie_id, table in enumerate(tables):
            offsets1.append(table.prefix1_offsets[:-1] + pair_base)
            offsets2.append(table.prefix2_value_offsets[:-1] + value_base)
            keys.append(table.prefix1_pair_keys + trie_id * self.vocab_size**2)
            pair_base += table.prefix1_values.size
            value_base += table.prefix2_values.size
        table = ConstraintTable(
            first_token_ids=np.concatenate([t.first_token_ids for t in tables]),
            first_token_offsets=first_offsets,
            prefix1_offsets=np.concatenate([*offsets1, np.asarray([pair_base], dtype=np.int32)]),
            prefix1_values=np.concatenate([t.prefix1_values for t in tables]),
            prefix1_pair_keys=np.concatenate(keys),
            prefix2_value_offsets=np.concatenate(
                [*offsets2, np.asarray([value_base], dtype=np.int32)]
            ),
            prefix2_values=np.concatenate([t.prefix2_values for t in tables]),
            metadata=ConstraintTableMetadata(
                format=MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
                artifact_digest=None,
                tokenizer_digest=self.tokenizer_digest,
                vocab_size=self.vocab_size,
                max_prefix1_degree=max(t.metadata.max_prefix1_degree for t in tables),
                max_prefix2_degree=max(t.metadata.max_prefix2_degree for t in tables),
                total_bytes=total_bytes,
                trie_count=len(tables),
            ),
        )
        validate_constraint_table(table)
        return table
