# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Compile triples documents for the canonical constraint-table artifact ABI."""

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
    """Build single-root and multi-root constraint tables for artifact serialization."""

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
        """Compile one Trie directly into a canonical table."""

        compiler = ConstraintTableCompiler(self.tokenizer, self.vocab_size)
        arrays, max_prefix1_degree, max_prefix2_degree = compiler.compile_arrays(document)
        table = ConstraintTable(
            **arrays._asdict(),
            metadata=ConstraintTableMetadata(
                format=CONSTRAINT_TABLE_FORMAT,
                artifact_digest=None,
                tokenizer_digest=self.tokenizer_digest,
                vocab_size=self.vocab_size,
                max_prefix1_degree=max_prefix1_degree,
                max_prefix2_degree=max_prefix2_degree,
                total_bytes=sum(array.nbytes for array in arrays),
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
        compiler = ConstraintTableCompiler(self.tokenizer, self.vocab_size)
        roots = []
        first_offsets = [0]
        pair_base = value_base = 0
        max_prefix1_degree = max_prefix2_degree = 0
        # Root offsets plus the two terminal int32 offsets shared by all roots.
        total_bytes = 4 * (len(documents) + 1) + 8
        for trie_id, document in enumerate(documents):
            arrays, degree1, degree2 = compiler.compile_arrays(document)
            total_bytes += sum(array.nbytes for array in arrays) - 8
            if total_bytes > MAX_CONSTRAINT_TABLE_BYTES:
                raise ValueError("multi-trie constraint table exceeds the artifact size limit")

            # These arrays are private to this compilation. Rebase them before joining roots.
            arrays.prefix1_offsets[:] += pair_base
            arrays.prefix2_value_offsets[:] += value_base
            arrays.prefix1_pair_keys[:] += trie_id * self.vocab_size**2
            first_offsets.append(first_offsets[-1] + arrays.first_token_ids.size)
            pair_base += arrays.prefix1_values.size
            value_base += arrays.prefix2_values.size
            max_prefix1_degree = max(max_prefix1_degree, degree1)
            max_prefix2_degree = max(max_prefix2_degree, degree2)
            roots.append(arrays)

        # Keep the final CSR boundary only for the last root.
        pair_offsets = [r.prefix1_offsets[:-1] for r in roots[:-1]]
        value_offsets = [r.prefix2_value_offsets[:-1] for r in roots[:-1]]
        table = ConstraintTable(
            first_token_ids=np.concatenate([r.first_token_ids for r in roots]),
            first_token_offsets=np.asarray(first_offsets, dtype=np.int32),
            prefix1_offsets=np.concatenate([*pair_offsets, roots[-1].prefix1_offsets]),
            prefix1_values=np.concatenate([r.prefix1_values for r in roots]),
            prefix1_pair_keys=np.concatenate([r.prefix1_pair_keys for r in roots]),
            prefix2_value_offsets=np.concatenate([*value_offsets, roots[-1].prefix2_value_offsets]),
            prefix2_values=np.concatenate([r.prefix2_values for r in roots]),
            metadata=ConstraintTableMetadata(
                format=MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
                artifact_digest=None,
                tokenizer_digest=self.tokenizer_digest,
                vocab_size=self.vocab_size,
                max_prefix1_degree=max_prefix1_degree,
                max_prefix2_degree=max_prefix2_degree,
                total_bytes=total_bytes,
                trie_count=len(roots),
            ),
        )
        validate_constraint_table(table)
        return table
