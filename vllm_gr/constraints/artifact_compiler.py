# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility compiler for the canonical constraint-table artifact ABI."""

from __future__ import annotations

from typing import Any

from vllm_gr.config.models import CONSTRAINT_TABLE_FORMAT
from vllm_gr.config.validation import validate_tokenizer_digest
from vllm_gr.constraints.compiler import ConstraintTableCompiler
from vllm_gr.constraints.table import (
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
