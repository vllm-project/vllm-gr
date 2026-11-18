# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_gr.constraints.artifact_compiler import (
    ConstraintTableArtifactCompiler,
)
from vllm_gr.constraints.compiler import (
    ConstraintTableCompiler,
    ConstraintTableResource,
)
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.constraints.table import (
    CONSTRAINT_TABLE_TENSOR_NAMES,
    ConstraintTable,
)

_TOKENIZER_DIGEST = "sha256:" + "b" * 64


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {str(i): i for i in range(20)}
    tokenizer.convert_tokens_to_ids.side_effect = lambda values: [int(value) for value in values]
    return tokenizer


def test_artifact_compiler_wraps_legacy_six_tables_exactly() -> None:
    tokenizer = _tokenizer()
    document = compile_relations_json([["1", "2", "3"], ["1", "4", "5"], ["7", "8", "9"]])

    legacy = ConstraintTableCompiler(tokenizer, 20).compile(document)
    canonical = ConstraintTableArtifactCompiler(
        tokenizer,
        20,
        tokenizer_digest=_TOKENIZER_DIGEST,
    ).compile(document)

    assert isinstance(legacy, ConstraintTableResource)
    assert isinstance(canonical, ConstraintTable)
    for name in CONSTRAINT_TABLE_TENSOR_NAMES:
        assert np.array_equal(getattr(canonical, name), getattr(legacy, name))
        assert getattr(canonical, name).dtype == getattr(legacy, name).dtype
    assert canonical.metadata.artifact_digest is None
    assert canonical.metadata.tokenizer_digest == _TOKENIZER_DIGEST
    assert canonical.metadata.vocab_size == legacy.vocab_size
    assert canonical.metadata.max_prefix1_degree == legacy.max_prefix1_degree
    assert canonical.metadata.max_prefix2_degree == legacy.max_prefix2_degree
    assert canonical.metadata.total_bytes == legacy.total_bytes


def test_legacy_compiler_return_type_remains_unchanged() -> None:
    resource = ConstraintTableCompiler(_tokenizer(), 20).compile(
        compile_relations_json([["1", "2", "3"]])
    )
    assert type(resource) is ConstraintTableResource


pytestmark = pytest.mark.cpu_test
