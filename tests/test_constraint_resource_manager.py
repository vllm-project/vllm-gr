# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Tests for ConstraintResourceManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm_gr.constraints.compiler import (
    ConstraintTableCompiler,
    ConstraintTableResource,
)
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.constraints.resource_manager import ConstraintResourceManager


@pytest.fixture  # type: ignore[misc]
def tokenizer() -> MagicMock:
    result = MagicMock()
    result.convert_tokens_to_ids.side_effect = lambda values: [int(value) for value in values]
    result.get_vocab.return_value = {str(i): i for i in range(100)}
    return result


def _build_resource(tokenizer: MagicMock, vocab_size: int = 20) -> ConstraintTableResource:
    document = compile_relations_json([["1", "2", "3"], ["1", "4", "5"]])
    return ConstraintTableCompiler(tokenizer, vocab_size).compile(document)


def test_starts_disabled(tokenizer: MagicMock) -> None:
    manager = ConstraintResourceManager()
    assert not manager.enabled
    assert manager.current_digest is None
    assert manager.resources == {}


def test_ensure_caches_and_enables(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    manager = ConstraintResourceManager()

    manager.ensure(resource)

    assert manager.enabled
    assert manager.current_digest == resource.digest
    assert manager.get(resource.digest) is resource


def test_ensure_same_digest_is_idempotent(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    manager = ConstraintResourceManager()

    manager.ensure(resource)
    manager.ensure(resource)  # same digest -> no-op, no error

    assert manager.current_digest == resource.digest
    assert manager.get(resource.digest) is resource


def test_ensure_different_digest_rejected(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer, vocab_size=20)
    manager = ConstraintResourceManager()
    manager.ensure(resource)

    other = _build_resource(tokenizer, vocab_size=30)  # different digest
    with pytest.raises(ValueError, match="cannot replace loaded constraint tables"):
        manager.ensure(other)


def test_require_returns_resource(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    manager = ConstraintResourceManager()
    manager.ensure(resource)

    assert manager.require(resource.digest) is resource


def test_require_unknown_digest_raises(tokenizer: MagicMock) -> None:
    manager = ConstraintResourceManager()

    with pytest.raises(ValueError, match="is not loaded"):
        manager.require("unknown-digest")


def test_clear_resets_state(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    manager = ConstraintResourceManager()
    manager.ensure(resource)

    manager.clear()

    assert not manager.enabled
    assert manager.current_digest is None
    assert manager.resources == {}


def test_clear_then_reload_allowed(tokenizer: MagicMock) -> None:
    """After clear(), a different digest can be injected again."""
    first = _build_resource(tokenizer, vocab_size=20)
    manager = ConstraintResourceManager()
    manager.ensure(first)
    manager.clear()

    second = _build_resource(tokenizer, vocab_size=30)
    manager.ensure(second)  # allowed after explicit clear

    assert manager.current_digest == second.digest


def test_manager_and_execution_paired_lifecycle(tokenizer: MagicMock) -> None:
    """Full paired flow: ensure/load -> clear/reset -> ensure/load, digests in sync."""
    from vllm_gr.constraints.execution import TorchConstrainedTopKExecution

    resource_a = _build_resource(tokenizer, vocab_size=20)
    resource_b = _build_resource(tokenizer, vocab_size=30)

    manager = ConstraintResourceManager()
    execution = TorchConstrainedTopKExecution(device="cpu")

    # Step 1: load A on both sides.
    manager.ensure(resource_a)
    execution.load_tables(resource_a)
    assert manager.current_digest == resource_a.digest
    assert execution._digest == resource_a.digest  # noqa: SLF001

    # Different digest on execution is rejected until reset.
    with pytest.raises(ValueError, match="reset"):
        execution.load_tables(resource_b)

    # Step 2: paired reset then load B.
    manager.clear()
    execution.reset()
    assert manager.current_digest is None
    assert execution._digest is None  # noqa: SLF001

    manager.ensure(resource_b)
    execution.load_tables(resource_b)
    assert manager.current_digest == resource_b.digest
    assert execution._digest == resource_b.digest  # noqa: SLF001

    # B's tables are usable for a run (vocab_size=30).
    import torch

    result = execution.run(
        torch.randn(2, 30), torch.zeros(2, 1, 1, dtype=torch.int32), torch.zeros(0), 0, 2
    )
    assert result.top_tokens.dtype == torch.int32


pytestmark = pytest.mark.cpu_test
