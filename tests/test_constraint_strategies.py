# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_gr.config import ConstraintBackend
from vllm_gr.constraints import (
    BeamConstraintCoordinator,
    CandidateLayout,
    ConstraintTableStrategy,
    ParentCandidateBuilder,
    TrieConstraintStrategy,
    create_constraint_strategy,
)
from vllm_gr.constraints.strategies import (
    validate_constraint_generation_limits,
    validate_constraint_table_protocol,
)


@pytest.fixture  # type: ignore[misc]
def tokenizer() -> MagicMock:
    result = MagicMock()
    result.convert_tokens_to_ids.side_effect = lambda values: [int(value) for value in values]
    return result


def test_trie_factory_loads_catalog_and_preserves_compatibility(
    tokenizer: MagicMock,
) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as file:
        json.dump([["1", "2", "3"], ["1", "4", "5"]], file)
        catalog_path = file.name
    try:
        strategy = create_constraint_strategy(
            backend="TRIE",
            tokenizer=tokenizer,
            catalog_path=catalog_path,
        )

        assert isinstance(strategy, TrieConstraintStrategy)
        assert strategy.valid_next_token_ids([]) == {1}
        assert strategy.valid_next_token_ids([1]) == {2, 4}
    finally:
        os.remove(catalog_path)


def test_constraint_table_factory_returns_strategy(tokenizer: MagicMock) -> None:
    strategy = create_constraint_strategy(
        backend=ConstraintBackend.CONSTRAINT_TABLE,
        tokenizer=tokenizer,
        catalog_path="catalog.json",
    )

    assert isinstance(strategy, ConstraintTableStrategy)
    assert strategy.constraint_data_path == "catalog.json"
    assert strategy.digest is None  # populated only after startup injection
    # Candidate generation lives in the Worker; the frontend handle never
    # filters candidates, so the strategy must refuse frontend APIs.
    with pytest.raises(RuntimeError, match="Worker"):
        strategy.valid_next_token_ids([1])
    with pytest.raises(RuntimeError, match="does not apply frontend"):
        strategy.filter_frontend_candidates(
            token_ids=[1],
            logprobs=[-0.1],
            ranks=None,
            decoded_tokens=None,
            valid_token_ids=set(),
            candidate_strategy="fill",
        )


def test_parent_candidate_builder_keeps_ragged_parent_mapping() -> None:
    builder = ParentCandidateBuilder(CandidateLayout.STANDARD_SAMPLER)
    builder.add_parent(
        parent_index=0,
        parent_cumulative_logprob=-0.5,
        token_ids=[],
        local_logprobs=[],
        ranks=None,
        decoded_tokens=None,
    )
    builder.add_parent(
        parent_index=1,
        parent_cumulative_logprob=-0.2,
        token_ids=[7, 8],
        local_logprobs=[-0.1, -0.3],
        ranks=[1, 2],
        decoded_tokens=["7", "8"],
    )

    batch = builder.build()

    assert batch.token_ids.tolist() == [7, 8]
    assert batch.cumulative_logprobs == pytest.approx(np.asarray([-0.3, -0.5]))
    assert batch.parent_indices.tolist() == [1, 1]
    assert batch.parent_offsets.tolist() == [0, 0, 2]
    assert len(batch.parent_cache) == 2


def test_resolve_catalog_strategy_no_strategy_passes_through() -> None:
    coordinator = BeamConstraintCoordinator(None)

    assert coordinator.resolve_catalog_strategy(None) is None
    assert coordinator.resolve_catalog_strategy("fill") == "fill"
    assert coordinator.resolve_catalog_strategy("null") == "null"


def test_resolve_catalog_strategy_trie_normalizes_none_to_fill() -> None:
    coordinator = BeamConstraintCoordinator(TrieConstraintStrategy(object()))

    # None is the new default and means the historical "fill" for the Trie.
    assert coordinator.resolve_catalog_strategy(None) == "fill"
    assert coordinator.resolve_catalog_strategy("fill") == "fill"
    assert coordinator.resolve_catalog_strategy("null") == "null"


def test_resolve_catalog_strategy_constraint_table_rejects_explicit_value() -> None:
    coordinator = BeamConstraintCoordinator(ConstraintTableStrategy("catalog.json"))

    assert coordinator.resolve_catalog_strategy(None) is None
    for rejected in ("fill", "null"):
        with pytest.raises(ValueError, match="Worker"):
            coordinator.resolve_catalog_strategy(rejected)


def test_constraint_generation_limits_accept_valid_fsm_window() -> None:
    # 3 generated steps = 1 constrained prefill + 2 constrained decode.
    validate_constraint_generation_limits(beam_width=2, max_generate_steps=3)
    validate_constraint_generation_limits(beam_width=256, max_generate_steps=0)
    validate_constraint_generation_limits(beam_width=1, max_generate_steps=1)


def test_constraint_generation_limits_reject_oversized_requests() -> None:
    with pytest.raises(ValueError, match="beam_width"):
        validate_constraint_generation_limits(beam_width=257, max_generate_steps=3)
    with pytest.raises(ValueError, match="3 generated steps"):
        validate_constraint_generation_limits(beam_width=2, max_generate_steps=4)


def test_constraint_table_protocol_passes_when_beam_protocol_available() -> None:
    validate_constraint_table_protocol(
        is_constraint_table=True,
        beam_protocol_available=True,
        required="the CUSTOM attention backend",
    )  # must not raise
    validate_constraint_table_protocol(
        is_constraint_table=False,
        beam_protocol_available=False,
        required="anything",
    )  # unconstrained requests may use the fallback path


def test_constraint_table_protocol_fails_fast_when_fallback_unavailable() -> None:
    """The degraded fallback cannot express step0/1/2 + beam_suffix.

    Regression for the review finding: under constraint_table the offline
    baseline ``generate()`` loop and the serving ``_add_batch_step`` fallback
    must fail fast instead of silently producing uncorrected samples.
    """
    with pytest.raises(RuntimeError, match="cannot express"):
        validate_constraint_table_protocol(
            is_constraint_table=True,
            beam_protocol_available=False,
            required="an engine client implementing beam_request_step_update",
        )


def test_constraint_table_protocol_appends_hint_when_given() -> None:
    with pytest.raises(RuntimeError, match="Configure attention_config"):
        validate_constraint_table_protocol(
            is_constraint_table=True,
            beam_protocol_available=False,
            required="the CUSTOM attention backend",
            hint="Configure attention_config={'backend': 'CUSTOM'}.",
        )


def test_coordinator_applies_fill_and_null_with_the_same_contract(
    tokenizer: MagicMock,
) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as file:
        json.dump([["1", "2"], ["1", "4"]], file)
        catalog_path = file.name
    try:
        strategy = TrieConstraintStrategy.load(tokenizer, catalog_path)
        coordinator = BeamConstraintCoordinator(strategy)
        valid = coordinator.valid_next_token_ids([1])

        fill_tokens, fill_logprobs, _, _ = coordinator.filter_candidates(
            token_ids=[9, 2],
            logprobs=[-0.1, -0.2],
            ranks=None,
            decoded_tokens=None,
            valid_token_ids=valid,
            candidate_strategy="fill",
        )
        null_tokens, null_logprobs, _, _ = coordinator.filter_candidates(
            token_ids=[9, 2],
            logprobs=[-0.1, -0.2],
            ranks=None,
            decoded_tokens=None,
            valid_token_ids=valid,
            candidate_strategy="null",
        )

        assert fill_tokens == [2, 2]
        assert fill_logprobs == pytest.approx([-10000.0, -0.2])
        assert null_tokens == [9, 2]
        assert null_logprobs[0] == -float("inf")
        assert null_logprobs[1] == pytest.approx(-0.2)
    finally:
        os.remove(catalog_path)
