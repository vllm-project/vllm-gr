# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Constraint strategy contracts, implementations, and factory helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from vllm_gr.config import MAX_BEAM_WIDTH, ConstraintBackend
from vllm_gr.utils.catalog import Catalog

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.tokenizers import TokenizerLike

# Constraint Tables implement a fixed 3-level FSM: the constrained prefill
# (step 0) plus two decode steps (1 and 2).  The Worker hard-errors on any
# other step, so the request-side loop is capped to match — never a silent
# fall back to unconstrained generation.
MAX_CONSTRAINT_GENERATION_STEPS = 3
MAX_CONSTRAINT_BEAM_WIDTH = MAX_BEAM_WIDTH


def validate_constraint_generation_limits(*, beam_width: int, max_generate_steps: int) -> None:
    """Fail fast when a Constraint Table request exceeds the fixed FSM limits."""
    if beam_width > MAX_CONSTRAINT_BEAM_WIDTH:
        raise ValueError(
            f"constraint_table beam_width must be <= {MAX_CONSTRAINT_BEAM_WIDTH}, got {beam_width}"
        )
    if max_generate_steps > MAX_CONSTRAINT_GENERATION_STEPS:
        raise ValueError(
            "constraint_table supports at most "
            f"{MAX_CONSTRAINT_GENERATION_STEPS} generated steps "
            "(1 constrained prefill + 2 constrained decode), got "
            f"{max_generate_steps}"
        )


def validate_constraint_table_protocol(
    *,
    is_constraint_table: bool,
    beam_protocol_available: bool,
    required: str,
    hint: str = "",
) -> None:
    """Fail fast when Constraint Tables cannot reach the beam-request path.

    Both execution paths (offline ``GRLLM`` and OpenAI ``beam_search``) have a
    degraded fallback that cannot express Constraint Table step0/1/2 and
    ``beam_suffix``: the baseline ``generate()`` loop when CUSTOM attention is
    absent, and ``_add_batch_step`` when the engine client lacks
    ``beam_request_step_update``.  Under ``constraint_backend=constraint_table``
    those fallbacks must fail fast, never silently produce uncorrected samples.
    """
    if is_constraint_table and not beam_protocol_available:
        suffix = f" {hint}" if hint else ""
        raise RuntimeError(
            "constraint_backend='constraint_table' requires the beam-request "
            f"protocol path ({required}); the fallback path cannot express "
            "Constraint Table step0/1/2 and beam_suffix." + suffix
        )


def validate_beam_request_prefix_cache(
    *, beam_request_active: bool, prefix_caching_enabled: bool | None
) -> None:
    """Reject a beam-request path that cannot inherit the prefill KV.

    Incremental Beam requests carry only the newest semantic token. Their
    shared prompt KV is recovered through vLLM's prefix cache, including the
    specially hashed trailing partial block. Explicitly disabling that cache
    would otherwise make decode silently attend to an incomplete prefix.

    ``None`` means the caller has no authoritative cache config (typically a
    test double); runtime engines always expose the boolean setting.
    """
    if beam_request_active and prefix_caching_enabled is False:
        raise RuntimeError(
            "the beam-request decode path requires prefix caching to reuse "
            "the complete prefill KV; enable_prefix_caching=False would "
            "produce incorrect beam scores"
        )


class ConstraintStrategy(ABC):
    """Base class for request-independent generation constraints."""

    backend: ConstraintBackend
    execution_site: Literal["frontend", "worker"]

    @abstractmethod
    def valid_next_token_ids(self, token_ids: Sequence[int]) -> set[int]:
        """Return valid next token IDs for a generated prefix."""

    @abstractmethod
    def filter_frontend_candidates(
        self,
        token_ids: Sequence[int],
        logprobs: Sequence[float],
        ranks: Sequence[int | None] | None,
        decoded_tokens: Sequence[str | None] | None,
        valid_token_ids: set[int],
        candidate_strategy: str,
    ) -> tuple[
        Sequence[int],
        Sequence[float],
        Sequence[int | None] | None,
        Sequence[str | None] | None,
    ]:
        """Apply a frontend constraint to one parent's candidate row."""


class TrieConstraintStrategy(ConstraintStrategy):
    """Catalog Trie implementation of the common constraint strategy."""

    backend = ConstraintBackend.TRIE
    execution_site = "frontend"

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    @classmethod
    def load(cls, tokenizer: TokenizerLike, catalog_path: str) -> TrieConstraintStrategy:
        catalog = Catalog(tokenizer)
        catalog.load(catalog_path)
        return cls(catalog)

    def valid_next_token_ids(self, token_ids: Sequence[int]) -> set[int]:
        # Beam tokens are already represented as lists. Avoid copying the
        # generated prefix for every beam on every decode step.
        return self.catalog.valid(token_ids)

    def filter_frontend_candidates(
        self,
        token_ids: Sequence[int],
        logprobs: Sequence[float],
        ranks: Sequence[int | None] | None,
        decoded_tokens: Sequence[str | None] | None,
        valid_token_ids: set[int],
        candidate_strategy: str,
    ) -> tuple[
        Sequence[int],
        Sequence[float],
        Sequence[int | None] | None,
        Sequence[str | None] | None,
    ]:
        return self.catalog.apply_catalog_filter(
            token_ids if isinstance(token_ids, list) else list(token_ids),
            logprobs if isinstance(logprobs, list) else list(logprobs),
            (ranks if isinstance(ranks, list) else list(ranks)) if ranks is not None else None,
            (decoded_tokens if isinstance(decoded_tokens, list) else list(decoded_tokens))
            if decoded_tokens is not None
            else None,
            valid_token_ids,
            candidate_strategy,
        )


class ConstraintTableStrategy(ConstraintStrategy):
    """Worker-side strategy for Constraint Table candidates.

    Candidates are generated by ``rec_constrained_topk`` at the raw-logits
    boundary (``execution_site = "worker"``), not filtered in the frontend, so
    this strategy carries no Trie and never runs frontend candidate filtering.
    ``constraint_data_path`` points at the task-5 triples JSON document (the
    ``tools/build_constraint_triples.py`` output); the driver compiles and
    injects it at startup via
    :func:`~vllm_gr.constraints.startup.install_constraint_table`.  ``digest``
    is populated after injection for observability.
    """

    backend = ConstraintBackend.CONSTRAINT_TABLE
    execution_site = "worker"

    def __init__(self, constraint_data_path: str) -> None:
        self.constraint_data_path = constraint_data_path
        self.digest: str | None = None

    def valid_next_token_ids(self, token_ids: Sequence[int]) -> set[int]:
        del token_ids
        raise RuntimeError("Constraint Table candidates are generated in the Worker")

    def filter_frontend_candidates(
        self,
        token_ids: Sequence[int],
        logprobs: Sequence[float],
        ranks: Sequence[int | None] | None,
        decoded_tokens: Sequence[str | None] | None,
        valid_token_ids: set[int],
        candidate_strategy: str,
    ) -> tuple[
        Sequence[int],
        Sequence[float],
        Sequence[int | None] | None,
        Sequence[str | None] | None,
    ]:
        del token_ids, logprobs, ranks, decoded_tokens, valid_token_ids, candidate_strategy
        raise RuntimeError("Constraint Table does not apply frontend candidate filtering")


def create_constraint_strategy(
    *,
    backend: ConstraintBackend | str,
    tokenizer: TokenizerLike,
    catalog_path: str | None,
) -> ConstraintStrategy | None:
    """Create the configured strategy without silently changing backends.

    ``catalog_path`` may be ``None`` or the
    :data:`~vllm_gr.config.models.AUTO_CONSTRAINT_TABLE_PATH` sentinel for the
    Constraint Table backend: the tables are auto-built from the tokenizer's
    s_a/s_b/s_c groups by ``install_constraint_table``, so no triples document
    is required.  Trie still requires a real catalog path.
    """
    from vllm_gr.config.models import AUTO_CONSTRAINT_TABLE_PATH

    normalized_backend = (
        backend if isinstance(backend, ConstraintBackend) else ConstraintBackend(backend.lower())
    )
    if catalog_path is None or catalog_path == AUTO_CONSTRAINT_TABLE_PATH:
        if normalized_backend is ConstraintBackend.CONSTRAINT_TABLE:
            return ConstraintTableStrategy(catalog_path)
        return None
    if normalized_backend is ConstraintBackend.TRIE:
        return TrieConstraintStrategy.load(tokenizer, catalog_path)

    # Constraint Tables generate candidates in the Worker; the strategy is a
    # frontend handle that records the data path and, after startup injection,
    # the compiled resource digest.  The worker runtime is wired by
    # install_constraint_table (constraints/startup.py), never by silently
    # running the Trie path.
    return ConstraintTableStrategy(catalog_path)
