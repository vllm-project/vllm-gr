# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Compile a catalog JSON into the validated three-token triples JSON.

Task-5 generation tooling: the business-data source (``catalog.json``) is a
list of token-string sequences.  Constraint Tables require exactly three
semantic tokens per sequence, optionally wrapped by begin/end sentinels.

This compiler:

- validates sentinel consistency and semantic depth via
  :mod:`vllm_gr.constraints.triples`,
- writes the reduced triples as a JSON document that the task-3
  ``ConstraintTableCompiler`` consumes without re-running tokenizer-based
  parsing.

The triples document is tokenizer-independent: it stores token *strings*, not
IDs.  ID assignment is the compiler's (task-3) responsibility because it needs
``model_config.vocab_size`` and the tokenizer.
"""

from __future__ import annotations

import json
from typing import Sequence

from vllm_gr.constraints.triples import (
    ConstraintTriple,
    build_relations,
    strip_sentinels,
    validate_consistent_sentinels,
    validate_semantic_depth,
)

# Schema version of the triples document emitted by this compiler.  Bump when
# the emitted layout changes so digesting and the ConstraintTableCompiler can
# reject documents produced by an incompatible tooling version.
FORMAT_VERSION = 1

# Keys used in the compiled triples document.
FORMAT_VERSION_KEY = "format_version"
TRIPLES_KEY = "triples"
SENTINELS_KEY = "sentinels"
BEGIN_KEY = "begin"
END_KEY = "end"
RELATIONS_KEY = "relations"


def compile_triples_json(
    catalog: Sequence[Sequence[str]],
    *,
    begin_token: str | None = None,
    end_token: str | None = None,
) -> dict:
    """Validate a catalog and reduce it to a triples document.

    Args:
        catalog: list of token-string sequences, e.g. ``[["<|sid_begin|>",
            "<s_a_1>", "<s_b_2>", "<s_c_3>", "<|sid_end|>"], ...]``.
        begin_token: optional begin sentinel.  When set, every sequence must
            start with exactly one copy.
        end_token: optional end sentinel.  When set, every sequence must end
            with exactly one copy.

    Returns:
        ``{"format_version": 1, "triples": [{"a": ..., "b": ..., "c": ...},
        ...], "sentinels": {"begin": ..., "end": ...}}``

    Raises:
        ValueError: on inconsistent sentinels, wrong semantic depth, empty
            catalog, or a sequence that is not a flat list of strings.
    """
    if not catalog:
        raise ValueError("Catalog is empty: Constraint Tables require at least one sequence")

    # validate_consistent_sentinels performs per-sequence shape validation for
    # every entry (even when no sentinel is configured), so there is no need to
    # pre-scan the catalog here.
    validate_consistent_sentinels(catalog, begin_token, end_token)

    triples: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for index, sequence in enumerate(catalog):
        semantic = strip_sentinels(sequence, begin_token, end_token)
        validate_semantic_depth(semantic)

        triple = ConstraintTriple(semantic[0], semantic[1], semantic[2])
        key = triple.values
        if key in seen:
            raise ValueError(
                f"Duplicate triple {list(key)!r} at sequence #{index}; "
                "Constraint Tables require unique (s_a, s_b, s_c) sequences"
            )
        seen.add(key)
        triples.append({"a": triple.a, "b": triple.b, "c": triple.c})

    # Deterministic ordering: sort by (a, b, c) so that the same logical
    # catalog serializes identically regardless of input order.  This keeps the
    # document stable for digesting and for ConstraintTableCompiler input.
    triples.sort(key=lambda t: (t["a"], t["b"], t["c"]))

    document: dict = {
        FORMAT_VERSION_KEY: FORMAT_VERSION,
        TRIPLES_KEY: triples,
        SENTINELS_KEY: {
            BEGIN_KEY: begin_token,
            END_KEY: end_token,
        },
    }
    return document


def compile_relations_json(
    catalog: Sequence[Sequence[str]],
    *,
    begin_token: str | None = None,
    end_token: str | None = None,
) -> dict:
    """Compile a catalog into the binary relation document.

    Wraps :func:`compile_triples_json` and adds the ``first`` / ``a_to_b`` /
    ``ab_to_c`` relations, matching the task document's binary-relation
    contract.  This is the document a ``ConstraintTableCompiler`` consumes
    directly.
    """
    document = compile_triples_json(catalog, begin_token=begin_token, end_token=end_token)
    triples = [ConstraintTriple(t["a"], t["b"], t["c"]) for t in document[TRIPLES_KEY]]
    document[RELATIONS_KEY] = build_relations(triples)
    return document


def compile_triples_json_string(
    catalog: Sequence[Sequence[str]],
    *,
    begin_token: str | None = None,
    end_token: str | None = None,
) -> str:
    """Serialize the compiled triples document to a JSON string."""
    return json.dumps(
        compile_triples_json(catalog, begin_token=begin_token, end_token=end_token),
        indent=2,
        ensure_ascii=False,
    )
