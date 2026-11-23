# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Validate three-token catalogs and produce tokenizer-independent triples documents."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ConstraintTriple:
    """One business sequence reduced to its three semantic tokens."""

    a: str
    b: str
    c: str

    def __post_init__(self) -> None:
        for field_name in ("a", "b", "c"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ValueError(
                    f"ConstraintTriple field {field_name!r} must be a str, "
                    f"got {type(value).__name__}"
                )

    @property
    def values(self) -> tuple[str, str, str]:
        return (self.a, self.b, self.c)


def validate_sequence_shape(sequence: object, index: int) -> None:
    """Require a catalog sequence to be a list or tuple of token strings.

    This guard runs before any sentinel inspection so that malformed entries
    (``None``, integers, nested lists) raise a ``ValueError`` instead of an
    uncaught ``TypeError`` deep inside ``list(sequence)`` or token counting.

    The parameter is typed ``object`` rather than ``Sequence[Any]`` because
    this is a runtime boundary check: callers may pass ``None``, integers, or
    other non-sequence objects, none of which are ``Sequence`` instances.
    """
    if not isinstance(sequence, (list, tuple)):
        raise ValueError(
            f"Catalog sequence #{index} must be a sequence of token strings, "
            f"got {type(sequence).__name__}"
        )

    non_strings = {type(token).__name__ for token in sequence if not isinstance(token, str)}
    if non_strings:
        raise ValueError(
            f"Catalog sequence #{index} must contain only token strings, "
            f"found element types: {sorted(non_strings)}"
        )


def strip_sentinels(
    sequence: Sequence[str],
    begin_token: str | None = None,
    end_token: str | None = None,
) -> list[str]:
    """Remove consistent begin/end sentinels from a catalog sequence.

    A sequence may be wrapped by exactly one begin token and one end token.
    When a sentinel is configured and present on every sequence, it is
    stripped.  Missing sentinels are tolerated only when the sequence already
    has exactly three semantic tokens.

    Returns the remaining tokens with sentinels removed.
    """
    tokens = list(sequence)
    if begin_token is not None and tokens and tokens[0] == begin_token:
        tokens = tokens[1:]
    if end_token is not None and tokens and tokens[-1] == end_token:
        tokens = tokens[:-1]
    return tokens


def validate_semantic_depth(tokens: Sequence[str]) -> None:
    """Require exactly three semantic tokens after sentinel stripping."""
    if len(tokens) != 3:
        raise ValueError(
            "Constraint Tables require exactly three semantic tokens per sequence "
            f"([s_a, s_b, s_c]) but got {len(tokens)}: {list(tokens)!r}. "
            "Catalogs with arbitrary depth must use the Trie backend."
        )


def build_relations(
    triples: Sequence[ConstraintTriple],
) -> dict:
    """Build the binary ``first`` / ``a_to_b`` / ``ab_to_c`` relations.

    These are the three lookup relations a three-level FSM needs, expressed as
    plain JSON-able data keyed by token *strings*:

    - ``first``: sorted unique ``s_a`` tokens.
    - ``a_to_b``: ``{s_a: [sorted unique s_b, ...], ...}``.
    - ``ab_to_c``: ``{json.dumps([s_a, s_b]): [sorted unique s_c, ...], ...}``.

    The ``ab_to_c`` key is the JSON array ``[s_a, s_b]`` (e.g. ``'["a","b"]'``)
    rather than a comma-joined string so that tokens containing commas cannot
    collide (``("a,b", "c")`` vs ``("a", "b,c")`` would both become
    ``"a,b,c"``).

    """
    first: set[str] = set()
    a_to_b: dict[str, set[str]] = {}
    ab_to_c: dict[str, set[str]] = {}

    for triple in triples:
        first.add(triple.a)
        a_to_b.setdefault(triple.a, set()).add(triple.b)
        ab_to_c.setdefault(json.dumps([triple.a, triple.b]), set()).add(triple.c)

    return {
        "first": sorted(first),
        "a_to_b": {a: sorted(bs) for a, bs in sorted(a_to_b.items())},
        "ab_to_c": {key: sorted(cs) for key, cs in sorted(ab_to_c.items())},
    }


def validate_consistent_sentinels(
    sequences: Sequence[Sequence[str]],
    begin_token: str | None,
    end_token: str | None,
) -> None:
    """Require sentinels to be either absent everywhere or present everywhere.

    Every sequence is shape-validated first (list/tuple of token strings) so a
    malformed entry fails with a ``ValueError`` even when no sentinel is
    configured.  When a sentinel is configured, every sequence must carry
    exactly one copy at the corresponding end; a sequence with zero or multiple
    copies is rejected so that stripping never produces ambiguous triples.
    """
    for index, sequence in enumerate(sequences):
        validate_sequence_shape(sequence, index)

        if begin_token is None and end_token is None:
            continue

        tokens = list(sequence)
        if begin_token is not None:
            count = tokens.count(begin_token)
            if count != 1:
                raise ValueError(
                    f"Sequence #{index} must contain exactly one begin sentinel "
                    f"{begin_token!r} (got {count})"
                )
            if tokens[0] != begin_token:
                raise ValueError(
                    f"Sequence #{index} has begin sentinel {begin_token!r} at the wrong "
                    "position; it must be the first token"
                )
        if end_token is not None:
            count = tokens.count(end_token)
            if count != 1:
                raise ValueError(
                    f"Sequence #{index} must contain exactly one end sentinel "
                    f"{end_token!r} (got {count})"
                )
            if tokens[-1] != end_token:
                raise ValueError(
                    f"Sequence #{index} has end sentinel {end_token!r} at the wrong "
                    "position; it must be the last token"
                )


def _compile_catalog(
    catalog: Sequence[Sequence[str]],
    begin_token: str | None,
    end_token: str | None,
    *,
    include_relations: bool,
) -> dict:
    if not catalog:
        raise ValueError("Catalog is empty: Constraint Tables require at least one sequence")

    validate_consistent_sentinels(catalog, begin_token, end_token)
    triples: set[ConstraintTriple] = set()
    for index, sequence in enumerate(catalog):
        semantic = strip_sentinels(sequence, begin_token, end_token)
        validate_semantic_depth(semantic)
        triple = ConstraintTriple(*semantic)
        if triple in triples:
            raise ValueError(
                f"Duplicate triple {list(triple.values)!r} at sequence #{index}; "
                "Constraint Tables require unique (s_a, s_b, s_c) sequences"
            )
        triples.add(triple)

    ordered = sorted(triples, key=lambda triple: triple.values)
    del triples
    document = {
        FORMAT_VERSION_KEY: FORMAT_VERSION,
        TRIPLES_KEY: [{"a": t.a, "b": t.b, "c": t.c} for t in ordered],
        SENTINELS_KEY: {BEGIN_KEY: begin_token, END_KEY: end_token},
    }
    if include_relations:
        document[RELATIONS_KEY] = build_relations(ordered)
    return document


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
    return _compile_catalog(catalog, begin_token, end_token, include_relations=False)


def compile_relations_json(
    catalog: Sequence[Sequence[str]],
    *,
    begin_token: str | None = None,
    end_token: str | None = None,
) -> dict:
    """Compile a triples document with the lookup relations used by the table compiler."""
    return _compile_catalog(catalog, begin_token, end_token, include_relations=True)


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
