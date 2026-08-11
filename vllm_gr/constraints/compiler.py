# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compile the constraint triples document into ACS-compatible lookup tables.

The ``ConstraintTableCompiler`` consumes the task-5 triples document (see
:mod:`vllm_gr.constraints.json_compiler`) and, together with the model
tokenizer and ``model_config.vocab_size``, produces the eight ACS-operator
tables consumed by ``rec_constrained_topk``:

- ``first_token_ids``: valid ``s_a`` token IDs for step 0.
- ``prefix1_offsets``: CSR offsets ``[vocab_size + 1]`` into ``prefix1_values``.
- ``prefix1_values``: valid ``s_b`` token IDs for step 1, grouped by ``s_a``.
- ``prefix1_pair_keys``: ``s_a * vocab_size + s_b`` (int64) for binary search.
- ``prefix2_value_offsets``: offsets ``[total_prefix1 + 1]`` into ``prefix2_values``.
- ``prefix2_values``: valid ``s_c`` token IDs for step 2, grouped by ``(s_a, s_b)``.
- ``max_prefix1_degree`` / ``max_prefix2_degree``: per-level max candidate count.

Layout and dtype follow the ACS ``rec_constrained_topk`` operator contract:
``pair_key`` is ``int64``; offsets and values are ``int32``.

The compiler performs the strict validation listed in the task document:
token-ID range, CSR offset monotonicity, pair-key ordering, dtypes, degree
derived-consistency, and total resource size.  ``top_k <= 256`` is a runtime /
operator constraint (validated at execution), not a compile-time limit on the
candidate degree: a prefix may have more valid continuations than the per-row
``top_k``, and the operator's bucketed top-k handles that.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm_gr.constraints.json_compiler import (
    FORMAT_VERSION,
    FORMAT_VERSION_KEY,
    RELATIONS_KEY,
    TRIPLES_KEY,
)
from vllm_gr.constraints.triples import ConstraintTriple, build_relations

# Maximum number of candidates per beam accepted by the ACS operator.
MAX_TOP_K = 256


@dataclass(frozen=True)
class ConstraintTableResource:
    """Immutable device-agnostic constraint tables plus their content digest.

    Values are plain NumPy arrays so the resource is CPU-resident until the
    platform layer moves it to device.  ``digest`` covers the normalized
    tables, ``vocab_size``, and the compiler format version, so Workers can
    load idempotently: same digest reuses, different digest rejects.
    """

    first_token_ids: np.ndarray
    prefix1_offsets: np.ndarray
    prefix1_values: np.ndarray
    prefix1_pair_keys: np.ndarray
    prefix2_value_offsets: np.ndarray
    prefix2_values: np.ndarray
    max_prefix1_degree: int
    max_prefix2_degree: int
    vocab_size: int
    digest: str = ""

    def __post_init__(self) -> None:
        # Freeze every array over an immutable bytes buffer.  ``frozen=True``
        # only guards field reassignment, and a plain ``setflags(write=False)``
        # array can still be re-enabled by its owner via ``setflags(write=True)``.
        # Building the array over an immutable ``bytes`` buffer makes both
        # reassignment and in-place mutation impossible, so a caller cannot
        # corrupt a table without the digest changing.
        for field_name in (
            "first_token_ids",
            "prefix1_offsets",
            "prefix1_values",
            "prefix1_pair_keys",
            "prefix2_value_offsets",
            "prefix2_values",
        ):
            array = getattr(self, field_name)
            frozen = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
            object.__setattr__(self, field_name, frozen)

        object.__setattr__(self, "digest", compute_resource_digest(self))


def compute_resource_digest(resource: ConstraintTableResource) -> str:
    """Stable content digest over tables, degrees, vocab_size, and format version.

    ``max_prefix1_degree`` / ``max_prefix2_degree`` are included because the
    operator uses them as candidate-width upper bounds; a silently-shrunk degree
    would truncate candidates without changing any array bytes.
    """
    hasher = hashlib.sha256()
    for array in (
        resource.first_token_ids,
        resource.prefix1_offsets,
        resource.prefix1_values,
        resource.prefix1_pair_keys,
        resource.prefix2_value_offsets,
        resource.prefix2_values,
    ):
        # Include dtype and shape so a reshape (e.g. [N] -> [N, 1]) changes the
        # digest even though the underlying bytes are identical.  The digest
        # must represent the full device-table layout the ABI depends on.
        hasher.update(str(array.dtype).encode())
        hasher.update(str(tuple(array.shape)).encode())
        hasher.update(np.ascontiguousarray(array).tobytes())
    hasher.update(str(resource.max_prefix1_degree).encode())
    hasher.update(str(resource.max_prefix2_degree).encode())
    hasher.update(str(resource.vocab_size).encode())
    hasher.update(str(FORMAT_VERSION).encode())
    return hasher.hexdigest()


def derive_degrees(
    prefix1_offsets: np.ndarray, prefix2_value_offsets: np.ndarray
) -> tuple[int, int]:
    """Derive max per-level degrees from the CSR offsets.

    These are the true degrees the operator observes; callers should compare
    them against any stored/payload degree so a mismatch is caught.
    """
    prefix1_degree = int(np.diff(prefix1_offsets).max()) if prefix1_offsets.size > 1 else 0
    prefix2_degree = (
        int(np.diff(prefix2_value_offsets).max()) if prefix2_value_offsets.size > 1 else 0
    )
    return prefix1_degree, prefix2_degree


def validate_resource_tables(resource: ConstraintTableResource) -> None:
    """Validate the full table contract of a resource.

    Run by the compiler after building a resource so an invalid table layout is
    rejected before it reaches any operator.  Checks:

    - dtype contract: five int32 tables, one int64 pair-key table;
    - CSR offset monotonicity / bounds;
    - pair-key strict ordering;
    - token-ID range vs ``vocab_size``;
    - degree bounds and consistency with the values derived from the offsets.
    """
    vocab_size = resource.vocab_size
    prefix1_offsets = resource.prefix1_offsets
    prefix1_values = resource.prefix1_values
    prefix1_pair_keys = resource.prefix1_pair_keys
    prefix2_value_offsets = resource.prefix2_value_offsets
    prefix2_values = resource.prefix2_values

    expected_dtypes = {
        "first_token_ids": np.dtype(np.int32),
        "prefix1_offsets": np.dtype(np.int32),
        "prefix1_values": np.dtype(np.int32),
        "prefix1_pair_keys": np.dtype(np.int64),
        "prefix2_value_offsets": np.dtype(np.int32),
        "prefix2_values": np.dtype(np.int32),
    }
    for name, want in expected_dtypes.items():
        array = getattr(resource, name)
        if array.ndim != 1:
            raise ValueError(
                f"table {name} must be 1-D (NPU ABI), got {array.ndim} dims "
                f"with shape {tuple(array.shape)}"
            )
        if array.dtype != want:
            raise ValueError(f"table {name}.dtype must be {want}, got {array.dtype}")

    # CSR offset monotonicity and bounds.
    if prefix1_offsets.shape != (vocab_size + 1,):
        raise ValueError(
            f"prefix1_offsets must have shape (vocab_size+1,), got {prefix1_offsets.shape}"
        )
    if len(prefix1_offsets) == 0 or prefix1_offsets[0] != 0:
        raise ValueError("prefix1_offsets must start at zero")
    if int(prefix1_offsets[-1]) != len(prefix1_values):
        raise ValueError(
            "prefix1_offsets end must equal the prefix1_values count "
            f"({int(prefix1_offsets[-1])} != {len(prefix1_values)})"
        )
    if np.any(np.diff(prefix1_offsets) < 0):
        raise ValueError("prefix1_offsets must be non-decreasing")

    if len(prefix2_value_offsets) != len(prefix1_values) + 1:
        raise ValueError(
            "prefix2_value_offsets must have one entry per (s_a, s_b) pair plus the end"
        )
    if int(prefix2_value_offsets[0]) != 0:
        raise ValueError("prefix2_value_offsets must start at zero")
    if int(prefix2_value_offsets[-1]) != len(prefix2_values):
        raise ValueError("prefix2_value_offsets end must equal the prefix2_values count")
    if np.any(np.diff(prefix2_value_offsets) < 0):
        raise ValueError("prefix2_value_offsets must be non-decreasing")

    # Pair keys must be strictly increasing for binary search.
    if len(prefix1_pair_keys) > 1 and np.any(np.diff(prefix1_pair_keys) <= 0):
        raise ValueError("prefix1_pair_keys must be strictly increasing")

    # Degree derived-consistency.  Note: degrees are NOT bounded by MAX_TOP_K;
    # a prefix may have far more valid continuations than the per-row output
    # top_k, and the operator's bucketed top-k handles that.  The top_k <= 256
    # limit is a runtime/operator concern, validated at execution time.
    derived1, derived2 = derive_degrees(prefix1_offsets, prefix2_value_offsets)
    if resource.max_prefix1_degree != derived1:
        raise ValueError(
            f"max_prefix1_degree {resource.max_prefix1_degree} does not match "
            f"derived degree {derived1} from prefix1_offsets"
        )
    if resource.max_prefix2_degree != derived2:
        raise ValueError(
            f"max_prefix2_degree {resource.max_prefix2_degree} does not match "
            f"derived degree {derived2} from prefix2_value_offsets"
        )

    # Token IDs in-range.
    for name, array in (
        ("first_token_ids", resource.first_token_ids),
        ("prefix1_values", prefix1_values),
        ("prefix2_values", prefix2_values),
    ):
        if array.size and (array.min() < 0 or array.max() >= vocab_size):
            raise ValueError(f"{name} contains an out-of-range token ID")

    # Cross-table semantic relations that keep step 0 / 1 / 2 consistent.
    first = resource.first_token_ids
    if first.size and (np.any(np.diff(first) <= 0) or len(np.unique(first)) != len(first)):
        raise ValueError("first_token_ids must be strictly increasing and unique")

    # first_token_ids must be exactly the tokens whose CSR bucket has degree > 0.
    # This comparison is unconditional: an empty first_token_ids must NOT be
    # accepted when offsets have positive-degree buckets (it would silently
    # disable the step-0 table while steps 1/2 still work).
    offsets = prefix1_offsets
    degree_mask = np.diff(offsets) > 0
    active_tokens = np.flatnonzero(degree_mask)
    if not np.array_equal(first, active_tokens):
        raise ValueError(
            "first_token_ids must exactly match the prefix1_offsets tokens with positive degree"
        )

    # pair_keys must align one-to-one with prefix1_values and follow the
    # a_id * vocab_size + b_id formula for the owning CSR bucket.
    if len(prefix1_pair_keys) != len(prefix1_values):
        raise ValueError(
            f"prefix1_pair_keys length {len(prefix1_pair_keys)} must equal "
            f"prefix1_values length {len(prefix1_values)}"
        )
    if len(prefix1_values):
        begins = offsets[:-1].astype(np.int64)
        bucket_ids = np.repeat(np.arange(len(begins)), np.diff(offsets).astype(np.int64))
        if len(bucket_ids) != len(prefix1_values):
            raise ValueError("prefix1_offsets bucket lengths do not cover prefix1_values")
        expected_keys = bucket_ids.astype(np.int64) * vocab_size + prefix1_values.astype(np.int64)
        if not np.array_equal(expected_keys, prefix1_pair_keys):
            raise ValueError(
                "prefix1_pair_keys must equal a_id * vocab_size + b_id for their owning CSR bucket"
            )


class ConstraintTableCompiler:
    """Compile a triples document into ACS-compatible constraint tables."""

    def __init__(self, tokenizer: Any, vocab_size: int) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def compile(self, document: dict) -> ConstraintTableResource:
        """Validate a triples document and build the lookup tables.

        Args:
            document: the task-5 triples document with ``format_version``,
                ``triples``, ``sentinels``, and ``relations``.

        Returns:
            A :class:`ConstraintTableResource` with NumPy table arrays.

        Raises:
            ValueError: on a wrong ``format_version``, unknown token, an
                out-of-range token ID, or an empty / inconsistent document.
        """
        self._validate_format_version(document)

        relations = document.get(RELATIONS_KEY)
        if not isinstance(relations, dict):
            raise ValueError(
                f"Constraint triples document must contain a {RELATIONS_KEY!r} object, "
                f"got {type(relations).__name__ if relations is not None else 'None'}"
            )

        # The `relations` field is derived data.  Rebuild the canonical
        # relations from `triples` and require them to match, so that an
        # inconsistent document (missing / orphaned pairs, mismatched first
        # tokens) cannot silently compile into an invalid resource.
        canonical = self._canonical_relations(document)
        self._validate_relations_match(canonical, relations)

        first_tokens: list[str] = canonical["first"]
        a_to_b: dict[str, list[str]] = canonical["a_to_b"]
        ab_to_c: dict[str, list[str]] = canonical["ab_to_c"]

        # Map every token string to its ID once.
        token_ids = self._map_tokens(
            set(first_tokens)
            | {b for bs in a_to_b.values() for b in bs}
            | {c for cs in ab_to_c.values() for c in cs}
        )

        first_ids = sorted(token_ids[t] for t in first_tokens)
        for token_id in first_ids:
            self._validate_token_id(token_id)

        # Step 1: prefix1_offsets / prefix1_values / prefix1_pair_keys.
        # CSR offsets are built as cumulative counts, matching the ACS
        # operator layout: counts[token_id] = number of s_b continuations,
        # then prefix1_offsets = [0] + cumsum(counts).
        counts = np.zeros(self.vocab_size, dtype=np.int32)
        prefix1_values_list: list[int] = []
        prefix1_pair_keys_list: list[int] = []
        prefix2_value_offsets_list = [0]
        prefix2_values_list: list[int] = []
        max_prefix1_degree = 0
        max_prefix2_degree = 0

        # Iterate s_a in ascending token-ID order; within each s_a, sort the
        # s_b list by token ID too.  This guarantees prefix1_pair_keys =
        # a_id * vocab_size + b_id is strictly increasing, which the operator's
        # binary search requires.
        for a_token in sorted(a_to_b, key=lambda t: token_ids[t]):
            a_id = token_ids[a_token]
            self._validate_token_id(a_id)
            bs = sorted(a_to_b[a_token], key=lambda t: token_ids[t])
            # Canonical relations guarantee every s_a has at least one s_b, so
            # bs is never empty here.
            max_prefix1_degree = max(max_prefix1_degree, len(bs))
            counts[a_id] = len(bs)

            for b_token in bs:
                b_id = token_ids[b_token]
                self._validate_token_id(b_id)
                prefix1_values_list.append(b_id)
                pair_key = a_id * self.vocab_size + b_id
                prefix1_pair_keys_list.append(pair_key)

                # Step 2: prefix2 for this (s_a, s_b) pair.  The canonical
                # relations are guaranteed to contain every (a, b) pair, so a
                # direct lookup can never produce an empty default.  Sort c by
                # token ID for determinism.
                pair_key_str = json.dumps([a_token, b_token])
                cs = sorted(ab_to_c[pair_key_str], key=lambda t: token_ids[t])
                max_prefix2_degree = max(max_prefix2_degree, len(cs))
                for c_token in cs:
                    c_id = token_ids[c_token]
                    self._validate_token_id(c_id)
                    prefix2_values_list.append(c_id)
                prefix2_value_offsets_list.append(len(prefix2_values_list))

        # Check resource size *before* narrowing to int32 so an overflow is
        # caught from the Python-side counts rather than silently wrapping.
        int32_max = np.iinfo(np.int32).max
        total_prefix1 = len(prefix1_values_list)
        total_prefix2 = len(prefix2_values_list)
        if total_prefix1 > int32_max:
            raise ValueError(f"prefix1_values count {total_prefix1} exceeds int32 max {int32_max}")
        if total_prefix2 > int32_max:
            raise ValueError(f"prefix2_values count {total_prefix2} exceeds int32 max {int32_max}")

        # Build offsets in int64 and narrow to int32 only after the size check.
        prefix1_offsets = np.zeros(self.vocab_size + 1, dtype=np.int64)
        np.cumsum(counts, out=prefix1_offsets[1:])
        prefix1_offsets = prefix1_offsets.astype(np.int32)
        prefix2_value_offsets = np.asarray(prefix2_value_offsets_list, dtype=np.int32)

        prefix1_values = np.asarray(prefix1_values_list, dtype=np.int32)
        prefix1_pair_keys = np.asarray(prefix1_pair_keys_list, dtype=np.int64)
        prefix2_values = np.asarray(prefix2_values_list, dtype=np.int32)
        first_token_ids = np.asarray(first_ids, dtype=np.int32)

        resource = ConstraintTableResource(
            first_token_ids=first_token_ids,
            prefix1_offsets=prefix1_offsets,
            prefix1_values=prefix1_values,
            prefix1_pair_keys=prefix1_pair_keys,
            prefix2_value_offsets=prefix2_value_offsets,
            prefix2_values=prefix2_values,
            max_prefix1_degree=max_prefix1_degree,
            max_prefix2_degree=max_prefix2_degree,
            vocab_size=self.vocab_size,
        )
        validate_resource_tables(resource)
        return resource

    def _validate_format_version(self, document: dict) -> None:
        version = document.get(FORMAT_VERSION_KEY)
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported constraint triples format_version={version!r}; "
                f"expected {FORMAT_VERSION}. Regenerate the document with the "
                "current task-5 tooling."
            )

    def _canonical_relations(self, document: dict) -> dict:
        """Rebuild the relations from the ``triples`` field.

        The ``triples`` field is the authoritative business data; ``relations``
        is derived from it.  Rebuilding here lets the compiler validate that
        the two agree instead of trusting a possibly-stale ``relations``.
        """
        triples = document.get(TRIPLES_KEY)
        if not isinstance(triples, list) or not triples:
            raise ValueError(
                f"Constraint triples document must contain a non-empty {TRIPLES_KEY!r} list"
            )
        parsed: list[ConstraintTriple] = []
        seen: set[tuple[str, str, str]] = set()
        for index, triple in enumerate(triples):
            if not isinstance(triple, dict):
                raise ValueError(
                    f"Constraint triples entry #{index} must be an object, "
                    f"got {type(triple).__name__}"
                )
            try:
                parsed_triple = ConstraintTriple(triple["a"], triple["b"], triple["c"])
            except KeyError as exc:
                raise ValueError(
                    f"Constraint triples entry #{index} is missing required field {exc.args[0]!r}"
                ) from exc
            except ValueError as exc:
                raise ValueError(f"Constraint triples entry #{index} is malformed: {exc}") from exc

            key = parsed_triple.values
            if key in seen:
                raise ValueError(
                    f"Duplicate triple {list(key)!r} at entry #{index}; "
                    "Constraint Tables require unique (s_a, s_b, s_c) sequences"
                )
            seen.add(key)
            parsed.append(parsed_triple)
        return build_relations(parsed)

    def _validate_relations_match(self, canonical: dict, relations: dict) -> None:
        """Require the document's relations to match the canonical ones.

        Rejects:
        - a pair present in ``a_to_b`` but missing from ``ab_to_c``;
        - an orphaned pair in ``ab_to_c`` not reachable from ``a_to_b``;
        - a ``first`` token not covered by ``a_to_b`` (or vice versa).
        """
        canonical_a_to_b = canonical["a_to_b"]
        canonical_ab_to_c = canonical["ab_to_c"]
        canonical_first = canonical["first"]

        first = relations.get("first", [])
        if not isinstance(first, list):
            raise ValueError(f"relations.first must be a list, got {type(first).__name__}")
        if sorted(first) != canonical_first:
            raise ValueError(
                "relations.first is inconsistent with the triples document: "
                f"got {sorted(first)!r}, expected {canonical_first!r}"
            )

        a_to_b = relations.get("a_to_b", {})
        if not isinstance(a_to_b, dict):
            raise ValueError(f"relations.a_to_b must be a dict, got {type(a_to_b).__name__}")
        if a_to_b != canonical_a_to_b:
            raise ValueError("relations.a_to_b is inconsistent with the triples document")

        ab_to_c = relations.get("ab_to_c", {})
        if not isinstance(ab_to_c, dict):
            raise ValueError(f"relations.ab_to_c must be a dict, got {type(ab_to_c).__name__}")
        if ab_to_c != canonical_ab_to_c:
            raise ValueError(
                "relations.ab_to_c is inconsistent with the triples document; "
                "every (s_a, s_b) pair must have a (possibly empty) s_c list"
            )

    def _map_tokens(self, tokens: set[str]) -> dict[str, int]:
        """Map token strings to IDs via the tokenizer, rejecting unknowns.

        A tokenizer may silently map an unknown token to a valid-looking
        ``unk_token_id`` (which is not ``-1``/``None``).  When the tokenizer
        exposes a real ``get_vocab()`` mapping we cross-check membership in the
        vocabulary so such tokens are rejected too; otherwise we fall back to
        the ``-1`` / ``None`` marker check.

        The ``get_vocab`` result is required to be a :class:`Mapping`.  This
        guards against test doubles / shims whose ``get_vocab`` exists but
        returns a non-mapping (e.g. a mock), which would otherwise falsely
        reject every token as "not in the vocabulary".
        """
        token_list = sorted(tokens)
        vocab: Mapping[str, int] | None = None
        get_vocab = getattr(self.tokenizer, "get_vocab", None)
        if callable(get_vocab):
            vocab_result = get_vocab()
            if isinstance(vocab_result, Mapping):
                vocab = vocab_result

        ids = self.tokenizer.convert_tokens_to_ids(token_list)
        if isinstance(ids, int):
            ids = [ids]
        if ids is None or len(ids) != len(token_list):
            raise ValueError(
                f"Tokenizer returned {len(ids) if ids is not None else 0} IDs for "
                f"{len(token_list)} tokens; expected one ID per token"
            )
        mapping: dict[str, int] = {}
        id_to_token: dict[int, str] = {}
        for token, token_id in zip(token_list, ids):
            if token_id is None or token_id == -1:
                raise ValueError(
                    f"Failed to map token {token!r} to a valid token ID "
                    f"(got {token_id!r}). Please verify that special/semantic "
                    "tokens are registered in the model's tokenizer."
                )
            if vocab is not None and token not in vocab:
                raise ValueError(
                    f"Failed to map token {token!r}: not present in the tokenizer "
                    "vocabulary (the tokenizer likely returned its unk_token_id). "
                    "Please verify that special/semantic tokens are registered."
                )
            token_id = int(token_id)
            existing = id_to_token.get(token_id)
            if existing is not None and existing != token:
                raise ValueError(
                    f"Tokenizer maps distinct constraint tokens to ID {token_id}: "
                    f"{existing!r} and {token!r}. Two different catalog tokens must "
                    "not share a token ID, or the compiled relation would be ambiguous."
                )
            id_to_token[token_id] = token
            mapping[token] = token_id
        return mapping

    def _validate_token_id(self, token_id: int) -> None:
        if not (0 <= token_id < self.vocab_size):
            raise ValueError(
                f"Token ID {token_id} is out of range for vocab_size={self.vocab_size}"
            )
