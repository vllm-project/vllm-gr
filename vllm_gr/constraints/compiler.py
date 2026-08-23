# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline-only compiler from constraint triples to the six-table ABI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from vllm_gr.config import CONSTRAINT_TABLE_FORMAT
from vllm_gr.constraints.json_compiler import (
    FORMAT_VERSION,
    FORMAT_VERSION_KEY,
    RELATIONS_KEY,
    TRIPLES_KEY,
)
from vllm_gr.constraints.table import (
    ConstraintTable,
    ConstraintTableInfo,
    validate_constraint_table,
)
from vllm_gr.constraints.triples import ConstraintTriple, build_relations

MAX_TOP_K = 256


class ConstraintTableCompiler:
    """Compile triples offline; this class is never imported by a Worker."""

    def __init__(
        self,
        tokenizer: Any,
        vocab_size: int,
        *,
        tokenizer_digest: str = "uncomputed",
    ) -> None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {vocab_size!r}")
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.tokenizer_digest = tokenizer_digest

    def compile(self, document: dict[str, Any]) -> ConstraintTable:
        self._validate_format_version(document)
        relations = document.get(RELATIONS_KEY)
        if not isinstance(relations, dict):
            raise ValueError(
                f"Constraint triples document must contain a {RELATIONS_KEY!r} object, "
                f"got {type(relations).__name__ if relations is not None else 'None'}"
            )
        canonical = self._canonical_relations(document)
        self._validate_relations_match(canonical, relations)

        first_tokens: list[str] = canonical["first"]
        a_to_b: dict[str, list[str]] = canonical["a_to_b"]
        ab_to_c: dict[str, list[str]] = canonical["ab_to_c"]
        token_ids = self._map_tokens(
            set(first_tokens)
            | {token for values in a_to_b.values() for token in values}
            | {token for values in ab_to_c.values() for token in values}
        )

        first_ids = sorted(token_ids[token] for token in first_tokens)
        for token_id in first_ids:
            self._validate_token_id(token_id)

        counts = np.zeros(self.vocab_size, dtype=np.int32)
        prefix1_values_list: list[int] = []
        prefix1_pair_keys_list: list[int] = []
        prefix2_value_offsets_list = [0]
        prefix2_values_list: list[int] = []
        max_prefix1_degree = 0
        max_prefix2_degree = 0

        for a_token in sorted(a_to_b, key=lambda token: token_ids[token]):
            a_id = token_ids[a_token]
            self._validate_token_id(a_id)
            b_tokens = sorted(a_to_b[a_token], key=lambda token: token_ids[token])
            max_prefix1_degree = max(max_prefix1_degree, len(b_tokens))
            counts[a_id] = len(b_tokens)
            for b_token in b_tokens:
                b_id = token_ids[b_token]
                self._validate_token_id(b_id)
                prefix1_values_list.append(b_id)
                prefix1_pair_keys_list.append(a_id * self.vocab_size + b_id)
                pair_key = json.dumps([a_token, b_token])
                c_tokens = sorted(ab_to_c[pair_key], key=lambda token: token_ids[token])
                max_prefix2_degree = max(max_prefix2_degree, len(c_tokens))
                for c_token in c_tokens:
                    c_id = token_ids[c_token]
                    self._validate_token_id(c_id)
                    prefix2_values_list.append(c_id)
                prefix2_value_offsets_list.append(len(prefix2_values_list))

        int32_max = np.iinfo(np.int32).max
        if len(prefix1_values_list) > int32_max:
            raise ValueError("prefix1_values count exceeds int32 max")
        if len(prefix2_values_list) > int32_max:
            raise ValueError("prefix2_values count exceeds int32 max")

        prefix1_offsets = np.zeros(self.vocab_size + 1, dtype=np.int64)
        np.cumsum(counts, out=prefix1_offsets[1:])
        arrays = {
            "first_token_ids": np.asarray(first_ids, dtype=np.int32),
            "prefix1_offsets": prefix1_offsets.astype(np.int32),
            "prefix1_values": np.asarray(prefix1_values_list, dtype=np.int32),
            "prefix1_pair_keys": np.asarray(prefix1_pair_keys_list, dtype=np.int64),
            "prefix2_value_offsets": np.asarray(
                prefix2_value_offsets_list,
                dtype=np.int32,
            ),
            "prefix2_values": np.asarray(prefix2_values_list, dtype=np.int32),
        }
        total_bytes = sum(int(array.nbytes) for array in arrays.values())
        table = ConstraintTable(
            **arrays,
            info=ConstraintTableInfo(
                format=CONSTRAINT_TABLE_FORMAT,
                artifact_digest="unpersisted",
                tokenizer_digest=self.tokenizer_digest,
                vocab_size=self.vocab_size,
                max_prefix1_degree=max_prefix1_degree,
                max_prefix2_degree=max_prefix2_degree,
                total_bytes=total_bytes,
            ),
        )
        validate_constraint_table(table)
        return table

    def _validate_format_version(self, document: dict[str, Any]) -> None:
        version = document.get(FORMAT_VERSION_KEY)
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported constraint triples format_version={version!r}; "
                f"expected {FORMAT_VERSION}. Regenerate the document with the "
                "current offline tooling."
            )

    def _canonical_relations(self, document: dict[str, Any]) -> dict[str, Any]:
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
            if parsed_triple.values in seen:
                raise ValueError(
                    f"Duplicate triple {list(parsed_triple.values)!r} at entry #{index}"
                )
            seen.add(parsed_triple.values)
            parsed.append(parsed_triple)
        return build_relations(parsed)

    def _validate_relations_match(
        self,
        canonical: dict[str, Any],
        relations: dict[str, Any],
    ) -> None:
        first = relations.get("first", [])
        if not isinstance(first, list):
            raise ValueError(f"relations.first must be a list, got {type(first).__name__}")
        if sorted(first) != canonical["first"]:
            raise ValueError("relations.first is inconsistent with the triples document")
        a_to_b = relations.get("a_to_b", {})
        if not isinstance(a_to_b, dict):
            raise ValueError(f"relations.a_to_b must be a dict, got {type(a_to_b).__name__}")
        if a_to_b != canonical["a_to_b"]:
            raise ValueError("relations.a_to_b is inconsistent with the triples document")
        ab_to_c = relations.get("ab_to_c", {})
        if not isinstance(ab_to_c, dict):
            raise ValueError(f"relations.ab_to_c must be a dict, got {type(ab_to_c).__name__}")
        if ab_to_c != canonical["ab_to_c"]:
            raise ValueError(
                "relations.ab_to_c is inconsistent with the triples document; "
                "every (s_a, s_b) pair must have an s_c list"
            )

    def _map_tokens(self, tokens: set[str]) -> dict[str, int]:
        token_list = sorted(tokens)
        vocab: Mapping[str, int] | None = None
        get_vocab = getattr(self.tokenizer, "get_vocab", None)
        if callable(get_vocab):
            result = get_vocab()
            if isinstance(result, Mapping):
                vocab = result
        ids = self.tokenizer.convert_tokens_to_ids(token_list)
        if isinstance(ids, int):
            ids = [ids]
        if ids is None or len(ids) != len(token_list):
            raise ValueError(
                f"Tokenizer returned {len(ids) if ids is not None else 0} IDs for "
                f"{len(token_list)} tokens; expected one per token"
            )
        mapping: dict[str, int] = {}
        id_to_token: dict[int, str] = {}
        for token, raw_id in zip(token_list, ids):
            if raw_id is None or raw_id == -1:
                raise ValueError(f"Failed to map token {token!r} to a valid token ID")
            if vocab is not None and token not in vocab:
                raise ValueError(f"Failed to map token {token!r}: not present in tokenizer vocab")
            token_id = int(raw_id)
            previous = id_to_token.get(token_id)
            if previous is not None and previous != token:
                raise ValueError(
                    f"Tokenizer maps distinct constraint tokens to ID {token_id}: "
                    f"{previous!r} and {token!r}"
                )
            id_to_token[token_id] = token
            mapping[token] = token_id
        return mapping

    def _validate_token_id(self, token_id: int) -> None:
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(
                f"Token ID {token_id} is out of range for vocab_size={self.vocab_size}"
            )
