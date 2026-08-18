# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from vllm_gr.constraints.trie_resource import (
    BeamConstraintResourceRegistry,
    TrieConstraintCompiler,
    trie_resource_from_wire,
    trie_resource_to_wire,
)


class _Tokenizer:
    ids: ClassVar = {"<begin>": 1, "a": 2, "b": 3, "c": 4, "<end>": 5}

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.ids.get(tokens, -1)
        return [self.ids.get(token, -1) for token in tokens]


def _resource(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            [
                ["<begin>", "a", "b", "<end>"],
                ["<begin>", "a", "c", "<end>"],
            ]
        )
    )
    return TrieConstraintCompiler.from_catalog(_Tokenizer(), str(path))


def test_trie_resource_lookup_and_wire_round_trip(tmp_path) -> None:
    resource = _resource(tmp_path)
    root = resource.root_state()
    assert resource.valid_next_token_ids(root) == {1}
    begin = resource.advance(root, 1)
    assert begin is not None
    assert resource.valid_next_token_ids(begin) == {2}
    a_state = resource.advance(begin, 2)
    assert a_state is not None
    assert resource.valid_next_token_ids(a_state) == {3, 4}

    rebuilt = trie_resource_from_wire(trie_resource_to_wire(resource))
    assert rebuilt.digest == resource.digest
    assert rebuilt.valid_next_token_ids(a_state) == {3, 4}
    assert rebuilt.node_offsets.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        rebuilt.edge_token_ids[0] = 99


def test_trie_resource_rejects_digest_mismatch(tmp_path) -> None:
    wire = trie_resource_to_wire(_resource(tmp_path))
    wire["digest"] = "deadbeef"
    with pytest.raises(ValueError, match="digest mismatch"):
        trie_resource_from_wire(wire)


def test_engine_registry_is_digest_keyed_and_idempotent(tmp_path) -> None:
    resource = _resource(tmp_path)
    registry = BeamConstraintResourceRegistry()
    assert registry.ensure(resource) is True
    assert registry.ensure(resource) is False
    assert registry.require(resource.digest) is resource
    with pytest.raises(KeyError, match="unknown Trie"):
        registry.require("unknown")
    assert registry.remove(resource.digest) is True
    assert registry.remove(resource.digest) is False
