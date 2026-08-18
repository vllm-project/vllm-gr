# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Immutable tokenizer-free Trie resources for EngineCore beam control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

TRIE_RESOURCE_FORMAT_VERSION = 1
_ARRAY_FIELDS = (
    "node_offsets",
    "edge_token_ids",
    "edge_child_nodes",
    "terminal_nodes",
)


@dataclass(frozen=True)
class TrieConstraintResource:
    """CSR-encoded token-ID Trie owned by EngineCore."""

    node_offsets: np.ndarray
    edge_token_ids: np.ndarray
    edge_child_nodes: np.ndarray
    terminal_nodes: np.ndarray
    digest: str = ""

    def __post_init__(self) -> None:
        for name in _ARRAY_FIELDS:
            array = np.ascontiguousarray(getattr(self, name))
            frozen = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
            object.__setattr__(self, name, frozen)
        validate_trie_resource(self)
        object.__setattr__(self, "digest", compute_trie_resource_digest(self))

    @property
    def num_nodes(self) -> int:
        return len(self.node_offsets) - 1

    def root_state(self) -> int:
        return 0

    def valid_next_token_ids(self, node_id: int) -> set[int]:
        start, end = self._bounds(node_id)
        return {int(token_id) for token_id in self.edge_token_ids[start:end]}

    def advance(self, node_id: int, token_id: int) -> int | None:
        start, end = self._bounds(node_id)
        edges = self.edge_token_ids[start:end]
        relative = int(np.searchsorted(edges, token_id))
        if relative >= len(edges) or int(edges[relative]) != token_id:
            return None
        return int(self.edge_child_nodes[start + relative])

    def is_terminal(self, node_id: int) -> bool:
        self._bounds(node_id)
        return bool(self.terminal_nodes[node_id])

    def _bounds(self, node_id: int) -> tuple[int, int]:
        if isinstance(node_id, bool) or not 0 <= node_id < self.num_nodes:
            raise ValueError(f"invalid Trie node id {node_id!r}")
        return int(self.node_offsets[node_id]), int(self.node_offsets[node_id + 1])


class TrieConstraintCompiler:
    """Compile catalog token strings into a tokenizer-free resource."""

    @classmethod
    def from_catalog(cls, tokenizer: Any, catalog_path: str) -> TrieConstraintResource:
        try:
            with open(catalog_path) as catalog_file:
                catalog_data = json.load(catalog_file)
        except FileNotFoundError as exc:
            raise ValueError(f"Catalog file not found: {catalog_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode JSON from catalog file: {catalog_path}") from exc

        if not isinstance(catalog_data, list):
            raise ValueError(  # noqa: TRY004
                "Catalog document must contain a list of token sequences"
            )

        nodes: list[dict[int, int]] = [{}]
        terminal: list[bool] = [False]
        for sequence_index, sequence in enumerate(catalog_data):
            if not isinstance(sequence, list) or any(
                not isinstance(token, str) for token in sequence
            ):
                raise ValueError(
                    f"Catalog sequence #{sequence_index} must be a list of token strings"
                )
            token_ids = tokenizer.convert_tokens_to_ids(sequence)
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            if len(token_ids) != len(sequence):
                raise ValueError(
                    f"Tokenizer returned {len(token_ids)} ids for catalog sequence "
                    f"#{sequence_index} with {len(sequence)} tokens"
                )
            node_id = 0
            for token_index, (token, token_id) in enumerate(zip(sequence, token_ids)):
                if token_id is None or token_id == -1:
                    raise ValueError(
                        f"Failed to map token {token!r} at position {token_index} "
                        f"of catalog sequence #{sequence_index}"
                    )
                if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                    raise ValueError(f"Invalid catalog token id {token_id!r} for token {token!r}")
                child = nodes[node_id].get(token_id)
                if child is None:
                    child = len(nodes)
                    nodes[node_id][token_id] = child
                    nodes.append({})
                    terminal.append(False)
                node_id = child
            terminal[node_id] = True

        node_offsets = [0]
        edge_token_ids: list[int] = []
        edge_child_nodes: list[int] = []
        for edges in nodes:
            for token_id, child_id in sorted(edges.items()):
                edge_token_ids.append(token_id)
                edge_child_nodes.append(child_id)
            node_offsets.append(len(edge_token_ids))

        return TrieConstraintResource(
            node_offsets=np.asarray(node_offsets, dtype=np.int64),
            edge_token_ids=np.asarray(edge_token_ids, dtype=np.int64),
            edge_child_nodes=np.asarray(edge_child_nodes, dtype=np.int64),
            terminal_nodes=np.asarray(terminal, dtype=np.bool_),
        )


def validate_trie_resource(resource: TrieConstraintResource) -> None:
    arrays = {name: getattr(resource, name) for name in _ARRAY_FIELDS}
    if any(array.ndim != 1 for array in arrays.values()):
        raise ValueError("Trie resource arrays must all be one-dimensional")
    if resource.node_offsets.dtype != np.int64:
        raise ValueError("Trie node_offsets must use int64")
    if resource.edge_token_ids.dtype != np.int64:
        raise ValueError("Trie edge_token_ids must use int64")
    if resource.edge_child_nodes.dtype != np.int64:
        raise ValueError("Trie edge_child_nodes must use int64")
    if resource.terminal_nodes.dtype != np.bool_:
        raise ValueError("Trie terminal_nodes must use bool")
    if len(resource.node_offsets) != len(resource.terminal_nodes) + 1:
        raise ValueError("Trie node_offsets must contain one boundary per node plus the end")
    if len(resource.node_offsets) < 2 or int(resource.node_offsets[0]) != 0:
        raise ValueError("Trie must contain a root node and offsets must start at zero")
    if np.any(np.diff(resource.node_offsets) < 0):
        raise ValueError("Trie node_offsets must be non-decreasing")
    edge_count = len(resource.edge_token_ids)
    if int(resource.node_offsets[-1]) != edge_count:
        raise ValueError("Trie node_offsets must end at the edge count")
    if len(resource.edge_child_nodes) != edge_count:
        raise ValueError("Trie edge arrays must have equal length")
    if edge_count:
        if np.any(resource.edge_token_ids < 0):
            raise ValueError("Trie token IDs must be non-negative")
        if np.any(resource.edge_child_nodes < 0) or np.any(
            resource.edge_child_nodes >= len(resource.terminal_nodes)
        ):
            raise ValueError("Trie child node IDs are out of range")
    for node_id in range(len(resource.terminal_nodes)):
        start = int(resource.node_offsets[node_id])
        end = int(resource.node_offsets[node_id + 1])
        tokens = resource.edge_token_ids[start:end]
        if len(tokens) > 1 and np.any(np.diff(tokens) <= 0):
            raise ValueError(f"Trie edges for node {node_id} must be sorted and unique")


def compute_trie_resource_digest(resource: TrieConstraintResource) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(TRIE_RESOURCE_FORMAT_VERSION).encode())
    for name in _ARRAY_FIELDS:
        array = getattr(resource, name)
        hasher.update(name.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(str(tuple(array.shape)).encode())
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def trie_resource_to_wire(resource: TrieConstraintResource) -> dict[str, Any]:
    arrays = {}
    for name in _ARRAY_FIELDS:
        array = getattr(resource, name)
        arrays[name] = {
            "data": np.ascontiguousarray(array).tobytes(),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    return {
        "format_version": TRIE_RESOURCE_FORMAT_VERSION,
        "backend": "trie",
        "arrays": arrays,
        "digest": resource.digest,
    }


def trie_resource_from_wire(payload: dict[str, Any]) -> TrieConstraintResource:
    try:
        if int(payload["format_version"]) != TRIE_RESOURCE_FORMAT_VERSION:
            raise ValueError(f"unsupported Trie resource format {payload['format_version']!r}")
        if payload["backend"] != "trie":
            raise ValueError(f"unexpected constraint backend {payload['backend']!r}")
        fields = payload["arrays"]
        wire_digest = str(payload["digest"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed Trie resource wire payload: {exc}") from exc

    arrays: dict[str, np.ndarray] = {}
    for name in _ARRAY_FIELDS:
        try:
            field = fields[name]
            dtype = np.dtype(field["dtype"])
            shape = tuple(field["shape"])
            data = field["data"]
            arrays[name] = np.frombuffer(data, dtype=dtype).reshape(shape)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed Trie array {name!r}: {exc}") from exc

    resource = TrieConstraintResource(**arrays)
    if resource.digest != wire_digest:
        raise ValueError(
            f"Trie resource digest mismatch: expected {wire_digest}, recomputed {resource.digest}"
        )
    return resource


class BeamConstraintResourceRegistry:
    """EngineCore-owned immutable Trie resource registry."""

    def __init__(self) -> None:
        self._resources: dict[str, TrieConstraintResource] = {}

    def ensure(self, resource: TrieConstraintResource) -> bool:
        existing = self._resources.get(resource.digest)
        if existing is not None:
            return False
        self._resources[resource.digest] = resource
        return True

    def require(self, digest: str) -> TrieConstraintResource:
        resource = self._resources.get(digest)
        if resource is None:
            raise KeyError(f"unknown Trie constraint resource {digest!r}")
        return resource

    def remove(self, digest: str) -> bool:
        return self._resources.pop(digest, None) is not None

    def clear(self) -> None:
        self._resources.clear()

    def __contains__(self, digest: str) -> bool:
        return digest in self._resources

    def __len__(self) -> int:
        return len(self._resources)
