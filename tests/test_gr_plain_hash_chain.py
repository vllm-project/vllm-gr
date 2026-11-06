# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from types import SimpleNamespace
from typing import Any, NoReturn, cast
from unittest.mock import Mock

import numpy as np
import pytest
from vllm.utils.hashing import sha256 as _hash
from vllm.v1.core import kv_cache_utils

from vllm_gr.v1.engine import engine_core_patch


def _request(length: int) -> Any:
    return SimpleNamespace(
        all_token_ids=np.arange(100000, 100000 + length, dtype=np.int64),
        num_tokens=length,
        block_hashes=[],
        mm_features=None,
        lora_request=None,
        cache_salt=None,
        prompt_embeds=None,
    )


def _direct_block_hash(tokens: Any, parent: bytes, extra_keys: Any) -> bytes:
    """Same encoding the memo computes, without going through the wrapper."""
    return cast(
        bytes,
        engine_core_patch._hash_block_bytes(
            engine_core_patch._token_bytes(tokens), parent, extra_keys
        ),
    )


def _factory(monkeypatch: Any, block_size: int, hash_fn: Any = _hash) -> Any:
    if getattr(kv_cache_utils, "NONE_HASH", None) is None:
        kv_cache_utils.init_none_hash(_hash)
    fallback = Mock(return_value=[])
    proxy = SimpleNamespace(
        get_request_block_hasher=lambda *_: fallback,
        BlockHash=kv_cache_utils.BlockHash,
        NONE_HASH=kv_cache_utils.NONE_HASH,
    )
    # Installation also updates core's from-import alias; restore it after the test.
    from vllm.v1.engine import core

    monkeypatch.setattr(core, "get_request_block_hasher", core.get_request_block_hasher)
    engine_core_patch._apply_block_hash_chain_memo_patch(proxy)
    return proxy.get_request_block_hasher(block_size, hash_fn), fallback, proxy


@pytest.mark.parametrize("block_size", [4, 16, 32])  # type: ignore[misc]
@pytest.mark.parametrize("length", [0, 3, 4, 15, 16, 17, 1023, 1024, 1025])  # type: ignore[misc]
def test_gr_chain_matches_direct_buffer_encoding(
    monkeypatch: Any, block_size: int, length: int
) -> None:
    hasher, fallback, _ = _factory(monkeypatch, block_size)
    request = _request(length)
    expected: list[bytes] = []
    parent = None
    for start in range(0, length - block_size + 1, block_size):
        digest = hashlib.sha256(request.all_token_ids[start : start + block_size])
        digest.update(b"\x00")
        digest.update(parent or kv_cache_utils.NONE_HASH)
        parent = digest.digest()
        expected.append(parent)
    actual = hasher(request)
    assert actual == expected
    if length >= block_size:
        fallback.assert_not_called()
        actual.clear()
        assert hasher(request) == expected  # Published lists cannot corrupt the memo.
    else:
        fallback.assert_called_once_with(request)


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_hashes", [b"parent"]),
        ("mm_features", [object()]),
        ("lora_request", object()),
        ("cache_salt", "salt"),
        ("prompt_embeds", object()),
    ],
)  # type: ignore[misc]
def test_extra_keys_and_incremental_requests_fall_back(
    monkeypatch: Any, field: str, value: Any
) -> None:
    hasher, fallback, _ = _factory(monkeypatch, 16)
    request = _request(64)
    setattr(request, field, value)
    hasher(request)
    fallback.assert_called_once_with(request)


def test_failed_hash_does_not_publish_partial_chain(monkeypatch: Any) -> None:
    calls = 0

    original = engine_core_patch._hash_block_bytes

    def fail_second(*args: Any) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("hash failed")
        return cast(bytes, original(*args))

    monkeypatch.setattr(engine_core_patch, "_hash_block_bytes", fail_second)
    hasher, _, proxy = _factory(monkeypatch, 16)
    with pytest.raises(ValueError, match="hash failed"):
        hasher(_request(64))
    assert engine_core_patch._CHAIN_FALLBACK_ACTIVE is False
    assert not proxy.get_request_block_hasher._gr_chain_memo_entries
    assert len(hasher(_request(64))) == 4


class _NoTokenCopy(np.ndarray):  # type: ignore[misc]
    def tobytes(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("token bytes copy")

    def copy(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("token array copy")

    def tolist(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("token list copy")


def test_chain_reads_views_without_token_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    hasher, _, _ = _factory(monkeypatch, 16)
    request = _request(64)
    request.all_token_ids = request.all_token_ids.view(_NoTokenCopy)
    first = hasher(request)
    assert len(first) == 4
    assert hasher(request) == first
    request.all_token_ids[0] += 1
    assert hasher(request) != first


@pytest.mark.parametrize("extra_keys", [("salt",), (range(3),)])  # type: ignore[misc]
def test_direct_block_hash_includes_parent_and_extra_keys(extra_keys: Any) -> None:
    tokens = _request(16).all_token_ids.view(_NoTokenCopy)
    fn = _direct_block_hash
    assert fn(tokens, b"parent", None) == fn(tokens[:], b"parent", None)
    assert fn(tokens, b"parent", None) != fn(tokens, b"other", None)
    assert fn(tokens, b"parent", None) != fn(tokens, b"parent", extra_keys)
    assert fn(tokens, b"parent", extra_keys) == fn(tokens[:], b"parent", extra_keys)
    assert fn(tokens, b"parent", extra_keys) != fn(tokens, b"other", extra_keys)


def test_block_memo_reads_views_without_token_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.engine import core

    monkeypatch.setattr(core, "get_request_block_hasher", core.get_request_block_hasher)
    monkeypatch.setattr(kv_cache_utils, "get_request_block_hasher", lambda *_: Mock())
    monkeypatch.setattr(kv_cache_utils, "hash_block_tokens", lambda *_: None)
    monkeypatch.setattr(engine_core_patch, "_CHAIN_FALLBACK_ACTIVE", False)
    engine_core_patch.apply_kv_hash_memo_patch()
    tokens = _request(16).all_token_ids.view(_NoTokenCopy)
    fn = kv_cache_utils.hash_block_tokens
    first = fn(_hash, b"parent", tokens, ("salt",))
    assert first == _direct_block_hash(tokens, b"parent", ("salt",))
    assert fn(_hash, b"parent", tokens, ("salt",)) == first
    assert fn(_hash, b"other", tokens, ("salt",)) != first
    tokens[0] += 1
    assert fn(_hash, b"parent", tokens, ("salt",)) != first


def test_python_list_tokens_use_the_same_chain_encoding(monkeypatch: Any) -> None:
    """Native list token storage normalizes instead of raising in hashlib."""
    hasher, _, _ = _factory(monkeypatch, 16)
    array_request = _request(64)
    list_request = SimpleNamespace(
        all_token_ids=list(range(100000, 100064)),
        num_tokens=64,
        block_hashes=[],
        mm_features=None,
        lora_request=None,
        cache_salt=None,
        prompt_embeds=None,
    )
    expected = hasher(array_request)
    assert hasher(list_request) == expected


def test_non_default_hash_function_falls_back(monkeypatch: Any) -> None:
    """A configured non-sha256 algorithm keeps the native implementation."""
    from vllm.utils.hashing import sha256_cbor

    hasher, fallback, _ = _factory(monkeypatch, 16, sha256_cbor)
    request = _request(64)
    hasher(request)
    fallback.assert_called_once_with(request)


@pytest.mark.parametrize("algorithm", ["sha256_cbor", "xxhash", "xxhash_cbor"])  # type: ignore[misc]
@pytest.mark.parametrize("salt", [None, "salt"])  # type: ignore[misc]
def test_non_default_hash_serializers_match_python_tokens(
    monkeypatch: pytest.MonkeyPatch, algorithm: str, salt: str | None
) -> None:
    from vllm.utils import hashing
    from vllm.v1.engine import core

    if algorithm.startswith("xxhash"):
        pytest.importorskip("xxhash")
    monkeypatch.setattr(core, "get_request_block_hasher", core.get_request_block_hasher)
    monkeypatch.setattr(
        kv_cache_utils, "get_request_block_hasher", kv_cache_utils.get_request_block_hasher
    )
    monkeypatch.setattr(kv_cache_utils, "hash_block_tokens", kv_cache_utils.hash_block_tokens)
    monkeypatch.setattr(engine_core_patch, "_CHAIN_FALLBACK_ACTIVE", False)
    if getattr(kv_cache_utils, "NONE_HASH", None) is None:
        kv_cache_utils.init_none_hash(_hash)
    engine_core_patch.apply_kv_hash_memo_patch()
    hasher = kv_cache_utils.get_request_block_hasher(4, getattr(hashing, algorithm))
    array_request = _request(9)
    list_request = _request(9)
    list_request.all_token_ids = array_request.all_token_ids.tolist()
    array_request.cache_salt = list_request.cache_salt = salt
    expected = hasher(list_request)
    assert len(expected) == 2
    assert hasher(array_request) == expected
    assert hasher(array_request) == expected

    # Exercise the incremental per-block path as well as the initial chain.
    array_request.all_token_ids = np.arange(100000, 100012, dtype=np.int64)
    list_request.all_token_ids = array_request.all_token_ids.tolist()
    array_request.num_tokens = list_request.num_tokens = 12
    array_request.block_hashes = list(expected)
    list_request.block_hashes = list(expected)
    incremental = hasher(list_request)
    assert len(incremental) == 1
    assert hasher(array_request) == incremental
