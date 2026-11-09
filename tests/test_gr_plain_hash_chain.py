# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast
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
def test_gr_chain_matches_block_byte_encoding(monkeypatch: Any, block_size: int, length: int) -> None:
    hasher, fallback, _ = _factory(monkeypatch, block_size)
    request = _request(length)
    expected: list[bytes] = []
    parent = None
    for start in range(0, length - block_size + 1, block_size):
        parent = _hash(
            (
                parent or kv_cache_utils.NONE_HASH,
                request.all_token_ids[start : start + block_size].tobytes(),
                None,
            )
        )
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

    def fail_second(value: Any) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("hash failed")
        return cast(bytes, _hash(value))

    hasher, _, proxy = _factory(monkeypatch, 16, fail_second)
    with pytest.raises(ValueError, match="hash failed"):
        hasher(_request(64))
    assert engine_core_patch._CHAIN_FALLBACK_ACTIVE is False
    assert not proxy.get_request_block_hasher._gr_chain_memo_entries
    assert len(hasher(_request(64))) == 4
