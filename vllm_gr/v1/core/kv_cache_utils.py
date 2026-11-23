# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Bounded prefix-cache hash memos for the EngineCore process."""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import msgspec
import numpy as np
from vllm.logger import init_logger
from vllm.utils.hashing import sha256 as _vllm_sha256

logger = init_logger(__name__)


_KV_HASH_MEMO_LIMIT = 65536

# True while the whole-chain memo's fallback hashes a from-scratch request
# that MISSED the chain memo.  Such traffic is (statistically) never
# repeated: inserting its ~64 blocks per request into the per-block memo
# costs key builds, big-dict probes, and FIFO evictions that evict hot
# entries, with no hits in return — the "varied prompts after memo
# saturation" regression.  The per-block wrapper checks this flag first and
# stays transparent, so chain-miss requests hash at stock vLLM speed while
# the per-block memo keeps serving the paths where repeats are real:
# incremental (block_hashes-seeded) beam requests and the mm / lora /
# salt / prompt-embeds fallbacks.  The two memos' recency windows coincide
# (65,536 blocks ~= 1,024 x 64-block prompts = the chain-memo limit), so a
# from-scratch prompt the chain memo cannot serve is beyond the per-block
# memo's reach anyway.  EngineCore is single-threaded; even if a concurrent
# caller ever observed the flag, bypassing the memo is semantically
# transparent (it only memoizes a pure function).
_CHAIN_FALLBACK_ACTIVE = False


def _token_bytes(tokens) -> memoryview:
    """Expose token storage as contiguous bytes without copying.

    GR requests already carry 1-D contiguous int64 arrays. Native requests
    (the ADD path) still pass Python lists, which normalize through the same
    boundary contract instead of raising inside ``hashlib``.
    """
    if not (
        isinstance(tokens, np.ndarray)
        and tokens.dtype == np.int64
        and tokens.ndim == 1
        and tokens.flags.c_contiguous
    ):
        from vllm_gr.v1.beam.batch_contracts import to_int64_tokens

        tokens = to_int64_tokens(tokens, name="block tokens")
    return memoryview(tokens).cast("B")


def _hash_block_bytes(block, parent_hash, extra_keys) -> bytes:
    """Hash token bytes, then a domain tag plus parent/extra-key metadata.

    The one-byte tag keeps the common no-extra-keys path free of per-block
    MessagePack encoding while staying unambiguous with the extra-keys path.
    """
    digest = hashlib.sha256(block)
    if extra_keys is None:
        digest.update(b"\x00")
        digest.update(parent_hash)
        return digest.digest()
    digest.update(b"\x01")
    try:
        digest.update(msgspec.msgpack.encode(extra_keys))
    except TypeError:
        # Preserve support for opaque extra keys through vLLM's native object
        # hasher instead of maintaining another serializer here.
        digest.update(b"\x02")
        digest.update(_vllm_sha256(extra_keys))
    digest.update(parent_hash)
    return digest.digest()


def apply_kv_hash_memo_patch() -> None:
    """Memoize vLLM's per-block prefix-cache hashing (EngineCore process).

    ``vllm.v1.core.kv_cache_utils.hash_block_tokens`` is a pure function of
    (hash_function, parent_block_hash, block token ids, extra keys): the same
    inputs always produce the same ``BlockHash``.  Repeated blocks
    (re-sent beam suffixes, popular prefixes) re-hash identical blocks on
    every request — ~1.3ms of sha256 work per 1024-token prompt — so a
    bounded FIFO memo turns those hits into dict lookups.  Keys are immutable
    byte snapshots of the token view (``bytes(memoryview)``; never
    ``tobytes()``), so a miss hashes each block exactly once, and non-array
    token sequences normalize through the same boundary contract.  The wrapper
    is idempotent and installs on the module attribute
    (``request_block_hasher`` resolves it as a module global at call time,
    so every existing and future hasher picks it up).

    From-scratch requests that miss the whole-chain memo set
    ``_CHAIN_FALLBACK_ACTIVE`` and bypass this wrapper entirely (see there);
    the per-block memo serves incremental beam requests and the mm / lora /
    salt / prompt-embeds fallbacks, where block-level repeats are real.
    """
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.engine import core

    _apply_block_hash_memo_patch(kv_cache_utils)
    _apply_block_hash_chain_memo_patch(kv_cache_utils)
    # EngineCore keeps its own from-import binding to the factory.
    core.get_request_block_hasher = kv_cache_utils.get_request_block_hasher


def _apply_block_hash_memo_patch(kv_cache_utils) -> None:
    global _CHAIN_FALLBACK_ACTIVE
    if getattr(kv_cache_utils.hash_block_tokens, "_gr_hash_memo", False):
        return

    _CHAIN_FALLBACK_ACTIVE = False
    original = kv_cache_utils.hash_block_tokens
    # OrderedDict (not plain dict): after the memo saturates, every miss
    # evicts the oldest entry.  ``dict.pop(next(iter(...)))`` is NOT
    # constant-time — the fresh iterator rescans the leading run of deleted
    # (dummy) slots and CPython only compacts the table on resize, so varied
    # traffic pays an ever-longer scan per eviction (measured 14-32x slowdown
    # on the saturated hashing path).  ``OrderedDict.popitem(last=False)``
    # unlinks the head of its doubly-linked list in O(1).
    memo: OrderedDict[tuple, object] = OrderedDict()

    def hash_block_tokens_memo(
        hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None
    ):
        if hash_function is not _vllm_sha256:
            # The direct-buffer encoding below is SHA-256 specific; keep the
            # configured algorithm authoritative for other hash functions.
            # Those requests take the native path and therefore do NOT get memo
            # acceleration. Identity comparison is intentional: upstream hands
            # us the module-level ``sha256`` itself, so a wrapper falls back.
            # Native serializers require Python ints, not NumPy scalars.
            if isinstance(curr_block_token_ids, np.ndarray):
                curr_block_token_ids = curr_block_token_ids.tolist()
            return original(hash_function, parent_block_hash, curr_block_token_ids, extra_keys)
        parent_hash = parent_block_hash or kv_cache_utils.NONE_HASH
        # Immutable content snapshot: exact memo key, no second block hash.
        token_bytes = bytes(_token_bytes(curr_block_token_ids))
        if _CHAIN_FALLBACK_ACTIVE:
            return kv_cache_utils.BlockHash(_hash_block_bytes(token_bytes, parent_hash, extra_keys))
        key = (hash_function, parent_hash, token_bytes, extra_keys)
        hit = memo.get(key)
        if hit is not None:
            return hit
        hit = kv_cache_utils.BlockHash(_hash_block_bytes(token_bytes, parent_hash, extra_keys))
        if len(memo) >= _KV_HASH_MEMO_LIMIT:
            # Constant-time FIFO eviction keeps the memo bounded in
            # long-running servers under varied (never-repeating) traffic.
            memo.popitem(last=False)
        memo[key] = hit
        return hit

    hash_block_tokens_memo._gr_hash_memo = True  # type: ignore[attr-defined]
    # Exposed for tests / diagnostics: the live memo store behind this wrapper.
    hash_block_tokens_memo._gr_hash_memo_entries = memo  # type: ignore[attr-defined]
    kv_cache_utils.hash_block_tokens = hash_block_tokens_memo
    logger.info("[vllm-gr] kv block-hash memo patch applied.")


_CHAIN_MEMO_LIMIT = 1024


def _apply_block_hash_chain_memo_patch(kv_cache_utils) -> None:
    """Whole-chain memo on top of the per-block memo (from-scratch requests).

    The per-block memo removes the sha256 work but a repeated prompt still
    walks the ``request_block_hasher`` loop (64 iterations of slicing,
    ``generate_block_hash_extra_keys``, tuple-key building — ~0.8ms for a
    1024-token prompt).  For a request hashing from scratch with no mm /
    lora / salt variation, the full hash chain is a pure function of
    (hash_function, block_size, salt, the full-block token prefix), so a
    hit can replay the whole list in one dict lookup.

    A MISS hashes the full-block byte view directly with SHA-256, using the
    same ``sha256(tokens || tag || parent)`` encoding as incremental hashing.
    Eligibility proves that extra keys are absent. The immutable token-byte
    snapshot doubles as the memo key, so no second whole-prompt hash is paid.
    No per-block memo bookkeeping is needed; the chain entry is still recorded
    for a repeat of this exact prompt.

    Fast-path eligibility is conservative: any incremental call
    (``block_hashes`` non-empty), mm_features, lora_request, a non-``None``
    cache_salt, or a non-``None`` prompt_embeds falls back to the original
    hasher.  prompt_embeds requests MUST bypass the chain memo: their
    ``all_token_ids`` are ``[0] * num_prompt_tokens`` placeholders (vLLM
    distinguishes their contents via per-block embeds hashes), so the token
    key alone would collide across distinct embeddings of equal length.
    """
    global _CHAIN_FALLBACK_ACTIVE
    if getattr(kv_cache_utils.get_request_block_hasher, "_gr_chain_memo", False):
        return

    original_factory = kv_cache_utils.get_request_block_hasher
    # Same constant-time FIFO rationale as the per-block memo above.
    chain_memo: OrderedDict[tuple, list] = OrderedDict()

    def get_request_block_hasher_memo(block_size, caching_hash_fn):
        original_hasher = original_factory(block_size, caching_hash_fn)

        def request_block_hasher_chain_memo(request):
            global _CHAIN_FALLBACK_ACTIVE
            if (
                request.block_hashes
                or request.mm_features
                or request.lora_request is not None
                or request.cache_salt is not None
                or getattr(request, "prompt_embeds", None) is not None
            ):
                return original_hasher(request)
            if caching_hash_fn is not _vllm_sha256:
                # Same contract as the per-block memo: non-default algorithms
                # keep native hashing and lose the chain memo.
                return original_hasher(request)

            num_tokens = request.num_tokens
            num_full_blocks = num_tokens // block_size
            if num_full_blocks == 0:
                return original_hasher(request)

            # Only the full-block prefix participates in hashing; trailing
            # partial-block tokens never reach hash_block_tokens.
            tokens = request.all_token_ids[: num_full_blocks * block_size]
            # Immutable content snapshot: an exact memo key that costs one
            # memcpy instead of a whole-prompt SHA-256.
            token_bytes = bytes(_token_bytes(tokens))
            key = (caching_hash_fn, block_size, token_bytes)
            hit = chain_memo.get(key)
            if hit is not None:
                return list(hit)

            _CHAIN_FALLBACK_ACTIVE = True
            try:
                # Eligibility above proves every block has no extra keys.
                # Use the same direct-buffer encoding as hash_block_tokens_memo.
                result = []
                parent_hash = None
                block_view = memoryview(token_bytes)
                step = block_size * 8
                for start in range(0, len(block_view), step):
                    parent_hash = kv_cache_utils.BlockHash(
                        _hash_block_bytes(
                            block_view[start : start + step],
                            parent_hash or kv_cache_utils.NONE_HASH,
                            None,
                        )
                    )
                    result.append(parent_hash)
            finally:
                _CHAIN_FALLBACK_ACTIVE = False
            if len(chain_memo) >= _CHAIN_MEMO_LIMIT:
                chain_memo.popitem(last=False)
            chain_memo[key] = list(result)
            return result

        return request_block_hasher_chain_memo

    get_request_block_hasher_memo._gr_chain_memo = True  # type: ignore[attr-defined]
    # Exposed for tests / diagnostics: the live chain-memo store.
    get_request_block_hasher_memo._gr_chain_memo_entries = chain_memo  # type: ignore[attr-defined]
    kv_cache_utils.get_request_block_hasher = get_request_block_hasher_memo


def validate_kv_hash_memo_patch() -> str | None:
    """Verify both prefix-hash wrappers and the factory used by EngineCore."""
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.engine import core

    if not getattr(kv_cache_utils.hash_block_tokens, "_gr_hash_memo", False):
        return "hash_block_tokens memo wrapper not installed"
    factory = kv_cache_utils.get_request_block_hasher
    if not getattr(factory, "_gr_chain_memo", False):
        return "get_request_block_hasher chain memo wrapper not installed"
    if core.get_request_block_hasher is not factory:
        return "EngineCore get_request_block_hasher is not bound to the chain memo wrapper"
    return None
