# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from importlib import import_module
from unittest import mock

import pytest

from vllm_gr.patch import (
    BeamPatch,
    BeamPatchInitializationError,
    BeamPatchReport,
    BeamPatchResult,
    BeamPatchStatus,
)


def test_patch_plans_preserve_process_responsibilities() -> None:
    from vllm_gr.patch import _engine_core_beam_patches, _required_beam_patches

    frontend_patches = [patch.name for patch in _required_beam_patches()]
    engine_core_patches = [patch.name for patch in _engine_core_beam_patches()]

    assert frontend_patches == [
        "flat_logprobs",
        "openai_beam_search",
        "fast_sid_encode",
        "chat_sampling_conversion",
        "engine_args",
        "serving_models",
        "batch_and_fork",
    ]
    assert engine_core_patches == [
        "engine_core_requests",
        "kv_hash_memo",
        "scheduler",
        "model_runner",
    ]
    assert set(frontend_patches).isdisjoint(engine_core_patches)


def test_runtime_patch_order_preserves_ascend_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch

    call_order: list[str] = []

    def apply_npu_spawn() -> int:
        call_order.append("npu_spawn")
        return 1

    def apply_attention(force: bool = False) -> int:
        call_order.append("attention")
        return 1

    def patch_npu_worker() -> int:
        call_order.append("npu_worker")
        return 1

    def apply_beam(force: bool = False) -> int:
        call_order.append("beam")
        return 1

    monkeypatch.setattr(
        runtime_patch.npu_spawn_patch,
        "apply_npu_spawn_patch",
        apply_npu_spawn,
    )
    monkeypatch.setattr(
        runtime_patch.ascend_compat,
        "apply_ascend_attention_backend_patch",
        apply_attention,
    )
    monkeypatch.setattr(
        runtime_patch.ascend_compat,
        "patch_npu_worker_init_device",
        patch_npu_worker,
    )
    monkeypatch.setattr(
        runtime_patch,
        "_apply_gr_beam_patches",
        apply_beam,
    )

    assert runtime_patch._apply_patches(force=True) == 4
    assert call_order == ["npu_spawn", "attention", "npu_worker", "beam"]


def test_optional_patch_count_tracks_actual_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch

    monkeypatch.setattr(
        runtime_patch,
        "initialize_required_beam_patches",
        lambda: BeamPatchReport(()),
    )
    monkeypatch.setattr(
        runtime_patch.sliced_logits_patch,
        "apply_sliced_logits_patch",
        mock.Mock(side_effect=(True, False)),
    )
    monkeypatch.setattr(
        runtime_patch.lmcache_patch,
        "is_lmcache_adapter_requested",
        lambda: True,
    )
    monkeypatch.setattr(
        runtime_patch.lmcache_patch,
        "apply_lmcache_adapter",
        mock.Mock(side_effect=(True, False)),
    )

    assert runtime_patch._apply_gr_beam_patches() == 2
    assert runtime_patch._apply_gr_beam_patches() == 0


def test_initialization_applies_required_patch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch

    state = {"installed": False}
    apply = mock.Mock(side_effect=lambda: state.update(installed=True))
    patch = BeamPatch(
        "beam",
        apply,
        lambda: None if state["installed"] else "not installed",
    )
    monkeypatch.setattr(runtime_patch, "_required_beam_patches", lambda: (patch,))

    first = runtime_patch.initialize_required_beam_patches()
    second = runtime_patch.initialize_required_beam_patches()

    apply.assert_called_once_with()
    assert state["installed"] is True
    assert first.applied_count == 1
    assert first.results[0].status is BeamPatchStatus.APPLIED
    assert second.results[0].status is BeamPatchStatus.ALREADY_APPLIED


def test_required_failures_are_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch

    later_apply = mock.Mock()

    def fail() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        runtime_patch,
        "_required_beam_patches",
        lambda: (
            BeamPatch("broken", fail, lambda: "not installed"),
            BeamPatch("later", later_apply, lambda: "not installed"),
        ),
    )

    with pytest.raises(BeamPatchInitializationError, match="broken.*boom") as exc_info:
        runtime_patch.initialize_required_beam_patches()

    later_apply.assert_not_called()
    assert exc_info.value.report.results[-1].status is BeamPatchStatus.FAILED
    assert runtime_patch.get_last_beam_patch_report() is exc_info.value.report

    monkeypatch.setattr(
        runtime_patch,
        "_required_beam_patches",
        lambda: (BeamPatch("broken", lambda: None, lambda: "still missing"),),
    )

    with pytest.raises(
        BeamPatchInitializationError,
        match="broken.*still missing",
    ):
        runtime_patch.initialize_required_beam_patches()


def test_engine_core_child_uses_minimal_authoritative_initialization_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch
    from vllm_gr.v1.engine import engine_core_patch

    calls: list[str] = []

    def initialize() -> BeamPatchReport:
        calls.append("initialize")
        return BeamPatchReport(())

    def original_run(*args: object, **kwargs: object) -> None:
        calls.append("run")

    monkeypatch.setattr(runtime_patch, "initialize_engine_core_beam_patches", initialize)
    monkeypatch.setattr(engine_core_patch, "_run_engine_core_active", False)
    monkeypatch.setattr(
        engine_core_patch.run_engine_core,
        "_original_run_engine_core",
        original_run,
        raising=False,
    )

    engine_core_patch.run_engine_core()

    assert calls == ["initialize", "run"]


def test_model_runner_verifier_checks_shared_discovery_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch
    from vllm_gr.v1.worker import model_runner_patch

    class PatchedRunner:
        pass

    class MissingRunner:
        pass

    monkeypatch.setattr(
        model_runner_patch,
        "discover_active_worker_runners",
        lambda: (
            ("PatchedRunner", PatchedRunner),
            ("MissingRunner", MissingRunner),
        ),
    )
    monkeypatch.setattr(
        model_runner_patch,
        "is_worker_runner_patched",
        lambda runner: runner is PatchedRunner,
    )

    assert (
        runtime_patch._validate_model_runner_patch()
        == "MissingRunner beam execution hooks are not installed"
    )

    monkeypatch.setattr(model_runner_patch, "discover_active_worker_runners", lambda: ())
    assert (
        runtime_patch._validate_model_runner_patch()
        == "No supported active-platform ModelRunner is available"
    )


def test_plugin_exposes_simple_per_patch_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_check = import_module("vllm_gr._env_check")
    plugin = import_module("vllm_gr.plugin")
    runtime_patch = import_module("vllm_gr.patch")

    report = BeamPatchReport((BeamPatchResult("flat_logprobs", BeamPatchStatus.APPLIED),))
    monkeypatch.setattr(env_check, "configure_model_runner_default", mock.Mock())
    monkeypatch.setattr(env_check, "check_environment", mock.Mock())
    reapply = mock.Mock(return_value=1)
    monkeypatch.setattr(runtime_patch, "re_apply_patches", reapply)
    monkeypatch.setattr(
        runtime_patch,
        "get_last_beam_patch_report",
        mock.Mock(return_value=report),
    )

    state = plugin.initialize_runtime()

    reapply.assert_called_once_with()
    assert state == {
        "patch_count": 1,
        "patches": {"flat_logprobs": "applied"},
    }


def test_kv_hash_memo_patch_equivalence_and_hits() -> None:
    """The memo wrapper returns byte-identical hashes and serves repeats.

    hash_block_tokens is pure in (hash_function, parent hash, tokens,
    extra_keys), so a bounded memo must return the exact same BlockHash for
    repeated inputs (dict hit) and fall through to the original for
    first-seen blocks.  FIFO eviction keeps the memo bounded.
    """
    import hashlib

    from vllm.v1.core import kv_cache_utils

    from vllm_gr.v1.engine import engine_core_patch

    # NONE_HASH is lazily initialized by init_none_hash; the hashing path
    # needs it for first-block (parent=None) calls.
    if "NONE_HASH" not in dir(kv_cache_utils) or kv_cache_utils.NONE_HASH is None:
        kv_cache_utils.init_none_hash(
            lambda payload: hashlib.sha256(str(payload).encode()).digest()
        )
    original = kv_cache_utils.hash_block_tokens
    from vllm.v1 import engine as _vllm_engine

    orig_core_factory = _vllm_engine.core.get_request_block_hasher
    # Snapshot the module attribute so the test can restore even if the
    # wrapper was already installed by an earlier plan run.
    try:
        engine_core_patch.apply_kv_hash_memo_patch()
        wrapped = kv_cache_utils.hash_block_tokens
        assert getattr(wrapped, "_gr_hash_memo", False)

        def sha_fn(payload: object) -> bytes:
            return hashlib.sha256(repr(payload).encode()).digest()

        tokens = list(range(100, 116))
        first = wrapped(sha_fn, None, tokens, None)
        # Purity: identical inputs -> byte-identical hash (memo hit).
        second = wrapped(sha_fn, None, tokens, None)
        assert first == second
        # Direct call to the original agrees bit-for-bit.
        assert first == original(sha_fn, None, tokens, None)
        # Different tokens -> different hash (no false hits).
        assert wrapped(sha_fn, None, tokens[:-1] + [999], None) != first
        # Parent hash participates in the key.
        assert wrapped(sha_fn, first, tokens, None) != first
        # Idempotent re-apply keeps the same wrapper.
        engine_core_patch.apply_kv_hash_memo_patch()
        assert kv_cache_utils.hash_block_tokens is wrapped
    finally:
        kv_cache_utils.hash_block_tokens = original
        kv_cache_utils.get_request_block_hasher = orig_core_factory
        _vllm_engine.core.get_request_block_hasher = orig_core_factory


def test_block_hash_chain_memo_replays_full_chain() -> None:
    """The chain memo replays a from-scratch request's whole hash list.

    Eligibility: fresh request (empty block_hashes), no mm / lora / salt.
    The second identical request must receive the SAME BlockHash list in
    the same order without re-running the per-block loop; an incremental
    request (non-empty block_hashes) must fall back to the original
    hasher path.
    """
    import hashlib
    from types import SimpleNamespace

    from vllm.v1.core import kv_cache_utils

    from vllm_gr.v1.engine import engine_core_patch

    if "NONE_HASH" not in dir(kv_cache_utils) or kv_cache_utils.NONE_HASH is None:
        kv_cache_utils.init_none_hash(
            lambda payload: hashlib.sha256(str(payload).encode()).digest()
        )

    orig_hash = kv_cache_utils.hash_block_tokens
    orig_factory = kv_cache_utils.get_request_block_hasher
    from vllm.v1 import engine as _vllm_engine

    orig_core_factory = _vllm_engine.core.get_request_block_hasher
    try:
        engine_core_patch.apply_kv_hash_memo_patch()
        assert getattr(kv_cache_utils.get_request_block_hasher, "_gr_chain_memo", False)
        # core.py binds the factory via from-import: it must hold the SAME
        # memoizing wrapper, not the original, or Requests keep using the
        # per-block loop.
        assert _vllm_engine.core.get_request_block_hasher is kv_cache_utils.get_request_block_hasher

        def sha_fn(payload: object) -> bytes:
            return hashlib.sha256(repr(payload).encode()).digest()

        block_size = 16
        hasher = kv_cache_utils.get_request_block_hasher(block_size, sha_fn)

        fresh = SimpleNamespace(
            all_token_ids=list(range(64)),
            num_tokens=64,
            block_hashes=[],
            mm_features=None,
            lora_request=None,
            cache_salt=None,
            prompt_embeds=None,
        )
        first = hasher(fresh)
        assert len(first) == 64 // block_size

        # Second identical request: chain-memo hit returns an equal list.
        fresh2 = SimpleNamespace(
            all_token_ids=list(range(64)),
            num_tokens=64,
            block_hashes=[],
            mm_features=None,
            lora_request=None,
            cache_salt=None,
            prompt_embeds=None,
        )
        second = hasher(fresh2)
        assert second == first
        # The memo returns a copy, not the cached list object itself, so a
        # consumer mutating it cannot corrupt the cache.
        assert second is not first

        # Incremental request (block_hashes non-empty) must bypass the chain
        # memo and compute through the original path: a request whose token
        # stream grew by one full block yields exactly one new hash.
        grown = SimpleNamespace(
            all_token_ids=list(range(80)),
            num_tokens=80,
            block_hashes=list(first),
            mm_features=None,
            lora_request=None,
            cache_salt=None,
            prompt_embeds=None,
        )
        result = hasher(grown)
        assert len(result) == 1

        # Salted requests never hit the chain memo (fall back; different
        # salt must produce a different chain via the original hasher).
        salted = SimpleNamespace(
            all_token_ids=list(range(64)),
            num_tokens=64,
            block_hashes=[],
            mm_features=None,
            lora_request=None,
            cache_salt="abc",
            prompt_embeds=None,
        )
        salted_result = hasher(salted)
        assert len(salted_result) == len(first)
        assert salted_result != first

        # Embedding requests MUST bypass the chain memo: their all_token_ids
        # are all-zero placeholders, so the token key alone is identical for
        # equal-length prompts whose embeddings differ.  Ground truth comes
        # from the original (unpatched) factory, which distinguishes embeds
        # via per-block extra keys.
        import torch

        emb_size = 64
        emb_a = torch.zeros((emb_size, 8), dtype=torch.float32)
        emb_b = torch.ones((emb_size, 8), dtype=torch.float32)

        def embed_req(embeds: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(
                all_token_ids=[0] * emb_size,
                num_tokens=emb_size,
                block_hashes=[],
                mm_features=[],
                lora_request=None,
                cache_salt=None,
                prompt_embeds=embeds,
                _prompt_embeds_per_block_hashes={},
            )

        emb_a_req = embed_req(emb_a)
        emb_b_req = embed_req(emb_b)
        # Sanity: the original hasher discriminates the two embeddings.
        ground_hasher = orig_factory(block_size, sha_fn)
        ground_a = ground_hasher(emb_a_req)
        ground_b = ground_hasher(emb_b_req)
        assert ground_a != ground_b
        # The patched factory must reproduce each ground-truth chain exactly
        # (a memo collision would hand request B request A's hash list).
        emb_hasher = kv_cache_utils.get_request_block_hasher(block_size, sha_fn)
        assert emb_hasher(emb_a_req) == ground_a
        assert emb_hasher(emb_b_req) == ground_b
        assert len(emb_hasher(emb_a_req)) == emb_size // block_size
    finally:
        kv_cache_utils.hash_block_tokens = orig_hash
        kv_cache_utils.get_request_block_hasher = orig_factory
        _vllm_engine.core.get_request_block_hasher = orig_core_factory


def test_hash_memos_survive_varied_traffic_beyond_capacity() -> None:
    """Oversubscribed varied traffic stays correct, bounded, and evicting.

    Regression for the FIFO eviction scan: after saturation,
    ``dict.pop(next(iter(...)))`` rescans the leading run of deleted entries
    on every miss (CPython only compacts on resize), so varied traffic pays
    an ever-longer scan per eviction.  ``OrderedDict.popitem(last=False)``
    is O(1).  With the limits shrunk, distinct blocks/requests beyond
    capacity must (a) still return hashes identical to the unpatched path,
    (b) keep the stores bounded at the limit, and (c) actually evict the
    oldest entries (a re-run of an evicted key recomputes through the
    original; a resident key does not).
    """
    import hashlib
    from collections.abc import Callable, Sequence
    from types import SimpleNamespace
    from typing import Any

    from vllm.v1 import engine as _vllm_engine
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_utils import BlockHash

    from vllm_gr.v1.engine import engine_core_patch

    if "NONE_HASH" not in dir(kv_cache_utils) or kv_cache_utils.NONE_HASH is None:
        kv_cache_utils.init_none_hash(
            lambda payload: hashlib.sha256(str(payload).encode()).digest()
        )

    orig_hash = kv_cache_utils.hash_block_tokens
    orig_factory = kv_cache_utils.get_request_block_hasher
    orig_core_factory = _vllm_engine.core.get_request_block_hasher
    orig_block_limit = engine_core_patch._KV_HASH_MEMO_LIMIT
    orig_chain_limit = engine_core_patch._CHAIN_MEMO_LIMIT
    try:
        # Spy on the original so the test can observe eviction-driven
        # recomputation (a memo hit never reaches it).
        calls = {"n": 0}

        def counting_original(
            hash_function: Callable[[Any], bytes],
            parent_block_hash: BlockHash | None,
            tokens: Sequence[int],
            extra_keys: tuple[Any, ...] | None,
        ) -> BlockHash:
            calls["n"] += 1
            return orig_hash(hash_function, parent_block_hash, tokens, extra_keys)

        kv_cache_utils.hash_block_tokens = counting_original
        engine_core_patch._KV_HASH_MEMO_LIMIT = 8
        engine_core_patch._CHAIN_MEMO_LIMIT = 4
        engine_core_patch.apply_kv_hash_memo_patch()
        wrapped = kv_cache_utils.hash_block_tokens
        assert wrapped is not counting_original
        block_store = wrapped._gr_hash_memo_entries  # type: ignore[attr-defined]

        def sha_fn(payload: object) -> bytes:
            return hashlib.sha256(repr(payload).encode()).digest()

        # 20 distinct blocks through a limit-8 memo: every value matches the
        # unpatched function and the store stays pinned at the limit.
        distinct = [list(range(200 + i, 216 + i)) for i in range(20)]
        for tokens in distinct:
            assert wrapped(sha_fn, None, tokens, None) == orig_hash(sha_fn, None, tokens, None)
        assert len(block_store) == 8
        assert calls["n"] == 20  # all misses

        # The oldest entry was evicted: re-running it recomputes...
        assert wrapped(sha_fn, None, distinct[0], None) == orig_hash(
            sha_fn, None, distinct[0], None
        )
        assert calls["n"] == 21
        # ...while the newest is still resident (served from the memo).
        assert wrapped(sha_fn, None, distinct[-1], None) == orig_hash(
            sha_fn, None, distinct[-1], None
        )
        assert calls["n"] == 21

        # Chain memo (limit 4): varied fresh requests beyond capacity stay
        # correct and bounded.
        memo_factory = kv_cache_utils.get_request_block_hasher
        chain_store = memo_factory._gr_chain_memo_entries
        block_size = 16
        hasher = memo_factory(block_size, sha_fn)
        ground = orig_factory(block_size, sha_fn)
        for i in range(10):
            req = SimpleNamespace(
                all_token_ids=list(range(i * 64, i * 64 + 64)),
                num_tokens=64,
                block_hashes=[],
                mm_features=None,
                lora_request=None,
                cache_salt=None,
                prompt_embeds=None,
            )
            assert hasher(req) == ground(req)
        assert len(chain_store) == 4
    finally:
        kv_cache_utils.hash_block_tokens = orig_hash
        kv_cache_utils.get_request_block_hasher = orig_factory
        _vllm_engine.core.get_request_block_hasher = orig_core_factory
        engine_core_patch._KV_HASH_MEMO_LIMIT = orig_block_limit
        engine_core_patch._CHAIN_MEMO_LIMIT = orig_chain_limit


def test_chain_miss_traffic_bypasses_per_block_memo() -> None:
    """From-scratch chain-MISS requests must not touch the per-block memo.

    Varied (never-repeating) prompts pay zero per-block bookkeeping: their
    blocks would only evict hot entries.  The per-block store must stay
    untouched by chain-miss traffic, chain-miss digests must equal the
    stock factory's, incremental (block_hashes-seeded) requests must still
    populate the per-block store, and the bypass flag must reset afterwards
    (direct per-block memoization keeps working).
    """
    import hashlib
    from types import SimpleNamespace

    import vllm.v1.engine.core  # noqa: F401  (bind the core submodule)
    from vllm.v1 import engine as _vllm_engine
    from vllm.v1.core import kv_cache_utils

    from vllm_gr.v1.engine import engine_core_patch

    if "NONE_HASH" not in dir(kv_cache_utils) or kv_cache_utils.NONE_HASH is None:
        kv_cache_utils.init_none_hash(
            lambda payload: hashlib.sha256(str(payload).encode()).digest()
        )

    orig_hash = kv_cache_utils.hash_block_tokens
    orig_factory = kv_cache_utils.get_request_block_hasher
    orig_core_factory = _vllm_engine.core.get_request_block_hasher
    try:
        engine_core_patch.apply_kv_hash_memo_patch()
        wrapped = kv_cache_utils.hash_block_tokens
        assert getattr(wrapped, "_gr_hash_memo", False)
        block_store = wrapped._gr_hash_memo_entries

        def sha_fn(payload: object) -> bytes:
            return hashlib.sha256(repr(payload).encode()).digest()

        block_size = 16
        hasher = kv_cache_utils.get_request_block_hasher(block_size, sha_fn)
        ground = orig_factory(block_size, sha_fn)

        def fresh_req(i: int) -> SimpleNamespace:
            return SimpleNamespace(
                all_token_ids=[i + 1000] + list(range(63)),
                num_tokens=64,
                block_hashes=[],
                mm_features=None,
                lora_request=None,
                cache_salt=None,
                prompt_embeds=None,
            )

        # Varied from-scratch requests: chain misses.  The per-block store
        # must not grow while they hash...
        varied = [fresh_req(i) for i in range(20)]
        before = len(block_store)
        head_digests = [hasher(req) for req in varied]
        assert len(block_store) == before
        # The chain memo still records them (a repeat would hit), and the
        # repeat is served by the chain memo — again without touching the
        # per-block store.
        chain_store = kv_cache_utils.get_request_block_hasher._gr_chain_memo_entries
        assert len(chain_store) >= 1
        assert hasher(varied[0]) == head_digests[0]
        assert len(block_store) == before

        # Digest parity with the stock factory.  (The ground hasher runs
        # through the memoized per-block path, so it may populate the store
        # from here on — assert the untouched property above, before this.)
        for req, digest in zip(varied, head_digests):
            assert ground(req) == digest

        # Incremental request (block_hashes seeded): routes to the per-block
        # memo and populates it.
        incremental = SimpleNamespace(
            all_token_ids=varied[0].all_token_ids + list(range(200, 232)),
            num_tokens=96,
            block_hashes=list(head_digests[0]),
            mm_features=None,
            lora_request=None,
            cache_salt=None,
            prompt_embeds=None,
        )
        inc_digest = hasher(incremental)
        assert ground(incremental) == inc_digest
        assert len(inc_digest) == 2  # blocks 5 and 6 are new full blocks
        assert len(block_store) > before

        # The bypass flag reset: direct per-block calls still memoize (the
        # first miss inserts, the second is served without further growth).
        probe = list(range(3000, 3016))
        assert wrapped(sha_fn, None, probe, None) == orig_hash(sha_fn, None, probe, None)
        grown = len(block_store)
        assert wrapped(sha_fn, None, probe, None) == orig_hash(sha_fn, None, probe, None)
        assert len(block_store) == grown
        assert engine_core_patch._CHAIN_FALLBACK_ACTIVE is False
    finally:
        kv_cache_utils.hash_block_tokens = orig_hash
        kv_cache_utils.get_request_block_hasher = orig_factory
        _vllm_engine.core.get_request_block_hasher = orig_core_factory


def test_npu_spawn_patch_wraps_upstream_force_spawn_and_both_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.utils.system_utils as system_utils
    import vllm.v1.executor.multiproc_executor as multiproc_executor

    from vllm_gr import npu_spawn_patch

    upstream_calls: list[str] = []

    def upstream() -> None:
        upstream_calls.append("upstream")

    monkeypatch.setattr(system_utils, "_maybe_force_spawn", upstream)
    monkeypatch.setattr(multiproc_executor, "_maybe_force_spawn", upstream)

    assert npu_spawn_patch.apply_npu_spawn_patch() is True
    wrapped = system_utils._maybe_force_spawn
    assert multiproc_executor._maybe_force_spawn is wrapped
    assert hasattr(wrapped, "_vllm_gr_npu_spawn_original")

    monkeypatch.setattr(npu_spawn_patch, "_npu_is_initialized", lambda: False)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    wrapped()
    assert upstream_calls == ["upstream"]
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ

    # Re-applying over the wrapped target must not stack another layer.
    assert npu_spawn_patch.apply_npu_spawn_patch() is False


def test_npu_spawn_patch_forces_spawn_only_when_npu_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr import npu_spawn_patch

    monkeypatch.setattr(npu_spawn_patch, "_npu_is_initialized", lambda: True)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    npu_spawn_patch._force_spawn_if_npu_initialized()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"

    # An explicit spawn choice or an uninitialized NPU leaves the env alone.
    npu_spawn_patch._force_spawn_if_npu_initialized()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"

    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr(npu_spawn_patch, "_npu_is_initialized", lambda: False)
    npu_spawn_patch._force_spawn_if_npu_initialized()
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ
