# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import os
from importlib import import_module
from unittest import mock

import numpy as np
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
        "chat_sampling_conversion",
        "engine_args",
        "batch_and_fork",
    ]
    assert engine_core_patches == [
        "engine_core_requests",
        "kv_hash_memo",
        "scheduler",
        "model_runner",
    ]
    assert set(frontend_patches).isdisjoint(engine_core_patches)


@pytest.mark.l1  # type: ignore[misc]
def test_flat_logprobs_patch_preserves_offsets_and_optional_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from itertools import repeat

    from vllm.logprobs import FlatLogprobs

    from vllm_gr.v1.engine.logprobs_patch import (
        patch_flat_logprobs,
        validate_flat_logprobs_patch,
    )

    monkeypatch.setattr(FlatLogprobs, "append_fast", FlatLogprobs.append_fast)
    patch_flat_logprobs()
    assert validate_flat_logprobs_patch() is None

    flat = FlatLogprobs()
    flat.append_fast([], [], iter(()), iter(()))
    flat.append_fast([2, 3], [-0.25, -0.5], repeat(1), iter(["b", "c"]))
    flat.append_fast([7], [-1.0], None, None)

    assert flat.start_indices == [0, 0, 2]
    assert flat.end_indices == [0, 2, 3]
    assert flat.token_ids == [2, 3, 7]
    assert flat.logprobs == [-0.25, -0.5, -1.0]
    assert flat.ranks == [1, 1, None]
    assert flat.decoded_tokens == ["b", "c", None]


def test_runtime_patch_order_preserves_ascend_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.patch as runtime_patch
    from vllm_gr.platforms import loader
    from vllm_gr.platforms.ascend import NPU_BACKEND, spawn_patch
    from vllm_gr.v1.worker.npu import startup_patch

    monkeypatch.setattr(loader, "activate_current_backend", lambda: NPU_BACKEND)
    call_order: list[str] = []

    def apply_npu_spawn() -> int:
        call_order.append("npu_spawn")
        return 1

    def apply_pangu() -> bool:
        call_order.append("pangu")
        return True

    def apply_attention(force: bool = False) -> int:
        call_order.append("attention")
        return 1

    def patch_npu_worker() -> int:
        call_order.append("npu_worker")
        return 1

    def apply_beam() -> int:
        call_order.append("beam")
        return 1

    monkeypatch.setattr(
        runtime_patch.pangu_compat,
        "apply_pangu_tied_embeddings_patch",
        apply_pangu,
    )
    monkeypatch.setattr(
        spawn_patch,
        "apply_npu_spawn_patch",
        apply_npu_spawn,
    )
    monkeypatch.setattr(
        runtime_patch.selector_patch,
        "apply_custom_attention_backend_patch",
        apply_attention,
    )
    monkeypatch.setattr(
        startup_patch,
        "patch_npu_worker_init_device",
        patch_npu_worker,
    )
    monkeypatch.setattr(
        runtime_patch,
        "_apply_gr_beam_patches",
        apply_beam,
    )

    assert runtime_patch._apply_patches(force=True) == 5
    assert call_order == ["pangu", "attention", "npu_spawn", "npu_worker", "beam"]


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

    assert runtime_patch._apply_gr_beam_patches() == 1
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
        "is_active_worker_runner_patched",
        lambda runner: runner is PatchedRunner,
    )

    assert (
        model_runner_patch.validate_model_runner_patch()
        == "MissingRunner beam execution hooks are not installed"
    )

    monkeypatch.setattr(model_runner_patch, "discover_active_worker_runners", lambda: ())
    assert (
        model_runner_patch.validate_model_runner_patch()
        == "No supported active-platform ModelRunner is available"
    )


def test_plugin_exposes_simple_per_patch_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup = import_module("vllm_gr.engine.startup")
    loader = import_module("vllm_gr.platforms.loader")
    plugin = import_module("vllm_gr.plugin")
    runtime_patch = import_module("vllm_gr.patch")

    report = BeamPatchReport((BeamPatchResult("flat_logprobs", BeamPatchStatus.APPLIED),))
    provider = mock.Mock()
    monkeypatch.setattr(startup, "configure_model_runner_default", mock.Mock())
    monkeypatch.setattr(loader, "activate_current_backend", mock.Mock(return_value=provider))
    reapply = mock.Mock(return_value=1)
    monkeypatch.setattr(runtime_patch, "re_apply_patches", reapply)
    monkeypatch.setattr(
        runtime_patch,
        "get_last_beam_patch_report",
        mock.Mock(return_value=report),
    )

    state = plugin.initialize_runtime()

    provider.check_environment.assert_called_once_with()
    reapply.assert_called_once_with(provider=provider)
    assert state == {
        "patch_count": 1,
        "patches": {"flat_logprobs": "applied"},
    }


def test_kv_hash_memo_patch_equivalence_and_hits() -> None:
    """The memo wrapper returns byte-identical hashes and serves repeats.

    hash_block_tokens is pure in (hash_function, parent hash, tokens,
    extra_keys), so a bounded memo must return the exact same BlockHash for
    repeated inputs (dict hit) and compute the byte encoding for
    first-seen blocks.  FIFO eviction keeps the memo bounded.
    """
    import hashlib

    from vllm.v1.core import kv_cache_utils

    from vllm_gr.v1.core import kv_cache_utils as gr_kv_cache_utils

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
        gr_kv_cache_utils.apply_kv_hash_memo_patch()
        wrapped = kv_cache_utils.hash_block_tokens
        assert getattr(wrapped, "_gr_hash_memo", False)

        from vllm.utils.hashing import sha256 as sha_fn

        tokens = np.arange(100, 116, dtype=np.int64)
        first = wrapped(sha_fn, None, tokens, None)
        # Purity: identical inputs -> byte-identical hash (memo hit).
        second = wrapped(sha_fn, None, tokens, None)
        assert first == second
        # Compute the direct-view encoding independently of the memo helper.
        expected = hashlib.sha256(tokens)
        expected.update(b"\x00")
        expected.update(kv_cache_utils.NONE_HASH)
        assert first == expected.digest()
        # Different tokens -> different hash (no false hits).
        assert wrapped(sha_fn, None, np.concatenate((tokens[:-1], [999])), None) != first
        # Parent hash participates in the key.
        assert wrapped(sha_fn, first, tokens, None) != first
        # Idempotent re-apply keeps the same wrapper.
        gr_kv_cache_utils.apply_kv_hash_memo_patch()
        assert kv_cache_utils.hash_block_tokens is wrapped
    finally:
        kv_cache_utils.hash_block_tokens = original
        kv_cache_utils.get_request_block_hasher = orig_core_factory
        _vllm_engine.core.get_request_block_hasher = orig_core_factory


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("missing", "intact"),
    [("update_from_output", "schedule"), ("schedule", "update_from_output")],
)
def test_scheduler_plan_repairs_partial_install(
    monkeypatch: pytest.MonkeyPatch, missing: str, intact: str
) -> None:
    from inspect import unwrap

    from vllm.v1.core.sched import scheduler as native

    from vllm_gr import patch

    class IsolatedScheduler(native.Scheduler):  # type: ignore[misc]
        pass

    original = {name: unwrap(getattr(native.Scheduler, name)) for name in (missing, intact)}
    for name, method in original.items():
        monkeypatch.setattr(IsolatedScheduler, name, method)
    monkeypatch.setattr(native, "Scheduler", IsolatedScheduler)
    monkeypatch.setattr(patch, "_LAST_BEAM_PATCH_REPORT", None)
    scheduler_patch = next(p for p in patch._engine_core_beam_patches() if p.name == "scheduler")

    def install() -> BeamPatchStatus:
        return patch._initialize_beam_patch_plan((scheduler_patch,)).results[0].status

    assert install() is BeamPatchStatus.APPLIED
    surviving_wrapper = getattr(IsolatedScheduler, intact)
    monkeypatch.setattr(IsolatedScheduler, missing, original[missing])
    assert scheduler_patch.verify() is not None
    assert install() is BeamPatchStatus.APPLIED
    assert getattr(IsolatedScheduler, missing).__wrapped__ is original[missing]
    assert getattr(IsolatedScheduler, intact) is surviving_wrapper
    assert install() is BeamPatchStatus.ALREADY_APPLIED


def test_kv_hash_memo_plan_repairs_partial_install(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.engine import core

    from vllm_gr import patch

    # Isolate all three process-wide bindings even if an earlier test installed them.
    def native_factory(*args: object) -> object:
        raise AssertionError("unexpected native factory call")

    monkeypatch.setattr(kv_cache_utils, "hash_block_tokens", lambda *_: None)
    monkeypatch.setattr(kv_cache_utils, "get_request_block_hasher", native_factory)
    monkeypatch.setattr(core, "get_request_block_hasher", native_factory)
    monkeypatch.setattr(patch, "_LAST_BEAM_PATCH_REPORT", None)
    (kv_patch,) = [p for p in patch._engine_core_beam_patches() if p.name == "kv_hash_memo"]

    def install() -> BeamPatchStatus:
        return patch._initialize_beam_patch_plan((kv_patch,)).results[0].status

    assert install() is BeamPatchStatus.APPLIED
    block = kv_cache_utils.hash_block_tokens
    cached_hash = block(sha256, b"parent", [1, 2], None)

    # The block wrapper survived, but the chain factory and its alias are native.
    monkeypatch.setattr(kv_cache_utils, "get_request_block_hasher", native_factory)
    monkeypatch.setattr(core, "get_request_block_hasher", native_factory)
    assert install() is BeamPatchStatus.APPLIED
    assert kv_cache_utils.hash_block_tokens is block
    assert block(sha256, b"parent", [1, 2], None) is cached_hash
    chain = kv_cache_utils.get_request_block_hasher
    assert core.get_request_block_hasher is chain

    # Repair only the stale EngineCore alias, retaining the existing chain memo.
    monkeypatch.setattr(core, "get_request_block_hasher", native_factory)
    assert install() is BeamPatchStatus.APPLIED
    assert core.get_request_block_hasher is kv_cache_utils.get_request_block_hasher is chain
    assert install() is BeamPatchStatus.ALREADY_APPLIED


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

    from vllm_gr.v1.core import kv_cache_utils as gr_kv_cache_utils

    if "NONE_HASH" not in dir(kv_cache_utils) or kv_cache_utils.NONE_HASH is None:
        kv_cache_utils.init_none_hash(
            lambda payload: hashlib.sha256(str(payload).encode()).digest()
        )

    orig_hash = kv_cache_utils.hash_block_tokens
    orig_factory = kv_cache_utils.get_request_block_hasher
    from vllm.v1 import engine as _vllm_engine

    orig_core_factory = _vllm_engine.core.get_request_block_hasher
    try:
        gr_kv_cache_utils.apply_kv_hash_memo_patch()
        assert getattr(kv_cache_utils.get_request_block_hasher, "_gr_chain_memo", False)
        # core.py binds the factory via from-import: it must hold the SAME
        # memoizing wrapper, not the original, or Requests keep using the
        # per-block loop.
        assert _vllm_engine.core.get_request_block_hasher is kv_cache_utils.get_request_block_hasher

        from vllm.utils.hashing import sha256 as sha_fn

        block_size = 16
        hasher = kv_cache_utils.get_request_block_hasher(block_size, sha_fn)

        fresh = SimpleNamespace(
            all_token_ids=np.arange(0, 64, dtype=np.int64),
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
            all_token_ids=np.arange(0, 64, dtype=np.int64),
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
            all_token_ids=np.arange(0, 80, dtype=np.int64),
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
            all_token_ids=np.arange(0, 64, dtype=np.int64),
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
                all_token_ids=np.zeros(emb_size, dtype=np.int64),
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


def test_hash_memos_survive_varied_traffic_beyond_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversubscribed varied traffic stays correct, bounded, and evicting.

    Regression for the FIFO eviction scan: after saturation,
    ``dict.pop(next(iter(...)))`` rescans the leading run of deleted entries
    on every miss (CPython only compacts on resize), so varied traffic pays
    an ever-longer scan per eviction.  ``OrderedDict.popitem(last=False)``
    is O(1).  With the limits shrunk, distinct blocks/requests beyond
    capacity must (a) still return hashes consistent with the GR block encoding,
    (b) keep the stores bounded at the limit, and (c) actually evict the
    oldest entries (a re-run of an evicted key recomputes through the
    hash function; a resident key does not).
    """
    import hashlib
    from types import SimpleNamespace

    from vllm.v1 import engine as _vllm_engine
    from vllm.v1.core import kv_cache_utils

    from vllm_gr.v1.core import kv_cache_utils as gr_kv_cache_utils

    if "NONE_HASH" not in dir(kv_cache_utils) or kv_cache_utils.NONE_HASH is None:
        kv_cache_utils.init_none_hash(
            lambda payload: hashlib.sha256(str(payload).encode()).digest()
        )

    orig_hash = kv_cache_utils.hash_block_tokens
    orig_factory = kv_cache_utils.get_request_block_hasher
    orig_core_factory = _vllm_engine.core.get_request_block_hasher
    orig_block_limit = gr_kv_cache_utils._KV_HASH_MEMO_LIMIT
    orig_chain_limit = gr_kv_cache_utils._CHAIN_MEMO_LIMIT
    try:
        # Count real block digests; the configured function is only a namespace.
        block_digest = mock.Mock(wraps=gr_kv_cache_utils._hash_block_bytes)
        monkeypatch.setattr(gr_kv_cache_utils, "_hash_block_bytes", block_digest)
        gr_kv_cache_utils._KV_HASH_MEMO_LIMIT = 8
        gr_kv_cache_utils._CHAIN_MEMO_LIMIT = 4
        gr_kv_cache_utils.apply_kv_hash_memo_patch()
        wrapped = kv_cache_utils.hash_block_tokens
        assert getattr(wrapped, "_gr_hash_memo", False)
        block_store = wrapped._gr_hash_memo_entries
        block_store.clear()

        from vllm.utils.hashing import sha256 as sha_fn

        # 20 distinct blocks through a limit-8 memo stay distinct; the
        # store stays pinned at the limit.
        distinct = [np.arange(200 + i, 216 + i, dtype=np.int64) for i in range(20)]
        results = [wrapped(sha_fn, None, tokens, None) for tokens in distinct]
        assert len(set(results)) == 20
        assert len(block_store) == 8
        assert block_digest.call_count == 20  # all misses

        # The oldest entry was evicted: re-running it recomputes...
        assert wrapped(sha_fn, None, distinct[0], None) == results[0]
        assert block_digest.call_count == 21
        # ...while the newest is still resident (served from the memo).
        assert wrapped(sha_fn, None, distinct[-1], None) == results[-1]
        assert block_digest.call_count == 21

        # Chain memo (limit 4): varied fresh requests beyond capacity stay
        # correct and bounded.
        memo_factory = kv_cache_utils.get_request_block_hasher
        chain_store = memo_factory._gr_chain_memo_entries
        chain_store.clear()
        block_size = 16
        hasher = memo_factory(block_size, sha_fn)
        ground = orig_factory(block_size, sha_fn)
        for i in range(10):
            req = SimpleNamespace(
                all_token_ids=np.arange(i * 64, i * 64 + 64, dtype=np.int64),
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
        gr_kv_cache_utils._KV_HASH_MEMO_LIMIT = orig_block_limit
        gr_kv_cache_utils._CHAIN_MEMO_LIMIT = orig_chain_limit


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

    from vllm_gr.v1.core import kv_cache_utils as gr_kv_cache_utils

    if "NONE_HASH" not in dir(kv_cache_utils) or kv_cache_utils.NONE_HASH is None:
        kv_cache_utils.init_none_hash(
            lambda payload: hashlib.sha256(str(payload).encode()).digest()
        )

    orig_hash = kv_cache_utils.hash_block_tokens
    orig_factory = kv_cache_utils.get_request_block_hasher
    orig_core_factory = _vllm_engine.core.get_request_block_hasher
    try:
        gr_kv_cache_utils.apply_kv_hash_memo_patch()
        wrapped = kv_cache_utils.hash_block_tokens
        assert getattr(wrapped, "_gr_hash_memo", False)
        block_store = wrapped._gr_hash_memo_entries

        from vllm.utils.hashing import sha256 as sha_fn

        block_size = 16
        hasher = kv_cache_utils.get_request_block_hasher(block_size, sha_fn)
        ground = orig_factory(block_size, sha_fn)

        def fresh_req(i: int) -> SimpleNamespace:
            return SimpleNamespace(
                all_token_ids=np.array([i + 1000, *range(63)], dtype=np.int64),
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
            all_token_ids=np.concatenate((varied[0].all_token_ids, np.arange(200, 232))),
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
        probe = np.arange(3000, 3016, dtype=np.int64)
        first_probe = wrapped(sha_fn, None, probe, None)
        grown = len(block_store)
        assert wrapped(sha_fn, None, probe, None) == first_probe
        assert len(block_store) == grown
        assert gr_kv_cache_utils._CHAIN_FALLBACK_ACTIVE is False
    finally:
        kv_cache_utils.hash_block_tokens = orig_hash
        kv_cache_utils.get_request_block_hasher = orig_factory
        _vllm_engine.core.get_request_block_hasher = orig_core_factory


def test_npu_spawn_patch_wraps_upstream_force_spawn_and_both_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.utils.system_utils as system_utils
    import vllm.v1.executor.multiproc_executor as multiproc_executor

    from vllm_gr.platforms.ascend import spawn_patch

    upstream_calls: list[str] = []

    def upstream() -> None:
        upstream_calls.append("upstream")

    monkeypatch.setattr(system_utils, "_maybe_force_spawn", upstream)
    monkeypatch.setattr(multiproc_executor, "_maybe_force_spawn", upstream)

    assert spawn_patch.apply_npu_spawn_patch() is True
    wrapped = system_utils._maybe_force_spawn
    assert multiproc_executor._maybe_force_spawn is wrapped
    assert hasattr(wrapped, "_vllm_gr_npu_spawn_original")

    monkeypatch.setattr(spawn_patch, "_npu_is_initialized", lambda: False)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    wrapped()
    assert upstream_calls == ["upstream"]
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ

    # Re-applying over the wrapped target must not stack another layer.
    assert spawn_patch.apply_npu_spawn_patch() is False
    assert system_utils._maybe_force_spawn is wrapped

    # A later native import can restore the directly imported executor alias.
    monkeypatch.setattr(multiproc_executor, "_maybe_force_spawn", upstream)
    assert spawn_patch.apply_npu_spawn_patch() is True
    assert system_utils._maybe_force_spawn is wrapped
    assert multiproc_executor._maybe_force_spawn is wrapped
    assert spawn_patch.apply_npu_spawn_patch() is False


def test_npu_spawn_patch_forces_spawn_only_when_npu_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.platforms.ascend import spawn_patch

    monkeypatch.setattr(spawn_patch, "_npu_is_initialized", lambda: True)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    spawn_patch._force_spawn_if_npu_initialized()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"

    # An explicit spawn choice or an uninitialized NPU leaves the env alone.
    spawn_patch._force_spawn_if_npu_initialized()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"

    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr(spawn_patch, "_npu_is_initialized", lambda: False)
    spawn_patch._force_spawn_if_npu_initialized()
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ


@pytest.mark.parametrize(
    ("class_name", "method_name"),
    [
        ("AsyncMPClient", "submit_gr_session_async"),
        ("SyncMPClient", "submit_gr_session"),
        ("InprocClient", "submit_gr_session"),
        ("InprocClient", "get_output"),
        ("InprocClient", "add_request"),
        ("InprocClient", "abort_requests"),
        ("InprocClient", "shutdown"),
    ],
)  # type: ignore[misc]
def test_client_validator_rejects_missing_client_patch(
    monkeypatch: pytest.MonkeyPatch, class_name: str, method_name: str
) -> None:
    from vllm.v1.engine import core_client

    from vllm_gr.v1.engine.core_client_patch import (
        apply_engine_client_patches,
        validate_engine_clients_patch,
    )

    apply_engine_client_patches()
    assert validate_engine_clients_patch() is None
    monkeypatch.delattr(getattr(core_client, class_name), method_name)
    assert validate_engine_clients_patch() == "EngineCore client methods are not fully installed"


@pytest.mark.parametrize(
    ("class_name", "method_name"),
    [
        ("AsyncLLM", "submit_gr_session"),
        ("AsyncLLM", "_add_request"),
        ("LLMEngine", "add_request"),
    ],
)  # type: ignore[misc]
def test_async_llm_validator_rejects_missing_merged_submission(
    monkeypatch: pytest.MonkeyPatch, class_name: str, method_name: str
) -> None:
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine

    from vllm_gr.v1.engine.core_client_patch import (
        apply_async_llm_patches,
        validate_async_llm_patch,
    )

    apply_async_llm_patches()
    assert validate_async_llm_patch() is None
    owner = {"AsyncLLM": AsyncLLM, "LLMEngine": LLMEngine}[class_name]
    monkeypatch.delattr(owner, method_name)
    assert validate_async_llm_patch() == "GR frontend lifecycle methods are not fully installed"


pytestmark = pytest.mark.cpu_test
