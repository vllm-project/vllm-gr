# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Exercise the real FIA update cache using CPU tensors and mocked NPU APIs."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.helpers.npu_fia_cache import CacheHarness, load_backend


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return load_backend(monkeypatch)


@pytest.fixture
def cache(backend: ModuleType, monkeypatch: pytest.MonkeyPatch) -> CacheHarness:
    return CacheHarness(backend, monkeypatch)


def test_equal_bindings_and_lengths_skip_all_update_work(cache: CacheHarness) -> None:
    cache.assert_full_update()
    assert cache.update(qlen=list(cache.qlen), kvlen=list(cache.kvlen)) is True
    # Equivalent tensor views are bindings, not Python object identities.
    assert cache.update(block_table=cache.block_table.view_as(cache.block_table)) is True
    assert cache.api.events == []
    assert cache.api.calls == []


def test_in_place_tensor_content_changes_keep_cache_hit(cache: CacheHarness) -> None:
    cache.assert_full_update()
    cache.block_table.fill_(7)
    cache.entry.workspace.fill_(3)
    for task in cache.entry.layers:
        for name in ("query", "key", "value", "mask", "output", "lse"):
            getattr(task, name).fill_(5)
    assert cache.update() is True
    assert cache.api.events == []
    assert cache.api.calls == []


@pytest.mark.parametrize("name", ["qlen", "kvlen"])
def test_mutated_host_length_list_updates_every_layer(cache: CacheHarness, name: str) -> None:
    cache.assert_full_update()
    getattr(cache, name)[0] += 1
    cache.assert_full_update()


@pytest.mark.parametrize(
    "name", ["query", "key", "value", "mask", "output", "lse", "workspace", "block_table"]
)
def test_replaced_tensor_binding_updates_every_layer(cache: CacheHarness, name: str) -> None:
    cache.assert_full_update()
    owner = cache if name == "block_table" else cache.entry
    if name not in ("workspace", "block_table"):
        owner = cache.entry.layers[1]
    original = getattr(owner, name)
    setattr(owner, name, original.clone())
    assert getattr(owner, name).data_ptr() != original.data_ptr()
    cache.assert_full_update()


@pytest.mark.parametrize("change", ["shape", "stride", "offset", "dtype"])
def test_tensor_layout_change_updates_every_layer(cache: CacheHarness, change: str) -> None:
    backing = torch.zeros(17, dtype=torch.int32, device="cpu")
    cache.block_table = backing[:16].view(4, 4)
    cache.assert_full_update()
    original = cache.block_table
    if change == "shape":
        cache.block_table = original.view(2, 8)
    elif change == "stride":
        cache.block_table = original.t()
    elif change == "offset":
        cache.block_table = backing[1:17].view(4, 4)
        assert cache.block_table.storage_offset() != original.storage_offset()
    else:
        cache.block_table = original.view(torch.float32)
    if change != "offset":
        assert cache.block_table.data_ptr() == original.data_ptr()
    cache.assert_full_update()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("block_size", 32),
        ("num_heads", 8),
        ("num_kv_heads", 1),
        ("scale", 0.25),
        ("sparse_mode", 0),
    ],
)
def test_changed_task_config_updates_every_layer(
    cache: CacheHarness, name: str, value: float
) -> None:
    cache.assert_full_update()
    setattr(cache.entry.layers[1], name, value)
    cache.assert_full_update()


def test_replaced_task_handle_updates_every_layer(cache: CacheHarness) -> None:
    cache.assert_full_update()
    cache.entry.layers[1].handle = object()
    cache.assert_full_update()


def test_changed_serving_stream_updates_every_layer(cache: CacheHarness) -> None:
    cache.assert_full_update()
    cache.api.current = object()
    cache.assert_full_update()


def test_optional_mask_transitions_update_every_layer(cache: CacheHarness) -> None:
    cache.assert_full_update()
    mask = cache.entry.layers[1].mask
    cache.entry.layers[1].mask = None
    cache.assert_full_update()
    cache.entry.layers[1].mask = mask
    cache.assert_full_update()


@pytest.mark.parametrize("name", ["workspace", "query", "key", "value", "output", "lse"])
def test_missing_binding_invalidates_previous_update(cache: CacheHarness, name: str) -> None:
    cache.assert_full_update()
    owner = cache.entry if name == "workspace" else cache.entry.layers[1]
    original = getattr(owner, name)
    setattr(owner, name, None)
    assert cache.update() is False
    assert cache.entry._last_update_signature is None
    assert cache.api.events == []
    setattr(owner, name, original)
    cache.assert_full_update()


@pytest.mark.parametrize("phase", ["begin", "fia", "end"])
def test_partial_failure_cannot_restore_old_cache_hit(cache: CacheHarness, phase: str) -> None:
    # A succeeds; B updates layer 1 but fails in layer 2; returning to A must
    # rewrite BOTH layers because the graph now contains a mixture of A/B.
    cache.assert_full_update()
    original = cache.kvlen[0]
    cache.kvlen[0] += 1
    first, second = cache.entry.layers
    cache.api.failure = (phase, second.handle)
    with pytest.raises(RuntimeError, match="injected task update failure"):
        cache.update()
    assert cache.entry._last_update_signature is None
    expected = [("enter", cache.update_stream), ("wait", cache.api.current)]
    expected.extend((step, first.handle) for step in ("begin", "fia", "end"))
    second_steps = ("begin", "fia", "end")
    expected.extend((step, second.handle) for step in second_steps[: second_steps.index(phase) + 1])
    expected.append(("exit", cache.update_stream))
    assert cache.api.events == expected
    cache.api.failure = None
    cache.api.clear()
    cache.kvlen[0] = original
    cache.assert_full_update()


def test_signature_validation_exception_invalidates_cache(cache: CacheHarness) -> None:
    cache.assert_full_update()
    original = cache.entry.layers[1].query

    def invalid_pointer() -> int:
        raise RuntimeError("invalid tensor binding")

    cache.entry.layers[1].query = SimpleNamespace(data_ptr=invalid_pointer)
    with pytest.raises(RuntimeError, match="invalid tensor binding"):
        cache.update()
    assert cache.entry._last_update_signature is None
    assert cache.api.events == []
    cache.entry.layers[1].query = original
    cache.assert_full_update()


@pytest.mark.parametrize("capture_raises", [False, True])
def test_recapture_invalidates_cache_and_restores_capture_state(
    cache: CacheHarness, capture_raises: bool
) -> None:
    cache.assert_full_update()
    state = cache.backend._capture_entry
    previous_entry = object()
    token = state.set(previous_entry)

    def recapture() -> None:
        with cache.backend.prefill_graph_capture(cache.entry):
            assert cache.entry._last_update_signature is None
            assert state.get() is cache.entry
            if capture_raises:
                raise RuntimeError("injected capture failure")

    if capture_raises:
        with pytest.raises(RuntimeError, match="injected capture failure"):
            recapture()
    else:
        recapture()
    assert state.get() is previous_entry
    state.reset(token)
    cache.assert_full_update()


pytestmark = pytest.mark.cpu_test
