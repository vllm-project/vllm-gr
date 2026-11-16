# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Exercise the real FIA update cache using CPU tensors and mocked NPU APIs."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Resolve ordinary dependencies before installing the torch_npu stub, so
    # platform detection cannot mistake it for an installed device extension.
    importlib.import_module("vllm_gr.v1.attention.backends.beam_attn_metadata")
    importlib.import_module("vllm_gr.v1.worker.prefill_graph_common")
    torch_npu = ModuleType("torch_npu")
    torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    # Load the entire production module under a private name. No fake NPU
    # globals or cached package attributes survive the fixture's teardown.
    path = Path(__file__).resolve().parents[1] / "vllm_gr/v1/attention/backends/beam_attn_npu.py"
    spec = importlib.util.spec_from_file_location("_fia_update_cache_backend", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _UpdateAPI:
    def __init__(self) -> None:
        self.current = object()
        self.handle: Any = None
        self.events: list[tuple[Any, ...]] = []
        self.calls: list[dict[str, Any]] = []
        self.failure: tuple[Any, ...] | None = None

    def record(self, *event: Any) -> None:
        self.events.append(event)
        if event == self.failure:
            raise RuntimeError("injected task update failure")

    def current_stream(self) -> Any:
        return self.current

    @contextmanager
    def stream(self, update_stream: Any) -> Iterator[None]:
        previous = self.current
        self.current = update_stream
        self.record("enter", update_stream)
        try:
            yield
        finally:
            self.current = previous
            self.record("exit", update_stream)

    def graph_task_update_begin(self, update_stream: Any, handle: Any) -> None:
        assert self.current is update_stream
        self.handle = handle
        self.record("begin", handle)

    def out(self, **kwargs: Any) -> None:
        # Snapshot host lists: callers intentionally mutate them between calls.
        self.calls.append(
            {
                **kwargs,
                "actual_seq_qlen": tuple(kwargs["actual_seq_qlen"]),
                "actual_seq_kvlen": tuple(kwargs["actual_seq_kvlen"]),
            }
        )
        self.record("fia", self.handle)

    def graph_task_update_end(self, update_stream: Any) -> None:
        assert self.current is update_stream
        self.record("end", self.handle)

    def clear(self) -> None:
        self.events.clear()
        self.calls.clear()


class _CacheHarness:
    def __init__(self, backend: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        self.backend = backend
        self.api = _UpdateAPI()
        monkeypatch.setattr(torch, "npu", self.api, raising=False)
        monkeypatch.setattr(
            backend.torch_npu,
            "npu_fused_infer_attention_score_v2",
            SimpleNamespace(out=self.api.out),
            raising=False,
        )
        self.update_stream = SimpleNamespace(
            wait_stream=lambda serving: self.api.record("wait", serving)
        )
        self.qlen = [2, 4]
        self.kvlen = [8, 12]
        self.block_table = torch.zeros((4, 4), dtype=torch.int32, device="cpu")
        self.entry = backend.NPUPrefillGraphEntry(bucket=4, workspace=torch.empty(16, device="cpu"))
        for _ in range(2):
            self.entry.layers.append(
                backend.NPUGraphLayerTask(
                    handle=object(),
                    **{
                        name: torch.zeros((4, 4), device="cpu")
                        for name in ("query", "key", "value", "mask", "output", "lse")
                    },
                    block_size=16,
                    num_heads=4,
                    num_kv_heads=2,
                    scale=0.5,
                    sparse_mode=3,
                )
            )

    def update(self, **overrides: Any) -> bool:
        result = self.backend.update_graph_params(
            self.update_stream,
            self.entry,
            overrides.get("qlen", self.qlen),
            overrides.get("kvlen", self.kvlen),
            overrides.get("block_table", self.block_table),
        )
        assert isinstance(result, bool)
        return result

    def assert_full_update(self) -> None:
        assert self.api.events == []
        serving = self.api.current
        assert self.update() is True
        expected = [("enter", self.update_stream), ("wait", serving)]
        for task in self.entry.layers:
            expected.extend((phase, task.handle) for phase in ("begin", "fia", "end"))
        expected.append(("exit", self.update_stream))
        assert self.api.events == expected
        assert self.api.current is serving
        assert len(self.api.calls) == len(self.entry.layers) == 2
        for task, call in zip(self.entry.layers, self.api.calls):
            for field in ("query", "key", "value"):
                assert call[field] is getattr(task, field)
            assert call["out"][0] is task.output
            assert call["out"][1] is task.lse
            assert call["atten_mask"] is task.mask
            assert call["block_table"] is self.block_table
            assert call["workspace"] is self.entry.workspace
            assert call["actual_seq_qlen"] == tuple(self.qlen)
            assert call["actual_seq_kvlen"] == tuple(self.kvlen)
            assert call["block_size"] == task.block_size
            assert call["num_query_heads"] == task.num_heads
            assert call["num_key_value_heads"] == task.num_kv_heads
            assert call["softmax_scale"] == task.scale
            assert call["sparse_mode"] == task.sparse_mode
        assert self.entry._last_update_signature is not None
        self.api.clear()


@pytest.fixture
def cache(backend: ModuleType, monkeypatch: pytest.MonkeyPatch) -> _CacheHarness:
    return _CacheHarness(backend, monkeypatch)


def test_equal_bindings_and_lengths_skip_all_update_work(cache: _CacheHarness) -> None:
    cache.assert_full_update()
    assert cache.update(qlen=list(cache.qlen), kvlen=list(cache.kvlen)) is True
    # Equivalent tensor views are bindings, not Python object identities.
    assert cache.update(block_table=cache.block_table.view_as(cache.block_table)) is True
    assert cache.api.events == []
    assert cache.api.calls == []


def test_in_place_tensor_content_changes_keep_cache_hit(cache: _CacheHarness) -> None:
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
def test_mutated_host_length_list_updates_every_layer(cache: _CacheHarness, name: str) -> None:
    cache.assert_full_update()
    getattr(cache, name)[0] += 1
    cache.assert_full_update()


@pytest.mark.parametrize(
    "name", ["query", "key", "value", "mask", "output", "lse", "workspace", "block_table"]
)
def test_replaced_tensor_binding_updates_every_layer(cache: _CacheHarness, name: str) -> None:
    cache.assert_full_update()
    owner = cache if name == "block_table" else cache.entry
    if name not in ("workspace", "block_table"):
        owner = cache.entry.layers[1]
    original = getattr(owner, name)
    setattr(owner, name, original.clone())
    assert getattr(owner, name).data_ptr() != original.data_ptr()
    cache.assert_full_update()


@pytest.mark.parametrize("change", ["shape", "stride", "offset", "dtype"])
def test_tensor_layout_change_updates_every_layer(cache: _CacheHarness, change: str) -> None:
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
    cache: _CacheHarness, name: str, value: int | float
) -> None:
    cache.assert_full_update()
    setattr(cache.entry.layers[1], name, value)
    cache.assert_full_update()


def test_replaced_task_handle_updates_every_layer(cache: _CacheHarness) -> None:
    cache.assert_full_update()
    cache.entry.layers[1].handle = object()
    cache.assert_full_update()


def test_changed_serving_stream_updates_every_layer(cache: _CacheHarness) -> None:
    cache.assert_full_update()
    cache.api.current = object()
    cache.assert_full_update()


def test_optional_mask_transitions_update_every_layer(cache: _CacheHarness) -> None:
    cache.assert_full_update()
    mask = cache.entry.layers[1].mask
    cache.entry.layers[1].mask = None
    cache.assert_full_update()
    cache.entry.layers[1].mask = mask
    cache.assert_full_update()


@pytest.mark.parametrize("name", ["workspace", "query", "key", "value", "output", "lse"])
def test_missing_binding_invalidates_previous_update(cache: _CacheHarness, name: str) -> None:
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
def test_partial_failure_cannot_restore_old_cache_hit(cache: _CacheHarness, phase: str) -> None:
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


def test_signature_validation_exception_invalidates_cache(cache: _CacheHarness) -> None:
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
    cache: _CacheHarness, capture_raises: bool
) -> None:
    cache.assert_full_update()
    state = cache.backend._GRAPH_CAPTURE_STATE
    previous_entry = object()
    state.update(capturing=True, entry=previous_entry)

    def recapture() -> None:
        with cache.backend.prefill_graph_capture(cache.entry):
            assert cache.entry._last_update_signature is None
            assert state == {"capturing": True, "entry": cache.entry}
            if capture_raises:
                raise RuntimeError("injected capture failure")

    if capture_raises:
        with pytest.raises(RuntimeError, match="injected capture failure"):
            recapture()
    else:
        recapture()
    assert state == {"capturing": True, "entry": previous_entry}
    cache.assert_full_update()


pytestmark = pytest.mark.cpu_test
