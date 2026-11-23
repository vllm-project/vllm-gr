# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Shared harness for the NPU FIA update cache over CPU tensors and mocked NPU APIs.

``load_backend`` loads the production ``beam_attn_npu`` module with a ``torch_npu``
stub and ``CacheHarness`` drives its graph-parameter update cache. The cache
suite and the Prefill update failure suite both build on them, so they live
here instead of in either test module.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from tests.helpers.paths import REPO_ROOT


def load_backend(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load production ``beam_attn_npu`` with a ``torch_npu`` stub, undone at teardown."""
    # Resolve ordinary dependencies before installing the torch_npu stub, so
    # platform detection cannot mistake it for an installed device extension.
    importlib.import_module("vllm_gr.v1.attention.backends.beam_attn_metadata")
    importlib.import_module("vllm_gr.v1.worker.prefill_graph_common")
    torch_npu = ModuleType("torch_npu")
    torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    # Load the entire production module under a private name. No fake NPU
    # globals or cached package attributes survive the fixture's teardown.
    path = REPO_ROOT / "vllm_gr/v1/attention/backends/beam_attn_npu.py"
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


class CacheHarness:
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
