# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"
"""Ordinary FULL attention task updates, exercised with CPU tensors only."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


@dataclass(frozen=True)
class _Descriptor:
    num_tokens: int
    uniform: bool = False


@dataclass
class _NativeACLGraphEntry:
    batch_descriptor: Any
    aclgraph: Any = None


class _NativeACLGraphWrapper:
    def __init__(self, runtime_mode: Any, enable_enpu: bool = True) -> None:
        self.runtime_mode = runtime_mode
        self.enable_enpu = enable_enpu
        self.concrete_aclgraph_entries: dict[Any, Any] = {}


@dataclass
class _ModelGraphProxy:
    ordinary_wrapper: Any

    @property
    def cudagraph_wrapper(self) -> Any:
        return self.ordinary_wrapper


class _Event:
    def __init__(self, api: _NPUAPI) -> None:
        self.api = api

    def wait(self, stream: Any) -> None:
        self.api.events.append(("event_wait", self, stream))

    def reset(self, stream: Any) -> None:
        self.api.events.append(("event_reset", self, stream))

    def record(self, stream: Any) -> None:
        self.api.events.append(("event_record", self, stream))


class _NPUAPI:
    def __init__(self) -> None:
        self.serving = object()
        self.current = self.serving
        self.events: list[tuple[Any, ...]] = []
        self.calls: list[dict[str, Any]] = []
        self.workspace_calls: list[dict[str, Any]] = []
        self.active_handle: Any = None

    def current_stream(self) -> Any:
        return self.current

    def ExternalEvent(self) -> _Event:
        return _Event(self)

    @contextmanager
    def stream(self, stream: Any) -> Iterator[None]:
        previous = self.current
        self.current = stream
        self.events.append(("enter", stream))
        try:
            yield
        finally:
            self.events.append(("exit", stream))
            self.current = previous

    def graph_task_group_begin(self, stream: Any) -> None:
        assert stream is self.current
        self.active_handle = object()
        self.events.append(("capture_begin", self.active_handle, stream))

    def graph_task_group_end(self, stream: Any) -> Any:
        assert stream is self.current
        self.events.append(("capture_end", self.active_handle, stream))
        return self.active_handle

    def graph_task_update_begin(self, stream: Any, handle: Any) -> None:
        assert stream is self.current
        self.active_handle = handle
        self.events.append(("update_begin", handle, stream))

    def graph_task_update_end(self, stream: Any) -> None:
        assert stream is self.current
        self.events.append(("update_end", self.active_handle, stream))

    @staticmethod
    def _snapshot(kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            **kwargs,
            "actual_seq_qlen": tuple(kwargs["actual_seq_qlen"]),
            "actual_seq_kvlen": tuple(kwargs["actual_seq_kvlen"]),
        }

    def out(self, **kwargs: Any) -> None:
        self.calls.append(self._snapshot(kwargs))
        self.events.append(("fia", self.active_handle, self.current))

    def workspace(self, **kwargs: Any) -> torch.Tensor:
        self.workspace_calls.append(self._snapshot(kwargs))
        return torch.empty(16, device="cpu")

    def clear(self) -> None:
        self.events.clear()
        self.calls.clear()


@pytest.fixture
def standard(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Resolve platform detection before installing the synthetic extension.
    importlib.import_module("vllm_gr.v1.attention.backends.beam_attn_metadata")
    torch_npu = ModuleType("torch_npu")
    torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    ascend_utils = ModuleType("vllm_ascend.utils")
    monkeypatch.setattr(ascend_utils, "weak_ref_tensors", lambda value: value, raising=False)
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", ascend_utils)
    acl_graph = ModuleType("vllm_ascend.compilation.acl_graph")
    monkeypatch.setattr(acl_graph, "get_graph_params", lambda: None, raising=False)
    monkeypatch.setattr(acl_graph, "ACLGraphWrapper", _NativeACLGraphWrapper, raising=False)
    monkeypatch.setattr(acl_graph, "ACLGraphEntry", _NativeACLGraphEntry, raising=False)
    monkeypatch.setitem(sys.modules, "vllm_ascend.compilation.acl_graph", acl_graph)
    path = (
        Path(__file__).resolve().parents[1] / "vllm_gr/v1/attention/backends/npu_standard_graph.py"
    )
    spec = importlib.util.spec_from_file_location("_ordinary_fia_graph_backend", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _Harness:
    def __init__(self, module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        from vllm.v1.cudagraph_dispatcher import CUDAGraphMode

        self.module = module
        self.api = _NPUAPI()
        self.owner = SimpleNamespace()
        self.update_stream = object()
        self.context = SimpleNamespace(
            batch_descriptor=_Descriptor(1),
            capturing=True,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            attn_metadata={},
            model_instance=_ModelGraphProxy(
                _NativeACLGraphWrapper(CUDAGraphMode.FULL, enable_enpu=False)
            ),
        )
        monkeypatch.setattr(torch, "npu", self.api, raising=False)
        monkeypatch.setattr(module, "get_forward_context", lambda: self.context)
        monkeypatch.setattr(module, "get_graph_params", lambda: self.owner)
        monkeypatch.setattr(
            module.torch_npu,
            "npu_fused_infer_attention_score_v2",
            SimpleNamespace(out=self.api.out),
            raising=False,
        )
        monkeypatch.setattr(
            module.torch_npu,
            "_npu_fused_infer_attention_score_v2_get_max_workspace",
            self.api.workspace,
            raising=False,
        )
        self.outputs: dict[str, torch.Tensor] = {}

    def metadata(self, num_reqs: int = 1, kvlen: int = 1) -> SimpleNamespace:
        return SimpleNamespace(
            num_actual_tokens=num_reqs,
            num_reqs=num_reqs,
            query_start_loc_cpu=list(range(num_reqs + 1)),
            seq_lens_cpu=[kvlen] * num_reqs,
            block_table=torch.zeros((num_reqs, 32), dtype=torch.int32, device="cpu"),
            slot_mapping=torch.zeros(num_reqs, dtype=torch.int64, device="cpu"),
            causal=True,
            npu_causal_atten_mask=torch.zeros((8, 8), dtype=torch.bool, device="cpu"),
            block_size=16,
            num_heads=2,
            num_kv_heads=1,
            is_prefill_graph_exec=True,
        )

    def capture(
        self,
        *,
        num_reqs: int = 1,
        descriptor: _Descriptor | None = None,
        names: tuple[str, ...] = ("layer.0", "layer.1"),
        query_lengths: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.float32,
        head_size: int = 4,
        cache_stride: int = 1,
        scale: float = 0.5,
    ) -> Any:
        query_lengths = query_lengths or (1,) * num_reqs
        assert len(query_lengths) == num_reqs
        num_tokens = sum(query_lengths)
        self.context.batch_descriptor = descriptor or _Descriptor(num_tokens)
        self.context.capturing = True
        self.context.attn_metadata = {name: self.metadata(num_reqs) for name in names}
        for name, metadata in self.context.attn_metadata.items():
            metadata.num_actual_tokens = num_tokens
            metadata.query_start_loc_cpu = [0]
            for length in query_lengths:
                metadata.query_start_loc_cpu.append(metadata.query_start_loc_cpu[-1] + length)
            metadata.seq_lens_cpu = list(query_lengths)
            metadata.slot_mapping = torch.zeros(num_tokens, dtype=torch.int64, device="cpu")
            query = torch.zeros((num_tokens, 2, head_size), dtype=dtype, device="cpu")
            output = torch.empty_like(query)
            self.outputs[name] = output
            key = torch.zeros((32, 16 * cache_stride, 1, head_size), dtype=dtype, device="cpu")
            value = torch.zeros_like(key)
            key = key[:, ::cache_stride]
            value = value[:, ::cache_stride]
            assert (
                self.module.capture_standard_fia(
                    output, query, key, value, metadata, scale, name, 512
                )
                is None
            )
        self.context.capturing = False
        return self.owner._vllm_gr_standard_fia_entries[self.context.batch_descriptor]

    def update(self) -> None:
        assert (
            self.module.update_standard_graph_params(
                self.update_stream, self.context, self.context.batch_descriptor.num_tokens
            )
            is None
        )


@pytest.fixture
def graph(standard: ModuleType, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    return _Harness(standard, monkeypatch)


def test_decode_lengths_are_rebound_for_every_layer(graph: _Harness) -> None:
    entry = graph.capture()
    assert set(entry.tasks) == {"layer.0", "layer.1"}
    for kvlen in (1, 2, 3):
        for metadata in graph.context.attn_metadata.values():
            metadata.seq_lens_cpu[:] = [kvlen]
        graph.api.clear()
        graph.update()
        assert len(graph.api.calls) == 2
        for name, call in zip(entry.tasks, graph.api.calls):
            task = entry.tasks[name]
            assert call["actual_seq_qlen"] == (1,)
            assert call["actual_seq_kvlen"] == (kvlen,)
            assert call["block_table"] is graph.context.attn_metadata[name].block_table
            assert call["out"][0] is task.output
            assert call["out"][1] is task.lse
            assert call["workspace"] is task.workspace
            assert call["softmax_scale"] == 0.5
            assert call["input_layout"] == "TND"
            assert call["num_query_heads"] == 2
            assert call["num_key_value_heads"] == 1
            assert call["block_size"] == 16
            assert call["sparse_mode"] == 3
            assert call["atten_mask"] is task.mask


def test_native_event_protocol_brackets_capture_and_update(graph: _Harness) -> None:
    entry = graph.capture(names=("layer.0",))
    task = entry.tasks["layer.0"]
    phases = [event[0] for event in graph.api.events]
    assert phases.index("event_wait") < phases.index("event_reset") < phases.index("capture_begin")
    assert phases.index("capture_begin") < phases.index("fia") < phases.index("capture_end")
    graph.api.clear()
    graph.update()
    phases = [event[0] for event in graph.api.events]
    assert phases == ["enter", "update_begin", "fia", "update_end", "event_record", "exit"]
    assert ("event_record", task.event, graph.update_stream) in graph.api.events
    assert graph.api.current is graph.api.serving


def test_mixed_decode_and_prefill_rebinds_per_request_query_boundaries(graph: _Harness) -> None:
    graph.capture(num_reqs=2, query_lengths=(1, 3))
    graph.api.clear()
    for metadata in graph.context.attn_metadata.values():
        metadata.query_start_loc_cpu[:] = [0, 2, 4]
        metadata.seq_lens_cpu[:] = [5, 9]
    graph.update()
    assert len(graph.api.calls) == 2
    for call in graph.api.calls:
        assert call["actual_seq_qlen"] == (2, 4)
        assert call["actual_seq_kvlen"] == (5, 9)


def test_same_token_count_keeps_exact_descriptors_separate(graph: _Harness) -> None:
    first_descriptor = _Descriptor(1, uniform=False)
    second_descriptor = _Descriptor(1, uniform=True)
    first = graph.capture(descriptor=first_descriptor)
    second = graph.capture(descriptor=second_descriptor)
    registry = graph.owner._vllm_gr_standard_fia_entries
    assert set(registry) == {first_descriptor, second_descriptor}
    assert registry[first_descriptor] is first
    assert registry[second_descriptor] is second
    assert first is not second
    assert first.tasks["layer.0"].handle is not second.tasks["layer.0"].handle
    graph.api.clear()
    graph.update()
    assert [event[1] for event in graph.api.events if event[0] == "update_begin"] == [
        task.handle for task in second.tasks.values()
    ]


def test_identical_descriptors_on_different_native_owners_do_not_alias(graph: _Harness) -> None:
    first_owner = graph.owner
    first = graph.capture()
    graph.owner = SimpleNamespace()
    second = graph.capture()
    descriptor = graph.context.batch_descriptor
    assert first_owner._vllm_gr_standard_fia_entries[descriptor] is first
    assert graph.owner._vllm_gr_standard_fia_entries[descriptor] is second
    assert first is not second
    assert first.tasks["layer.0"].event is not second.tasks["layer.0"].event
    graph.api.clear()
    graph.update()
    assert [event[1] for event in graph.api.events if event[0] == "update_begin"] == [
        task.handle for task in second.tasks.values()
    ]


@pytest.mark.parametrize(
    "damage",
    [
        "missing_metadata",
        "missing_task",
        "bad_entry",
        "missing_owner_entry",
        "bad_task",
        "missing_workspace",
        "changed_slot_binding",
    ],
)
def test_invalid_graph_is_rejected_before_any_layer_update(graph: _Harness, damage: str) -> None:
    entry = graph.capture()
    if damage == "missing_metadata":
        del graph.context.attn_metadata["layer.1"]
    elif damage == "missing_task":
        del entry.tasks["layer.1"]
    elif damage == "bad_entry":
        graph.owner._vllm_gr_standard_fia_entries[graph.context.batch_descriptor] = object()
    elif damage == "missing_owner_entry":
        del graph.owner._vllm_gr_standard_fia_entries[graph.context.batch_descriptor]
    elif damage == "bad_task":
        entry.tasks["layer.1"] = object()
    elif damage == "missing_workspace":
        entry.tasks["layer.1"].workspace = None
    else:
        metadata = graph.context.attn_metadata["layer.1"]
        metadata.slot_mapping = metadata.slot_mapping.clone()
    graph.api.clear()
    with pytest.raises(RuntimeError):
        graph.update()
    assert not any(
        event[0] in {"update_begin", "fia", "event_record"} for event in graph.api.events
    )
    assert graph.api.calls == []


def test_later_layer_validation_failure_does_not_partially_update(graph: _Harness) -> None:
    graph.capture()
    # A valid first layer must not be updated before the second is checked.
    graph.context.attn_metadata["layer.1"].seq_lens_cpu = []
    graph.api.clear()
    with pytest.raises(RuntimeError):
        graph.update()
    assert not any(
        event[0] in {"update_begin", "fia", "event_record"} for event in graph.api.events
    )
    assert graph.api.calls == []


def test_workspace_covers_all_64_ordinary_requests_and_max_context(graph: _Harness) -> None:
    graph.capture(num_reqs=64)
    assert graph.api.workspace_calls
    for call in graph.api.workspace_calls:
        assert call["actual_seq_qlen"] == tuple(range(1, 65))
        assert call["actual_seq_kvlen"] == (512,) * 64
    graph.api.clear()
    for metadata in graph.context.attn_metadata.values():
        metadata.seq_lens_cpu = [512] * 64
    graph.update()
    assert len(graph.api.calls) == 2
    assert all(call["actual_seq_kvlen"] == (512,) * 64 for call in graph.api.calls)


@pytest.mark.parametrize(("real_tokens", "bucket"), [(37, 40), (41, 48)])
def test_native_padding_preserves_full_query_and_zero_kv_dummy(
    graph: _Harness, real_tokens: int, bucket: int
) -> None:
    entry = graph.capture(num_reqs=bucket)
    expected_qlen = tuple(range(1, real_tokens + 1)) + (bucket,)
    expected_kvlen = (2,) * real_tokens + (0,)
    for metadata in graph.context.attn_metadata.values():
        metadata.num_actual_tokens = real_tokens
        metadata.num_reqs = real_tokens + 1
        metadata.query_start_loc_cpu = list(range(real_tokens + 1)) + [bucket]
        metadata.seq_lens_cpu = list(expected_kvlen)
        metadata.block_table = metadata.block_table[: real_tokens + 1]
        metadata.slot_mapping[real_tokens:].fill_(-1)
    graph.api.clear()
    graph.update()
    assert len(graph.api.calls) == 2
    for name, call in zip(entry.tasks, graph.api.calls):
        metadata = graph.context.attn_metadata[name]
        task = entry.tasks[name]
        assert call["query"].shape == (bucket, 2, 4)
        assert call["out"][0].shape == (bucket, 2, 4)
        assert call["actual_seq_qlen"] == expected_qlen
        assert call["actual_seq_kvlen"] == expected_kvlen
        assert call["block_table"] is metadata.block_table
        assert call["block_table"].shape[0] == real_tokens + 1
        assert call["block_table"][-1].tolist() == [0] * metadata.block_table.shape[1]
        assert metadata.seq_lens_cpu == list(expected_kvlen)
        assert task.slot_mapping is metadata.slot_mapping
        assert tuple(task.slot_mapping.shape) == (bucket,)
        assert task.slot_mapping[:real_tokens].tolist() == [0] * real_tokens
        assert task.slot_mapping[real_tokens:].tolist() == [-1] * (bucket - real_tokens)


@pytest.mark.parametrize(("real_tokens", "bucket"), [(37, 40), (41, 48)])
def test_full_bucket_rejects_unpadded_query_lengths_before_any_update(
    graph: _Harness, real_tokens: int, bucket: int
) -> None:
    graph.capture(num_reqs=bucket)
    for metadata in graph.context.attn_metadata.values():
        metadata.num_actual_tokens = real_tokens
        metadata.num_reqs = real_tokens
        metadata.query_start_loc_cpu = list(range(real_tokens + 1))
        metadata.seq_lens_cpu = [2] * real_tokens
        metadata.block_table = metadata.block_table[:real_tokens]
        metadata.slot_mapping[real_tokens:].fill_(-1)
    graph.api.clear()
    with pytest.raises(RuntimeError):
        graph.update()
    assert not any(
        event[0] in {"update_begin", "fia", "event_record"} for event in graph.api.events
    )
    assert graph.api.calls == []


@pytest.mark.parametrize("native_entry_exists", [False, True])
def test_fresh_enpu_capture_defers_updates_for_exact_uncaptured_native_entry(
    graph: _Harness, native_entry_exists: bool
) -> None:
    wrapper = _NativeACLGraphWrapper(graph.module.CUDAGraphMode.FULL)
    descriptor = graph.context.batch_descriptor
    if native_entry_exists:
        wrapper.concrete_aclgraph_entries[descriptor] = _NativeACLGraphEntry(descriptor)
        # Both a wholly new owner and an existing empty registry are legal.
        graph.owner._vllm_gr_standard_fia_entries = {}
    graph.context.model_instance = _ModelGraphProxy(wrapper)
    graph.context.capturing = False
    graph.context.attn_metadata = {"layer.0": graph.metadata()}
    graph.update()
    assert graph.api.events == []
    assert graph.api.calls == []


def test_captured_native_graph_cannot_hide_missing_standard_tasks(graph: _Harness) -> None:
    wrapper = _NativeACLGraphWrapper(graph.module.CUDAGraphMode.FULL)
    descriptor = graph.context.batch_descriptor
    wrapper.concrete_aclgraph_entries[descriptor] = _NativeACLGraphEntry(
        descriptor, aclgraph=object()
    )
    graph.context.model_instance = _ModelGraphProxy(wrapper)
    graph.context.capturing = False
    graph.context.attn_metadata = {"layer.0": graph.metadata()}
    with pytest.raises(RuntimeError):
        graph.update()
    assert graph.api.events == []
    assert graph.api.calls == []


def test_uncaptured_native_entry_cannot_hide_incomplete_standard_capture(graph: _Harness) -> None:
    entry = graph.capture()
    del entry.tasks["layer.1"]
    wrapper = _NativeACLGraphWrapper(graph.module.CUDAGraphMode.FULL)
    descriptor = graph.context.batch_descriptor
    wrapper.concrete_aclgraph_entries[descriptor] = _NativeACLGraphEntry(descriptor)
    graph.context.model_instance = _ModelGraphProxy(wrapper)
    graph.api.clear()
    with pytest.raises(RuntimeError):
        graph.update()
    assert graph.api.events == []
    assert graph.api.calls == []


@pytest.mark.parametrize(
    "unproven_state",
    [
        "missing_model",
        "unknown_wrapper",
        "enpu_disabled",
        "non_full_wrapper",
        "bad_native_entry",
        "mismatched_native_descriptor",
        "bad_native_registry",
        "bad_standard_registry",
    ],
)
def test_missing_tasks_only_defer_for_proven_fresh_enpu_capture(
    graph: _Harness, unproven_state: str
) -> None:
    wrapper = _NativeACLGraphWrapper(graph.module.CUDAGraphMode.FULL)
    descriptor = graph.context.batch_descriptor
    graph.context.model_instance = _ModelGraphProxy(wrapper)
    graph.context.capturing = False
    graph.context.attn_metadata = {"layer.0": graph.metadata()}
    if unproven_state == "missing_model":
        del graph.context.model_instance
    elif unproven_state == "unknown_wrapper":
        graph.context.model_instance = _ModelGraphProxy(
            SimpleNamespace(
                runtime_mode=graph.module.CUDAGraphMode.FULL,
                enable_enpu=True,
                concrete_aclgraph_entries={},
            )
        )
    elif unproven_state == "enpu_disabled":
        wrapper.enable_enpu = False
    elif unproven_state == "non_full_wrapper":
        wrapper.runtime_mode = graph.module.CUDAGraphMode.NONE
    elif unproven_state == "bad_native_entry":
        wrapper.concrete_aclgraph_entries[descriptor] = object()
    elif unproven_state == "mismatched_native_descriptor":
        wrapper.concrete_aclgraph_entries[descriptor] = _NativeACLGraphEntry(_Descriptor(2))
    elif unproven_state == "bad_native_registry":
        setattr(wrapper, "concrete_aclgraph_entries", None)
    else:
        graph.owner._vllm_gr_standard_fia_entries = object()
    with pytest.raises(RuntimeError):
        graph.update()
    assert graph.api.events == []
    assert graph.api.calls == []


@pytest.fixture
def prefill_wiring(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Exercise the actual facade without loading an NPU execution module."""
    from vllm.config import CUDAGraphMode

    from vllm_gr.v1.attention import backends
    from vllm_gr.v1.attention.backends import beam_attn
    from vllm_gr.v1.beam.context import BEAM_DATA_VAR

    config_module = importlib.import_module("vllm.config")
    context_module = importlib.import_module("vllm.forward_context")
    capture_calls: list[tuple[Any, ...]] = []
    fallback_calls: list[dict[str, Any]] = []
    global_config_calls: list[None] = []

    def unavailable_global_config() -> Any:
        global_config_calls.append(None)
        raise RuntimeError("global VllmConfig is unset during native graph profiling")

    def capture(*args: Any) -> None:
        capture_calls.append(args)

    def fallback(**kwargs: Any) -> None:
        fallback_calls.append(kwargs)

    def unexpected_cache_write(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("KV sharing must skip reshape_and_cache")

    npu_module = ModuleType("vllm_gr.v1.attention.backends.beam_attn_npu")
    monkeypatch.setattr(npu_module, "_GRAPH_CAPTURE_STATE", {"capturing": False}, raising=False)
    standard_module = ModuleType("vllm_gr.v1.attention.backends.npu_standard_graph")
    monkeypatch.setattr(standard_module, "capture_standard_fia", capture, raising=False)
    for name, module in (("beam_attn_npu", npu_module), ("npu_standard_graph", standard_module)):
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(backends, name, module, raising=False)
    monkeypatch.setattr(config_module, "get_current_vllm_config", unavailable_global_config)
    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        capturing=True,
        model_instance=SimpleNamespace(
            vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=512))
        ),
    )
    monkeypatch.setattr(context_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", True)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", False)
    monkeypatch.setattr(
        beam_attn,
        "_get_platform_execution_module",
        lambda: SimpleNamespace(reshape_and_cache=unexpected_cache_write),
    )
    implementation = SimpleNamespace(
        kv_sharing_target_layer_name="shared-cache",
        kv_cache_dtype="auto",
        sliding_window=None,
        scale=0.125,
        logits_soft_cap=0.0,
        vllm_flash_attn_version=None,
        prefill_flash_attn_version=None,
        sinks=None,
        alibi_slopes=None,
        prefill_attention=fallback,
    )
    query = torch.zeros((1, 2, 4), device="cpu")
    output = torch.empty_like(query)
    key_cache = torch.zeros((2, 16, 1, 4), device="cpu")
    value_cache = torch.zeros_like(key_cache)
    metadata = SimpleNamespace(slot_mapping=torch.zeros(1, dtype=torch.int64, device="cpu"))
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn")

    def invoke() -> Any:
        return beam_attn.BeamAttentionImpl.prefill_forward(
            implementation,
            layer,
            query,
            query,
            query,
            (key_cache, value_cache),
            metadata,
            output,
        )

    token = BEAM_DATA_VAR.set(None)
    try:
        yield SimpleNamespace(
            invoke=invoke,
            facade=beam_attn,
            npu=npu_module,
            capture_calls=capture_calls,
            fallback_calls=fallback_calls,
            global_config_calls=global_config_calls,
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            metadata=metadata,
            layer=layer,
        )
    finally:
        BEAM_DATA_VAR.reset(token)


def test_prefill_capture_uses_model_config_without_global_context(
    prefill_wiring: SimpleNamespace,
) -> None:
    wiring = prefill_wiring
    assert wiring.invoke() is wiring.output
    assert wiring.global_config_calls == []
    assert wiring.fallback_calls == []
    assert len(wiring.capture_calls) == 1
    capture = wiring.capture_calls[0]
    assert capture[0] is wiring.output
    assert capture[1] is wiring.query
    assert capture[2] is wiring.key_cache
    assert capture[3] is wiring.value_cache
    assert capture[4] is wiring.metadata
    assert capture[5:] == (0.125, wiring.layer.layer_name, 512)


@pytest.mark.parametrize("guard", ["gpu", "private_prefill"])
def test_prefill_capture_wiring_preserves_platform_and_private_graph_guards(
    prefill_wiring: SimpleNamespace,
    guard: str,
) -> None:
    wiring = prefill_wiring
    if guard == "gpu":
        wiring.facade.NPU_DEVICE = False
        wiring.facade.CUDA_DEVICE = True
    else:
        wiring.npu._GRAPH_CAPTURE_STATE["capturing"] = True
    assert wiring.invoke() is wiring.output
    assert wiring.capture_calls == []
    assert wiring.global_config_calls == []
    assert len(wiring.fallback_calls) == 1
    assert wiring.fallback_calls[0]["output"] is wiring.output


def test_workspace_pool_checks_nonmonotonic_sizes_and_retains_old_bindings(
    graph: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required_bytes = {64: 128, 32: 256, 16: 96}

    def allocate(**kwargs: Any) -> torch.Tensor:
        graph.api.workspace_calls.append(graph.api._snapshot(kwargs))
        return torch.empty(required_bytes[kwargs["query"].shape[0]], dtype=torch.uint8)

    monkeypatch.setattr(
        graph.module.torch_npu, "_npu_fused_infer_attention_score_v2_get_max_workspace", allocate
    )
    first = graph.capture(num_reqs=64, names=("layer.0",))
    first_context = SimpleNamespace(**vars(graph.context))
    first_workspace = first.tasks["layer.0"].workspace
    second = graph.capture(num_reqs=32, names=("layer.0",))
    third = graph.capture(num_reqs=16, names=("layer.0",))
    assert [call["query"].shape[0] for call in graph.api.workspace_calls] == [64, 32, 16]
    assert first_workspace.numel() * first_workspace.element_size() == 128
    assert second.tasks["layer.0"].workspace.numel() == 256
    assert second.tasks["layer.0"].workspace is not first_workspace
    assert third.tasks["layer.0"].workspace is first_workspace
    assert first.tasks["layer.0"].workspace is first_workspace
    pool = graph.owner._vllm_gr_standard_fia_workspace_pool
    assert len(pool) == 1
    slots = next(iter(pool.values()))
    assert len(slots) == 2
    assert slots[0] is first_workspace
    assert slots[1] is second.tasks["layer.0"].workspace
    graph.context = first_context
    graph.api.clear()
    graph.update()
    assert graph.api.calls[0]["workspace"] is first_workspace
    assert len(graph.api.workspace_calls) == 3, "replay must not allocate workspace"
    assert slots[0] is first_workspace, "weakening an entry must retain the owner pool"


@pytest.mark.parametrize("isolation", ["owner", "wrapper"])
def test_workspace_pool_never_crosses_native_owners_or_wrappers(
    graph: _Harness,
    isolation: str,
) -> None:
    first = graph.capture(num_reqs=8, names=("layer.0",))
    old_owner = graph.owner
    old_wrapper = graph.context.model_instance.cudagraph_wrapper
    if isolation == "owner":
        graph.owner = SimpleNamespace()
    else:
        graph.context.model_instance = _ModelGraphProxy(
            _NativeACLGraphWrapper(graph.module.CUDAGraphMode.FULL, enable_enpu=False)
        )
    second = graph.capture(num_reqs=4, names=("layer.0",))
    assert second.tasks["layer.0"].workspace is not first.tasks["layer.0"].workspace
    if isolation == "owner":
        assert graph.context.model_instance.cudagraph_wrapper is old_wrapper
        assert graph.owner._vllm_gr_standard_fia_workspace_pool is not (
            old_owner._vllm_gr_standard_fia_workspace_pool
        )
    else:
        assert graph.owner is old_owner
        assert graph.context.model_instance.cudagraph_wrapper is not old_wrapper
        assert len(graph.owner._vllm_gr_standard_fia_workspace_pool) == 2


@pytest.mark.parametrize("configuration", ["query_dtype", "head_size", "cache_stride", "scale"])
def test_workspace_pool_keeps_operator_configuration_and_cache_layout_separate(
    graph: _Harness,
    configuration: str,
) -> None:
    first = graph.capture(num_reqs=8, names=("layer.0",))
    options: dict[str, Any] = {
        "query_dtype": {"dtype": torch.bfloat16},
        "head_size": {"head_size": 8},
        "cache_stride": {"cache_stride": 2},
        "scale": {"scale": 0.25},
    }[configuration]
    second = graph.capture(num_reqs=4, names=("layer.0",), **options)
    assert first.tasks["layer.0"].workspace is not second.tasks["layer.0"].workspace
    assert len(graph.owner._vllm_gr_standard_fia_workspace_pool) == 2
    assert len(graph.api.workspace_calls) == 2


def test_workspace_pool_separates_workspace_dtypes_at_equal_byte_capacity(
    graph: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = graph.capture(num_reqs=8, names=("layer.0",))

    def allocate(**kwargs: Any) -> torch.Tensor:
        graph.api.workspace_calls.append(graph.api._snapshot(kwargs))
        return torch.empty(64, dtype=torch.uint8)

    monkeypatch.setattr(
        graph.module.torch_npu, "_npu_fused_infer_attention_score_v2_get_max_workspace", allocate
    )
    second = graph.capture(num_reqs=4, names=("layer.0",))
    first_workspace = first.tasks["layer.0"].workspace
    second_workspace = second.tasks["layer.0"].workspace
    assert first_workspace.dtype != second_workspace.dtype
    assert first_workspace.numel() * first_workspace.element_size() == 64
    assert second_workspace.numel() * second_workspace.element_size() == 64
    assert first_workspace is not second_workspace
    assert len(graph.owner._vllm_gr_standard_fia_workspace_pool) == 2


@pytest.mark.parametrize("invalid_workspace", ["noncontiguous", "offset", "wrong_device"])
def test_workspace_pool_rejects_invalid_getter_buffers_before_capture(
    graph: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    invalid_workspace: str,
) -> None:
    def allocate(**kwargs: Any) -> torch.Tensor:
        graph.api.workspace_calls.append(graph.api._snapshot(kwargs))
        if invalid_workspace == "noncontiguous":
            return torch.empty(32)[::2]
        if invalid_workspace == "offset":
            return torch.empty(17)[1:]
        return torch.empty(16, device="meta")

    monkeypatch.setattr(
        graph.module.torch_npu, "_npu_fused_infer_attention_score_v2_get_max_workspace", allocate
    )
    with pytest.raises(RuntimeError):
        graph.capture(names=("layer.0",))
    assert graph.api.events == []
    assert graph.api.calls == []
