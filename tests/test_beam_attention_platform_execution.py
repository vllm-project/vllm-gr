# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform isolation and delegation contracts for Beam Attention."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.attention import backends
from vllm_gr.v1.attention.backends import beam_attn


def test_facade_import_does_not_load_platform_execution_modules() -> None:
    project_root = Path(__file__).resolve().parents[1]
    code = """
import sys

import vllm_gr.v1.attention.backends.beam_attn

assert "vllm_gr.v1.attention.backends.beam_attn_gpu" not in sys.modules
assert "vllm_gr.v1.attention.backends.beam_attn_npu" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    ("cuda_available", "npu_available", "expected"),
    [
        (True, False, "gpu"),
        (False, True, "npu"),
        (False, False, None),
        (True, True, None),
    ],
)
def test_platform_dispatch_selects_exactly_one_execution_module(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    npu_available: bool,
    expected: str | None,
) -> None:
    gpu_module = ModuleType("vllm_gr.v1.attention.backends.beam_attn_gpu")
    npu_module = ModuleType("vllm_gr.v1.attention.backends.beam_attn_npu")
    monkeypatch.setitem(sys.modules, gpu_module.__name__, gpu_module)
    monkeypatch.setitem(sys.modules, npu_module.__name__, npu_module)
    monkeypatch.setattr(backends, "beam_attn_gpu", gpu_module, raising=False)
    monkeypatch.setattr(backends, "beam_attn_npu", npu_module, raising=False)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", cuda_available)
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", npu_available)

    if expected is None:
        with pytest.raises(RuntimeError, match="exactly one active CUDA or NPU"):
            beam_attn._get_platform_execution_module()
        return

    selected = beam_attn._get_platform_execution_module()
    assert selected is (gpu_module if expected == "gpu" else npu_module)


def test_npu_prefill_attention_consumes_common_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fia_kwargs: dict[str, Any] = {}

    def fake_fia_v2(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        fia_kwargs.update(kwargs)
        return torch.zeros(7, 8, 64), torch.empty(1)

    fake_torch_npu = SimpleNamespace(npu_fused_infer_attention_score_v2=fake_fia_v2)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    npu_name = "vllm_gr.v1.attention.backends.beam_attn_npu"
    # ``from pkg import beam_attn_npu`` would return a stale module on machines
    # where the module was already imported (the package attribute short-circuits
    # _handle_fromlist), so resolve it explicitly and patch its torch_npu global.
    try:
        beam_attn_npu = sys.modules[npu_name]
    except KeyError:
        import importlib

        beam_attn_npu = importlib.import_module(npu_name)
    monkeypatch.setattr(beam_attn_npu, "torch_npu", fake_torch_npu)

    from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadata

    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    metadata = BeamAttentionMetadata(
        num_actual_tokens=7,
        max_query_len=4,
        query_start_loc=torch.tensor([0, 3, 7, 0, 0], dtype=torch.int32),
        max_seq_len=7,
        seq_lens=torch.tensor([7, 4, 0], dtype=torch.int32),
        block_table=block_table,
        slot_mapping=torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64),
        causal=True,
        is_prefill_graph_exec=True,
        num_reqs=2,
        num_heads=8,
        num_kv_heads=8,
        block_size=16,
        query_start_loc_cpu=[0, 3, 7, 0, 0],
        seq_lens_cpu=[7, 4, 0],
    )
    output = torch.zeros(7, 8, 64)
    query = torch.zeros(7, 8, 64)
    key_cache = torch.zeros(8, 16, 8, 64)
    value_cache = torch.zeros(8, 16, 8, 64)

    beam_attn_npu.prefill_attention(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        attn_metadata=metadata,
        sliding_window_size=None,
        scale=0.125,
        logits_soft_cap=0.0,
        vllm_flash_attn_version=0,
    )

    assert fia_kwargs["actual_seq_qlen"] == [3, 7]
    assert fia_kwargs["actual_seq_kvlen"] == [7, 4]
    assert fia_kwargs["block_table"] is block_table
    assert fia_kwargs["num_query_heads"] == 8
    assert fia_kwargs["num_key_value_heads"] == 8
    assert fia_kwargs["block_size"] == 16
    assert fia_kwargs["input_layout"] == "TND"
    assert torch.equal(output[:4], torch.zeros(4, 8, 64))


@pytest.mark.parametrize("tuple_cache", [False, True])  # type: ignore[misc]
def test_forward_preserves_cache_and_attention_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tuple_cache: bool,
) -> None:
    calls: dict[str, Any] = {}

    def reshape_and_cache(*args: Any) -> None:
        calls["reshape"] = args

    def cascade_attention(**kwargs: Any) -> None:
        calls["cascade"] = kwargs

    execution = SimpleNamespace(
        reshape_and_cache=reshape_and_cache,
        cascade_attention=cascade_attention,
    )
    monkeypatch.setattr(
        beam_attn,
        "_get_platform_execution_module",
        lambda: execution,
    )
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", tuple_cache)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", not tuple_cache)

    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_dtype = "auto"
    impl.sliding_window = (6, 0)
    impl.scale = 0.5
    impl.logits_soft_cap = 9.0
    impl.vllm_flash_attn_version = 3
    impl.sinks = torch.empty(1)
    impl.alibi_slopes = torch.empty(2)

    key_cache = torch.zeros(2, 2)
    value_cache = torch.ones(2, 2)
    kv_cache: Any
    if tuple_cache:
        kv_cache = (key_cache, value_cache)
    else:
        kv_cache = torch.stack((key_cache, value_cache))

    key = torch.empty(1)
    value = torch.empty(1)
    query = torch.empty(1)
    output = torch.empty(1)
    slot_mapping = torch.tensor([0])
    metadata = SimpleNamespace(slot_mapping=slot_mapping)
    k_scale = torch.tensor(1.0)
    v_scale = torch.tensor(2.0)
    layer = (
        SimpleNamespace() if tuple_cache else SimpleNamespace(_k_scale=k_scale, _v_scale=v_scale)
    )

    returned = impl.forward(
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=metadata,
        output=output,
    )

    assert returned is output
    reshape_args = calls["reshape"]
    assert reshape_args[0] is key
    assert reshape_args[1] is value
    torch.testing.assert_close(reshape_args[2], key_cache)
    torch.testing.assert_close(reshape_args[3], value_cache)
    assert reshape_args[4] is slot_mapping
    assert reshape_args[5] == "auto"
    assert reshape_args[6] is (None if tuple_cache else k_scale)
    assert reshape_args[7] is (None if tuple_cache else v_scale)

    cascade_args = calls["cascade"]
    assert cascade_args["output"] is output
    assert cascade_args["query"] is query
    torch.testing.assert_close(cascade_args["key_cache"], key_cache)
    torch.testing.assert_close(cascade_args["value_cache"], value_cache)
    assert cascade_args["attn_metadata"] is metadata
    assert cascade_args["sliding_window_size"] == [6, 0]
    assert cascade_args["scale"] == 0.5
    assert cascade_args["logits_soft_cap"] == 9.0
    assert cascade_args["vllm_flash_attn_version"] == 3
    assert cascade_args["s_aux"] is impl.sinks
    assert cascade_args["alibi_slopes"] is impl.alibi_slopes


def test_prefill_forward_applies_fp8_cache_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pure-prefill fast path converts FP8 KV caches like the ordinary path."""
    calls: dict[str, Any] = {}

    def fake_prefill_attention(**kwargs: Any) -> None:
        calls.update(kwargs)

    impl = SimpleNamespace(
        kv_sharing_target_layer_name=None,
        kv_cache_dtype="fp8_e4m3",
        sliding_window=None,
        scale=1.0,
        logits_soft_cap=0.0,
        vllm_flash_attn_version=3,
        sinks=None,
        alibi_slopes=None,
        prefill_attention=fake_prefill_attention,
    )
    platform = SimpleNamespace(reshape_and_cache=lambda *args, **kwargs: None)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", True)
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", False)
    monkeypatch.setattr(beam_attn, "_get_platform_execution_module", lambda: platform)

    layer = SimpleNamespace(_k_scale=None, _v_scale=None)
    kv_cache = torch.zeros(2, 8, 16, 8, 64)  # stacked [2, num_blocks, ...]
    query = torch.zeros(7, 8, 64)
    key = value = torch.zeros(7, 8, 64)
    attn_metadata = SimpleNamespace(slot_mapping=torch.zeros(7, dtype=torch.int64))
    output = torch.zeros(7, 8, 64)

    beam_attn.BeamAttentionImpl.prefill_forward(
        impl, layer, query, key, value, kv_cache, attn_metadata, output
    )

    assert calls["key_cache"].dtype is torch.float8_e4m3fn
    assert calls["value_cache"].dtype is torch.float8_e4m3fn
    # view(dtype) reinterprets bytes: fp32 (4B) -> fp8 (1B) quadruples the
    # trailing dimension (8 * 16 * 8 * 64 fp32 elems = 8 * 16 * 8 * 256 fp8).
    assert calls["key_cache"].shape == (8, 16, 8, 256)
    assert calls["value_cache"].shape == (8, 16, 8, 256)


def test_prefill_forward_keeps_non_fp8_cache_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-fp8 KV caches pass through the prefill fast path unchanged."""
    calls: dict[str, Any] = {}

    def fake_prefill_attention(**kwargs: Any) -> None:
        calls.update(kwargs)

    impl = SimpleNamespace(
        kv_sharing_target_layer_name=None,
        kv_cache_dtype="auto",
        sliding_window=None,
        scale=1.0,
        logits_soft_cap=0.0,
        vllm_flash_attn_version=3,
        sinks=None,
        alibi_slopes=None,
        prefill_attention=fake_prefill_attention,
    )
    platform = SimpleNamespace(reshape_and_cache=lambda *args, **kwargs: None)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", True)
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", False)
    monkeypatch.setattr(beam_attn, "_get_platform_execution_module", lambda: platform)

    layer = SimpleNamespace(_k_scale=None, _v_scale=None)
    kv_cache = torch.zeros(2, 8, 16, 8, 64)
    query = torch.zeros(7, 8, 64)
    key = value = torch.zeros(7, 8, 64)
    attn_metadata = SimpleNamespace(slot_mapping=torch.zeros(7, dtype=torch.int64))
    output = torch.zeros(7, 8, 64)

    beam_attn.BeamAttentionImpl.prefill_forward(
        impl, layer, query, key, value, kv_cache, attn_metadata, output
    )

    assert calls["key_cache"].dtype is torch.float32
    assert calls["value_cache"].dtype is torch.float32
    assert calls["key_cache"].shape == (8, 16, 8, 64)


def test_gpu_prefill_attention_forwards_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GPU pure-prefill fast path forwards attention sinks to FlashAttention."""
    from vllm_gr.v1.attention.backends import beam_attn_gpu

    calls: dict[str, Any] = {}

    def fake_fa(*args: Any, **kwargs: Any) -> Any:
        calls.update(kwargs)
        return torch.zeros(4, 8, 64)

    monkeypatch.setattr(beam_attn_gpu, "flash_attn_varlen_func", fake_fa)

    sink = torch.zeros(4, 8, 64)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        max_query_len=4,
        seq_lens=torch.tensor([4], dtype=torch.int32),
        max_seq_len=4,
        block_table=None,
        causal=True,
    )
    output = torch.zeros(4, 8, 64)
    query = torch.zeros(4, 8, 64)
    key_cache = torch.zeros(1, 4, 8, 64)
    value_cache = torch.zeros(1, 4, 8, 64)

    beam_attn_gpu.prefill_attention(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        attn_metadata=metadata,
        sliding_window_size=None,
        scale=0.125,
        logits_soft_cap=0.0,
        vllm_flash_attn_version=3,
        s_aux=sink,
    )

    assert calls["s_aux"] is sink
