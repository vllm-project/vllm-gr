# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host checks of the ROCm Beam attention policy and FlashAttention adapter."""

import sys
from types import ModuleType
from typing import Any

import pytest
import torch

from vllm_gr.utils import accelerator
from vllm_gr.v1.attention.backends import beam_attn, beam_attn_gpu, rocm_flash_attn
from vllm_gr.v1.attention.backends.beam_attention_state import (
    normalize_flash_attn_state,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture  # type: ignore[misc]
def on_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "is_rocm", lambda: True)
    monkeypatch.setattr(beam_attn_gpu, "is_available", lambda: True)


@pytest.fixture  # type: ignore[misc]
def upstream(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Install a fake upstream ``flash_attn`` and record its calls."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        out = torch.zeros_like(args[0])
        lse = torch.zeros(args[0].shape[1], args[0].shape[0])
        return (out, lse, None) if kwargs["return_attn_probs"] else out

    module = ModuleType("flash_attn")
    module.flash_attn_varlen_func = flash_attn_varlen_func  # type: ignore[attr-defined]
    interface = ModuleType("flash_attn.flash_attn_interface")
    interface.USE_TRITON_ROCM = False  # type: ignore[attr-defined]
    module.flash_attn_interface = interface  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flash_attn", module)
    monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", interface)
    return calls


def test_adapter_maps_seqused_k_and_drops_cuda_only_options(upstream: list[Any]) -> None:
    q = torch.zeros(3, 4, 8)
    k = torch.zeros(2, 128, 2, 8)
    out, lse = rocm_flash_attn.flash_attn_varlen_func(
        q=q,
        k=k,
        v=k,
        cu_seqlens_q=torch.tensor([0, 1, 3], dtype=torch.int32),
        seqused_k=torch.tensor([5, 130], dtype=torch.int32),
        max_seqlen_q=2,
        max_seqlen_k=130,
        softmax_scale=0.5,
        causal=False,
        window_size=None,
        softcap=None,
        block_table=torch.zeros(2, 2, dtype=torch.int32),
        return_softmax_lse=True,
        fa_version=2,
        num_splits=0,
        scheduler_metadata=None,
    )

    assert out.shape == q.shape and lse.shape == (4, 3)
    (args, kwargs) = upstream[0]
    assert args[4].tolist() == [0, 5, 135]
    assert args[5:] == (2, 130)
    assert kwargs["window_size"] == (-1, -1) and kwargs["softcap"] == 0.0
    assert kwargs["return_attn_probs"] is True
    assert not {"fa_version", "num_splits", "scheduler_metadata", "seqused_k"} & kwargs.keys()


def test_gather_paged_kv_packs_rows_in_request_order() -> None:
    blocks, block_size = 5, 4
    key = torch.arange(blocks * block_size, dtype=torch.float32).view(blocks, block_size, 1, 1)
    table = torch.tensor([[3, 1], [0, 4]], dtype=torch.int32)
    lengths = torch.tensor([6, 2], dtype=torch.int32)

    packed_k, packed_v, cu = rocm_flash_attn.gather_paged_kv(key, -key, table, lengths, 8)

    expected = [12, 13, 14, 15, 4, 5, 0, 1]
    assert cu.tolist() == [0, 6, 8]
    assert packed_k[:8].flatten().tolist() == expected
    assert packed_v[:8].flatten().tolist() == [-x for x in expected]


def test_triton_backend_gathers_paged_kv(upstream: list[Any]) -> None:
    sys.modules["flash_attn.flash_attn_interface"].USE_TRITON_ROCM = True  # type: ignore[attr-defined]
    q = torch.zeros(2, 2, 8)
    cache = torch.zeros(3, 16, 2, 8)

    rocm_flash_attn.flash_attn_varlen_func(
        q=q,
        k=cache,
        v=cache,
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        seqused_k=torch.tensor([20, 3], dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=32,
        block_table=torch.tensor([[0, 1], [2, 0]], dtype=torch.int32),
    )

    (args, kwargs) = upstream[0]
    assert kwargs["block_table"] is None
    assert args[1].shape == (2 * 32, 2, 8)
    assert args[4].tolist() == [0, 20, 23]


def test_adapter_rejects_sinks_and_unaligned_paged_blocks(upstream: list[Any]) -> None:
    q = torch.zeros(1, 2, 8)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    common: dict[str, Any] = {"q": q, "cu_seqlens_q": cu, "max_seqlen_q": 1, "max_seqlen_k": 1}
    with pytest.raises(NotImplementedError, match="sinks"):
        rocm_flash_attn.flash_attn_varlen_func(
            k=q, v=q, cu_seqlens_k=cu, s_aux=torch.zeros(2), **common
        )
    paged = torch.zeros(1, 16, 2, 8)
    with pytest.raises(ValueError, match="divisible by 128"):
        rocm_flash_attn.flash_attn_varlen_func(
            k=paged,
            v=paged,
            seqused_k=torch.tensor([1], dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            **common,
        )
    assert upstream == []


def test_rocm_single_query_gqa_lse_is_already_canonical(on_rocm: None) -> None:
    out = torch.zeros(3, 4, 8)
    lse = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    state = normalize_flash_attn_state(out, lse, fa_version=2, num_kv_heads=2, max_query_len=1)

    assert torch.equal(state.lse, lse)


def test_rocm_attention_policy(on_rocm: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", True)

    assert beam_attn_gpu.get_flash_attn_version() == 2
    assert not beam_attn_gpu.supports_sinks()
    assert not beam_attn_gpu.short_suffix_attention_supported()
    resolve: dict[str, Any] = {
        "decode_version": 2,
        "head_size": 128,
        "kv_cache_dtype": "auto",
        "requires_alibi": False,
        "has_sinks": False,
        "batch_invariant": False,
    }
    assert beam_attn_gpu.resolve_prefill_flash_attn_version("auto", **resolve) == 2
    with pytest.raises(ValueError, match="ROCm provides FlashAttention 2 only"):
        beam_attn_gpu.resolve_prefill_flash_attn_version(3, **resolve)
    sizes = beam_attn.BeamAttentionBackend.get_supported_kernel_block_sizes()
    assert [size.base for size in sizes] == [rocm_flash_attn.ROCM_PAGED_KV_BLOCK_SIZE]
    assert not beam_attn.BeamAttentionBackend.supports_block_size(16)
    assert beam_attn.BeamAttentionBackend.supports_block_size(256)
