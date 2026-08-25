# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Triton fused select_unshared_kv fast path.

The fused path (select_unshared_kv_fused) must be element-wise identical to
the Python reference (select_unshared_kv aux-buf path) across the full
semantic surface: fan-out beam selection (not a permutation), permutations,
shrinking beam counts (n < W), multi-step reorder ranges, and both fp16/bf16.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gr.v1.attention.backends.beam_kernels import (
    select_unshared_kv,
    select_unshared_kv_fused,
)

CUDA_OK = torch.cuda.is_available()


def _make_pool(
    max_batch: int,
    W: int,
    H: int,
    max_steps: int,
    D: int,
    dtype: torch.dtype,
    seed: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    key = torch.randn(max_batch, W, H, max_steps, D, generator=g, dtype=torch.float32)
    value = torch.randn(max_batch, W, H, max_steps, D, generator=g, dtype=torch.float32)
    return key.to(dtype).to(device), value.to(dtype).to(device)


def _reference(
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    beam_index: torch.Tensor,
    slot: int,
    decode_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Python reference via select_unshared_kv with aux bufs (fresh copies)."""
    k = pool_k.clone()
    v = pool_v.clone()
    W, H, max_steps, D = k.shape[1:]
    aux_k = torch.zeros(W, H, max_steps, D, dtype=k.dtype, device=k.device)
    aux_v = torch.zeros_like(aux_k)
    select_unshared_kv(
        beam_index,
        slot,
        [k],
        [v],
        decode_step,
        aux_key_bufs=[aux_k],
        aux_value_bufs=[aux_v],
    )
    return k, v


def _fused(
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    beam_index: torch.Tensor,
    slot: int,
    decode_step: int,
) -> tuple[bool, torch.Tensor, torch.Tensor]:
    k = pool_k.clone()
    v = pool_v.clone()
    W, H, max_steps, D = k.shape[1:]
    aux_k = torch.zeros(1, W, H, max_steps, D, dtype=k.dtype, device=k.device)
    aux_v = torch.zeros_like(aux_k)
    handled = select_unshared_kv_fused(
        beam_index,
        slot,
        [k],
        [v],
        decode_step,
        aux_key=aux_k,
        aux_value=aux_v,
    )
    return handled, k, v


BEAM_INDEX_CASES = {
    "identity": [0, 1, 2, 3],
    "permutation": [3, 1, 0, 2],
    "fanout_2from0": [0, 0, 1, 2],
    "fanout_all_from0": [0, 0, 0, 0],
}


@pytest.mark.skipif(not CUDA_OK, reason="requires CUDA + Triton")  # type: ignore
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore
@pytest.mark.parametrize("case", BEAM_INDEX_CASES.keys())  # type: ignore
def test_fused_matches_reference(dtype: torch.dtype, case: str) -> None:
    W, H, max_steps, D = 4, 2, 3, 16
    slot, max_batch = 0, 1
    pool_k, pool_v = _make_pool(max_batch, W, H, max_steps, D, dtype, seed=42)
    beam_index = torch.tensor(BEAM_INDEX_CASES[case], dtype=torch.int32, device="cuda")

    # decode_step 0 → reorder 1 step; decode_step 2 → reorder all 3 steps
    for decode_step in (0, 1, 2):
        rk, rv = _reference(pool_k, pool_v, beam_index, slot, decode_step)
        handled, fk, fv = _fused(pool_k, pool_v, beam_index, slot, decode_step)
        assert handled, "fused path declined on CUDA — check guards"
        torch.testing.assert_close(
            fk.cpu(), rk.cpu(), msg=f"key mismatch case={case} step={decode_step}"
        )
        torch.testing.assert_close(
            fv.cpu(), rv.cpu(), msg=f"value mismatch case={case} step={decode_step}"
        )


@pytest.mark.skipif(not CUDA_OK, reason="requires CUDA + Triton")  # type: ignore
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore
def test_fused_shrink_beam_count(dtype: torch.dtype) -> None:
    """n < W: only the first n pool rows are reordered; trailing rows keep
    the reference's semantics (stale-but-unreferenced), which the reference
    also produces because both write only rows [0:n]."""
    W = 8
    H, max_steps, D = 2, 3, 32
    pool_k, pool_v = _make_pool(1, W, H, max_steps, D, dtype, seed=7)
    beam_index = torch.tensor([4, 4, 0, 1, 3], dtype=torch.int32, device="cuda")
    decode_step = 1

    rk, rv = _reference(pool_k, pool_v, beam_index, 0, decode_step)
    handled, fk, fv = _fused(pool_k, pool_v, beam_index, 0, decode_step)
    assert handled
    torch.testing.assert_close(fk.cpu(), rk.cpu())
    torch.testing.assert_close(fv.cpu(), rv.cpu())


@pytest.mark.skipif(not CUDA_OK, reason="requires CUDA + Triton")  # type: ignore
def test_fused_multi_layer_and_slot() -> None:
    """The real call pattern: per-layer pools with distinct data, per-layer
    aux buffers ([L, W, H, S, D]), and a nonzero batch slot — a cross-layer
    staging clobber would be visible here."""
    L, max_batch, W, H, max_steps, D = 3, 2, 6, 2, 3, 16
    pools_k, pools_v = [], []
    for layer in range(L):  # DISTINCT seed per layer — layers must differ
        k, v = _make_pool(max_batch, W, H, max_steps, D, torch.bfloat16, seed=100 + layer)
        pools_k.append(k)
        pools_v.append(v)
    aux_k = torch.zeros(L, W, H, max_steps, D, dtype=torch.bfloat16, device="cuda")
    aux_v = torch.zeros_like(aux_k)
    beam_index = torch.tensor([1, 1, 0, 5, 5, 2], dtype=torch.int32, device="cuda")
    slot, decode_step = 1, 2

    ref_k = [k.clone() for k in pools_k]
    ref_v = [v.clone() for v in pools_v]
    # Key and value need SEPARATE scratch tensors: the reference stages both
    # copies before either write-back, so one shared buffer would let the
    # value copy overwrite the key staging.
    ref_aux_k = torch.zeros(W, H, max_steps, D, dtype=torch.bfloat16, device="cuda")
    ref_aux_v = torch.zeros(W, H, max_steps, D, dtype=torch.bfloat16, device="cuda")
    for k, v in zip(ref_k, ref_v):
        select_unshared_kv(
            beam_index,
            slot,
            [k],
            [v],
            decode_step,
            aux_key_bufs=[ref_aux_k],
            aux_value_bufs=[ref_aux_v],
        )

    handled = select_unshared_kv_fused(
        beam_index, slot, pools_k, pools_v, decode_step, aux_key=aux_k, aux_value=aux_v
    )
    assert handled
    for i, (k, v) in enumerate(zip(pools_k, pools_v)):
        torch.testing.assert_close(k.cpu(), ref_k[i].cpu(), msg=f"layer {i} key")
        torch.testing.assert_close(v.cpu(), ref_v[i].cpu(), msg=f"layer {i} value")


@pytest.mark.skipif(not CUDA_OK, reason="requires CUDA + Triton")  # type: ignore
def test_fused_returns_false_off_cuda() -> None:
    pool_k, pool_v = _make_pool(1, 4, 2, 3, 16, torch.bfloat16, 1, device="cpu")
    aux = torch.zeros(1, 4, 2, 3, 16, dtype=torch.bfloat16)
    handled = select_unshared_kv_fused(
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        0,
        [pool_k],
        [pool_v],
        0,
        aux_key=aux,
        aux_value=aux,
    )
    assert handled is False
