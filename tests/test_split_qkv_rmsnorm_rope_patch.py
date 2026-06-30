# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code=misc

import gc

import numpy as np
import pytest
import torch

from vllm_gr.patch import apply_patches_if_needed

try:
    import vllm_ascend.ops.register_custom_ops  # noqa: F401
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
except Exception:  # pragma: no cover - only available in NPU runtime
    vllm_ascend = None
    init_device_properties_triton = None


MAX_POSITION_EMBEDDINGS = [262144]
NUM_TOKENS = [1, 16, 1024]
NUM_QKV_HEADS = [(12, 1), (64, 4)]
HEAD_SIZES = [128]
ROPE_DIMS = [64, 128]
EPS = [1e-6]
DTYPES = [torch.bfloat16]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-3


def _has_npu_runtime() -> bool:
    return (
        hasattr(torch, "npu")
        and torch.npu.is_available()
        and init_device_properties_triton is not None
    )


pytestmark = pytest.mark.skipif(
    not _has_npu_runtime(), reason="requires torch_npu + vllm_ascend runtime"
)


def custom_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = sin.shape[-1]
    sin = sin.to(torch.float32)
    cos = cos.to(torch.float32)
    q_rot = q[..., :rotary_dim]
    k_rot = k[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_pass = k[..., rotary_dim:]

    x1 = q_rot[..., : rotary_dim // 2]
    x2 = q_rot[..., rotary_dim // 2 :]
    cat_x = torch.cat([-x2, x1], axis=-1)
    mul1 = cat_x * sin
    mul2 = q_rot * cos
    q_rot = mul1 + mul2
    res1 = torch.cat([q_rot, q_pass], dim=-1)

    x1 = k_rot[..., : rotary_dim // 2]
    x2 = k_rot[..., rotary_dim // 2 :]
    cat_x = torch.cat([-x2, x1], axis=-1)
    mul1 = cat_x * sin
    mul2 = k_rot * cos
    k_rot = mul1 + mul2
    res2 = torch.cat([k_rot, k_pass], dim=-1)
    return res1, res2


def rms_norm(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    norm_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    input = input.to(torch.float32)
    norm_weight = norm_weight.to(torch.float32)
    reciprocal_std = 1 / torch.sqrt(torch.mean(input**2, axis=-1, keepdims=True) + eps)
    out = input * reciprocal_std * norm_weight
    if norm_bias is not None:
        norm_bias = norm_bias.to(torch.float32)
        out = out + norm_bias
    return out


def test_split_qkv_rmsnorm_rope_patch_installed() -> None:
    apply_patches_if_needed()
    assert getattr(torch.ops.vllm, "qkv_rmsnorm_rope", None) is not None


@pytest.mark.parametrize("max_position_embeddings", MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_QKV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rope_dim", ROPE_DIMS)
@torch.inference_mode()
def test_split_qkv_rmsnorm_rope_patch(
    max_position_embeddings: int,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    dtype: torch.dtype,
    seed: int,
    device: str,
    rope_dim: int,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()
    apply_patches_if_needed()

    q_hidden_size = num_q_heads * head_size
    kv_hidden_size = num_kv_heads * head_size
    qkv = torch.randn(num_tokens, q_hidden_size + kv_hidden_size * 2, dtype=dtype, device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)
    cos_sin_cache = (
        torch.from_numpy(np.random.uniform(0, 1, [max_position_embeddings, rope_dim]))
        .to(dtype)
        .npu()
    )
    positions = torch.randint(
        low=0, high=max_position_embeddings, size=(num_tokens,), dtype=torch.int64, device=device
    )

    q, k, v = torch.ops.vllm.qkv_rmsnorm_rope(
        input=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_size,
        eps=eps,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
    )

    cos, sin = (
        cos_sin_cache.index_select(0, positions)
        .view(num_tokens, 2, -1)
        .repeat(1, 1, 2)
        .chunk(2, dim=-2)
    )
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    _q, _k, v_gold = qkv.cpu().split([q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
    _q = rms_norm(_q.reshape(-1, head_size), q_weight.cpu(), eps)
    _k = rms_norm(_k.reshape(-1, head_size), k_weight.cpu(), eps)
    _q = _q.reshape(num_tokens, 1, -1, head_size)
    _k = _k.reshape(num_tokens, 1, -1, head_size)

    q_gold, k_gold = custom_rope(_q, _k, sin.cpu(), cos.cpu())
    q_gold = q_gold.reshape(num_tokens, -1)
    k_gold = k_gold.reshape(num_tokens, -1)

    torch.testing.assert_close(
        q.to(torch.float32).cpu(), q_gold, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL
    )
    torch.testing.assert_close(
        k.to(torch.float32).cpu(), k_gold, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL
    )
    torch.testing.assert_close(
        v.to(torch.float32).cpu(), v_gold.to(torch.float32), atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("max_position_embeddings", MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_QKV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rope_dim", ROPE_DIMS)
@torch.inference_mode()
def test_split_qkv_rmsnorm_rope_patch_with_bias(
    max_position_embeddings: int,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    dtype: torch.dtype,
    seed: int,
    device: str,
    rope_dim: int,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()
    apply_patches_if_needed()

    q_hidden_size = num_q_heads * head_size
    kv_hidden_size = num_kv_heads * head_size
    qkv = torch.randn(num_tokens, q_hidden_size + kv_hidden_size * 2, dtype=dtype, device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)
    q_bias = torch.randn(head_size, dtype=dtype, device=device)
    k_bias = torch.randn(head_size, dtype=dtype, device=device)
    cos_sin_cache = (
        torch.from_numpy(np.random.uniform(0, 1, [max_position_embeddings, rope_dim]))
        .to(dtype)
        .npu()
    )
    positions = torch.randint(
        low=0, high=max_position_embeddings, size=(num_tokens,), dtype=torch.int64, device=device
    )

    q, k, v = torch.ops.vllm.qkv_rmsnorm_rope(
        input=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_size,
        eps=eps,
        q_bias=q_bias,
        k_bias=k_bias,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
    )

    cos, sin = (
        cos_sin_cache.index_select(0, positions)
        .view(num_tokens, 2, -1)
        .repeat(1, 1, 2)
        .chunk(2, dim=-2)
    )
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    _q, _k, v_gold = qkv.cpu().split([q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
    _q = rms_norm(_q.reshape(-1, head_size), q_weight.cpu(), eps, norm_bias=q_bias.cpu())
    _k = rms_norm(_k.reshape(-1, head_size), k_weight.cpu(), eps, norm_bias=k_bias.cpu())
    _q = _q.reshape(num_tokens, 1, -1, head_size)
    _k = _k.reshape(num_tokens, 1, -1, head_size)

    q_gold, k_gold = custom_rope(_q, _k, sin.cpu(), cos.cpu())
    q_gold = q_gold.reshape(num_tokens, -1)
    k_gold = k_gold.reshape(num_tokens, -1)

    torch.testing.assert_close(
        q.to(torch.float32).cpu(), q_gold, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL
    )
    torch.testing.assert_close(
        k.to(torch.float32).cpu(), k_gold, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL
    )
    torch.testing.assert_close(
        v.to(torch.float32).cpu(), v_gold.to(torch.float32), atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
