# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging

import torch
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger("vllm_gr.ops.npu.qkv_rmsnorm_rope_patch")

try:
    from vllm.triton_utils import tl, triton
except ImportError:  # pragma: no cover - Triton unavailable in some environments.
    tl = None
    triton = None


def _resolve_triton_ascend_op(op_name: str):
    if not HAS_TRITON:
        raise RuntimeError(f"Triton op '{op_name}' cannot be resolved because HAS_TRITON is False")

    extension_module = None
    try:
        import triton.language.extra.cann.extension as extension_module  # type: ignore
    except ImportError:
        extension_module = None

    if extension_module is not None:
        extension_op = getattr(extension_module, op_name, None)
        if extension_op is not None:
            return extension_op

    tl_op = getattr(tl, op_name, None)
    if tl_op is not None:
        return tl_op

    raise RuntimeError(
        f"Failed to resolve Triton op '{op_name}': "
        "neither triton.language.extra.cann.extension nor triton.language provides it."
    )


try:
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
except Exception:  # pragma: no cover - resolved at runtime in NPU env.
    get_vectorcore_num = None

if HAS_TRITON:
    try:
        _extract_slice = _resolve_triton_ascend_op("extract_slice")
        _get_element = _resolve_triton_ascend_op("get_element")
        _insert_slice = _resolve_triton_ascend_op("insert_slice")
    except Exception:  # pragma: no cover - resolved at runtime in NPU env.
        _extract_slice = None
        _get_element = None
        _insert_slice = None
else:
    _extract_slice = None
    _get_element = None
    _insert_slice = None

_PATCH_OP_NAME = "qkv_rmsnorm_rope_gr_patch"
_PATCHED = False
_REGISTERED = False


def _helpers_available() -> bool:
    return all(
        helper is not None
        for helper in (
            tl,
            triton,
            _extract_slice,
            _get_element,
            _insert_slice,
            get_vectorcore_num,
        )
    )


if triton is not None:

    @triton.jit
    def split_qkv_rmsnorm_rope_kernel(
        input_gm_ptr,
        q_gm_ptr,
        k_gm_ptr,
        v_gm_ptr,
        q_weight_ptr,
        q_bias_ptr,
        k_weight_ptr,
        k_bias_ptr,
        batch_size,
        q_hidden_size: tl.constexpr,
        kv_hidden_size: tl.constexpr,
        total_hidden_size: tl.constexpr,
        eps: tl.constexpr,
        BIAS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROPE_DIM: tl.constexpr,
        HALF_ROPE_DIM: tl.constexpr,
        IS_PARTIAL_ROPE: tl.constexpr,
        num_vectorcore: tl.constexpr,
        batch_size_per_iter_per_vec: tl.constexpr,
        qk_head_nums_per_iter_per_vec: tl.constexpr,
        q_head_num: tl.constexpr,
        kv_head_num: tl.constexpr,
        qk_head_num_sum: tl.constexpr,
        v_batch_size_per_iter_per_vec: tl.constexpr,
        positions_gm_ptr,
        cos_sin_cache_gm_ptr,
    ):
        row_pid = tl.program_id(0)

        q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
        k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))

        batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore)
        iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)
        v_iter_num_per_vec = tl.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)
        input_batch_offset = row_pid * batch_size_per_vec
        mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
        nblk_idx = tl.arange(0, q_hidden_size + kv_hidden_size)
        nmask = nblk_idx < total_hidden_size

        input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

        pos_indices = input_batch_offset + tl.arange(0, batch_size_per_iter_per_vec)
        output_q_nblk_idx = tl.arange(0, q_hidden_size)
        output_q_nmask = output_q_nblk_idx < q_hidden_size
        output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
        output_kv_nmask = output_kv_nblk_idx < kv_hidden_size
        sin_cos_range = tl.arange(0, ROPE_DIM)
        cos_sin_cache_offset = cos_sin_cache_gm_ptr + sin_cos_range

        for iter in tl.range(iter_num_per_vec):
            pos_offset = iter * batch_size_per_iter_per_vec
            x = tl.load(
                positions_gm_ptr + pos_indices + pos_offset,
                mask=(pos_indices + pos_offset) < input_batch_offset_end,
            )
            mmask = (mblk_idx + pos_offset) < input_batch_offset_end
            mask = (mmask[:, None]) & (nmask[None, :])
            idx = (mblk_idx + pos_offset)[:, None] * total_hidden_size + nblk_idx[None, :]
            values_tmp1 = tl.load(input_gm_ptr + idx, mask=mask).reshape(
                qk_head_nums_per_iter_per_vec, HEAD_DIM
            )
            if BIAS:
                q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
                k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))

            values_tmp3 = tl.zeros((batch_size_per_iter_per_vec, ROPE_DIM), dtype=tl.bfloat16)
            for i in tl.range(batch_size_per_iter_per_vec):
                pos = _get_element(x, (i,))
                values_tmp3 = _insert_slice(
                    values_tmp3.reshape(batch_size_per_iter_per_vec, ROPE_DIM),
                    tl.load(pos * ROPE_DIM + cos_sin_cache_offset[:, None]).reshape(1, ROPE_DIM),
                    offsets=(i, 0),
                    sizes=(1, ROPE_DIM),
                    strides=(1, 1),
                )
            values_tmp3 = values_tmp3.reshape(batch_size_per_iter_per_vec, 1, ROPE_DIM)
            cos = _extract_slice(
                values_tmp3,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            sin = _extract_slice(
                values_tmp3,
                offsets=(0, 0, HALF_ROPE_DIM),
                sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )

            normalized_values = values_tmp1.to(tl.float32)
            normalized_values = normalized_values * normalized_values
            normalized_values = tl.sum(normalized_values, axis=1) / HEAD_DIM
            normalized_values = 1 / tl.sqrt(normalized_values + eps).reshape(
                qk_head_nums_per_iter_per_vec, 1
            )
            normalized_values = values_tmp1 * normalized_values

            normalized_values_tmp = _extract_slice(
                normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM),
                strides=(1, 1, 1),
            )

            if BIAS:
                normalized_values_tmp = (
                    normalized_values_tmp * q_weight_values + q_bias_values
                ).to(tl.bfloat16)
            else:
                normalized_values_tmp = (normalized_values_tmp * q_weight_values).to(tl.bfloat16)

            values_tmp = tl.zeros(
                (batch_size_per_iter_per_vec, q_head_num, ROPE_DIM), dtype=tl.bfloat16
            )
            x1 = _extract_slice(
                normalized_values_tmp,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            x2 = _extract_slice(
                normalized_values_tmp,
                offsets=(0, 0, HALF_ROPE_DIM),
                sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            values_tmp = _insert_slice(
                values_tmp,
                x1 * cos - x2 * sin,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            values_tmp = _insert_slice(
                values_tmp,
                x2 * cos + x1 * sin,
                offsets=(0, 0, HALF_ROPE_DIM),
                sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            q_output_idx = (
                output_q_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * q_hidden_size
            )
            mask = (mmask[:, None]) & (output_q_nmask[None, :])
            if IS_PARTIAL_ROPE:
                normalized_values_tmp = _insert_slice(
                    normalized_values_tmp,
                    values_tmp,
                    offsets=(0, 0, 0),
                    sizes=(batch_size_per_iter_per_vec, q_head_num, ROPE_DIM),
                    strides=(1, 1, 1),
                )
                tl.store(
                    q_gm_ptr + q_output_idx,
                    normalized_values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                    mask=mask,
                )
            else:
                tl.store(
                    q_gm_ptr + q_output_idx,
                    values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                    mask=mask,
                )

            normalized_values_tmp1 = _extract_slice(
                normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
                offsets=(0, q_head_num, 0),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
                strides=(1, 1, 1),
            )

            if BIAS:
                normalized_values_tmp1 = (
                    normalized_values_tmp1 * k_weight_values + k_bias_values
                ).to(tl.bfloat16)
            else:
                normalized_values_tmp1 = (normalized_values_tmp1 * k_weight_values).to(tl.bfloat16)

            values_tmp2 = tl.zeros(
                (batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM), dtype=tl.bfloat16
            )
            x1 = _extract_slice(
                normalized_values_tmp1,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            x2 = _extract_slice(
                normalized_values_tmp1,
                offsets=(0, 0, HALF_ROPE_DIM),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            values_tmp2 = _insert_slice(
                values_tmp2,
                x1 * cos - x2 * sin,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )
            values_tmp2 = _insert_slice(
                values_tmp2,
                x2 * cos + x1 * sin,
                offsets=(0, 0, HALF_ROPE_DIM),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
                strides=(1, 1, 1),
            )

            kv_output_idx = (
                output_kv_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * kv_hidden_size
            )
            mask = (mmask[:, None]) & (output_kv_nmask[None, :])
            if IS_PARTIAL_ROPE:
                normalized_values_tmp1 = _insert_slice(
                    normalized_values_tmp1,
                    values_tmp2,
                    offsets=(0, 0, 0),
                    sizes=(batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM),
                    strides=(1, 1, 1),
                )
                tl.store(
                    k_gm_ptr + kv_output_idx,
                    normalized_values_tmp1.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                    mask=mask,
                )
            else:
                tl.store(
                    k_gm_ptr + kv_output_idx,
                    values_tmp2.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                    mask=mask,
                )

        mblk_idx = tl.arange(0, v_batch_size_per_iter_per_vec) + input_batch_offset
        nblk_idx = tl.arange(q_hidden_size + kv_hidden_size, total_hidden_size)
        nmask = nblk_idx < total_hidden_size
        out_nblk_idx = tl.arange(0, kv_hidden_size)
        out_nmask = out_nblk_idx < kv_hidden_size

        for _ in tl.range(v_iter_num_per_vec):
            mmask = mblk_idx < input_batch_offset_end
            mask = (mmask[:, None]) & (nmask[None, :])
            idx = mblk_idx[:, None] * total_hidden_size + nblk_idx[None, :]
            values = tl.load(input_gm_ptr + idx, mask=mask)
            out_idx = mblk_idx[:, None] * kv_hidden_size + out_nblk_idx[None, :]
            out_mask = (mmask[:, None]) & (out_nmask[None, :])
            tl.store(v_gm_ptr + out_idx, values, mask=out_mask)
            mblk_idx += v_batch_size_per_iter_per_vec


def _convert_sin_cos_to_cache_and_positions(
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sin.shape != cos.shape:
        raise ValueError(f"sin/cos shape mismatch: sin={tuple(sin.shape)} cos={tuple(cos.shape)}")
    if sin.ndim != 4:
        raise ValueError(
            "Expected 4D sin/cos tensors like [1, T, 1, rope_dim] or [T, 1, 1, rope_dim], "
            f"got {tuple(sin.shape)}"
        )

    if sin.shape[0] == 1:
        sin_rows = sin.squeeze(0).squeeze(1)
        cos_rows = cos.squeeze(0).squeeze(1)
    else:
        sin_rows = sin.squeeze(1).squeeze(1)
        cos_rows = cos.squeeze(1).squeeze(1)

    if sin_rows.ndim != 2:
        raise ValueError(
            "Unable to normalize sin/cos tensors into [num_tokens, rope_dim]; "
            f"got {tuple(sin_rows.shape)}"
        )

    rope_dim = sin_rows.shape[-1]
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")

    half = rope_dim // 2
    cache = torch.cat((cos_rows[:, :half], sin_rows[:, :half]), dim=-1).contiguous()
    positions = torch.arange(cache.shape[0], device=cache.device, dtype=torch.int64)
    return cache, positions


def split_qkv_rmsnorm_rope_gr_impl(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _helpers_available():
        raise RuntimeError(
            "split_qkv_rmsnorm_rope GR patch requires Triton + vllm_ascend Triton helpers"
        )

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()

    if cos_sin_cache is None or positions is None:
        if sin is None or cos is None:
            raise ValueError(
                "Expected either (cos_sin_cache, positions) or legacy (sin, cos) inputs for qkv_rmsnorm_rope"
            )
        cos_sin_cache, positions = _convert_sin_cos_to_cache_and_positions(sin=sin, cos=cos)

    num_vectorcore = get_vectorcore_num()
    rope_dim = cos_sin_cache.shape[-1]
    batch_size = input.shape[0]
    bias = q_bias is not None
    is_partial_rope = rope_dim != head_dim
    total_hidden_size = q_hidden_size + kv_hidden_size * 2

    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    q_head_num = q_hidden_size // head_dim
    kv_head_num = kv_hidden_size // head_dim

    ub_size = 87040
    if is_partial_rope:
        factor = 5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 4 + q_head_num * rope_dim
        batch_size_per_iter_per_vec = int(ub_size / input.element_size()) // factor
    else:
        factor = 5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 2 + q_head_num * rope_dim // 2
        batch_size_per_iter_per_vec = int(ub_size / input.element_size()) // factor
    batch_size_per_iter_per_vec = max(1, batch_size_per_iter_per_vec)
    qk_head_num_sum = int(q_head_num + kv_head_num)
    qk_head_nums_per_iter_per_vec = batch_size_per_iter_per_vec * qk_head_num_sum

    grid = (num_vectorcore, 1, 1)
    v_batch_size_per_iter_per_vec = ub_size / torch.bfloat16.itemsize // (kv_hidden_size + 1)

    split_qkv_rmsnorm_rope_kernel[grid](
        input,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        bias,
        head_dim,
        rope_dim,
        rope_dim // 2,
        is_partial_rope,
        num_vectorcore,
        int(batch_size_per_iter_per_vec),
        int(qk_head_nums_per_iter_per_vec),
        q_head_num,
        kv_head_num,
        qk_head_num_sum,
        int(v_batch_size_per_iter_per_vec),
        positions,
        cos_sin_cache,
    )
    return q_output, k_output, v_output


def split_qkv_rmsnorm_rope_gr_impl_fake(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, head_dim, eps, q_bias, k_bias, cos, sin, cos_sin_cache, positions
    batch_size = input.shape[0]
    q_output = torch.empty(batch_size, int(q_hidden_size), device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, int(kv_hidden_size), device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, int(kv_hidden_size), device=input.device, dtype=input.dtype)
    return q_output, k_output, v_output


def register_gr_qkv_rmsnorm_rope_op(force: bool = False) -> bool:
    global _REGISTERED
    if _REGISTERED and not force:
        return True

    if not _helpers_available():
        logger.debug(
            "Skipping GR qkv_rmsnorm_rope patch op registration because Triton helpers are unavailable"
        )
        return False

    try:
        direct_register_custom_op(
            op_name=_PATCH_OP_NAME,
            op_func=split_qkv_rmsnorm_rope_gr_impl,
            fake_impl=split_qkv_rmsnorm_rope_gr_impl_fake,
            mutates_args=[],
            dispatch_key="PrivateUse1",
        )
        logger.info("vllm_gr: registered patched op %s", _PATCH_OP_NAME)
    except Exception as exc:
        text = str(exc).lower()
        if "already" not in text and "same name" not in text:
            raise
        logger.debug("GR qkv_rmsnorm_rope patch op already registered: %s", exc)

    _REGISTERED = True
    return True


def apply_split_qkv_rmsnorm_rope_patch(force: bool = False) -> int:
    global _PATCHED
    if _PATCHED and not force:
        return 0

    if not register_gr_qkv_rmsnorm_rope_op(force=force):
        return 0

    ns = torch.ops.vllm
    original = getattr(ns, "qkv_rmsnorm_rope", None)
    patched = getattr(ns, _PATCH_OP_NAME)
    if original is patched:
        _PATCHED = True
        return 0

    setattr(ns, "qkv_rmsnorm_rope", patched)
    _PATCHED = True
    logger.info("vllm_gr: patched torch.ops.vllm.qkv_rmsnorm_rope -> %s", _PATCH_OP_NAME)
    return 1
