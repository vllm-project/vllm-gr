# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_gr.v1.attention.backends.beam_attention_state import (
    normalize_flash_attn_state,
)


@pytest.mark.parametrize("version", [2, 3, 4])  # type: ignore[misc]
def test_multi_query_state_preserves_strides_and_storage(version: int) -> None:
    out = torch.randn(7, 8, 128)[..., ::2]
    lse = torch.randn(7, 8).T
    state = normalize_flash_attn_state(
        out, lse, fa_version=version, num_kv_heads=4, max_query_len=7
    )
    assert state.out is out
    assert state.lse is lse


@pytest.mark.parametrize("version", [2, 3, 4])  # type: ignore[misc]
def test_single_query_modern_fa_keeps_canonical_head_order(version: int) -> None:
    out = torch.randn(3, 8, 64)
    lse = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    state = normalize_flash_attn_state(
        out, lse, fa_version=version, num_kv_heads=4, max_query_len=1, paged_kv=True
    )
    assert state.lse is lse


@pytest.mark.parametrize("version", [2, 3, 4])  # type: ignore[misc]
def test_empty_state_sentinels_do_not_require_device_work(version: int) -> None:
    out = torch.full((3, 8, 64), float("nan"))
    lse = torch.full((8, 3), -float("inf"))
    lse[:, 0] = float("inf")
    state = normalize_flash_attn_state(
        out, lse, fa_version=version, num_kv_heads=4, max_query_len=3
    )
    assert state.out is out
    assert state.lse is lse


@pytest.mark.parametrize("version", [2, 3, 4])  # type: ignore[misc]
def test_rejects_incompatible_lse_shape_and_dtype(version: int) -> None:
    out = torch.randn(3, 8, 64)
    for lse in (torch.randn(3, 8), torch.randn(8, 3, dtype=torch.float16)):
        with pytest.raises(ValueError, match="LSE must be FP32"):
            normalize_flash_attn_state(
                out, lse, fa_version=version, num_kv_heads=4, max_query_len=3
            )


def test_unknown_backend_requires_an_explicit_adapter() -> None:
    with pytest.raises(ValueError, match="Unsupported FlashAttention state version"):
        normalize_flash_attn_state(
            torch.randn(3, 8, 64),
            torch.randn(8, 3),
            fa_version=5,
            num_kv_heads=4,
            max_query_len=3,
        )
