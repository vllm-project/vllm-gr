# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU precision guard tests for the fused ``x_attention`` op.

Derived from xllm-ops' ``test_x_attention.py``.  These tests exercise the
combination vllm-gr-dev targets for the NPU decode path: a paged shared KV
cache (with ``shared_block_tables``) plus a contiguous unshared beam KV pool
(``unshared_block_tables=None``), GQA query heads, and per-request variable
shared prefix lengths.
"""

from __future__ import annotations

import math

import pytest
import torch

# mypy: disable-error-code="misc"


pytestmark = pytest.mark.npu


try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"torch-npu is not installed: {exc}", allow_module_level=True)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is not available", allow_module_level=True)

try:
    import custom_ops_lib as custom_ops
except ImportError as exc:
    pytest.skip(f"custom_ops_lib is not installed: {exc}", allow_module_level=True)


DEVICE = torch.device("npu")

# OneRec-shaped attention: GQA 32 query heads over 8 KV heads.
NUM_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 128
MAX_DECODE_STEPS = 3


def _reference_x_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    block_tables: torch.Tensor,
    kvlen_list: list[int],
    decode_step: int,
) -> torch.Tensor:
    """CPU fp32 reference for shared-paged + unshared-contiguous attention.

    Vectorized following the xllm-ops golden structure: the shared part is
    computed once per request for every beam query (the shared prefix is
    identical across beams), the unshared part is computed per beam row, and
    both are merged with a single softmax over the concatenated key axis.
    """
    request_num = block_tables.shape[0]
    beam_size = query.shape[0] // request_num
    group = NUM_HEADS // KV_HEADS
    scale = 1.0 / math.sqrt(HEAD_DIM)
    out = torch.zeros_like(query, dtype=torch.float32)

    for request in range(request_num):
        rows = slice(request * beam_size, (request + 1) * beam_size)
        length = kvlen_list[request]
        blocks = block_tables[request].tolist()
        # shared prefix: [kv_heads, L, D] / [kv_heads, D, L]
        shared_k = (key_cache[blocks].reshape(-1, KV_HEADS, HEAD_DIM)[:length].float()).permute(
            1, 0, 2
        )
        shared_v = (value_cache[blocks].reshape(-1, KV_HEADS, HEAD_DIM)[:length].float()).permute(
            1, 2, 0
        )
        # beam queries of this request: [beam, kv_heads, group, D]
        q = query[rows].float().reshape(beam_size, KV_HEADS, group, HEAD_DIM)
        # unshared pool rows: [beam, kv_heads, T, D]
        unshared_k = unshared_key[rows, :, :decode_step, :].float()
        unshared_v = unshared_value[rows, :, :decode_step, :].float()

        sim_shared = torch.einsum("bhgd,hld->bhgl", q, shared_k) * scale
        sim_unshared = torch.einsum("bhgd,bhtd->bhgt", q, unshared_k) * scale
        probs = torch.softmax(torch.cat([sim_shared, sim_unshared], dim=-1), dim=-1)

        out_shared = torch.einsum("bhgl,hdl->bhgd", probs[..., :length], shared_v)
        out_unshared = torch.einsum("bhgt,bhtd->bhgd", probs[..., length:], unshared_v)
        out[rows] = (out_shared + out_unshared).reshape(beam_size, NUM_HEADS, HEAD_DIM)
    return out


def _run_x_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    block_tables: torch.Tensor,
    kvlen_list: list[int],
    decode_step: int,
    scale_value: float,
) -> torch.Tensor:
    return custom_ops.x_attention(
        query.to(DEVICE),
        key_cache.to(DEVICE),
        value_cache.to(DEVICE),
        unshared_key.to(DEVICE),
        unshared_value.to(DEVICE),
        block_tables.to(DEVICE),
        None,  # unshared contiguous pool -> no block table
        torch.tensor(kvlen_list, dtype=torch.int32, device=DEVICE),
        torch.tensor([decode_step], dtype=torch.int32, device=DEVICE),
        scale_value,
    )


@pytest.mark.parametrize(
    "request_num,beam_size,kvlen_list,decode_step",
    [
        (1, 128, [128], 1),
        (1, 128, [256], 2),
        (2, 128, [256, 128], 2),
        (2, 128, [128, 256], 3),
    ],
)
def test_x_attention_npu_matches_reference(
    request_num: int,
    beam_size: int,
    kvlen_list: list[int],
    decode_step: int,
) -> None:
    """GQA shared-paged + unshared-contiguous attention vs CPU fp32 golden."""
    torch.manual_seed(1234)
    dtype = torch.bfloat16

    max_blocks_per_seq = max((length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in kvlen_list)
    num_blocks = request_num * max_blocks_per_seq
    block_tables = torch.tensor(
        [
            [max_blocks_per_seq * request + block for block in range(max_blocks_per_seq)]
            for request in range(request_num)
        ],
        dtype=torch.int32,
    )

    query = torch.empty((request_num * beam_size, NUM_HEADS, HEAD_DIM), dtype=dtype).uniform_(-1, 1)
    key_cache = torch.empty((num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM), dtype=dtype).uniform_(
        -1, 1
    )
    value_cache = torch.empty((num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM), dtype=dtype).uniform_(
        -1, 1
    )
    unshared_key = torch.empty(
        (request_num * beam_size, KV_HEADS, MAX_DECODE_STEPS, HEAD_DIM), dtype=dtype
    ).uniform_(-1, 1)
    unshared_value = torch.empty_like(unshared_key).uniform_(-1, 1)

    expected = _reference_x_attention(
        query,
        key_cache,
        value_cache,
        unshared_key,
        unshared_value,
        block_tables,
        kvlen_list,
        decode_step,
    )

    actual = (
        _run_x_attention(
            query,
            key_cache,
            value_cache,
            unshared_key,
            unshared_value,
            block_tables,
            kvlen_list,
            decode_step,
            0.0,  # <= 0 -> kernel derives 1/sqrt(head_dim)
        )
        .cpu()
        .float()
    )

    assert actual.shape == (request_num * beam_size, NUM_HEADS, HEAD_DIM)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_x_attention_npu_explicit_scale_matches_default() -> None:
    """An explicit 1/sqrt(head_dim) scale must match the auto-scale path."""
    torch.manual_seed(4321)
    dtype = torch.bfloat16
    beam_size, decode_step = 128, 2
    kvlen_list = [256]
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)

    query = torch.empty((beam_size, NUM_HEADS, HEAD_DIM), dtype=dtype).uniform_(-1, 1)
    key_cache = torch.empty((2, BLOCK_SIZE, KV_HEADS, HEAD_DIM), dtype=dtype).uniform_(-1, 1)
    value_cache = torch.empty_like(key_cache).uniform_(-1, 1)
    unshared_key = torch.empty(
        (beam_size, KV_HEADS, MAX_DECODE_STEPS, HEAD_DIM), dtype=dtype
    ).uniform_(-1, 1)
    unshared_value = torch.empty_like(unshared_key).uniform_(-1, 1)
    common = (
        query,
        key_cache,
        value_cache,
        unshared_key,
        unshared_value,
        block_tables,
        kvlen_list,
        decode_step,
    )

    auto = _run_x_attention(*common, 0.0).cpu()
    explicit = _run_x_attention(*common, 1.0 / math.sqrt(HEAD_DIM)).cpu()

    torch.testing.assert_close(auto, explicit, rtol=0, atol=0)
