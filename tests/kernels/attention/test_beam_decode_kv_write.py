# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disallow-untyped-decorators=False
"""FP32 reference and graph replay checks for the CUDA Beam KV writer."""

import pytest
import torch

from vllm_gr.v1.attention.backends.beam_graph_kernels import cache_unshared_kv_graph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

SLOT = 1
STEPS = 3


def _reference_write(
    pool: torch.Tensor,
    source: torch.Tensor,
    rows: torch.Tensor,
    active: torch.Tensor,
    step: int,
) -> None:
    """Update a FP32 pool using Torch indexing, independent of the writer."""
    if 0 <= step < pool.shape[3]:
        selected = source.float().index_select(0, rows.long())
        destination = pool[SLOT, : rows.numel(), :, step, :]
        destination[active] = selected[active]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mapped", [False, True], ids=["identity", "mapped"])
def test_decode_kv_write_matches_fp32_reference(dtype: torch.dtype, mapped: bool) -> None:
    width, heads, dim = 128, 8, 128
    input_rows = width + 7 if mapped else width
    generator = torch.Generator(device="cuda").manual_seed(1234)
    # Qwen's K/V are views split from the same projection, with wider row
    # strides than a compact [beam, head, dim] allocation.
    qkv = torch.randn(input_rows, heads * 4, dim, device="cuda", dtype=dtype, generator=generator)
    key = qkv[:, 2 * heads : 3 * heads]
    value = qkv[:, 3 * heads :]
    beams = torch.arange(width, device="cuda")
    rows = (7 * beams + 3) % input_rows if mapped else beams
    if mapped:
        rows[1] = rows[0]
    active = beams % 3 != 2 if mapped else torch.ones_like(beams, dtype=torch.bool)
    shape = (3, width + 3, heads, STEPS, dim)
    pool_key = torch.full(shape, -17, device="cuda", dtype=dtype)
    pool_value = torch.full_like(pool_key, 31)
    reference_key = pool_key.float().clone()
    reference_value = pool_value.float().clone()
    step = 1
    step_buffer = torch.tensor([step], device="cuda", dtype=torch.int32)

    cache_unshared_kv_graph(
        pool_key,
        pool_value,
        key,
        value,
        SLOT,
        step_buffer,
        active if mapped else None,
        row_indices=rows if mapped else None,
    )
    _reference_write(reference_key, key, rows, active, step)
    _reference_write(reference_value, value, rows, active, step)
    torch.testing.assert_close(pool_key.float(), reference_key, atol=0, rtol=0)
    torch.testing.assert_close(pool_value.float(), reference_value, atol=0, rtol=0)


def test_decode_kv_write_graph_replays_current_inputs_and_metadata() -> None:
    width, input_rows, heads, dim = 7, 11, 3, 64
    generator = torch.Generator(device="cuda").manual_seed(5678)
    key = torch.randn(
        input_rows, heads, 2 * dim, device="cuda", dtype=torch.bfloat16, generator=generator
    )[..., ::2]
    value = torch.randn(
        input_rows, heads, 3 * dim, device="cuda", dtype=torch.bfloat16, generator=generator
    )[..., ::3]
    shape = (3, width + 2, heads, STEPS, dim)
    pool_key = torch.full(shape, -17, device="cuda", dtype=key.dtype)
    pool_value = torch.full_like(pool_key, 31)
    reference_key = pool_key.float()
    reference_value = pool_value.float()
    beams = torch.arange(width, device="cuda", dtype=torch.int32)
    rows = beams.clone()
    active = torch.ones(width, device="cuda", dtype=torch.bool)
    step_buffer = torch.full((1,), -1, device="cuda", dtype=torch.int32)

    def write() -> None:
        cache_unshared_kv_graph(
            pool_key, pool_value, key, value, SLOT, step_buffer, active, row_indices=rows
        )

    write()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()
    for replay, step in enumerate((0, 2, -1, STEPS, 0)):
        rows.copy_((3 * beams + replay) % input_rows)
        rows[1] = rows[0]
        active.copy_((beams + replay) % 3 != 2)
        step_buffer.fill_(step)
        key.add_(0.25)
        value.sub_(0.5)
        graph.replay()
        _reference_write(reference_key, key, rows, active, step)
        _reference_write(reference_value, value, rows, active, step)
        torch.testing.assert_close(pool_key.float(), reference_key, atol=0, rtol=0)
        torch.testing.assert_close(pool_value.float(), reference_value, atol=0, rtol=0)
