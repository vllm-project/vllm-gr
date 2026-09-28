# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests: xllm fused decode vs the cascade incremental path on NPU.

Drives the real ``BeamAttentionMetadataBuilder`` with a live
``BeamSearchSession`` (so the metadata carries the incremental-decode fields
``is_beam_decode`` / ``beam_session`` / ``prefix_indices``) and compares
``beam_attn_npu_xllm.cascade_attention`` against
``beam_attn_npu.cascade_attention`` on identical pool and query state.

Both paths must agree because they compute the same quantity -- attention of
each beam query over ``[shared paged prefix || own unshared pool history]`` --
with the same bottom-right causal convention on the unshared axis.

Also covers:
  * ``beam_attn_npu_xllm.cache_unshared_kv`` vs the reference pool write.
  * mixed std+beam batches: the xllm module must refuse them loudly instead
    of silently dropping the std rows.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.npu_integration

try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"torch-npu is not installed: {exc}", allow_module_level=True)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is not available", allow_module_level=True)

from vllm_gr.ops.npu import beam_ops  # noqa: E402
from vllm_gr.v1.attention.backends import (  # noqa: E402
    beam_attn_npu,
    beam_attn_npu_xllm,
)
from vllm_gr.v1.attention.backends.beam_attn_metadata import (  # noqa: E402
    BeamAttentionMetadata,
    BeamAttentionMetadataBuilder,
)
from vllm_gr.v1.beam import (  # noqa: E402
    BeamAttentionContext,
    BeamAttentionPool,
    BeamSearchSession,
)
from vllm_gr.v1.beam.context import BEAM_DATA_VAR  # noqa: E402

DEVICE = torch.device("npu")
DTYPE = torch.bfloat16

NUM_HEADS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 128
MAX_DECODE_STEPS = 8
NUM_LAYERS = 1


def _build_incremental_metadata(
    *,
    beam_width: int,
    prefix_len: int,
    decode_step: int,
    session: BeamSearchSession,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    std_req: tuple[int, int, int] | None = None,
) -> BeamAttentionMetadata:
    """Build real incremental-decode metadata through the real builder.

    Mirrors ``cascade_beam_attn_utils.build_beam_attn_metadata`` but wires a
    live ``BeamSearchSession`` into ``BEAM_DATA_VAR`` so the builder takes the
    incremental path (``is_incremental=True``).
    """
    beam_seq_len = prefix_len + beam_width * (decode_step + 1)
    num_reqs = block_tables.shape[0] + (1 if std_req else 0)

    # std tokens are not part of the beam query tensor here
    query_start_loc_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    # Beam request occupies the first w query rows; each beam contributes 1.
    if std_req is None:
        query_start_loc_cpu[1] = beam_width
        query_start_loc_cpu[2:] = beam_width
        num_actual_tokens = beam_width
    else:
        std_idx, std_q_len, _ = std_req
        # std rows come first (std_indices gather), beam rows after.
        query_start_loc_cpu[1 : std_idx + 1] = std_q_len
        query_start_loc_cpu[std_idx + 1 :] = std_q_len + beam_width
        num_actual_tokens = std_q_len + beam_width

    seq_lens = torch.tensor([beam_seq_len] * num_reqs, dtype=torch.int32, device=DEVICE)

    class _Common:
        num_reqs: int
        query_start_loc_cpu: torch.Tensor
        query_start_loc: torch.Tensor
        seq_lens: torch.Tensor
        block_table_tensor: torch.Tensor
        num_actual_tokens: int
        max_query_len: int
        max_seq_len: int
        causal: bool
        slot_mapping: torch.Tensor

    common = _Common()
    common.num_reqs = num_reqs
    common.query_start_loc_cpu = query_start_loc_cpu
    common.query_start_loc = query_start_loc_cpu.to(DEVICE)
    common.seq_lens = seq_lens
    common.block_table_tensor = block_tables.to(DEVICE)
    common.num_actual_tokens = num_actual_tokens
    common.max_query_len = max(beam_width, 1)
    common.max_seq_len = beam_seq_len
    common.causal = True
    common.slot_mapping = torch.zeros(num_actual_tokens, dtype=torch.long, device=DEVICE)

    beam_data: dict[str, object] = {
        "req_0": {
            "is_beam_decode": True,
            "prefix_len": prefix_len,
            "beam_width": beam_width,
            "beam_decode_steps": decode_step + 1,
            "cache_len": prefix_len // BLOCK_SIZE * BLOCK_SIZE,
            "decode_step": decode_step,
        }
    }
    beam_info: dict[str, object] = {
        "beam_data": beam_data,
        "req_ids": [f"req_{i}" for i in range(num_reqs)],
        "beam_session": session,
    }

    token = BEAM_DATA_VAR.set(beam_info)
    try:
        builder = BeamAttentionMetadataBuilder.__new__(BeamAttentionMetadataBuilder)
        builder.device = DEVICE
        builder.block_size = BLOCK_SIZE
        builder.aot_schedule = False
        builder.aot_sliding_window = None
        builder.use_full_cuda_graph = False
        builder.max_cudagraph_size = None
        builder.max_num_splits = 0
        builder.num_heads_q = NUM_HEADS
        builder.num_kv_heads = NUM_KV_HEADS
        builder._is_cuda = False
        builder._is_npu = True
        # The builder was created via __new__ (no __init__ buffer allocation);
        # build() reads these attributes, so mirror the non-graph defaults.
        builder._fixed_buffers = False
        builder.npu_causal_atten_mask = (
            torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(DEVICE)
        )
        return builder.build(
            common_prefix_len=prefix_len,
            common_attn_metadata=common,
            fast_build=False,
        )
    finally:
        BEAM_DATA_VAR.reset(token)


def _run_mode(
    module: object,
    metadata: BeamAttentionMetadata,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    pool: BeamAttentionPool,
    scale: float,
) -> torch.Tensor:
    output = torch.zeros_like(query)
    module.cascade_attention(  # type: ignore[attr-defined]
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        attn_metadata=metadata,
        sliding_window_size=None,
        scale=scale,
        logits_soft_cap=0.0,
        vllm_flash_attn_version=0,
        s_aux=None,
        alibi_slopes=None,
        beam_pool=pool,
    )
    torch.npu.synchronize()
    return output


@pytest.mark.parametrize("beam_width", [2, 4])  # type: ignore[misc]
@pytest.mark.parametrize("decode_step", [0, 1, 3, 7])  # type: ignore[misc]
def test_xllm_incremental_matches_cascade(beam_width: int, decode_step: int) -> None:
    """x_attention fused path vs cascade incremental path, allclose."""
    torch.manual_seed(beam_width * 100 + decode_step)
    prefix_len = 256  # 2 full blocks; no remainder-block std spill
    scale = 1.0 / math.sqrt(HEAD_DIM)

    context = BeamAttentionContext(
        max_batch=1,
        max_beams=beam_width,
        num_kv_heads=NUM_KV_HEADS,
        max_decode_steps=MAX_DECODE_STEPS,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
        device=DEVICE,
        dtype=DTYPE,
    )
    slot = context.registry.allocate("test", beam_width=beam_width, prefix_len=prefix_len)
    session = BeamSearchSession(session_id="test", slot=slot, context=context)
    session.begin_session(beam_width=beam_width, prefix_len=prefix_len)
    # advance to the requested step
    for _ in range(decode_step):
        session._state.decode_step += 1
    # Mirror the runner refresh (model_runner_common.py): the xllm backend
    # reads the context's fixed-address device buffer, which starts at zero —
    # without this fill the fused path would attend over an empty unshared
    # history while the cascade path reads the host-side step.
    context.decode_step_1based_buf.fill_(decode_step + 1)
    pool = session.pools[0]

    # Fill pool history for steps [0, decode_step] with distinct values.
    steps = decode_step + 1
    hist_k = torch.randn(beam_width, NUM_KV_HEADS, steps, HEAD_DIM, dtype=DTYPE) * scale
    hist_v = torch.randn(beam_width, NUM_KV_HEADS, steps, HEAD_DIM, dtype=DTYPE) * scale
    pool.unshared_key[slot, :beam_width, :, :steps, :] = hist_k.to(DEVICE)
    pool.unshared_value[slot, :beam_width, :, :steps, :] = hist_v.to(DEVICE)

    # Paged shared prefix KV.
    blocks_per_req = prefix_len // BLOCK_SIZE
    key_cache = (
        torch.randn(blocks_per_req, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE) * scale
    ).to(DEVICE)
    value_cache = (
        torch.randn(blocks_per_req, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE) * scale
    ).to(DEVICE)
    block_tables = torch.arange(blocks_per_req, dtype=torch.int32).unsqueeze(0)

    query = (torch.randn(beam_width, NUM_HEADS, HEAD_DIM, dtype=DTYPE) * scale).to(DEVICE)

    metadata = _build_incremental_metadata(
        beam_width=beam_width,
        prefix_len=prefix_len,
        decode_step=decode_step,
        session=session,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
    )
    assert metadata.is_beam_decode is True
    assert metadata.beam_session is session
    assert metadata.prefix_indices is not None
    assert metadata.std_indices is None

    cascade_out = _run_mode(beam_attn_npu, metadata, query, key_cache, value_cache, pool, scale)
    xllm_out = _run_mode(beam_attn_npu_xllm, metadata, query, key_cache, value_cache, pool, scale)

    assert xllm_out.shape == cascade_out.shape
    torch.testing.assert_close(xllm_out.float(), cascade_out.float(), rtol=2e-2, atol=2e-2)

    session.end_session()


@pytest.mark.parametrize("beam_width", [2, 4])  # type: ignore[misc]
def test_xllm_submax_width_matches_cascade(beam_width: int) -> None:
    """Parity when beam_width < configured max_beams (eager path).

    The runtime pool is sized by ``beam_max_width`` (an upper bound), so a
    request narrower than the maximum attends over ``pool[slot, :beam_width]``
    — a leading-dim slice of the contiguous pool, which stays contiguous
    (verified on device: kernel output identical to a packed ``.contiguous()``
    copy).  This test locks that contract down: the xllm fused path must read
    exactly the active rows and agree with the cascade path.  FULL-graph
    replay never runs for sub-max widths (the determine patch gates on exact
    width), so this is the eager path exercised end to end.
    """
    torch.manual_seed(500 + beam_width)
    max_beams = 8  # > beam_width: pool keeps max-width layout
    prefix_len = 256
    decode_step = 3
    scale = 1.0 / math.sqrt(HEAD_DIM)

    context = BeamAttentionContext(
        max_batch=1,
        max_beams=max_beams,
        num_kv_heads=NUM_KV_HEADS,
        max_decode_steps=MAX_DECODE_STEPS,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
        device=DEVICE,
        dtype=DTYPE,
    )
    slot = context.registry.allocate("submax", beam_width=beam_width, prefix_len=prefix_len)
    session = BeamSearchSession(session_id="submax", slot=slot, context=context)
    session.begin_session(beam_width=beam_width, prefix_len=prefix_len)
    for _ in range(decode_step):
        session._state.decode_step += 1
    context.decode_step_1based_buf.fill_(decode_step + 1)
    pool = session.pools[0]

    # Active rows carry distinct values; the tail rows [beam_width, max_beams)
    # carry DIFFERENT values — if the kernel misread the row mapping the two
    # backends would diverge.
    steps = decode_step + 1
    hist_k = torch.randn(beam_width, NUM_KV_HEADS, steps, HEAD_DIM, dtype=DTYPE) * scale
    hist_v = torch.randn(beam_width, NUM_KV_HEADS, steps, HEAD_DIM, dtype=DTYPE) * scale
    pool.unshared_key[slot, :beam_width, :, :steps, :] = hist_k.to(DEVICE)
    pool.unshared_value[slot, :beam_width, :, :steps, :] = hist_v.to(DEVICE)
    tail_k = torch.randn(max_beams - beam_width, NUM_KV_HEADS, steps, HEAD_DIM, dtype=DTYPE)
    pool.unshared_key[slot, beam_width:, :, :steps, :] = tail_k.to(DEVICE)

    blocks_per_req = prefix_len // BLOCK_SIZE
    key_cache = (
        torch.randn(blocks_per_req, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE) * scale
    ).to(DEVICE)
    value_cache = (
        torch.randn(blocks_per_req, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE) * scale
    ).to(DEVICE)
    block_tables = torch.arange(blocks_per_req, dtype=torch.int32).unsqueeze(0)
    query = (torch.randn(beam_width, NUM_HEADS, HEAD_DIM, dtype=DTYPE) * scale).to(DEVICE)

    metadata = _build_incremental_metadata(
        beam_width=beam_width,
        prefix_len=prefix_len,
        decode_step=decode_step,
        session=session,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
    )

    cascade_out = _run_mode(beam_attn_npu, metadata, query, key_cache, value_cache, pool, scale)
    xllm_out = _run_mode(beam_attn_npu_xllm, metadata, query, key_cache, value_cache, pool, scale)

    assert xllm_out.shape == cascade_out.shape
    torch.testing.assert_close(xllm_out.float(), cascade_out.float(), rtol=2e-2, atol=2e-2)

    session.end_session()


def test_xllm_cache_unshared_kv_matches_reference() -> None:
    """The flattened B*W attention view must preserve B request-local writes."""
    torch.manual_seed(99)
    batch_size, beam_width, max_steps = 2, 4, 8
    shape = (
        batch_size,
        beam_width,
        NUM_KV_HEADS,
        max_steps,
        HEAD_DIM,
    )
    pool = torch.zeros(shape, dtype=DTYPE, device=DEVICE)
    pool_v = torch.zeros_like(pool)
    pool_ref = torch.zeros_like(pool)
    pool_ref_v = torch.zeros_like(pool)

    rows = batch_size * beam_width
    curr_k = (torch.randn(rows, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE) * 0.1).to(DEVICE)
    curr_v = (torch.randn(rows, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE) * 0.1).to(DEVICE)
    decode_step = 3
    context = SimpleNamespace(
        xllm_block_table_buf=torch.arange(batch_size, dtype=torch.int32, device=DEVICE),
        decode_step_1based_buf=torch.tensor([decode_step + 1], dtype=torch.int32, device=DEVICE),
    )
    session = SimpleNamespace(
        context=context,
        request_width=beam_width,
        beam_width=beam_width,
    )

    beam_attn_npu_xllm.cache_unshared_kv(
        pool.view(1, rows, *pool.shape[2:]),
        pool_v.view(1, rows, *pool_v.shape[2:]),
        curr_k,
        curr_v,
        slot=0,
        decode_step=decode_step,
        beam_session=session,
    )
    pool_ref[:, :, :, decode_step : decode_step + 1, :] = curr_k.view(
        batch_size, beam_width, NUM_KV_HEADS, HEAD_DIM
    ).unsqueeze(3)
    pool_ref_v[:, :, :, decode_step : decode_step + 1, :] = curr_v.view(
        batch_size, beam_width, NUM_KV_HEADS, HEAD_DIM
    ).unsqueeze(3)

    torch.testing.assert_close(pool, pool_ref, rtol=0, atol=0)
    torch.testing.assert_close(pool_v, pool_ref_v, rtol=0, atol=0)

    pool2 = torch.zeros_like(pool)
    pool2_v = torch.zeros_like(pool)
    beam_ops.cache_unshared_kv(
        pool2,
        pool2_v,
        curr_k,
        curr_v,
        context.xllm_block_table_buf,
        context.decode_step_1based_buf,
    )
    torch.testing.assert_close(pool2, pool_ref, rtol=0, atol=0)
    torch.testing.assert_close(pool2_v, pool_ref_v, rtol=0, atol=0)


def test_xllm_mixed_std_beam_batch_falls_back_to_cascade() -> None:
    """Mixed std+beam batches must fall back to cascade and produce output.

    The fused x_attention kernel only encodes beam rows, so a mixed batch is
    delegated to beam_attn_npu.cascade_attention (std rows via paged
    attention, beam rows via the pool-slice incremental path).  This is the
    documented fallback (docs/quickstart-ascend.md): a std request sharing a
    batch with beam decode must succeed, not raise.
    """
    beam_width, prefix_len, decode_step = 2, 256, 1
    scale = 1.0 / math.sqrt(HEAD_DIM)

    context = BeamAttentionContext(
        max_batch=1,
        max_beams=beam_width,
        num_kv_heads=NUM_KV_HEADS,
        max_decode_steps=MAX_DECODE_STEPS,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
        device=DEVICE,
        dtype=DTYPE,
    )
    slot = context.registry.allocate("mixed", beam_width=beam_width, prefix_len=prefix_len)
    session = BeamSearchSession(session_id="mixed", slot=slot, context=context)
    session.begin_session(beam_width=beam_width, prefix_len=prefix_len)
    session._state.decode_step = decode_step
    context.decode_step_1based_buf.fill_(decode_step + 1)
    pool = session.pools[0]

    # Fill the pool history so the beam rows attend over real data.
    steps = decode_step + 1
    hist_k = torch.randn(beam_width, NUM_KV_HEADS, steps, HEAD_DIM, dtype=DTYPE) * scale
    hist_v = torch.randn(beam_width, NUM_KV_HEADS, steps, HEAD_DIM, dtype=DTYPE) * scale
    pool.unshared_key[slot, :beam_width, :, :steps, :] = hist_k.to(DEVICE)
    pool.unshared_value[slot, :beam_width, :, :steps, :] = hist_v.to(DEVICE)

    blocks_per_req = prefix_len // BLOCK_SIZE
    key_cache = torch.randn(
        blocks_per_req + 1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    value_cache = torch.randn_like(key_cache)
    # std request occupies block `blocks_per_req`
    block_tables = torch.arange(blocks_per_req + 1, dtype=torch.int32).unsqueeze(0)

    # 1 std token + w beam tokens
    query = (torch.randn(1 + beam_width, NUM_HEADS, HEAD_DIM, dtype=DTYPE) * scale).to(DEVICE)

    metadata = _build_incremental_metadata(
        beam_width=beam_width,
        prefix_len=prefix_len,
        decode_step=decode_step,
        session=session,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        std_req=(0, 1, prefix_len),
    )
    # Force a mixed batch: std rows exist alongside beam rows.  The std
    # branch needs the std routing fields the builder would have produced
    # (std_block_table / std_cu_seqlens_q_cpu / std_seqlens_kv_cpu).
    metadata.std_indices = torch.tensor([0], dtype=torch.long, device=DEVICE)
    metadata.num_std_indices = 1
    metadata.prefix_indices = torch.arange(1, 1 + beam_width, dtype=torch.long, device=DEVICE)
    metadata.num_prefix_indices = beam_width
    metadata.std_block_table = block_tables.to(DEVICE)
    metadata.std_cu_seqlens_q_cpu = [1]
    metadata.std_seqlens_kv_cpu = [prefix_len]

    xllm_out = _run_mode(beam_attn_npu_xllm, metadata, query, key_cache, value_cache, pool, scale)
    cascade_out = _run_mode(beam_attn_npu, metadata, query, key_cache, value_cache, pool, scale)

    # The delegated mixed batch must succeed and match the cascade module
    # bit-for-bit (same code path) — std rows and beam rows alike.
    assert xllm_out.shape == (1 + beam_width, NUM_HEADS, HEAD_DIM)
    torch.testing.assert_close(xllm_out.float(), cascade_out.float(), rtol=0, atol=0)
    # Sanity: real attention output, not zeros.
    assert xllm_out.float().abs().sum() > 0

    session.end_session()
