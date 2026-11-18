# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
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
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.npu_test

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
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView  # noqa: E402
from vllm_gr.v1.beam.npu_execution_batch import NPUBeamExecutionBatch  # noqa: E402

DEVICE = torch.device("npu")
DTYPE = torch.bfloat16

NUM_HEADS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 128
MAX_DECODE_STEPS = 8
NUM_LAYERS = 1


@pytest.mark.parametrize(  # type: ignore[misc]
    ("case", "expected"),
    [
        ("eligible", None),
        ("not_beam_decode", "not_beam_decode"),
        ("missing_beam_session", "missing_beam_session"),
        ("inactive_beam_session", "inactive_beam_session"),
        ("mixed_std_beam_batch", "mixed_std_beam_batch"),
        ("missing_prefix_indices", "missing_prefix_indices"),
        ("sliding_window", "sliding_window"),
        ("logits_soft_cap", "logits_soft_cap"),
        ("attention_sinks", "attention_sinks"),
        ("alibi", "alibi"),
        ("missing_beam_pool", "missing_beam_pool"),
    ],
)
def test_xllm_cascade_fallback_reason(case: str, expected: str | None) -> None:
    metadata = SimpleNamespace(
        is_beam_decode=True,
        beam_session=SimpleNamespace(is_active=True),
        std_indices=None,
        prefix_indices=object(),
    )
    kwargs: dict[str, Any] = {
        "sliding_window_size": None,
        "logits_soft_cap": 0.0,
        "s_aux": None,
        "alibi_slopes": None,
        "beam_pool": object(),
    }
    if case == "not_beam_decode":
        metadata.is_beam_decode = False
    elif case == "missing_beam_session":
        metadata.beam_session = None
    elif case == "inactive_beam_session":
        metadata.beam_session.is_active = False
    elif case == "mixed_std_beam_batch":
        metadata.std_indices = object()
    elif case == "missing_prefix_indices":
        metadata.prefix_indices = None
    elif case == "sliding_window":
        kwargs["sliding_window_size"] = (-1, 0)
    elif case == "logits_soft_cap":
        kwargs["logits_soft_cap"] = 1.0
    elif case == "attention_sinks":
        kwargs["s_aux"] = object()
    elif case == "alibi":
        kwargs["alibi_slopes"] = object()
    elif case == "missing_beam_pool":
        kwargs["beam_pool"] = None

    assert beam_attn_npu_xllm._cascade_fallback_reason(metadata, **kwargs) == expected


def _build_incremental_metadata(
    *,
    beam_width: int,
    prefix_len: int,
    decode_step: int,
    session: BeamSearchSession | NPUBeamExecutionBatch,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    std_req: tuple[int, int, int] | None = None,
    request_ids: tuple[str, ...] | None = None,
    num_query_heads: int = NUM_HEADS,
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
    # Requests occupy consecutive w-row groups; each beam contributes 1.
    if std_req is None:
        query_start_loc_cpu[:] = torch.arange(num_reqs + 1) * beam_width
        num_actual_tokens = num_reqs * beam_width
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

    req_ids = request_ids or tuple(f"req_{i}" for i in range(num_reqs))
    beam_req_ids = req_ids if std_req is None else ("req_0",)
    beam_data: dict[str, object] = {
        req_id: {
            "is_beam_decode": True,
            "prefix_len": prefix_len,
            "beam_width": beam_width,
            "beam_decode_steps": decode_step + 1,
            "cache_len": prefix_len // BLOCK_SIZE * BLOCK_SIZE,
            "decode_step": decode_step,
        }
        for req_id in beam_req_ids
    }
    beam_info: dict[str, object] = {
        "beam_data": beam_data,
        "req_ids": list(req_ids),
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
        builder.num_heads_q = num_query_heads
        builder.num_kv_heads = NUM_KV_HEADS
        # The builder was created via __new__ (no __init__ provider lookup), so
        # supply the profile the active Ascend backend would have returned.
        from vllm_gr.platforms.ascend_services import AscendBeamMetadataProfile

        builder._profile = AscendBeamMetadataProfile()
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


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("decode_step", [0, 2])  # type: ignore[misc]
@pytest.mark.parametrize("num_query_heads", [NUM_HEADS, 2 * NUM_KV_HEADS])  # type: ignore[misc]
def test_cascade_execution_batch_matches_individual_requests(
    batch_size: int, decode_step: int, num_query_heads: int
) -> None:
    """The production batch facade must read all B*W request-local histories.

    Compare the flattened execution with independent single-request Cascade
    calls and a dense FP32 reference for both MHA and GQA. Distinct queries,
    prefix blocks, and suffix values make missing or cross-request KV rows
    observable. The dense comparison uses the existing BF16 attention
    tolerance, since FIA computes prefix and suffix separately before their
    LSE-weighted merge.
    """
    torch.manual_seed(1000 + batch_size * 10 + decode_step)
    beam_width, prefix_len = 32, 256
    scale = 1.0 / math.sqrt(HEAD_DIM)
    steps = decode_step + 1
    request_ids = tuple(f"cascade_req_{index}" for index in range(batch_size))
    context = BeamAttentionContext(
        max_batch=batch_size,
        max_beams=beam_width,
        num_kv_heads=NUM_KV_HEADS,
        max_decode_steps=MAX_DECODE_STEPS,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
        device=DEVICE,
        dtype=DTYPE,
    )
    sessions = []
    for request_id in request_ids:
        slot = context.registry.allocate(request_id, beam_width=beam_width, prefix_len=prefix_len)
        session = BeamSearchSession(session_id=request_id, slot=slot, context=context)
        session.begin_session(beam_width=beam_width, prefix_len=prefix_len)
        session._state.decode_step = decode_step
        sessions.append(session)
    execution = BeamBatchExecutionView(
        session_id="cascade_batch",
        dispatch_id=0,
        stage_index=decode_step + 1,
        kind="decode",
        decode_step=decode_step,
        owner_generation=0,
        request_ids=request_ids,
        num_prompt_tokens_by_request=(prefix_len,) * batch_size,
        beam_width=beam_width,
        produces_output=True,
    )
    batch = NPUBeamExecutionBatch(execution, sessions)
    try:
        blocks_per_req = prefix_len // BLOCK_SIZE
        cache_shape = (
            batch_size * blocks_per_req,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
        )
        key_cache_cpu = torch.randn(cache_shape, dtype=DTYPE) * 0.5
        value_cache_cpu = torch.randn(cache_shape, dtype=DTYPE) * 0.5
        history_shape = (batch_size, beam_width, NUM_KV_HEADS, steps, HEAD_DIM)
        history_k_cpu = torch.randn(history_shape, dtype=DTYPE) * 0.5
        history_v_cpu = torch.randn(history_shape, dtype=DTYPE) * 0.5
        for index in range(batch_size):
            block_start = index * blocks_per_req
            value_cache_cpu[block_start : block_start + blocks_per_req] += index * 0.5
            history_v_cpu[index] += index + 1
        key_cache = key_cache_cpu.to(DEVICE)
        value_cache = value_cache_cpu.to(DEVICE)
        source_pool = context.pools[0]
        source_pool.unshared_key[:, :, :, :steps, :] = history_k_cpu.to(DEVICE)
        source_pool.unshared_value[:, :, :, :steps, :] = history_v_cpu.to(DEVICE)
        block_tables = torch.arange(batch_size * blocks_per_req, dtype=torch.int32).reshape(
            batch_size, blocks_per_req
        )
        query_cpu = (
            torch.randn(batch_size * beam_width, num_query_heads, HEAD_DIM, dtype=DTYPE) * 0.5
        )
        query = query_cpu.to(DEVICE)
        metadata = _build_incremental_metadata(
            beam_width=beam_width,
            prefix_len=prefix_len,
            decode_step=decode_step,
            session=batch,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            request_ids=request_ids,
            num_query_heads=num_query_heads,
        )
        assert metadata.beam_session is batch
        assert metadata.num_prefix_indices == batch_size * beam_width
        assert metadata.suffix_cu_seqlens_kv_cpu is not None
        assert metadata.suffix_cu_seqlens_kv_cpu[-1] == batch_size * beam_width * steps
        batch_output = _run_mode(
            beam_attn_npu, metadata, query, key_cache, value_cache, batch.pools[0], scale
        )
        for index, session in enumerate(sessions):
            row_start = index * beam_width
            query_slice = query[row_start : row_start + beam_width]
            single_metadata = _build_incremental_metadata(
                beam_width=beam_width,
                prefix_len=prefix_len,
                decode_step=decode_step,
                session=session,
                query=query_slice,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=block_tables[index : index + 1],
                request_ids=(session.session_id,),
                num_query_heads=num_query_heads,
            )
            individual_output = _run_mode(
                beam_attn_npu,
                single_metadata,
                query_slice,
                key_cache,
                value_cache,
                source_pool,
                scale,
            )
            actual = batch_output[row_start : row_start + beam_width].float().cpu()
            torch.testing.assert_close(
                actual, individual_output.float().cpu(), rtol=2e-2, atol=2e-2
            )

            blocks = block_tables[index].long()
            prefix_k = key_cache_cpu[blocks].reshape(prefix_len, NUM_KV_HEADS, HEAD_DIM)
            prefix_v = value_cache_cpu[blocks].reshape(prefix_len, NUM_KV_HEADS, HEAD_DIM)
            full_k = torch.cat(
                [
                    prefix_k.permute(1, 0, 2).unsqueeze(0).expand(beam_width, -1, -1, -1),
                    history_k_cpu[index],
                ],
                dim=2,
            ).float()
            full_v = torch.cat(
                [
                    prefix_v.permute(1, 0, 2).unsqueeze(0).expand(beam_width, -1, -1, -1),
                    history_v_cpu[index],
                ],
                dim=2,
            ).float()
            heads_per_kv_head = num_query_heads // NUM_KV_HEADS
            full_k = full_k.repeat_interleave(heads_per_kv_head, dim=1)
            full_v = full_v.repeat_interleave(heads_per_kv_head, dim=1)
            q = query_cpu[row_start : row_start + beam_width].float()
            logits = torch.einsum("whd,whld->whl", q, full_k) * scale
            expected = torch.einsum("whl,whld->whd", logits.softmax(dim=-1), full_v)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    finally:
        batch.end_session()


@pytest.mark.parametrize("cache_layout", ["contiguous", "interleaved"])  # type: ignore[misc]
@pytest.mark.parametrize("beam_width", [2, 4])  # type: ignore[misc]
@pytest.mark.parametrize("decode_step", [0, 1, 3, 7])  # type: ignore[misc]
def test_xllm_incremental_matches_cascade(
    beam_width: int, decode_step: int, cache_layout: str
) -> None:
    """x_attention fused path vs cascade incremental path, allclose.

    ``interleaved`` mirrors the vllm-ascend 0.30 pool allocation (K/V
    interleaved per block, per-layer views non-contiguous), exercising the
    pool-base gather in the xllm shared-prefix path.
    """
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
    key_cache_cpu = (
        torch.randn(blocks_per_req, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE) * scale
    )
    value_cache_cpu = (
        torch.randn(blocks_per_req, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE) * scale
    )
    if cache_layout == "interleaved":
        # vllm-ascend 0.30 layout: one [N, 2, block, H, D] allocation with
        # the per-layer K/V views as its strided halves.
        pool_kv = torch.zeros(blocks_per_req, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
        pool_kv[:, 0] = key_cache_cpu
        pool_kv[:, 1] = value_cache_cpu
        pool_kv = pool_kv.to(DEVICE)
        key_cache, value_cache = pool_kv[:, 0], pool_kv[:, 1]
        assert not key_cache.is_contiguous()
    else:
        key_cache = key_cache_cpu.to(DEVICE)
        value_cache = value_cache_cpu.to(DEVICE)
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


def test_gather_shared_blocks_bounds_fixed_width_table() -> None:
    """The eager gather copies only the block-table columns a prefix uses.

    The NPU metadata builder allocates the prefix block table at
    ``max_model_len // block_size`` columns, so a short prefix leaves most
    columns as padding. The gather must slice them off so the copy scales
    with the real prefix instead of ``max_model_len``.
    """
    num_blocks, block, heads, head_dim = 16, 4, 2, 8
    pool = torch.arange(
        num_blocks * 2 * block * heads * head_dim, dtype=torch.float32, device=DEVICE
    ).reshape(num_blocks, 2, block, heads, head_dim)
    key_cache, value_cache = pool[:, 0], pool[:, 1]

    # One live block per request, but a fixed-width table of 8 columns.
    table = torch.zeros(2, 8, dtype=torch.int32, device=DEVICE)
    table[0, 0] = 0
    table[1, 0] = 5

    unbounded_key, _, unbounded_table = beam_attn_npu_xllm._gather_shared_blocks(
        key_cache, value_cache, table, 0
    )
    bounded_key, bounded_value, bounded_table = beam_attn_npu_xllm._gather_shared_blocks(
        key_cache, value_cache, table, 1
    )

    # Only the referenced blocks are copied when the bound is applied.
    assert bounded_key.shape[0] == table.shape[0] * 1
    assert unbounded_key.shape[0] == table.shape[0] * table.shape[1]
    assert tuple(bounded_table.shape) == (table.shape[0], 1)

    # The remapped table must still point at the same real blocks.
    assert bounded_table[:, 0].tolist() == [0, 1]
    assert bounded_key[0].equal(pool[0, 0])
    assert bounded_key[1].equal(pool[5, 0])
    assert bounded_value[0].equal(pool[0, 1])
    assert bounded_value[1].equal(pool[5, 1])

    # A bound at or above the table width is a no-op, not a truncation.
    full_key, _, full_table = beam_attn_npu_xllm._gather_shared_blocks(
        key_cache, value_cache, table, table.shape[1]
    )
    assert full_key.shape[0] == table.shape[0] * table.shape[1]
    assert tuple(full_table.shape) == tuple(table.shape)
