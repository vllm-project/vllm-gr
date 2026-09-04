# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""xLLM fused-operator execution primitives for the Beam Attention backend.

NPU-only execution module selected by ``_get_platform_execution_module`` when
``vllm_gr_npu_decode_execution == "xllm"``.  It replaces the cascade
per-operator path only for the incremental beam decode branch; every other
branch (std requests, prefill, suffix recompute) forwards to
``beam_attn_npu`` unchanged, and the CUDA path is never touched.

The incremental branch becomes a single fused ``x_attention`` call that
attends over the paged shared prefix and the contiguous unshared beam pool
in one kernel, replacing "two npu_fused_infer_attention_score_v2 calls +
pool->buffer copies + the merge operator chain".
"""

from __future__ import annotations

import math

import torch

from vllm_gr.ops.npu import beam_ops
from vllm_gr.v1.attention.backends import beam_attn_npu
from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
)

# Fail fast when the mode selects this module but the operator library is
# missing: without this check the failure would surface at the first beam
# decode step, deep inside model execution.
if not beam_ops.is_available():
    raise RuntimeError(
        "vllm_gr_npu_decode_execution='xllm' requires the xllm-ops library "
        f"({beam_ops.unavailable_reason()}); install it with "
        "tools/install_xllm_ops.sh or select 'cascade' explicitly via "
        "additional_config['vllm_gr_npu_decode_execution']='cascade'"
    )

# Forwarded primitives: std requests, prefill and cache writes behave
# exactly like the cascade module.
is_available = beam_attn_npu.is_available
get_flash_attn_version = beam_attn_npu.get_flash_attn_version
supports_sinks = beam_attn_npu.supports_sinks
reshape_and_cache = beam_attn_npu.reshape_and_cache


def prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: BeamAttentionMetadata,
    sliding_window_size: list | tuple | None,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
) -> None:
    """Validation guard: the prefill path must never execute through xllm.

    The xllm module is decode-only by design; BeamAttentionImpl routes the
    prefill branch to ``beam_attn_npu`` directly on NPU. Reaching this
    function means a routing regression, so fail loudly.
    """
    raise RuntimeError(
        "prefill attention reached the xllm execution module — prefill must "
        "route to the cascade module (beam_attn_npu), not the decode-only "
        "xllm path"
    )


def cache_unshared_kv(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    slot: int,
    decode_step: int,
    beam_session: object = None,
) -> None:
    """Fused pool write: slot -> block_table, 0-based step -> 1-based.

    When ``beam_session`` carries the context's fixed-address device buffers
    (``xllm_block_table_buf`` / ``decode_step_1based_buf``), those are passed
    to the kernel directly instead of allocating fresh tensors — required for
    ACL graph capture/replay (address stability + fresh values at replay).
    """
    context = getattr(beam_session, "context", None) if beam_session is not None else None
    block_table = getattr(context, "xllm_block_table_buf", None)
    step = getattr(context, "decode_step_1based_buf", None)
    if block_table is None or step is None:
        # Eager fallback: per-call tensors (previous behavior).
        block_table = torch.tensor([slot], dtype=torch.int32, device=unshared_key.device)
        step = decode_step + 1
    beam_ops.cache_unshared_kv(
        unshared_key,
        unshared_value,
        curr_key,
        curr_value,
        block_table,
        step,
    )


def _xllm_incremental_decode(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata,
    beam_session,
    beam_pool,
    scale: float,
) -> None:
    """One fused x_attention call for the incremental beam decode branch.

    Graph-friendly (ACL graph capture/replay): every runtime-varying kernel
    input is either a fixed-address buffer refreshed outside the graph
    (prefix_indices / prefix_seqlens_kv_device / prefix_block_table from the
    metadata builder's fixed-buffer pool; decode_step_1based_buf from the
    beam context) or a stable pool slice.  Host-int fallbacks are kept for the
    eager path (no fixed buffers configured).
    """
    step = beam_session.decode_step  # 0-based host int
    slot = beam_session.slot
    beam_width = beam_session.beam_width
    head_dim = key_cache.shape[-1]

    if abs(scale - (1.0 / math.sqrt(head_dim))) > 1e-6:
        # The kernel derives its own scale when scale_value <= 0; only the
        # standard 1/sqrt(head_dim) matches the model contract.
        raise RuntimeError(
            f"xllm decode execution only supports softmax_scale = 1/sqrt(head_dim), got {scale}"
        )

    # Beam queries: [W, num_heads, D] (already gathered at prefix_indices).
    beam_query = query[attn_metadata.prefix_indices]

    # Contiguous unshared pool view: [W, H_kv, max_steps, D] -- exactly the
    # layout x_attention expects for the contiguous unshared branch
    # (unshared_block_tables=None, implicit row mapping).
    unshared_key = beam_pool.unshared_key[slot, :beam_width]
    unshared_value = beam_pool.unshared_value[slot, :beam_width]

    # Shared prefix KV length: fixed-address device buffer when the metadata
    # builder runs with fixed buffers (graph path); CPU-list -> fresh tensor
    # otherwise (eager path).
    seqlens_kv = getattr(attn_metadata, "prefix_seqlens_kv_device", None)
    if seqlens_kv is None:
        seqlens_kv = torch.tensor(
            attn_metadata.prefix_seqlens_kv_cpu,
            dtype=torch.int32,
            device=beam_query.device,
        )

    # 1-based step count: fixed-address device buffer when available.
    step_count = getattr(beam_session.context, "decode_step_1based_buf", None)
    if step_count is None:
        step_count = step + 1  # dev is 0-based; kernel takes 1-based count

    out = beam_ops.x_attention(
        beam_query,
        key_cache,
        value_cache,
        unshared_key,
        unshared_value,
        attn_metadata.prefix_block_table,
        seqlens_kv,
        step_count,
        0.0,  # auto 1/sqrt(head_dim), validated above
    )
    # Scatter into the layer output: slice copy for a pure beam batch (rows
    # [0, num_prefix_indices) are exactly the beam rows; index_put_ here was
    # a host-blocking op), fancy-index scatter otherwise.
    if attn_metadata.std_indices is None:
        output[: attn_metadata.num_prefix_indices] = out
    else:
        output[attn_metadata.prefix_indices] = out


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata,
    sliding_window_size: list[int] | tuple[int, int] | None,
    scale: float,
    logits_soft_cap: float,
    vllm_flash_attn_version: int,
    s_aux: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    beam_pool: object = None,
) -> None:
    """Dispatch incremental beam decode to x_attention; forward the rest.

    Shapes the fused kernel cannot express are delegated to the cascade
    module instead of raising: mixed std/beam batches, sliding-window /
    soft-cap / sink / ALiBI configs, or a missing prefix_indices/pool all
    fall back to ``beam_attn_npu.cascade_attention``, which computes the
    same quantity per-operator (std rows via paged attention, beam rows via
    the pool-slice incremental path — numerically equivalent to the fused
    kernel, locked by the parity tests).  Without this delegation the
    documented fallback (docs/quickstart-ascend.md) would be a user-visible
    request failure whenever a std request shares a batch with beam decode.
    """
    beam_session = getattr(attn_metadata, "beam_session", None)
    is_beam_decode = getattr(attn_metadata, "is_beam_decode", False)

    def _fallback_to_cascade() -> None:
        beam_attn_npu.cascade_attention(
            output,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            sliding_window_size,
            scale,
            logits_soft_cap,
            vllm_flash_attn_version,
            s_aux=s_aux,
            alibi_slopes=alibi_slopes,
            beam_pool=beam_pool,
        )

    if is_beam_decode and beam_session is not None and beam_session.is_active:
        if attn_metadata.std_indices is not None:
            # Mixed std/beam batch: the fused kernel only encodes beam rows;
            # the cascade module handles both branches.
            _fallback_to_cascade()
            return
        if attn_metadata.prefix_indices is None:
            # No beam-row routing: cannot gather beam queries for the fused
            # kernel.  Loud failure would be a user-visible crash for a shape
            # cascade handles (its beam branch falls back to full gather).
            _fallback_to_cascade()
            return
        # (-1, -1) is the "no sliding window" sentinel the impl passes for
        # models without one; only a real window is unsupported here (the
        # cascade path ignores the argument entirely on the beam branch).
        if sliding_window_size is not None and any(
            w is not None and w >= 0 for w in sliding_window_size
        ):
            _fallback_to_cascade()
            return
        if logits_soft_cap:
            _fallback_to_cascade()
            return
        if s_aux is not None:
            _fallback_to_cascade()
            return
        if alibi_slopes is not None:
            _fallback_to_cascade()
            return
        if beam_pool is None:
            # No per-layer pool passed: the fused kernel requires it, the
            # cascade module falls back to metadata-based pool resolution.
            _fallback_to_cascade()
            return
        _xllm_incremental_decode(
            output,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            beam_session,
            beam_pool,
            scale,
        )
        return

    # Std requests, prefill, recompute and mixed batches keep the cascade
    # implementation.
    _fallback_to_cascade()


__all__ = [
    "cache_unshared_kv",
    "cascade_attention",
    "get_flash_attn_version",
    "is_available",
    "prefill_attention",
    "reshape_and_cache",
    "supports_sinks",
]
