# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Asynchronous-scheduling Beam attention metadata entrypoints."""

from __future__ import annotations

from typing import Any

import torch
from vllm import envs

from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadata
from vllm_gr.v1.attention.backends.beam_attn_metadata_layout import (
    build_async_batch_layout as build_async_batch_layout,
)
from vllm_gr.v1.attention.backends.beam_attn_metadata_layout import (
    build_async_layout as build_async_layout,
)


def build_async(
    *,
    layout: dict[str, list[int]],
    views: dict[str, torch.Tensor],
    session: Any,
    mask: torch.Tensor,
    slots: torch.Tensor,
    block_size: int,
    max_model_len: int,
    num_heads: int,
    num_kv_heads: int,
) -> BeamAttentionMetadata:
    """Build a fresh metadata description over stable device storage each step.

    Native execution uses block-aligned model/pool bounds. A matching
    measured profile enables bounded prefix-length graph buckets.
    """
    if session is None or not session.is_active:
        raise RuntimeError("Beam decode requires an active beam session and KV pool")
    rows = len(layout["positions"])
    batch = len(layout["prefix_lengths"])
    width = rows // batch
    blocks = views["blocks"].view(batch, -1)
    from vllm_gr.v1.attention.backends.beam_prefix_policy import (
        PrefixShape,
        available_prefix_versions,
        device_fingerprint,
        explicit_fa_version,
        multi_processor_count,
        policy_mode,
        prefix_bucket,
        resolve_prefix_plan,
    )

    bucket = 0
    if not envs.VLLM_BATCH_INVARIANT and policy_mode() != "off":
        candidate = prefix_bucket(max(layout["prefix_lengths"]), block_size, max_model_len)
        versions = available_prefix_versions(blocks.device.index)
        explicit = explicit_fa_version()
        plan = None
        if versions:
            plan = resolve_prefix_plan(
                PrefixShape(
                    batch=batch,
                    width=width,
                    length=candidate,
                    heads=num_heads,
                    kv_heads=num_kv_heads,
                    dim=session.context.pools[0].unshared_key.shape[-1],
                    dtype=str(session.context.dtype),
                    block_size=block_size,
                ),
                # An explicit choice decides the backend; otherwise the best
                # supported one (FA3/FA4) keeps its own split-K heuristic and
                # does not need a length-scoped graph bound.
                default_version=explicit if explicit is not None else max(versions),
                fingerprint=device_fingerprint(blocks.device.index),
                supported_versions=versions,
                sm_count=multi_processor_count(blocks.device.index),
                explicit_version=explicit,
            )
        if plan is not None and plan.segments > 1:
            pages = (candidate + block_size - 1) // block_size
            chunk = max(1, (pages + plan.segments - 1) // plan.segments)
            if chunk * plan.segments <= blocks.shape[1]:
                bucket = candidate
    return BeamAttentionMetadata(
        num_actual_tokens=rows,
        num_reqs=batch,
        max_query_len=width,
        query_start_loc=views["query"],
        query_start_loc_cpu=list(layout["query"]),
        max_seq_len=max_model_len,
        seq_lens=views["sequence"],
        seq_lens_cpu=list(layout["sequence"]),
        block_table=blocks,
        slot_mapping=slots,
        is_beam_decode=True,
        beam_session=session,
        beam_active_mask=mask,
        prefix_indices=views["logits"],
        num_prefix_indices=rows,
        prefix_cu_seqlens_q_device=views["query"],
        prefix_seqlens_kv_device=views["prefix_lengths"],
        prefix_max_q_len=width,
        prefix_max_seq_len=bucket or ((max_model_len + block_size - 1) // block_size) * block_size,
        prefix_length_bucket=bucket,
        prefix_block_table=blocks,
        prefix_seqlens_q_cpu=[width] * batch,
        prefix_cu_seqlens_q_cpu=list(layout["query"]),
        prefix_seqlens_kv_cpu=list(layout["prefix_lengths"]),
        prefix_cu_seqlens_kv_cpu=None,
        suffix_cu_seqlens_q_device=views["suffix_query"],
        suffix_cu_seqlens_k_device=views["suffix_offsets"],
        suffix_seqlens_kv_device=views["suffix_lengths"],
        suffix_max_q_len=1,
        suffix_max_seq_len=session.context.max_decode_steps,
        suffix_seqlens_q_cpu=[1] * rows,
        suffix_cu_seqlens_q_cpu=list(layout["suffix_query"]),
        suffix_seqlens_kv_cpu=list(layout["suffix_lengths"]),
        suffix_cu_seqlens_kv_cpu=list(layout["suffix_offsets"]),
        incremental_suffix_kv_len=layout["suffix_lengths"][0],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
    )
