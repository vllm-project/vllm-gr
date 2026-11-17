# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Asynchronous-scheduling Beam attention metadata entrypoints."""

from __future__ import annotations

from typing import Any

import torch
from vllm import envs

from vllm_gr.config.models import DEFAULT_DEPLOYMENT
from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadata


def build_async_batch_layout(
    *,
    width: int,
    prompt_lengths: list[int],
    decode_step: int,
    block_ids_by_request: list[list[int]],
    max_blocks: int,
) -> dict[str, list[int]]:
    """Build request-major metadata for B logical requests and B*W rows."""
    batch = len(prompt_lengths)
    if (
        type(width) is not int
        or width < 1
        or type(decode_step) is not int
        or decode_step < 0
        or not batch
        or len(block_ids_by_request) != batch
        or any(type(length) is not int or length < 1 for length in prompt_lengths)
    ):
        raise ValueError("Asynchronous scheduling batch geometry is invalid")
    query = [item * width for item in range(batch + 1)]
    rows = batch * width
    blocks: list[int] = []
    for request_blocks in block_ids_by_request:
        if len(request_blocks) > max_blocks:
            raise ValueError("Asynchronous scheduling prefix block table exceeds capacity")
        blocks.extend(request_blocks)
        blocks.extend([0] * (max_blocks - len(request_blocks)))
    suffix_length = decode_step + 1
    return {
        "positions": [prompt + decode_step for prompt in prompt_lengths for _ in range(width)],
        "query": query,
        "sequence": [prompt + suffix_length for prompt in prompt_lengths],
        "prefix_lengths": list(prompt_lengths),
        "blocks": blocks,
        "logits": list(range(rows)),
        "suffix_query": list(range(rows + 1)),
        "suffix_lengths": [suffix_length] * rows,
        "suffix_offsets": [item * suffix_length for item in range(rows + 1)],
    }


def build_async_layout(
    *,
    width: int,
    prompt_len: int,
    decode_step: int,
    block_ids: list[int],
    max_blocks: int,
    query_start_loc_cpu: list[int] | tuple[int, ...],
    seq_lens_cpu: list[int] | tuple[int, ...],
) -> dict[str, list[int]]:
    """Complete CPU layout for B=1 incremental Decode, with no cache lookup.

    CPU mirrors are mandatory. This entry never falls back to device reads,
    geometry caches, cached views, cached schedules or the legacy uploader.
    Common attention has one packed request; suffix attention has W beams.
    """
    if any(
        not isinstance(values, (list, tuple)) or any(type(value) is not int for value in values)
        for values in (query_start_loc_cpu, seq_lens_cpu)
    ):
        raise ValueError(
            "Asynchronous scheduling attention requires explicit CPU query/sequence mirrors"
        )
    if list(query_start_loc_cpu) != [0, width] or list(seq_lens_cpu) != [
        prompt_len + decode_step + 1
    ]:
        raise ValueError(
            "Asynchronous scheduling common CPU mirrors disagree with the planned Decode"
        )
    if len(block_ids) > max_blocks:
        raise ValueError("Asynchronous scheduling prefix block table exceeds model capacity")
    return build_async_batch_layout(
        width=width,
        prompt_lengths=[prompt_len],
        decode_step=decode_step,
        block_ids_by_request=[block_ids],
        max_blocks=max_blocks,
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
    prefix_policy: str = DEFAULT_DEPLOYMENT.prefix_policy,
    prefix_min_chain_us: float = DEFAULT_DEPLOYMENT.prefix_min_chain_us,
    prefix_calibrations: dict | None = None,
    explicit_fa_version: int | None = None,
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
        multi_processor_count,
        prefix_bucket,
        resolve_prefix_plan,
    )

    bucket = 0
    if not envs.VLLM_BATCH_INVARIANT and prefix_policy != "off":
        candidate = prefix_bucket(max(layout["prefix_lengths"]), block_size, max_model_len)
        versions = available_prefix_versions(blocks.device.index)
        explicit = explicit_fa_version
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
                policy=prefix_policy,
                min_chain_us=prefix_min_chain_us,
                calibrations=prefix_calibrations,
            )
        if plan is not None and plan.segments > 1:
            pages = (candidate + block_size - 1) // block_size
            chunk = max(1, (pages + plan.segments - 1) // plan.segments)
            if chunk * plan.segments <= blocks.shape[1]:
                bucket = candidate
    return BeamAttentionMetadata(
        num_actual_tokens=rows,
        prefix_policy=prefix_policy,
        prefix_min_chain_us=prefix_min_chain_us,
        explicit_fa_version=explicit_fa_version,
        prefix_calibrations=prefix_calibrations,
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
