# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Run native KDA checkpoint chunks before the grouped, output-producing Prefill."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_gr.v1.core.sched.beam_session import GRSessionState

HYBRID_PREFILL_ATTR = "gr_hybrid_prefill_chunk"


@contextmanager
def hybrid_prefill_scope(scheduler: Any, session: GRSessionState) -> Iterator[str | None]:
    """Admit one checkpoint producer until all requests can finish Prefill together."""
    if (
        not getattr(scheduler, "need_mamba_block_aligned_split", False)
        or not session.hybrid_beam_block_counts
        or session.issued_stages
    ):
        yield None
        return
    request_id = next(
        (r for r in session.request_ids if r not in session._hybrid_prefill_ready), None
    )
    if request_id is None:
        yield None
        return

    from vllm_gr.v1.core.sched.beam_async_scheduler import _progress

    original_split = scheduler._mamba_block_aligned_split
    original_running = scheduler.running
    paused = [request for request in original_running if request.request_id != request_id]
    scheduler.running = [
        request for request in original_running if request.request_id == request_id
    ]

    def split(
        request: Any,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        allowed = original_split(
            request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens
        )
        end = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
            + allowed
        )
        if end < request.num_prompt_tokens:
            return allowed
        # Park the final chunk until all batch members can emit their logits.
        # A waiting request has not yet claimed the cached blocks looked up by
        # native scheduling. Pin that exact prefix while other members warm up.
        if num_new_local_computed_tokens:
            manager = scheduler.kv_cache_manager
            groups, cached, _ = manager.coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1
            )
            if cached != num_new_local_computed_tokens:
                raise RuntimeError("Hybrid Beam prefix changed during admission")
            manager.block_pool.touch([block for group in groups for block in group])
            session._hybrid_prefill_pinned.extend(tuple(reversed(group)) for group in groups)
        session._hybrid_prefill_ready.add(request_id)
        _progress(scheduler)
        return 0

    scheduler._mamba_block_aligned_split = split
    try:
        yield request_id
    finally:
        original_running[:] = [*paused, *scheduler.running]
        scheduler.running = original_running
        scheduler._mamba_block_aligned_split = original_split


def attach_hybrid_prefill_chunk(scheduler: Any, output: Any, request_id: str) -> None:
    if not output.num_scheduled_tokens:
        return
    from vllm_gr.v1.core.sched.beam_async_scheduler import (
        _progress,
        _request_sessions,
        _sessions,
    )

    if (
        tuple(output.num_scheduled_tokens) != (request_id,)
        or not scheduler.requests[request_id].is_prefill_chunk
    ):
        raise RuntimeError("Hybrid Beam checkpoint dispatch must contain only a partial Prefill")
    session = _sessions(scheduler)[_request_sessions(scheduler)[request_id]]
    if session._hybrid_prefill_dispatch is not None:
        raise RuntimeError("Hybrid Beam checkpoint already has an outstanding dispatch")
    manager = scheduler.kv_cache_manager
    groups = manager.get_blocks(request_id).blocks
    manager.block_pool.touch([block for group in groups for block in group])
    session._hybrid_prefill_leases = tuple(tuple(reversed(group)) for group in groups)
    _progress(scheduler)
    identity = (session.session_id, session.owner_generation, scheduler._gr_progress)
    session._hybrid_prefill_dispatch = identity
    setattr(output, HYBRID_PREFILL_ATTR, identity)


def complete_hybrid_prefill_chunk(scheduler: Any, session: GRSessionState) -> None:
    from vllm_gr.v1.core.sched.beam_async_scheduler import _drain_resources, _progress
    from vllm_gr.v1.core.sched.beam_session import GRResourceStatus

    pool = scheduler.kv_cache_manager.block_pool
    for group in session._hybrid_prefill_leases:
        pool.free_blocks(group)
    session._hybrid_prefill_leases = ()
    session._hybrid_prefill_dispatch = None
    _progress(scheduler)
    if session.resource_status is GRResourceStatus.DRAINING:
        _drain_resources(scheduler, session)
