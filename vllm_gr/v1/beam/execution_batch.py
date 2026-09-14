# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronized multi-request views for fixed-width beam decode."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from typing import Any


class BeamExecutionBatch:
    """Zero-copy attention view over one ordered, synchronized cohort.

    The NPU attention operators see B * W contiguous suffix rows while
    the scheduler and result channel retain B independent request IDs.
    The view is valid only while slots are the contiguous range 0..B-1.
    """

    def __init__(self, sessions: Sequence[Any]):
        self.sessions = tuple(sessions)
        if not self.sessions:
            raise ValueError("an execution batch must contain requests")
        first = self.sessions[0]
        self.context = first.context
        self.slot = 0
        self.request_width = first.beam_width
        self.batch_size = len(self.sessions)
        self.session_id = first.session_id
        if [session.slot for session in self.sessions] != list(range(self.batch_size)):
            raise RuntimeError("beam batches require ordered contiguous slots 0..B-1")
        if any(session.context is not self.context for session in self.sessions):
            raise RuntimeError("beam batch sessions must share a context")
        geometry = (first.beam_width, first.decode_step, first._state.is_last_step)
        if any(
            (session.beam_width, session.decode_step, session._state.is_last_step) != geometry
            for session in self.sessions
        ):
            raise RuntimeError(
                "beam batches require equal beam widths, decode steps, and terminal state"
            )
        self.beam_width = self.request_width
        self.attention_width = self.batch_size * self.request_width
        if self.request_width != self.context.max_beams:
            raise RuntimeError("multi-request decode requires the configured exact beam width")
        self.pools = []
        for source in self.context.pools:
            pool = copy(source)
            pool.max_batch = 1
            storage_width = int(source.unshared_key.shape[0]) * self.request_width
            pool.max_beams = storage_width
            for name in ("unshared_key", "unshared_value"):
                tensor = getattr(source, name)
                setattr(pool, name, tensor.view(1, storage_width, *tensor.shape[2:]))
            self.pools.append(pool)

    @property
    def is_active(self) -> bool:
        return all(session.is_active for session in self.sessions)

    @property
    def decode_step(self) -> int:
        return int(self.sessions[0].decode_step)

    @property
    def _state(self) -> Any:
        return self.sessions[0]._state

    @property
    def suffix_k_buf(self) -> Any:
        size = self.attention_width * self.context.max_decode_steps
        return self.context.suffix_k_buf[:size]

    @property
    def suffix_v_buf(self) -> Any:
        size = self.attention_width * self.context.max_decode_steps
        return self.context.suffix_v_buf[:size]

    def end_session(self) -> None:
        """Release every session in the execution cohort."""
        for session in self.sessions:
            session.end_session()


def restore_cache_pool_layout(tensor: Any, request_width: int) -> Any:
    """Restore [blocks, W, ...] for xllm cache writes from a flat view."""
    if request_width < 1:
        raise ValueError("request_width must be positive")
    if tensor.shape[0] != 1 or tensor.shape[1] == request_width:
        return tensor
    if tensor.shape[1] % request_width:
        raise RuntimeError("flattened xllm pool width is not divisible by request width")
    storage_batch = tensor.shape[1] // request_width
    return tensor.view(storage_batch, request_width, *tensor.shape[2:])


def select_execution_pool(cached_pool: Any, current_pool: Any) -> Any:
    """Reuse a layer cache only while its shape and backing storage match."""
    if cached_pool is None:
        return current_pool
    cached_shape = tuple(cached_pool.unshared_key.shape[:2])
    current_shape = tuple(current_pool.unshared_key.shape[:2])
    same_storage = (
        cached_pool.unshared_key.data_ptr() == current_pool.unshared_key.data_ptr()
    )
    if cached_shape == current_shape and same_storage:
        return cached_pool
    return current_pool


def validate_batch_order(batch: Any, req_ids: Sequence[str], beam_data: Mapping[str, Any]) -> None:
    """Fail before a kernel can pair one request's prefix with another's suffix."""
    if not isinstance(batch, BeamExecutionBatch):
        return
    expected = [session.session_id for session in batch.sessions]
    if list(req_ids) != expected or set(beam_data) != set(expected):
        raise RuntimeError("beam decode cohort was split, mixed, or reordered by the scheduler")


def split_batch_decision(
    request_ids: Sequence[str],
    width: int,
    host: Mapping[str, Sequence[Sequence[Any]]],
    is_last_step: bool,
) -> dict[str, dict[str, Any]]:
    """Split global operator rows into request-local result payloads."""
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("duplicate request IDs in beam result")
    if any(len(value) != len(request_ids) for value in host.values()):
        raise ValueError("beam result batch dimension does not match request IDs")
    results = {}
    for row, request_id in enumerate(request_ids):
        item = {key: [value[row]] for key, value in host.items()}
        parents = [int(parent) - row * width for parent in host["out_token_index"][row]]
        if any(parent < 0 or parent >= width for parent in parents):
            raise RuntimeError("beam result contains an invalid or cross-request parent")
        item["out_token_index"] = [parents]
        item["is_last_step"] = bool(is_last_step)
        item["stage"] = "decode"
        item["session_id"] = request_id
        results[request_id] = item
    return results
