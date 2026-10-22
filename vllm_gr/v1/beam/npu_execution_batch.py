# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU attention views for one fixed-width Beam Batch Session."""

from __future__ import annotations

from copy import copy
from typing import Any, Sequence

from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView


class NPUBeamExecutionBatch:
    """Zero-copy NPU attention facade over ordered per-request sessions.

    The cache operator keeps its physical ``[B, W, ...]`` storage, while
    x_attention consumes a flattened ``[1, B * W, ...]`` view of the same
    storage. Scheduler and result accounting remain at B logical requests.
    """

    def __init__(
        self,
        execution: BeamBatchExecutionView,
        sessions: Sequence[Any],
    ) -> None:
        self.execution = execution
        self.sessions = tuple(sessions)
        if len(self.sessions) != execution.batch_size:
            raise ValueError("NPU execution needs one Beam session per request")
        first = self.sessions[0]
        self.context = first.context
        self.slot = 0
        self.session_id = execution.session_id
        self.request_width = execution.input_width
        self.beam_width = self.request_width
        self.batch_size = execution.batch_size
        self.attention_width = self.batch_size * self.request_width
        execution.validate_request_order(tuple(session.session_id for session in self.sessions))
        if [session.slot for session in self.sessions] != list(range(self.batch_size)):
            raise RuntimeError("NPU Beam batch slots must be contiguous and request ordered")
        if any(session.context is not self.context for session in self.sessions):
            raise RuntimeError("NPU Beam batch sessions must share one attention context")
        if any(session.beam_width != execution.beam_width for session in self.sessions):
            raise RuntimeError("NPU Beam batch sessions must use the fixed Beam width")
        if self.context.max_beams != execution.beam_width:
            raise RuntimeError("NPU Beam pool width must equal the execution capacity")

        sources = self.context.pools
        if execution.dynamic_beam_widths is not None:
            from vllm_gr.v1.beam.npu_dynamic_beam import get_npu_stage_pools

            sources = get_npu_stage_pools(self.context, self.request_width)

        self.pools = []
        for source in sources:
            pool = copy(source)
            pool.max_batch = 1
            pool.max_beams = self.attention_width
            for name in ("unshared_key", "unshared_value"):
                tensor = getattr(source, name)
                setattr(
                    pool,
                    name,
                    tensor.view(1, self.attention_width, *tensor.shape[2:]),
                )
            self.pools.append(pool)

    @property
    def is_active(self) -> bool:
        return all(session.is_active for session in self.sessions)

    @property
    def decode_step(self) -> int:
        steps = {int(session.decode_step) for session in self.sessions}
        if len(steps) != 1:
            raise RuntimeError("NPU Beam batch sessions left the same Decode step")
        return steps.pop()

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
        for session in self.sessions:
            if session.is_active:
                session.end_session()


def restore_npu_cache_pool_layout(tensor: Any, request_width: int) -> Any:
    """Restore ``[B, W, ...]`` before the xllm cache-write operator."""

    if request_width < 1:
        raise ValueError("request_width must be positive")
    if tensor.shape[0] != 1 or tensor.shape[1] == request_width:
        return tensor
    if tensor.shape[1] % request_width:
        raise RuntimeError("flattened NPU pool width is not divisible by Beam width")
    batch = tensor.shape[1] // request_width
    return tensor.view(batch, request_width, *tensor.shape[2:])


__all__ = ["NPUBeamExecutionBatch", "restore_npu_cache_pool_layout"]
