# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-resident state for one persistent GR request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_gr.v1.engine.types import GRRequestStage

GR_FINAL_OUTPUT_FORMAT = "gr_final_i32_f32_v1"


@dataclass(slots=True)
class GRModelRunnerState:
    request_id: str
    session: Any
    max_tokens: int
    continue_token_id: int
    finish_token_id: int
    stage: GRRequestStage
    step: int
    device_step: int
    next_token_ids: torch.Tensor
    control_token_ids: torch.Tensor
    beam_scores: torch.Tensor
    parent_ids: torch.Tensor
    sequences: torch.Tensor
    active_mask: torch.Tensor
    selected_logprobs: torch.Tensor
    prefix_block_table: torch.Tensor | None
    scratch_parent_ids: torch.Tensor
    scratch_sequences: torch.Tensor
    scratch_logprobs: torch.Tensor
    scratch_scores: torch.Tensor
    final_result: torch.Tensor
    host_final_result: torch.Tensor
    final_copy_event: Any | None
    released: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        request_id: str,
        session: Any,
        beam_width: int,
        max_tokens: int,
        continue_token_id: int,
        finish_token_id: int,
        device: torch.device,
        prefix_block_table: torch.Tensor | None = None,
    ) -> "GRModelRunnerState":
        history_shape = (beam_width, max_tokens)
        result_words = 3 * beam_width * max_tokens + beam_width
        return cls(
            request_id=request_id,
            session=session,
            max_tokens=max_tokens,
            continue_token_id=continue_token_id,
            finish_token_id=finish_token_id,
            stage=GRRequestStage.PREFILL_RUNNING,
            step=0,
            device_step=0,
            next_token_ids=torch.empty(beam_width, dtype=torch.int32, device=device),
            control_token_ids=torch.full(
                (1, 1), continue_token_id, dtype=torch.int32, device=device
            ),
            beam_scores=torch.empty(beam_width, dtype=torch.float32, device=device),
            parent_ids=torch.zeros(history_shape, dtype=torch.int32, device=device),
            sequences=torch.zeros(history_shape, dtype=torch.int32, device=device),
            active_mask=torch.ones(beam_width, dtype=torch.bool, device=device),
            selected_logprobs=torch.zeros(history_shape, dtype=torch.float32, device=device),
            prefix_block_table=prefix_block_table,
            scratch_parent_ids=torch.empty(history_shape, dtype=torch.int32, device=device),
            scratch_sequences=torch.empty(history_shape, dtype=torch.int32, device=device),
            scratch_logprobs=torch.empty(history_shape, dtype=torch.float32, device=device),
            scratch_scores=torch.empty(beam_width, dtype=torch.float32, device=device),
            final_result=torch.empty(result_words, dtype=torch.int32, device=device),
            host_final_result=torch.empty(
                result_words,
                dtype=torch.int32,
                device="cpu",
                pin_memory=device.type == "cuda",
            ),
            final_copy_event=(torch.Event(enable_timing=False) if device.type == "cuda" else None),
        )

    @property
    def beam_width(self) -> int:
        return self.next_token_ids.numel()

    @property
    def device_memory_bytes(self) -> int:
        tensors = (
            self.next_token_ids,
            self.control_token_ids,
            self.beam_scores,
            self.parent_ids,
            self.sequences,
            self.active_mask,
            self.selected_logprobs,
            self.scratch_parent_ids,
            self.scratch_sequences,
            self.scratch_logprobs,
            self.scratch_scores,
            self.final_result,
        )
        total = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        if self.prefix_block_table is not None:
            total += self.prefix_block_table.numel() * self.prefix_block_table.element_size()
        return total

    def bootstrap(self, token_ids: torch.Tensor, scores: torch.Tensor) -> None:
        width = self.beam_width
        ids = token_ids.reshape(-1)[:width]
        values = scores.reshape(-1)[:width]
        self.next_token_ids.copy_(ids)
        self.beam_scores.copy_(values)
        self.sequences[:, 0].copy_(ids)
        self.selected_logprobs[:, 0].copy_(values)
        self.parent_ids[:, 0].zero_()
        self.step = 1
        self.device_step = 1
        self.stage = (
            GRRequestStage.FINAL_READY
            if self.step == self.max_tokens
            else GRRequestStage.PREFILL_BOOTSTRAPPED
        )

    def advance(
        self,
        token_ids: torch.Tensor,
        scores: torch.Tensor,
        parent_ids: torch.Tensor,
    ) -> None:
        width = self.beam_width
        if self.step >= self.max_tokens:
            raise RuntimeError("persistent GR state advanced beyond max_tokens")
        ids = token_ids.reshape(-1)[:width].to(dtype=torch.int32)
        values = scores.reshape(-1)[:width].to(dtype=torch.float32)
        parents = parent_ids.reshape(-1)[:width].to(dtype=torch.int64)
        if self.step:
            torch.index_select(
                self.sequences,
                0,
                parents,
                out=self.scratch_sequences,
            )
            torch.index_select(
                self.parent_ids,
                0,
                parents,
                out=self.scratch_parent_ids,
            )
            torch.index_select(
                self.selected_logprobs,
                0,
                parents,
                out=self.scratch_logprobs,
            )
            torch.index_select(
                self.beam_scores,
                0,
                parents,
                out=self.scratch_scores,
            )
            self.sequences.copy_(self.scratch_sequences)
            self.parent_ids.copy_(self.scratch_parent_ids)
            self.selected_logprobs.copy_(self.scratch_logprobs)
        self.sequences[:, self.step].copy_(ids)
        self.parent_ids[:, self.step].copy_(parents.to(torch.int32))
        self.selected_logprobs[:, self.step].copy_(values - self.scratch_scores)
        self.next_token_ids.copy_(ids)
        self.beam_scores.copy_(values)
        self.step += 1
        self.device_step += 1
        self.stage = (
            GRRequestStage.FINAL_READY
            if self.step == self.max_tokens
            else GRRequestStage.DECODE_RUNNING
        )

    def set_control_token(self, finished: bool) -> None:
        self.control_token_ids.fill_(self.finish_token_id if finished else self.continue_token_id)

    def make_final_output(self) -> dict[str, Any]:
        width = self.beam_width
        count = width * self.step
        self.final_result[:count].copy_(self.sequences[:, : self.step].reshape(-1))
        self.final_result[count : 2 * count].copy_(self.parent_ids[:, : self.step].reshape(-1))
        self.final_result[2 * count : 3 * count].copy_(
            self.selected_logprobs[:, : self.step].reshape(-1).view(torch.int32)
        )
        self.final_result[3 * count : 3 * count + width].copy_(self.beam_scores.view(torch.int32))
        words = 3 * count + width
        host = self.host_final_result[:words]
        if self.final_copy_event is None:
            host.copy_(self.final_result[:words])
        else:
            host.copy_(self.final_result[:words], non_blocking=True)
            self.final_copy_event.record()
            self.final_copy_event.synchronize()
        return {
            "request_id": self.request_id,
            "buffer_format": GR_FINAL_OUTPUT_FORMAT,
            "shape": [width, self.step],
            "buffer": host.numpy().tobytes(),
            "is_last_step": True,
            "step": self.step,
            "device_step": self.device_step,
        }

    def release(self) -> bool:
        if self.released:
            return False
        self.released = True
        self.stage = GRRequestStage.RELEASED
        self.session.end_session()
        return True

    def reset_for_request(
        self,
        *,
        request_id: str,
        session: Any,
        continue_token_id: int,
        finish_token_id: int,
        prefix_block_table: torch.Tensor | None,
    ) -> None:
        """Rebind an inactive fixed-shape allocation to a new GR request."""
        if not self.released:
            raise RuntimeError("cannot reuse an active persistent GR state")
        self.request_id = request_id
        self.session = session
        self.continue_token_id = continue_token_id
        self.finish_token_id = finish_token_id
        self.prefix_block_table = prefix_block_table
        self.stage = GRRequestStage.PREFILL_RUNNING
        self.step = 0
        self.device_step = 0
        self.released = False
        self.control_token_ids.fill_(continue_token_id)
        self.active_mask.fill_(True)


def acquire_gr_model_runner_state(
    worker_or_runner: Any,
    *,
    request_id: str,
    session: Any,
    beam_width: int,
    max_tokens: int,
    continue_token_id: int,
    finish_token_id: int,
    device: torch.device,
    prefix_block_table: torch.Tensor | None = None,
) -> GRModelRunnerState:
    """Acquire a request state, reusing the runner's last fixed allocation."""
    runner = getattr(worker_or_runner, "model_runner", worker_or_runner)
    state = getattr(runner, "_gr_model_runner_state_cache", None)
    runner._gr_model_runner_state_cache = None
    if (
        state is not None
        and state.beam_width == beam_width
        and state.max_tokens == max_tokens
        and state.next_token_ids.device == device
    ):
        state.reset_for_request(
            request_id=request_id,
            session=session,
            continue_token_id=continue_token_id,
            finish_token_id=finish_token_id,
            prefix_block_table=prefix_block_table,
        )
        return state
    return GRModelRunnerState.allocate(
        request_id=request_id,
        session=session,
        beam_width=beam_width,
        max_tokens=max_tokens,
        continue_token_id=continue_token_id,
        finish_token_id=finish_token_id,
        device=device,
        prefix_block_table=prefix_block_table,
    )

def release_gr_model_runner_states(
    worker_or_runner: Any,
    request_ids: list[str] | tuple[str, ...] | None = None,
) -> int:
    """Release persistent GR states owned by one ModelRunner.

    This is also mounted as a named Worker RPC. Passing no IDs is the worker
    shutdown/exception path; abort passes only the affected request IDs.
    """

    runner = getattr(worker_or_runner, "model_runner", worker_or_runner)
    states = getattr(runner, "_gr_model_runner_states", None)
    if not states:
        if request_ids is None:
            runner._gr_model_runner_state_cache = None
        return 0
    ids = tuple(states) if request_ids is None else tuple(request_ids)
    sessions = getattr(runner, "_beam_sessions", None)
    freed = 0
    for request_id in ids:
        state = states.pop(request_id, None)
        if state is not None:
            released = state.release()
            freed += int(released)
            if released:
                runner._gr_model_runner_state_cache = state
        if sessions is not None:
            sessions.pop(request_id, None)
    runner._gr_state_freed_count = (
        int(getattr(runner, "_gr_state_freed_count", 0)) + freed
    )
    if request_ids is None:
        runner._gr_model_runner_state_cache = None
    return freed


def get_gr_model_runner_state_stats(worker_or_runner: Any) -> dict[str, int]:
    """Return persistent GR state lifecycle counts for validation."""

    runner = getattr(worker_or_runner, "model_runner", worker_or_runner)
    states = getattr(runner, "_gr_model_runner_states", {})
    return {
        "created": int(getattr(runner, "_gr_state_created_count", 0)),
        "freed": int(getattr(runner, "_gr_state_freed_count", 0)),
        "active": len(states),
    }


def get_gr_memory_stats(worker_or_runner: Any) -> dict[str, int]:
    """Return accelerator memory owned by the native KV and GR V1 pools."""

    runner = getattr(worker_or_runner, "model_runner", worker_or_runner)

    def tensor_bytes(tensors: Any) -> int:
        seen: set[int] = set()
        total = 0
        for tensor in tensors or ():
            if not isinstance(tensor, torch.Tensor):
                continue
            pointer = tensor.untyped_storage().data_ptr()
            if pointer in seen:
                continue
            seen.add(pointer)
            total += tensor.untyped_storage().nbytes()
        return total

    native_kv_bytes = tensor_bytes(getattr(runner, "kv_caches", ()))
    context = getattr(runner, "_beam_context", None)
    beam_kv_tensors = []
    if context is not None:
        for pool in getattr(context, "pools", ()):
            beam_kv_tensors.extend((pool.unshared_key, pool.unshared_value))
    beam_kv_bytes = tensor_bytes(beam_kv_tensors)
    workspaces = getattr(runner, "_beam_postprocess_workspaces", None)
    workspace_bytes = int(getattr(workspaces, "device_memory_bytes", 0))
    graph_pool_allocated_bytes = 0
    graph_pool_reserved_bytes = 0
    device = getattr(runner, "device", None)
    if str(getattr(device, "type", "")) == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        snapshot = torch.cuda.memory._snapshot()
        for segment in snapshot.get("segments", ()):
            if segment.get("device") != device_index:
                continue
            if tuple(segment.get("segment_pool_id", (0, 0))) == (0, 0):
                continue
            graph_pool_allocated_bytes += int(segment.get("allocated_size", 0))
            graph_pool_reserved_bytes += int(segment.get("total_size", 0))
    return {
        "native_kv_bytes": native_kv_bytes,
        "beam_kv_bytes": beam_kv_bytes,
        "workspace_bytes": workspace_bytes,
        "gr_state_peak_bytes": int(getattr(runner, "_gr_state_peak_bytes", 0)),
        "graph_pool_allocated_bytes": graph_pool_allocated_bytes,
        "graph_pool_reserved_bytes": graph_pool_reserved_bytes,
    }


def unpack_gr_final_output(
    output: Any,
) -> tuple[list[list[int]], list[list[int]], list[list[float]], list[float]] | None:
    if not isinstance(output, dict) or output.get("buffer_format") != GR_FINAL_OUTPUT_FORMAT:
        return None
    shape = output.get("shape")
    data = output.get("buffer")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or not all(isinstance(value, int) and value > 0 for value in shape)
        or not isinstance(data, (bytes, bytearray, memoryview))
    ):
        return None
    width, steps = shape
    count = width * steps
    expected_bytes = (3 * count + width) * 4
    if len(data) != expected_bytes:
        return None
    import numpy as np

    words = np.frombuffer(data, dtype=np.int32)
    sequences = words[:count].reshape(width, steps).tolist()
    parents = words[count : 2 * count].reshape(width, steps).tolist()
    logprobs = words[2 * count : 3 * count].view(np.float32).reshape(width, steps).tolist()
    scores = words[3 * count : 3 * count + width].view(np.float32).tolist()
    return sequences, parents, logprobs, scores
