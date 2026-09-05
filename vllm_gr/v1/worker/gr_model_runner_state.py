# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-neutral lifecycle for one persistent GR request.

This module deliberately owns no accelerator allocations. The active V1
adapter owns device buffers, copy events, streams and graph handles through
``backend_state``. ``GRModelRunnerState`` only records protocol state and
references the single authoritative ``BeamSearchSession``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from vllm_gr.v1.engine.types import GRRequestStage

GR_FINAL_OUTPUT_FORMAT = "gr_final_i32_f32_v1"


@dataclass(frozen=True, slots=True)
class GRBackendStep:
    """References published by a backend after one device generation stage.

    ``control_output`` must alternate between two distinct backend-owned,
    fixed-address objects.  Reusing an identity for its parity slot is
    required because the one-frame look-ahead may still be copying the other
    slot; allocating or cloning a fresh object per step is rejected.
    """

    device_step: int
    next_token_ids: Any
    beam_scores: Any
    active_mask: Any
    control_output: Any


@runtime_checkable
class GRDeviceBackend(Protocol):
    """Thin boundary between the shared lifecycle and a V1 device adapter."""

    def bootstrap_from_prefill(
        self,
        runner: Any,
        state: "GRModelRunnerState",
        sampler_output: Any,
        metadata: dict[str, Any],
    ) -> GRBackendStep: ...

    def prepare_decode_inputs(
        self,
        runner: Any,
        state: "GRModelRunnerState",
        scheduler_output: Any,
        num_scheduled_tokens: Any,
        metadata: dict[str, Any],
    ) -> None: ...

    def advance_beam(
        self,
        runner: Any,
        state: "GRModelRunnerState",
        sampler_output: Any,
        metadata: dict[str, Any],
        *,
        is_last_step: bool,
    ) -> GRBackendStep: ...

    def reorder_kv(
        self,
        runner: Any,
        state: "GRModelRunnerState",
        step_result: GRBackendStep,
        *,
        is_last_step: bool,
    ) -> None: ...

    def finalize_output(
        self,
        runner: Any,
        state: "GRModelRunnerState",
    ) -> dict[str, Any]: ...

    def release(self, state: "GRModelRunnerState") -> None:
        """Make session storage safe to reclaim; support retry after failure."""

        ...


def require_gr_device_backend(runner: Any) -> GRDeviceBackend:
    """Return the backend selected at runner initialization, or fail loudly."""

    backend = getattr(runner, "_gr_device_backend", None)
    if backend is None:
        raise RuntimeError(
            "persistent GR ModelRunner V1 backend is not configured for the "
            "active platform; Step 4 installs only the shared backend contract "
            "and GPU/NPU adapters must be supplied by their platform integration"
        )
    if not isinstance(backend, GRDeviceBackend):
        raise TypeError("runner._gr_device_backend does not satisfy GRDeviceBackend")
    return backend


@dataclass(slots=True)
class GRModelRunnerState:
    """Device-independent state machine and references for one GR session."""

    request_id: str
    session: Any
    backend: GRDeviceBackend
    beam_width: int
    max_tokens: int
    continue_token_id: int
    finish_token_id: int
    stage: GRRequestStage = GRRequestStage.NEW
    step: int = 0
    device_step: int = 0
    next_token_ids: Any = None
    beam_scores: Any = None
    active_mask: Any = None
    prefix_binding: Any = None
    backend_state: Any = None
    released: bool = False
    _control_outputs: list[Any] = field(default_factory=lambda: [None, None])
    _control_slot: int = -1
    _prepared_step: int | None = None
    _release_started: bool = False
    _backend_release_completed: bool = False
    _release_lock: Any = field(default_factory=Lock, repr=False, compare=False)

    def _ensure_active(self) -> None:
        if self._release_started or self.released:
            raise RuntimeError("persistent GR state release has started; execution is closed")

    def _transition(self, next_stage: GRRequestStage) -> None:
        if next_stage is not GRRequestStage.RELEASED:
            self._ensure_active()
        if not self.stage.can_transition_to(next_stage):
            raise RuntimeError(
                f"illegal persistent GR state transition: {self.stage.value} -> {next_stage.value}"
            )
        self.stage = next_stage

    def start_prefill(self) -> None:
        self._ensure_active()
        self._transition(GRRequestStage.PREFILL_RUNNING)

    @property
    def control_token_ids(self) -> Any:
        """Current O(B) control output (legacy name retained for transport)."""

        if self._control_slot < 0:
            raise RuntimeError("persistent GR control output is not ready")
        return self._control_outputs[self._control_slot]

    @property
    def control_slots(self) -> tuple[Any, Any]:
        return self._control_outputs[0], self._control_outputs[1]

    @property
    def device_memory_bytes(self) -> int:
        """Backend-reported storage; the common state owns no device tensors."""

        value = getattr(self.backend_state, "device_memory_bytes", 0)
        return int(value() if callable(value) else value)

    def _validate_step_result(self, result: GRBackendStep, next_stage: GRRequestStage) -> None:
        self._ensure_active()
        expected = self.device_step + 1
        if result.device_step != expected:
            raise RuntimeError(
                "persistent GR backend/device step mismatch: "
                f"expected={expected}, backend={result.device_step}"
            )
        if result.control_output is None:
            raise RuntimeError("persistent GR backend returned no control output")
        slot = expected % 2
        existing = self._control_outputs[slot]
        if existing is not None and result.control_output is not existing:
            raise RuntimeError(
                f"persistent GR backend replaced a fixed control buffer slot: slot={slot}"
            )
        other = self._control_outputs[1 - slot]
        if other is not None and result.control_output is other:
            raise RuntimeError(
                "persistent GR backend must provide two distinct control buffer slots"
            )
        if not self.stage.can_transition_to(next_stage):
            raise RuntimeError(
                f"illegal persistent GR state transition: {self.stage.value} -> {next_stage.value}"
            )

    def _accept_step(self, result: GRBackendStep, next_stage: GRRequestStage) -> None:
        self._validate_step_result(result, next_stage)
        expected = self.device_step + 1
        slot = expected % 2
        self._control_outputs[slot] = result.control_output
        self._control_slot = slot
        self.next_token_ids = result.next_token_ids
        self.beam_scores = result.beam_scores
        self.active_mask = result.active_mask
        self.step += 1
        self.device_step = result.device_step
        self._transition(next_stage)

    def bootstrap_from_prefill(
        self,
        runner: Any,
        sampler_output: Any,
        metadata: dict[str, Any],
    ) -> None:
        self._ensure_active()
        if self.stage is not GRRequestStage.PREFILL_RUNNING or self.step != 0:
            raise RuntimeError("persistent GR prefill bootstrap is out of sequence")
        result = self.backend.bootstrap_from_prefill(runner, self, sampler_output, metadata)
        next_stage = (
            GRRequestStage.FINAL_READY
            if self.max_tokens == 1
            else GRRequestStage.PREFILL_BOOTSTRAPPED
        )
        self._accept_step(result, next_stage)

    def prepare_decode_inputs(
        self,
        runner: Any,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
        metadata: dict[str, Any],
    ) -> None:
        self._ensure_active()
        if self.stage not in (
            GRRequestStage.PREFILL_BOOTSTRAPPED,
            GRRequestStage.DECODE_RUNNING,
        ):
            raise RuntimeError("persistent GR decode preparation is out of sequence")
        generation_index = metadata.get("generation_index")
        if generation_index != self.step:
            raise RuntimeError(
                "persistent GR generation/device step mismatch: "
                f"issued={generation_index}, device={self.step}"
            )
        if self._prepared_step is not None:
            raise RuntimeError(
                f"persistent GR decode step is already prepared: step={self._prepared_step}"
            )
        self.backend.prepare_decode_inputs(
            runner,
            self,
            scheduler_output,
            num_scheduled_tokens,
            metadata,
        )
        self._ensure_active()
        self._prepared_step = self.step

    def advance_beam(
        self,
        runner: Any,
        sampler_output: Any,
        metadata: dict[str, Any],
    ) -> bool:
        self._ensure_active()
        if self.stage not in (
            GRRequestStage.PREFILL_BOOTSTRAPPED,
            GRRequestStage.DECODE_RUNNING,
        ):
            raise RuntimeError("persistent GR Beam advance is out of sequence")
        if self.step >= self.max_tokens:
            raise RuntimeError("persistent GR state advanced beyond max_tokens")
        generation_index = metadata.get("generation_index")
        if generation_index != self.step:
            raise RuntimeError(
                "persistent GR generation/device step mismatch: "
                f"issued={generation_index}, device={self.step}"
            )
        if self._prepared_step != self.step:
            raise RuntimeError(
                "persistent GR Beam advance requires prepare_decode_inputs "
                f"for device step {self.step}"
            )
        is_last_step = self.step + 1 == self.max_tokens
        result = self.backend.advance_beam(
            runner,
            self,
            sampler_output,
            metadata,
            is_last_step=is_last_step,
        )
        next_stage = GRRequestStage.FINAL_READY if is_last_step else GRRequestStage.DECODE_RUNNING
        self._validate_step_result(result, next_stage)
        self.backend.reorder_kv(
            runner,
            self,
            result,
            is_last_step=is_last_step,
        )
        self._accept_step(
            result,
            next_stage,
        )
        self._prepared_step = None
        return is_last_step

    def finalize_output(self, runner: Any) -> dict[str, Any]:
        self._ensure_active()
        if self.stage is not GRRequestStage.FINAL_READY:
            raise RuntimeError("persistent GR final output requested before FINAL_READY")
        output = self.backend.finalize_output(runner, self)
        self._ensure_active()
        if output.get("request_id") != self.request_id:
            raise RuntimeError("persistent GR backend returned a mismatched request_id")
        return output

    def release(self) -> bool:
        """Drain backend ownership before making the Beam session reusable.

        A failed drain leaves execution closed and the session retained for an
        explicit retry. Backends must support retry after partial cleanup; a
        successful backend release is never repeated when session cleanup fails.
        Concurrent/reentrant cleanup fails immediately instead of double-freeing.
        """

        if not self._release_lock.acquire(blocking=False):
            raise RuntimeError("persistent GR state release is already in progress")
        try:
            if self.released:
                return False
            self._release_started = True
            if not self._backend_release_completed:
                self.backend.release(self)
                self._backend_release_completed = True
            try:
                self.session.end_session()
            except KeyError:
                # A compatibility path already removed the registry slot.
                # This is safe only after backend ownership was drained.
                pass
            self._transition(GRRequestStage.RELEASED)
            self.released = True
            return True
        finally:
            self._release_lock.release()

    def reset_for_request(
        self,
        *,
        request_id: str,
        session: Any,
        continue_token_id: int,
        finish_token_id: int,
    ) -> None:
        """Rebind inactive backend storage without duplicating Beam history."""

        if not self._release_lock.acquire(blocking=False):
            raise RuntimeError("cannot reuse a persistent GR state during release")
        try:
            if not self.released:
                raise RuntimeError("cannot reuse an active persistent GR state")
            self.request_id = request_id
            self.session = session
            self.continue_token_id = continue_token_id
            self.finish_token_id = finish_token_id
            self.stage = GRRequestStage.NEW
            self.step = 0
            self.device_step = 0
            self.next_token_ids = None
            self.beam_scores = None
            self.active_mask = None
            self.prefix_binding = None
            self._control_outputs[:] = [None, None]
            self._control_slot = -1
            self._prepared_step = None
            self._release_started = False
            self._backend_release_completed = False
            self.released = False
            self.start_prefill()
        finally:
            self._release_lock.release()


def acquire_gr_model_runner_state(
    worker_or_runner: Any,
    *,
    request_id: str,
    session: Any,
    backend: GRDeviceBackend,
    beam_width: int,
    max_tokens: int,
    continue_token_id: int,
    finish_token_id: int,
) -> GRModelRunnerState:
    """Acquire one semantic state, reusing only backend-compatible storage."""

    runner = getattr(worker_or_runner, "model_runner", worker_or_runner)
    states = getattr(runner, "_gr_model_runner_states", None) or {}
    if any(state._release_started and not state.released for state in states.values()):
        raise RuntimeError("cannot acquire persistent GR state while release is incomplete")
    state = getattr(runner, "_gr_model_runner_state_cache", None)
    if state is not None and not state.released:
        raise RuntimeError("cannot reuse an active persistent GR state cache")
    runner._gr_model_runner_state_cache = None
    if (
        state is not None
        and state.backend is backend
        and state.beam_width == beam_width
        and state.max_tokens == max_tokens
    ):
        state.reset_for_request(
            request_id=request_id,
            session=session,
            continue_token_id=continue_token_id,
            finish_token_id=finish_token_id,
        )
        return state
    state = GRModelRunnerState(
        request_id=request_id,
        session=session,
        backend=backend,
        beam_width=beam_width,
        max_tokens=max_tokens,
        continue_token_id=continue_token_id,
        finish_token_id=finish_token_id,
    )
    state.start_prefill()
    return state


def release_gr_model_runner_states(
    worker_or_runner: Any,
    request_ids: list[str] | tuple[str, ...] | None = None,
) -> int:
    """Idempotently release success/abort/error/shutdown state ownership."""

    runner = getattr(worker_or_runner, "model_runner", worker_or_runner)
    states = getattr(runner, "_gr_model_runner_states", None)
    if not states:
        if request_ids is None:
            runner._gr_model_runner_state_cache = None
        return 0
    ids = tuple(states) if request_ids is None else tuple(request_ids)
    sessions = getattr(runner, "_beam_sessions", None)
    freed = 0
    first_error: BaseException | None = None
    for request_id in ids:
        state = states.get(request_id)
        if state is not None:
            try:
                released = state.release()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                # Retain both registry ownerships: queued device work may still
                # reference session storage. Other requests can still be drained.
                continue
            freed += int(released)
            if state.released and states.get(request_id) is state:
                states.pop(request_id)
            if released and first_error is None:
                runner._gr_model_runner_state_cache = state
        if sessions is not None:
            sessions.pop(request_id, None)
    runner._gr_state_freed_count = int(getattr(runner, "_gr_state_freed_count", 0)) + freed
    if request_ids is None:
        runner._gr_model_runner_state_cache = None
    if first_error is not None:
        raise first_error
    return freed


def get_gr_model_runner_state_stats(worker_or_runner: Any) -> dict[str, int]:
    runner = getattr(worker_or_runner, "model_runner", worker_or_runner)
    states = getattr(runner, "_gr_model_runner_states", {})
    return {
        "created": int(getattr(runner, "_gr_state_created_count", 0)),
        "freed": int(getattr(runner, "_gr_state_freed_count", 0)),
        "active": len(states),
    }


def get_gr_memory_stats(worker_or_runner: Any) -> dict[str, int]:
    """Return generic pool accounting without importing accelerator APIs."""

    runner = getattr(worker_or_runner, "model_runner", worker_or_runner)

    def tensor_bytes(tensors: Any) -> int:
        seen: set[int] = set()
        total = 0
        for tensor in tensors or ():
            storage_fn = getattr(tensor, "untyped_storage", None)
            if not callable(storage_fn):
                continue
            storage = storage_fn()
            pointer = int(storage.data_ptr())
            if pointer in seen:
                continue
            seen.add(pointer)
            total += int(storage.nbytes())
        return total

    native_kv_bytes = tensor_bytes(getattr(runner, "kv_caches", ()))
    context = getattr(runner, "_beam_context", None)
    beam_kv_tensors = []
    if context is not None:
        for pool in getattr(context, "pools", ()):
            beam_kv_tensors.extend((pool.unshared_key, pool.unshared_value))
    workspaces = getattr(runner, "_beam_postprocess_workspaces", None)
    return {
        "native_kv_bytes": native_kv_bytes,
        "beam_kv_bytes": tensor_bytes(beam_kv_tensors),
        "workspace_bytes": int(getattr(workspaces, "device_memory_bytes", 0)),
        "gr_state_peak_bytes": int(getattr(runner, "_gr_state_peak_bytes", 0)),
        "graph_pool_allocated_bytes": int(getattr(runner, "_gr_graph_pool_allocated_bytes", 0)),
        "graph_pool_reserved_bytes": int(getattr(runner, "_gr_graph_pool_reserved_bytes", 0)),
    }


def unpack_gr_final_output(
    output: Any,
) -> tuple[list[list[int]], list[list[int]], list[list[float]], list[float]] | None:
    """Decode the stable final ABI without imposing storage on common state."""

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
    if len(data) != (3 * count + width) * 4:
        return None
    import numpy as np

    words = np.frombuffer(data, dtype=np.int32)
    sequences = words[:count].reshape(width, steps).tolist()
    parents = words[count : 2 * count].reshape(width, steps).tolist()
    logprobs = words[2 * count : 3 * count].view(np.float32).reshape(width, steps).tolist()
    scores = words[3 * count : 3 * count + width].view(np.float32).tolist()
    return sequences, parents, logprobs, scores
