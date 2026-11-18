# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA ModelRunner contract used by the persistent GR stage core.

Implementations are peers, selected by the installed runner entrypoint. This
module has no runtime CUDA imports and is safe for shared CUDA/NPU dispatchers
to reference.

The runtime abstract base class rejects missing operations and policies at
construction, before admission on every device, without a type checker.
The class-side Protocol also checks constructor compatibility at binding sites.
Native Worker internals remain version-specific and dynamically typed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch

    from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
        BeamRequestBinding,
        GPUBeamStageRunner,
    )


class CudaStageHooks(ABC):
    """Native CUDA operations; this contract does not define an NPU adapter.

    The core owns stage/session lifetime and dispatch ordering. Hooks own the
    runner-specific native operations below. A Worker retains one concrete
    hooks type even after an individual stage owner has been released.
    """

    @property
    @abstractmethod
    def persistent_graph_install(self) -> bool:
        """Whether capture keeps the model wrapper installed for replay."""

    @property
    @abstractmethod
    def supports_transposed_head(self) -> bool:
        """Whether sliced sampling can use the transposed LM head."""

    @property
    @abstractmethod
    def specialize_constraint_steps(self) -> bool:
        """Whether device-driven Top-K uses complete SID-stage capacities."""

    def __init__(self, worker: Any) -> None:
        self.worker = worker

    @staticmethod
    @abstractmethod
    def required_constraint_rows(batch_size: int, beam_width: int) -> int:
        """Return Backend admission rows: W for V1 and B*W for V2."""

    @abstractmethod
    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention: Any,
        slot_mappings: Any,
        capture: bool,
    ) -> torch.Tensor:
        """Forward the supplied physical rows, preserving row order and device."""

    @abstractmethod
    def prefill_logits(
        self, original: Callable[..., Any], scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Return one logits row per logical member in GR request order.

        Consume native Prefill output as required by this runner, including
        counters normally advanced by its sampler. V2 clears its pending native
        state here; V1 retains it until finish_native_output. Counter updates
        use actual query lengths, including after a Prefix Cache hit. GR
        admission requires complete, output-producing Prefill; this operation
        does not enable Chunked Prefill.
        """

    @abstractmethod
    def finish_native_output(self) -> None:
        """Drop native result references after GR output work has been submitted.

        Called before clearing the pending dispatch, including cancelled
        dispatches. Earlier execute, sampling or output-submission failures
        must not reach this operation. A hook failure propagates without
        clearing pending or returning a successful receipt.

        This neither waits for GPU/output consumers nor releases GR resources.
        It does not advance native counters: V2's earlier Prefill consumption
        and bookkeeping remain in prefill_logits.
        """

    @abstractmethod
    def prefill_stage_fence(self) -> None:
        """Protect reused pinned Host inputs before subsequent CPU mutation.

        V1 synchronizes the native input event; V2 owns fresh per-step inputs.
        A GPU stream wait alone cannot protect CPU writes.
        """

    @abstractmethod
    def consume_native_state(self, scheduler_output: Any) -> None:
        """Apply native request/KV bookkeeping exactly once for this dispatch."""

    @abstractmethod
    def decode_block_ids(self, session_id: str) -> list[list[int]]:
        """Provide Host KV blocks only when scheduler metadata is absent.

        V2 requires published metadata and raises if this fallback is reached.
        """

    @abstractmethod
    def initialize_inputs(self, owner: GPUBeamStageRunner, rows: int) -> None:
        """Bind model-input storage with the lifetime required by graph replay.

        V1 allocates stage-owned tensors; V2 binds Worker-owned fixed storage.
        Captured addresses must remain valid until their graphs are discarded.
        """

    @abstractmethod
    def prepare_new_adapter(self, view: BeamBatchExecutionView) -> None:
        """Invalidate graphs whose storage will be replaced by a new owner."""

    @abstractmethod
    def decode_graph(self, batch_size: int, beam_width: int) -> Any:
        """Select the runner's graph implementation without changing its policy."""

    @abstractmethod
    def decode_logits(self, owner: GPUBeamStageRunner) -> torch.Tensor | None:
        """Return Decode logits after the core has prepared owner.inputs.

        With vocabulary slicing, return None and publish staged owner.hidden
        and owner._active_slice_cols for sampling. Staged hidden states must
        remain valid when another graph replay overwrites its output buffer.
        Each implementation retains its capture/replay policy. Unsupported
        capture may fall back only after cleanup and device health checks;
        unexpected missing graphs and programming errors remain fatal.
        """

    @abstractmethod
    def consume_retirement_state(
        self, scheduler_output: Any, bindings: tuple[BeamRequestBinding, ...]
    ) -> None:
        """Consume retirement bookkeeping, including all native batch members."""


class CudaStageHooksType(Protocol):
    """Class-side construction and admission operations checked at each binding.

    ``type[CudaStageHooks]`` alone permits subclasses with incompatible
    constructors. Keep this callable boundary separate from the runtime ABC.
    """

    def __call__(self, worker: Any) -> CudaStageHooks: ...

    def required_constraint_rows(self, batch_size: int, beam_width: int) -> int: ...


def validate_stage_hooks_type(hooks: CudaStageHooks, hooks_type: CudaStageHooksType) -> None:
    """Reject a different hooks class when reusing an active or released owner.

    The ABC guarantees both classes implement the same operations; this only
    enforces the lifetime rule that one Worker keeps one concrete hooks type.
    """
    if type(hooks) is not hooks_type:
        raise ValueError("GR execution hooks cannot change during the runner lifetime")
