# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""ModelRunner V1 implementation of the CUDA stage hooks contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch

from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
    BeamRequestBinding,
    GPUBeamStageRunner,
    _hidden_logits,
    should_replay_decode_graph,
)
from vllm_gr.v1.worker.cuda.gpu_stage_hooks import CudaStageHooks


class CudaStageHooksV1(CudaStageHooks):
    """V1 native operations and graph policy for the shared CUDA stage core."""

    persistent_graph_install: ClassVar[bool] = True
    supports_transposed_head: ClassVar[bool] = True
    specialize_constraint_steps: ClassVar[bool] = False

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention: Any,
        slot_mappings: Any,
        capture: bool,
    ) -> torch.Tensor:
        """Run one beam-decode forward, returning hidden states."""
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.cudagraph_dispatcher import BatchDescriptor

        worker = self.worker
        with set_forward_context(
            attention,
            worker.vllm_config,
            num_tokens=input_ids.numel(),
            cudagraph_runtime_mode=CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=input_ids.numel(), uniform=True),
            slot_mapping=slot_mappings,
        ):
            return worker._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
            )

    def prefill_logits(
        self, original: Callable[..., Any], scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Run the native prefill execution and extract its logits."""
        worker = self.worker
        out = original(worker, scheduler_output, *args, **kwargs)
        if out is not None or worker.execute_model_state is None:
            raise RuntimeError("native GR Prefill must return logits through execute_model_state")
        return worker.execute_model_state.logits

    def finish_native_output(self) -> None:
        """Drop native results after GR has submitted its output work."""
        self.worker.execute_model_state = None
        self.worker.kv_connector_output = None

    def prefill_stage_fence(self) -> None:
        """Wait for any native prefill reads of reused CPU input buffers.

        V1: the pinned InputBatch buffers the Prefill forward reads
        asynchronously are mutated on the CPU by ``_update_states``; a stream
        wait on the decode stream cannot protect those Host writes.  V2
        builds a fresh InputBatch per step and needs no fence.
        """
        event = getattr(self.worker, "prepare_inputs_event", None)
        if event is not None:
            event.synchronize()

    def consume_native_state(self, scheduler_output: Any) -> None:
        """Apply the runner's native request-state bookkeeping."""
        self.worker._update_states(scheduler_output)

    def decode_block_ids(self, session_id: str) -> list[list[int]]:
        """Host KV block ids for the session's prompt, grouped per KV group."""
        native = self.worker.requests[session_id]
        block_ids: list[list[int]] = native.block_ids
        return block_ids

    def initialize_inputs(self, owner: GPUBeamStageRunner, rows: int) -> None:
        """V1 model inputs belong to the stage; upload storage stays lazy."""
        owner.input_ids = torch.empty(rows, device=owner.beam.state.device, dtype=torch.int32)
        owner.positions = torch.empty_like(owner.input_ids, dtype=torch.int64)
        owner.slots = torch.full_like(owner.positions, -1)
        owner.execution_mask = torch.empty_like(owner.input_ids, dtype=torch.bool)

    @staticmethod
    def required_constraint_rows(batch_size: int, beam_width: int) -> int:
        return beam_width

    def prepare_new_adapter(self, view: BeamBatchExecutionView) -> None:
        graph = getattr(self.worker, "_beam_decode_graph", None)
        if graph is not None:
            graph.discard_geometry(view.batch_size, view.beam_width)

    def decode_graph(self, batch_size: int, beam_width: int) -> Any:
        from vllm_gr.v1.worker.cuda.beam_decode_graph import get_beam_decode_graph

        return get_beam_decode_graph(self.worker, batch_size=batch_size, beam_width=beam_width)

    def decode_logits(self, owner: GPUBeamStageRunner) -> torch.Tensor | None:
        """V1 can capture a missing graph against a new stage's buffers."""
        assert owner.inputs is not None
        worker = self.worker
        from vllm_gr.v1.worker.gr_context import (
            _get_gr_config,
        )

        graph_enabled = (
            not worker.model_config.enforce_eager
            and _get_gr_config(worker).beam_graph_enabled
            and not getattr(worker, "_gr_async_decode_graph_capture_failed", False)
        )
        if graph_enabled:
            graph = self.decode_graph(len(owner.bindings), owner.inputs.width)
            bucket = getattr(getattr(owner, "metadata", None), "prefix_length_bucket", 0)
            if bucket:
                graph.select(len(owner.bindings), owner.inputs.width, bucket)
            if not graph.ready:
                graph.capture(
                    owner._forward_hidden,
                    persistent_install=self.persistent_graph_install,
                )
            owner.graph = graph
        if should_replay_decode_graph(worker, owner.graph):
            return _hidden_logits(owner, owner.graph.replay())
        # Eager Decode keeps the stage's ``_forward`` entry point; ``_forward``
        # publishes ``hidden`` itself when the LM head is sliced.
        return owner._forward()

    def consume_retirement_state(
        self, scheduler_output: Any, bindings: tuple[BeamRequestBinding, ...]
    ) -> None:
        self.consume_native_state(scheduler_output)
