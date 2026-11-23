# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""NPU input geometry for Beam batch stages.

Scheduler accounting stays at B logical requests. This adapter derives
request-major B * W device views. The stage runner owns Beam state, KV reorder,
terminal transport, and graph execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from vllm_gr.v1.beam.dispatch import GR_STAGE_OUTPUT_ATTR
from vllm_gr.v1.beam.execution_batch import (
    BeamBatchExecutionView,
    BeamExecutionContractError,
)


@dataclass(frozen=True, slots=True)
class NPUEagerStagePlan:
    """Validated NPU view of one fixed Batch Session stage."""

    execution: BeamBatchExecutionView
    logical_scheduled_tokens: tuple[int, ...]
    model_token_count: int
    logits_indices: tuple[int, ...]
    sampling_row_indices: tuple[int, ...]
    decode_positions: tuple[int, ...] | None

    @classmethod
    def from_scheduler_output(cls, scheduler_output: object) -> NPUEagerStagePlan:
        mapping = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None)
        if not isinstance(mapping, Mapping) or len(mapping) != 1:
            raise BeamExecutionContractError(
                "NPU eager execution requires exactly one fixed Batch Session"
            )
        session_id, metadata = next(iter(mapping.items()))
        execution = BeamBatchExecutionView.from_stage_metadata(metadata)
        if session_id != execution.session_id:
            raise BeamExecutionContractError(
                "GR stage map key does not match the Batch Session identity"
            )
        if getattr(scheduler_output, "beam_data", None):
            raise BeamExecutionContractError("NPU eager execution cannot mix legacy Beam metadata")
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not isinstance(scheduled, Mapping):
            raise BeamExecutionContractError(
                "scheduler output is missing logical scheduled-token counts"
            )
        execution.validate_request_order(tuple(scheduled))
        counts = tuple(scheduled[request_id] for request_id in execution.request_ids)
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in counts
        ):
            raise BeamExecutionContractError(
                "logical scheduled-token counts must be positive integers"
            )
        total = sum(counts)
        if getattr(scheduler_output, "total_num_scheduled_tokens", total) != total:
            raise BeamExecutionContractError(
                "scheduler total does not match logical request token counts"
            )
        if execution.kind == "prefill":
            if not execution.produces_output:
                raise BeamExecutionContractError(
                    "NPU eager execution requires a complete output-producing Prefill"
                )
            # Native new-request data snapshots the cached prefix before this
            # dispatch. Only the remaining prompt tokens have hidden-state rows.
            computed_by_request = {
                request.req_id: request.num_computed_tokens
                for request in getattr(scheduler_output, "scheduled_new_reqs", ())
            }
            logits = []
            offset = 0
            for request_id, count, length in zip(
                execution.request_ids, counts, execution.num_prompt_tokens_by_request, strict=True
            ):
                computed = computed_by_request.get(request_id, 0)
                if (
                    isinstance(computed, bool)
                    or not isinstance(computed, int)
                    or computed < 0
                    or computed + count != length
                ):
                    raise BeamExecutionContractError(
                        "NPU Prefill must complete each prompt: "
                        f"request_id={request_id!r}, cached_tokens={computed!r}, "
                        f"scheduled_tokens={count}, prompt_tokens={length}"
                    )
                offset += count
                logits.append(offset - 1)
            return cls(
                execution,
                counts,
                total,
                tuple(logits),
                tuple(range(execution.batch_size)),
                None,
            )
        if any(count != 1 for count in counts):
            raise BeamExecutionContractError(
                "Scheduler Decode accounting must remain one token per request"
            )
        step = execution.decode_step
        assert step is not None
        positions = execution.expand_per_request(
            tuple(length + step for length in execution.num_prompt_tokens_by_request),
            name="decode positions",
        )
        rows = execution.num_execution_rows
        return cls(
            execution,
            counts,
            rows,
            tuple(range(rows)),
            execution.row_request_indices,
            positions,
        )

    def prepare_decode_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        winner_token_ids: torch.Tensor,
    ) -> None:
        """Bind winners and materialize NPU positions directly on device."""

        if self.execution.kind != "decode" or self.decode_positions is None:
            raise BeamExecutionContractError("Decode input binding requires a Decode plan")
        rows = self.execution.num_execution_rows
        if input_ids.ndim != 1 or input_ids.numel() < rows:
            raise BeamExecutionContractError("input_ids has insufficient Decode capacity")
        if positions.ndim != 1 or positions.numel() < rows:
            raise BeamExecutionContractError("positions has insufficient Decode capacity")
        expected = (self.execution.batch_size, self.execution.input_width)
        if tuple(winner_token_ids.shape) not in (expected, (rows,)):
            raise BeamExecutionContractError(
                f"winner_token_ids must have shape {expected} or ({rows},)"
            )
        if winner_token_ids.device != input_ids.device:
            raise BeamExecutionContractError(
                "winner_token_ids and input_ids must be on the same device"
            )
        input_ids[:rows].copy_(winner_token_ids.reshape(-1))
        if positions.device.type == "npu":
            # Positions are constant within each request's physical row range.
            # Follow the execution layout (including dynamic stage widths) and
            # merge adjacent equal requests without scanning every beam row.
            start = 0
            value = self.decode_positions[0]
            for request_index in range(1, self.execution.batch_size):
                stop = self.execution.rows_for_request(request_index).start
                next_value = self.decode_positions[stop]
                if next_value != value:
                    positions[start:stop].fill_(value)
                    start, value = stop, next_value
            positions[start:rows].fill_(value)
        else:
            positions[:rows].copy_(
                torch.as_tensor(
                    self.decode_positions,
                    dtype=positions.dtype,
                    device=positions.device,
                )
            )


def validate_npu_eager_capabilities(worker: object) -> None:
    """Fail before mutation when the NPU eager-or-graph adapter is unsupported.

    Graph misses intentionally execute through the same eager data path, so
    enforce_eager=False is valid once the NPU graph adapter is enabled.
    """

    device_type = str(getattr(worker, "device", "cpu")).split(":", 1)[0].lower()
    parallel = getattr(worker, "parallel_config", None)
    parallel_sizes = (
        int(getattr(parallel, "tensor_parallel_size", 1)),
        int(getattr(parallel, "pipeline_parallel_size", 1)),
        int(getattr(parallel, "data_parallel_size", 1)),
    )
    model = getattr(worker, "model_config", None)
    if device_type != "npu":
        raise ValueError("NPU eager Beam execution requires an NPU worker")
    config = getattr(worker, "vllm_config", None)
    if config is not None:
        from vllm_gr.v1.engine.beam_admission import validate_gr_device_config

        validate_gr_device_config(config)
    if not bool(getattr(worker, "use_async_scheduling", False)):
        raise ValueError("fixed Batch Session execution requires async scheduling")
    context_parallel = (
        int(getattr(parallel, "decode_context_parallel_size", 1)),
        int(getattr(parallel, "prefill_context_parallel_size", 1)),
    )
    if parallel_sizes != (1, 1, 1) or context_parallel != (1, 1):
        raise ValueError("initial NPU Multi-Batch execution requires TP/PP/DP/DCP/PCP=1")
    if getattr(worker, "speculative_config", None) is not None:
        raise ValueError("initial NPU Multi-Batch execution does not support speculative decode")
    if getattr(model, "dtype", None) not in (torch.float16, torch.bfloat16):
        raise ValueError("initial NPU Multi-Batch execution requires FP16 or BF16")
    if (
        bool(getattr(model, "is_encoder_decoder", False))
        or bool(getattr(model, "is_hybrid", False))
        or bool(getattr(worker, "supports_mm_inputs", False))
        or getattr(worker, "lora_config", None) is not None
        or getattr(getattr(worker, "vllm_config", None), "kv_transfer_config", None) is not None
    ):
        raise ValueError("initial NPU Multi-Batch execution requires a plain text decoder")


__all__ = [
    "NPUEagerStagePlan",
    "validate_npu_eager_capabilities",
]
