# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU eager input geometry for fixed Beam batch stages.

Scheduler accounting stays at B logical requests. This adapter derives
request-major B * W device views. Beam state, KV reorder, terminal transport,
and graph execution intentionally belong to later stages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass

import torch

from vllm_gr.v1.beam.execution_batch import (
    BeamBatchExecutionView,
    BeamExecutionContractError,
)

GR_STAGE_OUTPUT_ATTR = "gr_stage_metadata"
_SAMPLING_TENSOR_ATTRS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalties",
    "presence_penalties",
    "repetition_penalties",
)


@dataclass(frozen=True, slots=True)
class NPUDecodeAttentionLayout:
    """Request-isolated geometry for the NPU attention binder."""

    request_ids: tuple[str, ...]
    row_request_indices: tuple[int, ...]
    row_beam_indices: tuple[int, ...]
    query_start_locations: tuple[int, ...]
    prefix_sequence_lengths: tuple[int, ...]
    suffix_sequence_lengths: tuple[int, ...]
    prefix_block_tables: tuple[tuple[int, ...], ...]
    pool_slots: tuple[int, ...]

    @property
    def num_rows(self) -> int:
        return len(self.row_request_indices)


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
        """Bind request-major winner tokens and positions without a D2H copy."""

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
        positions[:rows].copy_(
            torch.as_tensor(
                self.decode_positions,
                dtype=positions.dtype,
                device=positions.device,
            )
        )

    def remap_sampling_metadata(self, metadata: object) -> object:
        """Copy and expand request-level sampling tensors onto physical rows."""

        remapped = copy(metadata)
        rows = self.execution.num_execution_rows
        batch = self.execution.batch_size
        if rows == batch:
            return remapped
        index_cache: dict[torch.device, torch.Tensor] = {}
        for name in _SAMPLING_TENSOR_ATTRS:
            value = getattr(metadata, name, None)
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                continue
            if value.ndim == 0 or value.shape[0] == rows:
                continue
            if value.shape[0] != batch:
                raise BeamExecutionContractError(
                    f"sampling metadata {name!r} must have B or B*W leading rows"
                )
            index = index_cache.get(value.device)
            if index is None:
                index = torch.as_tensor(
                    self.sampling_row_indices,
                    dtype=torch.long,
                    device=value.device,
                )
                index_cache[value.device] = index
            setattr(remapped, name, value.index_select(0, index))
        return remapped

    def build_decode_attention_layout(
        self,
        *,
        block_ids_by_request: Sequence[Sequence[int]],
        pool_slots: Sequence[int],
    ) -> NPUDecodeAttentionLayout:
        """Build request-local prefix/suffix geometry for NPU eager attention."""

        if self.execution.kind != "decode":
            raise BeamExecutionContractError("attention layout requires a Decode plan")
        batch = self.execution.batch_size
        if len(block_ids_by_request) != batch or len(pool_slots) != batch:
            raise BeamExecutionContractError(
                "attention inputs need one block table and pool slot per request"
            )
        blocks: list[tuple[int, ...]] = []
        for request_index, request_blocks in enumerate(block_ids_by_request):
            normalized = tuple(request_blocks)
            if not normalized or any(
                isinstance(block, bool) or not isinstance(block, int) or block < 0
                for block in normalized
            ):
                raise BeamExecutionContractError(
                    f"block_ids_by_request[{request_index}] is invalid"
                )
            blocks.append(normalized)
        slots = tuple(pool_slots)
        if any(isinstance(slot, bool) or not isinstance(slot, int) or slot < 0 for slot in slots):
            raise BeamExecutionContractError("pool slots must be non-negative integers")

        step = self.execution.decode_step
        assert step is not None
        row_blocks = tuple(
            blocks[request_index] for request_index in self.execution.row_request_indices
        )
        return NPUDecodeAttentionLayout(
            request_ids=self.execution.request_ids,
            row_request_indices=self.execution.row_request_indices,
            row_beam_indices=self.execution.row_beam_indices,
            query_start_locations=tuple(range(self.execution.num_execution_rows + 1)),
            prefix_sequence_lengths=self.execution.expand_per_request(
                self.execution.num_prompt_tokens_by_request,
                name="prefix sequence lengths",
            ),
            suffix_sequence_lengths=(step + 1,) * self.execution.num_execution_rows,
            prefix_block_tables=row_blocks,
            pool_slots=slots,
        )


@dataclass(frozen=True, slots=True)
class NPUEagerFirstDecodeHandoff:
    """Device-resident Prefill winners consumed by Decode step zero."""

    prefill: BeamBatchExecutionView
    winner_token_ids: torch.Tensor

    @classmethod
    def from_prefill(
        cls,
        prefill_plan: NPUEagerStagePlan,
        winner_token_ids: torch.Tensor,
    ) -> NPUEagerFirstDecodeHandoff:
        execution = prefill_plan.execution
        expected = (execution.batch_size, execution.output_width)
        if execution.kind != "prefill":
            raise BeamExecutionContractError("winner handoff requires a Prefill plan")
        if tuple(winner_token_ids.shape) != expected:
            raise BeamExecutionContractError(f"Prefill winner_token_ids must have shape {expected}")
        return cls(execution, winner_token_ids)

    def bind_first_decode(
        self,
        decode_plan: NPUEagerStagePlan,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        execution = decode_plan.execution
        if execution.kind != "decode" or execution.decode_step != 0:
            raise BeamExecutionContractError(
                "Prefill winner handoff can bind only Decode step zero"
            )
        self.prefill.assert_same_logical_batch(execution)
        decode_plan.prepare_decode_inputs(
            input_ids,
            positions,
            self.winner_token_ids,
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
    "NPUDecodeAttentionLayout",
    "NPUEagerFirstDecodeHandoff",
    "NPUEagerStagePlan",
    "validate_npu_eager_capabilities",
]
