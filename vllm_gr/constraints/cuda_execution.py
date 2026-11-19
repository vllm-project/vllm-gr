# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""CUDA execution for the legacy compiled constraint tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_gr.constraints.execution import (
    ConstrainedTopKTensorResult,
    _ConstrainedTopKExecutionBase,
)

if TYPE_CHECKING:
    from vllm_gr.constraints.compiler import ConstraintTableResource


class CUDAConstrainedTopKExecution(_ConstrainedTopKExecutionBase):
    """CUDA constrained Top-K backed by the Triton lookup/gather kernel."""

    implementation_id = "vllm_gr.constraints.execution.CUDAConstrainedTopKExecution"

    def load_tables(self, resource: "ConstraintTableResource") -> None:
        from vllm_gr.ops.cuda.rec_constrained_topk import validate_cuda_sort_capacity

        validate_cuda_sort_capacity(
            int(resource.first_token_ids.size),
            resource.max_prefix1_degree,
            resource.max_prefix2_degree,
        )
        super().load_tables(resource)

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def _default_device(self) -> torch.device:
        return torch.device("cuda")

    def _vocab_window(self, logits: torch.Tensor) -> tuple[int, int]:
        """Resolve ``(offset, full_vocab_size)`` for full or sliced logits."""

        if self._vocab_size is None:
            raise RuntimeError("loaded constraint tables are missing vocab_size")
        if logits.shape[1] == self._vocab_size:
            return 0, self._vocab_size

        from vllm_gr.ops.sliced_logits import get_vocab_slice_config

        offset, slice_size, _ = get_vocab_slice_config()
        if offset is None or slice_size is None:
            raise ValueError(
                f"CUDA constraint logits vocab {logits.shape[1]} differs from table "
                f"vocab_size {self._vocab_size}, but vocab slicing is not configured"
            )
        if logits.shape[1] != slice_size:
            raise ValueError(
                f"CUDA constraint logits vocab {logits.shape[1]} must equal configured "
                f"vocab_slice_size {slice_size}"
            )
        if offset < 0 or offset + slice_size > self._vocab_size:
            raise ValueError(
                "configured CUDA vocabulary slice is outside the constraint table "
                f"vocabulary: offset={offset}, size={slice_size}, "
                f"table_vocab_size={self._vocab_size}"
            )
        return offset, self._vocab_size

    def _expected_logits_vocab_size(self, logits: torch.Tensor) -> int:
        self._vocab_window(logits)
        return int(logits.shape[1])

    def _call_op(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_gr.ops.cuda.rec_constrained_topk import rec_constrained_topk_cuda

        vocab_offset, table_vocab_size = self._vocab_window(logits)
        tokens, logprobs, _, _ = rec_constrained_topk_cuda(
            logits,
            sequence_group,
            self._tables["first_token_ids"],
            self._tables["prefix1_offsets"],
            self._tables["prefix1_values"],
            self._tables["prefix1_pair_keys"],
            self._tables["prefix2_value_offsets"],
            self._tables["prefix2_values"],
            temperatures,
            current_step,
            top_k,
            self._max_degrees["max_prefix1_degree"],
            self._max_degrees["max_prefix2_degree"],
            vocab_offset=vocab_offset,
            table_vocab_size=table_vocab_size,
        )
        return tokens, logprobs

    def run_out(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
        *,
        out_token_ids: torch.Tensor,
        out_logprobs: torch.Tensor,
        out_valid_mask: torch.Tensor,
        out_valid_counts: torch.Tensor,
        out_wire_logprobs: torch.Tensor | None = None,
    ) -> ConstrainedTopKTensorResult:
        """CUDA-only caller-owned output path used by Worker postprocessing."""

        return self._run_out_impl(
            logits,
            sequence_group,
            temperatures,
            current_step,
            top_k,
            out_token_ids=out_token_ids,
            out_logprobs=out_logprobs,
            out_valid_mask=out_valid_mask,
            out_valid_counts=out_valid_counts,
            out_wire_logprobs=out_wire_logprobs,
            validate=True,
        )

    def run_out_unchecked(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
        *,
        out_token_ids: torch.Tensor,
        out_logprobs: torch.Tensor,
        out_valid_mask: torch.Tensor,
        out_valid_counts: torch.Tensor,
        out_wire_logprobs: torch.Tensor | None = None,
        out_prefill_status: torch.Tensor | None = None,
        out_prefill_payload: torch.Tensor | None = None,
    ) -> ConstrainedTopKTensorResult:
        """Launch into a prewarmed Worker workspace without repeated validation."""

        return self._run_out_impl(
            logits,
            sequence_group,
            temperatures,
            current_step,
            top_k,
            out_token_ids=out_token_ids,
            out_logprobs=out_logprobs,
            out_valid_mask=out_valid_mask,
            out_valid_counts=out_valid_counts,
            out_wire_logprobs=out_wire_logprobs,
            out_prefill_status=out_prefill_status,
            out_prefill_payload=out_prefill_payload,
            validate=False,
        )

    def _run_out_impl(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
        *,
        out_token_ids: torch.Tensor,
        out_logprobs: torch.Tensor,
        out_valid_mask: torch.Tensor,
        out_valid_counts: torch.Tensor,
        out_wire_logprobs: torch.Tensor | None,
        out_prefill_status: torch.Tensor | None = None,
        out_prefill_payload: torch.Tensor | None = None,
        validate: bool,
    ) -> ConstrainedTopKTensorResult:
        """Shared checked/unchecked caller-owned launch implementation."""

        if self._digest is None:
            raise RuntimeError("constraint tables not loaded; call load_tables first")
        if validate:
            self._validate_runtime_inputs(
                logits,
                sequence_group,
                temperatures,
                current_step,
                top_k,
            )
        from vllm_gr.ops.cuda.rec_constrained_topk import (
            _rec_constrained_topk_cuda_out_unchecked,
            rec_constrained_topk_cuda_out,
        )

        vocab_offset, table_vocab_size = self._vocab_window(logits)
        args = (
            logits,
            sequence_group,
            self._tables["first_token_ids"],
            self._tables["prefix1_offsets"],
            self._tables["prefix1_values"],
            self._tables["prefix1_pair_keys"],
            self._tables["prefix2_value_offsets"],
            self._tables["prefix2_values"],
            temperatures,
            current_step,
            top_k,
            self._max_degrees["max_prefix1_degree"],
            self._max_degrees["max_prefix2_degree"],
            out_token_ids,
            out_logprobs,
            out_valid_mask,
            out_valid_counts,
        )
        if validate:
            rec_constrained_topk_cuda_out(
                *args,
                vocab_offset=vocab_offset,
                table_vocab_size=table_vocab_size,
                out_wire_logprobs=out_wire_logprobs,
            )
        else:
            _rec_constrained_topk_cuda_out_unchecked(
                *args,
                vocab_offset=vocab_offset,
                table_vocab_size=table_vocab_size,
                out_wire_logprobs=out_wire_logprobs,
                out_prefill_status=out_prefill_status,
                out_prefill_payload=out_prefill_payload,
            )
        return ConstrainedTopKTensorResult(
            top_tokens=out_token_ids,
            top_logprobs=out_logprobs,
            valid_mask=out_valid_mask,
        )
