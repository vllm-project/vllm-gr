# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Backend owning the canonical six-table device representation."""

from __future__ import annotations

from enum import Enum, auto

import torch

from vllm_gr.constraints.backend import (
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
    build_constraint_output,
    validate_select_inputs,
)
from vllm_gr.ops.cuda.rec_constrained_topk import (
    _rec_constrained_topk_cuda_out_unchecked,
    rec_constrained_topk_cuda,
    rec_constrained_topk_cuda_out,
    validate_cuda_sort_capacity,
)


class _BackendState(Enum):
    INITIALIZING = auto()
    READY = auto()
    CLOSED = auto()


class CUDAConstraintBackend:
    """Own GPU-resident canonical tables and dispatch CUDA constrained Top-K."""

    __slots__ = (
        "_first_token_ids",
        "_info",
        "_prefix1_offsets",
        "_prefix1_pair_keys",
        "_prefix1_values",
        "_prefix2_value_offsets",
        "_prefix2_values",
        "_state",
    )

    def __init__(
        self,
        *,
        first_token_ids: torch.Tensor,
        prefix1_offsets: torch.Tensor,
        prefix1_values: torch.Tensor,
        prefix1_pair_keys: torch.Tensor,
        prefix2_value_offsets: torch.Tensor,
        prefix2_values: torch.Tensor,
        info: ConstraintBackendInfo,
    ) -> None:
        validate_cuda_sort_capacity(
            int(first_token_ids.numel()),
            info.table_metadata.max_prefix1_degree,
            info.table_metadata.max_prefix2_degree,
        )
        self._first_token_ids: torch.Tensor | None = first_token_ids
        self._prefix1_offsets: torch.Tensor | None = prefix1_offsets
        self._prefix1_values: torch.Tensor | None = prefix1_values
        self._prefix1_pair_keys: torch.Tensor | None = prefix1_pair_keys
        self._prefix2_value_offsets: torch.Tensor | None = prefix2_value_offsets
        self._prefix2_values: torch.Tensor | None = prefix2_values
        self._info = info
        self._state = _BackendState.INITIALIZING

    @property
    def info(self) -> ConstraintBackendInfo:
        return self._info

    def _tables(self) -> tuple[torch.Tensor, ...]:
        tables = (
            self._first_token_ids,
            self._prefix1_offsets,
            self._prefix1_values,
            self._prefix1_pair_keys,
            self._prefix2_value_offsets,
            self._prefix2_values,
        )
        if self._state is _BackendState.CLOSED or any(table is None for table in tables):
            raise RuntimeError("CUDA constraint backend is closed")
        return tuple(table for table in tables if table is not None)

    def _invoke(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
        *,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        out_wire_logprobs: torch.Tensor | None = None,
        out_prefill_status: torch.Tensor | None = None,
        out_prefill_payload: torch.Tensor | None = None,
        validate: bool = True,
        build_output: bool = True,
    ) -> ConstraintOutput | None:
        table_vocab_size = self._info.table_metadata.vocab_size
        vocab_offset = 0
        expected_logits_vocab_size = table_vocab_size
        if logits.shape[1] != table_vocab_size:
            from vllm_gr.ops.sliced_logits import get_vocab_slice_config

            offset, slice_size, _ = get_vocab_slice_config()
            if offset is None or slice_size is None or logits.shape[1] != slice_size:
                raise ValueError(
                    f"CUDA constraint logits vocab {logits.shape[1]} does not match "
                    f"table vocab_size {table_vocab_size} or the configured vocabulary slice"
                )
            if offset < 0 or offset + slice_size > table_vocab_size:
                raise ValueError("configured CUDA vocabulary slice is outside table vocabulary")
            vocab_offset = offset
            expected_logits_vocab_size = slice_size
        if validate:
            validate_select_inputs(
                logits,
                metadata,
                self._info,
                expected_logits_vocab_size=expected_logits_vocab_size,
                allow_strided_logits=True,
            )
            if logits.device.type != "cuda":
                raise ValueError(
                    f"CUDA constraint backend requires CUDA logits, got {logits.device}"
                )
        tables = self._tables()
        if validate:
            for table in tables:
                if table.device != logits.device:
                    raise RuntimeError(
                        "all CUDA constraint tables must share logits.device; "
                        f"got table={table.device}, logits={logits.device}"
                    )
        op_args = (
            logits,
            metadata.prefix_token_ids,
            *tables,
            metadata.temperatures,
            metadata.current_step,
            metadata.top_k,
            self._info.table_metadata.max_prefix1_degree,
            self._info.table_metadata.max_prefix2_degree,
        )
        if outputs is None:
            token_ids, logprobs, valid_mask, _ = rec_constrained_topk_cuda(
                *op_args,
                vocab_offset=vocab_offset,
                table_vocab_size=table_vocab_size,
            )
        else:
            token_ids, logprobs, valid_mask, valid_counts = outputs
            launch_out = (
                rec_constrained_topk_cuda_out
                if validate
                else _rec_constrained_topk_cuda_out_unchecked
            )
            launch_kwargs = {
                "vocab_offset": vocab_offset,
                "table_vocab_size": table_vocab_size,
                "out_wire_logprobs": out_wire_logprobs,
            }
            if not validate:
                launch_kwargs["out_prefill_status"] = out_prefill_status
                launch_kwargs["out_prefill_payload"] = out_prefill_payload
            launch_out(
                *op_args,
                token_ids,
                logprobs,
                valid_mask,
                valid_counts,
                **launch_kwargs,
            )
        if not build_output:
            return None
        return build_constraint_output(
            token_ids,
            logprobs,
            valid_mask,
            logits=logits,
            top_k=metadata.top_k,
        )

    def _select_for_readiness(
        self, logits: torch.Tensor, metadata: ConstraintMetadata
    ) -> ConstraintOutput:
        if self._state is not _BackendState.INITIALIZING:
            raise RuntimeError("CUDA readiness can only run while initializing")
        output = self._invoke(logits, metadata)
        assert output is not None
        return output

    def _mark_ready(self) -> None:
        if self._state is not _BackendState.INITIALIZING:
            raise RuntimeError("only an initializing CUDA constraint backend can become ready")
        self._state = _BackendState.READY

    def select(self, logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        if self._state is not _BackendState.READY:
            raise RuntimeError("CUDA constraint backend is not ready")
        output = self._invoke(logits, metadata)
        assert output is not None
        return output

    def select_out(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
        *,
        out_token_ids: torch.Tensor,
        out_logprobs: torch.Tensor,
        out_valid_mask: torch.Tensor,
        out_valid_counts: torch.Tensor,
        out_wire_logprobs: torch.Tensor | None = None,
    ) -> ConstraintOutput:
        """Write CUDA candidates into caller-owned fixed-address tensors."""

        if self._state is not _BackendState.READY:
            raise RuntimeError("CUDA constraint backend is not ready")
        output = self._invoke(
            logits,
            metadata,
            outputs=(
                out_token_ids,
                out_logprobs,
                out_valid_mask,
                out_valid_counts,
            ),
            out_wire_logprobs=out_wire_logprobs,
        )
        assert output is not None
        return output

    def select_out_unchecked(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
        *,
        out_token_ids: torch.Tensor,
        out_logprobs: torch.Tensor,
        out_valid_mask: torch.Tensor,
        out_valid_counts: torch.Tensor,
        out_wire_logprobs: torch.Tensor | None = None,
        out_prefill_status: torch.Tensor | None = None,
        out_prefill_payload: torch.Tensor | None = None,
    ) -> None:
        """Write to a prevalidated persistent Worker workspace.

        Initialization and ``prepare_cuda_prefill_constraint`` establish the
        device, dtype, table, shape, and capacity invariants.  Keeping only the
        backend state check here avoids repeating the complete public input
        validation between LM-Head and the CUDA TopK launch.
        """

        if self._state is not _BackendState.READY:
            raise RuntimeError("CUDA constraint backend is not ready")
        self._invoke(
            logits,
            metadata,
            outputs=(
                out_token_ids,
                out_logprobs,
                out_valid_mask,
                out_valid_counts,
            ),
            out_wire_logprobs=out_wire_logprobs,
            out_prefill_status=out_prefill_status,
            out_prefill_payload=out_prefill_payload,
            validate=False,
            build_output=False,
        )

    def close(self) -> None:
        if self._state is _BackendState.CLOSED:
            return
        self._first_token_ids = None
        self._prefix1_offsets = None
        self._prefix1_values = None
        self._prefix1_pair_keys = None
        self._prefix2_value_offsets = None
        self._prefix2_values = None
        self._state = _BackendState.CLOSED


__all__ = ["CUDAConstraintBackend"]
