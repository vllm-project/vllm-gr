# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""NPU Backend owning one six-table device representation per Trie."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum, auto
from typing import Any

import torch

from vllm_gr.constraints.backend import (
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
    build_constraint_output,
    validate_select_inputs,
)
from vllm_gr.constraints.npu_stage_vocab import NPUStageVocab
from vllm_gr.constraints.validity import (
    constraint_candidate_counts,
    valid_mask_from_counts,
)
from vllm_gr.ops.npu.rec_constrained_topk import NPUConstraintOp


class _BackendState(Enum):
    INITIALIZING = auto()
    READY = auto()
    CLOSED = auto()


class NPUConstraintBackend:
    """Own per-Trie tables, shared stage maps, one real op, and state."""

    __slots__ = (
        "_first_token_ids",
        "_info",
        "_op",
        "_prefix1_offsets",
        "_prefix1_pair_keys",
        "_prefix1_values",
        "_prefix2_value_offsets",
        "_prefix2_values",
        "_stage_vocab",
        "_state",
        "_trie_tables",
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
        op: NPUConstraintOp,
        stage_vocab: NPUStageVocab | None = None,
        trie_tables: tuple[tuple[torch.Tensor, ...], ...] | None = None,
    ) -> None:
        self._first_token_ids: torch.Tensor | None = first_token_ids
        self._prefix1_offsets: torch.Tensor | None = prefix1_offsets
        self._prefix1_values: torch.Tensor | None = prefix1_values
        self._prefix1_pair_keys: torch.Tensor | None = prefix1_pair_keys
        self._prefix2_value_offsets: torch.Tensor | None = prefix2_value_offsets
        self._prefix2_values: torch.Tensor | None = prefix2_values
        self._info = info
        self._op: NPUConstraintOp | None = op
        self._stage_vocab = stage_vocab
        self._trie_tables = trie_tables
        if trie_tables is not None and len(trie_tables) != info.table_metadata.trie_count:
            raise ValueError("NPU tables must contain one six-table group per Trie")
        self._state = _BackendState.INITIALIZING

    @property
    def info(self) -> ConstraintBackendInfo:
        return self._info

    @property
    def stage_vocab(self) -> NPUStageVocab | None:
        return self._stage_vocab

    def _tables(self, trie_id: int = 0) -> tuple[torch.Tensor, ...]:
        tables = (
            self._first_token_ids,
            self._prefix1_offsets,
            self._prefix1_values,
            self._prefix1_pair_keys,
            self._prefix2_value_offsets,
            self._prefix2_values,
        )
        if self._state is _BackendState.CLOSED or any(table is None for table in tables):
            raise RuntimeError("NPU constraint backend is closed")
        if self._trie_tables is not None:
            return self._trie_tables[trie_id]
        if trie_id != 0:
            raise RuntimeError("NPU constraint tables are missing the selected Trie")
        return tuple(table for table in tables if table is not None)

    def _active_stage_vocab(self, logits: torch.Tensor) -> NPUStageVocab | None:
        stage_vocab = self._stage_vocab
        if stage_vocab is not None and (
            logits.dim() != 2
            or logits.shape[1] != stage_vocab.vocab_size
            or stage_vocab.vocab_size >= self._info.table_metadata.vocab_size
        ):
            return None
        return stage_vocab

    def _invoke(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
        *,
        trie_id: int = 0,
        validate: bool = True,
    ) -> ConstraintOutput:
        stage_vocab = self._active_stage_vocab(logits)
        if validate:
            validate_select_inputs(
                logits,
                metadata,
                self._info,
                expected_logits_vocab_size=stage_vocab.vocab_size
                if stage_vocab is not None
                else None,
            )
        tables = self._tables(trie_id)
        op_tables = stage_vocab.tables_for_trie(trie_id) if stage_vocab is not None else tables
        if self._op is None:
            raise RuntimeError("NPU constraint backend is closed")
        for table in op_tables:
            if table.device != logits.device:
                raise RuntimeError(
                    "all NPU constraint tables must share logits.device; "
                    f"got table={table.device}, logits={logits.device}"
                )

        # The fused operator consumes [B, W, max_steps]. A singleton W is a
        # zero-copy view of the public flat [rows, 3] Backend metadata.
        prefixes = metadata.prefix_token_ids
        if stage_vocab is not None:
            prefixes = stage_vocab.local_prefixes(prefixes, metadata.current_step)
        sequence_group = prefixes.unsqueeze(1)
        # The fused kernel's bf16 load path reads stale unified-buffer data on
        # ascend910b and produces garbage logprobs, while its fp32 path is
        # correct; bf16 -> fp32 is a lossless bit extension, so widen first.
        # (Same normalization as vllm_gr.ops.npu.rec_constrained_topk.)
        op_logits = logits.to(torch.float32) if logits.dtype == torch.bfloat16 else logits
        token_ids, logprobs = self._op(
            op_logits,
            sequence_group,
            *op_tables,
            metadata.temperatures,
            metadata.current_step,
            metadata.top_k,
            self._info.table_metadata.max_prefix1_degree,
            self._info.table_metadata.max_prefix2_degree,
        )
        valid_counts = constraint_candidate_counts(
            metadata.prefix_token_ids,
            tables[0],
            tables[1],
            tables[3],
            tables[4],
            current_step=metadata.current_step,
            vocab_size=self._info.table_metadata.vocab_size,
        )
        valid_mask = valid_mask_from_counts(valid_counts, metadata.top_k)
        output = build_constraint_output(
            token_ids,
            logprobs,
            valid_mask,
            logits=logits,
            top_k=metadata.top_k,
        )
        if stage_vocab is not None:
            return ConstraintOutput(
                token_ids=stage_vocab.global_tokens(
                    output.token_ids, valid_mask, metadata.current_step
                ),
                logprobs=output.logprobs,
                valid_mask=valid_mask,
            )
        return output

    def _select_checked(
        self, logits: torch.Tensor, metadata: ConstraintMetadata
    ) -> ConstraintOutput:
        if self._info.table_metadata.trie_count == 1:
            return self._invoke(logits, metadata)
        stage_vocab = self._active_stage_vocab(logits)
        validate_select_inputs(
            logits,
            metadata,
            self._info,
            expected_logits_vocab_size=stage_vocab.vocab_size if stage_vocab is not None else None,
        )
        assert metadata.trie_ids is not None  # Validated above.
        # Checked public calls may synchronize. The Worker uses Host routing
        # from session admission through select_for_rows instead.
        roots = tuple(metadata.trie_ids.to(device="cpu").tolist())
        return self._select_rows(logits, metadata, roots)

    def _select_rows(
        self, logits: torch.Tensor, metadata: ConstraintMetadata, trie_ids: tuple[int, ...]
    ) -> ConstraintOutput:
        parts = []
        start = 0
        while start < len(trie_ids):
            root = trie_ids[start]
            stop = start + 1
            while stop < len(trie_ids) and trie_ids[stop] == root:
                stop += 1
            temperatures = metadata.temperatures
            part = ConstraintMetadata(
                prefix_token_ids=metadata.prefix_token_ids[start:stop],
                temperatures=temperatures
                if temperatures.numel() == 1
                else temperatures[start:stop],
                current_step=metadata.current_step,
                top_k=metadata.top_k,
            )
            parts.append(self._invoke(logits[start:stop], part, trie_id=root, validate=False))
            start = stop
        if len(parts) == 1:
            return parts[0]
        return ConstraintOutput(
            token_ids=torch.cat([part.token_ids for part in parts], dim=0),
            logprobs=torch.cat([part.logprobs for part in parts], dim=0),
            valid_mask=torch.cat([part.valid_mask for part in parts], dim=0),
        )

    def select_for_rows(
        self, logits: torch.Tensor, metadata: ConstraintMetadata, trie_ids: tuple[int, ...]
    ) -> ConstraintOutput:
        """Select prevalidated Worker rows without reading device root IDs.

        There is one Host Trie ID per physical logits row, including when a
        Worker chunk splits a request. Adjacent equal roots use one native call.
        """
        if self._state is not _BackendState.READY:
            raise RuntimeError("NPU constraint backend is not ready or is closed")
        if not isinstance(trie_ids, tuple) or len(trie_ids) != logits.shape[0]:
            raise ValueError("Host trie_ids must be a tuple with one ID per logits row")
        if any(
            type(root) is not int or not 0 <= root < self._info.table_metadata.trie_count
            for root in trie_ids
        ):
            raise ValueError("trie_id is outside the constraint table")
        # Retain structural ABI checks but trust the Host root IDs validated
        # at admission. Never invoke a device reduction on metadata.trie_ids.
        unrooted = replace(metadata, trie_ids=None)
        info = replace(self._info, table_metadata=replace(self._info.table_metadata, trie_count=1))
        stage_vocab = self._active_stage_vocab(logits)
        validate_select_inputs(
            logits,
            unrooted,
            info,
            expected_logits_vocab_size=stage_vocab.vocab_size if stage_vocab is not None else None,
        )
        return self._select_rows(logits, unrooted, trie_ids)

    def _select_for_readiness(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput:
        if self._state is not _BackendState.INITIALIZING:
            raise RuntimeError("NPU readiness can only run while initializing")
        return self._select_checked(logits, metadata)

    def _mark_ready(self) -> None:
        if self._state is not _BackendState.INITIALIZING:
            raise RuntimeError("only an initializing NPU constraint backend can become ready")
        self._state = _BackendState.READY

    def select(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput:
        if self._state is _BackendState.CLOSED:
            raise RuntimeError("NPU constraint backend is closed")
        if self._state is not _BackendState.READY:
            raise RuntimeError("NPU constraint backend is not ready")
        return self._select_checked(logits, metadata)

    def prepare_runner(self, model_config: Any) -> None:
        """The Ascend operator is resolved eagerly at load, so nothing is deferred."""
        del model_config

    def close(self) -> None:
        if self._state is _BackendState.CLOSED:
            return
        self._first_token_ids = None
        self._prefix1_offsets = None
        self._prefix1_values = None
        self._prefix1_pair_keys = None
        self._prefix2_value_offsets = None
        self._prefix2_values = None
        self._op = None
        self._stage_vocab = None
        self._trie_tables = None
        self._state = _BackendState.CLOSED


__all__ = ["NPUConstraintBackend"]
