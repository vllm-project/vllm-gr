# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Asynchronous scheduling CPU preparation and ordered device binding, independent of output consumption."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm_gr.v1.attention.backends.beam_attn_metadata_async import (
    build_async_batch_layout,
)
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.engine.beam_search_v1 import GRStageMetadata

ERROR_DISPATCH_STATE = 6


@dataclass(frozen=True)
class PreparedGRInputs:
    metadata: GRStageMetadata
    binding: Any
    staging_slot: int
    layout: dict[str, list[int]]
    batch_view: BeamBatchExecutionView | None = None

    @property
    def identity(self) -> tuple[str, int, int]:
        m = self.metadata
        return m.session_id, m.owner_generation, m.dispatch_id


class GRInputBuffers:
    """Two pinned upload slots and one fixed-address compute view.

    A slot belongs to a physical dispatch until its output is consumed. Its
    H2D event additionally protects the CPU source even with a delayed copy.
    All uploads and device binding run on the same stream as Forward.
    """

    def __init__(
        self,
        width: int,
        max_model_len: int,
        block_size: int,
        device: torch.device | str,
        *,
        max_batch: int = 1,
    ) -> None:
        if type(max_batch) is not int or max_batch < 1:
            raise ValueError("max_batch must be a positive integer")
        self.width = width
        self.max_batch = max_batch
        self.max_model_len, self.block_size = max_model_len, block_size
        self.max_blocks = (max_model_len + block_size - 1) // block_size
        rows = max_batch * width
        sizes = {
            "positions": rows,
            "query": max_batch + 1,
            "sequence": max_batch,
            "prefix_lengths": max_batch,
            "blocks": max_batch * self.max_blocks,
            "logits": rows,
            "suffix_query": rows + 1,
            "suffix_lengths": rows,
            "suffix_offsets": rows + 1,
        }
        self.host = torch.empty((2, sum(sizes.values())), dtype=torch.int32, pin_memory=True)
        self.device = torch.empty(sum(sizes.values()), dtype=torch.int32, device=device)
        self.slices: dict[str, slice] = {}
        offset = 0
        for name, size in sizes.items():
            self.slices[name] = slice(offset, offset + size)
            offset += size
        self.h2d_done = [torch.Event(), torch.Event()]
        self.uploaded = [False, False]
        self.identities: list[tuple[str, int, int] | None] = [None, None]

    def prepare_cpu(
        self, metadata: GRStageMetadata, binding: Any, *, block_ids: list[list[int]], slot: int
    ) -> PreparedGRInputs:
        return self.prepare_cpu_batch(
            metadata,
            (binding,),
            block_ids_by_request=(block_ids,),
            slot=slot,
        )

    def prepare_cpu_batch(
        self,
        metadata: GRStageMetadata,
        bindings: tuple[Any, ...],
        *,
        block_ids_by_request: tuple[list[list[int]], ...],
        slot: int,
    ) -> PreparedGRInputs:
        view = BeamBatchExecutionView.from_stage_metadata(metadata)
        if view.kind != "decode" or view.input_width != self.width:
            raise ValueError(
                "Asynchronous scheduling input preparation requires fixed-width Decode metadata"
            )
        if (
            view.batch_size > self.max_batch
            or len(bindings) != view.batch_size
            or len(block_ids_by_request) != view.batch_size
            or tuple(binding.native_request_id for binding in bindings) != view.request_ids
            or any(binding.session_id != view.session_id for binding in bindings)
            or slot not in (0, 1)
        ):
            raise ValueError("Asynchronous scheduling input binding or staging slot mismatch")
        step = view.decode_step
        if (
            type(step) is not int
            or step < 0
            or step >= metadata.beam_params.max_tokens - 1
            or any(
                prompt + step >= self.max_model_len for prompt in view.num_prompt_tokens_by_request
            )
        ):
            raise ValueError(
                "Asynchronous scheduling inputs require supported CPU geometry and Host block IDs"
            )
        prompt_blocks: list[list[int]] = []
        for prompt, block_ids in zip(view.num_prompt_tokens_by_request, block_ids_by_request):
            if (
                not isinstance(block_ids, (tuple, list))
                or len(block_ids) != 1
                or not isinstance(block_ids[0], (tuple, list))
                or any(type(block) is not int or block < 0 for block in block_ids[0])
            ):
                raise ValueError("Asynchronous scheduling inputs require Host block IDs")
            required = (prompt + self.block_size - 1) // self.block_size
            if len(block_ids[0]) < required:
                raise ValueError("Asynchronous scheduling request is missing prompt KV block IDs")
            prompt_blocks.append(list(block_ids[0][:required]))
        layout = build_async_batch_layout(
            width=self.width,
            prompt_lengths=list(view.num_prompt_tokens_by_request),
            decode_step=step,
            block_ids_by_request=prompt_blocks,
            max_blocks=self.max_blocks,
        )
        # Only the selected slot's previous H2D can block this CPU write.
        if self.uploaded[slot] and not self.h2d_done[slot].query():
            self.h2d_done[slot].synchronize()
        host = self.host[slot].numpy()
        for name, values in layout.items():
            span = self.slices[name]
            host[span.start : span.start + len(values)] = values
        prepared = PreparedGRInputs(metadata, bindings, slot, layout, view)
        self.identities[slot] = prepared.identity
        return prepared

    def upload(self, prepared: PreparedGRInputs) -> dict[str, torch.Tensor]:
        slot = prepared.staging_slot
        if self.identities[slot] != prepared.identity:
            raise RuntimeError("Asynchronous scheduling staging identity changed before upload")
        self.device.copy_(self.host[slot], non_blocking=True)
        self.h2d_done[slot].record(torch.cuda.current_stream(self.device.device))
        self.uploaded[slot] = True
        # Fresh views are intentional: no layout/view cache participates.
        return {
            name: self.device[span.start : span.start + len(prepared.layout[name])]
            for name, span in self.slices.items()
        }


def bind_device_inputs(prepared: PreparedGRInputs, owner: Any) -> None:
    """Read the predecessor's device state on the compute stream, without D2H."""
    m = prepared.metadata
    bindings = prepared.binding if isinstance(prepared.binding, tuple) else (prepared.binding,)
    owner_bindings = getattr(owner, "bindings", (getattr(owner, "binding", None),))
    if bindings != owner_bindings or any(b.session_id != m.session_id for b in bindings):
        raise ValueError(
            "Asynchronous scheduling prepared inputs belong to a different execution owner"
        )
    state = owner.beam.state
    rows = owner.state_rows
    step = m.decode_step
    if step is None:
        raise ValueError("Asynchronous scheduling device binding requires Decode metadata")
    initialized, finished = state.is_initialized[rows], state.is_finished[rows]
    error = state.error_code[rows]
    count, actual_step = state.num_generated_tokens[rows], state.num_decode_steps[rows]
    mismatch = (
        ~finished & (error == 0) & (~initialized | (actual_step != step) | (count != step + 1))
    )
    dispatch_error = torch.where(mismatch, ERROR_DISPATCH_STATE, error)
    state.error_code[rows] = dispatch_error
    running = initialized & ~finished & (dispatch_error == 0)
    width = BeamBatchExecutionView.from_stage_metadata(m).input_width
    mask = running[:, None] & state.active_mask[rows, :width]
    flat_mask = mask.reshape(-1)
    owner.execution_mask.copy_(flat_mask)
    owner.input_ids.copy_(torch.where(flat_mask, state.input_ids[rows, :width].reshape(-1), 0))
    owner.positions.copy_(torch.where(flat_mask, owner.device_positions, 0))
    owner.beam.session.context.decode_step_buf.copy_(
        torch.where(running.all(), step, -1).reshape(1)
    )
    owner.prefixes.fill_(-1)
    columns = min(step + 1, owner.prefixes.shape[1], state.sequence.shape[2])
    owner.prefixes[:, :columns].copy_(
        torch.where(
            mask.reshape(-1, 1),
            state.sequence[rows, :width, :columns].reshape(-1, columns),
            -1,
        )
    )
    lengths = torch.where(running, step + 1, 1).repeat_interleave(width)
    owner.metadata.suffix_seqlens_kv_device.copy_(lengths)
    owner.metadata.suffix_cu_seqlens_k_device[0].zero_()
    torch.cumsum(
        lengths,
        dim=0,
        out=owner.metadata.suffix_cu_seqlens_k_device[1:],
    )
