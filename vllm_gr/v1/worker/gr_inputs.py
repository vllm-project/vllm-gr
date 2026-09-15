# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous scheduling CPU preparation and ordered device binding, independent of output consumption."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadataBuilder,
)
from vllm_gr.v1.engine.beam_search_v1 import GRStageMetadata

ERROR_DISPATCH_STATE = 6


@dataclass(frozen=True)
class PreparedGRInputs:
    metadata: GRStageMetadata
    binding: Any
    staging_slot: int
    layout: dict[str, list[int]]

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
        self, width: int, max_model_len: int, block_size: int, device: torch.device | str
    ) -> None:
        self.width, self.max_model_len, self.block_size = width, max_model_len, block_size
        self.max_blocks = (max_model_len + block_size - 1) // block_size
        sizes = {
            "positions": width,
            "query": 2,
            "sequence": 1,
            "prefix_lengths": 1,
            "blocks": self.max_blocks,
            "logits": width,
            "suffix_query": width + 1,
            "suffix_lengths": width,
            "suffix_offsets": width + 1,
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
        if metadata.decode_step is None or metadata.beam_params.beam_width != self.width:
            raise ValueError(
                "Asynchronous scheduling input preparation requires fixed-width Decode metadata"
            )
        if metadata.session_id != binding.session_id or slot not in (0, 1):
            raise ValueError("Asynchronous scheduling input binding or staging slot mismatch")
        prompt, step = metadata.num_prompt_tokens, metadata.decode_step
        if (
            type(step) is not int
            or step < 0
            or step > 1
            or prompt + step >= self.max_model_len
            or not isinstance(block_ids, (tuple, list))
            or any(not isinstance(group, (tuple, list)) for group in block_ids)
            or any(type(block) is not int or block < 0 for group in block_ids for block in group)
        ):
            raise ValueError(
                "Asynchronous scheduling inputs require supported CPU geometry and Host block IDs"
            )
        required = (prompt + self.block_size - 1) // self.block_size
        if len(block_ids) != 1 or len(block_ids[0]) < required:
            raise ValueError("Asynchronous scheduling request is missing prompt KV block IDs")
        layout = BeamAttentionMetadataBuilder.build_async_layout(
            width=self.width,
            prompt_len=prompt,
            decode_step=step,
            block_ids=block_ids[0][:required],
            max_blocks=self.max_blocks,
            query_start_loc_cpu=[0, self.width],
            seq_lens_cpu=[prompt + step + 1],
        )
        # Only the selected slot's previous H2D can block this CPU write.
        if self.uploaded[slot] and not self.h2d_done[slot].query():
            self.h2d_done[slot].synchronize()
        host = self.host[slot].numpy()
        for name, values in layout.items():
            host[self.slices[name]] = values
        prepared = PreparedGRInputs(metadata, binding, slot, layout)
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
        return {name: self.device[span] for name, span in self.slices.items()}


def bind_device_inputs(prepared: PreparedGRInputs, owner: Any) -> None:
    """Read the predecessor's device state on the compute stream, without D2H."""
    m, b = prepared.metadata, prepared.binding
    if b != owner.binding or m.session_id != owner.binding.session_id:
        raise ValueError(
            "Asynchronous scheduling prepared inputs belong to a different execution owner"
        )
    state = owner.beam.state
    row = slice(b.state_row, b.state_row + 1)
    step = m.decode_step
    if step is None:
        raise ValueError("Asynchronous scheduling device binding requires Decode metadata")
    initialized, finished = state.is_initialized[row], state.is_finished[row]
    error = state.error_code[row]
    count, actual_step = state.num_generated_tokens[row], state.num_decode_steps[row]
    mismatch = (
        ~finished & (error == 0) & (~initialized | (actual_step != step) | (count != step + 1))
    )
    error.copy_(torch.where(mismatch, ERROR_DISPATCH_STATE, error))
    running = initialized & ~finished & (error == 0)
    mask = running & state.active_mask[b.state_row]
    owner.execution_mask.copy_(mask)
    owner.input_ids.copy_(torch.where(mask, state.input_ids[b.state_row], 0))
    owner.positions.copy_(torch.where(mask, owner.device_positions, 0))
    owner.beam.session.context.decode_step_buf.copy_(torch.where(running, step, -1))
    owner.prefixes.fill_(-1)
    columns = min(step + 1, owner.prefixes.shape[1], state.sequence.shape[2])
    owner.prefixes[:, :columns].copy_(
        torch.where(mask[:, None], state.sequence[b.state_row, :, :columns], -1)
    )
    length = torch.where(running, step + 1, 1)
    owner.metadata.suffix_seqlens_kv_device.copy_(length.expand(state.beam_width))
    # The prebuilt query offsets also equal [0, 1, ..., W].
    offsets = owner.metadata.suffix_cu_seqlens_q_device
    owner.metadata.suffix_cu_seqlens_k_device.copy_(offsets * length)
