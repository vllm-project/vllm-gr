# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Bind native attention pages and recurrent states for CUDA Beam Decode."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.beam.hybrid_cache import hybrid_block_budget, hybrid_group_spec
from vllm_gr.v1.worker.cuda.gr_inputs import bind_beam_decode_state

if TYPE_CHECKING:
    from vllm_gr.v1.beam.dispatch import GRStageMetadata
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import GPUBeamStageRunner


@dataclass(slots=True)
class HybridDecodeBatch:
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    tables: tuple[torch.Tensor, ...]
    slots: tuple[torch.Tensor, ...]


def _kernel_block_sizes(worker: Any) -> list[int]:
    sizes = getattr(worker, "_kernel_block_sizes", None)
    return worker.kernel_block_sizes if sizes is None else sizes


def validate_hybrid_cache(worker: Any, metadata: GRStageMetadata) -> None:
    groups = worker.kv_cache_config.kv_cache_groups
    view = BeamBatchExecutionView.from_stage_metadata(metadata)
    steps = metadata.beam_params.max_tokens - 1
    if steps:
        blocks = metadata.hybrid_beam_blocks
        prompt_blocks = metadata.hybrid_prompt_blocks
        if blocks is None or prompt_blocks is None:
            raise ValueError("Hybrid Beam dispatch is missing native cache tables")
        for length, private, prompt in zip(
            view.num_prompt_tokens_by_request, blocks, prompt_blocks, strict=True
        ):
            _, expected = hybrid_block_budget(
                groups, prompt_tokens=length, decode_steps=steps, beam_width=view.beam_width
            )
            if tuple(map(len, private)) != expected or any(
                i >= worker.kv_cache_config.num_blocks
                for group in (*private, *prompt)
                for i in group
            ):
                raise ValueError("Hybrid Beam dispatch has incompatible cache blocks")
    for group, kernel_size in zip(groups, _kernel_block_sizes(worker), strict=True):
        spec = hybrid_group_spec(group)
        if isinstance(spec, AttentionSpec) and spec.block_size % kernel_size:
            raise ValueError(
                "Hybrid Beam requires attention pages aligned to the native kernel block size"
            )


def _copy_blocks(
    tensors: Iterable[torch.Tensor], source: torch.Tensor, destination: torch.Tensor
) -> None:
    # Snapshot each layer's sources before writing any destinations. Parents
    # can repeat or swap positions across the same set of physical blocks.
    for tensor in {tensor.data_ptr(): tensor for tensor in tensors}.values():
        tensor.index_copy_(0, destination, tensor.index_select(0, source))


def _kernel_blocks(ids: torch.Tensor, fragments: int) -> torch.Tensor:
    if fragments == 1:
        return ids
    offsets = torch.arange(fragments, dtype=ids.dtype, device=ids.device)
    return (ids.unsqueeze(-1) * fragments + offsets).flatten(-2)


def _attention_decode_blocks(
    caches: tuple[torch.Tensor, ...],
    native_ids: Sequence[int],
    private_blocks: torch.Tensor,
    parents: torch.Tensor,
    *,
    block_size: int,
    prompt_length: int,
    step: int,
    fragments: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    shared_pages, prompt_tail = divmod(prompt_length, block_size)
    live = private_blocks[: parents.numel()]
    relative_pos = prompt_tail + step
    preceding_pages = cdiv(relative_pos, block_size)
    if preceding_pages:
        destinations = live[:, :preceding_pages].reshape(-1).long()
        sources = (
            private_blocks.index_select(0, parents)[:, :preceding_pages].reshape(-1).long()
            if step
            else torch.full_like(destinations, native_ids[shared_pages])
        )
        _copy_blocks(
            caches, _kernel_blocks(sources, fragments), _kernel_blocks(destinations, fragments)
        )

    page, offset = divmod(relative_pos, block_size)
    write_blocks = live[:, page].long()
    if not offset:
        for cache in caches:
            cache.index_fill_(0, _kernel_blocks(write_blocks, fragments), 0)

    shared = private_blocks.new_tensor(list(native_ids[:shared_pages])).expand(parents.numel(), -1)
    table = _kernel_blocks(torch.cat((shared, live), dim=1), fragments)
    return table, write_blocks * block_size + offset


def _recurrent_decode_blocks(
    caches: tuple[tuple[torch.Tensor, ...], ...],
    native_ids: Sequence[int],
    private_blocks: torch.Tensor,
    parents: torch.Tensor,
    *,
    prompt_length: int,
    step: int,
    spec: MambaSpec,
) -> torch.Tensor:
    width = parents.numel()
    table = private_blocks[:width]
    targets = table.flatten().long()
    initial = 0 if spec.mamba_cache_mode == "none" else (prompt_length - 1) // spec.block_size
    sources = (
        private_blocks.flatten().long().index_select(0, parents)
        if step
        else torch.full_like(targets, native_ids[initial])
    )
    _copy_blocks((state for layer_states in caches for state in layer_states), sources, targets)
    if spec.mamba_cache_mode == "align":
        # The native builder indexes by context length. Each beam keeps a
        # single recurrent state throughout Decode, including page boundaries.
        return table.expand(width, cdiv(prompt_length + step + 1, spec.block_size))
    return table


def prepare_hybrid_decode(
    owner: GPUBeamStageRunner, metadata: GRStageMetadata
) -> HybridDecodeBatch:
    """Fork native caches using device-resident Beam parents and build native metadata."""
    worker = owner.worker
    step = metadata.decode_step
    blocks = metadata.hybrid_beam_blocks
    prompt_blocks = metadata.hybrid_prompt_blocks
    if step is None or blocks is None or prompt_blocks is None:
        raise ValueError("Hybrid Beam Decode requires native cache block IDs")
    view = BeamBatchExecutionView.from_stage_metadata(metadata)
    width = view.input_width
    device = owner.beam.state.device
    lengths_cpu = torch.tensor(view.num_prompt_tokens_by_request, dtype=torch.int32)
    lengths_cpu = (lengths_cpu + step + 1).repeat_interleave(width)
    lengths = lengths_cpu.to(device)
    owner.device_positions = lengths.long() - 1
    bind_beam_decode_state(metadata, owner)

    # Inactive rows still get distinct writable state. Their candidates are
    # masked by the Beam state selector.
    parents = owner.beam.state.parent_ids[: view.batch_size, :width]
    parents = parents.clamp(0, view.beam_width - 1).long()
    recurrent_slots = torch.full((width,), -1, dtype=torch.long, device=device)
    layers = worker.compilation_config.static_forward_context
    tables = []
    all_slots = []
    for group_id, (group, kernel_size) in enumerate(
        zip(worker.kv_cache_config.kv_cache_groups, _kernel_block_sizes(worker), strict=True)
    ):
        spec = hybrid_group_spec(group)
        caches = tuple(layers[name].kv_cache for name in group.layer_names)
        member_tables = []
        member_slots = []
        for index, length in enumerate(view.num_prompt_tokens_by_request):
            native_ids = prompt_blocks[index][group_id]
            private_ids = blocks[index][group_id]
            private = torch.tensor(private_ids, dtype=torch.int32, device=device)
            private = private.view(view.beam_width, -1)
            if isinstance(spec, MambaSpec):
                table = _recurrent_decode_blocks(
                    caches,
                    native_ids,
                    private,
                    parents[index],
                    prompt_length=length,
                    step=step,
                    spec=spec,
                )
                slots = recurrent_slots
            else:
                table, slots = _attention_decode_blocks(
                    caches,
                    native_ids,
                    private,
                    parents[index],
                    block_size=spec.block_size,
                    prompt_length=length,
                    step=step,
                    fragments=spec.block_size // kernel_size,
                )
            member_tables.append(table)
            member_slots.append(slots)
        cols = max(table.shape[1] for table in member_tables)
        tables.append(
            torch.cat(
                [torch.nn.functional.pad(t, (0, cols - t.shape[1])) for t in member_tables], dim=0
            )
        )
        all_slots.append(torch.cat(member_slots))
    return HybridDecodeBatch(lengths, lengths_cpu, tuple(tables), tuple(all_slots))


def build_hybrid_attention(
    worker: Any,
    batch: HybridDecodeBatch,
    positions: torch.Tensor,
    *,
    max_seq_len: int,
    graph_state_ids: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    rows = positions.numel()
    query_cpu = torch.arange(rows + 1, dtype=torch.int32)
    query = query_cpu.to(positions.device)
    is_prefilling = torch.zeros(rows, dtype=torch.bool)
    attention: dict[str, Any] = {}
    slot_mappings: dict[str, torch.Tensor] = {}
    for group_id, (table, slots) in enumerate(zip(batch.tables, batch.slots, strict=True)):
        common = CommonAttentionMetadata(
            query_start_loc=query,
            query_start_loc_cpu=query_cpu,
            seq_lens=batch.seq_lens,
            seq_lens_cpu_upper_bound=batch.seq_lens_cpu,
            num_reqs=rows,
            num_actual_tokens=rows,
            max_query_len=1,
            max_seq_len=max_seq_len,
            block_table_tensor=table,
            slot_mapping=slots,
            positions=positions,
            is_prefilling=is_prefilling,
        )
        for attn_group in worker.attn_groups[group_id]:
            builder = attn_group.get_metadata_builder()
            spec = attn_group.kv_cache_spec
            if (
                isinstance(spec, MambaSpec)
                and spec.mamba_cache_mode == "align"
                and hasattr(builder, "mamba_aligned_state_indices")
            ):
                # Each beam's private state occupies every logical column.
                # V2's native builder otherwise expects indices computed from
                # its own request-major block table before build().
                native_indices = builder.mamba_aligned_state_indices
                builder.mamba_aligned_state_indices = table[:, :1]
                try:
                    built = builder.build(0, common)
                finally:
                    builder.mamba_aligned_state_indices = native_indices
            else:
                built = builder.build(0, common)
            if graph_state_ids is not None:
                built.is_beam_decode = True
                if group_id in graph_state_ids:
                    built.non_spec_state_indices_tensor = graph_state_ids[group_id]
            attention.update(dict.fromkeys(attn_group.layer_names, built))
            slot_mappings.update(dict.fromkeys(attn_group.layer_names, slots))
    return attention, slot_mappings
