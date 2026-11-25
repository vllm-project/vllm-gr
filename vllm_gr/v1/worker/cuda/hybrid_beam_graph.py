# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""CUDA graph bindings for native single-token MLA and KDA Decode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.logger import init_logger
from vllm.model_executor.offloader import get_offloader
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.beam.hybrid_cache import hybrid_group_spec
from vllm_gr.v1.worker.cuda.beam_decode_graph import (
    BeamDecodeCaptureUnsupported,
    BeamDecodeGraph,
    BeamDecodeGraphSignature,
    get_beam_decode_graph,
)
from vllm_gr.v1.worker.cuda.hybrid_beam_cache import (
    HybridDecodeBatch,
    _kernel_block_sizes,
    build_hybrid_attention,
    prepare_hybrid_decode,
)
from vllm_gr.v1.worker.gr_context import _get_gr_config

if TYPE_CHECKING:
    from vllm_gr.v1.beam.dispatch import GRStageMetadata
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import GPUBeamStageRunner

logger = init_logger(__name__)


class HybridGraphInputs:
    def __init__(self, worker: Any, rows: int) -> None:
        device = worker.device
        max_len = worker.model_config.max_model_len
        self.input_ids = torch.zeros(rows, dtype=torch.int32, device=device)
        self.positions = torch.zeros(rows, dtype=torch.int64, device=device)
        self.state_ids: dict[int, torch.Tensor] = {}
        tables = []
        for group_id, (group, kernel_size) in enumerate(
            zip(worker.kv_cache_config.kv_cache_groups, _kernel_block_sizes(worker), strict=True)
        ):
            spec = hybrid_group_spec(group)
            if isinstance(spec, MambaSpec):
                self.state_ids[group_id] = torch.zeros(rows, dtype=torch.int32, device=device)
                columns = 1 if spec.mamba_cache_mode == "none" else cdiv(max_len, spec.block_size)
            else:
                columns = cdiv(max_len, spec.block_size) * (spec.block_size // kernel_size)
            tables.append(torch.zeros((rows, columns), dtype=torch.int32, device=device))
        self.batch = HybridDecodeBatch(
            seq_lens=torch.ones(rows, dtype=torch.int32, device=device),
            seq_lens_cpu=torch.ones(rows, dtype=torch.int32),
            tables=tuple(tables),
            slots=tuple(torch.full((rows,), -1, dtype=torch.int64, device=device) for _ in tables),
        )
        # Dummy state 0 and slot -1 disable KDA and MLA cache writes during
        # warmup and capture. A replay binds the admitted private block IDs.
        self.attention, self.slots = build_hybrid_attention(
            worker, self.batch, self.positions, max_seq_len=max_len, graph_state_ids=self.state_ids
        )

    def load(self, owner: GPUBeamStageRunner, batch: HybridDecodeBatch) -> None:
        self.input_ids.copy_(owner.input_ids)
        self.positions.copy_(owner.positions)
        self.batch.seq_lens.copy_(batch.seq_lens)
        for index, (target, source, slot, actual_slots) in enumerate(
            zip(self.batch.tables, batch.tables, self.batch.slots, batch.slots, strict=True)
        ):
            target.zero_()
            target[:, : source.shape[1]].copy_(source)
            slot.copy_(actual_slots)
            if index in self.state_ids:
                self.state_ids[index].copy_(source[:, 0])


class HybridBeamDecodeGraph(BeamDecodeGraph):
    def __init__(self, worker: Any, width: int) -> None:
        super().__init__(worker, width)
        self.inputs: dict[BeamDecodeGraphSignature, HybridGraphInputs] = {}
        self.refused: set[BeamDecodeGraphSignature] = set()

    def discard_selected(self) -> None:
        super().discard_selected()
        self.inputs.pop(self.signature, None)
        self.refused.discard(self.signature)

    def clear(self) -> None:
        super().clear()
        self.inputs.clear()
        self.refused.clear()

    def run(self, owner: GPUBeamStageRunner, batch: HybridDecodeBatch) -> torch.Tensor | None:
        if self.signature in self.refused:
            return None
        inputs = self.inputs.get(self.signature)
        if inputs is None:
            inputs = HybridGraphInputs(self.worker, self.signature.num_model_rows)

            def forward(capture: bool) -> torch.Tensor:
                return owner.hooks.forward_hidden(
                    inputs.input_ids, inputs.positions, inputs.attention, inputs.slots, capture
                )

            try:
                self.capture(forward, persistent_install=False)
            except BeamDecodeCaptureUnsupported as error:
                torch.accelerator.synchronize(self.worker.device)
                self.refused.add(self.signature)
                logger.warning("Hybrid Beam Decode graph refused for %s: %s", self.signature, error)
                return None
            self.inputs[self.signature] = inputs
        inputs.load(owner, batch)
        # This replay goes directly through the Beam graph entry, so retain the
        # native wrapper's ordering with any weight prefetch from the last forward.
        get_offloader().sync_prev_onload()
        hidden = self.replay()
        self.worker._beam_graph_full_dispatch_count = (
            getattr(self.worker, "_beam_graph_full_dispatch_count", 0) + 1
        )
        return hidden


def hybrid_decode_logits(
    owner: GPUBeamStageRunner, metadata: GRStageMetadata
) -> torch.Tensor | None:
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import _hidden_logits

    worker = owner.worker
    view = BeamBatchExecutionView.from_stage_metadata(metadata)
    batch = prepare_hybrid_decode(owner, metadata)
    cfg = _get_gr_config(worker)
    if (
        not worker.model_config.enforce_eager
        and cfg.beam_graph_enabled
        and view.batch_size in cfg.deployment.beam_batch_sizes
        and _supports_hybrid_graph(worker)
    ):
        graph = cast(
            HybridBeamDecodeGraph,
            get_beam_decode_graph(
                worker,
                batch_size=view.batch_size,
                beam_width=view.input_width,
                graph_type=HybridBeamDecodeGraph,
            ),
        )
        hidden = graph.run(owner, batch)
        if hidden is not None:
            owner.graph = graph
            return _hidden_logits(owner, hidden)
    owner.graph = None
    assert metadata.decode_step is not None
    owner.attention, owner.slot_mappings = build_hybrid_attention(
        worker,
        batch,
        owner.positions,
        max_seq_len=max(view.num_prompt_tokens_by_request) + metadata.decode_step + 1,
    )
    return owner._forward()


def _supports_hybrid_graph(worker: Any) -> bool:
    # The graph stages live lengths and GDN state indices in fixed device
    # buffers. Other native builders can require host-selected launch geometry
    # or different state metadata and use eager Decode.
    for group_id, group in enumerate(worker.kv_cache_config.kv_cache_groups):
        spec = hybrid_group_spec(group)
        if isinstance(spec, MambaSpec):
            if spec.mamba_type is not MambaAttentionBackendEnum.GDN_ATTN:
                return False
        elif any(
            attention.backend.get_name() != "TRITON_MLA"
            for attention in worker.attn_groups[group_id]
        ):
            return False
    return True
