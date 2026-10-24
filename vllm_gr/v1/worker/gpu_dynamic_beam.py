# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA adapter for shared stage-width selection and request-local KV banks.

Selection stays outside model graphs. Each input width owns stable forward
buffers; the existing GPU owner retains control/D2H and retirement ownership.
"""

import weakref
from typing import Any

import torch

from vllm_gr.constraints.backend import ConstraintOutput
from vllm_gr.v1.beam.dynamic_beam import DynamicBeamState, prune_stage_pools
from vllm_gr.v1.worker.gr_inputs import GRInputBuffers


def uses_dynamic_beam(params: Any) -> bool:
    """Only search geometry requires dynamic state; output limits do not."""
    widths = params.dynamic_beam_widths
    return widths is not None and any(w != params.beam_width for w in widths)


class GPUDynamicBeam:
    def __init__(self, owner: Any) -> None:
        self.owner = weakref.proxy(owner)
        self.buffers: dict[int, dict[str, Any]] = {}
        self.candidates: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        self.reset()

    def reset(self) -> None:
        owner = self.owner
        state = owner.beam.state
        self.state = DynamicBeamState(
            owner.beam.session.context, len(owner.bindings), state.max_new_tokens
        )

    def activate(self, width: int) -> None:
        owner = self.owner
        if width not in self.buffers:
            worker, state = owner.worker, owner.beam.state
            batch = len(owner.bindings)
            rows = batch * width
            device = state.device
            ids = torch.empty(rows, dtype=torch.int32, device=device)
            self.buffers[width] = {
                "input_ids": ids,
                "positions": torch.empty(rows, dtype=torch.int64, device=device),
                "slots": torch.full((rows,), -1, dtype=torch.int64, device=device),
                "execution_mask": torch.empty(rows, dtype=torch.bool, device=device),
                "prefixes": torch.empty((rows, 3), dtype=torch.int32, device=device),
                "inputs": GRInputBuffers(
                    width,
                    worker.model_config.max_model_len,
                    worker.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size,
                    device,
                    max_batch=batch,
                ),
            }
        for name, value in self.buffers[width].items():
            setattr(owner, name, value)

    def candidate_buffers(self, rows: int, width: int) -> tuple[torch.Tensor, ...]:
        key = rows, width
        if key not in self.candidates:
            device = self.owner.beam.state.device
            self.candidates[key] = (
                torch.empty(key, dtype=torch.int32, device=device),
                torch.empty(key, dtype=torch.float32, device=device),
                torch.empty(key, dtype=torch.bool, device=device),
            )
        return self.candidates[key]

    def advance(self, view: Any, tensors: tuple[torch.Tensor, ...], last: bool) -> None:
        tokens, local_scores, mask = tensors
        finite = torch.isfinite(local_scores)
        bad = (mask & ~finite).reshape(view.batch_size, -1).any(dim=1)
        target = self.owner.beam.state
        errors = target.error_code[: view.batch_size]
        errors.copy_(torch.where(bad, 3, errors))  # ERROR_INVALID_CANDIDATE
        candidates = ConstraintOutput(tokens, local_scores, mask & finite)
        if view.kind == "prefill":
            self.state.initialize(candidates, view.output_width)
        else:
            self.state.select(
                candidates,
                view.input_width,
                view.output_width,
                view.stage_index,
                is_last=last,
            )
        # Publish into the existing capacity-sized control state. Input binding
        # reads only the live-width slice; addresses used by controls stay fixed.
        target = self.owner.beam.state
        batch, width = view.batch_size, view.output_width
        scores, sequence = self.state.scores, self.state.sequence
        assert scores is not None and sequence is not None
        valid = torch.isfinite(scores)
        target.active_mask[:batch].zero_()
        target.active_mask[:batch, :width].copy_(valid)
        target.input_ids[:batch, :width].copy_(sequence[:, :, view.stage_index])
        target.sequence[:batch, :width, : sequence.shape[2]].copy_(sequence)
        target.beam_scores[:batch, :width].copy_(scores)
        target.num_active_beams[:batch].copy_(valid.sum(dim=1))
        target.is_initialized[:batch].fill_(True)
        target.num_generated_tokens[:batch].fill_(view.stage_index + 1)
        target.num_decode_steps[:batch].fill_(view.stage_index)
        target.is_finished[:batch].fill_(last)

    def pack_final(self, output: Any) -> None:
        """Fill the existing compact payload, preserving its async lifecycle.

        ``length_penalty`` needs no separate ranking pass: V1 always finishes
        at the fixed horizon, so the native ratio
        ``(prompt + max_tokens) / effective_length`` is exactly 1 and the
        retained order is already the native ranking order.
        """
        state = self.state
        assert state.scores is not None and state.sequence is not None
        assert state.parent_history is not None
        width = output.options.num_return_sequences
        steps = output.layout.steps
        # Prefill-only sessions have no final select: filter/rank the complete
        # candidate set before truncating, just as the merged NPU terminal path.
        order = state.scores.argsort(dim=1, descending=True, stable=True)
        final_scores = state.scores.gather(1, order)
        history_order = order[:, :, None].expand_as(state.sequence)
        final_sequence = state.sequence.gather(1, history_order)
        final_parents = state.parent_history.gather(1, history_order)
        for i in range(state.batch_size):
            start = 3 + i * output.layout.member_stride
            rows = output.device[start + 1 : start + output.layout.member_stride].view(
                output.layout.width, output.layout.row_stride
            )
            scores = final_scores[i, :width]
            output.device[start].copy_(torch.isfinite(scores).sum())
            rows[:width, :steps].copy_(final_sequence[i, :width, :steps])
            rows[:width, steps : 2 * steps].copy_(final_parents[i, :width, :steps])
            rows[:width, 2 * steps].copy_(scores.contiguous().view(torch.int32))
            rows[:width, 2 * steps + 1].fill_(steps)
            rows[:width, 2 * steps + 2].fill_(2)  # FINISH_LENGTH: fixed horizon
            rows[:width, 2 * steps + 3].fill_(-1)


def discard_dynamic_graphs(worker: Any, context: Any) -> None:
    """Drain-time disposal before an adapter's captured buffers are replaced."""
    graph = getattr(worker, "_beam_decode_graph", None)
    if graph is not None:
        graph.clear()
    worker._gr_dynamic_graphs_discarded = True
    prune_stage_pools(context, ())
