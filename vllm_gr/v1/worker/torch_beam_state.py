# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-resident full-vocabulary Beam state, without accelerator-specific ops.

EOS candidates leave the active frontier immediately. Only the best W completed
sequences need to be retained: a discarded completion can never enter final
Top-W. Inactive physical rows remain allocated for fixed-shape attention, but
their scores stay -inf and they cannot produce candidates in a later stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class TorchBeamOutput:
    sequences: torch.Tensor
    scores: torch.Tensor
    parents: torch.Tensor
    lengths: torch.Tensor
    stopped: torch.Tensor
    stop_token_ids: torch.Tensor


class TorchBeamState:
    """Fixed-capacity active and completed frontiers for a logical batch.

    Shape and stage validation uses host metadata only. All score-dependent
    selection, including an entirely completed request, stays on the device.
    """

    def __init__(
        self,
        *,
        beam_width: int,
        max_steps: int,
        prompt_lengths: Sequence[int],
        stop_token_ids: Sequence[int],
        eos_token_id: int | None,
        length_penalty: float,
        device: torch.device | str,
        ignore_stop_tokens: bool = False,
    ) -> None:
        if beam_width < 1 or max_steps < 1 or not prompt_lengths:
            raise ValueError("Beam state requires positive width, steps and batch size")
        if any(length < 1 for length in prompt_lengths):
            raise ValueError("Beam state requires nonempty prompts")
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.batch_size = len(prompt_lengths)
        self.stage_index = -1
        self.length_penalty = length_penalty
        self.eos_token_id = eos_token_id
        self.prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.float32, device=device)
        self.stop_ids = tuple(stop_token_ids) if not ignore_stop_tokens else ()
        shape = (self.batch_size, beam_width)
        history_shape = (*shape, max_steps)
        self.sequences = torch.zeros(history_shape, dtype=torch.int32, device=device)
        self.parents = torch.zeros_like(self.sequences)
        self.scores = torch.full(shape, -torch.inf, dtype=torch.float32, device=device)
        self.valid = torch.zeros(shape, dtype=torch.bool, device=device)
        self.token_ids = torch.zeros(shape, dtype=torch.int32, device=device)
        self.local_parents = torch.zeros(shape, dtype=torch.int64, device=device)
        self.completed_sequences = torch.zeros_like(self.sequences)
        self.completed_parents = torch.zeros_like(self.parents)
        self.completed_scores = torch.full_like(self.scores, -torch.inf)
        self.completed_lengths = torch.zeros(shape, dtype=torch.int32, device=device)
        self.completed_stop_ids = torch.full(shape, -1, dtype=torch.int32, device=device)

    def _normalized(
        self, scores: torch.Tensor, lengths: torch.Tensor, stop_ids: torch.Tensor
    ) -> torch.Tensor:
        lengths = lengths.to(torch.float32)
        if self.eos_token_id is not None:
            lengths = lengths - (stop_ids == self.eos_token_id).to(torch.float32)
        denominator = (self.prompt_lengths[:, None] + lengths).pow(self.length_penalty)
        return scores / denominator

    @staticmethod
    def _take_history(history: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return history.gather(1, indices.unsqueeze(2).expand(-1, -1, history.shape[2]))

    @torch.inference_mode()
    def advance(
        self,
        token_ids: torch.Tensor,
        logprobs: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        stage_index: int,
    ) -> None:
        if stage_index != self.stage_index + 1 or stage_index >= self.max_steps:
            raise ValueError("Beam stages must be contiguous and within the generation budget")
        batch, width = self.batch_size, self.beam_width
        parents_per_item = 1 if stage_index == 0 else width
        if (
            token_ids.ndim != 2
            or token_ids.shape[0] != batch * parents_per_item
            or token_ids.shape[1] < width
            or logprobs.shape != token_ids.shape
            or valid_mask.shape != token_ids.shape
        ):
            raise ValueError("Beam candidates must have B*P rows and a shared K >= W")
        candidate_width = token_ids.shape[1]
        candidate_count = parents_per_item * candidate_width
        tokens = token_ids.reshape(batch, candidate_count).to(torch.int32)
        scores = logprobs.reshape(batch, parents_per_item, candidate_width).to(torch.float32)
        valid = valid_mask.reshape(batch, parents_per_item, candidate_width)
        if stage_index:
            scores = scores + self.scores[:, :, None]
            valid = valid & self.valid[:, :, None]
        scores = scores.reshape(batch, candidate_count)
        valid = valid.reshape(batch, candidate_count) & torch.isfinite(scores)
        scores = scores.masked_fill(~valid, -torch.inf)
        local_parents = (
            torch.arange(parents_per_item, device=tokens.device, dtype=torch.int64)
            .repeat_interleave(candidate_width)
            .expand(batch, -1)
        )
        sequences = self._take_history(self.sequences, local_parents)
        histories = self._take_history(self.parents, local_parents)
        sequences[:, :, stage_index].copy_(tokens)
        histories[:, :, stage_index].copy_(local_parents.to(torch.int32))
        terminal = torch.zeros_like(valid)
        for stop_id in self.stop_ids:
            terminal |= tokens == stop_id
        terminal &= valid

        # Existing completions precede new completions, preserving the native
        # stable ordering when normalized scores are exactly equal.
        completed_scores = torch.cat(
            (self.completed_scores, scores.masked_fill(~terminal, -torch.inf)), dim=1
        )
        completed_lengths = torch.cat(
            (self.completed_lengths, torch.full_like(tokens, stage_index + 1)), dim=1
        )
        completed_stops = torch.cat((self.completed_stop_ids, tokens), dim=1)
        completed_order = torch.argsort(
            self._normalized(completed_scores, completed_lengths, completed_stops),
            dim=1,
            descending=True,
            stable=True,
        )[:, :width]
        self.completed_sequences.copy_(
            self._take_history(
                torch.cat((self.completed_sequences, sequences), dim=1), completed_order
            )
        )
        self.completed_parents.copy_(
            self._take_history(
                torch.cat((self.completed_parents, histories), dim=1), completed_order
            )
        )
        self.completed_scores.copy_(completed_scores.gather(1, completed_order))
        self.completed_lengths.copy_(completed_lengths.gather(1, completed_order))
        self.completed_stop_ids.copy_(completed_stops.gather(1, completed_order))

        active_scores = scores.masked_fill(terminal, -torch.inf)
        active_order = torch.argsort(active_scores, dim=1, descending=True, stable=True)[:, :width]
        self.scores.copy_(active_scores.gather(1, active_order))
        self.valid.copy_(torch.isfinite(self.scores))
        self.sequences.copy_(self._take_history(sequences, active_order))
        self.parents.copy_(self._take_history(histories, active_order))
        self.local_parents.copy_(local_parents.gather(1, active_order))
        self.local_parents.masked_fill_(~self.valid, 0)
        self.token_ids.copy_(tokens.gather(1, active_order))
        self.token_ids.masked_fill_(~self.valid, 0)
        self.stage_index = stage_index

    @torch.inference_mode()
    def finalize(self) -> TorchBeamOutput:
        if self.stage_index < 0:
            raise RuntimeError("Cannot finalize Beam state before Prefill")
        lengths = torch.cat(
            (
                self.completed_lengths,
                torch.full_like(self.completed_lengths, self.stage_index + 1),
            ),
            dim=1,
        )
        stops = torch.cat(
            (self.completed_stop_ids, torch.full_like(self.completed_stop_ids, -1)), dim=1
        )
        scores = torch.cat((self.completed_scores, self.scores), dim=1)
        order = torch.argsort(
            self._normalized(scores, lengths, stops), dim=1, descending=True, stable=True
        )[:, : self.beam_width]
        return TorchBeamOutput(
            self._take_history(torch.cat((self.completed_sequences, self.sequences), dim=1), order),
            scores.gather(1, order),
            self._take_history(torch.cat((self.completed_parents, self.parents), dim=1), order),
            lengths.gather(1, order),
            order < self.beam_width,
            stops.gather(1, order),
        )


@torch.inference_mode()
def reorder_beam_kv(pools: Sequence[Any], local_parents: torch.Tensor, *, decode_step: int) -> None:
    """Gather arbitrary request-local parents without CPU reads or fused-op ordering rules."""
    if local_parents.ndim != 2 or decode_step < 0:
        raise ValueError("KV reorder requires B*W local parent indices and a nonnegative step")
    batch, width = local_parents.shape
    for pool in pools:
        for name in ("unshared_key", "unshared_value"):
            cache = getattr(pool, name)
            if (
                cache.ndim != 5
                or cache.shape[0] < batch
                or cache.shape[1] != width
                or decode_step >= cache.shape[3]
            ):
                raise ValueError("Beam KV pool does not match the active batch geometry")
            written = cache[:batch, :, :, : decode_step + 1]
            indices = local_parents[:, :, None, None, None].expand_as(written)
            written.copy_(written.gather(1, indices))


__all__ = ["TorchBeamOutput", "TorchBeamState", "reorder_beam_kv"]
