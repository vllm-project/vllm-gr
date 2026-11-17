# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Fixed-capacity device state for one or more physical Beam storage rows.

Execution currently admits one session. The leading capacity dimension does
not imply concurrent execution or ownership of a session/KV registry.
"""

from __future__ import annotations

import torch

from vllm_gr.config.models import MAX_BEAM_WIDTH
from vllm_gr.v1.beam.beam_attention_context import BeamDecisionBuffers


class BeamSearchState:
    """Persistent Beam tensors; views remain valid until their row is reset.

    ``sequence`` and ``beam_scores`` alias the existing decision storage.
    Active views are overwritten each step. Completed records and the append-
    only history survive until final consumption and reset. Parent indices
    are request-local: a record of length n refers to history[n-2, parent],
    unless n == 1, in which case it refers to the implicit prefill root.

    Reset must be ordered after all users on the caller's stream (or through
    explicit events). This storage class neither waits nor releases KV slots.
    """

    #: Fixed-state tensors that survive every reset; they hold the request
    #: configuration rather than per-step progress.
    CONFIGURATION_TENSORS = frozenset(
        {
            "max_new_tokens_tensor",
            "stop_token_ids",
            "num_stop_tokens",
            "ignore_stop_tokens",
        }
    )
    #: Tensors whose rows are reset to ``-inf`` instead of zero.
    NEGATIVE_INFINITY_TENSORS = frozenset({"beam_scores", "completed_scores"})
    #: Views of the decision storage. A reset covers the whole decision row,
    #: so these must not be written a second time at the narrower beam width.
    ALIASED_TENSORS = frozenset({"sequence", "beam_scores"})

    def __init__(
        self,
        decision: BeamDecisionBuffers,
        *,
        beam_width: int,
        max_new_tokens: int,
        stop_token_ids: tuple[int, ...] = (),
        ignore_stop_tokens: bool = False,
        candidate_width: int | None = None,
    ) -> None:
        for name, value in (("beam_width", beam_width), ("max_new_tokens", max_new_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if candidate_width is None:
            candidate_width = beam_width
        if (
            isinstance(candidate_width, bool)
            or not isinstance(candidate_width, int)
            or candidate_width < 1
        ):
            raise ValueError("candidate_width must be a positive integer")
        sequence, scores = decision.sequence, decision.beam_scores
        if sequence.ndim != 3 or sequence.dtype != torch.int32:
            raise ValueError("decision sequence must be int32 [batch, beams, steps]")
        batch, beams, steps = sequence.shape
        if batch < 1:
            raise ValueError("decision storage must contain at least one batch row")
        if not 1 <= beam_width <= min(beams, MAX_BEAM_WIDTH):
            raise ValueError("beam_width exceeds decision or CUDA selection capacity")
        if max_new_tokens > steps:
            raise ValueError("max_new_tokens exceeds decision sequence capacity")
        if (
            scores.shape != (batch, beams)
            or scores.dtype != torch.float32
            or scores.device != sequence.device
            or not scores.is_contiguous()
            or not sequence.is_contiguous()
        ):
            raise ValueError("decision storage must have compatible contiguous sequence/scores")
        if not isinstance(ignore_stop_tokens, bool):
            raise TypeError("ignore_stop_tokens must be bool")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < 2**31
            for token in stop_token_ids
        ) or len(set(stop_token_ids)) != len(stop_token_ids):
            raise ValueError("stop_token_ids must be unique non-negative int32 values")
        # Completed records are appended once per finishing candidate, and a
        # step visits ``candidate_width`` candidates per parent row.  Prefill
        # contributes at most one row's worth of candidates, every later step
        # at most ``beam_width`` rows' worth.  With K == W this is exactly the
        # historical ``W + (N-1) * W**2``.
        capacity = candidate_width + (max_new_tokens - 1) * beam_width * candidate_width
        if max(batch * capacity, batch * max_new_tokens * beam_width, sequence.numel()) >= 2**31:
            raise ValueError("Beam state exceeds int32 indexing capacity")

        self.decision = decision
        self.max_batch_size = batch
        self.beam_width = beam_width
        self.max_new_tokens = max_new_tokens
        self.stop_token_ids_config = stop_token_ids
        self.ignore_stop_tokens_config = ignore_stop_tokens
        self.completed_capacity = capacity
        self.candidate_width = candidate_width
        self.device = sequence.device
        self.sequence = sequence[:, :beam_width]
        self.beam_scores = scores[:, :beam_width]

        def empty(shape: tuple[int, ...], dtype: torch.dtype = torch.int32) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=self.device)

        beam_shape = (batch, beam_width)
        self.input_ids = empty(beam_shape)
        self.parent_ids = empty(beam_shape)
        self.active_mask = empty(beam_shape, torch.bool)
        self.sequence_lengths = empty(beam_shape)
        self.num_generated_tokens = empty((batch,))
        self.num_decode_steps = empty((batch,))
        self.num_active_beams = empty((batch,))
        self.num_completed = empty((batch,))
        self.is_initialized = empty((batch,), torch.bool)
        self.is_finished = empty((batch,), torch.bool)
        self.error_code = empty((batch,))
        self.history_token_ids = empty((batch, max_new_tokens, beam_width))
        self.history_parent_ids = empty((batch, max_new_tokens, beam_width))
        completed_shape = (batch, capacity)
        self.completed_token_ids = empty(completed_shape)
        self.completed_scores = empty(completed_shape, torch.float32)
        self.completed_parent_ids = empty(completed_shape)
        self.completed_lengths = empty(completed_shape)
        self.completed_finish_reasons = empty(completed_shape)
        self.completed_tie_break_indices = empty(completed_shape)
        self.max_new_tokens_tensor = torch.full(
            (batch,), max_new_tokens, dtype=torch.int32, device=self.device
        )
        # A nonempty physical stop table simplifies the fixed kernel ABI.
        self.stop_token_ids = empty((batch, max(1, len(stop_token_ids))))
        self.stop_token_ids.fill_(-1)
        if stop_token_ids:
            self.stop_token_ids.copy_(torch.tensor(stop_token_ids, device=self.device))
        self.num_stop_tokens = torch.full(
            (batch,), len(stop_token_ids), dtype=torch.int32, device=self.device
        )
        self.ignore_stop_tokens = torch.full(
            (batch,), ignore_stop_tokens, dtype=torch.bool, device=self.device
        )
        for index in range(batch):
            self.reset(index)

    def owned_tensors(self) -> tuple[torch.Tensor, ...]:
        """Unique newly allocated storage, excluding aliased decision buffers."""
        return tuple(
            value
            for name, value in vars(self).items()
            if isinstance(value, torch.Tensor) and name not in ("sequence", "beam_scores")
        )

    @property
    def device_memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.owned_tensors())

    def reset(self, state_index: int = 0) -> None:
        """Clear one physical row in place, preserving its fixed configuration."""
        if isinstance(state_index, bool) or not isinstance(state_index, int):
            raise TypeError("state_index must be an integer")
        if not 0 <= state_index < self.max_batch_size:
            raise ValueError("state_index exceeds state capacity")
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and name not in self.CONFIGURATION_TENSORS:
                row = value[state_index : state_index + 1]
                if name in self.NEGATIVE_INFINITY_TENSORS:
                    row.fill_(-float("inf"))
                else:
                    row.zero_()
