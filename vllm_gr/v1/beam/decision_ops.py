# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Operators selected at runner bind time for canonical Beam candidates."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def workspace_candidate_width(runner: Any, beam_width: int) -> int:
    """Resolve K for a workspace lookup, falling back to the legacy K == W."""

    from vllm_gr.config import GRConfig, resolve_candidate_width

    try:
        config = GRConfig.from_vllm_config(getattr(runner, "vllm_config", None))
    except (AttributeError, TypeError, ValueError):
        return beam_width
    if config.beam_max_width is None:
        return beam_width
    try:
        return resolve_candidate_width(config)
    except ValueError:
        return beam_width


def reorder_sessions(session: Any, sessions: tuple[Any, ...], indices: Any, counts: Any) -> None:
    for row, item in enumerate(sessions):
        item.step_complete(indices[row], layout_prevalidated=True)


@dataclass(frozen=True)
class DecisionOps:
    implementation_id: str
    group: Callable[..., Any] | None
    final: Callable[..., Any] | None
    alignment: int
    reorder: Callable[..., None] = reorder_sessions
    equal_width: bool = False
    fallback: DecisionOps | None = None

    def select(
        self,
        scores: Any,
        tokens: Any,
        probs: Any,
        sequence: Any,
        step: int,
        width: int,
        result_width: int,
        last: bool,
        strict: bool,
    ) -> tuple[Any, ...]:
        if self.equal_width and tokens.shape[1] != width:
            raise NotImplementedError(
                "Beam decision requires candidate_width == beam_width, got "
                f"candidate_width={tokens.shape[1]}, beam_width={width}"
            )
        if self.group is None or self.final is None:
            if strict or self.fallback is None:
                raise RuntimeError("Required fused Beam decision operators are unavailable")
            return self.fallback.select(
                scores, tokens, probs, sequence, step, width, result_width, last, True
            )
        try:
            if last and result_width % self.alignment == 0:
                ids, parents, log_probs, seq = self.final(
                    scores, tokens, probs, sequence, step, result_width
                )
                return ids, parents, log_probs, None, seq, result_width
            ids, parents, log_probs, prefix, seq = self.group(
                scores, tokens, probs, sequence, step, width
            )
            return ids, parents, log_probs, prefix, seq, width
        except Exception:
            if strict or self.fallback is None:
                raise
            logger.warning(
                "Beam operator %s failed; using %s",
                self.implementation_id,
                self.fallback.implementation_id,
                exc_info=True,
            )
            return self.fallback.select(
                scores, tokens, probs, sequence, step, width, result_width, last, True
            )


def reference_ops() -> DecisionOps:
    from vllm_gr.ops.onerec_decode import (
        _beam_search_group_torch,
        _beam_search_rec_final_select_torch,
    )

    return DecisionOps(
        "torch.reference", _beam_search_group_torch, _beam_search_rec_final_select_torch, 32
    )
