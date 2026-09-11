# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Device-neutral contracts for persistent Beam Search V1 execution.

These types intentionally contain no torch, SchedulerOutput, or
ModelRunnerOutput references.  They are the narrow control-plane seam between
the future A3 scheduler integration and the A4 Worker-owned data path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from vllm_gr.config.models import MAX_BEAM_WIDTH


class GRStageKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class GRBeamSessionSpec:
    """Immutable B=1 session configuration consumed by the Worker.

    Phase A is a fixed-generation plan.  Stop-token handling is deliberately
    disabled until a device-resident completed-Beam pool is implemented.
    """

    session_id: str
    request_id: str
    beam_width: int
    max_tokens: int
    constraint_handle: str | None
    temperature: float = 1.0
    ignore_eos: bool = True
    early_stopping: bool = False
    ignore_stop_tokens: bool = True

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if isinstance(self.beam_width, bool) or not isinstance(self.beam_width, int):
            raise TypeError("beam_width must be an int")
        if not 1 <= self.beam_width <= MAX_BEAM_WIDTH:
            raise ValueError(f"beam_width must be in [1, {MAX_BEAM_WIDTH}], got {self.beam_width}")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise TypeError("max_tokens must be an int")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a finite non-negative number")
        if self.ignore_eos is not True:
            raise NotImplementedError(
                "beam_search_v1 persistent Beam requires ignore_eos=True; "
                "per-Beam EOS completion is not implemented"
            )
        if self.early_stopping is not False:
            raise NotImplementedError(
                "beam_search_v1 persistent Beam requires early_stopping=False; "
                "Phase A uses a fixed generation-stage plan"
            )
        if self.ignore_stop_tokens is not True:
            raise NotImplementedError(
                "beam_search_v1 persistent Beam requires ignore_stop_tokens=True; "
                "Phase A uses a fixed generation-stage plan"
            )


@dataclass(frozen=True, slots=True)
class GRWorkerStageSpec:
    """One fixed-plan Worker stage; Prefill is always stage zero."""

    session_id: str
    stage_index: int
    stage_kind: GRStageKind
    is_last_stage: bool

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if isinstance(self.stage_index, bool) or not isinstance(self.stage_index, int):
            raise TypeError("stage_index must be an int")
        if self.stage_index < 0:
            raise ValueError("stage_index must be non-negative")
        expected_kind = GRStageKind.PREFILL if self.stage_index == 0 else GRStageKind.DECODE
        if self.stage_kind is not expected_kind:
            raise ValueError(
                f"stage {self.stage_index} must use stage_kind={expected_kind.value!r}"
            )

    @property
    def budget_terminal(self) -> bool:
        """Compatibility name used by the architecture document."""

        return self.is_last_stage


@dataclass(frozen=True, slots=True)
class GRStageAck:
    """Lightweight control acknowledgement; never transports Beam data."""

    session_id: str
    stage_index: int
    enqueued: bool
    terminal: bool
    error: str | None = None


__all__ = [
    "GRBeamSessionSpec",
    "GRStageAck",
    "GRStageKind",
    "GRWorkerStageSpec",
]
