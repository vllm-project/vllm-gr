# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared types for vllm_gr engine patches.

Defines BeamRequestStepUpdate
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Final

import msgspec
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

GR_PROTOCOL_VERSION: Final = 1
GR_REQUEST_METADATA_KEY: Final = "vllm_gr"
GR_REQUEST_OUTPUT_KEY: Final = "vllm_gr_request_output"


class GRRequestStage(str, Enum):
    NEW = "new"
    PREFILL_RUNNING = "prefill_running"
    PREFILL_BOOTSTRAPPED = "prefill_bootstrapped"
    DECODE_RUNNING = "decode_running"
    FINAL_READY = "final_ready"
    RELEASED = "released"

    def can_transition_to(self, next_stage: "GRRequestStage") -> bool:
        if self is GRRequestStage.NEW:
            return next_stage in (
                GRRequestStage.PREFILL_RUNNING,
                GRRequestStage.RELEASED,
            )
        if self is GRRequestStage.PREFILL_RUNNING:
            return next_stage in (
                GRRequestStage.PREFILL_BOOTSTRAPPED,
                GRRequestStage.FINAL_READY,
                GRRequestStage.RELEASED,
            )
        if self is GRRequestStage.PREFILL_BOOTSTRAPPED:
            return next_stage in (
                GRRequestStage.DECODE_RUNNING,
                GRRequestStage.FINAL_READY,
                GRRequestStage.RELEASED,
            )
        if self is GRRequestStage.DECODE_RUNNING:
            return next_stage in (
                GRRequestStage.DECODE_RUNNING,
                GRRequestStage.FINAL_READY,
                GRRequestStage.RELEASED,
            )
        if self is GRRequestStage.FINAL_READY:
            return next_stage is GRRequestStage.RELEASED
        return next_stage is GRRequestStage.RELEASED


class GRRequestMetadata(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """Request-wide metadata carried by the single native request."""

    request_id: str
    beam_width: int
    max_tokens: int
    prefix_len: int
    continue_token_id: int
    finish_token_id: int
    num_return_sequences: int = 1
    return_logprobs: bool = False
    constraint_digest: str | None = None
    protocol_version: int = GR_PROTOCOL_VERSION


class GRSchedulerOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """Per-dispatch O(B) metadata attached to SchedulerOutput."""

    request_id: str
    stage: GRRequestStage
    step: int
    device_step: int
    is_prefill: bool
    is_last_prefill: bool = False
    is_final: bool = False
    prefix_block_ids: list[list[int]] | None = None
    protocol_version: int = GR_PROTOCOL_VERSION


class GRSequenceOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    token_ids: list[int]
    parent_ids: list[int]
    score: float
    logprobs: list[float] | None = None
    finish_reason: str = "length"
    stop_token_id: int | None = None


class GRRequestOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """Final output produced by the only Beam-state D2H copy."""

    request_id: str
    outputs: list[GRSequenceOutput]
    step: int
    device_step: int
    stage: GRRequestStage = GRRequestStage.FINAL_READY
    counters: dict[str, int] | None = None
    protocol_version: int = GR_PROTOCOL_VERSION


class GRModelRunnerOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """O(B) control output; Beam data exists only in final_output."""

    request_id: str
    token_id: int
    finished: bool
    step: int
    device_step: int
    stage: GRRequestStage
    final_output: GRRequestOutput | None = None
    error: str | None = None
    protocol_version: int = GR_PROTOCOL_VERSION

# ---------------------------------------------------------------------------
# BeamRequestStepUpdate — Improved Beam handling by a single Beam request
# ---------------------------------------------------------------------------


class BeamRequestStepUpdate(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Single logical message for all branches of a decode step."""

    session_id: str
    parent_beam_ids: list[str]
    child_beam_ids: list[str]
    beam_tokens: list[list[int]]
    pruned_ids: list[str]
    prefix_len: int
    beam_width: int
    sampling_params: SamplingParams
    eos_token_id: int | None = None
    client_index: int = 0
    current_wave: int = 0
    priority: int = 0
    data_parallel_rank: int | None = None
    lora_request: LoRARequest | None = None
    cache_salt: str | None = None
    trace_headers: Mapping[str, str] | None = None
    # Numeric parent beam indices for incremental decode KV reordering
    parent_beam_indices: list[int] | None = None
    # Maximum number of decode steps for KV pool pre-allocation
    max_decode_steps: int = 0
    # 0 = constrained prefill, 1/2 = constrained decode step for Constraint
    # Tables (issue #246).  Default 0 also means "not a constrained decode", so
    # omit_defaults keeps legacy code streams byte-identical.
    constraint_step: int = 0
