# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared types for vllm_gr engine patches.

Defines BeamRequestStepUpdate
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import re
from typing import Final

import msgspec
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

GR_PROTOCOL_VERSION: Final = 1
GR_REQUEST_METADATA_KEY: Final = "vllm_gr"
GR_REQUEST_OUTPUT_KEY: Final = "vllm_gr_request_output"
# Native vLLM requires an O(B) sampled-token result for every generation
# stage. This placeholder is transport-only: persistent SamplingParams ignore
# EOS/stops and expose FINAL_ONLY output, while the next real Beam token stays
# in the device-resident state.
GR_CONTROL_TOKEN_ID: Final = 0
_CANONICAL_SHA256_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_BARE_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def _validate_protocol_version(version: object) -> None:
    if type(version) is not int or version != GR_PROTOCOL_VERSION:
        raise ValueError(
            "unsupported persistent GR protocol_version "
            f"{version!r}; expected {GR_PROTOCOL_VERSION}"
        )


def _require_non_empty_request_id(request_id: object) -> None:
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("persistent GR request_id must be a non-empty string")


def canonicalize_constraint_digest(digest: object) -> str:
    """Return the request-wire form of an installed Constraint Table digest."""
    if isinstance(digest, str) and _BARE_SHA256_RE.fullmatch(digest) is not None:
        return f"sha256:{digest}"
    if isinstance(digest, str) and _CANONICAL_SHA256_RE.fullmatch(digest) is not None:
        return digest
    raise ValueError(
        "persistent GR constraint_digest must be 'sha256:' followed by "
        "64 lowercase hex digits"
    )


def _validate_non_negative_step(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"persistent GR {name} must be a non-negative integer")


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
    frozen=True,
):
    """Request-wide metadata carried by the single native request."""

    request_id: str
    beam_width: int
    generation_steps: int
    prefix_len: int
    result_width: int = 1
    constraint_digest: str = ""
    protocol_version: int = GR_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _validate_protocol_version(self.protocol_version)
        _require_non_empty_request_id(self.request_id)
        if type(self.beam_width) is not int or self.beam_width < 1:
            raise ValueError("persistent GR beam_width must be a positive integer")
        if type(self.generation_steps) is not int or self.generation_steps < 1:
            raise ValueError(
                "persistent GR generation_steps must be a positive integer"
            )
        if type(self.prefix_len) is not int or self.prefix_len < 0:
            raise ValueError("persistent GR prefix_len must be a non-negative integer")
        if (
            type(self.result_width) is not int
            or not 1 <= self.result_width <= self.beam_width
        ):
            raise ValueError(
                "persistent GR result_width must be in [1, beam_width]"
            )
        if canonicalize_constraint_digest(self.constraint_digest) != self.constraint_digest:
            raise ValueError("persistent GR constraint_digest must use canonical wire form")

    # Compatibility names used by the scheduler while its per-step sidecar is
    # migrated independently. They are derived, not serialized request state.
    @property
    def max_tokens(self) -> int:
        return self.generation_steps

    @property
    def num_return_sequences(self) -> int:
        return self.result_width

    @property
    def continue_token_id(self) -> int:
        return GR_CONTROL_TOKEN_ID

    @property
    def finish_token_id(self) -> int:
        return GR_CONTROL_TOKEN_ID


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

    def __post_init__(self) -> None:
        _validate_protocol_version(self.protocol_version)
        _require_non_empty_request_id(self.request_id)
        _validate_non_negative_step(self.step, "step")
        _validate_non_negative_step(self.device_step, "device_step")


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

    def __post_init__(self) -> None:
        _validate_protocol_version(self.protocol_version)
        _require_non_empty_request_id(self.request_id)
        _validate_non_negative_step(self.step, "step")
        _validate_non_negative_step(self.device_step, "device_step")
        if self.stage is not GRRequestStage.FINAL_READY:
            raise ValueError("persistent GR final output stage must be final_ready")


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

    def __post_init__(self) -> None:
        _validate_protocol_version(self.protocol_version)
        _require_non_empty_request_id(self.request_id)
        _validate_non_negative_step(self.step, "step")
        _validate_non_negative_step(self.device_step, "device_step")
        if self.final_output is not None:
            if not self.finished:
                raise ValueError("persistent GR final output requires finished=true")
            if self.final_output.request_id != self.request_id:
                raise ValueError(
                    "persistent GR final output request_id does not match control output"
                )
            if self.final_output.protocol_version != self.protocol_version:
                raise ValueError(
                    "persistent GR final output protocol_version does not match "
                    "control output"
                )
            if self.final_output.step != self.step:
                raise ValueError(
                    "persistent GR final output step does not match control output"
                )
            if self.final_output.device_step != self.device_step:
                raise ValueError(
                    "persistent GR final output device_step does not match control output"
                )
            if self.final_output.stage is not self.stage:
                raise ValueError(
                    "persistent GR final output stage does not match control output"
                )
        if self.stage is GRRequestStage.FINAL_READY and (
            not self.finished or self.final_output is None
        ):
            raise ValueError(
                "persistent GR final_ready control output requires finished=true "
                "and a final output"
            )

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
