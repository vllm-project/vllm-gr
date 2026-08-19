# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Data contracts for EngineCore-owned beam control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgspec

from vllm_gr.beam_output import BeamOutputSpec


class BeamControllerRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    beam_session_id: str
    external_request_id: str
    beam_width: int
    max_tokens: int
    prompt_token_ids: list[int]
    initial_tokens: list[int]
    prompt_length: int
    eos_token_id: int
    temperature: float = 0.0
    ignore_eos: bool = False
    include_stop_token: bool = False
    begin_token_id: int | None = None
    end_token_id: int | None = None
    catalog_strategy: str = "fill"
    constraint_backend: str = "trie"
    constraint_resource_id: str | None = None
    beam_control_mode: str = "engine_per_step_request"
    # None is reserved for mixed-version compatibility and means the legacy
    # full-candidate result contract.
    output_spec: BeamOutputSpec | None = None


@dataclass(frozen=True)
class BeamControllerStepOutput:
    """Raw EngineCore logprob row for one completed execution step."""

    token_ids: list[int]
    logprobs: list[float]
    ranks: list[int | None] | None = None
    decoded_tokens: list[str | None] | None = None
    error: str | None = None


@dataclass(frozen=True)
class BeamControllerFinalBeam:
    token_ids: list[int]
    decode_token_ids: list[int]
    cumulative_logprob: float
    logprobs: dict[str, list[Any]]
    finish_reason: str
    stop_reason: int | str | None


@dataclass(frozen=True)
class BeamControllerFinalBeamLite:
    token_ids: list[int]
    cumulative_logprob: float
    finish_reason: str
    stop_reason: int | str | None


@dataclass(frozen=True)
class BeamControllerMetrics:
    beam_search_overhead: float
    beam_search_prefill_time: float
    beam_search_decode_time: float
    beam_search_sort_time: float
    num_generation_tokens: int
    num_beams: int


@dataclass(frozen=True)
class BeamControllerFinalResult:
    beam_session_id: str
    external_request_id: str
    beams: list[BeamControllerFinalBeam]
    metrics: BeamControllerMetrics | None
    finish_reason: str
    error: str | None = None
    error_code: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "beam_final_full",
            "version": 1,
            "beam_session_id": self.beam_session_id,
            "external_request_id": self.external_request_id,
            "beams": [
                {
                    "token_ids": beam.token_ids,
                    "decode_token_ids": beam.decode_token_ids,
                    "cumulative_logprob": beam.cumulative_logprob,
                    "logprobs": beam.logprobs,
                    "finish_reason": beam.finish_reason,
                    "stop_reason": beam.stop_reason,
                }
                for beam in self.beams
            ],
            "metrics": (
                {
                    "beam_search_overhead": self.metrics.beam_search_overhead,
                    "beam_search_prefill_time": self.metrics.beam_search_prefill_time,
                    "beam_search_decode_time": self.metrics.beam_search_decode_time,
                    "beam_search_sort_time": self.metrics.beam_search_sort_time,
                    "num_generation_tokens": self.metrics.num_generation_tokens,
                    "num_beams": self.metrics.num_beams,
                }
                if self.metrics is not None
                else None
            ),
            "finish_reason": self.finish_reason,
            "error": self.error,
            "error_code": self.error_code,
        }

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> BeamControllerFinalResult:
        result_type = payload.get("type")
        if result_type == "beam_final_lite":
            return BeamControllerFinalResultLite.from_wire(payload)  # type: ignore[return-value]
        if result_type not in (None, "beam_final_full"):
            raise ValueError(f"unknown beam final result type {result_type!r}")
        if result_type is not None and payload.get("version") != 1:
            raise ValueError(f"unsupported beam final result version {payload.get('version')!r}")
        metrics_payload = payload.get("metrics")
        metrics = BeamControllerMetrics(**metrics_payload) if metrics_payload is not None else None
        return cls(
            beam_session_id=payload["beam_session_id"],
            external_request_id=payload["external_request_id"],
            beams=[BeamControllerFinalBeam(**beam) for beam in payload.get("beams", [])],
            metrics=metrics,
            finish_reason=payload["finish_reason"],
            error=payload.get("error"),
            error_code=payload.get("error_code"),
        )


@dataclass(frozen=True)
class BeamControllerFinalResultLite:
    beam_session_id: str
    external_request_id: str
    beams: list[BeamControllerFinalBeamLite]
    metrics: BeamControllerMetrics | None
    finish_reason: str
    error: str | None = None
    error_code: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "beam_final_lite",
            "version": 1,
            "beam_session_id": self.beam_session_id,
            "external_request_id": self.external_request_id,
            "beams": [
                {
                    "token_ids": beam.token_ids,
                    "cumulative_logprob": beam.cumulative_logprob,
                    "finish_reason": beam.finish_reason,
                    "stop_reason": beam.stop_reason,
                }
                for beam in self.beams
            ],
            "metrics": (
                {
                    "beam_search_overhead": self.metrics.beam_search_overhead,
                    "beam_search_prefill_time": self.metrics.beam_search_prefill_time,
                    "beam_search_decode_time": self.metrics.beam_search_decode_time,
                    "beam_search_sort_time": self.metrics.beam_search_sort_time,
                    "num_generation_tokens": self.metrics.num_generation_tokens,
                    "num_beams": self.metrics.num_beams,
                }
                if self.metrics is not None
                else None
            ),
            "finish_reason": self.finish_reason,
            "error": self.error,
            "error_code": self.error_code,
        }

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> BeamControllerFinalResultLite:
        if payload.get("type") != "beam_final_lite":
            raise ValueError(f"not a Beam FinalResultLite payload: {payload.get('type')!r}")
        if payload.get("version") != 1:
            raise ValueError(f"unsupported beam final result version {payload.get('version')!r}")
        metrics_payload = payload.get("metrics")
        metrics = BeamControllerMetrics(**metrics_payload) if metrics_payload is not None else None
        return cls(
            beam_session_id=payload["beam_session_id"],
            external_request_id=payload["external_request_id"],
            beams=[BeamControllerFinalBeamLite(**beam) for beam in payload.get("beams", [])],
            metrics=metrics,
            finish_reason=payload["finish_reason"],
            error=payload.get("error"),
            error_code=payload.get("error_code"),
        )


@dataclass(frozen=True)
class BeamControllerDecision:
    completed_step: int
    next_tokens: list[int]
    parent_beam_indices: list[int]
    active_width: int
    finished: bool
    finish_reason: str | None = None
    final_beams: list[BeamControllerFinalBeam] | None = None
    error: str | None = None
