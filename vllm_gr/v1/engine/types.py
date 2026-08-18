# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared types for vllm_gr engine patches.

Defines BeamRequestStepUpdate
"""

from __future__ import annotations

from collections.abc import Mapping

import msgspec
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.v1.beam.controller_types import BeamControllerRequest

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
    # Explicit decode step for EngineCore-owned control. Legacy Serving
    # updates omit it and continue deriving the step from beam_tokens length.
    decode_step: int | None = None
    controller_owned: bool = False


class BeamControllerStartRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    """One logical frontend submission for an EngineCore beam session."""

    controller: BeamControllerRequest
    engine_request: EngineCoreRequest
    call_id: int
    client_index: int = 0
    current_wave: int = 0
    data_parallel_rank: int | None = None
    constraint_resource: dict[str, object] | None = None


class BeamControllerCancelRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):
    beam_session_id: str
    client_index: int = 0
    data_parallel_rank: int | None = None
