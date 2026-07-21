# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-independent GR EngineCore commands."""

from __future__ import annotations

from collections.abc import Mapping

import msgspec
from vllm.sampling_params import SamplingParams

from vllm_gr.protocol.version import CURRENT_GR_PROTOCOL_VERSION


class GRBeamStepRequest(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
    tag="beam_step",
):
    """Execute one grouped Beam decode step for an existing session."""

    session_id: str
    sequence_no: int
    beam_width: int
    beam_token_ids: list[list[int]]
    prefix_len: int
    sampling_params: SamplingParams
    eos_token_id: int | None = None
    priority: int = 0
    trace_headers: Mapping[str, str] | None = None


class GREndBeamSessionRequest(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
    tag="end_session",
):
    """Idempotently release all state owned by a Beam Session."""

    session_id: str
    sequence_no: int


GRCommand = GRBeamStepRequest | GREndBeamSessionRequest


class GRCommandEnvelope(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
):
    """Versioned command plus transport context required by EngineCore."""

    command_id: str
    command: GRCommand
    protocol_major: int = CURRENT_GR_PROTOCOL_VERSION.major
    protocol_minor: int = CURRENT_GR_PROTOCOL_VERSION.minor
    client_index: int = 0
    current_wave: int = 0
