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
    # Appended after all existing fields to preserve msgspec's positional
    # array wire compatibility with older EngineCore processes.
    parent_cumulative_logprobs: list[float] | None = None
    # Atomic synchronous multi-request cohort. None preserves the legacy
    # single-session wire representation.
    cohort_id: int | None = None
    # Stage 1 resident form (async scheduling): when True, the first decode
    # step builds ONE Request that lives for the whole session
    # (max_tokens=max_decode_steps, ignore_eos=True); subsequent decode steps
    # are driven by the upstream scheduler as ordinary cached-reqs decode —
    # the worker overwrites input_ids/positions from the device beam decision
    # (9.diff "scheduler keeps bookkeeping, worker owns the inputs" model).
    # Default False keeps the Stage 0 per-step Request lifecycle
    # byte-identical on the wire.
    resident: bool = False
    # Stage 1 single-Request form: the initial prefill ADD carries the WHOLE
    # session lifecycle (max_tokens = max_decode_steps, ignore_eos=True) and
    # terminates naturally via check_stop (FINISHED_LENGTH_CAPPED).  There is
    # no engine-driven prefill→step0 ADD chain, no input_queue hop, and no
    # frontend re-registration: the prefill and every decode step run as
    # successive schedules of the SAME Request, and the worker seeds the device
    # beam-decision buffers (sequence[:,:,0] = prefill winners) at prefill
    # sample time so step 0's input overwrite reads the real winners.  Superset
    # of ``resident`` (the Request is still long-lived); the extra flag only
    # selects the no-re-ADD control flow.
    single_request: bool = False
