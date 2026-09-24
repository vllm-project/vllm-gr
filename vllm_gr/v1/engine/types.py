# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared types for vllm_gr engine patches.

Defines the upstream grouped beam update and the internal BeamProc plan type.
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
    # (pool-based BeamSearchSession). Defaults keep legacy updates
    # serialization-compatible (omit_defaults=True).
    parent_beam_indices: list[int] | None = None
    # Maximum number of decode steps for KV pool pre-allocation.
    max_decode_steps: int = 0
    # Parent cumulative logprobs for worker-side candidate pruning.
    parent_cum_logprobs: list[float] | None = None
    # Target beam width for worker-side candidate pruning (defaults to beam_width).
    target_beam_width: int = 0
    # True if this is the final decode step, so worker/engine teardown can happen inline.
    is_last_step: bool = False


class BeamForkRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Internal normalized BeamProc plan used inside EngineCore.

    This is no longer registered as an EngineCore wire request.  The public
    client protocol uses ``BeamRequestStepUpdate``; Rust plans are decoded by
    the dedicated BeamProc socket and normalized to this structure locally.
    """

    parent_ids: list[str]
    child_ids: list[str]
    token_ids: list[int]
    abort_ids: list[str]
    sampling_params: SamplingParams
    eos_token_id: int | None = None
    client_index: int = 0
    current_wave: int = 0
    priority: int = 0
    data_parallel_rank: int | None = None
    beam_width: int = 0
    active_parent_ids: list[str] = msgspec.field(default_factory=list)
    active_parent_cum_logprobs: list[float] = msgspec.field(default_factory=list)
    active_parent_valid_token_ids: list[list[int]] = msgspec.field(default_factory=list)
    child_cum_logprobs: list[float] = msgspec.field(default_factory=list)
    child_valid_token_ids: list[list[int]] = msgspec.field(default_factory=list)
    child_flat_indices: list[int] = msgspec.field(default_factory=list)
    lora_request: LoRARequest | None = None
    cache_salt: str | None = None
    trace_headers: Mapping[str, str] | None = None
