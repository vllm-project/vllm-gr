# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared types for vllm_gr engine patches.

Defines BeamForkRequest, BeamSearchRequest, BeamSearchResult,
BeamSearchSequenceResult, and PatchedEngineCoreOutputs.
"""

from __future__ import annotations

from collections.abc import Mapping

import msgspec

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams


# ---------------------------------------------------------------------------
# BeamForkRequest — lightweight beam fork (ADD_BATCH + BEAM_FORK path)
# ---------------------------------------------------------------------------

class BeamForkRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Lightweight beam search fork: clone parent state + append 1 token.

    Parallel arrays: parent_ids[i] -> child_ids[i] with token_ids[i] appended.
    A parent can appear multiple times (one beam forking into multiple children).
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
    lora_request: LoRARequest | None = None
    cache_salt: str | None = None
    trace_headers: Mapping[str, str] | None = None


# ---------------------------------------------------------------------------
# BeamSearchRequest — full beam search in EngineCore (BEAM_SEARCH path)
# ---------------------------------------------------------------------------

class BeamSearchRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Full beam search request sent to EngineCore."""

    request_id: str
    prompt_token_ids: list[int]
    beam_width: int
    max_tokens: int
    temperature: float = 0.0
    length_penalty: float = 1.0
    ignore_eos: bool = False
    include_stop_str_in_output: bool = False
    eos_token_id: int = 2
    sid_begin_token_id: int | None = None
    sid_end_token_id: int | None = None
    n_best: int = 0  # 0 means beam_width
    client_index: int = 0


# ---------------------------------------------------------------------------
# BeamSearchResult / BeamSearchSequenceResult
# ---------------------------------------------------------------------------

class BeamSearchSequenceResult(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """A single beam result sequence."""

    token_ids: list[int]
    cumulative_logprob: float
    finish_reason: str = "length"
    stop_reason: int | str | None = None


class BeamSearchResult(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Final beam search result from EngineCore."""

    request_id: str
    sequences: list[BeamSearchSequenceResult]


# ---------------------------------------------------------------------------
# PatchedEngineCoreOutputs — drop-in replacement with beam_search_result
# ---------------------------------------------------------------------------

class PatchedEngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """EngineCoreOutputs with an additional beam_search_result field.

    Field order must exactly match EngineCoreOutputs (array_like encoding),
    with new fields appended at the end.
    """

    # --- Fields mirrored from EngineCoreOutputs (same order) ---
    engine_index: int = 0
    outputs: list = msgspec.field(default_factory=list)
    scheduler_stats: object = None
    timestamp: float = 0.0
    utility_output: object = None
    finished_requests: set[str] | None = None
    wave_complete: int | None = None
    start_wave: int | None = None
    # --- New fields (appended) ---
    beam_search_result: BeamSearchResult | None = None
