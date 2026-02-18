# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam-search related msgspec structs injected by monkey-patching."""

import time

import msgspec

from vllm.v1.engine import (
    EngineCoreOutput,
    SchedulerStats,
    UtilityOutput,
)


class BeamSearchRequest(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    request_id: str
    prompt_token_ids: list[int]
    eos_token_id: int
    beam_width: int
    max_tokens: int
    n_best: int = 1
    temperature: float = 0.0
    length_penalty: float = 1.0
    ignore_eos: bool = False
    include_stop_str_in_output: bool = False
    sid_begin_token_id: int | None = None
    sid_end_token_id: int | None = None
    client_index: int = 0


class BeamSearchSequenceResult(
    msgspec.Struct,
    array_like=True,
    gc=False,
):
    token_ids: list[int]
    cumulative_logprob: float
    finish_reason: str
    stop_reason: int | str | None = None


class BeamSearchResult(
    msgspec.Struct,
    array_like=True,
    gc=False,
):
    request_id: str
    sequences: list[BeamSearchSequenceResult]
    error: str | None = None


class PatchedEngineCoreOutputs(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    """Drop-in replacement for EngineCoreOutputs with beam_search_result."""

    engine_index: int = 0
    outputs: list[EngineCoreOutput] = msgspec.field(default_factory=list)
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0
    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None
    wave_complete: int | None = None
    start_wave: int | None = None
    beam_search_result: BeamSearchResult | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()
