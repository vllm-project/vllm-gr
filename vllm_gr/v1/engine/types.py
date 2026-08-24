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


# ---------------------------------------------------------------------------
# BeamSearchRequest — engine-side beam-search session bootstrap
# ---------------------------------------------------------------------------


class BeamSearchRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """One message that starts a full engine-side beam-search session.

    The engine runs the complete beam loop (prefill wave + decode waves,
    catalog filtering, selection) internally and emits a single final
    EngineCoreOutputs for the session once all beams finish.
    """

    session_id: str
    prompt_token_ids: list[int]
    beam_width: int
    max_tokens: int
    sampling_params: SamplingParams
    eos_token_id: int | None = None
    sid_begin_token_id: int | None = None
    sid_end_token_id: int | None = None
    ignore_eos: bool = False
    include_stop_str_in_output: bool = False
    catalog_strategy: str = "fill"
    catalog_token_sequences: list[list[int]] | None = None
    # Cheap identity of the catalog content (computed once frontend-side).
    # The engine caches the built trie under this key, so the sequences only
    # need to be transferred once per catalog version.
    catalog_hash: int | None = None
    # Warm the engine-side catalog trie cache without starting a session.
    preload_only: bool = False
    client_index: int = 0
    current_wave: int = 0
    priority: int = 0
    data_parallel_rank: int | None = None
    lora_request: LoRARequest | None = None
    cache_salt: str | None = None
    trace_headers: Mapping[str, str] | None = None


class BeamSearchAbort(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Abort an engine-side beam-search session (client disconnect etc.)."""

    session_id: str
