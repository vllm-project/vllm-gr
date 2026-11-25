# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""EngineCore Beam admission checks."""

from __future__ import annotations

from typing import Any

from vllm_gr.config import GRConfig
from vllm_gr.config.models import (
    CONSTRAINT_TABLE_MAX_DECODE_STEPS,
    runtime_constraint_table,
)
from vllm_gr.platforms.beam_requests import (
    validate_decode_token_budget,
    validate_gr_device_config,
)
from vllm_gr.v1.beam.batch_contracts import BeamBatchParams, BeamBatchRequest
from vllm_gr.v1.beam.dispatch import validate_dispatch_request


def _reject_candidate_width_below_dynamic_schedule(
    gr_config: Any,
    params: BeamBatchParams,
) -> None:
    """A configured K must cover every dynamic stage width.

    An explicit request-level ``per_parent_beam_widths`` schedule takes
    precedence over the configured scalar and is validated on its own.

    K is forwarded to every stage, so each stage samples K candidates per
    parent and keeps its own ``output_width``: ``K=4`` with ``(2, 2, 2)`` runs
    TopK=4 and retains 2.  The one unsupported combination is a stage that
    retains more beams than its parents supply candidates.  This reads the
    *configured* K so the rejection is a request-level error at admission
    instead of a contract failure deep in the worker.  Leaving K unset keeps
    dynamic Beam widths working exactly as before.
    """

    if params.per_parent_beam_widths is not None:
        return
    widths = params.dynamic_beam_widths
    if widths is None:
        return
    configured = gr_config.beam_candidate_width
    if configured is None:
        return
    for index, width in enumerate(widths):
        if width > configured:
            raise ValueError(
                f"vllm-gr gr.candidate_width={configured} is below "
                f"dynamic_beam_widths[{index}]={width}: each parent must supply at "
                "least as many candidates as the stage retains"
            )


def validate_gr_generation_budget(gr_config: GRConfig, max_tokens: int) -> None:
    """Fix constrained SID length to table depth; otherwise bound it by capacity."""
    table = runtime_constraint_table(gr_config)
    if table is not None and table.enabled and max_tokens != CONSTRAINT_TABLE_MAX_DECODE_STEPS:
        raise ValueError(
            f"effective generation length {max_tokens} must equal {table.format} depth "
            f"({CONSTRAINT_TABLE_MAX_DECODE_STEPS})"
        )
    if max_tokens > gr_config.beam_max_decode_steps:
        raise ValueError(
            f"Decode steps {max_tokens} exceed runtime limit {gr_config.beam_max_decode_steps}"
        )


def validate_gr_request(
    request: BeamBatchRequest, *, config: Any = None, trie_count: int | None = None
) -> None:
    """Validate shared V1 semantics before native ADD or Worker allocation."""
    validate_dispatch_request(request, trie_count=trie_count)
    if config is not None:
        gr_config = GRConfig.from_vllm_config(config)
        validate_gr_generation_budget(gr_config, request.beam_params.max_tokens)
        if getattr(getattr(config, "model_config", None), "is_hybrid", False):
            params = request.beam_params
            widths = params.dynamic_beam_widths
            input_width = max(widths[:-1], default=0) if widths else params.beam_width
            scheduler = config.scheduler_config
            if params.max_tokens > 1 and len(request.items) * input_width > min(
                scheduler.max_num_seqs, scheduler.max_num_batched_tokens
            ):
                raise ValueError("Hybrid Beam Decode rows exceed native sequence or token capacity")
        _reject_candidate_width_below_dynamic_schedule(gr_config, request.beam_params)
        validate_decode_token_budget(request, config)


__all__ = [
    "validate_gr_device_config",
    "validate_gr_generation_budget",
    "validate_gr_request",
]
