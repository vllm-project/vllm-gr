# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler metadata construction for grouped beam-decode requests."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_beam_request_metadata(req: Any, num_scheduled_tokens: int) -> Optional[Dict[str, Any]]:
    """Build scheduler metadata for a grouped beam-decode request.

    ``beam_width`` is execution metadata carried by ``BeamRequestStepUpdate``;
    it must never be inferred from ``SamplingParams.n`` or silently defaulted.
    A missing or invalid value means the EngineCore protocol chain is broken,
    so fail before the model runner can execute with incorrect batch geometry.
    """
    if not getattr(req, "is_beam_decode", False):
        return None

    if not hasattr(req, "beam_width"):
        raise RuntimeError(
            f"Grouped beam request {req.request_id!r} is missing required beam_width metadata"
        )

    beam_width = req.beam_width
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
        raise RuntimeError(
            f"Grouped beam request {req.request_id!r} has invalid beam_width={beam_width!r}; "
            "expected a positive integer"
        )

    return {
        "is_beam_decode": True,
        "beam_width": beam_width,
        "beam_decode_steps": getattr(req, "beam_decode_steps", 0),
        "prefix_len": getattr(req, "prefix_len", 0),
        "cache_len": req.num_computed_tokens - num_scheduled_tokens,
    }
