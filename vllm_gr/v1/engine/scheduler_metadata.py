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

    In incremental decode mode, cache_len = prefix_len because decode KV
    is stored in the BeamAttentionPool, not in the paged cache.

    Constraint Table prefill requests (step 0) are plain prefills with
    ``is_constraint_prefill`` set by the EngineCore ADD_BATCH cache hook; they
    are not ``is_beam_decode``, so this branch runs before the decode early
    return.  ``beam_width`` is required: the Worker uses it as the constrained
    top-k width, so a missing/invalid value is a protocol error.
    """
    if getattr(req, "is_constraint_prefill", False):
        beam_width = getattr(req, "beam_width", None)
        if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
            raise RuntimeError(
                f"Constraint prefill request {req.request_id!r} is missing a valid "
                f"beam_width={beam_width!r}; expected a positive integer"
            )
        return {
            "is_beam_decode": False,
            "is_constraint_prefill": True,
            "is_constrained": True,
            "constraint_step": 0,
            "beam_width": beam_width,
            "beam_suffix": [],
            "prefix_len": getattr(req, "prefix_len", 0),
            "cache_len": req.num_computed_tokens - num_scheduled_tokens,
            "decode_step": 0,
            "max_decode_steps": getattr(req, "max_decode_steps", 0),
        }

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

    prefix_len = getattr(req, "prefix_len", 0)

    # In incremental decode: the paged cache only holds the block-aligned
    # prefix KV; decoder KV lives in the BeamAttentionPool. The scheduler
    # treats the full prefix as computed (see patch_get_computed_blocks in
    # engine_core_patch.py), so the scheduler-view cache_len is the number
    # of tokens the scheduler believes are already computed.
    cache_len = req.num_computed_tokens - num_scheduled_tokens

    # decode_step: current decode step (0-indexed)
    decode_step = getattr(req, "beam_decode_step", 0)

    metadata: Dict[str, Any] = {
        "is_beam_decode": True,
        "beam_width": beam_width,
        "beam_decode_steps": getattr(req, "beam_decode_steps", 0),
        "prefix_len": prefix_len,
        "cache_len": cache_len,
        "decode_step": decode_step,
        "parent_beam_ids": getattr(req, "parent_beam_indices", None),
        "max_decode_steps": getattr(req, "max_decode_steps", 0),
    }

    if getattr(req, "is_constrained", False):
        # Constraint Table decode: carries the step the Worker's raw-logits
        # hook must run and the beam suffixes it needs to build the operator
        # sequence_group.  Trie decodes never set is_constrained.
        #
        # Step 0 is Constraint Table prefill ONLY (the is_constraint_prefill
        # branch above); a decode request must carry step 1 or 2.  Accepting 0
        # here would make the Worker build a prefill-shaped sequence_group and
        # bypass the beam-suffix cross-check.  Missing/invalid is a protocol
        # error, mirroring the ``beam_width`` check above.
        constraint_step = getattr(req, "constraint_step", None)
        if (
            isinstance(constraint_step, bool)
            or not isinstance(constraint_step, int)
            or constraint_step not in (1, 2)
        ):
            raise RuntimeError(
                f"Grouped beam request {req.request_id!r} is missing a valid "
                f"constraint_step={constraint_step!r}; expected 1 or 2 "
                "(step 0 is Constraint Table prefill only)"
            )
        metadata["is_constrained"] = True
        metadata["constraint_step"] = constraint_step
        metadata["beam_suffix"] = getattr(req, "beam_suffix", [])

    return metadata
