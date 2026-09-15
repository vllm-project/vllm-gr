# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler metadata construction for grouped beam-decode requests."""

from __future__ import annotations

from typing import Any, Dict, Optional

from vllm_gr.config import MAX_BEAM_WIDTH


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

    Single-Request form: one Request carries the whole session, so the SAME
    request is scheduled first as a prefill (``num_scheduled_tokens == prompt
    length``) and then as 1-token decodes.  Branch on the scheduled token
    count: a multi-token schedule emits the prefill metadata (so the worker's
    unconstrained prefill-decision gate fires and seeds the device buffers);
    a 1-token schedule emits the resident decode metadata (the worker widens
    to W rows and overwrites input_ids from the device beam decision).
    """
    # ---- Single-Request form: one Request, branched by token count ----
    if getattr(req, "is_beam_single_request", False) is True:
        beam_width = getattr(req, "beam_width", None)
        if (
            isinstance(beam_width, bool)
            or not isinstance(beam_width, int)
            or not 1 <= beam_width <= MAX_BEAM_WIDTH
        ):
            raise RuntimeError(
                f"Single-request beam request {req.request_id!r} is missing a valid "
                f"beam_width={beam_width!r}; expected an integer in [1, {MAX_BEAM_WIDTH}]"
            )
        # Distinguish the prefill schedule from the 1-token decode schedules
        # on this single Request.  Under async scheduling the decode schedule
        # is issued BEFORE the prefill's output is consumed (its future is
        # still queued, ``update_from_output`` has not appended any output
        # token yet), so ``num_output_tokens`` is still 0 on the first decode
        # schedule and cannot be used to tell the two apart.  The reliable
        # signal is whether the prompt was already fully computed BEFORE this
        # schedule: ``_update_after_schedule`` (run inside ``schedule()``)
        # advances ``num_computed_tokens`` by ``num_scheduled_tokens``, so the
        # pre-schedule computed count is ``num_computed_tokens -
        # num_scheduled_tokens``.  That being < ``num_prompt_tokens`` means the
        # prompt was not yet fully computed → this is the prefill schedule;
        # otherwise the request is past its prompt → decode schedule.  This is
        # robust to single-token prompts and to the prefill output not yet
        # being consumed.
        num_computed_before = req.num_computed_tokens - num_scheduled_tokens
        is_prefill_schedule = num_computed_before < req.num_prompt_tokens
        if not is_prefill_schedule:
            cache_len = req.num_computed_tokens - num_scheduled_tokens
            return {
                "is_beam_decode": True,
                "is_beam_prefill": False,
                "beam_width": beam_width,
                "beam_decode_steps": getattr(req, "beam_decode_steps", 0),
                "prefix_len": getattr(req, "prefix_len", 0),
                "cache_len": cache_len,
                # Request-derived decode_step stays 0 (the Request is reused);
                # the worker derives the real step from its session state.
                "decode_step": getattr(req, "beam_decode_step", 0),
                "parent_beam_ids": None,
                "max_decode_steps": getattr(req, "max_decode_steps", 0),
                "resident": True,
                "single_request": True,
                "result_width": getattr(req, "result_width", None),
            }
        # Prefill schedule of the single Request.
        return {
            "is_beam_decode": False,
            "is_beam_prefill": True,
            "is_beam_single_request": True,
            "is_constraint_prefill": False,
            "beam_width": beam_width,
            "prefix_len": getattr(req, "prefix_len", 0),
            "decode_step": 0,
            "max_decode_steps": getattr(req, "max_decode_steps", 0),
        }

    if (
        getattr(req, "is_beam_prefill", False) is True
        and getattr(req, "is_constraint_prefill", False) is not True
    ):
        # Unconstrained beam-session prefill (Stage 1 resident form).  The
        # candidates come from the ordinary sampler's flat logprobs (top-K of
        # the prefill logits), so the worker's unconstrained prefill decision
        # reads the width from here.  is_constraint_prefill requests carry a
        # stricter contract (valid beam_width enforced, worker runs the
        # constrained top-k operator) and never take this branch.
        beam_width = getattr(req, "beam_width", None)
        if (
            isinstance(beam_width, bool)
            or not isinstance(beam_width, int)
            or not 1 <= beam_width <= MAX_BEAM_WIDTH
        ):
            raise RuntimeError(
                f"Beam prefill request {req.request_id!r} is missing a valid "
                f"beam_width={beam_width!r}; expected an integer in [1, {MAX_BEAM_WIDTH}]"
            )
        return {
            "is_beam_decode": False,
            "is_beam_prefill": True,
            "is_constraint_prefill": False,
            "beam_width": beam_width,
            "prefix_len": getattr(req, "prefix_len", 0),
            "decode_step": 0,
            "max_decode_steps": getattr(req, "max_decode_steps", 0),
        }

    if getattr(req, "is_constraint_prefill", False) is True:
        beam_width = getattr(req, "beam_width", None)
        if (
            isinstance(beam_width, bool)
            or not isinstance(beam_width, int)
            or not 1 <= beam_width <= MAX_BEAM_WIDTH
        ):
            raise RuntimeError(
                f"Constraint prefill request {req.request_id!r} is missing a valid "
                f"beam_width={beam_width!r}; expected an integer in [1, {MAX_BEAM_WIDTH}]"
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

    if getattr(req, "is_beam_decode", False) is not True:
        return None

    if not hasattr(req, "beam_width"):
        raise RuntimeError(
            f"Grouped beam request {req.request_id!r} is missing required beam_width metadata"
        )

    beam_width = req.beam_width
    if (
        isinstance(beam_width, bool)
        or not isinstance(beam_width, int)
        or not 1 <= beam_width <= MAX_BEAM_WIDTH
    ):
        raise RuntimeError(
            f"Grouped beam request {req.request_id!r} has invalid beam_width={beam_width!r}; "
            f"expected an integer in [1, {MAX_BEAM_WIDTH}]"
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
    resident = getattr(req, "is_beam_resident", False) is True
    metadata: Dict[str, Any] = {
        "is_beam_decode": True,
        "beam_width": beam_width,
        "beam_decode_steps": getattr(req, "beam_decode_steps", 0),
        "prefix_len": prefix_len,
        "cache_len": cache_len,
        "decode_step": decode_step,
        # Stage 1 resident form: the Request (and therefore its frozen
        # step-0 parent mapping) is re-used for every decode step, so the
        # per-step frontend mapping does not exist.  None keeps the worker
        # on its device-side reorder record (``state.kv_reorder``), which the
        # beam decision already applied — the "compare with frontend"
        # composite path must never run against a stale step-0 list.
        "parent_beam_ids": None if resident else getattr(req, "parent_beam_indices", None),
        "max_decode_steps": getattr(req, "max_decode_steps", 0),
        # Stage 1 resident form: every decode step of the session is
        # scheduled by the upstream scheduler as a cached-reqs step on the
        # SAME Request, so this Request-derived decode_step stays 0 after the
        # first step.  The worker must derive the real step from its beam
        # session state instead, and source input_ids from the device beam
        # decision (9.diff worker-overwrite model).
        "resident": resident,
        # Final-step output width for the worker beam decision (number of
        # returned sequences; ACS ``BeamSearchParams.result_width`` semantics:
        # independent of the decode step count, defaults to beam_width when
        # the request does not carry one).
        "result_width": getattr(req, "result_width", None),
    }
    parent_scores = getattr(req, "parent_cumulative_logprobs", None)
    if parent_scores is not None:
        metadata["parent_cumulative_logprobs"] = parent_scores

    if getattr(req, "is_constrained", False) is not True:
        # Plain beam decode also carries the parent sequences: the Worker
        # seeds its beam-decision state from them when a session is
        # (re)created and rejects parent scores without suffixes as a
        # protocol cross-check.
        beam_suffix = getattr(req, "beam_suffix", None)
        if beam_suffix:
            metadata["beam_suffix"] = beam_suffix

    if getattr(req, "is_constrained", False) is True:
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
