# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Lightweight installation markers and native runner completeness checks."""

from typing import Any

_WORKER_EXECUTE_PATCH_MARKER = "_vllm_gr_beam_execute_patch"
_WORKER_PREPARE_PATCH_MARKER = "_vllm_gr_beam_prepare_patch"
# Optional: only installed when the runner exposes ``_sample`` (NPU builds may
# not).  is_worker_runner_patched() deliberately does not require it.
_WORKER_SAMPLE_TOKENS_PATCH_MARKER = "_vllm_gr_beam_sample_tokens_patch"
_WORKER_DETERMINE_PATCH_MARKER = "_vllm_gr_beam_determine_patch"
_WORKER_CAPTURE_PATCH_MARKER = "_vllm_gr_beam_capture_patch"
_WORKER_MODEL_FORWARD_PATCH_MARKER = "_vllm_gr_prefill_model_forward_patch"

# ModelRunner V2 patch markers.  The V2 runner has a different method surface
# (``prepare_inputs``/``sample``/``sample_tokens`` instead of
# ``_prepare_inputs``/``_bookkeeping_sync``/``_sample``), so the V2 wrappers
# stamp their own markers. Independent V1/V2 execute wrappers share the
# execute installation marker.
_WORKER_V2_PREPARE_PATCH_MARKER = "_vllm_gr_beam_prepare_v2_patch"
_WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER = "_vllm_gr_beam_sample_tokens_v2_patch"
# Optional GR Prefill registration specialization.
_WORKER_V2_ADD_REQUESTS_PATCH_MARKER = "_vllm_gr_beam_add_requests_v2_patch"

# Canonical Constraint Backend load/shutdown markers.  Defined here (the shared
# home) so both the V1 dispatcher and the V2-aware validation gate can read the
# same constants without a circular import.
_CONSTRAINT_LOAD_PATCH_MARKER = "_vllm_gr_constraint_load_patched"
_CONSTRAINT_SHUTDOWN_PATCH_MARKER = "_vllm_gr_constraint_shutdown_patched"


def is_worker_runner_v2_patched(runner_class: Any) -> bool:
    """Return whether all grouped-beam hooks are present on a V2 runner class.

    The V2 runner (``vllm.v1.worker.gpu.model_runner.GPUModelRunner``) exposes
    ``sample_tokens`` instead of the V1 ``_bookkeeping_sync`` surface. The
    required V2 method set is the slim ``execute_model`` and ``sample_tokens``
    stage bypass plus the canonical Constraint Backend load/shutdown markers.
    Native ``sample`` remains untouched: V2 supports only beam_search_v1,
    whose stage runner owns constrained sampling. The optional ``prepare_inputs``
    specialization preserves the native InputBatch contract. ``add_requests``
    optionally skips unused sampler state for admitted GR Prefill;
    ``finish_requests`` remains native. The stage path also drives this state
    machine directly (``consume_native_state``).
    """
    return bool(
        getattr(runner_class.execute_model, _WORKER_EXECUTE_PATCH_MARKER, False)
        and getattr(runner_class.sample_tokens, _WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, False)
        and getattr(runner_class.load_model, _CONSTRAINT_LOAD_PATCH_MARKER, False)
        and getattr(runner_class.shutdown, _CONSTRAINT_SHUTDOWN_PATCH_MARKER, False)
    )


def is_worker_runner_patched(runner_class: Any) -> bool:
    """Return whether all grouped-beam hooks are present on a runner class.

    Dispatches between the V1 and V2 method surfaces: the V2 runner has a
    ``prepare_inputs`` method and no ``_prepare_inputs``.

    The ``_determine_batch_execution_and_padding`` hook stamps the beam
    determine marker on all V1 platforms.  On GPU it overrides the dispatch
    result to force FULL CUDA Graph for beam decode; on NPU it is a no-op
    wrapper that preserves the original result — but the marker must still
    be present for the gate-check to pass.
    """
    if hasattr(runner_class, "prepare_inputs") and not hasattr(runner_class, "_prepare_inputs"):
        return is_worker_runner_v2_patched(runner_class)

    has_determine = hasattr(runner_class, "_determine_batch_execution_and_padding")
    return bool(
        getattr(runner_class.execute_model, _WORKER_EXECUTE_PATCH_MARKER, False)
        and getattr(runner_class._prepare_inputs, _WORKER_PREPARE_PATCH_MARKER, False)
        and (
            runner_class._bookkeeping_sync
            is getattr(runner_class, "_vllm_gr_native_bookkeeping", None)
        )
        and (
            not has_determine
            or getattr(
                runner_class._determine_batch_execution_and_padding,
                _WORKER_DETERMINE_PATCH_MARKER,
                False,
            )
        )
    )
