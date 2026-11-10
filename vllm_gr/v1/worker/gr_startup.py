# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare the persistent stage execution view for shared Beam Forward capture."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_gr.v1.beam.batch_contracts import (
    SUPPORTED_BEAM_BATCH_SIZES,
    BeamBatchParams,
    BeamOutputOptions,
)
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRExecutionCapacity,
    GRStageKind,
    GRStageMetadata,
    validate_gr_device_config,
    validate_gr_execution_runtime,
)
from vllm_gr.v1.engine.gr_async_scheduler import GR_STAGE_OUTPUT_ATTR
from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeCaptureUnsupported

logger = init_logger(__name__)
ENV_V1_STARTUP_CAPTURE_BATCH_SIZE = "VLLM_GR_V1_STARTUP_CAPTURE_BATCH_SIZE"


def _startup_capture_batch_size() -> int:
    raw = os.getenv(ENV_V1_STARTUP_CAPTURE_BATCH_SIZE)
    if raw is None:
        return 1
    message = (
        f"{ENV_V1_STARTUP_CAPTURE_BATCH_SIZE} must be one of {list(SUPPORTED_BEAM_BATCH_SIZES)}"
    )
    try:
        batch_size = int(raw)
    except ValueError as exc:
        raise ValueError(message) from exc
    if batch_size not in SUPPORTED_BEAM_BATCH_SIZES:
        raise ValueError(message)
    return batch_size


if TYPE_CHECKING:
    from vllm_gr.v1.worker.gpu_beam_stage_runner import _StageExecutionHooks


@torch.inference_mode()
def capture_async_forward(
    worker: Any,
    *,
    hooks_type: type[_StageExecutionHooks] | None = None,
    allow_eager_fallback: bool = False,
) -> bool:
    if not _capture_enabled(worker):
        return False
    device_type = str(worker.device).split(":")[0]
    if device_type == "npu":
        from vllm_gr.v1.worker.npu_model_runner_patch import capture_beam_decode_graph

        return capture_beam_decode_graph(worker)
    if device_type != "cuda":
        return False
    # Measure (or load) the device-local prefix cost model before any Beam graph
    # is captured: the cache write is a YAML file and the probe needs the GPU to
    # itself. Failures degrade to "no calibration", never to a failed start-up.
    from vllm_gr.v1.attention.backends.beam_prefix_calibrate import (
        ensure_prefix_calibration,
    )
    from vllm_gr.v1.attention.backends.beam_prefix_policy import (
        remember_worker_flash_attn_version,
    )

    # The request path may run without a vLLM config context, so persist the
    # operator's explicit FlashAttention choice while it is still available.
    remember_worker_flash_attn_version(worker)
    ensure_prefix_calibration(worker)
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    if getattr(_get_gr_config(worker), "beam_graph_widths", None) is not None:
        logger.warning(
            "beam.graph_widths is NPU-only startup pre-capture; CUDA captures "
            "dynamic Decode (batch, width) shapes lazily on first use."
        )

    existing = getattr(worker, "_gr_gpu_stage_runner", None)
    if existing is not None:
        if (
            existing.graph is None
            or not existing.graph.ready
            or not existing.closed
            or getattr(worker, "_gr_async_decode_graph_capture_failed", False)
        ):
            raise RuntimeError(
                "Asynchronous scheduling startup encountered an incomplete capture state"
            )
        return True
    error = _capture_batch(
        worker,
        _startup_capture_batch_size(),
        hooks_type=hooks_type,
        allow_eager_fallback=allow_eager_fallback,
    )
    if error is not None:
        worker._gr_async_decode_graph_capture_failed = True
        worker._gr_async_decode_graph_capture_error = error
        return False
    return True


def _capture_enabled(worker: Any) -> bool:
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    from vllm_gr.config import ConstraintMode

    cfg = _get_gr_config(worker)
    validate_gr_execution_runtime(worker.vllm_config)
    unconstrained = cfg.constraint_config.mode is ConstraintMode.NONE
    if (
        not cfg.beam_worker_decision
        or cfg.beam_max_width is None
        or (
            not unconstrained and (cfg.constraint_table is None or not cfg.constraint_table.enabled)
        )
    ):
        raise ValueError(
            "Beam V1 requires worker decisions, fixed width and either an enabled "
            "canonical table or constrained_generation=False"
        )
    if unconstrained and str(worker.device).split(":")[0] not in ("cuda", "npu"):
        raise ValueError("unconstrained Beam V1 requires CUDA or NPU")
    if unconstrained and str(worker.device).split(":")[0] == "npu":
        validate_gr_device_config(worker.vllm_config)
    # A one-token session completes in Prefill and never replays a Decode graph.
    return (
        not worker.model_config.enforce_eager
        and cfg.beam_graph_enabled
        and cfg.beam_max_decode_steps > 1
    )


def _capture_batch(
    worker: Any,
    batch_size: int,
    *,
    hooks_type: type[_StageExecutionHooks] | None,
    allow_eager_fallback: bool,
) -> str | None:
    """Capture the explicit batch selected by the V1 or V2 entry point."""
    from vllm_gr.v1.worker.gpu_beam_stage_runner import (
        _StageExecutionHooks,
        get_stage_runner,
    )
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    cfg = _get_gr_config(worker)
    width = cfg.beam_max_width
    GRExecutionCapacity(width)
    sessions = getattr(worker, "_beam_sessions", {})
    if sessions:
        raise RuntimeError("Asynchronous scheduling startup capture requires drained Beam sessions")
    worker._beam_sessions = sessions
    name = "__async_scheduling_startup_capture__"
    request_ids = (
        (name,) if batch_size == 1 else tuple(f"{name}_{index}" for index in range(batch_size))
    )
    m = GRStageMetadata(
        session_id=name,
        dispatch_id=0,
        stage_index=0,
        kind=GRStageKind.PREFILL,
        decode_step=None,
        is_last_stage=False,
        input_device_state=None,
        output_device_state=GRDeviceStateRef(name, 0),
        # The capture includes Decode step 0 after the first Prefill token.
        beam_params=BeamBatchParams(width, 2, 0.0, (), True),
        output_options=BeamOutputOptions(width, True, True, True),
        num_prompt_tokens=batch_size,
        request_ids=request_ids,
        num_prompt_tokens_by_request=(1,) * batch_size,
    )
    scheduled = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(request_ids, 1))
    setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {name: m})
    owner = get_stage_runner(
        worker, scheduled, hooks_type=hooks_type or _StageExecutionHooks, for_startup_capture=True
    )
    owner.previous = m
    # The null block is a safe capture prefix. Real requests replace its
    # contents and actual length while the device table address stays fixed.
    # Both runner surfaces consume the same published KV metadata, without
    # creating a fake entry in native request state.
    graph = None
    fallback_error = None
    try:
        try:
            decode = replace(
                m,
                kind=GRStageKind.DECODE,
                stage_index=1,
                decode_step=0,
                is_last_stage=True,
                input_device_state=GRDeviceStateRef(name, 0),
                output_device_state=GRDeviceStateRef(name, 1),
                kv_blocks=((0,),) * batch_size,
            )
            owner.prepared = owner.prepare_cpu(decode, 0)
            owner._bind_attention(decode)
            # Keep the startup fallback graph at model capacity. Measured
            # length buckets are captured lazily against real request buffers.
            if getattr(getattr(owner, "metadata", None), "prefix_length_bucket", 0):
                owner.metadata.prefix_length_bucket = 0
                owner.metadata.prefix_max_seq_len = (
                    (owner.inputs.max_model_len + owner.inputs.block_size - 1)
                    // owner.inputs.block_size
                    * owner.inputs.block_size
                )
            owner.input_ids.zero_()
            owner.positions.zero_()
            owner.execution_mask.zero_()
            owner.beam.session.context.decode_step_buf.fill_(-1)
            try:
                graph = owner.capture_forward()
            except BeamDecodeCaptureUnsupported as error:
                if not allow_eager_fallback:
                    raise
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Decode capture did not leave capture mode") from error
                # Failed capture ran on a side stream. Drain it before ending
                # the dummy session; a device error here aborts startup.
                torch.accelerator.synchronize(worker.device)
                fallback_error = error
        finally:
            done = torch.Event()
            done.record()
            owner.release(owner.binding, consumer_done=done)
        if not owner.closed:
            raise RuntimeError("Decode startup resources were not released")
        torch.cuda.current_stream(worker.device).synchronize()
        if graph is not None and not graph.ready:
            raise RuntimeError("Decode capture completed without a ready graph")
    except Exception:
        if graph is not None:
            graph.clear()
        raise

    if fallback_error is not None:
        logger.warning(
            "GR Decode capture unsupported; cleanup and device checks succeeded. "
            "Using eager Decode for B=%d: %s",
            batch_size,
            fallback_error,
        )
        return str(fallback_error)
    if graph is None:
        raise RuntimeError("Decode capture completed without a graph candidate")
    # Publishing is the final step: no serving path can see this graph before
    # the startup owner has been released and its GPU work has completed.
    owner.graph = graph
    return None
