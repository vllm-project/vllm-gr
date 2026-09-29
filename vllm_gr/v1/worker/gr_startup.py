# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare the persistent stage execution view for shared Beam Forward capture."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_gr.v1.beam.batch_contracts import BeamBatchParams, BeamOutputOptions
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRExecutionCapacity,
    GRStageKind,
    GRStageMetadata,
    validate_gr_execution_runtime,
)
from vllm_gr.v1.engine.gr_async_scheduler import GR_STAGE_OUTPUT_ATTR
from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeCaptureUnsupported

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_gr.v1.worker.gpu_beam_stage_runner import _StageExecutionHooks


@torch.inference_mode()
def capture_async_forward(
    worker: Any,
    *,
    hooks_type: type[_StageExecutionHooks] | None = None,
    allow_eager_fallback: bool = False,
) -> bool:
    from vllm_gr.v1.worker.gpu_beam_stage_runner import (
        _StageExecutionHooks,
        get_stage_runner,
    )
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    cfg = _get_gr_config(worker)
    validate_gr_execution_runtime(worker.vllm_config)
    if (
        not cfg.beam_worker_decision
        or cfg.beam_max_width is None
        or cfg.constraint_table is None
        or not cfg.constraint_table.enabled
    ):
        raise ValueError("Beam V1 requires worker decisions, fixed width and a canonical table")
    device_type = str(worker.device).split(":")[0]
    if worker.model_config.enforce_eager or not cfg.beam_graph_enabled:
        return False
    if device_type == "npu":
        from vllm_gr.v1.worker.npu_model_runner_patch import capture_beam_decode_graph

        return capture_beam_decode_graph(worker)
    if device_type != "cuda":
        return False

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
    width = cfg.beam_max_width
    GRExecutionCapacity(width)
    sessions = getattr(worker, "_beam_sessions", {})
    if sessions:
        raise RuntimeError("Asynchronous scheduling startup capture requires drained Beam sessions")
    worker._beam_sessions = sessions
    name = "__async_scheduling_startup_capture__"
    m = GRStageMetadata(
        session_id=name,
        dispatch_id=0,
        stage_index=0,
        kind=GRStageKind.PREFILL,
        decode_step=None,
        is_last_stage=True,
        input_device_state=None,
        output_device_state=GRDeviceStateRef(name, 0),
        beam_params=BeamBatchParams(width, 1, 0.0, (), True),
        output_options=BeamOutputOptions(width, True, True, True),
        num_prompt_tokens=1,
        request_ids=(name,),
        num_prompt_tokens_by_request=(1,),
    )
    scheduled = SimpleNamespace(num_scheduled_tokens={name: 1})
    setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {name: m})
    owner = get_stage_runner(worker, scheduled, hooks_type=hooks_type or _StageExecutionHooks)
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
                m, kind=GRStageKind.DECODE, stage_index=1, decode_step=0, kv_blocks=((0,),)
            )
            owner.prepared = owner.prepare_cpu(decode, 0)
            owner._bind_attention(decode)
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
        worker._gr_async_decode_graph_capture_failed = True
        worker._gr_async_decode_graph_capture_error = str(fallback_error)
        logger.warning(
            "GR Decode capture unsupported; cleanup and device checks succeeded. "
            "Using eager Decode: %s",
            fallback_error,
        )
        return False
    if graph is None:
        raise RuntimeError("Decode capture completed without a graph candidate")
    # Publishing is the final step: no serving path can see this graph before
    # the startup owner has been released and its GPU work has completed.
    owner.graph = graph
    return True
