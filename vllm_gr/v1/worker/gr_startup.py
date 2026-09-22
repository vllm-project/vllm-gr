# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare the V1 execution view for shared Beam Forward capture."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch

from vllm_gr.config import BeamExecutionMode
from vllm_gr.v1.beam.batch_contracts import BeamBatchParams, BeamOutputOptions
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRExecutionCapacity,
    GRStageKind,
    GRStageMetadata,
    validate_gr_execution_config,
)
from vllm_gr.v1.engine.gr_async_scheduler import GR_STAGE_OUTPUT_ATTR


@torch.inference_mode()
def capture_async_forward(worker: Any) -> bool:
    from vllm_gr.v1.worker.gpu_beam_stage_runner import get_stage_runner
    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

    cfg = _get_gr_config(worker)
    if cfg.beam_execution_mode is not BeamExecutionMode.V1:
        return False
    validate_gr_execution_config(worker.vllm_config)
    if (
        not cfg.beam_worker_decision
        or cfg.beam_max_width is None
        or cfg.constraint_table is None
        or not cfg.constraint_table.enabled
    ):
        raise ValueError("Beam V1 requires worker decisions, fixed width and a canonical table")
    if (
        str(worker.device).split(":")[0] != "cuda"
        or worker.model_config.enforce_eager
        or not cfg.beam_graph_enabled
    ):
        return False
    existing = getattr(worker, "_gr_gpu_stage_runner", None)
    if existing is not None:
        if existing.graph is None:
            raise RuntimeError(
                "Asynchronous scheduling startup encountered an execution view without a graph"
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
    )
    scheduled = SimpleNamespace(num_scheduled_tokens={name: 1})
    setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {name: m})
    owner = get_stage_runner(worker, scheduled)
    owner.previous = m
    # The null block is a safe capture prefix. Real requests replace its
    # contents and actual length while the device table address stays fixed.
    # Published as kv_blocks so both runner surfaces (V1 request state and
    # V2 scheduler-published metadata) find it identically.
    from vllm_gr.v1.worker.model_runner_common import _use_v2_model_runner

    if _use_v2_model_runner():
        decode_meta = replace(m, kv_blocks=((0,),))
        setattr(scheduled, GR_STAGE_OUTPUT_ATTR, {name: decode_meta})
    else:
        worker.requests[name] = SimpleNamespace(block_ids=[[0]])
    try:
        decode = replace(m, kind=GRStageKind.DECODE, stage_index=1, decode_step=0)
        if _use_v2_model_runner():
            decode = replace(decode, kv_blocks=((0,),))
        owner.prepared = owner.prepare_cpu(decode, 0)
        owner._bind_attention(decode)
        owner.input_ids.zero_()
        owner.positions.zero_()
        owner.execution_mask.zero_()
        owner.beam.session.context.decode_step_buf.fill_(-1)
        owner.capture_forward()
        torch.cuda.current_stream(worker.device).synchronize()
    finally:
        if not _use_v2_model_runner():
            worker.requests.pop(name, None)
        done = torch.Event()
        done.record()
        owner.release(owner.binding, consumer_done=done)
    return True
