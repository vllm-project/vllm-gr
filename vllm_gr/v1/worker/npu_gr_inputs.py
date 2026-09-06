# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU V1 input geometry boundary for persistent GR Decode.

This module is intentionally not a backend factory.  It only scopes the one
logical Native token to ``W`` physical ModelRunner rows and replaces the
pinned vllm-ascend V1 ``_prepare_input_ids`` Host/control-token copy with one
device-to-device copy from persistent state. During native preparation it also
suppresses the unrelated Native slot lookup for physical Beam rows, whose KV
writes belong to the session's suffix pool. Runtime enablement still
requires the NPU backend, decision workspace, KV reorder and output transport.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import torch
from vllm.distributed.ec_transfer import has_ec_transfer
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_gr.v1.engine.types import GRRequestStage, GRSchedulerOutput
from vllm_gr.v1.worker.model_runner_common import (
    _WORKER_EXECUTE_PATCH_MARKER,
)
from vllm_gr.v1.worker.model_runner_common import (
    make_patched_execute_model as _make_common_patched_execute_model,
)

_NPU_EXECUTE_MARKER = "_vllm_gr_npu_physical_execute_patched"
_NPU_PREPARE_IDS_MARKER = "_vllm_gr_npu_prepare_input_ids_patched"
_FRAME_ATTR = "_gr_npu_input_frame"
_GEOMETRY_ATTR = "_gr_npu_logical_geometry"


@dataclass(slots=True)
class NPUInputFrame:
    """Validated, execute-local ownership of one persistent Decode geometry."""

    scheduler_output: Any
    request_id: str
    sidecar: GRSchedulerOutput
    metadata: dict[str, Any]
    physical_rows: int
    active: bool = True
    input_copy_done: bool = False


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"persistent NPU GR {name} is invalid")
    return value


def _persistent_decode_frame(scheduler_output: Any) -> NPUInputFrame | None:
    beam_data = getattr(scheduler_output, "beam_data", None)
    if not beam_data:
        return None
    if not isinstance(beam_data, dict):
        raise RuntimeError("persistent NPU GR beam_data must be a dict")
    sidecars = getattr(scheduler_output, "gr_outputs", None)
    if not isinstance(sidecars, dict):
        sidecars = {}
    for request_id, metadata in beam_data.items():
        sidecar = sidecars.get(request_id)
        if (
            isinstance(sidecar, GRSchedulerOutput)
            and sidecar.stage is GRRequestStage.DECODE_RUNNING
            and not isinstance(metadata, dict)
        ):
            raise RuntimeError("persistent NPU GR Decode metadata must be a dict")
        if (
            isinstance(metadata, dict)
            and metadata.get("persistent_runner_state") is True
            and isinstance(sidecar, GRSchedulerOutput)
            and sidecar.stage is GRRequestStage.DECODE_RUNNING
            and metadata.get("is_beam_decode") is not True
        ):
            raise RuntimeError("persistent NPU GR Decode metadata lost its stage marker")
    persistent = [
        (request_id, metadata)
        for request_id, metadata in beam_data.items()
        if isinstance(metadata, dict)
        and metadata.get("persistent_runner_state") is True
        and metadata.get("is_beam_decode") is True
    ]
    if not persistent:
        return None
    if len(persistent) != 1 or len(beam_data) != 1:
        raise RuntimeError("persistent NPU GR Decode currently requires one isolated B=1 request")
    request_id, metadata = persistent[0]
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("persistent NPU GR request_id must be non-empty")
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled, dict) or set(scheduled) != {request_id}:
        raise RuntimeError("persistent NPU GR Decode must be the only scheduled request")
    sidecars_value = getattr(scheduler_output, "gr_outputs", None)
    if not isinstance(sidecars_value, dict):
        raise RuntimeError("persistent NPU GR Decode is missing its typed sidecar")
    sidecar = sidecars_value.get(request_id)
    if not isinstance(sidecar, GRSchedulerOutput):
        raise RuntimeError("persistent NPU GR Decode requires a typed GRSchedulerOutput")
    if sidecar.request_id != request_id:
        raise RuntimeError("persistent NPU GR sidecar request_id mismatch")
    if sidecar.stage is not GRRequestStage.DECODE_RUNNING or sidecar.is_prefill:
        raise RuntimeError("persistent NPU GR sidecar is not a Decode stage")
    if sidecar.is_last_prefill:
        raise RuntimeError("persistent NPU GR Decode cannot be a Prefill boundary")
    step = _strict_int(sidecar.step, "sidecar step", minimum=1)
    device_step = _strict_int(sidecar.device_step, "sidecar device_step", minimum=1)
    if device_step != step:
        raise RuntimeError("persistent NPU GR sidecar step/device_step mismatch")
    logical = _strict_int(sidecar.logical_token_count, "logical token count", minimum=1)
    scheduled_logical = _strict_int(
        scheduled.get(request_id), "scheduled logical token count", minimum=1
    )
    if logical != 1 or scheduled_logical != 1:
        raise RuntimeError("persistent NPU GR native accounting must remain one logical token")
    total_logical = _strict_int(
        getattr(scheduler_output, "total_num_scheduled_tokens", None),
        "total logical token count",
        minimum=1,
    )
    if total_logical != 1:
        raise RuntimeError("persistent NPU GR total logical token count must be one")
    width = _strict_int(sidecar.physical_row_count, "physical row count", minimum=1)
    metadata_width = _strict_int(metadata.get("beam_width"), "Beam width", minimum=1)
    if width != metadata_width:
        raise RuntimeError("persistent NPU GR physical row count must equal Beam width")
    for name in ("generation_index", "constraint_step", "beam_decode_steps"):
        if _strict_int(metadata.get(name), name, minimum=1) != step:
            raise RuntimeError(f"persistent NPU GR metadata {name}/step mismatch")
    if _strict_int(metadata.get("decode_step"), "decode_step") != step - 1:
        raise RuntimeError("persistent NPU GR metadata decode_step mismatch")
    if metadata.get("is_constrained") is not True:
        raise RuntimeError("persistent NPU GR Decode must remain constrained")
    if metadata.get("persistent_request") is not True:
        raise RuntimeError("persistent NPU GR metadata lost persistent request ownership")
    if metadata.get("is_last_step") is not sidecar.is_final:
        raise RuntimeError("persistent NPU GR final marker mismatch")
    if metadata.get("next_token_ids") is not None:
        raise RuntimeError("persistent NPU GR next tokens must not come from Host metadata")
    if metadata.get("beam_suffix") != [] or metadata.get("parent_beam_ids") is not None:
        raise RuntimeError("persistent NPU GR Decode must not carry Host Beam state")
    return NPUInputFrame(scheduler_output, request_id, sidecar, metadata, width)


def _reject_unsupported_input_paths(runner: Any, scheduler_output: Any) -> None:
    if getattr(runner, "use_async_scheduling", None) is not True:
        raise RuntimeError("persistent NPU GR requires upstream async scheduling")
    if _strict_int(getattr(runner, "pcp_size", 1), "PCP size", minimum=1) != 1:
        raise RuntimeError("persistent NPU GR does not support PCP")
    if _strict_int(getattr(runner, "dcp_size", 1), "DCP size", minimum=1) != 1:
        raise RuntimeError("persistent NPU GR does not support DCP")
    parallel = getattr(runner, "parallel_config", None)
    if (
        _strict_int(
            getattr(parallel, "pipeline_parallel_size", 1),
            "pipeline parallel size",
            minimum=1,
        )
        != 1
    ):
        raise RuntimeError("persistent NPU GR does not support pipeline parallelism")
    if getattr(runner, "use_cp", False):
        raise RuntimeError("persistent NPU GR does not support CP input rebuild")
    if getattr(runner, "speculative_config", None) is not None:
        raise RuntimeError("persistent NPU GR does not support speculative decode")
    if getattr(scheduler_output, "scheduled_spec_decode_tokens", None):
        raise RuntimeError("persistent NPU GR does not support draft-token input rebuild")
    if getattr(scheduler_output, "has_structured_output_requests", False):
        raise RuntimeError("persistent NPU GR does not support grammar input")
    if getattr(scheduler_output, "scheduled_encoder_inputs", None):
        raise RuntimeError("persistent NPU GR does not support encoder inputs")
    input_batch = getattr(runner, "input_batch", None)
    if getattr(runner, "enable_prompt_embeds", False) or getattr(
        input_batch, "req_prompt_embeds", None
    ):
        raise RuntimeError("persistent NPU GR does not support prompt embeddings")
    if has_kv_transfer_group():
        raise RuntimeError("persistent NPU GR does not support an active KV connector")
    if has_ec_transfer():
        raise RuntimeError("persistent NPU GR does not support active EC transfer")


def make_patched_execute_model(original: Any) -> Any:
    """Publish physical W only inside one validated persistent NPU Decode."""

    common_execute = _make_common_patched_execute_model(original)
    if getattr(common_execute, _NPU_EXECUTE_MARKER, False):
        return common_execute

    @wraps(common_execute)
    def execute_with_npu_geometry(
        self: Any, scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if getattr(self, _FRAME_ATTR, None) is not None:
            raise RuntimeError("nested or stale persistent NPU GR input frame")
        frame = _persistent_decode_frame(scheduler_output)
        if frame is None:
            return common_execute(self, scheduler_output, *args, **kwargs)
        _reject_unsupported_input_paths(self, scheduler_output)
        if hasattr(scheduler_output, _GEOMETRY_ATTR):
            raise RuntimeError("stale persistent NPU GR SchedulerOutput geometry")

        request_id = frame.request_id
        scheduled = scheduler_output.num_scheduled_tokens
        original_request_tokens = scheduled[request_id]
        original_total_tokens = scheduler_output.total_num_scheduled_tokens
        scheduled[request_id] = frame.physical_rows
        scheduler_output.total_num_scheduled_tokens = frame.physical_rows
        setattr(scheduler_output, _GEOMETRY_ATTR, (request_id, 1, frame.physical_rows))
        setattr(self, _FRAME_ATTR, frame)
        try:
            return common_execute(self, scheduler_output, *args, **kwargs)
        finally:
            frame.active = False
            scheduled[request_id] = original_request_tokens
            scheduler_output.total_num_scheduled_tokens = original_total_tokens
            if getattr(self, _FRAME_ATTR, None) is frame:
                delattr(self, _FRAME_ATTR)
            if hasattr(scheduler_output, _GEOMETRY_ATTR):
                delattr(scheduler_output, _GEOMETRY_ATTR)

    setattr(execute_with_npu_geometry, _WORKER_EXECUTE_PATCH_MARKER, True)
    setattr(execute_with_npu_geometry, _NPU_EXECUTE_MARKER, True)
    return execute_with_npu_geometry


def _validate_cumulative_counts(cu_num_tokens: Any, width: int) -> None:
    if not isinstance(cu_num_tokens, np.ndarray):
        raise RuntimeError("persistent NPU GR cumulative token counts must be a NumPy array")
    if cu_num_tokens.shape != (1,) or not np.issubdtype(cu_num_tokens.dtype, np.integer):
        raise RuntimeError("persistent NPU GR cumulative token counts must be integer [1]")
    if cu_num_tokens[0] != width:
        raise RuntimeError("persistent NPU GR cumulative token counts must equal [W]")


def make_patched_prepare_input_ids(original: Any) -> Any:
    """D2D-publish persistent Beam tokens at the pinned NPU V1 copy boundary."""

    if getattr(original, _NPU_PREPARE_IDS_MARKER, False):
        return original

    @wraps(original)
    def prepare_input_ids(
        self: Any,
        scheduler_output: Any,
        num_reqs: int,
        total_num_scheduled_tokens: int,
        cu_num_tokens: Any,
    ) -> Any:
        frame = getattr(self, _FRAME_ATTR, None)
        if frame is None:
            if _persistent_decode_frame(scheduler_output) is not None:
                raise RuntimeError(
                    "persistent NPU GR Decode reached input copy without an active frame"
                )
            return original(
                self,
                scheduler_output,
                num_reqs,
                total_num_scheduled_tokens,
                cu_num_tokens,
            )
        if not isinstance(frame, NPUInputFrame) or not frame.active:
            raise RuntimeError("invalid persistent NPU GR input frame")
        if frame.input_copy_done:
            raise RuntimeError("persistent NPU GR input frame already copied its tokens")
        if frame.scheduler_output is not scheduler_output:
            raise RuntimeError("persistent NPU GR input frame SchedulerOutput changed")
        current_beam_data = getattr(scheduler_output, "beam_data", None)
        current_sidecars = getattr(scheduler_output, "gr_outputs", None)
        if (
            not isinstance(current_beam_data, dict)
            or not isinstance(current_sidecars, dict)
            or current_beam_data.get(frame.request_id) is not frame.metadata
            or current_sidecars.get(frame.request_id) is not frame.sidecar
        ):
            raise RuntimeError("persistent NPU GR input frame metadata identity changed")
        if getattr(scheduler_output, _GEOMETRY_ATTR, None) != (
            frame.request_id,
            1,
            frame.physical_rows,
        ):
            raise RuntimeError("persistent NPU GR physical geometry marker changed")
        if isinstance(num_reqs, bool) or not isinstance(num_reqs, int) or num_reqs != 1:
            raise RuntimeError("persistent NPU GR input preparation requires B=1")
        if (
            isinstance(total_num_scheduled_tokens, bool)
            or not isinstance(total_num_scheduled_tokens, int)
            or total_num_scheduled_tokens != frame.physical_rows
        ):
            raise RuntimeError("persistent NPU GR physical input count mismatch")
        _validate_cumulative_counts(cu_num_tokens, frame.physical_rows)
        scheduled_physical = _strict_int(
            scheduler_output.num_scheduled_tokens.get(frame.request_id),
            "scheduled physical token count",
            minimum=1,
        )
        if scheduled_physical != frame.physical_rows:
            raise RuntimeError("persistent NPU GR per-request physical count changed")
        total_physical = _strict_int(
            scheduler_output.total_num_scheduled_tokens,
            "total physical token count",
            minimum=1,
        )
        if total_physical != frame.physical_rows:
            raise RuntimeError("persistent NPU GR total physical count changed")

        input_batch = getattr(self, "input_batch", None)
        if list(getattr(input_batch, "req_ids", ())) != [frame.request_id]:
            raise RuntimeError("persistent NPU GR input batch request identity mismatch")
        states = getattr(self, "_gr_model_runner_states", None)
        state = states.get(frame.request_id) if isinstance(states, dict) else None
        if state is None or getattr(state, "released", None) is not False:
            raise RuntimeError("persistent NPU GR ModelRunner state is missing or released")
        if getattr(state, "request_id", None) != frame.request_id:
            raise RuntimeError("persistent NPU GR ModelRunner state request_id mismatch")
        sidecar = frame.sidecar
        step = sidecar.step
        if not (
            getattr(state, "_prepared_step", None)
            == getattr(state, "step", None)
            == getattr(state, "device_step", None)
            == step
            == sidecar.device_step
        ):
            raise RuntimeError("persistent NPU GR prepared/device step mismatch")

        destination = getattr(getattr(self, "input_ids", None), "gpu", None)
        source = getattr(state, "next_token_ids", None)
        if not torch.is_tensor(destination) or destination.ndim != 1:
            raise RuntimeError("persistent NPU GR input_ids.gpu must be a 1-D Tensor")
        if destination.dtype is not torch.int32:
            raise RuntimeError("persistent NPU GR input_ids.gpu must use int32")
        try:
            runner_device = torch.device(getattr(self, "device"))
        except (TypeError, RuntimeError):
            raise RuntimeError("persistent NPU GR runner device is invalid") from None
        if destination.device != runner_device:
            raise RuntimeError("persistent NPU GR input buffer is on the wrong device")
        if destination.numel() < frame.physical_rows:
            raise RuntimeError("persistent NPU GR input buffer is smaller than Beam width")
        if not torch.is_tensor(source) or tuple(source.shape) != (1, frame.physical_rows):
            raise RuntimeError("persistent NPU GR next_token_ids must have shape [1, W]")
        if source.dtype is not torch.int32 or source.device != destination.device:
            raise RuntimeError("persistent NPU GR next_token_ids dtype/device mismatch")
        # Bootstrap publishes ``canonical_sequence[:, :, 0]``, whose Beam
        # dimension is intentionally strided by the fixed generation extent.
        # ``copy_`` consumes that device view directly, without allocating a
        # contiguous staging tensor or materializing Beam tokens on the Host.
        destination[: frame.physical_rows].copy_(source[0])
        frame.input_copy_done = True
        return None

    setattr(prepare_input_ids, _NPU_PREPARE_IDS_MARKER, True)
    return prepare_input_ids


class _BeamOnlyBlockTable:
    """Execute-local proxy: Beam suffix writes belong to the session pool.

    Ascend's native slot kernel interprets W queries as consecutive Native
    positions, potentially indexing past the logical request's block row.
    Do not launch it just because incremental Beam attention ignores its output.
    Keep the original slot buffer, marked invalid, for native metadata plumbing.
    """

    def __init__(self, original: Any, frame: NPUInputFrame, slots: torch.Tensor):
        self.original = original
        self.frame = frame
        self.slots = slots
        self.active = True
        self.called = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def __getitem__(self, index: int) -> Any:
        return self.original[index]

    def compute_slot_mapping(
        self,
        num_reqs,
        query_start_loc,
        positions,
        positions_compressed_list=None,
        req_indices_compressed_list=None,
    ) -> None:
        if not self.active or not self.frame.active or self.called:
            raise RuntimeError("stale or repeated persistent NPU GR slot preparation")
        if not self.frame.input_copy_done:
            raise RuntimeError("persistent NPU GR slots require prepared device tokens")
        if type(num_reqs) is not int or num_reqs != 1:
            raise RuntimeError("persistent NPU GR slot preparation requires B=1")
        if positions_compressed_list is not None or req_indices_compressed_list is not None:
            raise RuntimeError("persistent NPU GR does not support compressed slot mapping")
        for value, shape, dtype in (
            (query_start_loc, (2,), torch.int32),
            (positions, (self.frame.physical_rows,), torch.int64),
        ):
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != shape
                or value.dtype is not dtype
                or value.device != self.slots.device
            ):
                raise RuntimeError("persistent NPU GR native slot input geometry mismatch")
        # Include padding: an old native slot must not survive in the tail.
        self.slots.fill_(PAD_SLOT_ID)
        self.called = True

    def compute_slot_mapping_draft(self, *args, **kwargs) -> None:
        raise RuntimeError("persistent NPU GR does not support draft slot mapping")


@contextmanager
def suspend_native_slot_mapping(runner: Any, scheduler_output: Any):
    """Replace only native slot computation during persistent NPU preparation."""
    frame = getattr(runner, _FRAME_ATTR, None)
    if frame is None:
        if _persistent_decode_frame(scheduler_output) is not None:
            raise RuntimeError("persistent NPU GR slot preparation requires an active frame")
        yield
        return
    if (
        not isinstance(frame, NPUInputFrame)
        or not frame.active
        or frame.scheduler_output is not scheduler_output
    ):
        raise RuntimeError("persistent NPU GR slot frame identity mismatch")
    _reject_unsupported_input_paths(runner, scheduler_output)
    if (
        getattr(runner, "use_compress", False)
        or getattr(runner, "uses_mrope", False)
        or getattr(runner, "uses_xdrope_dim", 0)
        or getattr(runner, "use_async_spec_decode", False)
    ):
        raise RuntimeError("unsupported persistent NPU GR position/slot preparation")
    batch = runner.input_batch
    if list(batch.req_ids) != [frame.request_id]:
        raise RuntimeError("persistent NPU GR slot request identity mismatch")
    original = batch.block_table
    if isinstance(original, _BeamOnlyBlockTable):
        raise RuntimeError("nested persistent NPU GR block-table proxy")
    groups = getattr(original, "block_tables", None)
    if not isinstance(groups, (tuple, list)) or len(groups) != 1:
        raise RuntimeError("persistent NPU GR slots require one native KV group")
    group = groups[0]
    if (
        getattr(group, "is_mamba_group", None) is not False
        or getattr(group, "use_hybrid_blocks", None) is not False
        or getattr(group, "pcp_world_size", None) != 1
        or getattr(group, "dcp_world_size", None) != 1
    ):
        raise RuntimeError("unsupported persistent NPU GR block-table group")
    slots = getattr(getattr(group, "slot_mapping", None), "gpu", None)
    if (
        not torch.is_tensor(slots)
        or slots.ndim != 1
        or slots.dtype is not torch.int32
        or slots.numel() < frame.physical_rows
        or slots.device != torch.device(runner.device)
    ):
        raise RuntimeError("persistent NPU GR native slot buffer is invalid")
    proxy = _BeamOnlyBlockTable(original, frame, slots)
    batch.block_table = proxy
    try:
        yield
        if not proxy.called:
            raise RuntimeError("persistent NPU GR native prepare skipped slot boundary")
    finally:
        proxy.active = False
        batch.block_table = original


__all__ = [
    "NPUInputFrame",
    "make_patched_execute_model",
    "make_patched_prepare_input_ids",
    "suspend_native_slot_mapping",
]
