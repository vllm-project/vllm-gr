# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU V1 persistent-GR prefix and fixed metadata binding.

The helpers mutate only backend-owned/fixed runner buffers and the Beam
context binding consumed by attention metadata.  They never query Prefix
Cache during Decode, rebuild a Native block table, copy Beam state to Host,
overwrite input tokens, or advance either lifecycle cursor.  The NPU backend
factory remains disabled until decision, KV-reorder, transport, and hardware
contracts are complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_gr.v1.engine.types import GRSchedulerOutput, GRRequestStage


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"persistent NPU GR {name} is invalid")
    return value


def _device(runner: Any) -> torch.device:
    try:
        return torch.device(getattr(runner, "device"))
    except (TypeError, RuntimeError):
        raise RuntimeError("persistent NPU GR runner device is invalid") from None


def _tensor(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    ndim: int,
) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != ndim:
        raise RuntimeError(f"persistent NPU GR {name} must be a {ndim}-D Tensor")
    if value.dtype is not dtype or value.device != device:
        raise RuntimeError(f"persistent NPU GR {name} dtype/device mismatch")
    if not value.is_contiguous():
        raise RuntimeError(f"persistent NPU GR {name} must be contiguous")
    return value


def _active_session(runner: Any, state: Any) -> tuple[Any, Any, torch.device]:
    if getattr(state, "released", None) is not False:
        raise RuntimeError("persistent NPU GR state is missing or released")
    request_id = getattr(state, "request_id", None)
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("persistent NPU GR state request_id is invalid")
    session = getattr(state, "session", None)
    if session is None or getattr(session, "session_id", None) != request_id:
        raise RuntimeError("persistent NPU GR state/session ownership mismatch")
    if getattr(session, "is_active", None) is not True:
        raise RuntimeError("persistent NPU GR Beam session is not active")
    if _integer(getattr(session, "slot", None), "session slot") != 0:
        raise RuntimeError("persistent NPU GR currently requires isolated slot zero")
    context = getattr(session, "context", None)
    if context is None or getattr(runner, "_beam_context", None) is not context:
        raise RuntimeError("persistent NPU GR Beam context ownership mismatch")
    sessions = getattr(runner, "_beam_sessions", None)
    if not isinstance(sessions, dict) or sessions.get(request_id) is not session:
        raise RuntimeError("persistent NPU GR runner/session ownership mismatch")
    return session, context, _device(runner)


def _input_batch(runner: Any, request_id: str) -> tuple[Any, int]:
    input_batch = getattr(runner, "input_batch", None)
    input_batch = getattr(input_batch, "_obj", input_batch)
    if input_batch is None:
        raise RuntimeError("persistent NPU GR input batch is missing")
    if _integer(getattr(input_batch, "num_reqs", None), "input batch size", minimum=1) != 1:
        raise RuntimeError("persistent NPU GR metadata requires B=1")
    if list(getattr(input_batch, "req_ids", ())) != [request_id]:
        raise RuntimeError("persistent NPU GR input batch request identity mismatch")
    mapping = getattr(input_batch, "req_id_to_index", None)
    if not isinstance(mapping, dict) or mapping.get(request_id) != 0:
        raise RuntimeError("persistent NPU GR request row mapping mismatch")
    return input_batch, 0


def _prefix_length(runner: Any, state: Any, session: Any) -> int:
    requests = getattr(runner, "requests", None)
    request = requests.get(state.request_id) if isinstance(requests, dict) else None
    if request is None or getattr(request, "req_id", state.request_id) != state.request_id:
        raise RuntimeError("persistent NPU GR Native request state is missing")
    request_prefix = _integer(
        getattr(request, "num_prompt_tokens", None), "Native request prefix length", minimum=1
    )
    session_state = getattr(session, "_state", None)
    session_prefix = _integer(
        getattr(session_state, "prefix_len", None), "session prefix length", minimum=1
    )
    beam_data = getattr(runner, "_current_beam_data", None)
    metadata = beam_data.get(state.request_id) if isinstance(beam_data, dict) else None
    if not isinstance(metadata, dict):
        raise RuntimeError("persistent NPU GR request metadata is missing")
    metadata_prefix = _integer(metadata.get("prefix_len"), "metadata prefix length", minimum=1)
    if request_prefix != session_prefix or request_prefix != metadata_prefix:
        raise RuntimeError("persistent NPU GR prefix length ownership mismatch")
    return request_prefix


def bind_prefill_prefix(runner: Any, state: Any) -> tuple[torch.Tensor]:
    """Bind the one Native prefix row after final Prefill, without a clone."""

    session, _, device = _active_session(runner, state)
    if getattr(state, "stage", None) is not GRRequestStage.PREFILL_RUNNING:
        raise RuntimeError("persistent NPU GR prefix binding requires Prefill state")
    if getattr(state, "step", None) != 0 or getattr(state, "device_step", None) != 0:
        raise RuntimeError("persistent NPU GR prefix binding is not at bootstrap step")
    if _integer(getattr(state, "beam_width", None), "Beam width", minimum=1) != _integer(
        getattr(session, "beam_width", None), "session Beam width", minimum=1
    ):
        raise RuntimeError("persistent NPU GR Beam width ownership mismatch")
    if (
        _integer(getattr(runner, "pcp_size", None), "PCP size", minimum=1) != 1
        or _integer(getattr(runner, "dcp_size", None), "DCP size", minimum=1) != 1
        or getattr(runner, "use_cp", None) is not False
    ):
        raise RuntimeError("persistent NPU GR prefix binding does not support CP")
    input_batch, request_index = _input_batch(runner, state.request_id)
    prefix_len = _prefix_length(runner, state, session)

    multi_group = getattr(input_batch, "block_table", None)
    groups = getattr(multi_group, "block_tables", None)
    if not isinstance(groups, (list, tuple)) or len(groups) != 1:
        raise RuntimeError("persistent NPU GR requires one Native KV-cache group")
    block_table = groups[0]
    if getattr(block_table, "is_mamba_group", None) is not False:
        raise RuntimeError("persistent NPU GR prefix binding does not support Mamba KV")
    if getattr(block_table, "use_hybrid_blocks", None) is not False:
        raise RuntimeError("persistent NPU GR prefix binding requires physical block IDs")
    block_size = _integer(getattr(block_table, "block_size", None), "block size", minimum=1)
    num_blocks = (prefix_len + block_size - 1) // block_size
    get_device_tensor = getattr(block_table, "get_device_tensor", None)
    if not callable(get_device_tensor):
        raise RuntimeError("persistent NPU GR Native block table has no device API")
    table = get_device_tensor(input_batch.num_reqs)
    table = _tensor(
        table,
        name="Native block table",
        dtype=torch.int32,
        device=device,
        ndim=2,
    )
    if table.shape[0] != 1 or table.shape[1] < num_blocks:
        raise RuntimeError("persistent NPU GR Native prefix block-table capacity mismatch")
    # A basic slice is a stable device-storage view; Native logical suffix
    # columns are deliberately excluded from the persistent attention binding.
    binding = table[request_index, :num_blocks]
    if not binding.is_contiguous() or binding.data_ptr() != table[request_index].data_ptr():
        raise RuntimeError("persistent NPU GR prefix binding is not a stable row view")
    return (binding,)


@dataclass(frozen=True, slots=True)
class _DecodeContract:
    session: Any
    context: Any
    device: torch.device
    sidecar: GRSchedulerOutput
    prefix_binding: tuple[torch.Tensor]
    beam_width: int
    prefix_len: int


def _native_metadata_buffers(
    runner: Any, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens = _tensor(
        getattr(runner, "seq_lens", None),
        name="seq_lens",
        dtype=torch.int32,
        device=device,
        ndim=1,
    )
    optimistic = _tensor(
        getattr(runner, "optimistic_seq_lens_cpu", None),
        name="optimistic_seq_lens_cpu",
        dtype=torch.int32,
        device=torch.device("cpu"),
        ndim=1,
    )
    query_start = getattr(getattr(runner, "query_start_loc", None), "gpu", None)
    query_start = _tensor(
        query_start,
        name="query_start_loc.gpu",
        dtype=torch.int32,
        device=device,
        ndim=1,
    )
    if seq_lens.numel() < 1 or optimistic.numel() < 1 or query_start.numel() < 2:
        raise RuntimeError("persistent NPU GR native metadata buffer capacity mismatch")
    return seq_lens, optimistic, query_start


def _decode_contract(
    runner: Any,
    state: Any,
    scheduler_output: Any,
    metadata: dict[str, Any],
) -> _DecodeContract:
    session, context, device = _active_session(runner, state)
    if not isinstance(metadata, dict):
        raise RuntimeError("persistent NPU GR Decode metadata must be a dict")
    if getattr(runner, "_current_scheduler_output", None) is not scheduler_output:
        raise RuntimeError("persistent NPU GR current SchedulerOutput identity mismatch")
    beam_data = getattr(scheduler_output, "beam_data", None)
    if not isinstance(beam_data, dict) or beam_data.get(state.request_id) is not metadata:
        raise RuntimeError("persistent NPU GR Decode metadata identity mismatch")
    sidecars = getattr(scheduler_output, "gr_outputs", None)
    sidecar = sidecars.get(state.request_id) if isinstance(sidecars, dict) else None
    if not isinstance(sidecar, GRSchedulerOutput):
        raise RuntimeError("persistent NPU GR Decode requires typed sidecar")
    if (
        sidecar.request_id != state.request_id
        or sidecar.stage is not GRRequestStage.DECODE_RUNNING
        or sidecar.is_prefill
    ):
        raise RuntimeError("persistent NPU GR Decode sidecar ownership mismatch")
    if getattr(state, "stage", None) not in (
        GRRequestStage.PREFILL_BOOTSTRAPPED,
        GRRequestStage.DECODE_RUNNING,
    ):
        raise RuntimeError("persistent NPU GR state is not Decode-preparable")
    step = _integer(getattr(state, "step", None), "state step", minimum=1)
    max_tokens = _integer(getattr(state, "max_tokens", None), "generation steps", minimum=1)
    if step >= max_tokens:
        raise RuntimeError("persistent NPU GR Decode step is terminal or out of range")
    if not (
        _integer(getattr(state, "device_step", None), "state device_step", minimum=1)
        == _integer(getattr(state, "_prepared_step", None), "prepared step", minimum=1)
        == _integer(sidecar.step, "sidecar step", minimum=1)
        == _integer(sidecar.device_step, "sidecar device_step", minimum=1)
        == step
    ):
        raise RuntimeError("persistent NPU GR prepared/device step mismatch")
    if _integer(metadata.get("generation_index"), "generation index", minimum=1) != step:
        raise RuntimeError("persistent NPU GR metadata/state step mismatch")
    for name in ("constraint_step", "beam_decode_steps"):
        if _integer(metadata.get(name), name, minimum=1) != step:
            raise RuntimeError(f"persistent NPU GR metadata {name}/state step mismatch")
    if _integer(metadata.get("decode_step"), "metadata decode step") != step - 1:
        raise RuntimeError("persistent NPU GR metadata/session step mismatch")
    if metadata.get("is_last_step") is not sidecar.is_final:
        raise RuntimeError("persistent NPU GR final marker mismatch")
    if sidecar.is_final is not (step + 1 == max_tokens):
        raise RuntimeError("persistent NPU GR sidecar final step mismatch")
    session_step = _integer(getattr(session, "decode_step", None), "session decode step")
    if session_step != step - 1:
        raise RuntimeError("persistent NPU GR session/state step mismatch")
    width = _integer(getattr(state, "beam_width", None), "Beam width", minimum=1)
    if not (
        _integer(getattr(session, "beam_width", None), "session Beam width", minimum=1)
        == _integer(sidecar.physical_row_count, "physical row count", minimum=1)
        == _integer(metadata.get("beam_width"), "metadata Beam width", minimum=1)
        == width
    ):
        raise RuntimeError("persistent NPU GR Beam width ownership mismatch")
    if _integer(sidecar.logical_token_count, "logical token count", minimum=1) != 1:
        raise RuntimeError("persistent NPU GR sidecar must retain one logical token")
    if (
        metadata.get("persistent_runner_state") is not True
        or metadata.get("is_beam_decode") is not True
    ):
        raise RuntimeError("persistent NPU GR Decode metadata lost persistence markers")
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled, dict) or set(scheduled) != {state.request_id}:
        raise RuntimeError("persistent NPU GR Decode scheduling ownership mismatch")
    if (
        _integer(scheduled.get(state.request_id), "physical scheduled tokens", minimum=1) != width
        or _integer(
            getattr(scheduler_output, "total_num_scheduled_tokens", None),
            "total physical scheduled tokens",
            minimum=1,
        )
        != width
    ):
        raise RuntimeError("persistent NPU GR physical scheduling geometry mismatch")
    if getattr(scheduler_output, "_gr_npu_logical_geometry", None) != (
        state.request_id,
        1,
        width,
    ):
        raise RuntimeError("persistent NPU GR logical geometry marker mismatch")
    prefix_len = _integer(metadata.get("prefix_len"), "metadata prefix length", minimum=1)
    session_prefix = _integer(
        getattr(getattr(session, "_state", None), "prefix_len", None),
        "session prefix length",
        minimum=1,
    )
    if prefix_len != session_prefix:
        raise RuntimeError("persistent NPU GR Decode prefix length mismatch")
    if _integer(metadata.get("cache_len"), "metadata cache length") != prefix_len:
        raise RuntimeError("persistent NPU GR Decode cache length must equal prefix length")
    _input_batch(runner, state.request_id)
    binding = getattr(state, "prefix_binding", None)
    if not isinstance(binding, tuple) or len(binding) != 1:
        raise RuntimeError("persistent NPU GR prefix binding is missing or multi-group")
    prefix_row = _tensor(
        binding[0],
        name="prefix binding",
        dtype=torch.int32,
        device=device,
        ndim=1,
    )
    if prefix_row.numel() < 1:
        raise RuntimeError("persistent NPU GR prefix binding is empty")
    existing = getattr(context, "_gr_persistent_prefix_binding", None)
    if existing is not None and existing is not binding:
        raise RuntimeError("persistent NPU GR context prefix binding identity changed")

    decision = getattr(context, "decision", None)
    sequence = _tensor(
        getattr(decision, "sequence", None),
        name="canonical sequence",
        dtype=torch.int32,
        device=device,
        ndim=3,
    )
    scores = _tensor(
        getattr(decision, "beam_scores", None),
        name="canonical scores",
        dtype=torch.float32,
        device=device,
        ndim=2,
    )
    group_tokens = _tensor(
        getattr(decision, "group_token_num", None),
        name="canonical group token counts",
        dtype=torch.int32,
        device=device,
        ndim=2,
    )
    if (
        sequence.shape[0] != 1
        or sequence.shape[1] != width
        or sequence.shape[2] < max_tokens
        or scores.shape[0] != 1
        or scores.shape[1] != width
        or group_tokens.shape[0] != 1
        or group_tokens.shape[1] != width
    ):
        raise RuntimeError("persistent NPU GR canonical buffer capacity mismatch")
    if (
        _integer(
            getattr(decision, "worker_decode_steps", None),
            "canonical decision step",
            minimum=1,
        )
        != step
    ):
        raise RuntimeError("persistent NPU GR canonical decision/state step mismatch")
    decode_step = _tensor(
        getattr(context, "decode_step_buf", None),
        name="decode_step_buf",
        dtype=torch.int32,
        device=device,
        ndim=1,
    )
    decode_step_1based = _tensor(
        getattr(context, "decode_step_1based_buf", None),
        name="decode_step_1based_buf",
        dtype=torch.int32,
        device=device,
        ndim=1,
    )
    slot_table = _tensor(
        getattr(context, "xllm_block_table_buf", None),
        name="xllm_block_table_buf",
        dtype=torch.int32,
        device=device,
        ndim=1,
    )
    if decode_step.shape != (1,) or decode_step_1based.shape != (1,) or slot_table.numel() < 1:
        raise RuntimeError("persistent NPU GR fixed context buffer capacity mismatch")
    _native_metadata_buffers(runner, device)
    return _DecodeContract(session, context, device, sidecar, binding, width, prefix_len)


def prepare_decode_metadata(
    runner: Any,
    state: Any,
    scheduler_output: Any,
    metadata: dict[str, Any],
) -> None:
    """Publish current device step/pool slot before native NPU preparation."""

    contract = _decode_contract(runner, state, scheduler_output, metadata)
    context = contract.context
    session_step = contract.session.decode_step
    # All validation completes before any fixed buffer or binding is mutated.
    context.decode_step_buf.fill_(session_step)
    context.decode_step_1based_buf.fill_(session_step + 1)
    context.xllm_block_table_buf[:1].fill_(contract.session.slot)
    context._gr_persistent_prefix_binding = contract.prefix_binding


def finish_decode_metadata(runner: Any, state: Any, metadata: dict[str, Any]) -> None:
    """Correct native length buffers after physical-W NPU input preparation."""

    scheduler_output = getattr(runner, "_current_scheduler_output", None)
    contract = _decode_contract(runner, state, scheduler_output, metadata)
    seq_lens, optimistic, _ = _native_metadata_buffers(runner, contract.device)
    logical_length = contract.prefix_len + state.step
    # Host integers are request metadata, not device Beam state; no device
    # value is inspected or synchronized here.
    seq_lens[0].fill_(logical_length)
    optimistic[0].fill_(logical_length)


__all__ = ["bind_prefill_prefix", "finish_decode_metadata", "prepare_decode_metadata"]
