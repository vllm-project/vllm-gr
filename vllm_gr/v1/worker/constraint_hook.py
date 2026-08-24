# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-neutral Constraint Tables raw-logits sampler (issue #246).

Pure constrained batches bypass vLLM's ordinary sampler and construct a fresh,
fixed-width ``SamplerOutput`` directly from ``ConstraintBackend.select()``.
Ordinary and chunk-only batches still use the unmodified platform sampler.
``_beam_collapse_bookkeeping`` derives ``actual_K`` from the direct output's
width ``W`` (so decode produces ``W*W`` candidates per logical request).

Wire contract (P2):

- decode: one session flat ``[1, W*W]`` — ``W`` beams × ``W`` candidates,
  stride ``W`` per parent block (each block is exactly one operator row).
- prefill: one wide ``[1, W*W]`` segment — ``W`` valid candidates followed by
  ``-inf`` remux padding.
- padding slots in operator rows carry the finite ``INVALID_LOGIT`` sentinel;
  the frontend filters them by threshold, never by ``np.isfinite``.

Guards:

- ``constraint_step`` must be in ``(0, 1, 2)`` — a Constraint Table request
  never degrades to unconstrained sampling (P6).
- A constrained step must not share logits rows with unconstrained requests,
  and all constrained rows must share one ``beam_width`` (mixed routing is P8).
- The operator is only invoked on the last prefill chunk (``W_logits == 1``);
  intermediate chunks carry no logits rows and keep the normal dummy path.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintMetadata,
    effective_temperatures,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Fixed 3-level FSM: constrained prefill (0) plus two constrained decode steps.
_VALID_CONSTRAINT_STEPS = (0, 1, 2)
_MIN_CONSTRAINT_PREFIX_WIDTH = len(_VALID_CONSTRAINT_STEPS)

_DEFAULT_TEMPERATURE = 1.0

ConstrainedRow = tuple[Any, int, int, dict[str, Any]]


def validate_constraint_step(step: Any) -> int:
    """Return ``step`` normalized to a valid Constraint Table FSM step.

    ``constraint_step`` outside ``(0, 1, 2)`` is a hard error (never a silent
    fall back to unconstrained sampling).  Accepts any int-like value
    (``operator.index``), rejecting bools and floats.
    """
    if isinstance(step, bool):
        raise RuntimeError(
            f"constraint_step must be one of {_VALID_CONSTRAINT_STEPS}, got bool {step!r}"
        )
    try:
        normalized = operator.index(step)
    except TypeError:
        normalized = -1
    if normalized not in _VALID_CONSTRAINT_STEPS:
        raise RuntimeError(
            f"constraint_step must be one of {_VALID_CONSTRAINT_STEPS}, "
            f"got {step!r}; Constraint Tables never degrade to unconstrained "
            "sampling"
        )
    return normalized


def collect_constrained_rows(
    request_row_configs: Sequence[Any],
    beam_data: dict[str, Any],
) -> list[ConstrainedRow]:
    """Locate constrained requests and their sampler/logits row ranges.

    Walks ``request_row_configs`` with the same ``gpu_row_cursor`` traversal as
    ``_remux_logprobs_tensors`` (model_runner_common), so the returned
    ``row_start`` values index directly into the sampler-output tensors and the
    ``logits`` argument of the patched ``_sample``.

    Returns ``(req_id, row_start, W_logits, mdata)`` tuples in sampler-row
    order.  Requests with ``W_logits == 0`` (intermediate chunked-prefill
    chunks) produce no logits rows and never run the operator.  Every
    ``is_constrained`` request's ``constraint_step`` is validated, including
    chunk-only visits, so an out-of-range step fails fast.
    """
    collected: list[ConstrainedRow] = []
    gpu_row_cursor = 0
    for req_id, _, W_logits, *_ in request_row_configs:
        mdata = beam_data.get(req_id) if req_id is not None else None
        if mdata is not None and mdata.get("is_constrained", False):
            # ``constraint_step`` is mandatory for every constrained request
            # (the scheduler_metadata builder guarantees it).  Missing is a
            # protocol error: defaulting to 0 would silently treat a decode
            # request as prefill and build the wrong sequence_group.
            constraint_step = mdata.get("constraint_step")
            if constraint_step is None:
                raise RuntimeError(
                    f"constrained request {req_id!r} is missing constraint_step "
                    "in beam metadata; refusing to guess prefill — the "
                    "scheduler metadata chain must provide it"
                )
            validate_constraint_step(constraint_step)
            if W_logits > 0:
                collected.append((req_id, gpu_row_cursor, W_logits, mdata))
        gpu_row_cursor += W_logits
    return collected


def build_sequence_group_for_step(
    rows: Sequence[ConstrainedRow],
    device: Any,
    step: int,
) -> torch.Tensor:
    """Build flat ``prefix_token_ids [rows, max_steps]`` for the Backend.

    Each logits row receives one prefix row. Prefill content is ignored;
    decode rows carry the last ``step`` beam tokens in columns ``[:step]``.

    Cross-checks ``len(beam_suffix[b]) >= step``: the per-request
    ``constraint_step`` is authoritative and must not disagree with the suffix
    the request actually carries (a global counter inference would hide this).
    """
    step = validate_constraint_step(step)
    max_step = _MIN_CONSTRAINT_PREFIX_WIDTH
    for _, _, _, mdata in rows:
        request_steps = mdata.get("max_decode_steps") or 0
        if request_steps > max_step:
            max_step = request_steps

    if step == 0:
        for req_id, _, W_logits, _ in rows:
            if W_logits != 1:
                raise RuntimeError(
                    f"constrained prefill request {req_id!r} has {W_logits} logits rows, "
                    "expected exactly 1"
                )
        row_count = sum(W_logits for _, _, W_logits, _ in rows)
        return torch.zeros([row_count, max_step], dtype=torch.int32, device=device)

    widths = {mdata["beam_width"] for _, _, _, mdata in rows}
    if len(widths) != 1:
        raise RuntimeError(
            f"constrained step mixes beam_widths {sorted(widths)}; "
            "mixed-width Constraint Table batches are not supported yet (P8)"
        )
    width = next(iter(widths))

    flat: list[int] = []
    for req_id, _, W_logits, mdata in rows:
        if W_logits != width:
            raise RuntimeError(
                f"constrained decode request {req_id!r} has {W_logits} logits rows, "
                f"expected beam_width={width}"
            )
        suffix = mdata.get("beam_suffix") or []
        for b in range(width):
            seq = list(suffix[b][-step:]) if b < len(suffix) else []
            if len(seq) != step:
                raise RuntimeError(
                    f"constrained decode step {step} needs the last {step} "
                    f"token(s) per beam, got {len(seq)} from "
                    f"beam_suffix[{b}]={suffix[b]!r}; beam_suffix length is "
                    "inconsistent with constraint_step"
                )
            flat.extend(seq)
            flat.extend([0] * (max_step - step))
    return torch.tensor(flat, dtype=torch.int32, device=device).reshape(-1, max_step)


def _temperatures_for_rows(runner: Any, rows: Sequence[int], device: Any) -> torch.Tensor:
    """Return a ``[rows]`` / ``[1]`` temperature tensor for constrained rows.

    Prefers the proxy-mapped per-logits-row temperature on the runner's
    sampling metadata (``InputBatchProxy`` maps it to one entry per logits row);
    falls back to a scalar ``1.0`` default.  Always float32 on ``device``.
    """
    s_meta = getattr(getattr(runner, "input_batch", None), "sampling_metadata", None)
    temps = getattr(s_meta, "temperature", None) if s_meta is not None else None
    if isinstance(temps, torch.Tensor) and temps.numel() > 0:
        prepared = temps.detach().to(device=device, dtype=torch.float32).reshape(-1)
        if prepared.numel() == 1:
            return effective_temperatures(prepared).contiguous()
        if prepared.numel() >= max(rows) + 1:
            row_start = rows[0]
            row_end = row_start + len(rows)
            if list(rows) == list(range(row_start, row_end)):
                selected = prepared[row_start:row_end]
            else:
                selected = prepared.index_select(
                    0,
                    torch.tensor(rows, dtype=torch.long, device=prepared.device),
                )
            return effective_temperatures(selected).contiguous()
    return torch.tensor([_DEFAULT_TEMPERATURE], dtype=torch.float32, device=device)


def _prefix_width(mdata: dict[str, Any]) -> int:
    request_steps = mdata.get("max_decode_steps") or 0
    return max(_MIN_CONSTRAINT_PREFIX_WIDTH, int(request_steps))


def _expanded_global_rows(rows: Sequence[ConstrainedRow]) -> list[int]:
    return [
        global_row
        for _, row_start, W_logits, _ in rows
        for global_row in range(row_start, row_start + W_logits)
    ]


def _group_logits(logits: torch.Tensor, rows: Sequence[ConstrainedRow]) -> torch.Tensor:
    """Return a view for contiguous groups, copying only interleaved groups."""

    first_start = rows[0][1]
    expected_start = first_start
    for _, row_start, W_logits, _ in rows:
        if row_start != expected_start:
            return torch.cat(
                [logits[start : start + count] for _, start, count, _ in rows],
                dim=0,
            )
        expected_start += W_logits
    return logits[first_start:expected_start]


def run_constrained_partition(
    runner: Any,
    logits: torch.Tensor,
    rows: Sequence[ConstrainedRow],
    width: int,
    backend: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Run one Backend call per ``(step, prefix width)`` partition."""

    device = logits.device
    global_rows = _expanded_global_rows(rows)
    expected_rows = list(range(logits.shape[0]))
    if global_rows != expected_rows:
        raise RuntimeError(
            "constrained logits row mapping must cover every logits row exactly once; "
            f"mapped={global_rows}, logits_rows={logits.shape[0]}"
        )

    grouped: dict[tuple[int, int], list[ConstrainedRow]] = defaultdict(list)
    for row in rows:
        mdata = row[3]
        step = validate_constraint_step(mdata.get("constraint_step", 0))
        grouped[(step, _prefix_width(mdata))].append(row)

    parts: list[tuple[list[int], Any]] = []
    for (step, _), group in grouped.items():
        group_rows = _expanded_global_rows(group)
        prefix_token_ids = build_sequence_group_for_step(
            group,
            device,
            step,
        )
        result = backend.select(
            _group_logits(logits, group),
            ConstraintMetadata(
                prefix_token_ids=prefix_token_ids,
                temperatures=_temperatures_for_rows(runner, group_rows, device),
                current_step=step,
                top_k=width,
            ),
        )
        parts.append((group_rows, result))

    if len(parts) == 1:
        _, result = parts[0]
        return result.token_ids, result.logprobs, result.valid_mask, global_rows

    first_result = parts[0][1]
    shape = (logits.shape[0], width)
    top_tokens = torch.empty(shape, dtype=first_result.token_ids.dtype, device=device)
    top_logprobs = torch.empty(shape, dtype=first_result.logprobs.dtype, device=device)
    valid_mask = torch.empty(shape, dtype=torch.bool, device=device)
    for part_rows, result in parts:
        row_indices = torch.tensor(part_rows, dtype=torch.long, device=device)
        top_tokens.index_copy_(0, row_indices, result.token_ids)
        top_logprobs.index_copy_(0, row_indices, result.logprobs)
        valid_mask.index_copy_(0, row_indices, result.valid_mask)
    return top_tokens, top_logprobs, valid_mask, global_rows


def build_constrained_sampler_output(
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    valid_mask: torch.Tensor,
    width: int,
) -> Any:
    """Construct the fixed-width vLLM output without resizing sampler storage."""

    expected_shape = (top_tokens.shape[0], width)
    if tuple(top_tokens.shape) != expected_shape:
        raise RuntimeError(
            f"constraint token_ids shape {tuple(top_tokens.shape)} must be {expected_shape}"
        )
    if top_tokens.dtype != torch.int32:
        raise RuntimeError(f"constraint token_ids must be int32, got {top_tokens.dtype}")
    if tuple(top_logprobs.shape) != expected_shape:
        raise RuntimeError(
            f"constraint logprobs shape {tuple(top_logprobs.shape)} must be {expected_shape}"
        )
    if top_logprobs.dtype != torch.float32:
        raise RuntimeError(f"constraint logprobs must be float32, got {top_logprobs.dtype}")
    if valid_mask.shape != top_logprobs.shape:
        raise RuntimeError(
            f"constraint valid_mask shape {tuple(valid_mask.shape)} must match "
            f"logprobs shape {tuple(top_logprobs.shape)}"
        )
    if valid_mask.dtype != torch.bool:
        raise RuntimeError(f"constraint valid_mask must be bool, got {valid_mask.dtype}")
    if valid_mask.device != top_logprobs.device:
        raise RuntimeError(
            f"constraint valid_mask device {valid_mask.device} must match "
            f"logprobs device {top_logprobs.device}"
        )
    if top_tokens.device != top_logprobs.device:
        raise RuntimeError(
            f"constraint token_ids device {top_tokens.device} must match "
            f"logprobs device {top_logprobs.device}"
        )

    # valid_mask is the Backend validity contract. INVALID_LOGIT is its stable
    # encoding in the existing sampler/remux wire format.
    wire_logprobs = top_logprobs.masked_fill(~valid_mask, INVALID_LOGIT)

    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    return SamplerOutput(
        sampled_token_ids=top_tokens[:, :1].contiguous(),
        logprobs_tensors=LogprobsTensors(
            logprob_token_ids=top_tokens,
            logprobs=wire_logprobs,
            selected_token_ranks=torch.zeros(
                top_tokens.shape[0],
                dtype=torch.int64,
                device=top_tokens.device,
            ),
        ),
    )


def _validate_homogeneous_step(
    request_row_configs: Sequence[Any],
    beam_data: dict[str, Any],
) -> None:
    """Reject a constrained step that shares rows with unconstrained logprobs.

    The direct constrained output owns every logits row and has one fixed
    candidate width. A non-constrained logits row therefore cannot share this
    path. Rows without logits (``W_logits == 0``) are not part of the guard.
    """
    for req_id, _, W_logits, *_ in request_row_configs:
        if W_logits == 0:
            continue
        mdata = beam_data.get(req_id) if req_id is not None else None
        if mdata is None or not mdata.get("is_constrained", False):
            raise RuntimeError(
                "a constrained step cannot share the sampler output with "
                f"unconstrained logprobs rows (req_id={req_id!r}); mixed-batch "
                "Constraint Table routing is not supported yet (P8)"
            )


def try_sample_constrained(
    runner: Any,
    logits: torch.Tensor | None,
    spec_decode_metadata: Any = None,
) -> Any | None:
    """Return a direct constrained output, or ``None`` for the normal sampler.

    Short-circuits for pure-completion and ordinary Beam steps (no constrained
    rows).  Requires the runner to carry ``_current_beam_data`` (from
    ``make_patched_execute_model``) and ``_current_scheduler_output`` (its
    ``_gr_request_row_configs``), both published before sampling runs.
    """
    beam_data = getattr(runner, "_current_beam_data", None)
    if not beam_data:
        return None
    scheduler_output = getattr(runner, "_current_scheduler_output", None)
    request_row_configs = (
        getattr(scheduler_output, "_gr_request_row_configs", None)
        if scheduler_output is not None
        else None
    )
    if not request_row_configs:
        return None

    rows = collect_constrained_rows(request_row_configs, beam_data)
    if not rows:
        return None

    if logits is None:
        raise RuntimeError("constrained sampling requires raw logits, got None")
    if spec_decode_metadata is not None:
        raise RuntimeError("constraint_table does not support speculative decoding")

    _validate_homogeneous_step(request_row_configs, beam_data)

    widths = {mdata["beam_width"] for _, _, _, mdata in rows}
    if len(widths) != 1:
        raise RuntimeError(
            f"constrained step mixes beam_widths {sorted(widths)}; "
            "mixed-width Constraint Table batches are not supported yet (P8)"
        )
    width = next(iter(widths))

    backend = getattr(runner, "constraint_backend", None)
    if backend is None:
        from vllm_gr.config import GRConfig

        config = GRConfig.from_vllm_config(getattr(runner, "vllm_config", None))
        if config.constraint_table.enabled:
            raise RuntimeError(
                "constraint_table is enabled but runner.constraint_backend is None; "
                "Worker startup did not complete and unconstrained sampling is forbidden"
            )
        raise RuntimeError(
            "received constrained Beam metadata while constraint_table is disabled; "
            "unconstrained sampling is forbidden"
        )

    update_async_tokens = getattr(
        getattr(runner, "input_batch", None),
        "update_async_output_token_ids",
        None,
    )
    if callable(update_async_tokens):
        update_async_tokens()

    top_tokens, top_logprobs, valid_mask, global_rows = run_constrained_partition(
        runner, logits, rows, width, backend
    )
    if global_rows != list(range(logits.shape[0])):
        raise RuntimeError("direct constrained output rows must remain in global logits order")
    return build_constrained_sampler_output(top_tokens, top_logprobs, valid_mask, width)
