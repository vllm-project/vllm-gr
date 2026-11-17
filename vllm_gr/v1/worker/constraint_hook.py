# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Platform-neutral Constraint Tables raw-logits hooks (issue #246).

A runner-owned canonical ``ConstraintBackend`` bypasses vLLM's ordinary sampler
and constructs a fresh fixed-width ``SamplerOutput`` directly.  When no
canonical Backend exists, the legacy resource-injection path remains unchanged:
the ordinary sampler runs first and its constrained rows are overwritten in
place by the legacy execution adapter.

Wire contract (P2):

- constrained sampling emits ``[rows, K]`` candidate rows, where K is the
  candidate width and is independent of the beam width W (``rows == W``);
- the shared worker-side Beam decision consumes those rows on CUDA or NPU;
- Beam operators remain outside the current model-forward FULL Graph.

Guards:

- ``constraint_step`` must be in ``(0, 1, 2)`` — a Constraint Table request
  never degrades to unconstrained sampling (P6).
- A constrained step must not share the sampler output with unconstrained
  logprobs rows, and all constrained rows must share one ``beam_width``
  (mixed-batch / mixed-width routing is P8 scope).
- The operator is only invoked on the last prefill chunk (``W_logits == 1``);
  intermediate chunks carry no logits rows and keep the normal dummy path.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_gr.constraints.backend import ConstraintMetadata

logger = init_logger(__name__)

# Fixed 3-level FSM: constrained prefill (0) plus two constrained decode steps.
_VALID_CONSTRAINT_STEPS = (0, 1, 2)
_CONSTRAINT_PREFIX_WIDTH = len(_VALID_CONSTRAINT_STEPS)

_DEFAULT_TEMPERATURE = 1.0


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
) -> list[tuple[Any, int, int, dict[str, Any]]]:
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
    collected: list[tuple[Any, int, int, dict[str, Any]]] = []
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
    rows: Sequence[tuple[Any, int, int, dict[str, Any]]],
    device: Any,
    step: int,
) -> torch.Tensor:
    """Build the operator ``sequence_group`` tensor for constrained rows.

    - prefill (step 0): ``[B, 1, 1]`` zeros — the operator ignores content.
    - decode (steps 1/2): ``[B, W, max_step]`` (int32), one request per ``B``;
      each beam writes its last ``step`` beam tokens into ``[b, :step]``
      (step 1 reads position 0 = ``s_a``; step 2 reads positions 0..1 =
      ``s_a, s_b``).  ``max_step`` is ``max(step, max_decode_steps) + 1``, so
      the operator's ``max_step > current_step`` check always passes.

    Cross-checks ``len(beam_suffix[b]) >= step``: the per-request
    ``constraint_step`` is authoritative and must not disagree with the suffix
    the request actually carries (a global counter inference would hide this).
    """
    step = validate_constraint_step(step)
    if step == 0:
        return torch.zeros([len(rows), 1, 1], dtype=torch.int32, device=device)

    widths = {mdata["beam_width"] for _, _, _, mdata in rows}
    if len(widths) != 1:
        raise RuntimeError(
            f"constrained step mixes beam_widths {sorted(widths)}; "
            "mixed-width Constraint Table batches are not supported yet (P8)"
        )
    width = next(iter(widths))

    max_step = step + 1
    for _, _, _, mdata in rows:
        request_steps = mdata.get("max_decode_steps") or 0
        if request_steps + 1 > max_step:
            max_step = request_steps + 1

    flat: list[int] = []
    for _, _, _, mdata in rows:
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
    return torch.tensor(flat, dtype=torch.int32, device=device).reshape(len(rows), width, max_step)


def _temperatures_for_rows(runner: Any, rows: Sequence[int], device: Any) -> torch.Tensor:
    """Return a ``[rows]`` / ``[1]`` temperature tensor for constrained rows.

    Prefers the proxy-mapped per-logits-row temperature on the runner's
    sampling metadata (``InputBatchProxy`` maps it to one entry per logits row);
    falls back to the V2 ``Sampler.sampling_states`` UVA temperature indexed by
    ``expanded_idx_mapping`` when ``input_batch.sampling_metadata`` is absent;
    otherwise a scalar ``1.0`` default.  Always float32 on ``device``.
    """
    s_meta = getattr(getattr(runner, "input_batch", None), "sampling_metadata", None)
    temps = getattr(s_meta, "temperature", None) if s_meta is not None else None
    if isinstance(temps, torch.Tensor) and temps.numel() > 0:
        temps = temps.detach().float().to(device)
        if temps.numel() == 1:
            return temps
        if temps.numel() >= max(rows) + 1:
            return temps.index_select(0, torch.tensor(rows, dtype=torch.long, device=temps.device))

    # V2 adapter: no ``sampling_metadata``.  Read per-request temperatures from
    # the sampler's UVA-backed ``SamplingStates`` and map each logits row back
    # to its request via ``input_batch.expanded_idx_mapping``.
    sampler = getattr(runner, "sampler", None)
    sampling_states = getattr(sampler, "sampling_states", None)
    gpu_temps = getattr(getattr(sampling_states, "temperature", None), "gpu", None)
    input_batch = getattr(runner, "input_batch", None)
    expanded_idx_mapping = getattr(input_batch, "expanded_idx_mapping", None)
    if isinstance(gpu_temps, torch.Tensor) and gpu_temps.numel() > 0:
        if gpu_temps.numel() == 1:
            return gpu_temps.detach().float().to(device)
        if isinstance(expanded_idx_mapping, torch.Tensor):
            row_idx = torch.tensor(rows, dtype=torch.long, device=expanded_idx_mapping.device)
            return gpu_temps[expanded_idx_mapping[row_idx]].detach().float().to(device)
    return torch.tensor([_DEFAULT_TEMPERATURE], dtype=torch.float32, device=device)


def _update_async_output_token_ids(runner: Any) -> None:
    """Refresh async output token ids, no-oping when the batch has no such hook.

    The V1 ``InputBatchProxy`` exposes ``update_async_output_token_ids``; the
    V2 ``InputBatch`` dataclass does not.  Constraint sampling still works under
    V2 without that refresh (the V2 async output copies from the sampler output
    directly), so the call is a no-op there instead of a hard requirement.
    """
    update = getattr(getattr(runner, "input_batch", None), "update_async_output_token_ids", None)
    if callable(update):
        update()


def run_constrained_partition(
    runner: Any,
    logits: torch.Tensor,
    rows: Sequence[tuple[Any, int, int, dict[str, Any]]],
    width: int,
    adapter: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Run the constrained operator on every constrained row group.

    Requests are grouped by their ``constraint_step`` and each step group is
    submitted as a *single* batch ``rec_constrained_topk`` call (the operator
    contract is already ``[B*W, vocab]`` logits + ``[B, W, max_step]``
    sequence_group), replacing the previous one call per request.  Requests at
    different FSM steps still run in separate batches because the operator's
    candidate table selection depends on ``current_step``.  Operator output
    rows are re-gathered to the canonical sampler/logits row order (the
    ``global_rows`` returned here is ``range(n)``), so the result tensors are
    directly aligned with the caller's ``logits`` tensor.
    """
    device = getattr(runner, "device", logits.device)
    top_tokens_parts: list[torch.Tensor] = []
    top_logprobs_parts: list[torch.Tensor] = []
    valid_mask_parts: list[torch.Tensor] = []
    global_rows: list[int] = []

    # Step groups are submitted in ascending constraint_step order, which is
    # not necessarily the sampler/logits row order: requests at different FSM
    # decode steps can occupy interleaved or inverted row ranges.  The direct
    # constrained sampler output must stay aligned to the global sampler/logits
    # row indices, so the group-ordered operator results are re-gathered into
    # canonical row order before returning.
    by_step: dict[int, list[tuple[Any, int, int, dict[str, Any]]]] = {}
    for item in rows:
        step = validate_constraint_step(item[3].get("constraint_step", 0))
        by_step.setdefault(step, []).append(item)

    for step in sorted(by_step):
        step_rows = by_step[step]
        sequence_group = build_sequence_group_for_step(step_rows, device, step)
        group_rows: list[int] = []
        for _, row_start, W_logits, _ in step_rows:
            group_rows.extend(range(row_start, row_start + W_logits))
        result = adapter.run(
            logits[group_rows],
            sequence_group,
            _temperatures_for_rows(runner, group_rows, device),
            step,
            width,
        )
        top_tokens_parts.append(result.top_tokens)
        top_logprobs_parts.append(result.top_logprobs)
        valid_mask_parts.append(result.valid_mask)
        global_rows.extend(group_rows)

    top_tokens = torch.cat(top_tokens_parts, dim=0)
    top_logprobs = torch.cat(top_logprobs_parts, dim=0)
    valid_mask = torch.cat(valid_mask_parts, dim=0)

    n_rows = len(global_rows)
    if global_rows != list(range(n_rows)):
        if sorted(global_rows) != list(range(n_rows)):
            raise RuntimeError(
                "constrained step partition must cover every logits row "
                f"exactly once; mapped={global_rows}"
            )
        # global_rows is a permutation of range(n_rows): build the inverse
        # position mapping and gather each row back to its canonical index.
        inverse = [0] * n_rows
        for position, row in enumerate(global_rows):
            inverse[row] = position
        order = torch.tensor(inverse, dtype=torch.long, device=top_tokens.device)
        top_tokens = top_tokens.index_select(0, order)
        top_logprobs = top_logprobs.index_select(0, order)
        valid_mask = valid_mask.index_select(0, order)
        global_rows = list(range(n_rows))

    return (
        top_tokens,
        top_logprobs,
        valid_mask,
        global_rows,
    )


def patch_sampler_output_for_constrained(
    sampler_output: Any,
    global_rows: Sequence[int],
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
    valid_mask: torch.Tensor,
    width: int,
) -> None:
    """Overwrite constrained sampler-output rows with operator results.

    Writes the operator ``[rows, width]`` result in place over the sampled
    token / logprob tensors for ``global_rows``, then shrinks the logprob
    tensors to width ``width`` so ``_beam_collapse_bookkeeping`` derives
    ``actual_K == width`` (it reads the tensor width, not a request field).
    Structural padding is encoded as ``-inf`` only at this transport boundary;
    legal logprobs are never classified by magnitude. The explicit ``.to(...)``
    calls protect against dtype/device drift between the operator
    (int32/float32 on the runner device) and the sampler tensors.
    """
    lp_tensors = sampler_output.logprobs_tensors
    if lp_tensors is None or lp_tensors.logprobs is None:
        raise RuntimeError(
            "constrained rows require logprobs in the sampler output, but "
            "logprobs_tensors.logprobs is None; check sampling_params.logprobs"
        )
    if lp_tensors.logprobs.shape[1] < width:
        raise RuntimeError(
            f"sampler logprob width {lp_tensors.logprobs.shape[1]} is smaller "
            f"than constrained beam_width {width}; sampling_params.logprobs "
            "must be at least beam_width"
        )
    expected_shape = tuple(top_logprobs.shape)
    if tuple(top_tokens.shape) != expected_shape or tuple(valid_mask.shape) != expected_shape:
        raise RuntimeError("constrained tokens, logprobs, and valid_mask must have equal shapes")
    if valid_mask.dtype != torch.bool or valid_mask.device != top_logprobs.device:
        raise RuntimeError("constrained valid_mask must be bool on the logprobs device")

    rows_idx = torch.tensor(global_rows, dtype=torch.long, device=top_tokens.device)
    wire_logprobs = top_logprobs.masked_fill(~valid_mask, float("-inf"))

    # Shrink the logprob tensors to width W BEFORE the in-place scatter.  A
    # ``resize_`` reinterprets the flat storage, so doing it after a scatter
    # would misalign the rows; done first, the first ``num_rows * W`` storage
    # elements are exactly the row-aligned ``[:, :W]`` slice, which is what
    # ``_beam_collapse_bookkeeping`` needs (it derives ``actual_K`` from the
    # tensor width).
    num_rows = lp_tensors.logprobs.shape[0]
    lp_tensors.logprobs.resize_(num_rows, width)
    if lp_tensors.logprob_token_ids is not None:
        lp_tensors.logprob_token_ids.resize_(num_rows, width)

    st_ids = sampler_output.sampled_token_ids

    # Contiguous row ranges (the common pure-beam-decode case: global_rows
    # covers [0, W)) use slice assignment.  Advanced indexing (``[rows_idx]``)
    # dispatches the IndexPut AI_CPU kernel, which profiling (bw128_1k_t2)
    # showed as the single largest device-side cost — 10.3ms per step, 15.7%
    # of total kernel time, with the matching IndexPutImplGetWorkspaceSize
    # host call reaching 47ms — for what is a whole-tensor overwrite when the
    # rows are contiguous.  Slices stay on the vector-core fast path.
    contiguous = global_rows == list(range(global_rows[0], global_rows[-1] + 1))
    n_rows = len(global_rows)
    if contiguous:
        r0 = global_rows[0]

        if st_ids is not None:
            best_tokens = top_tokens[:, 0].to(st_ids.dtype)
            has_valid = valid_mask[:, 0]
            if st_ids.dim() == 2:
                st_ids[r0 : r0 + n_rows, 0] = torch.where(
                    has_valid, best_tokens, st_ids[r0 : r0 + n_rows, 0]
                )
            else:
                st_ids[r0 : r0 + n_rows] = torch.where(
                    has_valid, best_tokens, st_ids[r0 : r0 + n_rows]
                )

        # ``logprob_token_ids`` is nullable in the sampler-output contract;
        # see the non-contiguous branch below for the full note.
        if lp_tensors.logprob_token_ids is not None:
            lp_tensors.logprob_token_ids[r0 : r0 + n_rows, :width] = top_tokens.to(
                lp_tensors.logprob_token_ids.dtype
            )
        lp_tensors.logprobs[r0 : r0 + n_rows, :width] = wire_logprobs.to(lp_tensors.logprobs.dtype)

        ranks = getattr(lp_tensors, "selected_token_ranks", None)
        if ranks is not None:
            ranks[r0 : r0 + n_rows] = torch.zeros(n_rows, dtype=ranks.dtype, device=ranks.device)
        return

    if st_ids is not None:
        best_tokens = top_tokens[:, 0].to(st_ids.dtype)
        has_valid = valid_mask[:, 0]
        if st_ids.dim() == 2:
            st_ids[rows_idx, 0] = torch.where(has_valid, best_tokens, st_ids[rows_idx, 0])
        else:
            st_ids[rows_idx] = torch.where(has_valid, best_tokens, st_ids[rows_idx])

    # ``logprob_token_ids`` is nullable in the sampler-output contract (the
    # beam-collapse remux in model_runner_common guards it the same way); only
    # the logprobs tensor is mandatory.  When the sampler omits it, the
    # constrained candidates simply have no token-id side-channel.
    if lp_tensors.logprob_token_ids is not None:
        lp_tensors.logprob_token_ids[rows_idx, :width] = top_tokens.to(
            lp_tensors.logprob_token_ids.dtype
        )
    lp_tensors.logprobs[rows_idx, :width] = wire_logprobs.to(lp_tensors.logprobs.dtype)

    ranks = getattr(lp_tensors, "selected_token_ranks", None)
    if ranks is not None:
        ranks[rows_idx] = torch.zeros(len(global_rows), dtype=ranks.dtype, device=ranks.device)


def get_or_create_execution_adapter(runner: Any) -> Any:
    """Return the constrained execution adapter for the runner.

    Prefers a device-loaded adapter created at weight-load time
    (``load_constraint_tables_on_worker`` ran in ``init_device`` / injection).
    Falls back to the legacy lazy creation when preload did not happen (no
    injected tables yet or a mocked runner) — the lazy path keeps the
    fail-fast guarantees: our platform factory raises when unavailable,
    ``manager.require`` raises when tables are missing, and ``load_tables``
    rejects digest swaps.  Never falls back to Trie.
    """
    adapter = getattr(runner, "_gr_constraint_execution", None)
    if adapter is not None:
        return adapter

    # Weight-load-time preload may not have run (injection after init_device,
    # or a runner that was not yet built).  Try the preload helper first; it
    # is idempotent and caches the loaded adapter on the runner.
    from vllm_gr.constraints.worker_inject import load_constraint_tables_on_worker

    if load_constraint_tables_on_worker(runner):
        return runner._gr_constraint_execution

    from vllm_gr.constraints.worker_inject import get_worker_manager

    manager = get_worker_manager()
    digest = manager.current_digest
    if digest is None:
        raise RuntimeError(
            "constraint tables not injected into this worker; the driver must "
            "inject them via EngineCore.collective_rpc before a constrained "
            "request runs"
        )

    from vllm_gr.constraints.startup import create_constraint_execution

    adapter = create_constraint_execution(device=getattr(runner, "device", None))
    adapter.load_tables(manager.require(digest))
    runner._gr_constraint_execution = adapter
    return adapter


def _validate_homogeneous_step(
    request_row_configs: Sequence[Any],
    beam_data: dict[str, Any],
) -> None:
    """Reject a constrained step that shares rows with unconstrained logprobs.

    The whole logprob tensor is resized to the constrained ``W``, which would
    corrupt unconstrained rows; a constrained request next to any
    non-constrained logprobs row is a hard error (mixed-batch routing is P8).
    Rows without logits (``W_logits == 0``, e.g. regular completion decodes)
    keep the existing dummy path and are not part of this guard.
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


def _sampling_context(
    runner: Any,
) -> tuple[dict[str, Any], Sequence[Any]] | None:
    """Return the current Beam metadata and row map, when both are present."""

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
    return beam_data, request_row_configs


def _has_constrained_metadata(
    request_row_configs: Sequence[Any],
    beam_data: dict[str, Any],
) -> bool:
    """Check for canonical-shaped metadata without validating legacy rows."""

    for req_id, *_ in request_row_configs:
        mdata = beam_data.get(req_id) if req_id is not None else None
        if mdata is not None and mdata.get("is_constrained", False):
            return True
    return False


def _canonical_constraint_table_enabled(runner: Any) -> bool:
    from vllm_gr.v1.worker.gr_context import (
        _get_gr_config,
    )

    config = _get_gr_config(runner)
    table_config = config.constraint_table
    return table_config is not None and table_config.enabled


def _constraint_beam_width(mdata: dict[str, Any], req_id: Any) -> int:
    width = mdata.get("beam_width")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise RuntimeError(
            f"constrained request {req_id!r} has invalid beam_width={width!r}; "
            "expected a positive integer"
        )
    return width


def _constraint_candidate_width(
    runner: Any,
    mdata: dict[str, Any],
    req_id: Any,
    beam_width: int,
) -> int:
    """Return the constrained top-k width K (per-row candidates).

    K is independent of the beam width W: candidate geometry is ``(W, K)``.
    Precedence is metadata, then the engine configuration, and only then the
    legacy ``K == W``.

    The configuration lookup is what makes a decoupled K reach the operator.
    ``beam_data`` metadata carries no ``candidate_width`` key today -- no
    production producer writes one -- so falling straight back to the beam
    width here would silently sample W candidates per row for every
    constrained step, and would then trip the ``max_top_k`` capacity gate with
    the beam width instead of the candidate width.  Mirror
    ``resolve_candidate_width`` rather than re-deriving the value: the
    workspace bucket, the admission gate and this operator must agree.
    """

    width = mdata.get("candidate_width")
    if width is not None:
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise RuntimeError(
                f"constrained request {req_id!r} has invalid candidate_width={width!r}; "
                "expected a positive integer"
            )
        return width

    from vllm_gr.config import GRConfig, resolve_candidate_width

    config = GRConfig.from_vllm_config(getattr(runner, "vllm_config", None))
    if config.beam_max_width is None:
        # No Beam configuration at all: this is the legacy path the metadata
        # fallback existed for.
        return beam_width
    return resolve_candidate_width(config)


def _pack_beam_suffix_prefixes(
    suffix: Any,
    width: int,
    step: int,
    req_id: Any,
) -> np.ndarray:
    """Pack the last ``step`` tokens of the first ``width`` parent suffixes.

    Returns an int64 ``[width, step]`` array.  The fast path converts a
    rectangular ``list[list[int]]`` with one ``np.asarray`` + column slice;
    production suffixes are built as ``[list(t) for t in update.beam_tokens]``
    (one equal-length list per parent), so the O(width*step) Python loop it
    replaces — per-parent ``isinstance``/``int()``/``append`` work that cost
    ~2.4ms per decode step at width=128 — never runs on the hot path.  Any
    deviation (non-list rows, ragged or non-integer entries) falls back to
    the original validated per-parent loop with identical error messages.
    """

    block = suffix[:width] if type(suffix) is list and len(suffix) >= width else None
    if block is not None:
        try:
            packed = np.asarray(block, dtype=np.int64)
        except (TypeError, ValueError, OverflowError):
            packed = None  # ragged/foreign entries -> validated loop below
        if packed is not None and packed.ndim == 2 and packed.shape[1] >= step:
            return packed[:, packed.shape[1] - step :]

    if not isinstance(suffix, Sequence) or isinstance(suffix, (str, bytes)) or len(suffix) < width:
        suffix_count = len(suffix) if isinstance(suffix, Sequence) else 0
        raise RuntimeError(
            f"constrained decode request {req_id!r} needs {width} parent "
            f"beam suffixes, got {suffix_count}"
        )

    packed_rows: list[list[int]] = []
    for parent in range(width):
        parent_suffix = suffix[parent]
        if (
            not isinstance(parent_suffix, Sequence)
            or isinstance(parent_suffix, (str, bytes))
            or len(parent_suffix) < step
        ):
            suffix_length = (
                len(parent_suffix)
                if isinstance(parent_suffix, Sequence)
                and not isinstance(parent_suffix, (str, bytes))
                else 0
            )
            raise RuntimeError(
                f"constrained decode step {step} needs the last {step} token(s) "
                f"for parent {parent} of request {req_id!r}, got "
                f"{suffix_length}"
            )
        try:
            packed_rows.append([int(token_id) for token_id in parent_suffix[-step:]])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"constrained request {req_id!r} has invalid token IDs in beam_suffix[{parent}]"
            ) from exc
    return np.asarray(packed_rows, dtype=np.int64)


def build_constraint_backend_prefix_token_ids(
    req_id: Any,
    logits_rows: int,
    mdata: dict[str, Any],
    device: torch.device,
) -> tuple[int, int, torch.Tensor]:
    """Build one request's fixed ``int32 [rows, 3]`` canonical prefix."""

    step = validate_constraint_step(mdata.get("constraint_step"))
    width = _constraint_beam_width(mdata, req_id)

    if step == 0:
        if logits_rows != 1:
            raise RuntimeError(
                f"constrained prefill request {req_id!r} has {logits_rows} "
                "logits rows, expected exactly 1"
            )
        return (
            step,
            width,
            torch.zeros(
                (1, _CONSTRAINT_PREFIX_WIDTH),
                dtype=torch.int32,
                device=device,
            ),
        )

    if logits_rows != width:
        raise RuntimeError(
            f"constrained decode request {req_id!r} has {logits_rows} logits "
            f"rows, expected beam_width={width}"
        )

    suffix = mdata.get("beam_suffix")
    packed = _pack_beam_suffix_prefixes(suffix, width, step, req_id)
    prefix_rows = np.zeros((width, _CONSTRAINT_PREFIX_WIDTH), dtype=np.int32)
    prefix_rows[:, :step] = packed

    # One batched H2D replaces W tiny torch.tensor(device=...) allocations,
    # slice copies, and their associated allocator/synchronization overhead.
    return (
        step,
        width,
        torch.from_numpy(prefix_rows).to(device),
    )


def _validate_direct_backend_output(
    output: Any,
    *,
    req_id: Any,
    rows: int,
    width: int,
    device: torch.device,
    require_legal_candidate: bool = True,
) -> None:
    expected_shape = (rows, width)
    for name, dtype in (
        ("token_ids", torch.int32),
        ("logprobs", torch.float32),
        ("valid_mask", torch.bool),
    ):
        tensor = getattr(output, name, None)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"constraint Backend for request {req_id!r} did not return {name}")
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"constraint Backend {name} shape {tuple(tensor.shape)} must be "
                f"{expected_shape} for request {req_id!r}"
            )
        if tensor.dtype != dtype:
            raise RuntimeError(f"constraint Backend {name} must be {dtype}, got {tensor.dtype}")
        if tensor.device != device:
            raise RuntimeError(
                f"constraint Backend {name}.device {tensor.device} must match "
                f"logits.device {device}"
            )
        if not tensor.is_contiguous():
            raise RuntimeError(f"constraint Backend {name} must be contiguous")

    if require_legal_candidate and not bool(output.valid_mask[:, 0].all().item()):
        raise RuntimeError(
            f"constraint Backend returned no legal candidate for at least one "
            f"logits row of request {req_id!r}; refusing to sample a padding token"
        )


def build_direct_constrained_sampler_output(
    token_ids: torch.Tensor,
    logprobs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Any:
    """Construct a canonical fixed-width ``SamplerOutput`` from Backend data."""

    if (
        token_ids.dim() != 2
        or token_ids.dtype != torch.int32
        or tuple(logprobs.shape) != tuple(token_ids.shape)
        or logprobs.dtype != torch.float32
        or tuple(valid_mask.shape) != tuple(token_ids.shape)
        or valid_mask.dtype != torch.bool
    ):
        raise RuntimeError(
            "canonical constraint token_ids/logprobs/valid_mask have an invalid output ABI"
        )
    if token_ids.device != logprobs.device or valid_mask.device != logprobs.device:
        raise RuntimeError("canonical constraint outputs must share one device")
    if token_ids.shape[1] == 0 or not bool(valid_mask[:, 0].all().item()):
        raise RuntimeError("canonical constraint output has a logits row without a legal candidate")

    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    wire_logprobs = logprobs.masked_fill(~valid_mask, float("-inf"))
    return SamplerOutput(
        sampled_token_ids=token_ids[:, :1].contiguous(),
        logprobs_tensors=LogprobsTensors(
            logprob_token_ids=token_ids,
            logprobs=wire_logprobs,
            selected_token_ranks=torch.zeros(
                token_ids.shape[0],
                dtype=torch.int64,
                device=token_ids.device,
            ),
        ),
    )


def build_empty_constrained_sampler_output(device: torch.device) -> Any:
    """Return the no-token result for an intermediate chunked prefill.

    vLLM 0.30 calls ``_sample`` when the remapped constrained request has
    no logits rows. Passing that zero-row tensor to the ordinary sampler trips
    its compiled logprob shape guard. An empty SamplerOutput expresses the
    actual contract: this chunk advances KV state but generates no token.
    """

    from vllm.v1.outputs import SamplerOutput

    return SamplerOutput(
        # Upstream bookkeeping has one discard-mask entry for the logical
        # request and clears this placeholder into ``[]``. A zero-row tensor
        # makes that clear operation index past the returned list.
        sampled_token_ids=torch.full(
            (1, 1),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        logprobs_tensors=None,
    )


def try_sample_with_constraint_backend(
    runner: Any,
    logits: torch.Tensor | None,
    spec_decode_metadata: Any = None,
) -> Any | None:
    """Return direct constrained/device-Beam output, or ``None`` for Trie."""

    context = _sampling_context(runner)
    if context is None:
        return None
    beam_data, request_row_configs = context
    if not _has_constrained_metadata(request_row_configs, beam_data):
        return None

    backend = getattr(runner, "constraint_backend", None)
    execution_adapter = None
    if backend is None:
        # A prebuilt canonical artifact publishes runner.constraint_backend at
        # model load.  The offline triples path injects a CPU resource after
        # engine startup and creates the platform adapter lazily.  Both are
        # direct raw-logits paths; neither needs the ordinary vLLM sampler.
        if _canonical_constraint_table_enabled(runner):
            raise RuntimeError(
                "canonical constraint_table is enabled but "
                "runner.constraint_backend is None; unconstrained sampling "
                "is forbidden"
            )
        from vllm_gr.config import ConstraintBackend
        from vllm_gr.v1.worker.gr_context import (
            _get_gr_config,
        )

        gr_config = _get_gr_config(runner)
        if gr_config.constraint_backend is not ConstraintBackend.CONSTRAINT_TABLE:
            return None
        execution_adapter = get_or_create_execution_adapter(runner)

    # Speculative decoding is incompatible with the canonical Backend even on
    # an intermediate chunk that happens to carry no logits rows. Reject it
    # before the ordinary sampler can observe the constrained batch.
    if spec_decode_metadata is not None:
        raise RuntimeError("canonical constraint_table does not support speculative decoding")

    rows = collect_constrained_rows(request_row_configs, beam_data)
    if not rows:
        # Intermediate chunked-prefill visits carry metadata but no logits.
        # Do not fall through to the ordinary sampler: vLLM 0.30's compiled
        # logprob gather rejects zero rows. The scheduler needs an explicit
        # no-token result so it can continue the next prefill chunk.
        device = (
            logits.device
            if isinstance(logits, torch.Tensor)
            else torch.device(getattr(runner, "device", "cpu"))
        )
        return build_empty_constrained_sampler_output(device)
    if logits is None:
        raise RuntimeError("canonical constrained sampling requires raw logits")
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
        raise RuntimeError("canonical constrained logits must be a 2-D tensor")

    _validate_homogeneous_step(request_row_configs, beam_data)

    widths = {_constraint_beam_width(mdata, req_id) for req_id, _, _, mdata in rows}
    if len(widths) != 1:
        raise RuntimeError(
            f"constrained step mixes beam_widths {sorted(widths)}; mixed-width "
            "Constraint Table batches are not supported yet (P8)"
        )
    width = next(iter(widths))
    candidate_width = _constraint_candidate_width(runner, rows[0][3], rows[0][0], width)

    if backend is not None:
        info = getattr(backend, "info", None)
        max_top_k = getattr(info, "max_top_k", None)
        if (
            isinstance(max_top_k, bool)
            or not isinstance(max_top_k, int)
            or candidate_width > max_top_k
        ):
            raise RuntimeError(
                f"constrained candidate_width {candidate_width} exceeds Backend "
                f"max_top_k {max_top_k!r}"
            )
        max_rows = getattr(info, "max_rows", None)
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
            raise RuntimeError(f"constraint Backend has invalid max_rows={max_rows!r}")
    else:
        # The injected execution adapter validates table degree/top_k itself;
        # the grouped serving ABI still bounds one request to beam_width rows.
        max_rows = width

    expected_start = 0
    for req_id, row_start, row_count, _ in rows:
        if row_count > max_rows:
            raise RuntimeError(
                f"constrained request {req_id!r} has {row_count} logits rows, "
                f"exceeding Backend max_rows={max_rows}"
            )
        if row_start != expected_start:
            raise RuntimeError(
                "canonical constrained row mapping must cover every logits row "
                f"exactly once; expected row {expected_start}, got {row_start}"
            )
        expected_start += row_count
    if expected_start != logits.shape[0]:
        raise RuntimeError(
            "canonical constrained row mapping must cover every logits row "
            f"exactly once; mapped {expected_start}, expected {logits.shape[0]}"
        )

    # Refresh async output token ids when the batch exposes the hook (V1
    # InputBatchProxy); the V2 InputBatch has no such hook, so this is a no-op.
    _update_async_output_token_ids(runner)

    plans: list[tuple[Any, int, int, int, torch.Tensor, torch.Tensor]] = []
    global_rows: list[int] = []
    for req_id, row_start, row_count, mdata in rows:
        step, request_width, prefixes = build_constraint_backend_prefix_token_ids(
            req_id,
            row_count,
            mdata,
            logits.device,
        )
        if request_width != width:
            raise RuntimeError("mixed constrained beam widths escaped validation")
        request_rows = list(range(row_start, row_start + row_count))
        temperatures = _temperatures_for_rows(
            runner,
            request_rows,
            logits.device,
        ).contiguous()
        plans.append((req_id, row_start, row_count, step, prefixes, temperatures))
        global_rows.extend(request_rows)

    expected_rows = list(range(logits.shape[0]))
    if global_rows != expected_rows:
        raise RuntimeError(
            "canonical constrained row mapping must cover every logits row "
            f"exactly once; mapped={global_rows}, expected={expected_rows}"
        )

    if execution_adapter is not None:
        top_tokens, top_logprobs, valid_mask, adapter_rows = run_constrained_partition(
            runner,
            logits,
            rows,
            candidate_width,
            execution_adapter,
        )
        if adapter_rows != expected_rows:
            raise RuntimeError(
                "injected constraint adapter row order does not cover the "
                f"sampler rows exactly: {adapter_rows} != {expected_rows}"
            )
        return build_direct_constrained_sampler_output(
            top_tokens,
            top_logprobs,
            valid_mask,
        )

    # The common serving case has several requests at the same canonical
    # decode step. Flatten those rows into one Backend invocation so CUDA runs
    # one fused kernel for the whole scheduler batch instead of one launch per
    # request. Preserve the per-request path for a mixed prefill/decode batch;
    # its rows may require different table levels.
    steps = {plan[3] for plan in plans}
    if len(steps) == 1:
        if logits.shape[0] > max_rows:
            raise RuntimeError(
                f"constrained batch has {logits.shape[0]} logits rows, "
                f"exceeding Backend max_rows={max_rows}"
            )
        prefixes = torch.cat([plan[4] for plan in plans], dim=0).contiguous()
        temperatures = torch.cat([plan[5] for plan in plans], dim=0).contiguous()
        output = backend.select(
            logits,
            ConstraintMetadata(
                prefix_token_ids=prefixes,
                temperatures=temperatures,
                current_step=next(iter(steps)),
                top_k=candidate_width,
            ),
        )
        _validate_direct_backend_output(
            output,
            req_id="<canonical batch>",
            rows=logits.shape[0],
            width=candidate_width,
            device=logits.device,
            require_legal_candidate=logits.device.type != "cuda",
        )
        return build_direct_constrained_sampler_output(
            output.token_ids,
            output.logprobs,
            output.valid_mask,
        )

    token_parts: list[torch.Tensor] = []
    logprob_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    for req_id, row_start, row_count, step, prefixes, temperatures in plans:
        output = backend.select(
            logits[row_start : row_start + row_count],
            ConstraintMetadata(
                prefix_token_ids=prefixes,
                temperatures=temperatures,
                current_step=step,
                top_k=candidate_width,
            ),
        )
        _validate_direct_backend_output(
            output,
            req_id=req_id,
            rows=row_count,
            width=candidate_width,
            device=logits.device,
            require_legal_candidate=logits.device.type != "cuda",
        )
        token_parts.append(output.token_ids)
        logprob_parts.append(output.logprobs)
        mask_parts.append(output.valid_mask)

    token_ids = token_parts[0] if len(token_parts) == 1 else torch.cat(token_parts)
    logprobs = logprob_parts[0] if len(logprob_parts) == 1 else torch.cat(logprob_parts)
    valid_mask = mask_parts[0] if len(mask_parts) == 1 else torch.cat(mask_parts)
    return build_direct_constrained_sampler_output(token_ids, logprobs, valid_mask)


def apply_constrained_sample_override(
    runner: Any,
    logits: torch.Tensor,
    sampler_output: Any,
) -> None:
    """Override constrained sampler-output rows with operator results.

    Short-circuits for pure-completion and Trie-beam steps (no constrained
    rows).  Requires the runner to carry ``_current_beam_data`` (from
    ``make_patched_execute_model``) and ``_current_scheduler_output`` (its
    ``_gr_request_row_configs``), both published before sampling runs.
    """
    beam_data = getattr(runner, "_current_beam_data", None)
    if not beam_data:
        return
    scheduler_output = getattr(runner, "_current_scheduler_output", None)
    request_row_configs = (
        getattr(scheduler_output, "_gr_request_row_configs", None)
        if scheduler_output is not None
        else None
    )
    if not request_row_configs:
        return

    rows = collect_constrained_rows(request_row_configs, beam_data)
    if not rows:
        return

    _validate_homogeneous_step(request_row_configs, beam_data)

    widths = {mdata["beam_width"] for _, _, _, mdata in rows}
    if len(widths) != 1:
        raise RuntimeError(
            f"constrained step mixes beam_widths {sorted(widths)}; "
            "mixed-width Constraint Table batches are not supported yet (P8)"
        )
    width = next(iter(widths))
    candidate_width = _constraint_candidate_width(runner, rows[0][3], rows[0][0], width)

    adapter = get_or_create_execution_adapter(runner)
    top_tokens, top_logprobs, valid_mask, global_rows = run_constrained_partition(
        runner, logits, rows, candidate_width, adapter
    )
    patch_sampler_output_for_constrained(
        sampler_output, global_rows, top_tokens, top_logprobs, valid_mask, candidate_width
    )


def collect_constrained_override_rows(runner: Any) -> list[tuple[Any, int, int, dict[str, Any]]]:
    """Return the constrained rows for the runner's current step, if any.

    Thin pre-check for the sampler short-circuit path: same collection walk
    as ``apply_constrained_sample_override`` (rows + homogeneous validation)
    without running the operator, so ``patched_sample`` can decide whether
    the original sampler is needed before paying for it.
    """
    beam_data = getattr(runner, "_current_beam_data", None)
    if not beam_data:
        return []
    scheduler_output = getattr(runner, "_current_scheduler_output", None)
    request_row_configs = (
        getattr(scheduler_output, "_gr_request_row_configs", None)
        if scheduler_output is not None
        else None
    )
    if not request_row_configs:
        return []

    rows = collect_constrained_rows(request_row_configs, beam_data)
    if not rows:
        return []

    _validate_homogeneous_step(request_row_configs, beam_data)
    widths = {mdata["beam_width"] for _, _, _, mdata in rows}
    if len(widths) != 1:
        raise RuntimeError(
            f"constrained step mixes beam_widths {sorted(widths)}; "
            "mixed-width Constraint Table batches are not supported yet (P8)"
        )
    return rows


def build_constrained_sampler_output(
    runner: Any,
    logits: torch.Tensor,
    rows: Sequence[tuple[Any, int, int, dict[str, Any]]],
) -> Any:
    """Run the constrained operator and build a SamplerOutput directly.

    Asynchronous scheduling sampler short-circuit (``model_runner_v1`` decode path):
    when every sampling row is constrained, the sampler's log_softmax +
    topk + gather_logprobs is dead work — the operator already produces the
    fixed-width ``K`` candidates the beam decision consumes.  This builds
    the final sampler output from the operator result in one pass.

    Shape contract (matches ``patch_sampler_output_for_constrained``):
    ``logprob_token_ids``/``logprobs`` are ``[rows, K]`` with the top-1
    candidate in column 0; ``sampled_token_ids`` is ``[rows, 1]`` int32;
    ranks are all 1 (the top-1 candidate is rank 1 by definition).
    """
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    width = next(iter({mdata["beam_width"] for _, _, _, mdata in rows}))
    candidate_width = _constraint_candidate_width(runner, rows[0][3], rows[0][0], width)
    adapter = get_or_create_execution_adapter(runner)
    top_tokens, top_logprobs, valid_mask, _ = run_constrained_partition(
        runner, logits, rows, candidate_width, adapter
    )

    num_rows = top_tokens.shape[0]
    token_ranks = torch.ones(num_rows, dtype=torch.int32, device=top_tokens.device)
    return SamplerOutput(
        sampled_token_ids=top_tokens[:, :1].to(torch.int32),
        logprobs_tensors=LogprobsTensors(
            top_tokens.to(torch.int32),
            top_logprobs.masked_fill(~valid_mask, float("-inf")).to(torch.float32),
            token_ranks,
        ),
    )
