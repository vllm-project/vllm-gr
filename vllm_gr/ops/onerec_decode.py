# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""OneRec decode post-processing operators.

The public functions mirror the xllm-ops NPU signatures.  The ordinary
PyTorch implementations provide the GPU fallback while preserving the NPU
operator signatures.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _require_int32(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must use torch.int32")


def _require_fp32(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32")


def _require_same_device(reference: torch.Tensor, **tensors: torch.Tensor) -> None:
    for name, tensor in tensors.items():
        if tensor.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}, got {tensor.device}")


def _validate_select_unshared_kv(
    beam_index: torch.Tensor,
    x_key_block: Sequence[torch.Tensor],
    x_value_block: Sequence[torch.Tensor],
    block_table: torch.Tensor,
    group_token_num: torch.Tensor,
    decode_step: int,
    beam_size: int,
    layer_num: int,
) -> tuple[int, int, int, int]:
    _require_int32("beam_index", beam_index)
    _require_int32("block_table", block_table)
    _require_int32("group_token_num", group_token_num)
    if beam_index.ndim != 1 or group_token_num.ndim != 1:
        raise ValueError("beam_index and group_token_num must be one-dimensional")
    if block_table.ndim != 1:
        raise ValueError("block_table must be one-dimensional")
    if beam_size <= 0 or layer_num <= 0:
        raise ValueError("beam_size and layer_num must be positive")
    if len(x_key_block) != layer_num or len(x_value_block) != layer_num:
        raise ValueError("the K/V tensor-list lengths must equal layer_num")

    batch_size = block_table.numel()
    if beam_index.numel() != batch_size * beam_size:
        raise ValueError("beam_index must contain batch_size * beam_size values")
    if group_token_num.numel() != batch_size * beam_size:
        raise ValueError("group_token_num must contain batch_size * beam_size values")

    first_key = x_key_block[0]
    if first_key.ndim != 5:
        raise ValueError(
            "K/V caches must have shape "
            "[num_blocks, beam_size, num_heads, max_decode_steps, head_dim]"
        )
    num_blocks, cache_beam, num_heads, max_decode_steps, head_dim = first_key.shape
    if cache_beam != beam_size:
        raise ValueError("cache beam dimension must equal beam_size")
    if decode_step < 1 or decode_step > max_decode_steps:
        raise ValueError("decode_step must be in [1, max_decode_steps]")
    if first_key.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("K/V caches must use torch.float16 or torch.bfloat16")
    _require_same_device(
        first_key,
        beam_index=beam_index,
        block_table=block_table,
        group_token_num=group_token_num,
    )

    expected_shape = tuple(first_key.shape)
    for layer, (key, value) in enumerate(zip(x_key_block, x_value_block)):
        if tuple(key.shape) != expected_shape or tuple(value.shape) != expected_shape:
            raise ValueError(f"layer {layer} K/V cache shape mismatch")
        if key.dtype != first_key.dtype or value.dtype != first_key.dtype:
            raise TypeError(f"layer {layer} K/V cache dtype mismatch")
        if key.device != first_key.device or value.device != first_key.device:
            raise ValueError(f"layer {layer} K/V cache device mismatch")

    return batch_size, num_heads, max_decode_steps, head_dim


def _parent_beams_from_prefix(
    group_token_num: torch.Tensor, batch_size: int, beam_size: int
) -> torch.Tensor:
    prefix = group_token_num.reshape(batch_size, beam_size)
    request_offset = (
        torch.arange(batch_size, device=prefix.device, dtype=prefix.dtype)[:, None] * beam_size
    )
    # beam_search_group emits global prefix values, while the standalone NPU
    # select test uses request-local values.  Accept both representations.
    prefix = torch.where(prefix[:, -1:] > beam_size, prefix - request_offset, prefix)
    counts = torch.cat((prefix[:, :1], prefix[:, 1:] - prefix[:, :-1]), dim=1)
    parents = torch.arange(beam_size, dtype=torch.int64, device=group_token_num.device).expand(
        batch_size, -1
    )
    return torch.repeat_interleave(
        parents.reshape(-1),
        counts.to(torch.int64).reshape(-1),
        output_size=batch_size * beam_size,
    ).reshape(batch_size, beam_size)


def _select_unshared_kv_torch(
    beam_index: torch.Tensor,
    x_key_block: list[torch.Tensor],
    x_value_block: list[torch.Tensor],
    block_table: torch.Tensor,
    group_token_num: torch.Tensor,
    decode_step: int,
    beam_size: int,
    layer_num: int,
) -> None:
    """NPU-semantic in-place KV selection using ordinary PyTorch ops.

    xllm-ops derives the mapping from ``group_token_num`` and copies the full
    cache time axis.  ``beam_index`` and ``decode_step`` are retained because
    they are part of the public NPU ABI.
    """

    batch_size, _, _, _ = _validate_select_unshared_kv(
        beam_index,
        x_key_block,
        x_value_block,
        block_table,
        group_token_num,
        decode_step,
        beam_size,
        layer_num,
    )
    if batch_size == 0:
        return
    blocks = block_table.to(torch.int64)
    parents = _parent_beams_from_prefix(group_token_num, batch_size, beam_size)
    for key, value in zip(x_key_block, x_value_block):
        source_key = key[blocks].clone()
        source_value = value[blocks].clone()
        gather_index = parents[:, :, None, None, None].expand_as(source_key)
        key[blocks] = torch.gather(source_key, 1, gather_index)
        value[blocks] = torch.gather(source_value, 1, gather_index)


def select_unshared_kv(
    beam_index: torch.Tensor,
    x_key_block: list[torch.Tensor],
    x_value_block: list[torch.Tensor],
    block_table: torch.Tensor,
    group_token_num: torch.Tensor,
    decode_step: int,
    beam_size: int,
    layer_num: int,
) -> None:
    """Update K/V tensor lists in place; signature matches xllm-ops."""

    return _select_unshared_kv_torch(
        beam_index,
        x_key_block,
        x_value_block,
        block_table,
        group_token_num,
        decode_step,
        beam_size,
        layer_num,
    )


def _validate_final_select(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
) -> tuple[int, int, int, int]:
    _require_fp32("log_probs", log_probs)
    _require_fp32("top_probs", top_probs)
    _require_int32("top_tokens", top_tokens)
    _require_int32("sequence", sequence)
    if log_probs.ndim != 2 or log_probs.shape[1] != 1:
        raise ValueError("log_probs must have shape [batch * beam_width, 1]")
    if top_tokens.ndim != 2 or top_probs.shape != top_tokens.shape:
        raise ValueError("top_tokens and top_probs must have the same 2-D shape")
    if sequence.ndim != 3:
        raise ValueError("sequence must have shape [batch, beam_width, rounds]")
    batch_size, beam_width, total_rounds = sequence.shape
    candidate_width = top_tokens.shape[1]
    if top_tokens.shape[0] != batch_size * beam_width:
        raise ValueError("candidate row count must equal batch_size * beam_width")
    if log_probs.shape[0] != batch_size * beam_width:
        raise ValueError("log_probs row count must equal batch_size * beam_width")
    if current_step < 0 or current_step >= total_rounds:
        raise ValueError("current_step is outside the sequence time dimension")
    if result_width <= 0 or result_width > beam_width * candidate_width:
        raise ValueError("result_width must be in [1, beam_width * candidate_width]")
    _require_same_device(
        log_probs,
        top_tokens=top_tokens,
        top_probs=top_probs,
        sequence=sequence,
    )
    return batch_size, beam_width, candidate_width, total_rounds


def _stable_topk_indices(scores: torch.Tensor, width: int) -> torch.Tensor:
    # Stable descending sort gives the frozen (score, lower flat_index) order.
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :width]


def _beam_search_rec_final_select_torch(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, beam_width, candidate_width, total_rounds = _validate_final_select(
        log_probs,
        top_tokens,
        top_probs,
        sequence,
        current_step,
        result_width,
    )
    scores = (
        log_probs.reshape(batch_size, beam_width, 1)
        + top_probs.reshape(batch_size, beam_width, candidate_width)
    ).reshape(batch_size, -1)
    flat_indices = _stable_topk_indices(scores, result_width)
    out_log_probs = torch.gather(scores, 1, flat_indices)
    parent = torch.div(flat_indices, candidate_width, rounding_mode="floor")
    candidate = torch.remainder(flat_indices, candidate_width)
    request = torch.arange(batch_size, device=sequence.device)[:, None]
    candidates = top_tokens.reshape(batch_size, beam_width, candidate_width)
    out_token_ids = candidates[request, parent, candidate]
    out_token_index = (parent + request * beam_width).to(torch.int32)
    out_sequence = torch.zeros(
        (batch_size, result_width, total_rounds),
        dtype=torch.int32,
        device=sequence.device,
    )
    if current_step:
        out_sequence[:, :, :current_step] = sequence[
            request.expand_as(parent), parent, :current_step
        ]
    out_sequence[:, :, current_step] = out_token_ids
    return (
        out_token_ids.reshape(-1, 1),
        out_token_index.reshape(-1, 1),
        out_log_probs.reshape(-1, 1),
        out_sequence,
    )


def beam_search_rec_final_select(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if log_probs.device.type == "cuda":
        return _beam_search_rec_final_select_cuda_dispatch(
            log_probs,
            top_tokens,
            top_probs,
            sequence,
            current_step,
            result_width,
        )
    return _beam_search_rec_final_select_torch(
        log_probs, top_tokens, top_probs, sequence, current_step, result_width
    )


def _validate_group(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    top_k: int,
) -> tuple[int, int, int, int, int]:
    _require_fp32("log_probs", log_probs)
    _require_fp32("top_probs", top_probs)
    _require_int32("top_tokens", top_tokens)
    _require_int32("sequence", sequence)
    if log_probs.ndim != 2 or log_probs.shape[1] != 1:
        raise ValueError("log_probs must have shape [batch * beam_width, 1]")
    if top_tokens.ndim != 2 or top_probs.shape != top_tokens.shape:
        raise ValueError("top_tokens and top_probs must have the same 2-D shape")
    if sequence.ndim != 3:
        raise ValueError("sequence must have shape [batch, beam_width, rounds]")
    batch_size, beam_width, total_rounds = sequence.shape
    # Candidate width K is independent of the beam width W: (rows, columns)
    # is (batch * W, K).
    candidate_width = top_tokens.shape[1]
    if top_tokens.shape != (batch_size * beam_width, candidate_width):
        raise ValueError("group candidates must have shape [batch * beam_width, candidate_width]")
    if candidate_width < 1:
        raise ValueError("group candidates must have at least one column")
    if log_probs.shape[0] != batch_size * beam_width:
        raise ValueError("log_probs row count must equal batch_size * beam_width")
    if total_rounds < current_step + 1 or current_step < 0:
        raise ValueError("sequence must contain current_step + 1 positions")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    _require_same_device(
        log_probs,
        top_tokens=top_tokens,
        top_probs=top_probs,
        sequence=sequence,
    )
    return (
        batch_size,
        beam_width,
        candidate_width,
        min(top_k, beam_width * candidate_width),
        total_rounds,
    )


def _beam_search_group_torch(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, beam_width, candidate_width, effective_k, total_rounds = _validate_group(
        log_probs, top_tokens, top_probs, sequence, current_step, top_k
    )
    device = sequence.device
    out_token_ids = torch.zeros((batch_size, top_k), dtype=torch.int32, device=device)
    out_token_index = torch.zeros_like(out_token_ids)
    out_log_probs = torch.zeros((batch_size, top_k), dtype=torch.float32, device=device)
    out_prefix = torch.zeros((batch_size, beam_width), dtype=torch.int32, device=device)
    out_sequence = torch.zeros(
        (batch_size, top_k, total_rounds),
        dtype=torch.int32,
        device=device,
    )

    if current_step == 0:
        # Matches the NPU kernel's legacy initialization contract (verified on
        # device): only the flat sequence rows [0, batch_size) receive their
        # own row's first candidate token; every other output stays zero.
        # The Asynchronous scheduling integration bypasses this path through init_from_prefill.
        flat_out_sequence = out_sequence.reshape(-1, total_rounds)
        flat_out_sequence[:batch_size, 0] = top_tokens[:batch_size, 0].to(torch.int32)
        return (
            out_token_ids,
            out_token_index,
            out_log_probs,
            out_prefix,
            out_sequence,
        )

    scores = (
        log_probs.reshape(batch_size, beam_width, 1)
        + top_probs.reshape(batch_size, beam_width, candidate_width)
    ).reshape(batch_size, -1)
    flat = _stable_topk_indices(scores, effective_k)
    selected_scores = torch.gather(scores, 1, flat)
    parent = torch.div(flat, candidate_width, rounding_mode="floor")
    candidate = torch.remainder(flat, candidate_width)
    request = torch.arange(batch_size, device=device)[:, None]
    tokens = top_tokens.reshape(batch_size, beam_width, candidate_width)[request, parent, candidate]

    # Stable sort by parent retains score/flat-index TopK order per parent.
    group_order = torch.argsort(parent, dim=1, stable=True)
    grouped_parent = torch.gather(parent, 1, group_order)
    grouped_tokens = torch.gather(tokens, 1, group_order)
    grouped_scores = torch.gather(selected_scores, 1, group_order)
    global_parent = grouped_parent + request * beam_width
    parent_index = parent.to(torch.int64)
    counts = torch.zeros(
        (batch_size, beam_width),
        dtype=torch.int64,
        device=device,
    )
    counts.scatter_add_(1, parent_index, torch.ones_like(parent_index))
    prefix = torch.cumsum(counts, dim=1) + request * beam_width

    out_token_ids[:, :effective_k] = grouped_tokens
    out_token_index[:, :effective_k] = global_parent.to(torch.int32)
    out_log_probs[:, :effective_k] = grouped_scores
    out_prefix.copy_(prefix.to(torch.int32))

    source = sequence[:, :, :current_step].reshape(batch_size * beam_width, current_step)
    selected_history = source.index_select(0, global_parent.reshape(-1)).reshape(
        batch_size, effective_k, current_step
    )
    out_sequence[:, :effective_k, :current_step] = selected_history
    out_sequence[:, :effective_k, current_step] = grouped_tokens
    return (
        out_token_ids,
        out_token_index,
        out_log_probs,
        out_prefix,
        out_sequence,
    )


def beam_search_group(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select ``top_k`` survivors per request from ``[B*W, K]`` candidates.

    Three widths are in play and must not be conflated:

    - ``W`` (beam width) is ``sequence.shape[1]``, i.e. the number of rows per
      request;
    - ``K`` (candidate width) is the column extent of ``top_tokens`` /
      ``top_probs`` and is independent of ``W``;
    - ``top_k`` is the *survivor* count -- how many winners are kept per
      request.

    ``K != W`` is fully supported by both backends: each derives ``K`` from the
    tensors.  ``top_k != W`` is **not**: the CUDA backend fuses the survivor
    count into its ``W``-wide parent grouping (``out_parent_count_prefix`` is
    one entry per beam row), so it can only emit exactly ``W`` survivors.  That
    limitation is raised explicitly rather than silently ignored; narrowing the
    width between stages is expressed through ``dynamic_beam_widths`` and
    ``beam_state.select``, not through this operator.

    The torch reference does honour ``top_k < W``, but its outputs then mix
    widths (tokens/parents/scores/sequence are ``top_k`` wide while the
    per-parent prefix stays ``W`` wide, matching the NPU contract).
    """
    if log_probs.device.type == "cuda":
        return _beam_search_group_cuda_dispatch(
            log_probs,
            top_tokens,
            top_probs,
            sequence,
            current_step,
            top_k,
        )
    return _beam_search_group_torch(log_probs, top_tokens, top_probs, sequence, current_step, top_k)


def _beam_search_rec_final_select_cuda_dispatch(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm_gr.ops.cuda.beam_selection import beam_search_rec_final_select_cuda

    return beam_search_rec_final_select_cuda(
        log_probs,
        top_tokens,
        top_probs,
        sequence,
        current_step,
        result_width,
    )


class UnsupportedSurvivorCountError(NotImplementedError, ValueError):
    """``top_k != beam_width`` on the fixed-width CUDA group operator.

    It is a ``NotImplementedError`` because a narrower selection is a missing
    capability, not a bad value.  It is also a ``ValueError`` because that is
    what this case raised before the CUDA dispatch checked it itself, so a
    caller written against the old exception still catches it.
    """


def _beam_search_group_cuda_dispatch(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm_gr.ops.cuda.beam_selection import (
        _validate_beam_inputs,
        beam_search_group_cuda,
    )

    # The CUDA group operator is fixed-width: it always emits exactly one
    # survivor per beam row (``W``), and its ``candidate_width`` argument
    # selects the column extent ``K``, which it reads from the tensors.  So
    # ``K`` needs no plumbing here, but ``top_k`` -- the survivor count, the
    # torch convention -- cannot be honoured.  Raise instead of dropping it:
    # a silent ``del`` would make a narrower selection look supported while
    # returning ``W`` winners.
    #
    # Validate in the order the callee did before K was read from the tensors:
    # tensors and current_step first (the same validator, so malformed inputs
    # get its ValueError/TypeError rather than an IndexError from ``shape[1]``),
    # then the type of top_k (``8.0 != 8`` is False, so the width comparison
    # alone would accept a float), then the width itself.
    _, beam_width, candidate_width, _ = _validate_beam_inputs(
        log_probs, top_tokens, top_probs, sequence, current_step, None
    )
    if current_step not in (0, 1, 2):
        raise ValueError("CUDA BeamSearchGroup supports current_step 0, 1, or 2")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    # A non-positive survivor count is invalid rather than merely unsupported.
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k != beam_width:
        raise UnsupportedSurvivorCountError(
            "CUDA beam_search_group always emits beam_width survivors; "
            f"got top_k={top_k}, beam_width={beam_width}. Narrow the width "
            "through dynamic_beam_widths/beam_state.select instead."
        )
    return beam_search_group_cuda(
        log_probs,
        top_tokens,
        top_probs,
        sequence,
        current_step,
        candidate_width,
    )


def _beam_search_rec_final_select_meta(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del top_probs, current_step
    batch_size = sequence.shape[0]
    total_rounds = sequence.shape[2]
    return (
        top_tokens.new_empty((batch_size * result_width, 1), dtype=torch.int32),
        top_tokens.new_empty((batch_size * result_width, 1), dtype=torch.int32),
        log_probs.new_empty((batch_size * result_width, 1), dtype=torch.float32),
        sequence.new_empty((batch_size, result_width, total_rounds), dtype=torch.int32),
    )


def _beam_search_group_meta(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del top_probs, current_step
    batch_size, beam_width, total_rounds = sequence.shape
    return (
        top_tokens.new_empty((batch_size, top_k), dtype=torch.int32),
        top_tokens.new_empty((batch_size, top_k), dtype=torch.int32),
        log_probs.new_empty((batch_size, top_k), dtype=torch.float32),
        top_tokens.new_empty((batch_size, beam_width), dtype=torch.int32),
        sequence.new_empty((batch_size, top_k, total_rounds), dtype=torch.int32),
    )


# Keep explicit schemas alongside the Python API so aliasing and list mutation
# remain visible to torch.compile.  The library object must stay alive for the
# lifetime of the registrations.
_OP_LIBRARY = torch.library.Library("vllm_gr", "FRAGMENT")
_OP_LIBRARY.define(
    "select_unshared_kv(Tensor beam_index, Tensor(a!)[] x_key_block, "
    "Tensor(b!)[] x_value_block, Tensor block_table, Tensor group_token_num, "
    "int decode_step, int beam_size, int layer_num) -> ()"
)
_OP_LIBRARY.define(
    "beam_search_rec_final_select(Tensor log_probs, Tensor top_tokens, "
    "Tensor top_probs, Tensor sequence, int current_step, int result_width) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)
_OP_LIBRARY.define(
    "beam_search_group(Tensor log_probs, Tensor top_tokens, Tensor top_probs, "
    "Tensor sequence, int current_step, int top_k) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)
_OP_LIBRARY.impl("select_unshared_kv", _select_unshared_kv_torch, "CompositeExplicitAutograd")
_OP_LIBRARY.impl(
    "beam_search_rec_final_select",
    _beam_search_rec_final_select_torch,
    "CompositeExplicitAutograd",
)
_OP_LIBRARY.impl("beam_search_group", _beam_search_group_torch, "CompositeExplicitAutograd")
_OP_LIBRARY.impl(
    "beam_search_rec_final_select",
    _beam_search_rec_final_select_cuda_dispatch,
    "CUDA",
)
_OP_LIBRARY.impl("beam_search_group", _beam_search_group_cuda_dispatch, "CUDA")
_OP_LIBRARY.impl(
    "beam_search_rec_final_select",
    _beam_search_rec_final_select_meta,
    "Meta",
)
_OP_LIBRARY.impl("beam_search_group", _beam_search_group_meta, "Meta")


__all__ = [
    "_beam_search_group_torch",
    "_beam_search_rec_final_select_torch",
    "_select_unshared_kv_torch",
    "beam_search_group",
    "beam_search_rec_final_select",
    "select_unshared_kv",
]
