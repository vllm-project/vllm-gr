# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA precision guards for deterministic OneRec BeamSearchGroup."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vllm_gr.ops.cuda.beam_selection import (
    allocate_beam_selection_scratch,
    beam_search_group_cuda,
    beam_search_group_cuda_out,
    beam_selection_scratch_slots,
)
from vllm_gr.ops.onerec_decode import (
    UnsupportedSurvivorCountError,
    _beam_search_group_torch,
    beam_search_group,
)
from vllm_gr.v1.attention.backends.beam_kernels import (
    select_unshared_kv,
    select_unshared_kv_fused,
)

# pytest parametrization decorators are untyped under strict mypy.
# mypy: disable-error-code="misc"


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _inputs(
    batch_size: int,
    beam_width: int,
    current_step: int,
    *,
    max_steps: int = 3,
    candidate_width: int | None = None,
) -> tuple[torch.Tensor, ...]:
    # Rows are Beam rows; columns are the candidate width K, which is
    # independent of the Beam width W.  Default keeps the legacy K == W square.
    candidates = beam_width if candidate_width is None else candidate_width
    generator = torch.Generator(device="cuda").manual_seed(20260827)
    cumulative = torch.randn(
        (batch_size * beam_width, 1),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    top_tokens = torch.arange(
        batch_size * beam_width * candidates,
        dtype=torch.int32,
        device="cuda",
    ).reshape(batch_size * beam_width, candidates)
    top_logprobs = torch.randn(
        (batch_size * beam_width, candidates),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    sequence = torch.arange(
        batch_size * beam_width * max_steps,
        dtype=torch.int32,
        device="cuda",
    ).reshape(batch_size, beam_width, max_steps)
    assert current_step < max_steps
    return cumulative, top_tokens, top_logprobs, sequence


@pytest.mark.parametrize("beam_width", [1, 2, 4, 8, 32, 64, 128, 256])
@pytest.mark.parametrize("current_step", [1, 2])
def test_group_cuda_matches_fixed_extent_torch(
    beam_width: int,
    current_step: int,
) -> None:
    inputs = _inputs(1, beam_width, current_step)
    expected = _beam_search_group_torch(*inputs, current_step, beam_width)
    actual = beam_search_group_cuda(*inputs, current_step, beam_width)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert actual[-1].shape == (1, beam_width, 3)


# Candidate width K decoupled from the Beam width W: the hierarchical tile
# geometry must stay exact for narrower, wider and rectangular candidate sets.
@pytest.mark.parametrize(
    ("beam_width", "candidate_width"),
    [
        (8, 1),
        (8, 64),
        (64, 8),
        (64, 256),
        (128, 64),
        (256, 64),
        (256, 256),
        (512, 256),
        (1024, 256),
    ],
)
# Step 0 is excluded on purpose: the torch reference implements the NPU legacy
# init contract there (all-zero outputs plus the first sequence entry), while
# the CUDA bootstrap selects the best valid candidates from the first parent.
# The initialization-contract divergence predates K/W decoupling; test the
# CUDA contract independently below.
@pytest.mark.parametrize("current_step", [1, 2])
def test_group_cuda_matches_torch_across_candidate_widths(
    beam_width: int,
    candidate_width: int,
    current_step: int,
) -> None:
    inputs = _inputs(1, beam_width, current_step, candidate_width=candidate_width)
    expected = _beam_search_group_torch(*inputs, current_step, beam_width)
    # The torch reference takes the survivor count (W) as ``top_k``; the CUDA
    # entry point always emits exactly W survivors and takes the per-row
    # candidate width K instead.
    actual = beam_search_group_cuda(*inputs, current_step, candidate_width)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert actual[-1].shape == (1, beam_width, 3)


def test_group_cuda_multi_request_global_parent_and_prefix() -> None:
    batch_size, beam_width, current_step = 2, 8, 1
    inputs = _inputs(batch_size, beam_width, current_step)
    actual = beam_search_group_cuda(*inputs, current_step, beam_width)
    expected = _beam_search_group_torch(*inputs, current_step, beam_width)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert torch.all(actual[1][1] >= beam_width)
    assert actual[3][0, -1] == beam_width
    assert actual[3][1, -1] == 2 * beam_width


def test_group_cuda_step_zero_bootstraps_first_parent() -> None:
    batch_size, beam_width = 2, 8
    cumulative, top_tokens, top_logprobs, sequence = _inputs(batch_size, beam_width, 0)
    token, parent, score, prefix, next_sequence = beam_search_group_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        0,
        beam_width,
    )
    request = torch.arange(batch_size, dtype=torch.int32, device="cuda")[:, None]
    candidate_tokens = top_tokens.reshape(batch_size, beam_width, beam_width)[:, 0]
    candidate_scores = (
        cumulative.reshape(batch_size, beam_width, 1)[:, 0]
        + top_logprobs.reshape(batch_size, beam_width, beam_width)[:, 0]
    )
    order = torch.argsort(candidate_scores, dim=1, descending=True, stable=True)
    candidate_tokens = candidate_tokens.gather(1, order)
    candidate_scores = candidate_scores.gather(1, order)
    assert torch.equal(token, candidate_tokens)
    assert torch.equal(parent, (request * beam_width).expand_as(parent))
    torch.testing.assert_close(score, candidate_scores, rtol=0, atol=0)
    assert torch.equal(prefix, (request * beam_width + beam_width).expand_as(prefix))
    assert torch.equal(next_sequence[:, :, 0], token)
    assert torch.count_nonzero(next_sequence[:, :, 1:]) == 0


def test_group_cuda_step_zero_accepts_compact_rectopk_rows() -> None:
    batch_size, beam_width = 2, 8
    cumulative, top_tokens, top_logprobs, sequence = _inputs(batch_size, beam_width, 0)
    expected = beam_search_group_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        0,
        beam_width,
    )
    compact_cumulative = cumulative.reshape(batch_size, beam_width, 1)[:, 0].contiguous()
    compact_tokens = top_tokens.reshape(batch_size, beam_width, beam_width)[:, 0].contiguous()
    compact_logprobs = top_logprobs.reshape(batch_size, beam_width, beam_width)[:, 0].contiguous()
    compact_valid = torch.ones_like(compact_tokens, dtype=torch.bool)

    actual = beam_search_group_cuda(
        compact_cumulative,
        compact_tokens,
        compact_logprobs,
        sequence,
        0,
        beam_width,
        compact_valid,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_group_cuda_exact_tie_uses_flat_index_then_stable_parent_grouping() -> None:
    batch_size, beam_width, current_step = 1, 8, 1
    cumulative, top_tokens, top_logprobs, sequence = _inputs(batch_size, beam_width, current_step)
    cumulative.zero_()
    top_logprobs.zero_()
    token, parent, score, prefix, _ = beam_search_group_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        beam_width,
    )
    assert torch.equal(token[0], torch.arange(beam_width, dtype=torch.int32, device="cuda"))
    assert torch.equal(parent[0], torch.zeros(beam_width, dtype=torch.int32, device="cuda"))
    assert torch.equal(score, torch.zeros_like(score))
    assert torch.equal(
        prefix[0], torch.full((beam_width,), beam_width, dtype=torch.int32, device="cuda")
    )


def test_group_cuda_explicit_candidate_validity() -> None:
    batch_size, beam_width, current_step = 1, 8, 1
    cumulative, top_tokens, top_logprobs, sequence = _inputs(batch_size, beam_width, current_step)
    valid_mask = torch.zeros_like(top_tokens, dtype=torch.bool)
    valid_mask[0, :3] = True
    masked_logprobs = top_logprobs.masked_fill(~valid_mask, -float("inf"))
    expected = _beam_search_group_torch(
        cumulative,
        top_tokens,
        masked_logprobs,
        sequence,
        current_step,
        beam_width,
    )
    actual = beam_search_group_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        current_step,
        beam_width,
        valid_mask,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert torch.equal(actual[1], torch.zeros_like(actual[1]))
    assert torch.isneginf(actual[2][:, 3:]).all()
    assert actual[3][0, -1] == beam_width


def test_group_cuda_out_variant_graph_replay_uses_fixed_addresses() -> None:
    batch_size, beam_width, current_step = 2, 8, 2
    inputs = _inputs(batch_size, beam_width, current_step)
    scratch = allocate_beam_selection_scratch(
        batch_size,
        beam_width,
        beam_width,
        beam_width,
        torch.device("cuda"),
    )
    outputs = (
        torch.empty((batch_size, beam_width), dtype=torch.int32, device="cuda"),
        torch.empty((batch_size, beam_width), dtype=torch.int32, device="cuda"),
        torch.empty((batch_size, beam_width), dtype=torch.float32, device="cuda"),
        torch.empty((batch_size, beam_width), dtype=torch.int32, device="cuda"),
        torch.empty((batch_size, beam_width, 3), dtype=torch.int32, device="cuda"),
    )
    scratch_tensors = (
        scratch.local_scores,
        scratch.local_flat_indices,
        scratch.merge_scores,
        scratch.merge_flat_indices,
    )
    addresses = tuple(tensor.data_ptr() for tensor in (*outputs, *scratch_tensors))

    def run() -> None:
        beam_search_group_cuda_out(
            *inputs,
            current_step,
            beam_width,
            *outputs,
            scratch.local_scores,
            scratch.local_flat_indices,
            scratch.merge_scores,
            scratch.merge_flat_indices,
        )

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    inputs[2].copy_(inputs[2].flip(1))
    expected = _beam_search_group_torch(*inputs, current_step, beam_width)
    graph.replay()
    torch.accelerator.synchronize()
    assert tuple(tensor.data_ptr() for tensor in (*outputs, *scratch_tensors)) == addresses
    for actual_tensor, expected_tensor in zip(outputs, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_public_group_dispatches_to_cuda() -> None:
    inputs = _inputs(1, 4, 1)
    expected = beam_search_group_cuda(*inputs, 1, 4)
    results = (
        beam_search_group(*inputs, 1, 4),
        torch.ops.vllm_gr.beam_search_group(*inputs, 1, 4),
    )
    for actual in results:
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_group_parent_indices_drive_existing_select_unshared_kv_fused() -> None:
    beam_width, max_steps, head_dim = 4, 3, 8
    cumulative, top_tokens, top_logprobs, sequence = _inputs(1, beam_width, 1)
    cumulative.zero_()
    top_logprobs.fill_(-100.0)
    top_logprobs[2, :2] = torch.tensor([4.0, 3.0], device="cuda")
    top_logprobs[0, 0] = 2.0
    top_logprobs[3, 0] = 1.0
    _, parent, _, _, _ = beam_search_group_cuda(
        cumulative,
        top_tokens,
        top_logprobs,
        sequence,
        1,
        beam_width,
    )
    assert torch.equal(
        parent[0],
        torch.tensor([0, 2, 2, 3], dtype=torch.int32, device="cuda"),
    )

    shape = (1, beam_width, 1, max_steps, head_dim)
    key = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(shape).half().cuda()
    value = key + 1000
    expected_key = key.clone()
    expected_value = value.clone()
    reference_aux_key = torch.empty(
        (beam_width, 1, max_steps, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    reference_aux_value = torch.empty_like(reference_aux_key)
    select_unshared_kv(
        parent.reshape(-1),
        0,
        [expected_key],
        [expected_value],
        1,
        aux_key_bufs=[reference_aux_key],
        aux_value_bufs=[reference_aux_value],
    )

    aux_key = torch.empty(
        (1, beam_width, 1, max_steps, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    aux_value = torch.empty_like(aux_key)
    assert select_unshared_kv_fused(
        parent.reshape(-1),
        0,
        [key],
        [value],
        1,
        aux_key=aux_key,
        aux_value=aux_value,
    )
    assert torch.equal(key, expected_key)
    assert torch.equal(value, expected_value)


def test_group_cuda_fail_fast_fixed_width_and_dtype() -> None:
    inputs = _inputs(1, 4, 1)
    # The historical ``top_k`` argument denotes candidate width K, so the guard
    # compares it against the tensor's column extent rather than the beam width.
    with pytest.raises(ValueError, match=r"candidate_width == top_tokens\.shape\[1\]"):
        beam_search_group_cuda(*inputs, 1, 2)
    with pytest.raises(TypeError, match="cumulative_logprobs"):
        beam_search_group_cuda(inputs[0].half(), *inputs[1:], 1, 4)


@pytest.mark.parametrize(
    ("beam_width", "candidate_width", "top_k"),
    [(8, 8, 4), (8, 64, 2), (1024, 256, 512)],
)
def test_group_public_cuda_rejects_survivor_count_below_beam_width(
    beam_width: int,
    candidate_width: int,
    top_k: int,
) -> None:
    """``K != W`` is supported; ``top_k != W`` is not, and must not be silent.

    The CUDA backend emits exactly one survivor per beam row and fuses that
    width into its parent grouping, so a narrower selection cannot be honoured.
    ``K`` stays free: the candidate width is read from the tensors, and every
    case here also exercises ``candidate_width != beam_width``.
    """
    inputs = _inputs(1, beam_width, 1, candidate_width=candidate_width)
    with pytest.raises(NotImplementedError, match="always emits beam_width survivors"):
        beam_search_group(*inputs, 1, top_k)


def test_group_public_cuda_survivor_count_error_is_still_a_value_error() -> None:
    """Before the dispatch checked ``top_k``, the callee raised ``ValueError``.

    The new error stays catchable as one so a caller written against the old
    exception keeps working.
    """
    inputs = _inputs(1, 8, 1)
    with pytest.raises(ValueError, match="always emits beam_width survivors"):
        beam_search_group(*inputs, 1, 4)
    assert issubclass(UnsupportedSurvivorCountError, NotImplementedError)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("reshape", "error", "match"),
    [
        (lambda t: t.reshape(-1), ValueError, "sequence must be contiguous int32"),
        (lambda t: t.reshape(1, -1), ValueError, "sequence must be contiguous int32"),
        (lambda t: t.unsqueeze(0), ValueError, "sequence must be contiguous int32"),
    ],
)
def test_group_public_cuda_malformed_sequence_keeps_the_validator_error(
    reshape: Any, error: type[Exception], match: str
) -> None:
    """A wrong-rank ``sequence`` must not surface as an ``IndexError``.

    The dispatch used to read ``sequence.shape[1]`` before any validation.
    """
    cumulative, top_tokens, top_logprobs, sequence = _inputs(1, 4, 1)
    with pytest.raises(error, match=match):
        beam_search_group(cumulative, top_tokens, top_logprobs, reshape(sequence), 1, 4)


@pytest.mark.parametrize(  # type: ignore[misc]
    "reshape", [lambda t: t.reshape(-1), lambda t: t.unsqueeze(0)]
)
def test_group_public_cuda_malformed_top_tokens_keeps_the_validator_error(
    reshape: Any,
) -> None:
    """A wrong-rank ``top_tokens`` must not surface as an ``IndexError``.

    The dispatch used to forward ``top_tokens.shape[1]`` as the candidate width.
    """
    cumulative, top_tokens, top_logprobs, sequence = _inputs(1, 4, 1)
    with pytest.raises(TypeError, match="top_tokens must be a 2-D torch.int32 tensor"):
        beam_search_group(cumulative, reshape(top_tokens), top_logprobs, sequence, 1, 4)


@pytest.mark.parametrize("top_k", [8.0, True, "8", None])  # type: ignore[misc]
def test_group_public_cuda_rejects_a_non_int_survivor_count(top_k: Any) -> None:
    """``8.0 != 8`` is False, so the width check alone accepted a float."""
    inputs = _inputs(1, 8, 1)
    with pytest.raises(TypeError, match="top_k must be an int"):
        beam_search_group(*inputs, 1, top_k)


@pytest.mark.parametrize("top_k", [0, -1])  # type: ignore[misc]
def test_group_public_cuda_rejects_a_non_positive_survivor_count(top_k: int) -> None:
    inputs = _inputs(1, 8, 1)
    with pytest.raises(ValueError, match="top_k must be positive"):
        beam_search_group(*inputs, 1, top_k)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("widths", "name"),
    [
        ((4.0, 4, 4), "beam_width"),
        ((4, 4.0, 4), "candidate_width"),
        ((4, 4, 4.0), "selection_width"),
    ],
)
def test_scratch_slots_rejects_non_int_widths(widths: tuple[Any, Any, Any], name: str) -> None:
    """The tiling does integer bit arithmetic; a float must fail up front."""
    with pytest.raises(TypeError, match=f"{name} must be an int"):
        beam_selection_scratch_slots(*widths)


def test_group_public_cuda_accepts_survivor_count_equal_to_beam_width() -> None:
    """The supported geometry still runs, with ``K != W`` decoupled."""
    beam_width, candidate_width = 8, 64
    inputs = _inputs(1, beam_width, 1, candidate_width=candidate_width)
    actual = beam_search_group(*inputs, 1, beam_width)
    expected = _beam_search_group_torch(*inputs, 1, beam_width)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("case", ["holes", "unordered", "narrow", "all_invalid", "ties", "wide"])
def test_group_cuda_step_zero_masked_stable_selection(compact: bool, case: str) -> None:
    beam_width, candidate_width = {
        "holes": (2, 4),
        "unordered": (2, 4),
        "narrow": (8, 1),
        "all_invalid": (3, 5),
        "ties": (3, 5),
        "wide": (512, 256),
    }[case]
    batch_size = 2
    cumulative, tokens, scores, sequence = _inputs(
        batch_size,
        beam_width,
        0,
        candidate_width=candidate_width,
    )
    valid = torch.ones_like(tokens, dtype=torch.bool)
    if case == "holes":
        scores[:] = torch.tensor([9.0, 8.0, 7.0, 6.0], device="cuda")
        valid[:, 1] = False
    elif case == "unordered":
        scores[:] = torch.tensor([6.0, 7.0, 9.0, 8.0], device="cuda")
    elif case == "all_invalid":
        valid.zero_()
    elif case == "ties":
        scores.zero_()
        valid[:, 1] = False
    rows = torch.arange(batch_size, device="cuda") * beam_width
    first_scores = cumulative[rows] + scores[rows]
    first_valid = valid[rows]
    # Stable sorting only the valid indices keeps invalid mask holes out of Top-W.
    expected_flat = torch.full((batch_size, beam_width), -1, dtype=torch.int32, device="cuda")
    expected_scores = torch.full_like(expected_flat, -float("inf"), dtype=torch.float32)
    expected_tokens = torch.zeros_like(expected_flat)
    expected_parent = torch.full_like(expected_flat, -1)
    expected_prefix = torch.empty_like(expected_flat)
    for request in range(batch_size):
        candidates = torch.nonzero(first_valid[request], as_tuple=True)[0]
        order = torch.argsort(first_scores[request, candidates], descending=True, stable=True)
        selected = candidates[order[:beam_width]]
        count = selected.numel()
        expected_flat[request, :count] = selected.to(torch.int32)
        expected_scores[request, :count] = first_scores[request, selected]
        expected_tokens[request, :count] = tokens[rows[request], selected]
        expected_parent[request, :count] = request * beam_width
        expected_prefix[request] = request * beam_width + count
    if compact:
        cumulative, tokens, scores, valid = (
            value[rows].contiguous() for value in (cumulative, tokens, scores, valid)
        )
    inputs = (cumulative, tokens, scores, sequence)
    # Exercise the historical keyword on both public entry points.
    outputs = beam_search_group_cuda(
        *inputs,
        current_step=0,
        top_k=candidate_width,
        candidate_valid_mask=valid,
    )
    scratch = allocate_beam_selection_scratch(2, 1, 1, 1, torch.device("cuda"))
    flat = torch.full_like(expected_flat, 12345)
    beam_search_group_cuda_out(
        *inputs,
        current_step=0,
        top_k=candidate_width,
        out_token_ids=outputs[0],
        out_parent_indices=outputs[1],
        out_scores=outputs[2],
        out_parent_count_prefix=outputs[3],
        out_sequence=outputs[4],
        local_scores=scratch.local_scores,
        local_flat_indices=scratch.local_flat_indices,
        merge_scores=scratch.merge_scores,
        merge_flat_indices=scratch.merge_flat_indices,
        candidate_valid_mask=valid,
        out_grouped_flat_indices=flat,
    )
    for actual, expected in zip(
        outputs[:4],
        (
            expected_tokens,
            expected_parent,
            expected_scores,
            expected_prefix,
        ),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(flat, expected_flat)
    assert torch.equal(outputs[4][:, :, 0], expected_tokens)
    assert torch.count_nonzero(outputs[4][:, :, 1:]) == 0
