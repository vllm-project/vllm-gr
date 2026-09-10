# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.ops.cuda.rec_constrained_topk import (
    _rec_constrained_topk_cuda_out_unchecked,
    rec_constrained_topk_cuda,
    rec_constrained_topk_cuda_out,
    validate_cuda_sort_capacity,
)
from vllm_gr.ops.rec_constrained_topk import rec_constrained_topk_torch

# pytest parametrization decorators are untyped under strict mypy.
# mypy: disable-error-code="misc"


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _inputs(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    table = make_table()
    device = torch.device("cuda")
    logits = torch.tensor([[-8, -5, 1, 7, 3, 9], [9, 4, -3, 6, 1, 2]], dtype=dtype, device=device)
    prefixes = torch.tensor([[1, 2, 0], [3, 5, 0]], dtype=torch.int32, device=device)
    names = (
        "first_token_ids",
        "prefix1_offsets",
        "prefix1_values",
        "prefix1_pair_keys",
        "prefix2_value_offsets",
        "prefix2_values",
    )
    tables = tuple(
        torch.from_numpy(getattr(table, name).copy()).to(device=device) for name in names
    )
    return (
        logits,
        prefixes,
        *tables,
        torch.tensor([0.0, 2.0], dtype=torch.float32, device=device),
    )


def _uniform_degree_inputs(
    dtype: torch.dtype,
    rows: int,
    degree: int,
) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    vocab_size = max(512, degree + 1)
    generator = torch.Generator(device=device).manual_seed(20260827)
    logits = torch.randn(
        (rows, vocab_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    prefixes = torch.zeros((rows, 3), dtype=torch.int32, device=device)
    candidates = torch.arange(degree, dtype=torch.int32, device=device)
    prefix1_offsets = torch.full(
        (vocab_size + 1,),
        degree,
        dtype=torch.int32,
        device=device,
    )
    prefix1_offsets[0] = 0
    pair_keys = torch.zeros(1, dtype=torch.int64, device=device)
    prefix2_offsets = torch.tensor([0, degree], dtype=torch.int32, device=device)
    temperatures = torch.tensor(
        [float(index % 3) for index in range(rows)],
        dtype=torch.float32,
        device=device,
    )
    return (
        logits,
        prefixes,
        candidates,
        prefix1_offsets,
        candidates,
        pair_keys,
        prefix2_offsets,
        candidates,
        temperatures,
    )


def _candidate_domain_logprobs(
    inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    current_step: int,
    vocab_offset: int = 0,
) -> torch.Tensor:
    """Reference scores normalized over each row's legal candidate set."""
    (
        logits,
        sequence_group,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        temperatures,
    ) = inputs
    prefixes = sequence_group.reshape(sequence_group.shape[0], -1)
    vocab_end = vocab_offset + logits.shape[1]
    output = torch.full(token_ids.shape, -1.0e20, dtype=torch.float32, device=logits.device)

    for row in range(logits.shape[0]):
        if current_step == 0:
            candidates = first_token_ids
        elif current_step == 1:
            first = int(prefixes[row, 0].item())
            if first < 0 or first + 1 >= prefix1_offsets.numel():
                continue
            begin = int(prefix1_offsets[first].item())
            end = int(prefix1_offsets[first + 1].item())
            candidates = prefix1_values[begin:end]
        else:
            vocab_size = prefix1_offsets.numel() - 1
            first = int(prefixes[row, 0].item())
            second = int(prefixes[row, 1].item())
            if not (0 <= first < vocab_size and 0 <= second < vocab_size):
                continue
            key = first * vocab_size + second
            pair_index = int(
                torch.searchsorted(
                    prefix1_pair_keys,
                    torch.tensor(key, device=prefix1_pair_keys.device),
                ).item()
            )
            if (
                pair_index >= prefix1_pair_keys.numel()
                or int(prefix1_pair_keys[pair_index].item()) != key
            ):
                continue
            begin = int(prefix2_value_offsets[pair_index].item())
            end = int(prefix2_value_offsets[pair_index + 1].item())
            candidates = prefix2_values[begin:end]

        candidates = candidates[candidates.ge(vocab_offset) & candidates.lt(vocab_end)].long()
        if candidates.numel() == 0:
            continue
        temperature = temperatures[0] if temperatures.numel() == 1 else temperatures[row]
        divisor = torch.where(temperature == 0, torch.ones_like(temperature), temperature)
        row_logits = logits[row].float() / divisor
        candidate_logits = row_logits.index_select(0, candidates - vocab_offset)
        row_valid = valid_mask[row]
        selected = token_ids[row, row_valid].long() - vocab_offset
        output[row, row_valid] = row_logits.index_select(0, selected) - torch.logsumexp(
            candidate_logits,
            dim=0,
        )
    return output


def _assert_cuda_supported_shape(
    dtype: torch.dtype,
    step: int,
    rows: int,
    top_k: int,
) -> None:
    degree = 300
    inputs = _uniform_degree_inputs(dtype, rows, degree)
    actual = rec_constrained_topk_cuda(
        *inputs,
        current_step=step,
        top_k=top_k,
        max_prefix1_degree=degree,
        max_prefix2_degree=degree,
    )
    expected_tokens, _ = rec_constrained_topk_torch(
        *inputs,
        current_step=step,
        top_k=top_k,
        max_prefix1_degree=degree,
        max_prefix2_degree=degree,
    )
    expected_counts = torch.full((rows,), degree, dtype=torch.int32, device="cuda")
    expected_logprobs = _candidate_domain_logprobs(
        inputs,
        expected_tokens,
        actual[2],
        current_step=step,
    )
    assert torch.equal(actual[0], expected_tokens.to(torch.int32))
    torch.testing.assert_close(actual[1], expected_logprobs, rtol=2e-3, atol=2e-3)
    assert actual[2].all()
    assert torch.equal(actual[3], expected_counts)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("top_k", [1, 2, 4, 8, 32, 64, 128, 256])
def test_cuda_supported_specialization_matrix(
    dtype: torch.dtype,
    step: int,
    top_k: int,
) -> None:
    _assert_cuda_supported_shape(dtype, step, rows=4, top_k=top_k)


@pytest.mark.parametrize("rows", [1, 2, 256])
def test_cuda_supported_row_boundaries(rows: int) -> None:
    _assert_cuda_supported_shape(torch.float16, step=1, rows=rows, top_k=8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_cuda_matches_torch_reference(dtype: torch.dtype, step: int, top_k: int) -> None:
    table = make_table()
    inputs = _inputs(dtype)
    actual_tokens, actual_logprobs, valid_mask, valid_counts = rec_constrained_topk_cuda(
        *inputs,
        current_step=step,
        top_k=top_k,
        max_prefix1_degree=table.metadata.max_prefix1_degree,
        max_prefix2_degree=table.metadata.max_prefix2_degree,
    )
    expected_tokens, _ = rec_constrained_topk_torch(
        *inputs,
        current_step=step,
        top_k=top_k,
        max_prefix1_degree=table.metadata.max_prefix1_degree,
        max_prefix2_degree=table.metadata.max_prefix2_degree,
    )
    expected_counts = torch.tensor(
        [{0: 2, 1: 2, 2: 2}[step], {0: 2, 1: 1, 2: 2}[step]],
        dtype=torch.int32,
        device="cuda",
    )
    assert torch.equal(valid_counts, expected_counts)
    expected_mask = torch.arange(top_k, device="cuda").unsqueeze(0) < valid_counts.unsqueeze(1)
    assert torch.equal(valid_mask, expected_mask)
    assert torch.equal(actual_tokens, expected_tokens.to(torch.int32))
    expected_logprobs = _candidate_domain_logprobs(
        inputs,
        expected_tokens,
        expected_mask,
        current_step=step,
    )
    torch.testing.assert_close(actual_logprobs, expected_logprobs, rtol=2e-3, atol=2e-3)
    assert actual_tokens.dtype == torch.int32
    assert actual_logprobs.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_sliced_logits_keep_global_table_token_ids(dtype: torch.dtype) -> None:
    table = make_table()
    inputs = list(_inputs(dtype))
    full_logits = inputs[0]
    inputs[0] = full_logits[:, 1:].contiguous()

    actual_tokens, actual_logprobs, valid_mask, valid_counts = rec_constrained_topk_cuda(
        *inputs,
        current_step=1,
        top_k=2,
        max_prefix1_degree=table.metadata.max_prefix1_degree,
        max_prefix2_degree=table.metadata.max_prefix2_degree,
        vocab_offset=1,
        table_vocab_size=table.metadata.vocab_size,
    )
    reference_inputs = list(inputs)
    reference_inputs[0] = full_logits
    expected_tokens, _ = rec_constrained_topk_torch(
        *reference_inputs,
        current_step=1,
        top_k=2,
        max_prefix1_degree=table.metadata.max_prefix1_degree,
        max_prefix2_degree=table.metadata.max_prefix2_degree,
    )

    assert torch.equal(actual_tokens, expected_tokens.to(torch.int32))
    expected_logprobs = _candidate_domain_logprobs(
        inputs,
        actual_tokens,
        valid_mask,
        current_step=1,
        vocab_offset=1,
    )
    torch.testing.assert_close(actual_logprobs, expected_logprobs, rtol=2e-3, atol=2e-3)
    assert valid_counts.cpu().tolist() == [2, 1]
    expected_mask = torch.tensor([[True, True], [True, False]], device="cuda")
    assert torch.equal(valid_mask, expected_mask)


def test_cuda_sliced_logits_exclude_table_tokens_outside_window() -> None:
    table = make_table()
    inputs = list(_inputs(torch.float16))
    inputs[0] = inputs[0][:, 2:].contiguous()

    tokens, logprobs, valid_mask, valid_counts = rec_constrained_topk_cuda(
        *inputs,
        current_step=0,
        top_k=2,
        max_prefix1_degree=table.metadata.max_prefix1_degree,
        max_prefix2_degree=table.metadata.max_prefix2_degree,
        vocab_offset=2,
        table_vocab_size=table.metadata.vocab_size,
    )

    assert tokens[:, 0].cpu().tolist() == [3, 3]
    assert valid_counts.cpu().tolist() == [1, 1]
    assert torch.equal(valid_mask, torch.tensor([[True, False], [True, False]], device="cuda"))
    torch.testing.assert_close(logprobs[:, 0], torch.zeros(2, device="cuda"), rtol=0, atol=0)


@pytest.mark.parametrize("step", [0, 1, 2])
def test_cuda_exact_tie_uses_lower_table_offset(step: int) -> None:
    table = make_table()
    inputs = list(_inputs(torch.float16))
    inputs[0].zero_()
    actual = rec_constrained_topk_cuda(
        *inputs,
        current_step=step,
        top_k=2,
        max_prefix1_degree=table.metadata.max_prefix1_degree,
        max_prefix2_degree=table.metadata.max_prefix2_degree,
    )
    expected_tokens, _ = rec_constrained_topk_torch(
        *inputs,
        current_step=step,
        top_k=2,
        max_prefix1_degree=table.metadata.max_prefix1_degree,
        max_prefix2_degree=table.metadata.max_prefix2_degree,
    )
    expected_logprobs = _candidate_domain_logprobs(
        inputs,
        expected_tokens,
        actual[2],
        current_step=step,
    )
    assert torch.equal(actual[0], expected_tokens.to(torch.int32))
    torch.testing.assert_close(actual[1], expected_logprobs, rtol=2e-3, atol=2e-3)


def test_cuda_invalid_prefix_returns_only_padding() -> None:
    inputs = list(_inputs(torch.float16))
    inputs[1] = torch.tensor([[-1, 0, 0], [6, 0, 0]], dtype=torch.int32, device="cuda")
    tokens, logprobs, valid_mask, counts = rec_constrained_topk_cuda(
        *inputs,
        current_step=1,
        top_k=4,
        max_prefix1_degree=2,
        max_prefix2_degree=2,
    )
    assert tokens.shape == (2, 4)
    assert torch.equal(tokens, inputs[4][0].expand_as(tokens))
    assert counts.cpu().tolist() == [0, 0]
    assert not valid_mask.any()
    assert (logprobs == -1.0e20).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_step_zero_supports_top_k_256(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    vocab_size = 512
    first = torch.arange(300, dtype=torch.int32, device=device)
    logits = torch.randn((2, vocab_size), dtype=dtype, device=device)
    prefixes = torch.zeros((2, 3), dtype=torch.int32, device=device)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
    offsets = torch.zeros(vocab_size + 1, dtype=torch.int32, device=device)
    pair_offsets = torch.zeros(1, dtype=torch.int32, device=device)
    inputs = (
        logits,
        prefixes,
        first,
        offsets,
        empty_i32,
        empty_i64,
        pair_offsets,
        empty_i32,
        torch.ones(1, dtype=torch.float32, device=device),
    )
    tokens, logprobs, valid_mask, counts = rec_constrained_topk_cuda(
        *inputs, current_step=0, top_k=256
    )
    expected_tokens, _ = rec_constrained_topk_torch(*inputs, current_step=0, top_k=256)
    expected_logprobs = _candidate_domain_logprobs(
        inputs,
        expected_tokens,
        valid_mask,
        current_step=0,
    )
    assert torch.equal(tokens, expected_tokens.to(torch.int32))
    torch.testing.assert_close(logprobs, expected_logprobs, rtol=2e-3, atol=2e-3)
    assert counts.cpu().tolist() == [300, 300]
    assert valid_mask.all()


def test_cuda_out_variant_supports_fixed_address_graph_replay() -> None:
    inputs = _inputs(torch.float16)
    rows, top_k = inputs[0].shape[0], 4
    outputs = (
        torch.empty((rows, top_k), dtype=torch.int32, device="cuda"),
        torch.empty((rows, top_k), dtype=torch.float32, device="cuda"),
        torch.empty((rows, top_k), dtype=torch.bool, device="cuda"),
        torch.empty((rows,), dtype=torch.int32, device="cuda"),
    )
    addresses = tuple(tensor.data_ptr() for tensor in outputs)

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rec_constrained_topk_cuda_out(
            *inputs,
            1,
            top_k,
            2,
            2,
            *outputs,
        )
        return outputs

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    inputs[0].copy_(inputs[0].flip(1))
    expected_tokens, _ = rec_constrained_topk_torch(
        *inputs,
        current_step=1,
        top_k=top_k,
        max_prefix1_degree=2,
        max_prefix2_degree=2,
    )
    graph.replay()
    torch.accelerator.synchronize()
    expected_logprobs = _candidate_domain_logprobs(
        inputs,
        expected_tokens,
        outputs[2],
        current_step=1,
    )
    assert tuple(tensor.data_ptr() for tensor in outputs) == addresses
    assert torch.equal(outputs[0], expected_tokens.to(torch.int32))
    torch.testing.assert_close(outputs[1], expected_logprobs, rtol=2e-3, atol=2e-3)


def test_cuda_out_variant_writes_optional_wire_logprobs_in_same_kernel() -> None:
    inputs = _inputs(torch.float16)
    rows, top_k = inputs[0].shape[0], 4
    outputs = (
        torch.empty((rows, top_k), dtype=torch.int32, device="cuda"),
        torch.empty((rows, top_k), dtype=torch.float32, device="cuda"),
        torch.empty((rows, top_k), dtype=torch.bool, device="cuda"),
        torch.empty((rows,), dtype=torch.int32, device="cuda"),
    )
    wire = torch.empty((rows, top_k), dtype=torch.float32, device="cuda")

    rec_constrained_topk_cuda_out(
        *inputs,
        1,
        top_k,
        2,
        2,
        *outputs,
        out_wire_logprobs=wire,
    )

    assert torch.equal(wire[outputs[2]], outputs[1][outputs[2]])
    assert torch.isneginf(wire[~outputs[2]]).all()
    assert torch.equal(
        outputs[1][~outputs[2]],
        torch.full_like(outputs[1][~outputs[2]], -1.0e20),
    )


@pytest.mark.parametrize("offset", [0, 1])
def test_cuda_out_variant_rejects_exact_and_partial_output_alias(offset: int) -> None:
    inputs = _inputs(torch.float16)
    rows, top_k = inputs[0].shape[0], 4
    element_count = rows * top_k
    shared = torch.empty(element_count + offset, dtype=torch.float32, device="cuda")
    out_logprobs = shared[:element_count].view(rows, top_k)
    out_wire_logprobs = shared[offset : offset + element_count].view(rows, top_k)
    outputs = (
        torch.empty((rows, top_k), dtype=torch.int32, device="cuda"),
        out_logprobs,
        torch.empty((rows, top_k), dtype=torch.bool, device="cuda"),
        torch.empty((rows,), dtype=torch.int32, device="cuda"),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        rec_constrained_topk_cuda_out(
            *inputs,
            1,
            top_k,
            2,
            2,
            *outputs,
            out_wire_logprobs=out_wire_logprobs,
        )


def test_cuda_sort_capacity_boundary() -> None:
    validate_cuda_sort_capacity(65536, 65536, 65536)
    for values in ((65537, 1, 1), (1, 65537, 1), (1, 1, 65537)):
        with pytest.raises(ValueError, match="above supported 65536"):
            validate_cuda_sort_capacity(*values)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_unchecked_out_matches_validated_prewarmed_launch(
    dtype: torch.dtype,
) -> None:
    inputs = _uniform_degree_inputs(dtype, rows=2, degree=300)
    rows, top_k = 2, 128

    def make_outputs() -> tuple[torch.Tensor, ...]:
        return (
            torch.empty((rows, top_k), dtype=torch.int32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.float32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.bool, device="cuda"),
            torch.empty((rows,), dtype=torch.int32, device="cuda"),
            torch.empty((rows, top_k), dtype=torch.float32, device="cuda"),
        )

    expected = make_outputs()
    actual = make_outputs()
    rec_constrained_topk_cuda_out(
        *inputs,
        0,
        top_k,
        300,
        300,
        *expected[:4],
        table_vocab_size=inputs[0].shape[1],
        out_wire_logprobs=expected[4],
    )
    _rec_constrained_topk_cuda_out_unchecked(
        *inputs,
        0,
        top_k,
        300,
        300,
        *actual[:4],
        vocab_offset=0,
        table_vocab_size=inputs[0].shape[1],
        out_wire_logprobs=actual[4],
    )

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


def test_cuda_step2_empty_pair_table_returns_padding() -> None:
    inputs = list(_inputs(torch.float16))
    inputs[5] = torch.empty(0, dtype=torch.int64, device="cuda")
    inputs[6] = torch.zeros(1, dtype=torch.int32, device="cuda")
    inputs[7] = torch.empty(0, dtype=torch.int32, device="cuda")
    tokens, logprobs, valid_mask, counts = rec_constrained_topk_cuda(
        *inputs,
        current_step=2,
        top_k=4,
        max_prefix1_degree=2,
        max_prefix2_degree=0,
    )
    assert torch.equal(tokens, torch.zeros_like(tokens))
    assert torch.equal(counts, torch.zeros_like(counts))
    assert not valid_mask.any()
    assert (logprobs == -1.0e20).all()


def test_cuda_step_zero_accepts_empty_prefix_extent() -> None:
    inputs = list(_inputs(torch.float16))
    inputs[1] = torch.empty((2, 0), dtype=torch.int32, device="cuda")
    actual = rec_constrained_topk_cuda(
        *inputs,
        current_step=0,
        top_k=2,
        max_prefix1_degree=2,
        max_prefix2_degree=2,
    )
    expected_tokens, _ = rec_constrained_topk_torch(
        *inputs,
        current_step=0,
        top_k=2,
        max_prefix1_degree=2,
        max_prefix2_degree=2,
    )
    expected_logprobs = _candidate_domain_logprobs(
        inputs,
        expected_tokens,
        actual[2],
        current_step=0,
    )
    assert torch.equal(actual[0], expected_tokens.to(torch.int32))
    torch.testing.assert_close(actual[1], expected_logprobs, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("degree", [4, 8, 16])
def test_cuda_degree_less_equal_greater_than_top_k(step: int, degree: int) -> None:
    top_k = 8
    inputs = _uniform_degree_inputs(torch.float16, 4, degree)
    actual = rec_constrained_topk_cuda(
        *inputs,
        current_step=step,
        top_k=top_k,
        max_prefix1_degree=degree,
        max_prefix2_degree=degree,
    )
    expected_tokens, _ = rec_constrained_topk_torch(
        *inputs,
        current_step=step,
        top_k=top_k,
        max_prefix1_degree=degree,
        max_prefix2_degree=degree,
    )
    expected_logprobs = _candidate_domain_logprobs(
        inputs,
        expected_tokens,
        actual[2],
        current_step=step,
    )
    assert torch.equal(actual[0], expected_tokens.to(torch.int32))
    torch.testing.assert_close(actual[1], expected_logprobs, rtol=2e-3, atol=2e-3)
    expected_counts = torch.full((4,), degree, dtype=torch.int32, device="cuda")
    expected_mask = torch.arange(top_k, device="cuda") < expected_counts[:, None]
    assert torch.equal(actual[2], expected_mask)
    assert torch.equal(actual[3], expected_counts)
    assert (actual[1][~expected_mask] == -1.0e20).all()


def test_cuda_fail_fast_dtype_and_buffer_contiguity() -> None:
    inputs = list(_inputs(torch.float16))
    with pytest.raises(TypeError, match="fp16/bf16"):
        rec_constrained_topk_cuda(
            inputs[0].float(),
            *inputs[1:],
            current_step=1,
            top_k=2,
            max_prefix1_degree=2,
            max_prefix2_degree=2,
        )

    inputs = list(_inputs(torch.float16))
    padded_logits = torch.empty(
        (inputs[0].shape[0], inputs[0].shape[1] * 2),
        dtype=torch.float16,
        device="cuda",
    )
    noncontiguous_logits = padded_logits[:, ::2]
    noncontiguous_logits.copy_(inputs[0])
    assert not noncontiguous_logits.is_contiguous()
    strided_actual = rec_constrained_topk_cuda(
        noncontiguous_logits,
        *inputs[1:],
        current_step=1,
        top_k=2,
        max_prefix1_degree=2,
        max_prefix2_degree=2,
    )
    contiguous_expected = rec_constrained_topk_cuda(
        *inputs,
        current_step=1,
        top_k=2,
        max_prefix1_degree=2,
        max_prefix2_degree=2,
    )
    for actual, expected in zip(strided_actual, contiguous_expected, strict=True):
        assert torch.equal(actual, expected)

    inputs = list(_inputs(torch.float16))
    inputs[2] = inputs[2].to(torch.int64)
    with pytest.raises(TypeError, match="first_token_ids"):
        rec_constrained_topk_cuda(
            *inputs,
            current_step=1,
            top_k=2,
            max_prefix1_degree=2,
            max_prefix2_degree=2,
        )

    inputs = list(_inputs(torch.float16))
    inputs[1] = torch.zeros((2, 3, 2), dtype=torch.int32, device="cuda")[:, :, 0]
    assert not inputs[1].is_contiguous()
    with pytest.raises(ValueError, match="sequence_group must be contiguous"):
        rec_constrained_topk_cuda(
            *inputs,
            current_step=1,
            top_k=2,
            max_prefix1_degree=2,
            max_prefix2_degree=2,
        )

    inputs = list(_inputs(torch.float16))
    with pytest.raises(TypeError, match="current_step"):
        rec_constrained_topk_cuda(
            *inputs,
            current_step=True,
            top_k=2,
            max_prefix1_degree=2,
            max_prefix2_degree=2,
        )
