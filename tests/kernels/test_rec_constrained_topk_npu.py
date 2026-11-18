# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""NPU precision guard tests for the fused ``rec_constrained_topk`` op.

These tests intentionally call the NPU custom-op wrapper directly instead of
the public dispatcher.  That prevents a missing custom op from silently
falling back to the Torch implementation and makes the test a real NPU-op
precision check.
"""

from __future__ import annotations

import pytest
import torch

from tools.hardware_marks import hardware_marks
from vllm_gr.ops.npu.rec_constrained_topk import (
    is_available as rec_constrained_topk_npu_available,
)
from vllm_gr.ops.npu.rec_constrained_topk import (
    rec_constrained_topk_npu,
)
from vllm_gr.ops.npu.rec_constrained_topk import (
    unavailable_reason as rec_constrained_topk_npu_unavailable_reason,
)
from vllm_gr.ops.rec_constrained_topk import rec_constrained_topk_torch


# pytest parametrization decorators are untyped in the strict mypy environment.
# Keep this exception local to the NPU-only test module.
# mypy: disable-error-code="misc"


pytestmark = hardware_marks(res={"npu": "A3"}, num_cards=1)


try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"torch-npu is not installed: {exc}", allow_module_level=True)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is not available", allow_module_level=True)

if not rec_constrained_topk_npu_available():
    pytest.skip(
        f"NPU rec_constrained_topk is unavailable: {rec_constrained_topk_npu_unavailable_reason()}",
        allow_module_level=True,
    )


DEVICE = torch.device("npu")
DTYPES = [torch.float16, torch.bfloat16]


def _constraint_tables(
    vocab_size: int,
    candidate_width: int,
    pair_count: int = 4096,
) -> tuple[torch.Tensor, ...]:
    """Build a compact, sorted three-level OneRec constraint table.

    Every token has ``candidate_width`` step-1 candidates.  The first two
    tokens of every step-2 pair are valid, and each pair has the same number
    of step-2 candidates.  The regular layout keeps the expected result easy
    to reproduce while still exercising the large-table NPU path.
    """
    first_token_ids = torch.arange(candidate_width, dtype=torch.int32)

    prefix1_offsets = torch.arange(
        0,
        (vocab_size + 1) * candidate_width,
        candidate_width,
        dtype=torch.int32,
    )
    prefix1_values = (
        torch.arange(vocab_size, dtype=torch.int32).unsqueeze(1)
        + torch.arange(1, candidate_width + 1, dtype=torch.int32)
    ).remainder(vocab_size)
    prefix1_values = prefix1_values.reshape(-1)

    # Pair keys are sorted as required by the operator's binary search.
    # The runtime only requires sorted keys for the pairs used by the batch;
    # materialising all vocab_size**2 pairs would make a correctness test
    # needlessly consume gigabytes of host/NPU memory.
    pair_count = min(pair_count, vocab_size * vocab_size)
    pair_keys = torch.arange(pair_count, dtype=torch.int64)
    prefix2_degree = candidate_width
    prefix2_value_offsets = torch.arange(
        0,
        (pair_keys.numel() + 1) * prefix2_degree,
        prefix2_degree,
        dtype=torch.int32,
    )
    prefix2_values = (
        (
            torch.arange(pair_keys.numel(), dtype=torch.int64).unsqueeze(1)
            + torch.arange(1, prefix2_degree + 1, dtype=torch.int64)
        )
        .remainder(vocab_size)
        .to(torch.int32)
    )
    prefix2_values = prefix2_values.reshape(-1)

    return (
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        pair_keys,
        prefix2_value_offsets,
        prefix2_values,
    )


def _build_case(
    *,
    step: int,
    dtype: torch.dtype,
    rows: int = 32,
    vocab_size: int = 8192,
    candidate_width: int = 8,
    top_k: int = 4,
    per_row_temperature: bool = False,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], int]:
    tables = _constraint_tables(vocab_size, candidate_width)
    (
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
    ) = tables

    # Use valid prefixes for every row.  The first two columns are sufficient
    # for step 1/2; the extra column also matches the runtime's sequence view.
    sequence_group = torch.zeros((rows, 3), dtype=torch.int32)
    if step == 2:
        # Pair keys are [0, 1, ..., pair_count); use the first rows so every
        # step-2 row exercises a valid binary-search hit.
        sequence_group[:, 0] = 0
        sequence_group[:, 1] = torch.arange(rows, dtype=torch.int32)
    else:
        sequence_group[:, 0] = torch.arange(rows, dtype=torch.int32) % vocab_size
        sequence_group[:, 1] = (sequence_group[:, 0] + 1) % vocab_size
    sequence_group[:, 2] = (sequence_group[:, 1] + 1) % vocab_size

    # Build deterministic logits with a stable Top-K margin. Random bf16 logits
    # can put several constrained candidates extremely close to each other; then
    # CPU reference and NPU fused kernels may disagree on a few boundary tokens
    # due to different rounding/order stability. The precision guard should check
    # the operator contract, not accidental near-ties in random test data.
    logits = torch.full(
        (rows, vocab_size),
        -20.0,
        dtype=torch.float32,
    )

    for row in range(rows):
        if step == 0:
            candidate_ids = first_token_ids
        elif step == 1:
            prev_token = int(sequence_group[row, 0].item())
            start = int(prefix1_offsets[prev_token].item())
            end = int(prefix1_offsets[prev_token + 1].item())
            candidate_ids = prefix1_values[start:end]
        else:
            # The test constructs pair keys as [0, 1, ...]. With
            # sequence_group[row] = [0, row, ...], the pair key is exactly row.
            pair_index = int(sequence_group[row, 1].item())
            start = int(prefix2_value_offsets[pair_index].item())
            end = int(prefix2_value_offsets[pair_index + 1].item())
            candidate_ids = prefix2_values[start:end]

        # Make candidate ranking deterministic and well separated even after bf16
        # conversion. Higher-ranked candidates get larger logits.
        for rank, token_id in enumerate(candidate_ids.tolist()):
            logits[row, token_id] = 10.0 + float(candidate_ids.numel() - rank)

    logits = logits.to(dtype)

    if per_row_temperature:
        temperatures = torch.linspace(0.7, 1.3, rows, dtype=torch.float32)
    else:
        temperatures = torch.tensor(1.0, dtype=torch.float32)

    cpu_inputs = (
        logits,
        sequence_group,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        temperatures,
    )
    npu_inputs = tuple(tensor.to(DEVICE) for tensor in cpu_inputs)
    return cpu_inputs, npu_inputs, top_k


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("per_row_temperature", [False, True])
def test_rec_constrained_topk_npu_matches_torch_reference(
    step: int,
    dtype: torch.dtype,
    per_row_temperature: bool,
) -> None:
    """Guard token ids and log-prob precision for all OneRec decode steps."""
    cpu_inputs, npu_inputs, top_k = _build_case(
        step=step,
        dtype=dtype,
        per_row_temperature=per_row_temperature,
    )

    expected_tokens, expected_logprobs = rec_constrained_topk_torch(
        *cpu_inputs,
        current_step=step,
        top_k=top_k,
    )
    actual_tokens, actual_logprobs = rec_constrained_topk_npu(
        *npu_inputs,
        current_step=step,
        top_k=top_k,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual_tokens.cpu().to(expected_tokens.dtype),
        expected_tokens,
        rtol=0,
        atol=0,
    )
    # The reference accumulates in fp32 just like the NPU contract, but the
    # input logits may be fp16/bf16 and can differ by a few ulps after gather.
    torch.testing.assert_close(
        actual_logprobs.cpu(),
        expected_logprobs,
        rtol=2e-3,
        atol=2e-3,
    )


def test_rec_constrained_topk_npu_preserves_output_contract() -> None:
    """Check the NPU operator's dtype/shape contract on a larger batch."""
    cpu_inputs, npu_inputs, top_k = _build_case(
        step=1,
        dtype=torch.bfloat16,
        rows=64,
        vocab_size=16384,
        candidate_width=16,
        top_k=8,
    )
    del cpu_inputs

    tokens, logprobs = rec_constrained_topk_npu(
        *npu_inputs,
        current_step=1,
        top_k=top_k,
        max_prefix1_degree=16,
    )
    torch.npu.synchronize()

    assert tokens.shape == (64, top_k)
    assert logprobs.shape == (64, top_k)
    # The NPU custom op follows the xLLM contract and returns int32 token ids;
    # the Torch reference widens them to int64 for indexing convenience.
    assert tokens.dtype == torch.int32
    assert logprobs.dtype == torch.float32
    assert tokens.device.type == "npu"
    assert logprobs.device.type == "npu"
