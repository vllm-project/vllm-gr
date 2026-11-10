# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU precision guard tests for the fused ``beam_search_group`` op.

Derived from xllm-ops' ``test_beam_search_group.py``.  The CPU golden is the
package's own ``_beam_search_group_torch`` reference (stable tie-breaking,
global beam indices, prefix sums), whose semantics were already locked down by
``test_onerec_decode_ops.py``.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gr.ops.onerec_decode import _beam_search_group_torch

# mypy: disable-error-code="misc"


pytestmark = pytest.mark.npu


try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"torch-npu is not installed: {exc}", allow_module_level=True)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is not available", allow_module_level=True)

try:
    import custom_ops_lib as custom_ops
except ImportError as exc:
    pytest.skip(f"custom_ops_lib is not installed: {exc}", allow_module_level=True)


DEVICE = torch.device("npu")


def _build_inputs(
    request_num: int, beam_width: int, current_step: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(10)
    log_probs = torch.randn((request_num * beam_width, 1), dtype=torch.float32)
    top_tokens = torch.randint(0, 10000, (request_num * beam_width, beam_width), dtype=torch.int32)
    top_probs = torch.randn((request_num * beam_width, beam_width), dtype=torch.float32)
    sequence = torch.randint(
        0, 10000, (request_num, beam_width, current_step + 1), dtype=torch.int32
    )
    return log_probs, top_tokens, top_probs, sequence


@pytest.mark.parametrize(
    "request_num,beam_width,top_k,current_step",
    [
        (1, 8, 8, 0),
        (1, 8, 8, 1),
        (1, 8, 8, 2),
        (2, 8, 8, 1),
        (2, 16, 16, 2),
        (1, 8, 16, 1),
    ],
)
def test_beam_search_group_npu_matches_torch_reference(
    request_num: int,
    beam_width: int,
    top_k: int,
    current_step: int,
) -> None:
    """Non-final decode steps: tokens/parents/prefix sums vs CPU golden."""
    inputs = _build_inputs(request_num, beam_width, current_step)
    log_probs, top_tokens, top_probs, sequence = inputs

    expected = _beam_search_group_torch(*inputs, current_step, top_k)
    actual = custom_ops.beam_search_group(
        log_probs.to(DEVICE),
        top_tokens.to(DEVICE),
        top_probs.to(DEVICE),
        sequence.to(DEVICE),
        current_step,
        top_k,
    )
    torch.npu.synchronize()
    assert len(actual) == 5

    (
        exp_tokens,
        exp_index,
        exp_logprobs,
        exp_prefix,
        exp_sequence,
    ) = expected
    (
        npu_tokens,
        npu_index,
        npu_logprobs,
        npu_prefix,
        npu_sequence,
    ) = actual

    assert npu_tokens.shape == (request_num, top_k)
    assert npu_index.shape == (request_num, top_k)
    assert npu_logprobs.shape == (request_num, top_k)
    assert npu_prefix.shape == (request_num, beam_width)
    assert npu_sequence.shape == (request_num, top_k, current_step + 1)

    if current_step == 0:
        # Step 0 is a legacy kernel-initialization path (flat rows [0,
        # request_num) get their own first candidate; other outputs stay
        # zero).  The torch reference implements the same contract since the
        # R10 alignment, so compare against it directly.
        torch.testing.assert_close(npu_sequence.cpu(), exp_sequence, rtol=0, atol=0)
        torch.testing.assert_close(npu_tokens.cpu(), exp_tokens, rtol=0, atol=0)
        torch.testing.assert_close(npu_index.cpu(), exp_index, rtol=0, atol=0)
        torch.testing.assert_close(npu_logprobs.cpu(), exp_logprobs, rtol=0, atol=0)
        torch.testing.assert_close(npu_prefix.cpu(), exp_prefix, rtol=0, atol=0)
        return

    torch.testing.assert_close(npu_tokens.cpu(), exp_tokens, rtol=0, atol=0)
    torch.testing.assert_close(npu_index.cpu(), exp_index, rtol=0, atol=0)
    torch.testing.assert_close(npu_logprobs.cpu(), exp_logprobs, rtol=0, atol=1e-5)
    torch.testing.assert_close(npu_prefix.cpu(), exp_prefix, rtol=0, atol=0)
    torch.testing.assert_close(npu_sequence.cpu(), exp_sequence, rtol=0, atol=0)


def test_beam_search_group_npu_output_contract() -> None:
    """Output dtypes/devices follow the xllm contract."""
    request_num, beam_width, top_k, current_step = 2, 8, 8, 1
    inputs = _build_inputs(request_num, beam_width, current_step)

    outputs = custom_ops.beam_search_group(
        inputs[0].to(DEVICE),
        inputs[1].to(DEVICE),
        inputs[2].to(DEVICE),
        inputs[3].to(DEVICE),
        current_step,
        top_k,
    )
    torch.npu.synchronize()

    tokens, index, logprobs, prefix, sequence = outputs
    assert tokens.dtype == torch.int32
    assert index.dtype == torch.int32
    assert logprobs.dtype == torch.float32
    assert prefix.dtype == torch.int32
    assert sequence.dtype == torch.int32
    for tensor in outputs:
        assert tensor.device.type == "npu"
