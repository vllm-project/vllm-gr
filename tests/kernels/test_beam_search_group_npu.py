# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright 2025 The xLLM Authors. All Rights Reserved.
# Adapted from https://github.com/xLLM-AI/xllm-ops

"""NPU precision guard tests for the fused ``beam_search_group`` op.

Derived from xllm-ops' ``test_beam_search_group.py``.  The CPU golden is the
package's own ``_beam_search_group_torch`` reference (stable tie-breaking,
global beam indices, prefix sums), whose semantics were already locked down by
``test_onerec_decode_ops.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.ops.onerec_decode import _beam_search_group_torch

# mypy: disable-error-code="misc"


pytestmark = pytest.mark.npu_test


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
        (1, 8, 8, 3),
        (2, 16, 16, 3),
        (1, 8, 8, 4),
        (2, 16, 16, 4),
        (1, 8, 8, 5),
        (2, 16, 16, 5),
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


@pytest.mark.l2
@pytest.mark.parametrize("request_num", [1, 2])
@pytest.mark.parametrize("capacity,current_step", [(4, 2), (6, 2), (6, 4), (6, 5)])
def test_group_compacts_history_from_configured_capacity(
    request_num: int, capacity: int, current_step: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shorter terminal request must not expose unused history to the op."""
    from vllm_gr.ops.npu import beam_ops
    from vllm_gr.v1.beam.decision import _run_npu_decision

    beam_width = 8
    sentinel = -1234567
    inputs = _build_inputs(request_num, beam_width, current_step)
    log_probs, top_tokens, top_probs, history = inputs
    expected = _beam_search_group_torch(*inputs, current_step, beam_width)
    sequence = torch.full(
        (request_num, beam_width, capacity), sentinel, dtype=torch.int32, device=DEVICE
    )
    sequence[:, :, : current_step + 1].copy_(history.to(DEVICE))
    buffers = SimpleNamespace(
        worker_decode_steps=current_step,
        sequence=sequence,
        beam_scores=log_probs.to(DEVICE).reshape(request_num, beam_width),
        group_token_num=torch.zeros((request_num, beam_width), dtype=torch.int32, device=DEVICE),
    )
    session = SimpleNamespace(
        context=SimpleNamespace(decision=buffers),
        beam_width=beam_width,
        sessions=tuple(
            SimpleNamespace(_state=SimpleNamespace(is_last_step=True)) for _ in range(request_num)
        ),
    )
    native_group = beam_ops.beam_search_group
    native_calls = 0

    def checked_group(
        scores: torch.Tensor,
        tokens: torch.Tensor,
        probabilities: torch.Tensor,
        compact_history: torch.Tensor,
        step: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal native_calls
        # The raw binding allocates current_step + 1 output slots. Its input
        # history must have the same physical stride before native execution.
        assert compact_history.shape == (request_num, beam_width, current_step + 1)
        assert compact_history.is_contiguous()
        native_calls += 1
        out_tokens, out_index, out_scores, out_prefix, out_sequence = native_group(
            scores, tokens, probabilities, compact_history, step, top_k
        )
        return out_tokens, out_index, out_scores, out_prefix, out_sequence

    monkeypatch.setattr(beam_ops, "beam_search_group", checked_group)
    actual = _run_npu_decision(
        session, top_tokens.to(DEVICE), top_probs.to(DEVICE), True, beam_width
    )
    torch.npu.synchronize()
    assert actual is not None
    assert native_calls == 1
    exp_tokens, exp_index, exp_scores, exp_prefix, exp_sequence = expected
    torch.testing.assert_close(actual["out_token_ids"].cpu(), exp_tokens, rtol=0, atol=0)
    torch.testing.assert_close(actual["out_token_index"].cpu(), exp_index, rtol=0, atol=0)
    torch.testing.assert_close(actual["out_log_probs"].cpu(), exp_scores, rtol=0, atol=1e-5)
    torch.testing.assert_close(actual["out_sequence"].cpu(), exp_sequence, rtol=0, atol=0)
    torch.testing.assert_close(
        buffers.sequence[:, :, : current_step + 1].cpu(), exp_sequence, rtol=0, atol=0
    )
    torch.testing.assert_close(buffers.beam_scores.cpu(), exp_scores, rtol=0, atol=1e-5)
    request_offsets = torch.arange(request_num, dtype=torch.int32).unsqueeze(1) * beam_width
    torch.testing.assert_close(
        buffers.group_token_num.cpu(), exp_prefix - request_offsets, rtol=0, atol=0
    )
    assert buffers.sequence[:, :, current_step + 1 :].eq(sentinel).all().item()
    assert buffers.worker_decode_steps == current_step + 1


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
