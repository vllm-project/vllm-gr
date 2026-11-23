# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Adapted from https://github.com/xLLM-AI/xllm-ops

"""NPU precision guard tests for the fused ``beam_search_rec_final_select`` op.

Derived from xllm-ops' ``test_onerec_final_beam_select.py``.  The CPU golden
is the package's own ``_beam_search_rec_final_select_torch`` reference, whose
semantics were already locked down by ``test_onerec_decode_ops.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.ops.onerec_decode import _beam_search_rec_final_select_torch

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
    request_num: int, beam_width: int, candidate_top_k: int, current_step: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(2026)
    log_probs = torch.randn((request_num * beam_width, 1), dtype=torch.float32)
    top_tokens = torch.randint(
        0, 10000, (request_num * beam_width, candidate_top_k), dtype=torch.int32
    )
    top_probs = torch.randn((request_num * beam_width, candidate_top_k), dtype=torch.float32)
    sequence = torch.randint(
        0, 10000, (request_num, beam_width, current_step + 1), dtype=torch.int32
    )
    return log_probs, top_tokens, top_probs, sequence


@pytest.mark.parametrize(
    "request_num,beam_width,candidate_top_k,result_width,current_step",
    [
        # Kernel constraint: result_width is a multiple of 32 and
        # candidate_top_k is a multiple of 8.
        (1, 8, 32, 32, 1),
        (1, 16, 128, 64, 2),
        (2, 16, 64, 64, 2),
        (1, 8, 32, 32, 2),
        (1, 8, 64, 32, 1),
        (1, 32, 32, 32, 3),
        (2, 32, 32, 32, 3),
        (1, 32, 32, 32, 4),
        (2, 32, 32, 32, 4),
        (1, 32, 32, 32, 5),
        (2, 32, 32, 32, 5),
    ],
)
def test_final_select_npu_matches_torch_reference(
    request_num: int,
    beam_width: int,
    candidate_top_k: int,
    result_width: int,
    current_step: int,
) -> None:
    """Last-step selection: tokens/parents/sequences vs CPU golden."""
    inputs = _build_inputs(request_num, beam_width, candidate_top_k, current_step)
    log_probs, top_tokens, top_probs, sequence = inputs

    expected = _beam_search_rec_final_select_torch(*inputs, current_step, result_width)
    actual = custom_ops.beam_search_rec_final_select(
        log_probs.to(DEVICE),
        top_tokens.to(DEVICE),
        top_probs.to(DEVICE),
        sequence.to(DEVICE),
        current_step,
        result_width,
    )
    torch.npu.synchronize()
    assert len(actual) == 4

    exp_tokens, exp_index, exp_logprobs, exp_sequence = expected
    npu_tokens, npu_index, npu_logprobs, npu_sequence = actual

    assert npu_tokens.shape == (request_num * result_width, 1)
    assert npu_index.shape == (request_num * result_width, 1)
    assert npu_logprobs.shape == (request_num * result_width, 1)
    assert npu_sequence.shape == (request_num, result_width, current_step + 1)

    torch.testing.assert_close(npu_tokens.cpu(), exp_tokens, rtol=0, atol=0)
    torch.testing.assert_close(npu_index.cpu(), exp_index, rtol=0, atol=0)
    torch.testing.assert_close(npu_logprobs.cpu(), exp_logprobs, rtol=0, atol=1e-5)
    torch.testing.assert_close(npu_sequence.cpu(), exp_sequence, rtol=0, atol=0)


@pytest.mark.l2
@pytest.mark.parametrize("request_num", [1, 2])
@pytest.mark.parametrize("capacity,current_step", [(4, 2), (6, 2), (6, 4), (6, 5)])
def test_final_select_compacts_history_from_configured_capacity(
    request_num: int, capacity: int, current_step: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final selection retains the active prefix and leaves its tail unused."""
    from vllm_gr.ops.npu import beam_ops
    from vllm_gr.v1.worker.npu.beam_decision import _run_npu_decision

    beam_width = 32
    sentinel = -1234567
    inputs = _build_inputs(request_num, beam_width, beam_width, current_step)
    log_probs, top_tokens, top_probs, history = inputs
    expected = _beam_search_rec_final_select_torch(*inputs, current_step, beam_width)
    sequence = torch.full(
        (request_num, beam_width, capacity), sentinel, dtype=torch.int32, device=DEVICE
    )
    sequence[:, :, : current_step + 1].copy_(history.to(DEVICE))
    buffers = SimpleNamespace(
        worker_decode_steps=current_step,
        sequence=sequence,
        beam_scores=log_probs.to(DEVICE).reshape(request_num, beam_width),
    )
    session = SimpleNamespace(context=SimpleNamespace(decision=buffers), beam_width=beam_width)
    native_select = beam_ops.beam_search_rec_final_select
    native_calls = 0

    def checked_select(
        scores: torch.Tensor,
        tokens: torch.Tensor,
        probabilities: torch.Tensor,
        compact_history: torch.Tensor,
        step: int,
        result_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal native_calls
        assert compact_history.shape == (request_num, beam_width, current_step + 1)
        assert compact_history.is_contiguous()
        native_calls += 1
        out_tokens, out_index, out_scores, out_sequence = native_select(
            scores, tokens, probabilities, compact_history, step, result_width
        )
        return out_tokens, out_index, out_scores, out_sequence

    monkeypatch.setattr(beam_ops, "beam_search_rec_final_select", checked_select)
    actual = _run_npu_decision(
        session, top_tokens.to(DEVICE), top_probs.to(DEVICE), True, beam_width
    )
    torch.npu.synchronize()
    assert actual is not None
    assert native_calls == 1
    exp_tokens, exp_index, exp_scores, exp_sequence = expected
    shape = (request_num, beam_width)
    torch.testing.assert_close(
        actual["out_token_ids"].cpu(), exp_tokens.reshape(shape), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["out_token_index"].cpu(), exp_index.reshape(shape), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["out_log_probs"].cpu(), exp_scores.reshape(shape), rtol=0, atol=1e-5
    )
    torch.testing.assert_close(actual["out_sequence"].cpu(), exp_sequence, rtol=0, atol=0)
    torch.testing.assert_close(
        buffers.sequence[:, :, : current_step + 1].cpu(), exp_sequence, rtol=0, atol=0
    )
    torch.testing.assert_close(
        buffers.beam_scores.cpu(), exp_scores.reshape(shape), rtol=0, atol=1e-5
    )
    assert buffers.sequence[:, :, current_step + 1 :].eq(sentinel).all().item()
    assert buffers.worker_decode_steps == current_step + 1


def test_final_select_npu_output_contract() -> None:
    """Output dtypes/devices follow the xllm contract."""
    request_num, beam_width, candidate_top_k, current_step = 1, 8, 32, 1
    result_width = 32  # kernel requires a multiple of 32
    inputs = _build_inputs(request_num, beam_width, candidate_top_k, current_step)

    outputs = custom_ops.beam_search_rec_final_select(
        inputs[0].to(DEVICE),
        inputs[1].to(DEVICE),
        inputs[2].to(DEVICE),
        inputs[3].to(DEVICE),
        current_step,
        result_width,
    )
    torch.npu.synchronize()

    tokens, index, logprobs, sequence = outputs
    assert tokens.dtype == torch.int32
    assert index.dtype == torch.int32
    assert logprobs.dtype == torch.float32
    assert sequence.dtype == torch.int32
    for tensor in outputs:
        assert tensor.device.type == "npu"
