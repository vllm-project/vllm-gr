# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
import torch

from vllm_gr.constraints.backend import INVALID_LOGIT, ConstraintMetadata
from vllm_gr.constraints.cpu_backend import CPUConstraintBackend

from .helpers import make_backend_info, make_table

INVALID_LOGIT_FP32 = torch.tensor(INVALID_LOGIT, dtype=torch.float32).item()


def _backend() -> CPUConstraintBackend:
    table = make_table()
    return CPUConstraintBackend(table, make_backend_info(table))


def _metadata(
    prefixes: list[list[int]],
    step: int,
    top_k: int = 2,
    temperature: float = 1.0,
) -> ConstraintMetadata:
    return ConstraintMetadata(
        prefix_token_ids=torch.tensor(prefixes, dtype=torch.int32),
        temperatures=torch.tensor([temperature], dtype=torch.float32),
        current_step=step,
        top_k=top_k,
    )


def _reference(
    logits: torch.Tensor, allowed: list[int], top_k: int
) -> tuple[list[int], list[float]]:
    values = logits[allowed].float()
    logprobs = torch.log_softmax(values, dim=0)
    order = torch.argsort(logprobs, descending=True)
    tokens = [allowed[int(index)] for index in order]
    probs = [float(logprobs[int(index)]) for index in order]
    while len(tokens) < top_k:
        tokens.append(0)
        probs.append(INVALID_LOGIT)
    return tokens[:top_k], probs[:top_k]


@pytest.mark.parametrize(
    ("step", "prefix", "allowed"),
    [
        (0, [0, 0], [1, 2]),
        (1, [1, 0], [3, 4]),
        (2, [1, 3], [5]),
    ],
)
def test_cpu_steps_match_independent_reference(
    step: int,
    prefix: list[int],
    allowed: list[int],
) -> None:
    backend = _backend()
    logits = torch.tensor([[9.0, -4.0, -3.0, 0.5, 2.0, 4.0, 1.0, -2.0]])

    output = backend.select(logits, _metadata([prefix], step))
    expected_tokens, expected_probs = _reference(logits[0], allowed, 2)

    assert output.token_ids.tolist()[0] == expected_tokens
    for actual, expected in zip(output.logprobs.tolist()[0], expected_probs):
        if expected == INVALID_LOGIT:
            assert actual == INVALID_LOGIT_FP32
        else:
            assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
    assert output.valid_mask.tolist()[0] == [True] * min(len(allowed), 2) + [False] * max(
        0, 2 - len(allowed)
    )


def test_legal_token_outside_global_top_k_is_still_selected() -> None:
    backend = _backend()
    # Global top-2 are illegal tokens 0 and 7. Step-1 legal tokens are 3 and 4.
    logits = torch.tensor([[100.0, 0.0, 0.0, -20.0, -30.0, 0.0, 0.0, 90.0]])
    output = backend.select(logits, _metadata([[1, 0]], step=1, top_k=2))
    assert output.token_ids.tolist() == [[3, 4]]
    assert output.valid_mask.all()


def test_greedy_zero_temperature_matches_unscaled_temperature() -> None:
    backend = _backend()
    logits = torch.tensor([[9.0, -4.0, -3.0, 0.5, 2.0, 4.0, 1.0, -2.0]])

    greedy = backend.select(logits, _metadata([[1, 0]], step=1, temperature=0.0))
    unscaled = backend.select(logits, _metadata([[1, 0]], step=1, temperature=1.0))

    assert torch.equal(greedy.token_ids, unscaled.token_ids)
    assert torch.equal(greedy.logprobs, unscaled.logprobs)


def test_cpu_backend_keeps_tables_on_host_and_no_workspace() -> None:
    backend = _backend()
    table = backend._table
    assert table is not None
    for name in table.__dataclass_fields__:
        value = getattr(table, name)
        if hasattr(value, "device"):
            assert str(value.device) == "cpu"
    assert set(CPUConstraintBackend.__slots__) == {"_table", "_info", "_closed"}


def test_unknown_prefix_produces_only_sentinel_padding() -> None:
    backend = _backend()
    output = backend.select(torch.zeros((1, 8)), _metadata([[7, 7]], step=2, top_k=2))
    assert output.valid_mask.tolist() == [[False, False]]
    assert output.logprobs.tolist() == [[INVALID_LOGIT_FP32, INVALID_LOGIT_FP32]]


def test_close_is_idempotent() -> None:
    backend = _backend()
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.select(torch.zeros((1, 8)), _metadata([[0, 0]], step=0))


@pytest.mark.npu_integration
def test_cpu_host_tables_query_npu_logits() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    backend = _backend()
    logits = torch.zeros((1, 8), dtype=torch.float16, device="npu")
    logits[0, 3] = 2
    output = backend.select(logits, _metadata([[1, 0]], step=1, top_k=2))
    torch.npu.synchronize()
    assert output.token_ids.device.type == "npu"
    assert output.token_ids[0, 0].item() == 3
