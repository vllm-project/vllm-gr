# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-call"
"""Real NPU contracts for independent candidate K and survivor W.

The fused NPU selector retains its established legal-set normalization.
Its oracle below runs on CPU and enumerates that set explicitly; the CPU
Backend's full-vocabulary score domain is deliberately not substituted.
These tests do not invoke the native square-only BeamSearchGroup binding.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_gr.config import (
    CONSTRAINT_TABLE_FORMAT,
    ConstraintTableBackendEnum,
    ConstraintTableSpec,
)
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintMetadata,
    ConstraintOutput,
)
from vllm_gr.constraints.loader import load_constraint_backend
from vllm_gr.constraints.npu_backend import NPUConstraintBackend
from vllm_gr.constraints.table import (
    CONSTRAINT_TABLE_TENSOR_NAMES,
    ConstraintTable,
    ConstraintTableMetadata,
    validate_constraint_table,
)
from vllm_gr.ops.npu.rec_constrained_topk import is_available, unavailable_reason
from vllm_gr.v1.beam.attention_pool import BeamAttentionPool
from vllm_gr.v1.beam.dynamic_width import (
    DynamicBeamState,
    get_stage_pools,
    initialize_packed_stage_pools,
)

pytestmark = [pytest.mark.npu_test, pytest.mark.l2]

try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"torch-npu is unavailable: {exc}", allow_module_level=True)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is unavailable", allow_module_level=True)

DEVICE = torch.device("npu")


def _table() -> ConstraintTable:
    """Eight SA, eight SB per SA, and eight SC per pair, in a 32-token vocab."""
    offsets = np.asarray([8 * min(token, 8) for token in range(33)], dtype=np.int32)
    arrays = {
        "first_token_ids": np.arange(8, dtype=np.int32),
        "prefix1_offsets": offsets,
        "prefix1_values": np.tile(np.arange(8, 16, dtype=np.int32), 8),
        "prefix1_pair_keys": np.asarray(
            [a * 32 + b for a in range(8) for b in range(8, 16)], dtype=np.int64
        ),
        "prefix2_value_offsets": np.arange(0, 513, 8, dtype=np.int32),
        "prefix2_values": np.tile(np.arange(16, 24, dtype=np.int32), 64),
    }
    table = ConstraintTable(
        first_token_ids=arrays["first_token_ids"],
        prefix1_offsets=arrays["prefix1_offsets"],
        prefix1_values=arrays["prefix1_values"],
        prefix1_pair_keys=arrays["prefix1_pair_keys"],
        prefix2_value_offsets=arrays["prefix2_value_offsets"],
        prefix2_values=arrays["prefix2_values"],
        metadata=ConstraintTableMetadata(
            format=CONSTRAINT_TABLE_FORMAT,
            artifact_digest=None,
            tokenizer_digest="sha256:" + "b" * 64,
            vocab_size=32,
            max_prefix1_degree=8,
            max_prefix2_degree=8,
            total_bytes=int(sum(arrays[name].nbytes for name in CONSTRAINT_TABLE_TENSOR_NAMES)),
        ),
    )
    validate_constraint_table(table)
    return table


@pytest.fixture(scope="module")
def backend(tmp_path_factory: pytest.TempPathFactory) -> Iterator[NPUConstraintBackend]:
    # A detected NPU with missing custom ops is an incomplete target runtime,
    # not a successful skip and not permission to use the Torch fallback.
    if not is_available():
        pytest.fail(f"NPU is present but real constrained TopK is missing: {unavailable_reason()}")
    path = tmp_path_factory.mktemp("npu-independent-k") / "table.safetensors"
    digest = save_constraint_table_artifact(_table(), path)
    loaded = load_constraint_backend(
        ConstraintTableSpec(
            enabled=True,
            backend=ConstraintTableBackendEnum.NPU,
            path=str(path),
            artifact_digest=digest,
            tokenizer_digest="sha256:" + "b" * 64,
            max_top_k=256,
        ),
        device=DEVICE,
        vocab_size=32,
        dtype=torch.float32,
        max_rows=4,
        enable_npu_stage_vocab=False,
    )
    assert isinstance(loaded, NPUConstraintBackend)
    try:
        yield loaded
    finally:
        loaded.close()


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("k", [1, 3, 7, 16, 32, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_real_npu_constrained_k_matches_legal_set_cpu_oracle(
    backend: NPUConstraintBackend, step: int, rows: int, k: int, dtype: torch.dtype
) -> None:
    logits = torch.full((rows, 32), -12.0, dtype=dtype)
    # A large illegal logit ensures that the assertion really checks the
    # NPU legal-set domain instead of a coincidentally equal full-vocab score.
    logits[:, 31] = 20.0
    prefixes = torch.zeros((rows, 3), dtype=torch.int32)
    temperatures = torch.tensor([0.0, 0.5, 1.0, 2.0][:rows], dtype=torch.float32)
    expected_tokens, expected_scores = [], []
    allowed = list(range(8 * step, 8 * step + 8))
    for row in range(rows):
        prefixes[row, 0] = row % 8
        prefixes[row, 1] = 8 + row % 8
        for rank in range(8):
            token = allowed[(3 * rank + row) % 8]
            logits[row, token] = 8.0 - rank * 0.75
        divisor = float(temperatures[row]) or 1.0
        legal = [(float(logits[row, token]) / divisor, token) for token in allowed]
        legal.sort(reverse=True)
        log_denominator = torch.logsumexp(
            torch.tensor([score for score, _ in legal], dtype=torch.float32), dim=0
        ).item()
        expected_tokens.append([token for _, token in legal[: min(k, 8)]])
        expected_scores.append([score - log_denominator for score, _ in legal[: min(k, 8)]])

    output = backend.select(
        logits.to(DEVICE),
        ConstraintMetadata(
            prefix_token_ids=prefixes.to(DEVICE),
            temperatures=temperatures.to(DEVICE),
            current_step=step,
            top_k=k,
        ),
    )
    torch.npu.synchronize()
    tokens, scores, valid = (
        output.token_ids.cpu(),
        output.logprobs.cpu(),
        output.valid_mask.cpu(),
    )
    assert tokens.dtype == torch.int32 and scores.dtype == torch.float32
    assert valid.dtype == torch.bool
    assert tokens.shape == scores.shape == valid.shape == (rows, k)
    assert all(tensor.device.type == "npu" for tensor in _output_tensors(output))
    count = min(k, 8)
    assert valid[:, :count].all() and not valid[:, count:].any()
    assert tokens[:, :count].tolist() == expected_tokens
    torch.testing.assert_close(
        scores[:, :count], torch.tensor(expected_scores), rtol=2e-3, atol=2e-3
    )
    # Invalid token IDs have no semantic value. Their mask and sentinel score
    # must nevertheless preserve the real operator's fixed-width ABI.
    assert (scores[:, count:] == torch.tensor(INVALID_LOGIT, dtype=torch.float32)).all()


def _output_tensors(output: ConstraintOutput) -> tuple[torch.Tensor, ...]:
    return output.token_ids, output.logprobs, output.valid_mask


def _context(batch: int, packed: bool) -> Any:
    context = SimpleNamespace(
        max_batch=batch + 1,
        max_beams=4,
        max_decode_steps=3,
        registry=SimpleNamespace(num_active_sessions=0),
        pools=[BeamAttentionPool(batch + 1, 4, 1, 3, 2, DEVICE, torch.float16) for _ in range(2)],
    )
    if packed:
        initialize_packed_stage_pools(context)
    return context


def _candidates(batch: int, width: int, k: int, stage: int) -> ConstraintOutput:
    tokens, scores = [], []
    for request in range(batch):
        preferred = 0 if request % 2 == 0 else width - 1
        for parent in range(width):
            tokens.append([1000 * request + 100 * stage + 10 * parent + col for col in range(k)])
            scores.append(
                [
                    -(col + 1) / 32 - stage / 1024
                    if parent == preferred
                    else -(8 + 2 * parent + col / 32 + stage / 1024)
                    for col in range(k)
                ]
            )
    return ConstraintOutput(
        torch.tensor(tokens, dtype=torch.int32),
        torch.tensor(scores, dtype=torch.float32),
        torch.ones(batch * width, k, dtype=torch.bool),
    )


def _scalar_winners(
    scores: list[list[float]], candidates: ConstraintOutput, output_width: int, last: bool
) -> list[list[tuple[float, int, int]]]:
    """Enumerate parent/candidate pairs without production selection helpers."""
    token_rows = candidates.token_ids.tolist()
    score_rows = candidates.logprobs.tolist()
    mask_rows = candidates.valid_mask.tolist()
    width, k = len(scores[0]), candidates.token_ids.shape[1]
    requests = []
    for request, parent_scores in enumerate(scores):
        ranked = []
        for parent, score in enumerate(parent_scores):
            row = request * width + parent
            for col in range(k):
                if mask_rows[row][col]:
                    ranked.append((score + score_rows[row][col], parent, token_rows[row][col]))
        assert len({item[0] for item in ranked}) == len(ranked), "fixture must avoid score ties"
        ranked.sort(key=lambda item: item[0], reverse=True)
        winners = ranked[:output_width]
        if not last:
            winners.sort(key=lambda item: item[1])
        requests.append(winners)
    return requests


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "widths,ks",
    [
        ((4, 4, 4), (4, 3, 1)),
        ((2, 4, 3), (2, 5, 1)),
        ((4, 3, 2), (4, 1, 3)),
    ],
)
def test_real_npu_independent_k_selection_and_kv_match_scalar_oracle(
    batch: int, packed: bool, widths: tuple[int, ...], ks: tuple[int, ...]
) -> None:
    context = _context(batch, packed)
    state = DynamicBeamState(context, batch, 3)
    prefill_tokens = [
        [1000 * request + beam for beam in range(widths[0])] for request in range(batch)
    ]
    scores = [[-(beam + 1) / 8 for beam in range(widths[0])] for _ in range(batch)]
    prefill = ConstraintOutput(
        torch.tensor(prefill_tokens, dtype=torch.int32, device=DEVICE),
        torch.tensor(scores, dtype=torch.float32, device=DEVICE),
        torch.ones(batch, ks[0], dtype=torch.bool, device=DEVICE),
    )
    state.initialize(prefill, widths[0])
    sequences = [[[token, 0, 0] for token in row] for row in prefill_tokens]
    histories = [[[0, 0, 0] for _ in row] for row in prefill_tokens]
    for stage in (1, 2):
        input_width, output_width = widths[stage - 1], widths[stage]
        last = stage == 2
        host_candidates = _candidates(batch, input_width, ks[stage], stage)
        candidates = ConstraintOutput(
            *(tensor.to(DEVICE) for tensor in _output_tensors(host_candidates))
        )
        old_pools = get_stage_pools(context, input_width)
        target_pools = get_stage_pools(context, output_width)
        for pool in target_pools:
            pool.unshared_key[batch:].fill_(-123)
            pool.unshared_value[batch:].fill_(-123)
        before = []
        for layer, pool in enumerate(old_pools):
            copied = []
            for kind, name in enumerate(("unshared_key", "unshared_value")):
                tensor = getattr(pool, name)
                for request in range(batch):
                    for parent in range(input_width):
                        tensor[request, parent].fill_(
                            300 * request + 50 * layer + 20 * kind + parent
                        )
                copied.append(tensor.cpu().clone())
            before.append(copied)
        target_before = [
            (pool.unshared_key.cpu().clone(), pool.unshared_value.cpu().clone())
            for pool in target_pools
        ]
        winners = _scalar_winners(scores, host_candidates, output_width, last)
        next_sequences, next_histories = [], []
        for request, row in enumerate(winners):
            request_sequences, request_histories = [], []
            for _, parent, token in row:
                sequence = sequences[request][parent].copy()
                history = histories[request][parent].copy()
                sequence[stage], history[stage] = token, parent
                request_sequences.append(sequence)
                request_histories.append(history)
            next_sequences.append(request_sequences)
            next_histories.append(request_histories)
        sequences, histories = next_sequences, next_histories
        scores = [[score for score, _, _ in row] for row in winners]
        returned = min(output_width, 2) if last else output_width
        result = state.select(
            candidates, input_width, output_width, stage, is_last=last, result_width=returned
        )
        torch.npu.synchronize()
        assert state.scores is not None and state.sequence is not None
        assert state.parent_history is not None
        torch.testing.assert_close(state.scores.cpu(), torch.tensor(scores), rtol=0, atol=0)
        assert state.sequence.cpu().tolist() == sequences
        assert state.parent_history.cpu().tolist() == histories
        assert result["out_token_ids"].cpu().tolist() == [
            [token for _, _, token in row[:returned]] for row in winners
        ]
        assert result["out_sequence"].cpu().tolist() == [
            [sequence[: stage + 1] for sequence in row[:returned]] for row in sequences
        ]
        assert result["local_parent_ids"].cpu().tolist() == [
            [parent for _, parent, _ in row[:returned]] for row in winners
        ]
        assert result["out_token_index"].cpu().tolist() == [
            [request * input_width + parent for _, parent, _ in row[:returned]]
            for request, row in enumerate(winners)
        ]
        if not last:
            if ks[stage] > 1:
                assert any(len({parent for _, parent, _ in row}) < len(row) for row in winners)
            for layer, pool in enumerate(target_pools):
                for kind, name in enumerate(("unshared_key", "unshared_value")):
                    actual = getattr(pool, name).cpu()
                    for request, row in enumerate(winners):
                        for child, (_, parent, _) in enumerate(row):
                            torch.testing.assert_close(
                                actual[request, child, :, :stage],
                                before[layer][kind][request, parent, :, :stage],
                                rtol=0,
                                atol=0,
                            )
                    assert (actual[batch:] == -123).all()
        else:
            for pools, snapshots in ((old_pools, before), (target_pools, target_before)):
                for pool, saved in zip(pools, snapshots):
                    for name, expected in zip(("unshared_key", "unshared_value"), saved):
                        torch.testing.assert_close(
                            getattr(pool, name).cpu(), expected, rtol=0, atol=0
                        )
