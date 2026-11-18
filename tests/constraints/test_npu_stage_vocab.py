# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"

"""Host regressions for stage-local tables and the real NPU select boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from tests.constraints.test_table import make_table
from vllm_gr.config import CONSTRAINT_TABLE_FORMAT, ConstraintTableBackendEnum
from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintBackendInfo,
    ConstraintMetadata,
)
from vllm_gr.constraints.npu_backend import NPUConstraintBackend
from vllm_gr.constraints.npu_stage_vocab import NPUStageVocab, build_npu_stage_vocab
from vllm_gr.constraints.table import (
    CONSTRAINT_TABLE_TENSOR_NAMES,
    ConstraintTable,
    ConstraintTableMetadata,
    validate_constraint_table,
)


def _upload(array: NDArray[Any], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.copy()).to(device)


def _sparse_table() -> ConstraintTable:
    """Keep all three stage ID domains sparse, distinct, and non-contiguous."""
    offsets = np.zeros(513, dtype=np.int32)
    offsets[12:] += 2
    offsets[238:] += 1
    arrays = {
        "first_token_ids": np.array([11, 237], dtype=np.int32),
        "prefix1_offsets": offsets,
        "prefix1_values": np.array([7, 137, 399], dtype=np.int32),
        "prefix1_pair_keys": np.array(
            [11 * 512 + 7, 11 * 512 + 137, 237 * 512 + 399], dtype=np.int64
        ),
        "prefix2_value_offsets": np.array([0, 2, 3, 5], dtype=np.int32),
        "prefix2_values": np.array([0, 451, 53, 53, 211], dtype=np.int32),
    }
    table = make_table(
        **arrays,
        metadata=ConstraintTableMetadata(
            format=CONSTRAINT_TABLE_FORMAT,
            artifact_digest=None,
            tokenizer_digest="sha256:" + "b" * 64,
            vocab_size=512,
            max_prefix1_degree=2,
            max_prefix2_degree=2,
            total_bytes=sum(array.nbytes for array in arrays.values()),
        ),
    )
    validate_constraint_table(table)
    return table


def _vocab(table: ConstraintTable) -> NPUStageVocab:
    vocab = build_npu_stage_vocab(table, torch.device("cpu"), _upload)
    assert vocab is not None
    return vocab


def _candidate_set_op(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    first: torch.Tensor,
    offsets1: torch.Tensor,
    values1: torch.Tensor,
    pair_keys: torch.Tensor,
    offsets2: torch.Tensor,
    values2: torch.Tensor,
    temperatures: torch.Tensor,
    step: int,
    top_k: int,
    max_prefix1_degree: int,
    max_prefix2_degree: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small CPU stand-in for the fused ABI, including its legal-set scoring.

    The portable CPU operator uses full-window scoring; that domain cannot
    serve as an oracle for the established NPU candidate-set normalization.
    Only the native call is replaced: production select still validates and
    dispatches the tables, maps prefixes, derives validity, and restores IDs.
    """
    assert logits.dtype == torch.float32  # Includes production's BF16 widening.
    rows, width = logits.shape
    prefixes = sequence_group.reshape(rows, 3)
    tokens = torch.zeros((rows, top_k), dtype=torch.int32)
    logprobs = torch.full((rows, top_k), INVALID_LOGIT, dtype=torch.float32)
    for row in range(rows):
        if step == 0:
            candidates = first
        else:
            a = int(prefixes[row, 0])
            # The native ABI reserves the last slot for an invalid first prefix.
            if not 0 <= a < width - 1:
                continue
            if step == 1:
                candidates = values1[int(offsets1[a]) : int(offsets1[a + 1])]
            else:
                b = int(prefixes[row, 1])
                if not 0 <= b < width:
                    continue
                positions = torch.nonzero(pair_keys == a * width + b).flatten()
                if positions.numel() == 0:
                    continue
                position = int(positions[0])
                candidates = values2[int(offsets2[position]) : int(offsets2[position + 1])]
        if candidates.numel() == 0:
            continue
        temperature = float(temperatures[0 if temperatures.numel() == 1 else row])
        scores = logits[row, candidates.long()] / (temperature or 1.0)
        count = min(top_k, candidates.numel())
        order = torch.argsort(scores, descending=True, stable=True)[:count]
        tokens[row, :count] = candidates[order].to(torch.int32)
        logprobs[row, :count] = scores[order] - torch.logsumexp(scores, dim=0)
    return tokens, logprobs


def _backend(
    table: ConstraintTable,
    vocab: NPUStageVocab,
    op: Callable[..., tuple[torch.Tensor, torch.Tensor]] = _candidate_set_op,
) -> NPUConstraintBackend:
    tables = {
        name: _upload(getattr(table, name), torch.device("cpu"))
        for name in CONSTRAINT_TABLE_TENSOR_NAMES
    }
    backend = NPUConstraintBackend(
        **tables,
        info=ConstraintBackendInfo(
            table_metadata=table.metadata,
            backend=ConstraintTableBackendEnum.NPU,
            max_top_k=3,
            max_rows=16,
            host_memory_bytes=0,
            device_memory_bytes=table.total_bytes,
        ),
        op=op,
        stage_vocab=vocab,
    )
    backend._mark_ready()  # noqa: SLF001
    return backend


def test_compaction_preserves_ordered_relations_and_immutable_global_table() -> None:
    table = _sparse_table()
    originals = {name: getattr(table, name).copy() for name in CONSTRAINT_TABLE_TENSOR_NAMES}
    vocab = _vocab(table)

    assert vocab.vocab_size == 128
    assert [ids.tolist() for ids in vocab.stage_token_ids] == [
        [11, 237],
        [7, 137, 399],
        [0, 53, 211, 451],
    ]
    first, offsets, second, keys, offsets2, third = vocab.tables
    assert first.tolist() == [0, 1]
    assert offsets[:3].tolist() == [0, 2, 3]
    assert offsets[3:].tolist() == [3] * 126
    assert second.tolist() == [0, 1, 2]
    assert keys.tolist() == [0, 1, 130]
    assert offsets2.tolist() == [0, 2, 3, 5]
    assert third.tolist() == [0, 3, 1, 1, 2]
    # Every inactive SA slot, including the kernel's reserved last slot, is empty.
    assert int(offsets[-2]) == int(offsets[-1])
    for name, expected in originals.items():
        np.testing.assert_array_equal(getattr(table, name), expected)
        assert not getattr(table, name).flags.writeable
    torch.testing.assert_close(
        vocab.stage_token_ids[1][second.long()],
        torch.from_numpy(table.prefix1_values.copy()).long(),
    )
    torch.testing.assert_close(
        vocab.stage_token_ids[2][third.long()], torch.from_numpy(table.prefix2_values.copy()).long()
    )
    owned = vocab.owned_tensors()
    assert len(owned) == 11
    assert all(tensor.device.type == "cpu" and tensor.is_contiguous() for tensor in owned.values())


@pytest.mark.parametrize(("count", "expected_width"), [(126, 128), (127, 128), (128, 256)])
def test_alignment_always_leaves_a_reserved_prefix_slot(count: int, expected_width: int) -> None:
    # A dense first-stage domain forces the alignment boundary; SB and SC stay small.
    offsets = np.arange(513, dtype=np.int32).clip(max=count)
    first = np.arange(count, dtype=np.int32)
    values = np.ones(count, dtype=np.int32)
    table = make_table(
        first_token_ids=first,
        prefix1_offsets=offsets,
        prefix1_values=values,
        prefix1_pair_keys=first.astype(np.int64) * 512 + 1,
        prefix2_value_offsets=np.arange(count + 1, dtype=np.int32),
        prefix2_values=values,
    )
    table = replace(
        table,
        metadata=replace(
            table.metadata, vocab_size=512, max_prefix1_degree=1, max_prefix2_degree=1
        ),
    )
    validate_constraint_table(table)
    vocab = _vocab(table)
    assert vocab.vocab_size == expected_width
    assert int(vocab.tables[0][-1]) < vocab.vocab_size - 1
    assert int(vocab.tables[1][-2]) == int(vocab.tables[1][-1])


@pytest.mark.parametrize("vocab_size", [6, 128])
def test_no_savings_returns_none_without_uploading_optional_tensors(vocab_size: int) -> None:
    def unexpected_upload(array: NDArray[Any], device: torch.device) -> torch.Tensor:
        pytest.fail("a non-beneficial compact domain must not allocate optional tensors")

    original = make_table()
    table = make_table(
        prefix1_offsets=np.pad(original.prefix1_offsets, (0, vocab_size - 6), constant_values=3),
        prefix1_pair_keys=np.array(
            [vocab_size + 2, vocab_size + 4, 3 * vocab_size + 5], dtype=np.int64
        ),
    )
    table = replace(table, metadata=replace(table.metadata, vocab_size=vocab_size))
    validate_constraint_table(table)
    assert build_npu_stage_vocab(table, torch.device("cpu"), unexpected_upload) is None


@pytest.mark.parametrize("step", [0, 1, 2])
def test_prefix_mapping_uses_reserved_padding_for_absent_or_out_of_range_ids(step: int) -> None:
    vocab = _vocab(_sparse_table())
    prefixes = torch.tensor(
        [
            [11, 7, 123],
            [237, 399, 123],
            [7, 137, 123],
            [-1, 512, 123],
            [512, -1, 123],
            [11, 11, 123],
        ],
        dtype=torch.int32,
    )
    before = prefixes.clone()
    expected = torch.zeros_like(prefixes)
    if step >= 1:
        expected[:, 0] = torch.tensor([0, 1, 127, 127, 127, 0])
    if step >= 2:
        expected[:, 1] = torch.tensor([0, 2, 1, 127, 127, 127])
    torch.testing.assert_close(vocab.local_prefixes(prefixes, step), expected, rtol=0, atol=0)
    torch.testing.assert_close(prefixes, before, rtol=0, atol=0)


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_k", [1, 3])
@pytest.mark.parametrize("shared_temperature", [False, True])
def test_select_preserves_global_ids_and_logprobs_for_all_stages(
    step: int,
    dtype: torch.dtype,
    top_k: int,
    shared_temperature: bool,
) -> None:
    table = _sparse_table()
    vocab = _vocab(table)
    backend = _backend(table, vocab)
    prefixes = torch.tensor(
        [
            [11, 7, 0],
            [11, 137, 0],
            [237, 399, 0],
            [237, 7, 0],
            [11, 399, 0],
            [7, 137, 0],
            [-1, 519, 0],
            [512, 7, 0],
            [11, -1, 0],
        ],
        dtype=torch.int32,
    )
    full = torch.linspace(-3, 7, 512).repeat(prefixes.shape[0], 1).to(dtype)
    compact = torch.full((prefixes.shape[0], vocab.vocab_size), 1000.0, dtype=dtype)
    stage_ids = vocab.stage_token_ids[step]
    compact[:, : stage_ids.numel()] = full[:, stage_ids]
    temperatures = torch.tensor([0.0] if shared_temperature else [0, 0.5, 1, 2, 0, 0.5, 1, 2, 1])
    metadata = ConstraintMetadata(prefixes, temperatures, step, top_k)

    try:
        full_result = backend.select(full, metadata)
        compact_result = backend.select(compact, metadata)
        torch.testing.assert_close(compact_result.token_ids, full_result.token_ids, rtol=0, atol=0)
        torch.testing.assert_close(compact_result.logprobs, full_result.logprobs, rtol=0, atol=0)
        torch.testing.assert_close(
            compact_result.valid_mask, full_result.valid_mask, rtol=0, atol=0
        )
        expected_first_tokens = (
            [237] * 9
            if step == 0
            else [137, 137, 399, 399, 137, 0, 0, 0, 137]
            if step == 1
            else [451, 53, 211, 0, 0, 0, 0, 0, 0]
        )
        assert compact_result.token_ids[:, 0].tolist() == expected_first_tokens
        counts = (
            [2] * 9
            if step == 0
            else ([2, 2, 1, 1, 2, 0, 0, 0, 2] if step == 1 else [2, 1, 2, 0, 0, 0, 0, 0, 0])
        )
        assert compact_result.valid_mask.sum(dim=1).tolist() == [
            min(top_k, count) for count in counts
        ]
        assert (compact_result.token_ids[~compact_result.valid_mask] == 0).all()
        assert (compact_result.logprobs[~compact_result.valid_mask] == INVALID_LOGIT).all()
    finally:
        backend.close()


@pytest.mark.parametrize("step", [0, 1, 2])
def test_select_restores_minus_one_padding_without_remapping_it_to_a_valid_token(step: int) -> None:
    table = _sparse_table()
    vocab = _vocab(table)

    def padded_op(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, scores = _candidate_set_op(*args)
        return tokens.masked_fill(scores == INVALID_LOGIT, -1), scores

    backend = _backend(table, vocab, padded_op)
    metadata = ConstraintMetadata(
        torch.tensor([[11, 7, 0]], dtype=torch.int32),
        torch.ones(1),
        step,
        3,
    )
    try:
        output = backend.select(torch.zeros(1, vocab.vocab_size), metadata)
        assert output.valid_mask.tolist() == [[True, True, False]]
        expected = [[11, 237, 0], [7, 137, 0], [0, 451, 0]][step]
        assert output.token_ids.tolist() == [expected]
        assert output.logprobs[0, 2] == INVALID_LOGIT
        if step == 2:
            # Global token zero is a legal SC ID; the mask distinguishes it from padding.
            assert output.token_ids[0, 0] == output.token_ids[0, 2] == 0
            assert output.valid_mask[0, 0] and not output.valid_mask[0, 2]
    finally:
        backend.close()


pytestmark = pytest.mark.cpu_test
