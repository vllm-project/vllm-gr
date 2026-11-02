# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"
"""Catalog-free SID family selection and tokenizer map regressions."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from vllm_gr.ops.npu.sid_stage_vocab import (
    NPUSIDStageVocab,
    build_sid_stage_vocab,
    select_sid_stage_candidates,
)
from vllm_gr.ops.npu.unconstrained_topk import select_unconstrained_candidates
from vllm_gr.v1.worker.npu_beam_stage_runner import NPUEagerBeamStageRunner


def _tokenizer(vocabulary: Any) -> Any:
    return SimpleNamespace(get_vocab=lambda: vocabulary)


@pytest.fixture
def stage_vocab() -> NPUSIDStageVocab:
    # Name order, numeric SID order, and global tokenizer-ID order differ.
    return build_sid_stage_vocab(
        _tokenizer(
            {
                "<s_a_0>": 9,
                "<s_a_1>": 2,
                "<s_a_4>": 6,
                "<s_b_5>": 11,
                "<s_c_0>": 3,
                "<s_c_3>": 15,
                "ordinary": 0,
            }
        ),
        19,
        "cpu",
    )


@pytest.mark.parametrize("counts", [(2, 2, 2), (3, 1, 2)])
def test_tokenizer_maps_accept_unequal_sizes_and_noncontiguous_ids(
    counts: tuple[int, int, int],
) -> None:
    ids_by_stage = ((17, 2, 7), (11, 4, 19), (3, 13, 23))
    entries = {
        f"<s_{family}_{number}>": token_id
        for family, count, candidates in zip("abc", counts, ids_by_stage, strict=True)
        for number, token_id in enumerate(candidates[:count])
    }
    vocab = build_sid_stage_vocab(_tokenizer(entries), 31, "cpu")
    assert vocab.full_vocab_size == 31 and vocab.vocab_size == max(counts)
    assert vocab.stage_token_counts == counts
    assert vocab.device_memory_bytes == 8 * sum(counts)
    for actual, candidates, count in zip(vocab.stage_token_ids, ids_by_stage, counts, strict=True):
        assert actual.dtype == torch.int64 and actual.device.type == "cpu"
        assert actual.tolist() == sorted(candidates[:count])


def test_sid_matching_is_exact_and_uses_tokenizer_ids() -> None:
    entries = {"<s_a_1>": 9, "<s_b_1>": 3, "<s_c_1>": 7}
    for malformed in (
        "prefix<s_a_2>",
        "<s_b_2>suffix",
        "<s_c_2>\n",
        "<s_A_2>",
        "<s_a_-1>",
        "<s_b_1.0>",
        "<s_c_١>",
        "<|s_a_2|>",
    ):
        entries[malformed] = 999
    vocab = build_sid_stage_vocab(_tokenizer(entries), 11, "cpu")
    assert [ids.tolist() for ids in vocab.stage_token_ids] == [[9], [3], [7]]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"<s_a_1>": 3}, "share ID"),
        ({"<s_a_2>": 9}, "share ID"),
        ({"<s_a_01>": 8}, "duplicate numeric index"),
        ({"<s_a_2>": -1}, "invalid model vocabulary ID"),
        ({"<s_a_2>": 11}, "invalid model vocabulary ID"),
        ({"<s_a_2>": True}, "invalid model vocabulary ID"),
        ({"<s_a_2>": 2.0}, "invalid model vocabulary ID"),
    ],
)
def test_invalid_sid_maps_fail_before_device_allocation(
    mutation: dict[str, Any], match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = {"<s_a_1>": 9, "<s_b_1>": 3, "<s_c_1>": 7}
    entries.update(mutation)
    with monkeypatch.context() as guard:
        guard.setattr(torch, "tensor", lambda *_a, **_kw: pytest.fail("allocated invalid SID map"))
        with pytest.raises(ValueError, match=match):
            build_sid_stage_vocab(_tokenizer(entries), 11, "cpu")


@pytest.mark.parametrize("missing", ["a", "b", "c"])
def test_all_three_sid_families_are_required(missing: str) -> None:
    entries = {f"<s_{family}_0>": index for index, family in enumerate("abc") if family != missing}
    with pytest.raises(ValueError, match="no tokens for families"):
        build_sid_stage_vocab(_tokenizer(entries), 8, "cpu")


@pytest.mark.parametrize("size", [0, -1, True, 3.0])
def test_model_vocabulary_size_must_be_a_positive_integer(size: Any) -> None:
    with pytest.raises(ValueError, match="positive model vocab_size"):
        build_sid_stage_vocab(_tokenizer({}), size, "cpu")


@pytest.mark.parametrize("tokenizer", [object(), _tokenizer([]), _tokenizer({1: 0})])
def test_invalid_tokenizer_interfaces_are_rejected(tokenizer: Any) -> None:
    with pytest.raises(TypeError):
        build_sid_stage_vocab(tokenizer, 8, "cpu")


def test_stage_topk_uses_global_id_ties_and_all_stage_tokens_in_denominator(
    stage_vocab: NPUSIDStageVocab,
) -> None:
    logits = torch.full((1, 19), 1000.0)
    logits[:, [2, 6, 9]] = torch.tensor([1.0, 2.0, 2.0])
    temperatures = torch.ones(1)
    narrow = select_sid_stage_candidates(logits, temperatures, 1, 0, stage_vocab)
    wide = select_sid_stage_candidates(logits, temperatures, 5, 0, stage_vocab)
    assert narrow.token_ids.tolist() == [[6]]
    assert wide.token_ids.tolist() == [[6, 9, 2, 0, 0]]
    expected = -math.log(2 + math.exp(-1))
    torch.testing.assert_close(narrow.logprobs, torch.tensor([[expected]]))
    torch.testing.assert_close(narrow.logprobs[:, 0], wide.logprobs[:, 0], rtol=0, atol=0)
    assert wide.valid_mask.tolist() == [[True, True, True, False, False]]
    assert torch.isneginf(wide.logprobs[:, 3:]).all()


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("top_k", [1, 3, 8])
def test_compact_and_full_head_stage_mask_are_equivalent_and_padding_is_excluded(
    stage_vocab: NPUSIDStageVocab,
    stage: int,
    top_k: int,
) -> None:
    generator = torch.Generator().manual_seed(476)
    hidden, weight = torch.randn(4, 5, generator=generator), torch.randn(19, 5, generator=generator)
    full = F.linear(hidden, weight)
    ids = stage_vocab.stage_token_ids[stage]
    compact = torch.full((4, stage_vocab.vocab_size), 99999.0)
    compact[:, : ids.numel()] = F.linear(hidden, weight.index_select(0, ids))
    temperature = torch.tensor([0.0, 0.5, 1.0, 2.0])
    baseline = select_sid_stage_candidates(full, temperature, top_k, stage, stage_vocab)
    actual = select_sid_stage_candidates(
        compact, temperature, top_k, stage, stage_vocab, compact_logits=True
    )
    torch.testing.assert_close(actual.token_ids, baseline.token_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual.logprobs, baseline.logprobs, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual.valid_mask, baseline.valid_mask)
    if top_k >= ids.numel():
        torch.testing.assert_close(actual.logprobs.exp().sum(1), torch.ones(4))


def test_stage_normalization_is_explicitly_different_from_full_vocabulary(
    stage_vocab: NPUSIDStageVocab,
) -> None:
    # SB has only token 11, so its stage probability is one. Generic Beam
    # normalizes across all 19 equal logits and assigns probability 1/19.
    full = torch.zeros(1, 19)
    stage = select_sid_stage_candidates(full, torch.ones(1), 19, 1, stage_vocab)
    generic = select_unconstrained_candidates(full, torch.ones(1), 19)
    assert stage.token_ids[0, 0] == 11 and stage.logprobs[0, 0] == 0
    assert stage.valid_mask.sum() == 1
    torch.testing.assert_close(generic.logprobs, torch.full((1, 19), -math.log(19)))


@pytest.mark.parametrize("compact", [False, True])
def test_nonfinite_stage_logits_and_missing_candidates_are_explicit(
    stage_vocab: NPUSIDStageVocab,
    compact: bool,
) -> None:
    full = torch.ones(3, 19)
    full[:, [2, 6, 9]] = torch.tensor(
        [
            [float("nan"), float("inf"), -float("inf")],
            [float("nan"), 2.0, 1.0],
            [3.0, -float("inf"), float("inf")],
        ]
    )
    values = full[:, [2, 6, 9]] if compact else full
    result = select_sid_stage_candidates(
        values, torch.ones(1), 5, 0, stage_vocab, compact_logits=compact
    )
    assert result.token_ids.tolist() == [[0] * 5, [6, 9, 0, 0, 0], [2, 0, 0, 0, 0]]
    assert result.valid_mask.tolist() == [
        [False] * 5,
        [True, True, False, False, False],
        [True, False, False, False, False],
    ]
    assert torch.isneginf(result.logprobs[~result.valid_mask]).all()
    assert not torch.isnan(result.logprobs).any()
    assert result.logprobs[2, 0] == 0


def test_sid_selection_never_reads_device_values_on_host(
    stage_vocab: NPUSIDStageVocab, monkeypatch: pytest.MonkeyPatch
) -> None:
    logits = torch.zeros(2, 19)
    temperature = torch.ones(1)
    with monkeypatch.context() as guard:

        def forbidden(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("unexpected host tensor read")

        for name in ("cpu", "item", "tolist"):
            guard.setattr(torch.Tensor, name, forbidden)
        result = select_sid_stage_candidates(logits, temperature, 5, 0, stage_vocab)
    assert result.token_ids.tolist() == [[2, 6, 9, 0, 0]] * 2


@pytest.mark.parametrize("stage", [-1, 3, True, None])
def test_invalid_sid_stage_indices_fail(stage_vocab: NPUSIDStageVocab, stage: Any) -> None:
    with pytest.raises(ValueError, match="SID stage_index"):
        select_sid_stage_candidates(torch.zeros(1, 19), torch.ones(1), 2, stage, stage_vocab)


@pytest.mark.parametrize("shape,compact", [((1, 2), False), ((1, 19), True), ((19,), False)])
def test_unexpected_sid_logits_shapes_fail(
    stage_vocab: NPUSIDStageVocab, shape: tuple[int, ...], compact: bool
) -> None:
    with pytest.raises(ValueError):
        select_sid_stage_candidates(
            torch.zeros(shape), torch.ones(1), 2, 0, stage_vocab, compact_logits=compact
        )


def _runner(
    stage_vocab: NPUSIDStageVocab | None, *, enabled: bool = True, projection: Any = None
) -> Any:
    runner = object.__new__(NPUEagerBeamStageRunner)
    runner.worker = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(beam_sid_stage_vocab=enabled),
        _gr_npu_sid_stage_vocab=stage_vocab,
        _gr_npu_stage_projection=projection,
    )
    runner.pending_plan = SimpleNamespace(
        execution=SimpleNamespace(num_execution_rows=1, candidate_top_k=3, stage_index=1)
    )
    return runner


def test_runner_applies_sid_semantics_with_full_head_when_projection_is_disabled(
    stage_vocab: NPUSIDStageVocab,
) -> None:
    runner = _runner(stage_vocab)
    logits = torch.zeros(1, 19)
    logits[:, 0] = 1000  # Unrelated text token must never enter SID selection.
    result = runner._select_unconstrained_candidates(
        logits, SimpleNamespace(beam_params=SimpleNamespace(temperature=0.0))
    )
    assert result.token_ids.tolist() == [[11, 0, 0]]
    assert result.valid_mask.tolist() == [[True, False, False]]
    assert result.logprobs[0, 0] == 0


def test_runner_requires_loaded_sid_maps() -> None:
    with pytest.raises(RuntimeError, match="loaded tokenizer stage maps"):
        _runner(None)._select_unconstrained_candidates(
            torch.zeros(1, 19), SimpleNamespace(beam_params=SimpleNamespace(temperature=0.0))
        )


@pytest.mark.parametrize("projection", [None, SimpleNamespace(closed=True)])
def test_runner_rejects_narrow_logits_without_active_projection(
    stage_vocab: NPUSIDStageVocab, projection: Any
) -> None:
    with pytest.raises(RuntimeError, match="unsupported compact logits"):
        _runner(stage_vocab, projection=projection)._select_unconstrained_candidates(
            torch.zeros(1, 3), SimpleNamespace(beam_params=SimpleNamespace(temperature=0.0))
        )


def test_runner_accepts_known_compact_projection(stage_vocab: NPUSIDStageVocab) -> None:
    runner = _runner(stage_vocab, projection=SimpleNamespace(closed=False))
    result = runner._select_unconstrained_candidates(
        torch.tensor([[1.0, 9999.0, 9999.0]]),
        SimpleNamespace(beam_params=SimpleNamespace(temperature=0.0)),
    )
    assert result.token_ids.tolist() == [[11, 0, 0]]
    assert result.logprobs[0, 0] == 0


def test_sid_maps_do_not_change_generic_unconstrained_mode(stage_vocab: NPUSIDStageVocab) -> None:
    logits = torch.zeros(1, 19)
    logits[:, 0] = 1000
    result = _runner(stage_vocab, enabled=False)._select_unconstrained_candidates(
        logits, SimpleNamespace(beam_params=SimpleNamespace(temperature=0.0))
    )
    assert result.token_ids[0, 0] == 0 and result.valid_mask[0, 0]
