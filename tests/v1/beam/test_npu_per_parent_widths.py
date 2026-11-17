# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-call"
"""Host regressions for NPU independent candidate and survivor widths."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.constraints.backend import ConstraintOutput
from vllm_gr.v1.beam.beam_attention_pool import BeamAttentionPool
from vllm_gr.v1.beam.npu_dynamic_beam import (
    NPUDynamicBeamState,
    get_npu_stage_pools,
    initialize_npu_packed_stage_pools,
)
from vllm_gr.v1.engine.gr_async_scheduler import GR_STAGE_OUTPUT_ATTR
from vllm_gr.v1.worker.npu import npu_beam_stage_runner as npu_runner

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def _scheduled(per_parent: tuple[int, ...] | None) -> Any:
    metadata = SimpleNamespace(
        beam_params=SimpleNamespace(per_parent_beam_widths=per_parent),
        output_options=SimpleNamespace(num_return_sequences=4),
    )
    return SimpleNamespace(**{GR_STAGE_OUTPUT_ATTR: {"session": metadata}})


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("widths", [None, (4, 4, 4)])
@pytest.mark.parametrize("per_parent", [(4, 4, 1), (4, 4, 4)])
def test_explicit_k_uses_one_dynamic_owner_for_the_whole_session(
    stage: int,
    widths: tuple[int, ...] | None,
    per_parent: tuple[int, ...],
) -> None:
    """Later unequal K must choose dynamic state even while the current K equals W."""
    runner = object.__new__(npu_runner.NPUEagerBeamStageRunner)
    runner.worker = SimpleNamespace(_current_scheduler_output=_scheduled(per_parent))
    plan = SimpleNamespace(
        execution=SimpleNamespace(
            beam_width=4,
            dynamic_beam_widths=widths,
            per_parent_beam_widths=per_parent,
            stage_index=stage,
        )
    )
    assert runner._uses_dynamic_state(plan)


@pytest.mark.parametrize(
    "widths,returned,expected",
    [(None, 32, False), ((32, 32, 32), 32, False), ((32, 16, 8), 8, True), (None, 8, True)],
)
def test_default_k_keeps_existing_native_route(
    widths: tuple[int, ...] | None, returned: int, expected: bool
) -> None:
    runner = object.__new__(npu_runner.NPUEagerBeamStageRunner)
    scheduled = _scheduled(None)
    metadata = next(iter(getattr(scheduled, GR_STAGE_OUTPUT_ATTR).values()))
    metadata.output_options.num_return_sequences = returned
    runner.worker = SimpleNamespace(_current_scheduler_output=scheduled)
    plan = SimpleNamespace(
        execution=SimpleNamespace(
            beam_width=32, dynamic_beam_widths=widths, per_parent_beam_widths=None
        )
    )
    assert runner._uses_dynamic_state(plan) is expected


def _worker() -> Any:
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={
                "vllm_gr": {
                    "gr": {
                        "beam_graph_enabled": False,
                        "beam_width": [4, 4, 4],
                        "candidate_width": [4, 4, 4],
                        "max_decode_steps": 3,
                        "constraint_mode": "table",
                        "constraint_table": {
                            "backend": "npu",
                            "path": "/tmp/independent-k.safetensors",
                            "format": "constraint_table_v1",
                            "digests": {
                                "artifact": "sha256:" + "a" * 64,
                                "tokenizer": "sha256:" + "b" * 64,
                            },
                            "max_top_k": 8,
                        },
                    },
                }
            }
        )
    )


@pytest.mark.parametrize("missing", ["backend", "info", "capacity", "too_small"])
def test_explicit_k_rejects_unavailable_capacity_before_owner_creation(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    worker = _worker()
    if missing != "backend":
        worker.constraint_backend = SimpleNamespace()
        if missing != "info":
            worker.constraint_backend.info = SimpleNamespace()
            if missing != "capacity":
                # The bootstrap fits; the later K=8 must still be rejected.
                worker.constraint_backend.info.max_top_k = 4
    constructed = []

    def forbidden_constructor(worker: Any) -> Any:
        constructed.append(worker)
        raise AssertionError("capacity rejection must precede owner creation")

    monkeypatch.setattr(npu_runner, "NPUEagerBeamStageRunner", forbidden_constructor)
    with pytest.raises(ValueError, match="per_parent_beam_widths|max_top_k|capacity|backend"):
        npu_runner.get_npu_stage_runner(worker, _scheduled((4, 8, 1)))
    assert constructed == []
    assert not hasattr(worker, "_gr_npu_stage_runner")


@pytest.mark.parametrize("per_parent", [None, (4, 8, 1)])
def test_capacity_boundary_and_legacy_creation(
    monkeypatch: pytest.MonkeyPatch, per_parent: tuple[int, ...] | None
) -> None:
    worker = _worker()
    if per_parent is not None:
        worker.constraint_backend = SimpleNamespace(info=SimpleNamespace(max_top_k=8))
    created = []
    owner = SimpleNamespace(closed=False)

    def constructor(actual_worker: Any) -> Any:
        created.append(actual_worker)
        return owner

    monkeypatch.setattr(npu_runner, "NPUEagerBeamStageRunner", constructor)
    assert npu_runner.get_npu_stage_runner(worker, _scheduled(per_parent)) is owner
    assert worker._gr_npu_stage_runner is owner
    assert created == [worker]


def _context(batch: int, packed: bool) -> Any:
    context = SimpleNamespace(
        max_batch=batch + 1,
        max_beams=4,
        max_decode_steps=3,
        registry=SimpleNamespace(num_active_sessions=0),
        pools=[
            BeamAttentionPool(batch + 1, 4, 1, 3, 2, torch.device("cpu"), torch.float16)
            for _ in range(2)
        ],
    )
    if packed:
        initialize_npu_packed_stage_pools(context)
    return context


def _stage_candidates(batch: int, width: int, k: int, stage: int) -> ConstraintOutput:
    tokens = []
    scores = []
    for request in range(batch):
        preferred = 0 if request % 2 == 0 else width - 1
        for parent in range(width):
            tokens.append(
                [1000 * request + 100 * stage + 10 * parent + column for column in range(k)]
            )
            scores.append(
                [
                    -(column + 1) / 32 - stage / 1024
                    if parent == preferred
                    else -(8 + 2 * parent + column / 32 + stage / 1024)
                    for column in range(k)
                ]
            )
    return ConstraintOutput(
        token_ids=torch.tensor(tokens, dtype=torch.int32),
        logprobs=torch.tensor(scores, dtype=torch.float32),
        valid_mask=torch.ones(batch * width, k, dtype=torch.bool),
    )


def _scalar_select(
    sequences: list[list[list[int]]],
    scores: list[list[float]],
    histories: list[list[list[int]]],
    candidates: ConstraintOutput,
    output_width: int,
    stage: int,
    last: bool,
) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any]]:
    """Enumerate all parent/candidate pairs without production selection helpers."""
    input_width = len(scores[0])
    k = candidates.token_ids.shape[1]
    token_rows = candidates.token_ids.tolist()
    score_rows = candidates.logprobs.tolist()
    mask_rows = candidates.valid_mask.tolist()
    next_sequences, next_scores, next_histories, next_parents, next_tokens = [], [], [], [], []
    for request, parent_scores in enumerate(scores):
        ranked = []
        for parent, parent_score in enumerate(parent_scores):
            row = request * input_width + parent
            for column in range(k):
                if mask_rows[row][column]:
                    ranked.append(
                        (parent_score + score_rows[row][column], parent, token_rows[row][column])
                    )
        ranked.sort(key=lambda item: item[0], reverse=True)
        winners = ranked[:output_width]
        # These binary-fraction fixtures intentionally have no score ties.
        assert len({item[0] for item in ranked}) == len(ranked)
        if not last:
            winners.sort(key=lambda item: item[1])
        request_sequences, request_histories = [], []
        for _, parent, token in winners:
            sequence = sequences[request][parent].copy()
            history = histories[request][parent].copy()
            sequence[stage] = token
            history[stage] = parent
            request_sequences.append(sequence)
            request_histories.append(history)
        next_sequences.append(request_sequences)
        next_histories.append(request_histories)
        next_scores.append([item[0] for item in winners])
        next_parents.append([item[1] for item in winners])
        next_tokens.append([item[2] for item in winners])
    return next_sequences, next_scores, next_histories, next_parents, next_tokens


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "widths,ks", [((4, 4, 4), (4, 3, 1)), ((2, 4, 3), (2, 3, 1)), ((4, 3, 2), (4, 1, 3))]
)
def test_independent_k_selection_and_parent_kv_match_scalar_reference(
    batch: int, packed: bool, widths: tuple[int, ...], ks: tuple[int, ...]
) -> None:
    context = _context(batch, packed)
    state = NPUDynamicBeamState(context, batch, 3)
    prefill_tokens = [
        [1000 * request + beam for beam in range(widths[0])] for request in range(batch)
    ]
    prefill_scores = [[-(beam + 1) / 8 for beam in range(widths[0])] for _ in range(batch)]
    prefill = ConstraintOutput(
        torch.tensor(prefill_tokens, dtype=torch.int32),
        torch.tensor(prefill_scores, dtype=torch.float32),
        torch.ones(batch, ks[0], dtype=torch.bool),
    )
    state.initialize(prefill, widths[0])
    sequences = [[[token, 0, 0] for token in row] for row in prefill_tokens]
    scores = prefill_scores
    histories = [[[0, 0, 0] for _ in row] for row in prefill_tokens]
    for stage in (1, 2):
        input_width, output_width = widths[stage - 1], widths[stage]
        last = stage == 2
        candidates = _stage_candidates(batch, input_width, ks[stage], stage)
        old_pools = get_npu_stage_pools(context, input_width)
        target_pools = get_npu_stage_pools(context, output_width)
        for pool in target_pools:
            pool.unshared_key[batch:].fill_(-123)
            pool.unshared_value[batch:].fill_(-123)
        previous = []
        for layer, pool in enumerate(old_pools):
            layer_copies = []
            for kind, name in enumerate(("unshared_key", "unshared_value")):
                tensor = getattr(pool, name)
                for request in range(batch):
                    for parent in range(input_width):
                        value = 300 * request + 50 * layer + 20 * kind + parent
                        tensor[request, parent].fill_(value)
                layer_copies.append(tensor.clone())
            previous.append(layer_copies)
        sequences, scores, histories, parents, tokens = _scalar_select(
            sequences, scores, histories, candidates, output_width, stage, last
        )
        returned = min(output_width, 2) if last else output_width
        result = state.select(
            candidates,
            input_width,
            output_width,
            stage,
            is_last=last,
            result_width=returned,
        )
        assert state.scores is not None and state.sequence is not None
        torch.testing.assert_close(
            state.scores, torch.tensor(scores, dtype=torch.float32), rtol=0, atol=0
        )
        assert state.sequence.tolist() == sequences
        assert state.parent_history is not None
        assert state.parent_history.tolist() == histories
        assert result["out_token_ids"].tolist() == [row[:returned] for row in tokens]
        assert result["out_sequence"].tolist() == [
            [sequence[: stage + 1] for sequence in row[:returned]] for row in sequences
        ]
        assert result["local_parent_ids"].tolist() == [row[:returned] for row in parents]
        assert result["out_token_index"].tolist() == [
            [request * input_width + parent for parent in row[:returned]]
            for request, row in enumerate(parents)
        ]
        if not last:
            if ks[stage] > 1:
                assert any(len(set(row)) < len(row) for row in parents)
            for layer, pool in enumerate(target_pools):
                for kind, name in enumerate(("unshared_key", "unshared_value")):
                    actual = getattr(pool, name)
                    for request, parent_row in enumerate(parents):
                        for child, parent in enumerate(parent_row):
                            torch.testing.assert_close(
                                actual[request, child, :, :stage],
                                previous[layer][kind][request, parent, :, :stage],
                                rtol=0,
                                atol=0,
                            )
                    assert (actual[batch:] == -123).all()
        else:
            # The final stage must not move KV that no subsequent forward consumes.
            for layer, pool in enumerate(old_pools):
                for kind, name in enumerate(("unshared_key", "unshared_value")):
                    torch.testing.assert_close(
                        getattr(pool, name), previous[layer][kind], rtol=0, atol=0
                    )
