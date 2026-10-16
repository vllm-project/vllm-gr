# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-vocabulary Beam semantics independent of native fused Beam operators."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.worker.full_vocab import full_vocab_topk
from vllm_gr.v1.worker.torch_beam_state import TorchBeamState, reorder_beam_kv


def _state(width: int = 2, *, steps: int = 3, prompts: tuple[int, ...] = (7,)) -> TorchBeamState:
    return TorchBeamState(
        beam_width=width,
        max_steps=steps,
        prompt_lengths=prompts,
        stop_token_ids=(0,),
        eos_token_id=0,
        length_penalty=1.0,
        device="cpu",
    )


def _advance(state: TorchBeamState, ids: list[list[int]], scores: list[list[float]]) -> None:
    values = torch.tensor(scores, dtype=torch.float32)
    state.advance(
        torch.tensor(ids, dtype=torch.int32),
        values,
        torch.isfinite(values),
        stage_index=state.stage_index + 1,
    )


def test_eos_removes_parent_without_discarding_lower_ranked_live_candidates() -> None:
    state = _state()
    _advance(state, [[0, 11, 12, 13]], [[-0.1, -0.2, -0.3, -0.4]])
    assert state.token_ids.tolist() == [[11, 12]]
    assert state.completed_lengths.tolist() == [[1, 0]]
    _advance(
        state,
        [[0, 21, 22, 23], [31, 0, 32, 33]],
        [[-0.5, -0.6, -0.7, -0.8], [-0.01, -0.02, -0.03, -0.04]],
    )
    # Both live winners originate in the second parent; the EOS from Prefill
    # has never been decoded, even though it remains the best final result.
    assert state.local_parents.tolist() == [[1, 1]]
    assert state.sequences[0, :, :2].tolist() == [[12, 31], [12, 32]]
    assert state.parents[0, :, :2].tolist() == [[0, 1], [0, 1]]
    output = state.finalize()
    assert output.sequences[0, 0, 0].item() == 0
    assert output.lengths[0, 0].item() == 1
    assert output.scores[0, 0].item() == pytest.approx(-0.1)


def test_completed_request_stays_completed_while_other_batch_rows_continue() -> None:
    state = _state(width=1, prompts=(3, 9))
    _advance(state, [[0, 1], [2, 0]], [[-0.2, -math.inf], [-0.1, -3.0]])
    assert state.valid.tolist() == [[False], [True]]
    # Bogus padded-row logits must never create descendants of the finished item.
    _advance(state, [[7, 8], [3, 0]], [[100.0, 99.0], [-0.1, -3.0]])
    _advance(state, [[7, 8], [4, 0]], [[100.0, 99.0], [-0.1, -3.0]])
    result = state.finalize()
    assert result.sequences[:, 0].tolist() == [[0, 0, 0], [2, 3, 4]]
    assert result.lengths.tolist() == [[1], [3]]
    assert result.stopped.tolist() == [[True], [False]]
    torch.testing.assert_close(result.scores, torch.tensor([[-0.2], [-0.3]]))


def test_final_order_uses_prompt_length_and_excludes_terminal_eos() -> None:
    state = _state(width=1, steps=2, prompts=(1,))
    _advance(state, [[0, 10]], [[-1.0, -1.1]])
    _advance(state, [[20, 0]], [[-0.1, -3.0]])
    result = state.finalize()
    # EOS: -1 / 1; live: -1.2 / 3. Ranking by raw cumulative score, or by
    # generated length with EOS included, chooses the wrong result.
    assert result.sequences.tolist() == [[[10, 20]]]
    assert result.stopped.tolist() == [[False]]
    assert result.scores.item() == pytest.approx(-1.2)


def test_equal_scores_keep_earlier_completions_before_later_candidates() -> None:
    state = _state(width=2, steps=2, prompts=(2,))
    _advance(state, [[0, 10, 11, 12]], [[-0.5, -0.75, -1.5, -2.0]])
    _advance(
        state,
        [[0, 20, 21, 22], [0, 30, 31, 32]],
        [[0.0, -0.25, -3.0, -4.0], [-1.0, -2.0, -3.0, -4.0]],
    )
    result = state.finalize()
    # All three leading normalized scores are -0.25: Prefill EOS, Decode
    # EOS, then live. Native stable completed-first final ordering wins.
    assert result.lengths.tolist() == [[1, 2]]
    assert result.sequences[0, :, :2].tolist() == [[0, 0], [10, 0]]


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_matches_independent_native_beam_enumeration(width: int, steps: int, batch: int) -> None:
    prompt_lengths = tuple(3 + row * 17 for row in range(batch))
    state = _state(width, steps=steps, prompts=prompt_lengths)
    active = [[((), 0.0, ())] for _ in range(batch)]
    completed: list[list[tuple[tuple[int, ...], float, tuple[int, ...]]]] = [
        [] for _ in range(batch)
    ]

    def logits(row: int, prefix: tuple[int, ...]) -> torch.Tensor:
        seed = 71 + row * 173 + sum((index + 1) * token * 31 for index, token in enumerate(prefix))
        return torch.randn(19, generator=torch.Generator().manual_seed(seed))

    for stage in range(steps):
        physical_logits = []
        for row in range(batch):
            row_prefixes = [entry[0] for entry in active[row]]
            physical_logits.extend(logits(row, prefix) for prefix in row_prefixes)
            if stage:
                physical_logits.extend(logits(row, ()) for _ in range(width - len(row_prefixes)))
        candidates = full_vocab_topk(torch.stack(physical_logits), 2 * width)
        state.advance(
            candidates.token_ids, candidates.logprobs, candidates.valid_mask, stage_index=stage
        )
        for row in range(batch):
            expanded = []
            for parent, (prefix, score, history) in enumerate(active[row]):
                logprobs = logits(row, prefix).log_softmax(0).tolist()
                ranked = sorted(range(len(logprobs)), key=lambda token: -logprobs[token])[
                    : 2 * width
                ]
                for token in ranked:
                    entry = (prefix + (token,), score + logprobs[token], history + (parent,))
                    if token == 0:
                        completed[row].append(entry)
                    else:
                        expanded.append(entry)
            active[row] = sorted(expanded, key=lambda entry: -entry[1])[:width]

    result = state.finalize()
    for row in range(batch):
        reference = sorted(
            completed[row] + active[row],
            key=lambda entry: (
                -entry[1] / (prompt_lengths[row] + len(entry[0]) - (entry[0][-1] == 0))
            ),
        )[:width]
        for index, (tokens, score, parents) in enumerate(reference):
            length = result.lengths[row, index].item()
            assert tuple(result.sequences[row, index, :length].tolist()) == tokens
            assert tuple(result.parents[row, index, :length].tolist()) == parents
            assert result.stopped[row, index].item() == (tokens[-1] == 0)
            assert result.scores[row, index].item() == pytest.approx(score, abs=2e-6)


def test_kv_reorder_handles_duplicate_unsorted_parents_without_crossing_requests() -> None:
    source = torch.arange(2 * 3 * 2 * 3 * 2).reshape(2, 3, 2, 3, 2)
    pools = [SimpleNamespace(unshared_key=source.clone(), unshared_value=source.clone() + 1000)]
    parents = torch.tensor([[2, 0, 2], [1, 1, 0]])
    addresses = (pools[0].unshared_key.data_ptr(), pools[0].unshared_value.data_ptr())
    reorder_beam_kv(pools, parents, decode_step=1)
    for row in range(2):
        torch.testing.assert_close(
            pools[0].unshared_key[row, :, :, :2], source[row, parents[row], :, :2]
        )
        torch.testing.assert_close(
            pools[0].unshared_value[row, :, :, :2], source[row, parents[row], :, :2] + 1000
        )
    torch.testing.assert_close(pools[0].unshared_key[:, :, :, 2], source[:, :, :, 2])
    assert addresses == (pools[0].unshared_key.data_ptr(), pools[0].unshared_value.data_ptr())


def test_selection_and_reorder_do_not_read_device_values_on_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("device values must not control host execution")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    _advance(state, [[0, 1, 2, 3]], [[-0.1, -0.2, -0.3, -0.4]])
    _advance(state, [[0, 4, 5, 6], [7, 0, 8, 9]], [[-0.1] * 4, [-0.2] * 4])
    state.finalize()
    pool = SimpleNamespace(
        unshared_key=torch.zeros(1, 2, 1, 3, 2), unshared_value=torch.zeros(1, 2, 1, 3, 2)
    )
    reorder_beam_kv([pool], state.local_parents, decode_step=0)
