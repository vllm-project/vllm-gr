# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the platform-neutral Constraint Tables raw-logits hook (N3)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_gr.constraints.compiler import ConstraintTableCompiler
from vllm_gr.constraints.execution import TorchConstrainedTopKExecution
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.v1.worker.constraint_hook import (
    apply_constrained_sample_override,
    build_sequence_group_for_step,
    collect_constrained_rows,
    get_or_create_execution_adapter,
    patch_sampler_output_for_constrained,
    run_constrained_partition,
    validate_constraint_step,
)


@pytest.fixture  # type: ignore[misc]
def tokenizer() -> MagicMock:
    result = MagicMock()
    result.convert_tokens_to_ids.side_effect = lambda values: [int(value) for value in values]
    return result


def _decode_mdata(
    step: int, width: int = 2, suffix: list[list[int]] | None = None
) -> dict[str, Any]:
    """A constrained-decode metadata dict (scheduler_metadata shape)."""
    return {
        "is_beam_decode": True,
        "is_constrained": True,
        "constraint_step": step,
        "beam_width": width,
        "beam_suffix": [[i] for i in range(width)] if suffix is None else suffix,
        "max_decode_steps": 2,
    }


def _prefill_mdata(width: int = 2) -> dict[str, Any]:
    return {
        "is_beam_decode": False,
        "is_constraint_prefill": True,
        "is_constrained": True,
        "constraint_step": 0,
        "beam_width": width,
        "beam_suffix": [],
    }


# --------------------------------------------------------------------------
# validate_constraint_step
# --------------------------------------------------------------------------


def test_validate_constraint_step_accepts_fsm_steps() -> None:
    for step in (0, 1, 2):
        assert validate_constraint_step(step) == step
    # int-like values (numpy / numpy scalars from decode) are accepted.
    assert validate_constraint_step(np.int64(1)) == 1


def test_validate_constraint_step_rejects_out_of_range() -> None:
    for bad in (-1, 3, 7, "1", 1.0, None):
        with pytest.raises(RuntimeError, match="constraint_step"):
            validate_constraint_step(bad)
    with pytest.raises(RuntimeError, match="bool"):
        validate_constraint_step(True)


# --------------------------------------------------------------------------
# collect_constrained_rows
# --------------------------------------------------------------------------


def test_collect_constrained_rows_locates_decode_and_prefill_rows() -> None:
    """Row cursors must mirror _remux_logprobs_tensors traversal order."""
    request_row_configs = [
        ("completion", 0, 1, []),  # regular last-chunk prefill, 1 logits row
        ("cdecode", 0, 2, [0, 1]),  # constrained decode, W=2
        ("trie", 0, 2, [0, 1]),  # Trie beam decode (not constrained)
        ("cprefill", 0, 1, []),  # constrained prefill, 1 logits row
    ]
    beam_data = {
        "cdecode": _decode_mdata(step=1, width=2),
        "trie": {"is_beam_decode": True, "beam_width": 2},
        "cprefill": _prefill_mdata(width=2),
    }

    collected = collect_constrained_rows(request_row_configs, beam_data)

    assert [(r, start, w) for r, start, w, _ in collected] == [
        ("cdecode", 1, 2),
        ("cprefill", 5, 1),
    ]
    assert collected[0][3]["constraint_step"] == 1
    assert collected[1][3]["constraint_step"] == 0


def test_collect_constrained_rows_skips_intermediate_chunk() -> None:
    """A constrained prefill middle chunk (W_logits=0) never runs the operator."""
    request_row_configs: list[tuple[str, int, int, list[int]]] = [("cprefill", 0, 0, [])]
    beam_data = {"cprefill": _prefill_mdata(width=2)}

    assert collect_constrained_rows(request_row_configs, beam_data) == []


def test_collect_constrained_rows_rejects_invalid_step_even_without_rows() -> None:
    """Chunk-only visits still validate constraint_step (never silently skip)."""
    request_row_configs: list[tuple[str, int, int, list[int]]] = [("cdecode", 0, 0, [])]
    beam_data = {"cdecode": _decode_mdata(step=5, width=2)}

    with pytest.raises(RuntimeError, match="constraint_step"):
        collect_constrained_rows(request_row_configs, beam_data)


def test_collect_constrained_rows_rejects_missing_step() -> None:
    """A constrained request without constraint_step must fail fast.

    Defaulting the missing field to 0 would silently treat a decode request as
    prefill and build the wrong sequence_group.
    """
    request_row_configs = [("cdecode", 0, 2, [0, 1])]
    beam_data = {
        "cdecode": {
            "is_beam_decode": True,
            "is_constrained": True,
            "beam_width": 2,
            "beam_suffix": [[0], [1]],
        }
    }

    with pytest.raises(RuntimeError, match="missing constraint_step"):
        collect_constrained_rows(request_row_configs, beam_data)


# --------------------------------------------------------------------------
# build_sequence_group_for_step
# --------------------------------------------------------------------------


def test_build_sequence_group_for_step_prefill_ignores_content() -> None:
    rows = [("a", 0, 1, _prefill_mdata(width=2)), ("b", 1, 1, _prefill_mdata(width=2))]

    seq_group = build_sequence_group_for_step(rows, "cpu", 0)

    assert tuple(seq_group.shape) == (2, 1, 1)
    assert seq_group.dtype == torch.int32
    assert (seq_group == 0).all()


def test_build_sequence_group_for_step_decode_step1_writes_sa() -> None:
    rows = [("a", 0, 2, _decode_mdata(step=1, width=2, suffix=[[1], [4]]))]

    seq_group = build_sequence_group_for_step(rows, "cpu", 1)

    # max_step = max(step+1, max_decode_steps+1) = max(2, 3) = 3.
    assert tuple(seq_group.shape) == (1, 2, 3)
    assert seq_group[0, 0, 0].item() == 1  # s_a of beam 0
    assert seq_group[0, 1, 0].item() == 4  # s_a of beam 1
    # Positions >= step are ignored by the operator.
    assert (seq_group[:, :, 1:] == 0).all()


def test_build_sequence_group_for_step_decode_step2_writes_sa_sb() -> None:
    rows = [("a", 0, 2, _decode_mdata(step=2, width=2, suffix=[[1, 5], [4, 6]]))]

    seq_group = build_sequence_group_for_step(rows, "cpu", 2)

    assert tuple(seq_group.shape) == (1, 2, 3)
    assert seq_group[0, 0].tolist() == [1, 5, 0]  # s_a, s_b
    assert seq_group[0, 1].tolist() == [4, 6, 0]


def test_build_sequence_group_for_step_rejects_short_suffix() -> None:
    """beam_suffix shorter than constraint_step is a protocol error."""
    rows = [("a", 0, 2, _decode_mdata(step=2, width=2, suffix=[[1], [4]]))]

    with pytest.raises(RuntimeError, match="inconsistent"):
        build_sequence_group_for_step(rows, "cpu", 2)


def test_build_sequence_group_for_step_rejects_mixed_widths() -> None:
    rows = [
        ("a", 0, 2, _decode_mdata(step=1, width=2)),
        ("b", 2, 3, _decode_mdata(step=1, width=3)),
    ]

    with pytest.raises(RuntimeError, match="mixed-width"):
        build_sequence_group_for_step(rows, "cpu", 1)


# --------------------------------------------------------------------------
# patch_sampler_output_for_constrained
# --------------------------------------------------------------------------


def _sampler_output(width: int = 3, rows: int = 2) -> SimpleNamespace:
    lp = SimpleNamespace(
        logprobs=torch.full((rows, width), -1.0, dtype=torch.float32),
        logprob_token_ids=torch.zeros((rows, width), dtype=torch.int64),
        selected_token_ranks=torch.zeros(rows, dtype=torch.long),
    )
    return SimpleNamespace(
        sampled_token_ids=torch.zeros((rows, 1), dtype=torch.long),
        logprobs_tensors=lp,
    )


def test_patch_sampler_output_overrides_and_resizes() -> None:
    out = _sampler_output(width=3, rows=2)
    top_tokens = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    top_logprobs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]], dtype=torch.float32)
    valid_mask = torch.ones_like(top_logprobs, dtype=torch.bool)

    patch_sampler_output_for_constrained(out, [0, 1], top_tokens, top_logprobs, valid_mask, width=2)

    # Sampled token = best constrained candidate per row.
    assert out.sampled_token_ids.tolist() == [[7], [9]]
    # Operator rows replace the constrained columns; width shrinks to W.
    assert out.logprobs_tensors.logprobs.shape == (2, 2)
    assert out.logprobs_tensors.logprob_token_ids.shape == (2, 2)
    assert out.logprobs_tensors.logprobs.tolist() == [
        pytest.approx([-0.1, -0.2]),
        pytest.approx([-0.3, -0.4]),
    ]
    assert out.logprobs_tensors.logprob_token_ids.tolist() == [[7, 8], [9, 10]]
    assert out.logprobs_tensors.selected_token_ranks.tolist() == [0, 0]


def test_patch_sampler_output_uses_mask_not_logprob_magnitude() -> None:
    out = _sampler_output(width=2, rows=2)
    top_tokens = torch.tensor([[7, 8], [9, 0]], dtype=torch.int32)
    top_logprobs = torch.tensor([[0.0, -1.0e20], [0.0, -1.0e20]])
    valid_mask = torch.tensor([[True, True], [True, False]])

    patch_sampler_output_for_constrained(out, [0, 1], top_tokens, top_logprobs, valid_mask, width=2)

    assert out.logprobs_tensors.logprobs[0].tolist() == pytest.approx([0.0, -1.0e20])
    assert out.logprobs_tensors.logprobs[1, 0].item() == 0.0
    assert out.logprobs_tensors.logprobs[1, 1].item() == -float("inf")


def test_patch_sampler_output_scatters_non_contiguous_rows() -> None:
    out = _sampler_output(width=3, rows=4)
    top_tokens = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    top_logprobs = torch.zeros((2, 2), dtype=torch.float32)
    valid_mask = torch.ones_like(top_logprobs, dtype=torch.bool)

    patch_sampler_output_for_constrained(out, [1, 3], top_tokens, top_logprobs, valid_mask, width=2)

    assert out.sampled_token_ids.tolist() == [[0], [1], [0], [3]]
    # Unpatched row 2 keeps its original (pre-resize) values dropped.
    assert out.logprobs_tensors.logprobs.shape == (4, 2)
    assert out.logprobs_tensors.logprob_token_ids[1].tolist() == [1, 2]
    assert out.logprobs_tensors.logprob_token_ids[3].tolist() == [3, 4]


def test_patch_sampler_output_requires_logprobs() -> None:
    out = SimpleNamespace(
        sampled_token_ids=torch.zeros((2, 1)),
        logprobs_tensors=SimpleNamespace(logprobs=None),
    )
    with pytest.raises(RuntimeError, match="logprobs"):
        patch_sampler_output_for_constrained(
            out,
            [0, 1],
            torch.zeros((2, 2), dtype=torch.int32),
            torch.zeros((2, 2)),
            torch.ones((2, 2), dtype=torch.bool),
            width=2,
        )


def test_patch_sampler_output_handles_nullable_logprob_token_ids() -> None:
    """A sampler without ``logprob_token_ids`` must not crash the patch.

    The sampler-output contract makes ``logprob_token_ids`` optional (the beam
    collapse in model_runner_common guards it the same way); ``logprobs`` is
    the only required tensor.
    """
    lp = SimpleNamespace(
        logprobs=torch.full((2, 3), -1.0, dtype=torch.float32),
        logprob_token_ids=None,
        selected_token_ranks=torch.zeros(2, dtype=torch.long),
    )
    out = SimpleNamespace(
        sampled_token_ids=torch.zeros((2, 1), dtype=torch.long),
        logprobs_tensors=lp,
    )
    top_tokens = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    top_logprobs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]], dtype=torch.float32)
    valid_mask = torch.ones_like(top_logprobs, dtype=torch.bool)

    patch_sampler_output_for_constrained(out, [0, 1], top_tokens, top_logprobs, valid_mask, width=2)

    assert out.sampled_token_ids.tolist() == [[7], [9]]
    assert lp.logprobs.shape == (2, 2)
    assert lp.logprobs.tolist() == [
        pytest.approx([-0.1, -0.2]),
        pytest.approx([-0.3, -0.4]),
    ]
    assert lp.selected_token_ranks.tolist() == [0, 0]


# --------------------------------------------------------------------------
# run_constrained_partition
# --------------------------------------------------------------------------


class _RecordingAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[int, ...], int, int]] = []

    def run(
        self,
        logits: Any,
        sequence_group: Any,
        temperatures: Any,
        current_step: int,
        top_k: int,
    ) -> Any:
        rows = logits.shape[0]
        self.calls.append((rows, tuple(sequence_group.shape), current_step, top_k))
        return SimpleNamespace(
            top_tokens=torch.arange(rows * top_k, dtype=torch.int32).reshape(rows, top_k),
            top_logprobs=torch.full((rows, top_k), -0.5, dtype=torch.float32),
            valid_mask=torch.ones((rows, top_k), dtype=torch.bool),
        )


def test_run_constrained_partition_runs_per_request_and_step() -> None:
    """Mixed FSM steps share one step: each request runs with its own step."""
    rows = [
        ("a", 0, 2, _decode_mdata(step=1, width=2, suffix=[[1], [4]])),
        ("b", 2, 2, _decode_mdata(step=2, width=2, suffix=[[7, 9], [8, 10]])),
    ]
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            sampling_metadata=SimpleNamespace(temperature=torch.tensor([1.0, 1.0, 0.5, 0.5]))
        ),
    )
    adapter = _RecordingAdapter()
    logits = torch.randn(4, 20)

    top_tokens, top_logprobs, valid_mask, global_rows = run_constrained_partition(
        runner, logits, rows, width=2, adapter=adapter
    )

    assert adapter.calls == [
        (2, (1, 2, 3), 1, 2),  # request a: [1, W, max_step], step 1
        (2, (1, 2, 3), 2, 2),  # request b: [1, W, max_step], step 2
    ]
    assert global_rows == [0, 1, 2, 3]
    assert tuple(top_tokens.shape) == (4, 2)
    assert tuple(top_logprobs.shape) == (4, 2)
    assert valid_mask.all()


def test_run_constrained_partition_uses_request_temperature() -> None:
    rows = [("a", 0, 2, _decode_mdata(step=1, width=2, suffix=[[1], [4]]))]
    seen: list[torch.Tensor] = []

    class _TempsAdapter(_RecordingAdapter):
        def run(
            self,
            logits: Any,
            sequence_group: Any,
            temperatures: Any,
            current_step: int,
            top_k: int,
        ) -> Any:
            seen.append(temperatures)
            return super().run(logits, sequence_group, temperatures, current_step, top_k)

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            sampling_metadata=SimpleNamespace(temperature=torch.tensor([0.7, 0.7]))
        ),
    )

    run_constrained_partition(runner, torch.randn(2, 20), rows, width=2, adapter=_TempsAdapter())

    assert len(seen) == 1
    assert seen[0].dtype == torch.float32
    assert seen[0].tolist() == pytest.approx([0.7, 0.7])


# --------------------------------------------------------------------------
# apply_constrained_sample_override
# --------------------------------------------------------------------------


def test_apply_override_shortcircuits_without_beam_data() -> None:
    runner = SimpleNamespace(_current_beam_data={}, _current_scheduler_output=None)
    apply_constrained_sample_override(runner, torch.randn(1, 5), object())  # must not raise


def test_apply_override_shortcircuits_without_constrained_rows() -> None:
    beam_data = {"trie": {"is_beam_decode": True, "beam_width": 2}}
    runner = SimpleNamespace(
        _current_beam_data=beam_data,
        _current_scheduler_output=SimpleNamespace(_gr_request_row_configs=[("trie", 0, 2, [])]),
    )
    apply_constrained_sample_override(runner, torch.randn(2, 5), object())  # must not raise


def test_apply_override_rejects_unconstrained_logprobs_in_same_step() -> None:
    beam_data = {"c": _decode_mdata(step=1, width=2), "completion": {"beam_width": 1}}
    runner = SimpleNamespace(
        _current_beam_data=beam_data,
        _current_scheduler_output=SimpleNamespace(
            _gr_request_row_configs=[("completion", 0, 1, []), ("c", 1, 2, [0, 1])]
        ),
    )

    with pytest.raises(RuntimeError, match="mixed-batch"):
        apply_constrained_sample_override(runner, torch.randn(3, 5), object())


def test_apply_override_rejects_mixed_beam_widths() -> None:
    beam_data = {
        "a": _decode_mdata(step=1, width=2),
        "b": _decode_mdata(step=1, width=3),
    }
    runner = SimpleNamespace(
        _current_beam_data=beam_data,
        _current_scheduler_output=SimpleNamespace(
            _gr_request_row_configs=[("a", 0, 2, [0, 1]), ("b", 2, 3, [0, 1, 2])]
        ),
    )

    with pytest.raises(RuntimeError, match="mixed-width"):
        apply_constrained_sample_override(runner, torch.randn(5, 5), object())


def test_apply_override_end_to_end_patches_sampler_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full path: collect -> run -> in-place sampler-output patch -> width W."""
    beam_data = {"c": _decode_mdata(step=1, width=2, suffix=[[1], [4]])}
    request_row_configs = [("c", 0, 2, [0, 1])]
    out = _sampler_output(width=3, rows=2)  # width W+1, as vllm produces
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        _current_beam_data=beam_data,
        _current_scheduler_output=SimpleNamespace(_gr_request_row_configs=request_row_configs),
        input_batch=SimpleNamespace(
            sampling_metadata=SimpleNamespace(temperature=torch.tensor([1.0, 1.0]))
        ),
    )
    adapter = _RecordingAdapter()
    monkeypatch.setattr(
        "vllm_gr.v1.worker.constraint_hook.get_or_create_execution_adapter",
        lambda runner: adapter,
    )
    logits = torch.randn(2, 20)

    apply_constrained_sample_override(runner, logits, out)

    assert adapter.calls == [(2, (1, 2, 3), 1, 2)]
    assert out.logprobs_tensors.logprobs.shape == (2, 2)
    assert out.logprobs_tensors.logprob_token_ids.shape == (2, 2)
    assert out.sampled_token_ids[:, 0].tolist() == [0, 2]
    assert out.logprobs_tensors.selected_token_ranks.tolist() == [0, 0]


# --------------------------------------------------------------------------
# get_or_create_execution_adapter
# --------------------------------------------------------------------------


def test_get_or_create_execution_adapter_creates_and_caches(
    tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = compile_relations_json([["1", "2", "3"], ["1", "4", "5"]])
    resource = ConstraintTableCompiler(tokenizer, 20).compile(document)
    manager = SimpleNamespace(current_digest=resource.digest)
    manager.require = MagicMock(return_value=resource)
    adapter = MagicMock()

    monkeypatch.setattr("vllm_gr.constraints.worker_inject.get_worker_manager", lambda: manager)
    monkeypatch.setattr(
        "vllm_gr.constraints.startup.create_constraint_execution",
        lambda *, device: adapter,
    )

    runner = SimpleNamespace(device=torch.device("cpu"))

    first = get_or_create_execution_adapter(runner)
    second = get_or_create_execution_adapter(runner)

    assert first is adapter
    assert second is adapter
    adapter.load_tables.assert_called_once_with(resource)
    assert runner._gr_constraint_execution is adapter


def test_get_or_create_execution_adapter_fails_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SimpleNamespace(current_digest=None)
    monkeypatch.setattr("vllm_gr.constraints.worker_inject.get_worker_manager", lambda: manager)

    with pytest.raises(RuntimeError, match="not injected"):
        get_or_create_execution_adapter(SimpleNamespace(device=torch.device("cpu")))


# --------------------------------------------------------------------------
# Real Torch operator integration (sequence_group geometry acceptance)
# --------------------------------------------------------------------------


def test_hook_sequence_groups_are_accepted_by_real_torch_operator(
    tokenizer: MagicMock,
) -> None:
    """The hook-built sequence_groups must pass the operator's shape checks."""
    document = compile_relations_json([["1", "2", "3"], ["1", "4", "5"], ["7", "8", "9"]])
    resource = ConstraintTableCompiler(tokenizer, 20).compile(document)
    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)

    # Constrained prefill: [B=1, 1, 1] zeros.
    prefill_seq = build_sequence_group_for_step([("c", 0, 1, _prefill_mdata(width=2))], "cpu", 0)
    prefill_result = execution.run(torch.randn(1, 20), prefill_seq, torch.zeros(0), 0, 2)
    assert tuple(prefill_result.top_tokens.shape) == (1, 2)

    # Constrained decode step 1: s_a at position 0, max_step > current_step.
    decode_seq = build_sequence_group_for_step(
        [("c", 0, 2, _decode_mdata(step=1, width=2, suffix=[[1], [4]]))], "cpu", 1
    )
    decode_result = execution.run(torch.randn(2, 20), decode_seq, torch.zeros(0), 1, 2)
    assert tuple(decode_result.top_tokens.shape) == (2, 2)
