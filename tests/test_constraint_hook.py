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

from vllm_gr.constraints.backend import ConstraintOutput
from vllm_gr.constraints.compiler import ConstraintTableCompiler
from vllm_gr.constraints.execution import TorchConstrainedTopKExecution
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.v1.worker.constraint_hook import (
    _prepare_workspace_constraint_metadata,
    apply_constrained_sample_override,
    build_constraint_backend_prefix_token_ids,
    build_direct_constrained_sampler_output,
    build_sequence_group_for_step,
    collect_constrained_rows,
    get_or_create_execution_adapter,
    patch_sampler_output_for_constrained,
    run_constrained_partition,
    try_sample_with_constraint_backend,
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


def test_direct_intermediate_chunk_returns_empty_sampler_output() -> None:
    backend = _CanonicalBackend(max_top_k=2, max_rows=2)
    runner, events = _direct_runner(
        {"req": _prefill_mdata(width=2)},
        [("req", 0, 0, [])],
        backend,
    )

    output = try_sample_with_constraint_backend(runner, torch.empty((0, 8)))

    assert output is not None
    assert output.sampled_token_ids.shape == (1, 1)
    assert output.sampled_token_ids.tolist() == [[-1]]
    assert output.sampled_token_ids.dtype == torch.int32
    assert output.logprobs_tensors is None
    assert backend.calls == []
    assert events == []


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


def test_run_constrained_partition_restores_canonical_row_order_across_steps() -> None:
    """Mixed FSM steps run step-grouped but output rows realign to logits rows.

    The step-1 request (``b``) occupies the *higher* canonical rows, so the
    ascending-step submission order would emit rows ``[2, 3, 0, 1]``; the
    operator results must be gathered back to canonical sampler row order.
    """
    rows = [
        ("a", 0, 2, _decode_mdata(step=2, width=2, suffix=[[7, 9], [8, 10]])),
        ("b", 2, 2, _decode_mdata(step=1, width=2, suffix=[[1], [4]])),
    ]
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            sampling_metadata=SimpleNamespace(temperature=torch.tensor([1.0, 1.0, 0.5, 0.5]))
        ),
    )

    class _RowIdentityAdapter:
        def run(
            self,
            logits: Any,
            sequence_group: Any,
            temperatures: Any,
            current_step: int,
            top_k: int,
        ) -> Any:
            rows_n = logits.shape[0]
            return SimpleNamespace(
                top_tokens=logits[:, :top_k].to(torch.int32).contiguous(),
                top_logprobs=torch.full((rows_n, top_k), -0.5, dtype=torch.float32),
                valid_mask=torch.ones((rows_n, top_k), dtype=torch.bool),
            )

    # Each logits column of row ``r`` holds the value ``r``, so the adapter's
    # identity ``top_tokens`` reveals which logits row every output row came from.
    logits = torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(4, 5).contiguous()

    top_tokens, top_logprobs, valid_mask, global_rows = run_constrained_partition(
        runner, logits, rows, width=2, adapter=_RowIdentityAdapter()
    )

    assert global_rows == [0, 1, 2, 3]
    assert top_tokens[:, 0].tolist() == [0, 1, 2, 3]
    assert top_tokens[:, 1].tolist() == [0, 1, 2, 3]
    assert tuple(top_logprobs.shape) == (4, 2)
    assert valid_mask.all()


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
# Canonical runner-owned Backend direct path
# --------------------------------------------------------------------------


class _CanonicalBackend:
    def __init__(self, *, max_top_k: int = 4, max_rows: int = 4) -> None:
        self.info = SimpleNamespace(max_top_k=max_top_k, max_rows=max_rows)
        self.calls: list[tuple[torch.Tensor, Any]] = []

    def select(self, logits: torch.Tensor, metadata: Any) -> ConstraintOutput:
        call_index = len(self.calls) + 1
        self.calls.append((logits.clone(), metadata))
        rows = logits.shape[0]
        width = metadata.top_k
        token_ids = (
            torch.arange(rows * width, dtype=torch.int32, device=logits.device)
            .reshape(rows, width)
            .add(call_index * 10)
        )
        logprobs = torch.arange(
            rows * width,
            dtype=torch.float32,
            device=logits.device,
        ).reshape(rows, width)
        logprobs = logprobs.neg()
        return ConstraintOutput(
            token_ids=token_ids,
            logprobs=logprobs,
            valid_mask=torch.ones_like(token_ids, dtype=torch.bool),
        )


def _direct_runner(
    beam_data: dict[str, Any],
    request_row_configs: list[tuple[Any, ...]],
    backend: Any,
) -> tuple[SimpleNamespace, list[str]]:
    events: list[str] = []
    logits_rows = sum(int(config[2]) for config in request_row_configs)
    input_batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(temperature=torch.ones(logits_rows, dtype=torch.float32)),
        update_async_output_token_ids=lambda: events.append("async"),
    )
    return (
        SimpleNamespace(
            constraint_backend=backend,
            device=torch.device("cpu"),
            input_batch=input_batch,
            _current_beam_data=beam_data,
            _current_scheduler_output=SimpleNamespace(_gr_request_row_configs=request_row_configs),
        ),
        events,
    )


def test_canonical_prefix_is_fixed_rows_by_three_for_steps_zero_one_two() -> None:
    step0, width0, prefix0 = build_constraint_backend_prefix_token_ids(
        "prefill",
        1,
        _prefill_mdata(width=2),
        torch.device("cpu"),
    )
    step1, width1, prefix1 = build_constraint_backend_prefix_token_ids(
        "step1",
        2,
        _decode_mdata(step=1, width=2, suffix=[[11], [12]]),
        torch.device("cpu"),
    )
    step2, width2, prefix2 = build_constraint_backend_prefix_token_ids(
        "step2",
        2,
        _decode_mdata(step=2, width=2, suffix=[[1, 11], [2, 12]]),
        torch.device("cpu"),
    )

    assert (step0, width0) == (0, 2)
    assert (step1, width1) == (1, 2)
    assert (step2, width2) == (2, 2)
    assert prefix0.dtype is torch.int32
    assert prefix0.shape == (1, 3)
    assert prefix0.tolist() == [[0, 0, 0]]
    assert prefix1.shape == (2, 3)
    assert prefix1.tolist() == [[11, 0, 0], [12, 0, 0]]
    assert prefix2.shape == (2, 3)
    assert prefix2.tolist() == [[1, 11, 0], [2, 12, 0]]


def test_canonical_prefix_rejects_missing_parent_or_short_suffix() -> None:
    with pytest.raises(RuntimeError, match="parent beam suffixes"):
        build_constraint_backend_prefix_token_ids(
            "req",
            2,
            _decode_mdata(step=1, width=2, suffix=[[1]]),
            torch.device("cpu"),
        )
    with pytest.raises(RuntimeError, match="last 2 token"):
        build_constraint_backend_prefix_token_ids(
            "req",
            2,
            _decode_mdata(step=2, width=2, suffix=[[1], [2, 3]]),
            torch.device("cpu"),
        )


def test_persistent_constraint_prefix_uses_authoritative_session_decision() -> None:
    sequence = torch.tensor(
        [[[11, 21, 0], [12, 22, 0]]],
        dtype=torch.int32,
    )
    session = SimpleNamespace(
        slot=0,
        context=SimpleNamespace(decision=SimpleNamespace(sequence=sequence)),
    )
    state = SimpleNamespace(step=2, session=session)
    runner = SimpleNamespace(
        _gr_model_runner_states={"gr-0": state},
        input_batch=SimpleNamespace(
            sampling_metadata=SimpleNamespace(
                temperature=torch.ones(2, dtype=torch.float32)
            )
        ),
    )
    host_prefix = torch.empty((1, 2, 3), dtype=torch.int32)
    workspace = SimpleNamespace(
        max_batch=1,
        key=SimpleNamespace(beam_width=2),
        host_prefix_token_ids=host_prefix,
        host_prefix_token_ids_numpy=host_prefix.numpy(),
        prefix_token_ids=torch.empty((1, 2, 3), dtype=torch.int32),
        temperatures=torch.empty((1, 2), dtype=torch.float32),
        host_temperatures=torch.empty((1, 2), dtype=torch.float32),
    )
    metadata = _decode_mdata(step=2, width=2, suffix=None)
    metadata["persistent_runner_state"] = True

    prepared = _prepare_workspace_constraint_metadata(
        runner,
        [("gr-0", 0, 2, metadata)],
        workspace,
        total_rows=2,
        device=torch.device("cpu"),
    )

    assert prepared is not None
    step, prefix, temperatures = prepared
    assert step == 2
    assert prefix.tolist() == [[11, 21, 0], [12, 22, 0]]
    assert temperatures.tolist() == [1.0, 1.0]


def test_direct_path_calls_backend_once_per_request_and_preserves_row_order() -> None:
    backend = _CanonicalBackend(max_top_k=2, max_rows=2)
    beam_data = {
        "prefill": _prefill_mdata(width=2),
        "step1": _decode_mdata(step=1, width=2, suffix=[[3], [4]]),
        "step2": _decode_mdata(
            step=2,
            width=2,
            suffix=[[5, 7], [6, 8]],
        ),
    }
    request_row_configs = [
        ("prefill", 0, 1, []),
        ("step1", 1, 2, [0, 1]),
        ("step2", 3, 2, [0, 1]),
    ]
    runner, events = _direct_runner(
        beam_data,
        request_row_configs,
        backend,
    )
    logits = torch.arange(40, dtype=torch.float32).reshape(5, 8)

    output = try_sample_with_constraint_backend(runner, logits)

    assert output is not None
    assert events == ["async"]
    assert len(backend.calls) == 3
    assert [call[0][:, 0].tolist() for call in backend.calls] == [
        [0.0],
        [8.0, 16.0],
        [24.0, 32.0],
    ]
    assert [call[1].current_step for call in backend.calls] == [0, 1, 2]
    assert [tuple(call[1].prefix_token_ids.shape) for call in backend.calls] == [
        (1, 3),
        (2, 3),
        (2, 3),
    ]
    assert [call[1].top_k for call in backend.calls] == [2, 2, 2]
    assert output.sampled_token_ids[:, 0].tolist() == [10, 20, 22, 30, 32]
    assert output.logprobs_tensors.logprob_token_ids.shape == (5, 2)
    assert output.logprobs_tensors.selected_token_ranks.tolist() == [0] * 5


def test_direct_path_batches_requests_at_the_same_step() -> None:
    backend = _CanonicalBackend(max_top_k=2, max_rows=4)
    beam_data = {
        "left": _decode_mdata(step=1, width=2, suffix=[[3], [4]]),
        "right": _decode_mdata(step=1, width=2, suffix=[[5], [6]]),
    }
    runner, events = _direct_runner(
        beam_data,
        [("left", 0, 2, [0, 1]), ("right", 2, 2, [0, 1])],
        backend,
    )
    logits = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    output = try_sample_with_constraint_backend(runner, logits)

    assert output is not None
    assert events == ["async"]
    assert len(backend.calls) == 1
    called_logits, metadata = backend.calls[0]
    assert torch.equal(called_logits, logits)
    assert metadata.current_step == 1
    assert metadata.prefix_token_ids.tolist() == [
        [3, 0, 0],
        [4, 0, 0],
        [5, 0, 0],
        [6, 0, 0],
    ]
    assert output.sampled_token_ids[:, 0].tolist() == [10, 12, 14, 16]


def test_direct_output_uses_mask_for_negative_infinity_padding() -> None:
    token_ids = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    logprobs = torch.tensor(
        [[-1.0e20, -1.0e20], [-0.5, -1.0e20]],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor([[True, True], [True, False]])

    output = build_direct_constrained_sampler_output(
        token_ids,
        logprobs,
        valid_mask,
    )

    assert output.sampled_token_ids.tolist() == [[7], [9]]
    assert output.logprobs_tensors.logprob_token_ids is token_ids
    assert output.logprobs_tensors.logprobs[0, 1].item() == pytest.approx(-1.0e20)
    assert torch.isneginf(output.logprobs_tensors.logprobs[1, 1])


def test_direct_path_rejects_empty_legal_candidate_row() -> None:
    class EmptyBackend(_CanonicalBackend):
        def select(self, logits: torch.Tensor, metadata: Any) -> ConstraintOutput:
            output = super().select(logits, metadata)
            output.valid_mask[0, :] = False
            return output

    backend = EmptyBackend(max_top_k=2, max_rows=2)
    runner, events = _direct_runner(
        {"req": _prefill_mdata(width=2)},
        [("req", 0, 1, [])],
        backend,
    )

    with pytest.raises(RuntimeError, match="no legal candidate"):
        try_sample_with_constraint_backend(runner, torch.zeros((1, 8)))
    assert events == ["async"]


def test_direct_path_rejects_spec_decode_before_backend_call() -> None:
    backend = _CanonicalBackend(max_top_k=2, max_rows=2)
    runner, events = _direct_runner(
        {"req": _prefill_mdata(width=2)},
        [("req", 0, 1, [])],
        backend,
    )

    with pytest.raises(RuntimeError, match="speculative"):
        try_sample_with_constraint_backend(
            runner,
            torch.zeros((1, 8)),
            spec_decode_metadata=object(),
        )
    assert backend.calls == []
    assert events == []

    chunk_runner, chunk_events = _direct_runner(
        {"chunk": _prefill_mdata(width=2)},
        [("chunk", 0, 0, [])],
        backend,
    )
    with pytest.raises(RuntimeError, match="speculative"):
        try_sample_with_constraint_backend(
            chunk_runner,
            None,
            spec_decode_metadata=object(),
        )
    assert backend.calls == []
    assert chunk_events == []


def test_direct_path_rejects_mixed_rows_and_widths_before_backend_call() -> None:
    backend = _CanonicalBackend(max_top_k=4, max_rows=4)
    mixed_rows_runner, row_events = _direct_runner(
        {"constrained": _prefill_mdata(width=2)},
        [("plain", 0, 1, []), ("constrained", 1, 1, [])],
        backend,
    )
    with pytest.raises(RuntimeError, match="mixed-batch"):
        try_sample_with_constraint_backend(
            mixed_rows_runner,
            torch.zeros((2, 8)),
        )

    mixed_width_runner, width_events = _direct_runner(
        {
            "a": _prefill_mdata(width=2),
            "b": _prefill_mdata(width=3),
        },
        [("a", 0, 1, []), ("b", 1, 1, [])],
        backend,
    )
    with pytest.raises(RuntimeError, match="mixed-width"):
        try_sample_with_constraint_backend(
            mixed_width_runner,
            torch.zeros((2, 8)),
        )

    assert backend.calls == []
    assert row_events == []
    assert width_events == []


def test_direct_path_rejects_beam_width_above_backend_top_k() -> None:
    backend = _CanonicalBackend(max_top_k=1, max_rows=2)
    runner, events = _direct_runner(
        {"req": _prefill_mdata(width=2)},
        [("req", 0, 1, [])],
        backend,
    )

    with pytest.raises(RuntimeError, match="max_top_k"):
        try_sample_with_constraint_backend(runner, torch.zeros((1, 8)))
    assert backend.calls == []
    assert events == []


def test_missing_canonical_backend_fails_but_legacy_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beam_data = {"req": _prefill_mdata(width=2)}
    runner, events = _direct_runner(
        beam_data,
        [("req", 0, 1, [])],
        None,
    )
    monkeypatch.setattr(
        "vllm_gr.v1.worker.constraint_hook._canonical_constraint_table_enabled",
        lambda runner: True,
    )
    with pytest.raises(RuntimeError, match="constraint_backend is None"):
        try_sample_with_constraint_backend(runner, torch.zeros((1, 8)))

    # A legacy request must return to the existing sampler-then-override path,
    # even when its metadata is invalid; legacy validation still occurs later.
    beam_data["req"].pop("constraint_step")
    monkeypatch.setattr(
        "vllm_gr.v1.worker.constraint_hook._canonical_constraint_table_enabled",
        lambda runner: False,
    )
    assert try_sample_with_constraint_backend(runner, torch.zeros((1, 8))) is None
    assert events == []


def test_first_direct_request_uses_only_preloaded_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"runtime attempted persistent loading: {args} {kwargs}")

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        forbidden,
    )
    monkeypatch.setattr(
        "vllm_gr.constraints.artifact.load_constraint_table_artifact",
        forbidden,
    )
    monkeypatch.setattr(
        "vllm_gr.constraints.startup.create_constraint_execution",
        forbidden,
    )

    backend = _CanonicalBackend(max_top_k=2, max_rows=2)
    runner, events = _direct_runner(
        {"req": _prefill_mdata(width=2)},
        [("req", 0, 1, [])],
        backend,
    )

    output = try_sample_with_constraint_backend(runner, torch.zeros((1, 8)))

    assert output is not None
    assert events == ["async"]
    assert len(backend.calls) == 1
    assert runner.constraint_backend is backend
    assert not hasattr(runner, "_gr_constraint_execution")


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
