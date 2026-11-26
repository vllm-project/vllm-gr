# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Tests for the ConstrainedTopKExecution protocol and platform adapters."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm_gr.constraints.compiler import (
    ConstraintTableCompiler,
    ConstraintTableResource,
)
from vllm_gr.constraints.execution import (
    INVALID_LOGIT,
    ConstrainedTopKTensorResult,
    NPUConstrainedTopKExecution,
    TorchConstrainedTopKExecution,
)
from vllm_gr.constraints.json_compiler import compile_relations_json


@pytest.fixture  # type: ignore[misc]
def tokenizer() -> MagicMock:
    result = MagicMock()
    result.convert_tokens_to_ids.side_effect = lambda values: [int(value) for value in values]
    result.get_vocab.return_value = {str(i): i for i in range(100)}
    return result


def _build_resource(tokenizer: MagicMock, vocab_size: int = 20) -> ConstraintTableResource:
    document = compile_relations_json([["1", "2", "3"], ["1", "4", "5"], ["7", "8", "9"]])
    return ConstraintTableCompiler(tokenizer, vocab_size).compile(document)


def test_tensor_result_validates_shape() -> None:
    result = ConstrainedTopKTensorResult(
        top_tokens=torch.zeros(2, 3, dtype=torch.int32),
        top_logprobs=torch.zeros(2, 3),
        valid_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    assert result.rows == 2
    assert result.top_k == 3

    with pytest.raises(ValueError, match="same shape"):
        ConstrainedTopKTensorResult(
            top_tokens=torch.zeros(2, 3),
            top_logprobs=torch.zeros(2, 4),
            valid_mask=torch.ones(2, 3, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="2-D"):
        ConstrainedTopKTensorResult(
            top_tokens=torch.zeros(3),
            top_logprobs=torch.zeros(3),
            valid_mask=torch.ones(3, dtype=torch.bool),
        )


def test_load_tables_copies_to_device_tensors(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)

    tables = execution._tables  # noqa: SLF001
    assert set(tables) == {
        "first_token_ids",
        "prefix1_offsets",
        "prefix1_values",
        "prefix1_pair_keys",
        "prefix2_value_offsets",
        "prefix2_values",
    }
    assert tables["prefix1_offsets"].dtype == torch.int32
    assert tables["prefix1_values"].dtype == torch.int32
    assert tables["prefix1_pair_keys"].dtype == torch.int64
    assert tables["first_token_ids"].dtype == torch.int32
    assert all(t.is_contiguous() for t in tables.values())


def test_load_tables_idempotent_same_digest(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)
    execution.load_tables(resource)  # same digest -> no-op, no error


def test_load_tables_rejects_different_digest(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer, vocab_size=20)
    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)

    other = _build_resource(tokenizer, vocab_size=30)  # different vocab -> different digest
    with pytest.raises(ValueError, match="reloading is not allowed"):
        execution.load_tables(other)


def test_run_before_load_fails(tokenizer: MagicMock) -> None:
    execution = TorchConstrainedTopKExecution(device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        execution.run(
            torch.randn(2, 20),
            torch.zeros(2, 1, 1, dtype=torch.int32),
            torch.zeros(0),
            0,
            2,
        )


def test_run_step0_prefill_returns_expected_shape(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)

    logits = torch.randn(2, 20)
    sequence_group = torch.zeros(2, 1, 1, dtype=torch.int32)
    result = execution.run(logits, sequence_group, torch.zeros(0), 0, 2)

    assert tuple(result.top_tokens.shape) == (2, 2)
    assert result.top_logprobs.shape == result.top_tokens.shape
    assert result.top_tokens.dtype == torch.int32
    assert result.top_logprobs.dtype == torch.float32
    assert result.valid_mask.dtype == torch.bool


def test_run_step1_decode_returns_expected_shape(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)

    # batch=2, width=2 -> 4 rows, one per parent beam.
    logits = torch.randn(4, 20)
    sequence_group = torch.zeros(2, 2, 5, dtype=torch.int32)
    result = execution.run(logits, sequence_group, torch.zeros(0), 1, 2)

    assert tuple(result.top_tokens.shape) == (4, 2)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("logits_shape", "sequence_group_shape", "temperatures", "step", "top_k", "match"),
    [
        pytest.param((2, 20), (2, 2, 1), 0, 0, 2, "must be \\[B, 1, 1\\]", id="prefill-rows"),
        pytest.param((2, 20), (2, 1, 5), 0, 0, 2, "must be \\[B, 1, 1\\]", id="prefill-depth"),
        # batch=2, width=2 -> rows must be 4, not 3.
        pytest.param((3, 20), (2, 2, 5), 0, 1, 2, "batch\\*width", id="decode-rows"),
        pytest.param((2, 20), (2, 1, 1), 0, 3, 2, "expected 0, 1, or 2", id="unsupported-step"),
        pytest.param((2, 5), (2, 1, 1), 0, 0, 2, "vocab_size", id="logits-vocab"),
        pytest.param((2, 20), (2, 1, 1), 0, 0, 0, "in \\[1, 256\\]", id="top-k-zero"),
        pytest.param((2, 20), (2, 1, 1), 0, 0, 300, "in \\[1, 256\\]", id="top-k-above-limit"),
        # rows=2, temperatures must be [0], [1], or [2].
        pytest.param((2, 20), (2, 1, 1), 3, 0, 2, "must have shape", id="temperature-shape"),
        pytest.param((4, 20), (2, 2, 1), 0, 1, 2, "> current_step", id="decode-max-step"),
    ],
)
@pytest.mark.l1  # type: ignore[misc]
def test_run_rejects_invalid_arguments(
    tokenizer: MagicMock,
    logits_shape: tuple[int, int],
    sequence_group_shape: tuple[int, int, int],
    temperatures: int,
    step: int,
    top_k: int,
    match: str,
) -> None:
    resource = _build_resource(tokenizer)
    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)

    with pytest.raises(ValueError, match=match):
        execution.run(
            torch.randn(*logits_shape),
            torch.zeros(*sequence_group_shape, dtype=torch.int32),
            torch.zeros(temperatures),
            step,
            top_k,
        )


def test_padding_validity_is_structural(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)

    # A catalog row whose valid candidates are < top_k: force a wide top_k.
    logits = torch.randn(2, 20)
    sequence_group = torch.zeros(2, 1, 1, dtype=torch.int32)
    result = execution.run(logits, sequence_group, torch.zeros(0), 0, 8)

    # Padding may retain the finite operator sentinel, but the explicit mask is
    # authoritative and independent of score magnitude.
    assert INVALID_LOGIT == -1.0e20
    assert (result.top_logprobs == INVALID_LOGIT).any()
    assert result.valid_mask.shape == result.top_logprobs.shape
    assert result.valid_mask.sum(dim=1).tolist() == [2, 2]


def test_execution_keeps_legal_logprob_at_padding_sentinel_magnitude(
    tokenizer: MagicMock,
) -> None:
    resource = _build_resource(tokenizer)
    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)
    logits = torch.zeros(1, 20)
    logits[0, 7] = 1.0e20

    result = execution.run(
        logits,
        torch.zeros(1, 1, 1, dtype=torch.int32),
        torch.ones(1),
        0,
        3,
    )

    assert result.valid_mask.tolist() == [[True, True, False]]
    assert set(result.top_tokens[0][result.valid_mask[0]].tolist()) == {1, 7}
    assert bool((result.top_logprobs[0][result.valid_mask[0]] < (INVALID_LOGIT / 2)).any())


def test_npu_adapter_fails_explicitly_when_op_unavailable(tokenizer: MagicMock) -> None:
    """The NPU adapter must not silently fall back to Torch."""
    resource = _build_resource(tokenizer)
    execution = NPUConstrainedTopKExecution(device="cpu")

    # In a non-NPU environment is_available() is False.
    if execution.is_available():
        pytest.skip("NPU custom op is available; skipping unavailability test")

    execution.load_tables(resource)
    with pytest.raises(RuntimeError, match="unavailable"):
        execution.run(
            torch.randn(2, 20),
            torch.zeros(2, 1, 1, dtype=torch.int32),
            torch.zeros(0),
            0,
            2,
        )


def test_npu_adapter_not_loaded_fails(tokenizer: MagicMock) -> None:
    execution = NPUConstrainedTopKExecution(device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        execution.run(
            torch.randn(2, 20),
            torch.zeros(2, 1, 1, dtype=torch.int32),
            torch.zeros(0),
            0,
            2,
        )


@pytest.mark.parametrize("device_class", ["cpu", "accelerator"])  # type: ignore[misc]
def test_npu_default_device_falls_back_without_crash(
    tokenizer: MagicMock, device_class: str
) -> None:
    """On a non-Ascend torch build, torch.device('npu') raises; load must not crash."""
    resource = _build_resource(tokenizer)
    execution = NPUConstrainedTopKExecution()  # no device passed
    # Classify the current default only when this case runs.  Collection must
    # not initialize an accelerator merely to select a CI level.
    try:
        selected = execution._default_device().type
        actual_class = "cpu" if selected == "cpu" else "accelerator"
        if actual_class != device_class:
            pytest.skip(f"default device is {actual_class}, not {device_class}")
        execution.load_tables(resource)
    except RuntimeError as exc:
        if "npu" in str(exc).lower():
            pytest.skip("torch build has no npu device type")
        raise


def test_npu_default_device_prefers_npu(
    tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NPUConstrainedTopKExecution must request prefer='npu' when no device given."""
    from vllm_gr.constraints import execution as execution_module

    resource = _build_resource(tokenizer)
    captured: dict[str, Any] = {}

    def fake_default_device(prefer: str | None = None) -> torch.device:
        captured["prefer"] = prefer
        return torch.device("cpu")

    monkeypatch.setattr(execution_module, "_default_device", fake_default_device)
    execution = NPUConstrainedTopKExecution()  # no device passed
    execution.load_tables(resource)

    assert captured.get("prefer") == "npu"


def test_npu_adapter_passes_arguments_and_normalizes_dtype(
    tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a fake custom op, the NPU adapter must forward all args and emit int32."""
    import vllm_gr.ops.npu.rec_constrained_topk as npu_ops

    resource = _build_resource(tokenizer)
    execution = NPUConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)

    captured: dict[str, Any] = {}

    def fake_npu_op(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        # The real op returns (tokens, logprobs); return int64 tokens to verify
        # the adapter normalizes to int32.
        logits = args[0]
        top_k = args[10]
        rows = logits.shape[0]
        return (
            torch.zeros(rows, top_k, dtype=torch.int64),
            torch.zeros(rows, top_k, dtype=torch.float32),
        )

    monkeypatch.setattr(npu_ops, "rec_constrained_topk_npu", fake_npu_op)
    monkeypatch.setattr(npu_ops, "is_available", lambda: True)

    logits = torch.randn(2, 20)
    seq = torch.zeros(2, 1, 1, dtype=torch.int32)
    temperatures = torch.zeros(1)
    result = execution.run(logits, seq, temperatures, 0, 2)

    # Positional contract of rec_constrained_topk_npu:
    # 0 logits, 1 sequence_group, 2..7 tables (first_token_ids, prefix1_offsets,
    # prefix1_values, prefix1_pair_keys, prefix2_value_offsets, prefix2_values),
    # 8 temperatures, 9 current_step, 10 top_k, 11 max_prefix1_degree,
    # 12 max_prefix2_degree.
    args = captured["args"]
    assert args[0] is logits
    assert args[1] is seq
    assert args[2] is execution._tables["first_token_ids"]  # noqa: SLF001
    assert args[3] is execution._tables["prefix1_offsets"]  # noqa: SLF001
    assert args[4] is execution._tables["prefix1_values"]  # noqa: SLF001
    assert args[5] is execution._tables["prefix1_pair_keys"]  # noqa: SLF001
    assert args[6] is execution._tables["prefix2_value_offsets"]  # noqa: SLF001
    assert args[7] is execution._tables["prefix2_values"]  # noqa: SLF001
    assert args[8] is temperatures
    assert args[9] == 0  # current_step
    assert args[10] == 2  # top_k
    assert args[11] == resource.max_prefix1_degree
    assert args[12] == resource.max_prefix2_degree
    assert captured["kwargs"] == {}  # no stray keyword arguments

    assert result.top_tokens.dtype == torch.int32  # normalized
    assert result.top_logprobs.dtype == torch.float32


def test_reset_allows_reload_with_new_digest(tokenizer: MagicMock) -> None:
    """reset() drops the loaded state so a different digest can be loaded."""
    resource_a = _build_resource(tokenizer, vocab_size=20)
    resource_b = _build_resource(tokenizer, vocab_size=30)
    execution = TorchConstrainedTopKExecution(device="cpu")

    execution.load_tables(resource_a)
    with pytest.raises(ValueError, match="reset\\(\\) first"):
        execution.load_tables(resource_b)

    execution.reset()
    assert execution._digest is None  # noqa: SLF001
    assert execution._tables == {}  # noqa: SLF001

    execution.load_tables(resource_b)
    assert execution._digest == resource_b.digest  # noqa: SLF001

    # run works with the new digest's vocab.
    result = execution.run(
        torch.randn(2, 30), torch.zeros(2, 1, 1, dtype=torch.int32), torch.zeros(0), 0, 2
    )
    assert result.top_tokens.dtype == torch.int32


def test_run_wide_degree_takes_bucketed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prefix with > 256 continuations must run via the bucketed top-k path.

    step 1 uses prefix1_offsets; with degree 512 and top_k=8 the width exceeds
    2*top_k, so the operator must dispatch to _bucketed_topk.  We patch the
    wrapper to prove the path is actually taken (step 0 would not exercise it).
    """
    import importlib

    # vllm_gr.ops exports rec_constrained_topk as a *function*; the module that
    # defines _bucketed_topk must be imported by its dotted path.
    ops_module = importlib.import_module("vllm_gr.ops.rec_constrained_topk")

    degree = 512
    wide_tokenizer = MagicMock()
    wide_tokenizer.convert_tokens_to_ids.side_effect = lambda values: [
        int(value) for value in values
    ]
    wide_tokenizer.get_vocab.return_value = {str(i): i for i in range(2000)}

    catalog = [["1", str(i), str(i)] for i in range(degree)]
    document = compile_relations_json(catalog)
    resource = ConstraintTableCompiler(wide_tokenizer, degree + 10).compile(document)
    assert resource.max_prefix1_degree == degree

    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)

    real_bucketed = ops_module._bucketed_topk
    calls: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real_bucketed(*args, **kwargs)

    monkeypatch.setattr(ops_module, "_bucketed_topk", spy)

    # step 1: sequence_group [[[1, 0]]] shape [1, 1, 2]; s_a token 1 has
    # degree 512 s_b continuations, so width = 512 > 2*8 -> bucketed path.
    logits = torch.randn(1, degree + 10)
    seq = torch.tensor([[[1, 0]]], dtype=torch.int32)
    result = execution.run(logits, seq, torch.zeros(0), 1, top_k=8)

    assert calls, "expected _bucketed_topk to be invoked for the wide degree"
    assert tuple(result.top_tokens.shape) == (1, 8)
    assert result.top_tokens.dtype == torch.int32


def test_load_tables_validates_resource(tokenizer: MagicMock) -> None:
    """load_tables must reject an invalid directly-constructed resource."""
    from vllm_gr.constraints.compiler import ConstraintTableResource

    resource = _build_resource(tokenizer)
    invalid = ConstraintTableResource(
        first_token_ids=resource.first_token_ids,
        prefix1_offsets=resource.prefix1_offsets,
        prefix1_values=resource.prefix1_values,
        prefix1_pair_keys=resource.prefix1_pair_keys,
        prefix2_value_offsets=resource.prefix2_value_offsets,
        prefix2_values=resource.prefix2_values,
        max_prefix1_degree=1,  # derived degree is 2
        max_prefix2_degree=resource.max_prefix2_degree,
        vocab_size=resource.vocab_size,
    )
    execution = TorchConstrainedTopKExecution(device="cpu")
    with pytest.raises(ValueError, match="does not match"):
        execution.load_tables(invalid)


def test_run_rejects_wrong_output_row_count(
    tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter must reject operator output with the wrong row count."""
    ops_module = importlib.import_module("vllm_gr.ops.rec_constrained_topk")

    resource = _build_resource(tokenizer)
    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)

    real_torch = ops_module.rec_constrained_topk_torch

    def bad_op(*args: Any, **kwargs: Any) -> Any:
        tokens, probs = real_torch(*args, **kwargs)
        return tokens[:1], probs[:1]

    monkeypatch.setattr(ops_module, "rec_constrained_topk_torch", bad_op)
    with pytest.raises(ValueError, match="expected"):
        execution.run(
            torch.randn(2, 20), torch.zeros(2, 1, 1, dtype=torch.int32), torch.zeros(0), 0, 2
        )


def test_run_rejects_non_contiguous_output(
    tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter must reject non-contiguous operator output."""
    ops_module = importlib.import_module("vllm_gr.ops.rec_constrained_topk")

    resource = _build_resource(tokenizer)
    execution = TorchConstrainedTopKExecution(device="cpu")
    execution.load_tables(resource)

    real_torch = ops_module.rec_constrained_topk_torch

    def noncontig_op(*args: Any, **kwargs: Any) -> Any:
        tokens, probs = real_torch(*args, **kwargs)
        return tokens.transpose(0, 1), probs.transpose(0, 1)

    monkeypatch.setattr(ops_module, "rec_constrained_topk_torch", noncontig_op)
    with pytest.raises(ValueError, match="contiguous"):
        execution.run(
            torch.randn(2, 20), torch.zeros(2, 1, 1, dtype=torch.int32), torch.zeros(0), 0, 2
        )


pytestmark = pytest.mark.cpu_test
