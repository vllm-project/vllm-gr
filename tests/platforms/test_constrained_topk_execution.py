# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts for provider-created ConstrainedTopKExecution adapters."""

from __future__ import annotations

from vllm_gr.platforms import cuda, npu


def test_npu_provider_creates_adapter() -> None:
    """NPU provider must create a concrete NPUConstrainedTopKExecution."""
    from vllm_gr.constraints.execution import NPUConstrainedTopKExecution

    execution = npu.NPU_BACKEND.create_constrained_topk_execution()

    assert isinstance(execution, NPUConstrainedTopKExecution)


def test_npu_provider_passes_device() -> None:
    """The requested device must be forwarded to the adapter."""
    from vllm_gr.constraints.execution import NPUConstrainedTopKExecution

    execution = npu.NPU_BACKEND.create_constrained_topk_execution(device="cpu")

    assert isinstance(execution, NPUConstrainedTopKExecution)
    assert execution._device == "cpu"  # noqa: SLF001


def test_npu_provider_creates_independent_instances() -> None:
    """Each call must create an independent stateful adapter (not a singleton)."""
    first = npu.NPU_BACKEND.create_constrained_topk_execution()
    second = npu.NPU_BACKEND.create_constrained_topk_execution()

    assert first is not second
    # Loading tables on one must not affect the other.
    from unittest.mock import MagicMock

    from vllm_gr.constraints.compiler import ConstraintTableCompiler
    from vllm_gr.constraints.json_compiler import compile_relations_json

    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids.side_effect = lambda values: [int(value) for value in values]
    tokenizer.get_vocab.return_value = {str(i): i for i in range(100)}
    document = compile_relations_json([["1", "2", "3"], ["1", "4", "5"]])
    resource = ConstraintTableCompiler(tokenizer, 10).compile(document)

    first.load_tables(resource)
    assert first._digest == resource.digest  # noqa: SLF001
    assert second._digest is None  # noqa: SLF001


def test_cuda_provider_returns_none() -> None:
    """CUDA provider must explicitly return None (no Constraint Table execution)."""
    execution = cuda.CUDA_BACKEND.create_constrained_topk_execution()

    assert execution is None


def test_cuda_provider_none_with_device_arg() -> None:
    """CUDA provider returns None regardless of the requested device."""
    execution = cuda.CUDA_BACKEND.create_constrained_topk_execution(device="cuda")

    assert execution is None
