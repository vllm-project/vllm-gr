# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts for provider-created ConstrainedTopKExecution adapters."""

from __future__ import annotations

import pytest

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
    # Table loading allocates tensors. Keep this CPU contract independent of NPU hardware.
    first = npu.NPU_BACKEND.create_constrained_topk_execution(device="cpu")
    second = npu.NPU_BACKEND.create_constrained_topk_execution(device="cpu")

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


def test_cuda_provider_returns_cuda_execution_and_preserves_device() -> None:
    """CUDA provider creates its execution and forwards an explicit device."""
    from vllm_gr.constraints.execution import CUDAConstrainedTopKExecution

    default_execution = cuda.CUDA_BACKEND.create_constrained_topk_execution()
    explicit_execution = cuda.CUDA_BACKEND.create_constrained_topk_execution(device="cuda")

    assert isinstance(default_execution, CUDAConstrainedTopKExecution)
    assert isinstance(explicit_execution, CUDAConstrainedTopKExecution)
    assert explicit_execution._device == "cuda"  # noqa: SLF001


pytestmark = pytest.mark.cpu_test
