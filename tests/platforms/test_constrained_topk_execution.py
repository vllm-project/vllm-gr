# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Contracts for platform-specific ConstrainedTopKExecution adapters."""

from __future__ import annotations

import pytest

from tests.helpers.cuda_platform import CudaGRPlatform
from vllm_gr.constraints.execution import (
    CUDAConstrainedTopKExecution,
    NPUConstrainedTopKExecution,
)

cuda = CudaGRPlatform()


def test_npu_integration_passes_device() -> None:
    """The device is optional and, when given, forwarded to the adapter."""
    from vllm_gr.constraints.execution import NPUConstrainedTopKExecution

    default_execution = NPUConstrainedTopKExecution()
    explicit_execution = NPUConstrainedTopKExecution(device="cpu")

    assert isinstance(default_execution, NPUConstrainedTopKExecution)
    assert default_execution._device is None
    assert isinstance(explicit_execution, NPUConstrainedTopKExecution)
    assert explicit_execution._device == "cpu"


def test_npu_integration_creates_independent_instances() -> None:
    """Each call must create an independent stateful adapter (not a singleton)."""
    # Table loading allocates tensors. Keep this CPU contract independent of NPU hardware.
    first = NPUConstrainedTopKExecution(device="cpu")
    second = NPUConstrainedTopKExecution(device="cpu")

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
    assert first._digest == resource.digest
    assert second._digest is None


def test_cuda_integration_returns_cuda_execution_and_preserves_device() -> None:
    """CUDA creates its execution and forwards an explicit device."""

    default_execution = CUDAConstrainedTopKExecution()
    explicit_execution = CUDAConstrainedTopKExecution(device="cuda")

    assert isinstance(default_execution, CUDAConstrainedTopKExecution)
    assert isinstance(explicit_execution, CUDAConstrainedTopKExecution)
    assert explicit_execution._device == "cuda"


pytestmark = pytest.mark.cpu_test
