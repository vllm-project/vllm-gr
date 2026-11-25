# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Eager V1 Beam hooks for vLLM's CPUModelRunner.

The Ascend eager stage runner is plain PyTorch outside a few lazily imported
NPU kernels, and vLLM's CPUModelRunner shares GPUModelRunner's input contract.
The CPU path therefore installs the same wrapper factories on CPUModelRunner
and leaves out the graph-capture and SID-vocabulary hooks.
"""

from __future__ import annotations

import sys

from vllm_gr.v1.worker.model_runner_contracts import is_worker_runner_patched
from vllm_gr.v1.worker.npu.npu_model_runner_patch import (
    make_patched_determine_batch_execution_and_padding_npu as make_patched_determine_batch_execution_and_padding,
)
from vllm_gr.v1.worker.npu.npu_model_runner_patch import (
    make_patched_prepare_inputs_npu as make_patched_prepare_inputs,
)
from vllm_gr.v1.worker.npu.npu_model_runner_patch import (
    make_patched_sample_tokens_npu as make_patched_sample_tokens,
)
from vllm_gr.v1.worker.npu.npu_model_runner_patch import (
    make_patched_update_states_npu as make_patched_update_states,
)

__all__ = [
    "discover_worker_runners",
    "install",
    "is_patched",
    "make_patched_determine_batch_execution_and_padding",
    "make_patched_prepare_inputs",
    "make_patched_sample_tokens",
    "make_patched_update_states",
]


def discover_worker_runners() -> tuple[tuple[str, type], ...]:
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    return (("CPUModelRunner", CPUModelRunner),)


def is_patched(runner_class: type) -> bool:
    from vllm_gr.v1.worker.runner_lifecycle import has_runner_lifecycle

    return is_worker_runner_patched(runner_class) and has_runner_lifecycle(runner_class, __name__)


def install() -> None:
    from vllm_gr.v1.worker.model_runner_patch import install_v1_worker_hooks

    install_v1_worker_hooks(sys.modules[__name__], implementation_id=__name__)
