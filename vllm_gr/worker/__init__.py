# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_TO_MODULE = {
    "GRWorker": "vllm_gr.worker.gr_worker",
    "GRModelRunner": "vllm_gr.worker.gr_model_runner",
    "GRNPUModelRunner": "vllm_gr.worker.gr_npu_model_runner",
    "GRNPUWorker": "vllm_gr.worker.gr_npu_worker",
    "GRFeatureRunnerMixin": "vllm_gr.worker.gr_runner_mixin",
}

__all__ = list(_EXPORT_TO_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
