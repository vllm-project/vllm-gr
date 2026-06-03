# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Original code from Meta Platforms, Inc. (generative-recommenders)

"""Core DLRM_v3 module exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_TO_MODULE = {
    "ActionEncoder": "vllm_gr.models.dlrm_v3.modules.action_encoder",
    "DlrmHSTU": "vllm_gr.models.dlrm_v3.modules.dlrm_hstu",
    "DlrmHSTUConfig": "vllm_gr.models.dlrm_v3.modules.dlrm_hstu",
    "STU": "vllm_gr.models.dlrm_v3.modules.stu",
    "STULayer": "vllm_gr.models.dlrm_v3.modules.stu",
    "STUStack": "vllm_gr.models.dlrm_v3.modules.stu",
    "STULayerConfig": "vllm_gr.models.dlrm_v3.modules.stu",
    "HSTUTransducer": "vllm_gr.models.dlrm_v3.modules.hstu_transducer",
    "HSTUPositionalEncoder": "vllm_gr.models.dlrm_v3.modules.positional_encoder",
    "InputPreprocessor": "vllm_gr.models.dlrm_v3.modules.preprocessors",
    "ContextualPreprocessor": "vllm_gr.models.dlrm_v3.modules.preprocessors",
    "OutputPostprocessor": "vllm_gr.models.dlrm_v3.modules.postprocessors",
    "L2NormPostprocessor": "vllm_gr.models.dlrm_v3.modules.postprocessors",
    "LayerNormPostprocessor": "vllm_gr.models.dlrm_v3.modules.postprocessors",
    "TimestampLayerNormPostprocessor": "vllm_gr.models.dlrm_v3.modules.postprocessors",
    "MultitaskModule": "vllm_gr.models.dlrm_v3.modules.multitask_module",
    "DefaultMultitaskModule": "vllm_gr.models.dlrm_v3.modules.multitask_module",
    "MultitaskTaskType": "vllm_gr.models.dlrm_v3.modules.multitask_module",
    "TaskConfig": "vllm_gr.models.dlrm_v3.modules.multitask_module",
}

__all__ = list(_EXPORT_TO_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
