# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Original code from Meta Platforms, Inc. (generative-recommenders)

"""DLRM_v3 model exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_TO_MODULE = {
    "DlrmHSTU": "vllm_gr.models.dlrm_v3.modules.dlrm_hstu",
    "DlrmHSTUConfig": "vllm_gr.models.dlrm_v3.modules.dlrm_hstu",
    "STU": "vllm_gr.models.dlrm_v3.modules.stu",
    "STULayer": "vllm_gr.models.dlrm_v3.modules.stu",
    "STUStack": "vllm_gr.models.dlrm_v3.modules.stu",
    "STULayerConfig": "vllm_gr.models.dlrm_v3.modules.stu",
    "HSTUTransducer": "vllm_gr.models.dlrm_v3.modules.hstu_transducer",
}

__all__ = list(_EXPORT_TO_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
