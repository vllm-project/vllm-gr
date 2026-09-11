# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared ModelRunner support for vLLM-GR."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "GRDecodeDeviceInputs": "vllm_gr.v1.worker.gr_persistent_beam",
    "GRFinalDeviceView": "vllm_gr.v1.worker.gr_persistent_beam",
    "GRModelRunnerState": "vllm_gr.v1.worker.gr_model_runner_state",
    "GRModelRunnerStateRegistry": "vllm_gr.v1.worker.gr_model_runner_state",
    "GRModelRunnerStatus": "vllm_gr.v1.worker.gr_model_runner_state",
    "GRPersistentBeamBackend": "vllm_gr.v1.worker.gr_persistent_beam",
    "GRPrefillDeviceInputs": "vllm_gr.v1.worker.gr_persistent_beam",
    "GPUModelRunnerPersistentBeamAdapter": "vllm_gr.v1.worker.gr_persistent_beam",
    "GPUModelRunnerPersistentBeamBackend": "vllm_gr.v1.worker.gr_persistent_beam",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
