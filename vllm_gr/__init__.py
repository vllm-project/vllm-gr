# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Public API package for vllm-GR.

Top-level ``import vllm_gr`` is intentionally lightweight:

- it does not apply monkey patches
- it does not register models/configs
- it does not require NPU-specific dependencies

Runtime initialization is performed explicitly by
``vllm_gr.plugin.initialize_runtime()`` and by the vLLM general plugin entry
point ``vllm_gr.plugin:register``.
"""

from importlib import import_module
from typing import Any
import warnings

__version__ = "0.3.0"

_PUBLIC_EXPORTS: dict[str, tuple[str, str, bool]] = {
    "GRPrompt": ("vllm_gr.inputs.data", "GRPrompt", False),
    "is_gr_prompt": ("vllm_gr.inputs.data", "is_gr_prompt", False),
    "RecommendedItem": ("vllm_gr.outputs", "RecommendedItem", True),
    "GRRequestOutput": ("vllm_gr.outputs", "GRRequestOutput", True),
    "GRLLM": ("vllm_gr.entrypoints.gr", "GRLLM", True),
}

_COMPAT_EXPORTS: dict[str, tuple[str, str, bool]] = {
    "BeamAttentionBackend": (
        "vllm_gr.v1.attention.backends.beam_attn",
        "BeamAttentionBackend",
        False,
    ),
    "HSTUConfig": ("vllm_gr.config.hstu_config", "HSTUConfig", False),
    "ensure_hstu_config_registered": (
        "vllm_gr.config.hstu_vllm_registration",
        "ensure_hstu_config_registered",
        True,
    ),
    "GRRequest": ("vllm_gr.request", "GRRequest", True),
    "is_gr_request": ("vllm_gr.request", "is_gr_request", True),
    "GREngineCoreRequest": ("vllm_gr.engine", "GREngineCoreRequest", True),
    "GRNewRequestData": ("vllm_gr.core.sched.output", "GRNewRequestData", True),
    "GRInputProcessor": (
        "vllm_gr.engine.gr_input_processor",
        "GRInputProcessor",
        True,
    ),
    "GROutputProcessor": (
        "vllm_gr.engine.gr_output_processor",
        "GROutputProcessor",
        True,
    ),
    "GRRequestState": (
        "vllm_gr.engine.gr_output_processor",
        "GRRequestState",
        True,
    ),
    "GREngineArgs": ("vllm_gr.engine.arg_utils", "GREngineArgs", True),
    "GRLLMEngine": ("vllm_gr.engine.gr_engine", "GRLLMEngine", True),
    "GRWorker": ("vllm_gr.worker.gr_worker", "GRWorker", True),
    "GRModelRunner": ("vllm_gr.worker.gr_model_runner", "GRModelRunner", True),
}

_LAZY_EXPORTS = {**_PUBLIC_EXPORTS, **_COMPAT_EXPORTS}

__all__ = [
    "__version__",
    "GRPrompt",
    "is_gr_prompt",
    "RecommendedItem",
    "GRRequestOutput",
    "GRLLM",
]


def _check_env_before_vllm_import() -> None:
    from vllm_gr._env_check import check_opencv_env

    check_opencv_env()


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name, needs_env_check = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    if name in _COMPAT_EXPORTS:
        warnings.warn(
            (
                f"`vllm_gr.{name}` is deprecated as a top-level import and will "
                "move behind its module namespace in a future cleanup. "
                "Prefer importing it from its defining submodule."
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    if needs_env_check:
        _check_env_before_vllm_import()

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
