# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Public API package for vllm-gr.

Top-level ``import vllm_gr`` is intentionally lightweight:

- it does not apply monkey patches
- it does not register models/configs
- it does not require NPU-specific dependencies

Runtime initialization is performed explicitly by
``vllm_gr.plugin.initialize_runtime()`` and by the vLLM general plugin entry
point ``vllm_gr.plugin:register``.

After the GR adapter / recommend runtime was removed, the only public symbol
re-exported here is :class:`GRLLM`, which is now a thin subclass of vLLM's
:class:`~vllm.entrypoints.llm.LLM` providing offline beam search with optional
catalog-constrained generation (the OneRec path).
"""

from typing import Any

from vllm_gr.version import __version__

_PUBLIC_EXPORTS: dict[str, tuple[str, str, bool]] = {
    "GRLLM": ("vllm_gr.entrypoints.llm", "GRLLM", True),
}

_LAZY_EXPORTS = dict(_PUBLIC_EXPORTS)

__all__ = [
    "__version__",
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

    if needs_env_check:
        _check_env_before_vllm_import()

    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
