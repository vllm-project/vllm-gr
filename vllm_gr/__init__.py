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

__all__ = [
    "__version__",
    "GRLLM",
]


def __getattr__(name: str) -> Any:
    if name != "GRLLM":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from vllm_gr._env_check import check_opencv_env

    check_opencv_env()

    from vllm_gr.entrypoints.llm import GRLLM

    globals()[name] = GRLLM
    return GRLLM


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
