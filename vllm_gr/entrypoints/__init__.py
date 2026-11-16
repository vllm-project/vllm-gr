# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Entrypoints package for vllm-GR."""

from typing import Any

__all__ = ["GRLLM"]


def __getattr__(name: str) -> Any:
    if name == "GRLLM":
        from vllm_gr.entrypoints.gr import GRLLM

        return GRLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
