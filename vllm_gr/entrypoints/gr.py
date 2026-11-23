# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility import for :mod:`vllm_gr.entrypoints.llm`."""

import sys
from typing import TYPE_CHECKING

from vllm_gr.entrypoints import llm as _impl

if TYPE_CHECKING:
    from .llm import GRLLM as GRLLM

sys.modules[__name__] = _impl
