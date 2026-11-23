# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility import for :mod:`vllm_gr.engine.arg_utils`."""

import sys

from vllm_gr.engine import arg_utils as _impl

sys.modules[__name__] = _impl
