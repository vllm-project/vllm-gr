# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility import for :mod:`vllm_gr.platforms.npu.admission`."""

import sys

from vllm_gr.platforms.npu import admission as _impl

sys.modules[__name__] = _impl
