# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility import for :mod:`vllm_gr.platforms.ascend.admission`."""

import sys

from vllm_gr.platforms.ascend import admission as _impl

sys.modules[__name__] = _impl
