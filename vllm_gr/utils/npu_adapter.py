# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility import for :mod:`vllm_gr.utils.accelerator`."""

import sys
from typing import TYPE_CHECKING

from vllm_gr.utils import accelerator as _impl

if TYPE_CHECKING:
    from vllm_gr.utils.accelerator import is_cuda_available as is_cuda_available
    from vllm_gr.utils.accelerator import is_npu_available as is_npu_available
    from vllm_gr.utils.accelerator import (
        release_accelerator_memory as release_accelerator_memory,
    )

sys.modules[__name__] = _impl
