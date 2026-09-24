# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patch for FlatLogprobs.append_fast to use extend instead of
per-element append."""

from vllm.logprobs import FlatLogprobs

from vllm_gr.logprobs import append_fast as _append_fast
from vllm_gr.logprobs import (
    append_logprobs_for_next_position as _append_logprobs_for_next_position,
)


def patch_flat_logprobs():
    FlatLogprobs.append_fast = _append_fast
    import vllm.logprobs

    vllm.logprobs.append_logprobs_for_next_position = _append_logprobs_for_next_position

    try:
        import vllm.v1.engine.logprobs

        vllm.v1.engine.logprobs.append_logprobs_for_next_position = (
            _append_logprobs_for_next_position
        )
    except ImportError:
        pass
