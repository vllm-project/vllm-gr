# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""KV writes shared by the CUDA and Ascend cascade attention paths."""

import torch


def cache_unshared_kv(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    slot: int,
    decode_step: int,
    beam_session: object = None,
) -> None:
    """Write one cascade decode step at a zero-based slot and offset."""
    from vllm_gr.v1.attention.backends.beam_kernels import (
        cache_unshared_kv as _cache_unshared_kv,
    )

    del beam_session
    _cache_unshared_kv(
        unshared_key,
        unshared_value,
        curr_key,
        curr_value,
        slot,
        decode_step,
    )
