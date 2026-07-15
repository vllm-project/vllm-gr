# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-gr-specific attention backend selection helpers."""

from collections.abc import Mapping
from typing import Any

VLLM_GR_ATTENTION_BACKEND_KEY = "vllm_gr_attention_backend"
VLLM_GR_BEAM_ATTENTION_BACKEND = "BEAM"


def is_beam_attention_enabled(vllm_config: Any | None) -> bool:
    """Return whether a config explicitly requested the Beam backend."""
    if vllm_config is None:
        return False

    additional_config = getattr(vllm_config, "additional_config", None)
    return (
        isinstance(additional_config, Mapping)
        and additional_config.get(VLLM_GR_ATTENTION_BACKEND_KEY) == VLLM_GR_BEAM_ATTENTION_BACKEND
    )
