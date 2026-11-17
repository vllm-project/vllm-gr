# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Beam startup requirements shared by frontends and workers."""

from vllm_gr.config.models import GRConfig


def require_beam_config(config: object) -> None:
    if GRConfig.from_vllm_config(config).beam_max_width is None:
        raise ValueError("Beam search requires gr.beam_width configured at engine startup")
