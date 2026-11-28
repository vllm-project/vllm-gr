# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Prepare the persistent stage execution view for shared Beam Forward capture."""

from __future__ import annotations

from typing import Any

from vllm_gr.config.execution import validate_gr_execution_runtime


def _capture_enabled(worker: Any) -> bool:
    from vllm_gr.config.models import configured_constraint_mode
    from vllm_gr.v1.worker.gr_context import _get_gr_config

    cfg = _get_gr_config(worker)
    validate_gr_execution_runtime(worker.vllm_config)
    mode = configured_constraint_mode(cfg)
    if not cfg.beam_worker_decision or cfg.beam_max_width is None or mode is None:
        raise ValueError(
            "Beam V1 requires worker decisions, fixed width and either an enabled "
            "canonical table or gr.constraint_mode=none"
        )
    # A one-token session completes in Prefill and never replays a Decode graph.
    return (
        not worker.model_config.enforce_eager
        and cfg.beam_graph_enabled
        and cfg.beam_max_decode_steps > 1
    )
