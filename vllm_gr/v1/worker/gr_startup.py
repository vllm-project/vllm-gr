# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Prepare the persistent stage execution view for shared Beam Forward capture."""

from __future__ import annotations

from typing import Any

import torch

from vllm_gr.v1.engine.beam_search_v1 import validate_gr_execution_runtime
from vllm_gr.v1.worker.runner_lifecycle import get_stage_execution


def capture_async_forward(
    worker: Any,
    *,
    allow_eager_fallback: bool = False,
) -> bool:
    with torch.inference_mode():
        if not _capture_enabled(worker):
            return False
        return bool(
            get_stage_execution(worker).capture_startup_graph(
                worker, allow_eager_fallback=allow_eager_fallback
            )
        )


def _capture_enabled(worker: Any) -> bool:
    from vllm_gr.v1.worker.gr_context import (
        _beam_graph_enabled_for_run,
        _get_gr_config,
    )

    cfg = _get_gr_config(worker)
    validate_gr_execution_runtime(worker.vllm_config)
    unconstrained = getattr(cfg, "constrained_generation", None) is False
    if (
        not cfg.beam_worker_decision
        or cfg.beam_max_width is None
        or (
            not unconstrained and (cfg.constraint_table is None or not cfg.constraint_table.enabled)
        )
    ):
        raise ValueError(
            "Beam V1 requires worker decisions, fixed width and either an enabled "
            "canonical table or constrained_generation=False"
        )
    if unconstrained:
        get_stage_execution(worker).admit_unconstrained_beam(worker)
    # A one-token session completes in Prefill and never replays a Decode graph.
    return _beam_graph_enabled_for_run(worker) and cfg.beam_max_decode_steps > 1
