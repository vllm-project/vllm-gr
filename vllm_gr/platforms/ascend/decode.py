# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Select Ascend Beam decode for a single engine or runner."""

from typing import Any


def resolve_mode(requested: str | None) -> str:
    if requested is not None and requested != "auto":
        return requested
    from vllm_gr.ops.npu import beam_ops

    return "xllm" if beam_ops.is_available() else "cascade"


def engine_mode(config: Any) -> str:
    from vllm_gr.config import GRConfig

    if config is None:
        return "cascade"
    mode = getattr(config, "_gr_npu_decode_execution", None)
    if mode is None:
        mode = resolve_mode(GRConfig.from_vllm_config(config).npu_decode_execution)
        object.__setattr__(config, "_gr_npu_decode_execution", mode)
    return mode


def create_beam_context(runner: Any, gr_cfg: Any, *, max_batch: int) -> Any:
    from vllm_gr.v1.worker.gr_context import _create_beam_context

    context = _create_beam_context(runner, gr_cfg, max_batch=max_batch)
    context.decode_mode = engine_mode(getattr(runner, "vllm_config", None))
    return context
