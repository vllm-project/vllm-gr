# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared CUDA Beam workspace initialization and model-load accounting."""

from __future__ import annotations

from functools import wraps

import torch
from vllm.logger import init_logger

from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES

logger = init_logger(__name__)

_BEAM_POSTPROCESS_LOAD_PATCH_MARKER = "_vllm_gr_beam_postprocess_load_patched"


def initialize_beam_postprocess_workspaces(
    runner,
    *,
    max_batch: int | None = None,
):
    """Prewarm the configured CUDA Beam postprocess bucket on ``runner``.

    V1 reserves the largest supported logical batch so every admitted size
    reuses the same fixed-address state and selection storage.
    """

    device = torch.device(getattr(runner, "device", "cpu"))
    if device.type != "cuda":
        return None

    from vllm_gr.config import resolve_candidate_width
    from vllm_gr.v1.worker.cuda.beam_postprocess_workspace import (
        BeamPostprocessWorkspace,
        BeamPostprocessWorkspaceKey,
        BeamPostprocessWorkspacePool,
    )
    from vllm_gr.v1.worker.gr_context import (
        _get_gr_config,
    )

    gr_cfg = _get_gr_config(runner)
    width = gr_cfg.beam_max_width
    if not bool(gr_cfg.beam_worker_decision) or width is None:
        return None
    candidate_width = resolve_candidate_width(gr_cfg)

    context = getattr(runner, "_beam_context", None)
    # The Beam context is created later, on the first admitted batch, with
    # max_batch=max(SUPPORTED_BEAM_BATCH_SIZES); the workspace built here must
    # match it or the stage runner rejects the session as incompatible.
    default_max_batch = max(SUPPORTED_BEAM_BATCH_SIZES)
    resolved_max_batch = (
        int(max_batch)
        if max_batch is not None
        else int(getattr(context, "max_batch", default_max_batch))
    )
    dtype = getattr(getattr(runner, "model_config", None), "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(
            f"CUDA Beam postprocess workspace requires model_config.dtype, got {dtype!r}"
        )

    key = BeamPostprocessWorkspaceKey(int(width), int(candidate_width), dtype)
    existing = getattr(runner, "_beam_postprocess_workspaces", None)
    if existing is not None:
        workspace = existing.require(key)
        if workspace.max_batch != resolved_max_batch or workspace.max_decode_steps != int(
            gr_cfg.beam_max_decode_steps
        ):
            raise RuntimeError(
                "prewarmed Beam postprocess workspace geometry changed after "
                "initialization; rebuilding fixed-address buffers is forbidden"
            )
        return existing

    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=resolved_max_batch,
        beam_width=int(width),
        top_k=int(candidate_width),
        max_decode_steps=int(gr_cfg.beam_max_decode_steps),
        device=device,
        logits_dtype=dtype,
        pin_host=True,
    )
    pool = BeamPostprocessWorkspacePool()
    pool.add(workspace)
    runner._beam_postprocess_workspaces = pool
    logger.info(
        "vllm-gr: prewarmed CUDA Beam postprocess workspace "
        "(batch=%d, width=%d, top_k=%d, dtype=%s, device_bytes=%d, pinned_bytes=%d)",
        resolved_max_batch,
        width,
        candidate_width,
        dtype,
        workspace.device_memory_bytes,
        workspace.pinned_memory_bytes,
    )
    return pool


def make_load_model_with_beam_postprocess_workspace(original):
    """Initialize fixed Beam postprocess storage after model weights load."""

    if getattr(original, _BEAM_POSTPROCESS_LOAD_PATCH_MARKER, False):
        return original

    @wraps(original)
    def load_model_with_beam_postprocess_workspace(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        workspace_pool = initialize_beam_postprocess_workspaces(self)
        if workspace_pool is not None:
            # The pool is allocated after native load_model has measured the
            # model weights, but before Worker memory profiling.  Treat its
            # device storage as model-owned memory so KV-cache capacity is not
            # computed from bytes that are already permanently resident.
            self.model_memory_usage = int(getattr(self, "model_memory_usage", 0)) + int(
                workspace_pool.device_memory_bytes
            )
        return result

    setattr(
        load_model_with_beam_postprocess_workspace,
        _BEAM_POSTPROCESS_LOAD_PATCH_MARKER,
        True,
    )
    return load_model_with_beam_postprocess_workspace
