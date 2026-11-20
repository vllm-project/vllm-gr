# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Install and refresh stage-local projections on a native NPU runner."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from vllm.logger import init_logger

from vllm_gr.ops.npu.stage_projection import (
    NPUStageProjection,
    supported_projection_parts,
)

logger = init_logger(__name__)
_RELOAD_PATCH = "_vllm_gr_npu_stage_reload_weights"
_UPDATE_PATCH = "_vllm_gr_npu_stage_weight_update"


def can_install_npu_stage_projection(runner: Any) -> bool:
    """Allow only supported TP=PP=1 models; callers select the NPU backend."""
    return supported_projection_parts(runner) is not None


def install_npu_stage_projection(runner: Any, backend: Any) -> NPUStageProjection | None:
    """Default to the fast path only when both model and table are supported."""
    parts = supported_projection_parts(runner)
    if parts is None:
        return None
    stage_vocab = getattr(backend, "stage_vocab", None)
    if stage_vocab is None:
        return None
    if stage_vocab.vocab_size >= backend.info.table_metadata.vocab_size:
        logger.info(
            "NPU stage LM head has no vocabulary reduction; using full-vocabulary projection"
        )
        return None
    lm_head, processor = parts
    projection = NPUStageProjection(runner, processor, lm_head, stage_vocab)
    projection.install()
    logger.info(
        "NPU stage LM head ready: full vocabulary=%d, compact vocabulary=%d, stage sizes=%s",
        lm_head.weight.shape[0],
        projection.vocab_size,
        projection.stage_token_counts,
    )
    return projection


def install_npu_sid_stage_projection(runner: Any, stage_vocab: Any) -> NPUStageProjection | None:
    """Optionally project tokenizer SID families without a catalog Backend.

    Disabling the projection through gr.npu_stage_lm_head=false retains
    the stage-format semantics: candidate selection then masks full logits.
    The same owner handles ordinary/dummy fallback and native weight updates.
    """
    parts = supported_projection_parts(runner)
    if parts is None:
        return None
    if stage_vocab.vocab_size >= stage_vocab.full_vocab_size:
        return None
    lm_head, processor = parts
    if lm_head.weight.shape[0] < stage_vocab.full_vocab_size:
        raise ValueError("SID stage vocabulary exceeds the loaded LM-head vocabulary")
    projection = NPUStageProjection(runner, processor, lm_head, stage_vocab)
    projection.install()
    logger.info(
        "NPU SID stage LM head ready: full vocabulary=%d, compact vocabulary=%d, stage sizes=%s",
        stage_vocab.full_vocab_size,
        projection.vocab_size,
        projection.stage_token_counts,
    )
    return projection


def _runner_projection(runner: Any) -> NPUStageProjection | None:
    projection = getattr(runner, "_gr_npu_stage_projection", None)
    return projection if projection is not None and not projection.closed else None


def make_npu_stage_reload_weights(original: Callable[..., Any]) -> Callable[..., Any]:
    """Refresh after native checkpoint finalization or direct kernel copies."""
    if getattr(original, _RELOAD_PATCH, False):
        return original

    @wraps(original)
    def reload_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
        projection = _runner_projection(self)
        if projection is None:
            return original(self, *args, **kwargs)
        projection.begin_weight_update()
        try:
            result = original(self, *args, **kwargs)
            projection.finish_weight_update()
        except BaseException:
            projection.fail_weight_update()
            raise
        return result

    setattr(reload_weights, _RELOAD_PATCH, True)
    return reload_weights


def _make_weight_update_hook(original: Callable[..., Any], method_name: str) -> Callable[..., Any]:
    @wraps(original)
    def weight_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        projection = _runner_projection(getattr(self, "model_runner", None))
        if projection is None:
            return original(self, *args, **kwargs)
        if method_name == "start_weight_update":
            projection.begin_weight_update()
        else:
            projection._require_weight_update()
        try:
            result = original(self, *args, **kwargs)
            if method_name == "finish_weight_update":
                projection.finish_weight_update()
        except BaseException:
            # Native code may already have written a subset of model weights.
            # Retrying start/finish must not make that cache valid again.
            projection.fail_weight_update()
            raise
        return result

    setattr(weight_update, _UPDATE_PATCH, True)
    return weight_update


def install_npu_stage_weight_update_hooks(worker_cls: type[Any]) -> None:
    """Wrap the NPU worker's complete start/chunk/finish update transaction.

    The native worker can copy kernel-format parameters without calling
    model.load_weights, and checkpoint chunks are not finalized until finish.
    Errors conservatively require a complete model reload before inference.
    """
    names = ("start_weight_update", "update_weights", "finish_weight_update")
    if not all(callable(getattr(worker_cls, name, None)) for name in names):
        return
    for name in names:
        original = getattr(worker_cls, name)
        if not getattr(original, _UPDATE_PATCH, False):
            setattr(worker_cls, name, _make_weight_update_hook(original, name))
