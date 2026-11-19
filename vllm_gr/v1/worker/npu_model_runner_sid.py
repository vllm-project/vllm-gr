# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""NPU ModelRunner SID stage loading and teardown."""

# Native NPU runner internals remain dynamically typed; the npu-worker mypy
# group still checks facade re-exports and caller-side helper signatures.
# mypy: disable-error-code="no-any-return,no-untyped-call,no-untyped-def,type-arg,var-annotated"

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def make_patched_load_model(original: Callable[..., Any]) -> Callable[..., Any]:
    """Install catalog-free SID state after the shared NPU model load."""
    marker = "_vllm_gr_npu_sid_load_patch"
    if getattr(original, marker, False):
        return original

    @wraps(original)
    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        # The inner shared loader closes the old projection before loading
        # native weights. Detach the NPU-owned maps at the same boundary.
        self._gr_npu_sid_stage_vocab = None
        projection = None
        base_memory_usage = None
        try:
            result = original(self, *args, **kwargs)

            from vllm_gr.config import GRConfig

            config = GRConfig.from_vllm_config(getattr(self, "vllm_config", None))
            if config.constrained_generation is not False or not config.beam_sid_stage_vocab:
                return result
            if str(getattr(self, "device", "")).split(":")[0] != "npu":
                raise ValueError("beam.sid_stage_vocab currently requires NPU")

            from vllm.tokenizers import cached_tokenizer_from_config

            from vllm_gr.ops.npu.sid_stage_vocab import build_sid_stage_vocab
            from vllm_gr.ops.npu_stage_logits import install_npu_sid_stage_projection
            from vllm_gr.v1.worker.model_runner_patch import _runner_vocab_size

            base_memory_usage = int(getattr(self, "model_memory_usage", 0))
            tokenizer = cached_tokenizer_from_config(self.model_config)
            if tokenizer is None:
                raise ValueError("beam.sid_stage_vocab requires tokenizer initialization")
            stage_vocab = build_sid_stage_vocab(tokenizer, _runner_vocab_size(self), self.device)
            projection = install_npu_sid_stage_projection(self, stage_vocab)
            memory_usage = (
                base_memory_usage
                + stage_vocab.device_memory_bytes
                + (projection.device_memory_bytes if projection is not None else 0)
            )
            stage_sizes = tuple(int(ids.numel()) for ids in stage_vocab.stage_token_ids)

            # Publish only when maps and the optional projection are complete.
            self._gr_npu_sid_stage_vocab = stage_vocab
            self._gr_npu_stage_projection = projection
            self.model_memory_usage = memory_usage
            logger.info(
                "NPU catalog-free SID vocabulary ready: stage sizes=%s, projection=%s",
                stage_sizes,
                "sliced" if projection is not None else "full",
            )
            return result
        except BaseException:
            self._gr_npu_sid_stage_vocab = None
            # A projection may be installed but not yet published if a later
            # preparation step fails. The shared closer owns published state.
            if (
                projection is not None
                and getattr(self, "_gr_npu_stage_projection", None) is not projection
            ):
                projection.close()
            from vllm_gr.v1.worker.model_runner_patch import close_constraint_backend

            close_constraint_backend(self)
            if base_memory_usage is not None:
                self.model_memory_usage = base_memory_usage
            raise

    setattr(load_model, marker, True)
    return load_model


def make_patched_shutdown(original: Callable[..., Any]) -> Callable[..., Any]:
    """Detach NPU SID maps before the shared projection/backend shutdown."""
    marker = "_vllm_gr_npu_sid_shutdown_patch"
    if getattr(original, marker, False):
        return original

    @wraps(original)
    def shutdown(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._gr_npu_sid_stage_vocab = None
        return original(self, *args, **kwargs)

    setattr(shutdown, marker, True)
    return shutdown


def make_patched_reload_weights(original):
    from vllm_gr.ops.npu_stage_logits import make_npu_stage_reload_weights

    return make_npu_stage_reload_weights(original)
