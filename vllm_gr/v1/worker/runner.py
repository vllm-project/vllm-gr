# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Model-owned GR attention and execute/sample lifecycle helpers."""

import logging
from functools import wraps
from typing import Any

from vllm_gr.config import GRAttentionBackend, GRConfig

logger = logging.getLogger(__name__)


def close_gr_attention(runner: Any) -> None:
    execution = getattr(runner, "_gr_attention_execution", None)
    runner._gr_attention_execution = None
    runner._gr_ready = False
    layers = getattr(runner, "_gr_attention_layers", ())
    runner._gr_attention_layers = ()
    errors = []
    for impl in layers:
        try:
            impl.release_gr_attention_execution(execution)
        except Exception as exc:
            errors.append(exc)
    close = getattr(execution, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


def initialize_gr_attention(runner: Any) -> None:
    from vllm_gr.platforms import current_gr_platform

    gr = GRConfig.from_vllm_config(runner.vllm_config)
    execution = None
    layers: tuple[Any, ...] = ()
    try:
        if gr.attention_backend is GRAttentionBackend.CUSTOM:
            execution = current_gr_platform.get_attention_execution()
            if not execution.is_available():
                raise RuntimeError("GR attention implementation is unavailable")
            model = (
                runner.get_model() if callable(getattr(runner, "get_model", None)) else runner.model
            )
            layers = tuple(
                impl
                for layer in model.modules()
                if callable(
                    getattr(
                        impl := getattr(layer, "impl", None), "bind_gr_attention_execution", None
                    )
                )
            )
            for impl in layers:
                impl.validate_execution(execution)
            for impl in layers:
                impl.bind_gr_attention_execution(execution)
        runner._gr_loaded_config = gr
        runner._gr_attention_execution = execution
        runner._gr_attention_layers = layers
        runner._gr_pending_stage_sample = False
        runner._gr_ready = True
    except BaseException:
        runner._gr_attention_execution = execution
        runner._gr_attention_layers = layers
        try:
            close_gr_attention(runner)
        except Exception:
            logger.exception("Failed to release attention after model initialization failure")
        raise


def load_gr_model(original, *, close_runtime=None):
    """Initialize model-owned resources and unwind a failed load."""

    @wraps(original)
    def load(self, *args, **kwargs):
        if getattr(self, "_gr_pending_stage_sample", False) is not False:
            raise RuntimeError("GR sampling must consume the previous dispatch before model reload")
        if close_runtime is not None:
            close_runtime(self)
        close_gr_attention(self)
        expected_gr = GRConfig.from_vllm_config(self.vllm_config)
        try:
            result = original(self, *args, **kwargs)
            if GRConfig.from_vllm_config(self.vllm_config) != expected_gr:
                raise RuntimeError("GR configuration changed while loading the model")
            initialize_gr_attention(self)
            return result
        except BaseException:
            from vllm_gr.v1.worker.model_runner_patch import close_constraint_backend

            try:
                if close_runtime is not None:
                    close_runtime(self)
            except Exception:
                logger.exception("Failed to release runtime after model load failure")
            try:
                close_constraint_backend(self)
            except Exception:
                logger.exception("Failed to release constraints after model load failure")
            raise

    return load


def sample_gr_tokens(original):
    """Preserve the native BatchQueue execute/sample error boundary."""

    @wraps(original)
    def sample(self, *args, **kwargs):
        pending = getattr(self, "_gr_pending_stage_sample", False)
        if pending is False:
            return original(self, *args, **kwargs)
        self._gr_pending_stage_sample = False
        if isinstance(pending, Exception):
            raise pending
        return self.sample_gr_stage(*args, **kwargs)

    return sample


def require_gr_runner(runner):
    if not getattr(runner, "_gr_ready", False):
        raise RuntimeError("GR stage received before model initialization")
