# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Runner-local vocabulary projection for CUDA Beam stages.

Ordinary requests must see the full vocabulary. Only an admitted GR stage may
use the catalog window and its original constrained scoring semantics. That
distinction is a task-local execution scope rather than a process-global
processor attribute, so ordinary generation in the same process -- including
the thinking phase of a CoT SID Beam request -- keeps full-vocabulary logits.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch

from vllm_gr.ops.sliced_logits import (
    compute_sliced_logits,
    get_rank_vocab_bounds,
    is_supported_unquantized_lm_head,
)

_MISSING = object()


def _cached_bytes(head: Any) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in (
            getattr(head, "_sliced_weight", None),
            getattr(head, "_sliced_bias", None),
        )
        if isinstance(value, torch.Tensor)
    )


class GPUStageProjection:
    """Own per-processor hooks and a task-local Beam execution scope."""

    def __init__(self, processors: tuple[Any, ...], window: tuple[int, int]) -> None:
        self.processors = processors
        self.window = window
        self._active: ContextVar[bool] = ContextVar("vllm_gr_gpu_beam_projection", default=False)
        self._reload_active: ContextVar[bool] = ContextVar("vllm_gr_gpu_reload", default=False)
        self._previous: list[tuple[Any, Any, Any]] = []
        self.closed = False
        self.beam_projections = 0
        self.full_projections = 0
        self.cached_projections = 0
        self.fallback_projections = 0
        self._cached_head: Any = None
        self._source_weight: Any = None
        self._source_bias: Any = None
        self._weight_version: int | None = None
        self._bias_version: int | None = None
        self.update_active = False
        self.update_failed = False
        self.cached_bytes = 0
        self._prime_bytes = 0

    def install(self) -> None:
        if self.closed or self._previous:
            raise RuntimeError("CUDA stage projection is already installed or closed")
        if not self.processors:
            raise ValueError("CUDA Beam stage projection requires a logits processor")
        try:
            for processor in self.processors:
                if getattr(processor, "logits_as_input", False):
                    raise ValueError("CUDA Beam stage projection requires an LM-head projection")
                previous_method = processor.__dict__.get("_get_logits", _MISSING)
                previous_window = processor.__dict__.get("_gr_vocab_slice", _MISSING)
                original = processor._get_logits
                self._previous.append((processor, previous_method, previous_window))

                def stage_get_logits(
                    hidden_states: Any,
                    lm_head: Any,
                    embedding_bias: Any,
                    skip_gather: bool = False,
                    *,
                    _processor: Any = processor,
                    _original: Any = original,
                ) -> Any:
                    if self.update_active or self.update_failed:
                        raise RuntimeError(
                            "CUDA Beam projection cannot run during or after a failed weight update"
                        )
                    if not self._active.get():
                        self.full_projections += 1
                        return _original(hidden_states, lm_head, embedding_bias, skip_gather)
                    self.beam_projections += 1

                    if not self._cache_usable(lm_head, embedding_bias):
                        if skip_gather:
                            try:
                                rank_start, _ = get_rank_vocab_bounds(lm_head)
                            except (AttributeError, TypeError):
                                rank_start = -1
                            if getattr(lm_head, "tp_size", 1) != 1 or rank_start != 0:
                                raise RuntimeError(
                                    "CUDA stage fallback requires a full vocabulary before slicing"
                                )
                        self.fallback_projections += 1
                        # Unknown heads and changed weights retain the original
                        # Beam window semantics without using a stale clone.
                        logits = _original(hidden_states, lm_head, embedding_bias, skip_gather)
                        if logits is None:
                            return None
                        start, size = self.window
                        return logits[..., start : start + size].contiguous()

                    def original_get_logits(
                        _unused: Any, hidden: Any, head: Any, bias: Any, skip_gather: bool
                    ) -> Any:
                        return _original(hidden, head, bias, skip_gather)

                    self.cached_projections += 1
                    return compute_sliced_logits(
                        hidden_states,
                        lm_head,
                        embedding_bias,
                        offset=self.window[0],
                        slice_size=self.window[1],
                        logits_processor=_processor,
                        orig_get_logits=original_get_logits,
                        skip_gather=skip_gather,
                    )

                # Override legacy process-global slice for ordinary generation.
                processor._gr_vocab_slice = None
                processor._get_logits = stage_get_logits
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _version(tensor: Any) -> int | None:
        if tensor is None:
            return None
        try:
            return int(tensor._version)
        except RuntimeError:  # inference tensors have no version counter
            return None

    def _cache_usable(self, head: Any, bias: Any) -> bool:
        weight = getattr(head, "weight", None)
        return (
            head is self._cached_head
            and weight is self._source_weight
            and bias is self._source_bias
            and (self._weight_version is None or self._version(weight) == self._weight_version)
            and (self._bias_version is None or self._version(bias) == self._bias_version)
        )

    def _record_head(self, head: Any) -> None:
        """Adopt ``head`` as the projection source and drop its derived cache."""
        self._clear_cache(head)
        weight = getattr(head, "weight", None)
        bias = getattr(head, "bias", None)
        self._cached_head, self._source_weight, self._source_bias = head, weight, bias
        self._weight_version, self._bias_version = self._version(weight), self._version(bias)

    def _release_head(self) -> None:
        self._clear_cache(self._cached_head)
        self._cached_head = None
        self._source_weight = None
        self._source_bias = None
        self._weight_version = None
        self._bias_version = None
        self.cached_bytes = 0

    def prime(self, runner: Any) -> None:
        """Allocate the standard head's cached window before KV profiling.

        Unknown or quantized heads use a full projection followed by slicing.
        """
        model = getattr(runner, "model", None)
        head = getattr(model, "lm_head", None)
        weight = getattr(head, "weight", None)
        if (
            len(self.processors) != 1
            or not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or weight.device.type != "cuda"
            or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or getattr(head, "tp_size", 1) != 1
            or not callable(getattr(getattr(head, "quant_method", None), "apply", None))
            or not is_supported_unquantized_lm_head(head)
        ):
            return
        self._record_head(head)
        with torch.inference_mode(), self.beam_scope():
            hidden = torch.zeros((1, weight.shape[1]), device=weight.device, dtype=weight.dtype)
            self.processors[0]._get_logits(hidden, head, getattr(head, "bias", None))
        self.beam_projections = 0
        self.cached_projections = 0
        # The cache exists only after the priming projection populated it.
        self._prime_bytes = self.cached_bytes = _cached_bytes(head)

    def begin_weight_update(self) -> None:
        if self.closed or self.update_active or self.update_failed:
            raise RuntimeError("CUDA stage projection cannot start another weight update")
        self.update_active = True
        # The native loader must not observe the derived cache, and a clone made
        # before the update would score the next Beam stage from stale weights.
        self._release_head()

    def finish_weight_update(self, head: Any = None) -> None:
        if not self.update_active or self.update_failed:
            raise RuntimeError("CUDA stage projection has no active weight update")
        self.update_active = False
        # A native update can mutate inference tensors without version counters.
        # Adopting the head again keeps the identity check valid while forcing
        # the next Beam stage to re-clone from the updated parameter instead of
        # reusing pre-update weights.
        if head is not None:
            self._record_head(head)
            # The retired cache is already counted against the profiled budget,
            # so keep that residency until the next stage rebuilds it.
            self.cached_bytes = self._prime_bytes
        else:
            self._release_head()

    def fail_weight_update(self) -> None:
        self.update_failed = True

    @staticmethod
    def _clear_cache(head: Any) -> None:
        if head is None:
            return
        for name in (
            "_sliced_weight",
            "_sliced_bias",
            "_sliced_cache_key",
            "_sliced_rel_start",
            "_sliced_rel_end",
            "_sliced_need_crop",
            "_sliced_proxy",
        ):
            if hasattr(head, name):
                delattr(head, name)

    @contextmanager
    def beam_scope(self) -> Iterator[None]:
        if self.closed:
            raise RuntimeError("CUDA stage projection is closed")
        if self._active.get():
            raise RuntimeError("Nested CUDA Beam projection scopes are unsupported")
        token = self._active.set(True)
        try:
            yield
        finally:
            self._active.reset(token)

    def stats(self) -> dict[str, Any]:
        return {
            "full": self.full_projections,
            "beam_slice": self.beam_projections,
            "beam_cached": self.cached_projections,
            "beam_fallback": self.fallback_projections,
        }

    def close(self) -> None:
        for processor, method, window in reversed(self._previous):
            if method is _MISSING:
                delattr(processor, "_get_logits")
            else:
                processor._get_logits = method
            if window is _MISSING:
                delattr(processor, "_gr_vocab_slice")
            else:
                processor._gr_vocab_slice = window
        self._previous.clear()
        self._release_head()
        self.closed = True


def install_gpu_stage_projection(
    processors: tuple[Any, ...], window: tuple[int, int]
) -> GPUStageProjection:
    projection = GPUStageProjection(processors, window)
    projection.install()
    return projection


def _projection_head(runner: Any) -> Any:
    return getattr(getattr(runner, "model", None), "lm_head", None)


def make_gpu_stage_reload_weights(original: Any) -> Any:
    """Protect cached slices for both direct and Worker-delegated reloads."""
    if getattr(original, "_gr_gpu_stage_reload", False):
        return original

    @wraps(original)
    def reload_weights(runner: Any, *args: Any, **kwargs: Any) -> Any:
        projection = getattr(runner, "_gr_gpu_stage_projection", None)
        if projection is None:
            return original(runner, *args, **kwargs)
        # V2 delegates its native implementation to V1. If both classes have
        # been patched in this process, only the outer wrapper owns the update.
        if projection._reload_active.get():
            return original(runner, *args, **kwargs)
        projection.begin_weight_update()
        token = projection._reload_active.set(True)
        try:
            result = original(runner, *args, **kwargs)
            projection.finish_weight_update(_projection_head(runner))
            return result
        except BaseException:
            projection.fail_weight_update()
            raise
        finally:
            projection._reload_active.reset(token)

    setattr(reload_weights, "_gr_gpu_stage_reload", True)
    return reload_weights


def install_gpu_weight_update_hooks(worker_cls: type[Any]) -> None:
    """Invalidate cloned heads for the native Worker weight-update APIs."""
    for name in (
        "start_weight_update",
        "start_draft_weight_update",
        "update_weights",
        "finish_weight_update",
    ):
        original = getattr(worker_cls, name, None)
        if not callable(original) or getattr(original, "_gr_gpu_stage_update", False):
            continue

        def make_wrapper(method: Any, method_name: str) -> Any:
            @wraps(method)
            def update(worker: Any, *args: Any, **kwargs: Any) -> Any:
                runner = getattr(worker, "model_runner", None)
                projection = getattr(runner, "_gr_gpu_stage_projection", None)
                if projection is None:
                    return method(worker, *args, **kwargs)
                if method_name in ("start_weight_update", "start_draft_weight_update"):
                    projection.begin_weight_update()
                elif not projection.update_active or projection.update_failed:
                    raise RuntimeError("CUDA stage projection has no active weight update")
                try:
                    result = method(worker, *args, **kwargs)
                    if method_name == "finish_weight_update":
                        projection.finish_weight_update(_projection_head(runner))
                    return result
                except BaseException:
                    projection.fail_weight_update()
                    raise

            setattr(update, "_gr_gpu_stage_update", True)
            return update

        setattr(worker_cls, name, make_wrapper(original, name))
