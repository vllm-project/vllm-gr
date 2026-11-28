# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared model-owned constraint resources for GR ModelRunner subclasses."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger

from vllm_gr.config import ConstraintBackendKind
from vllm_gr.config.models import runtime_constraint_table
from vllm_gr.v1.worker.model_runner_contracts import (
    _CONSTRAINT_LOAD_PATCH_MARKER,
    _CONSTRAINT_SHUTDOWN_PATCH_MARKER,
)

if TYPE_CHECKING:
    from vllm_gr.constraints.cuda_backend import CUDAConstraintBackend

logger = init_logger(__name__)


def close_constraint_backend(runner: object) -> None:
    """Detach and idempotently close the runner-owned canonical Backend."""

    from vllm_gr.v1.worker.runner import close_gr_attention

    errors: list[Exception] = []
    try:
        close_gr_attention(runner)
    except Exception as exc:
        errors.append(exc)
    backend = getattr(runner, "constraint_backend", None)
    setattr(runner, "constraint_backend", None)
    projection = getattr(runner, "_gr_npu_stage_projection", None)
    if projection is not None:
        cast(Any, runner)._gr_npu_stage_projection = None
    for processor in getattr(runner, "_gr_vocab_slice_processors", ()):
        processor._gr_vocab_slice = None
    setattr(runner, "_gr_vocab_slice_processors", ())
    # One failing vendor close must not keep the other resources reachable.
    for resource in (projection, backend):
        if resource is not None:
            try:
                resource.close()
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise errors[0]


def _remove_model_bytecode_hooks(runner: object) -> None:
    """Release native compiler callbacks owned by this runner's model."""
    model = getattr(runner, "model", None)
    if model is None:
        return

    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    # vLLM 0.30 keeps each Dynamo RemovableHandle on its compile wrapper and
    # exposes cleanup(). GPUModelRunner.shutdown drops self.model without
    # calling it, so release the callbacks before native shutdown. The model's
    # modules() traversal also works through CUDAGraphWrapper.
    for module in model.modules():
        if isinstance(module, TorchCompileWithNoGuardsWrapper):
            module.cleanup()


def _runner_vocab_size(runner: object) -> int:
    """Read the effective model vocabulary through the upstream public API."""

    model_config = getattr(runner, "model_config", None)
    get_vocab_size = getattr(model_config, "get_vocab_size", None)
    vocab_size = (
        get_vocab_size() if callable(get_vocab_size) else getattr(model_config, "vocab_size", None)
    )
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise RuntimeError(
            f"constraint backend requires a positive model vocab_size, got {vocab_size!r}"
        )
    return cast(int, vocab_size)


def make_load_model_with_constraint_backend(
    original: Any,
    *,
    max_batch: int = 1,
    close_runtime: Any = None,
    prepare_backend: Callable[[Any, Any], None] | None = None,
) -> Any:
    """Load weights first, then atomically publish a ready canonical Backend."""

    if getattr(original, _CONSTRAINT_LOAD_PATCH_MARKER, False):
        return original

    @wraps(original)
    def load_model_with_constraint_backend(self: Any, *args: Any, **kwargs: Any) -> Any:
        from vllm_gr.config import GRConfig

        # A complete load replaces the model runtime. Release the previous
        # Backend before loading weights so a failed reload cannot leave stale
        # canonical tables reachable from this runner.
        if close_runtime is not None:
            close_runtime(self)
        close_constraint_backend(self)
        # Validate the deployment before the native loader allocates model weights.
        GRConfig.from_vllm_config(getattr(self, "vllm_config", None))
        parallel = getattr(getattr(self, "vllm_config", None), "parallel_config", None)
        result = original(self, *args, **kwargs)

        from vllm_gr.v1.worker.lm_head_cache import install_lm_head_cache_reload_hooks

        install_lm_head_cache_reload_hooks(self.model)
        backend = None
        try:
            gr_config = GRConfig.from_vllm_config(getattr(self, "vllm_config", None))
            table_config = runtime_constraint_table(gr_config)
            if table_config is None or not table_config.enabled:
                return result

            max_rows = gr_config.beam_max_width
            if max_rows is None:
                raise ValueError(
                    "canonical constraint_table.enabled=true requires "
                    "beam_max_width to be configured"
                )
            # Rows are bounded by the beam width; each row's candidate count is
            # the independent candidate width K, which is what the table's
            # top_k bound actually limits.
            from vllm_gr.config import resolve_candidate_width

            candidate_width = resolve_candidate_width(gr_config)
            max_top_k = table_config.max_top_k
            assert max_top_k is not None
            if candidate_width > max_top_k:
                raise ValueError(
                    "candidate_width must be <= constraint_table.max_top_k "
                    f"for canonical constraint sampling, got {candidate_width} > "
                    f"{table_config.max_top_k}"
                )

            from vllm_gr.constraints.loader import load_constraint_backend

            stage_options = {}
            if (
                table_config.backend is ConstraintBackendKind.NPU
                and gr_config.beam_max_width is not None
                and gr_config.beam_max_decode_steps <= 3
            ):
                from vllm_gr.v1.worker.npu.stage_projection import (
                    can_install_npu_stage_projection,
                )

                # Check the loaded model before allocating compact tables.
                if can_install_npu_stage_projection(self):
                    stage_options["enable_npu_stage_vocab"] = True
                    stage_options["stage_lm_head"] = gr_config.deployment.npu_stage_lm_head
            backend = load_constraint_backend(
                config=table_config,
                device=getattr(self, "device", None),
                vocab_size=_runner_vocab_size(self),
                dtype=getattr(getattr(self, "model_config", None), "dtype", None),
                # NPU V1 and CUDA V2 opt into batched sampling.
                max_rows=max_batch * max_rows,
                **stage_options,
            )
            if backend is None:
                raise RuntimeError("enabled canonical constraint table loader returned no Backend")

            model_memory_usage = int(getattr(self, "model_memory_usage", 0))
            self.model_memory_usage = model_memory_usage + backend.info.device_memory_bytes

            if getattr(backend.info, "backend", None) is ConstraintBackendKind.CUDA:
                # Import GPU layers only after selecting the CUDA backend.
                from vllm.model_executor.layers.logits_processor import LogitsProcessor

                cuda_backend = cast("CUDAConstraintBackend", backend)
                window = cuda_backend.vocab_slice
                if getattr(parallel, "tensor_parallel_size", 1) != 1:
                    # All ranks must retain the original full-logits collective path.
                    window = None
                    cuda_backend.vocab_slice = None
                    logger.info(
                        "vllm-gr: canonical CUDA vocabulary slicing disabled for TP>1; "
                        "using full-vocabulary projection"
                    )
                if window == (0, _runner_vocab_size(self)):
                    window = None  # No benefit from cloning the entire LM head.
                processors = tuple(
                    module for module in self.model.modules() if isinstance(module, LogitsProcessor)
                )
                for processor in processors:
                    # Explicit None also overrides unrelated legacy global settings.
                    processor._gr_vocab_slice = window
                self._gr_vocab_slice_processors = processors
            elif (
                getattr(backend.info, "backend", None) is ConstraintBackendKind.NPU
                and getattr(backend, "stage_vocab", None) is not None
            ):
                # Keep NPU projection imports isolated from CPU/CUDA startup.
                from vllm_gr.v1.worker.npu.stage_projection import (
                    install_npu_stage_projection,
                )

                projection = install_npu_stage_projection(self, backend)
                if projection is not None:
                    self._gr_npu_stage_projection = projection
                    self.model_memory_usage += projection.device_memory_bytes
            from vllm_gr.v1.worker.hot_update import versioned_constraint_table

            # Publish only after readiness and runner-specific kernel warmup.
            self.constraint_backend = versioned_constraint_table(
                self, backend, prepare_backend=prepare_backend
            )
            return result
        except BaseException:
            projection = getattr(self, "_gr_npu_stage_projection", None)
            if projection is not None:
                self._gr_npu_stage_projection = None
                projection.close()
            if backend is not None:
                backend.close()
            self.constraint_backend = None
            raise

    setattr(load_model_with_constraint_backend, _CONSTRAINT_LOAD_PATCH_MARKER, True)
    return load_model_with_constraint_backend


def make_shutdown_with_constraint_backend(
    original: Any = None, *, close_runtime: Any = None
) -> Any:
    """Release model callbacks and Backend before native runner shutdown."""

    if original is not None and getattr(original, _CONSTRAINT_SHUTDOWN_PATCH_MARKER, False):
        return original

    if original is None:

        def shutdown_with_constraint_backend(self: Any, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if close_runtime is not None:
                close_runtime(self)
            _remove_model_bytecode_hooks(self)
            close_constraint_backend(self)

    else:

        @wraps(original)
        def shutdown_with_constraint_backend(self: Any, *args: Any, **kwargs: Any) -> Any:
            if close_runtime is not None:
                close_runtime(self)
            _remove_model_bytecode_hooks(self)
            close_constraint_backend(self)
            return original(self, *args, **kwargs)

    setattr(
        shutdown_with_constraint_backend,
        _CONSTRAINT_SHUTDOWN_PATCH_MARKER,
        True,
    )
    return shutdown_with_constraint_backend
