# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Active-platform ModelRunner Beam patch dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger

from vllm_gr.config import ConstraintMode, ConstraintTableBackendEnum
from vllm_gr.config.models import runtime_constraint_table
from vllm_gr.platforms.interface import ModelRunnerInstaller, ModelRunnerPatchHooks
from vllm_gr.v1.worker.gr_context import (
    _get_gr_config,
    get_beam_graph_runtime_stats,
)
from vllm_gr.v1.worker.gr_gc import (
    freeze_gc_after_graph_capture,
    freeze_gc_after_warmup,
)
from vllm_gr.v1.worker.model_runner_contracts import (
    _CONSTRAINT_LOAD_PATCH_MARKER,
    _CONSTRAINT_SHUTDOWN_PATCH_MARKER,
    _WORKER_CAPTURE_PATCH_MARKER,
    _WORKER_DETERMINE_PATCH_MARKER,
    is_worker_runner_patched,
)

if TYPE_CHECKING:
    from vllm_gr.constraints.cuda_backend import CUDAConstraintBackend
    from vllm_gr.v1.worker.cuda.gpu_stage_hooks import CudaStageHooksType

logger = init_logger(__name__)

_BEAM_GRAPH_STATS_HOOK_MARKER = "_vllm_gr_beam_graph_stats_hook_installed"


def close_constraint_backend(runner: object) -> None:
    """Detach and idempotently close the runner-owned canonical Backend."""

    from vllm_gr.v1.worker.runner_binding import close_backend_binding

    close_backend_binding(runner)
    backend = getattr(runner, "constraint_backend", None)
    # Detach first so no concurrent observer can discover a Backend while it is
    # being closed, and so repeated shutdown/load calls are naturally idempotent.
    setattr(runner, "constraint_backend", None)
    projection = getattr(runner, "_gr_npu_stage_projection", None)
    if projection is not None:
        cast(Any, runner)._gr_npu_stage_projection = None
        projection.close()
    for processor in getattr(runner, "_gr_vocab_slice_processors", ()):
        processor._gr_vocab_slice = None
    setattr(runner, "_gr_vocab_slice_processors", ())
    if backend is not None:
        backend.close()


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
        gr_config = GRConfig.from_vllm_config(getattr(self, "vllm_config", None))
        table_config = runtime_constraint_table(gr_config)
        parallel = getattr(getattr(self, "vllm_config", None), "parallel_config", None)
        result = original(self, *args, **kwargs)

        from vllm_gr.v1.worker.lm_head_cache import install_lm_head_cache_reload_hooks

        install_lm_head_cache_reload_hooks(self.model)
        backend = None
        try:
            gr_config = GRConfig.from_vllm_config(getattr(self, "vllm_config", None))
            if gr_config.constraint_config.mode is ConstraintMode.NONE:
                return result
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
                table_config.backend is ConstraintTableBackendEnum.NPU
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

            if getattr(backend.info, "backend", None) is ConstraintTableBackendEnum.CUDA:
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
                getattr(backend.info, "backend", None) is ConstraintTableBackendEnum.NPU
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


def get_runner_hooks() -> ModelRunnerInstaller | ModelRunnerPatchHooks:
    """Use the process-local provider, also for a fresh EngineCore child."""
    from vllm_gr.platforms.errors import BackendNotActiveError
    from vllm_gr.platforms.loader import activate_current_backend
    from vllm_gr.platforms.registry import get_active_backend

    try:
        backend = get_active_backend()
    except BackendNotActiveError:
        backend = activate_current_backend()
    return backend.get_model_runner_hooks()


def discover_active_worker_runners() -> tuple[tuple[str, type], ...]:
    return cast(tuple[tuple[str, type], ...], get_runner_hooks().discover_worker_runners())


def is_active_worker_runner_patched(runner_class: type) -> bool:
    hooks = get_runner_hooks()
    if isinstance(hooks, ModelRunnerInstaller):
        return bool(hooks.is_patched(runner_class))
    return is_worker_runner_patched(runner_class)


def install_worker_utility_hooks(worker_cls: type) -> None:
    """Install named utility RPCs shared by CUDA and NPU Worker classes."""
    from vllm_gr.v1.worker.hot_update import install_worker_hot_update_hooks

    install_worker_hot_update_hooks(worker_cls)
    if not getattr(worker_cls, _BEAM_GRAPH_STATS_HOOK_MARKER, False):
        # Utility RPC dispatch resolves named Worker methods. Keep these
        # importable functions instead of relying on callable serialization.
        worker_cls.get_gr_beam_graph_runtime_stats = get_beam_graph_runtime_stats  # type: ignore[attr-defined]
        worker_cls.gr_freeze_gc_after_warmup = freeze_gc_after_warmup  # type: ignore[attr-defined]
        setattr(worker_cls, _BEAM_GRAPH_STATS_HOOK_MARKER, True)


def apply_worker_patches() -> None:
    """Install the shared async guard and the selected provider's runner hooks."""
    from vllm_gr.v1.worker.gr_async_output import install_async_output_guard

    install_async_output_guard()
    hooks = get_runner_hooks()
    if isinstance(hooks, ModelRunnerInstaller):
        hooks.install()
    else:
        # Existing providers can continue to supply V1 wrapper factories.
        install_v1_worker_hooks(hooks)


def install_v1_worker_hooks(
    platform_module: Any,
    *,
    cuda_hooks_type: CudaStageHooksType | None = None,
    workspace_load_patcher: Callable[..., Any] | None = None,
    constraint_max_batch: int = 1,
    implementation_id: str | None = None,
) -> None:
    """Apply the V1 wrapper factories to the native classes supplied by a provider."""
    from vllm_gr.v1.worker.model_runner_common import make_patched_execute_model

    if workspace_load_patcher is None:
        workspace_load_patcher = getattr(
            platform_module, "make_load_model_with_beam_postprocess_workspace", None
        )
    from vllm_gr.v1.worker.runner_binding import bind_after_model_load
    from vllm_gr.v1.worker.runner_lifecycle import install_runner_lifecycle

    for runner_name, runner_class in platform_module.discover_worker_runners():
        if is_worker_runner_patched(runner_class):
            if implementation_id is not None:
                install_runner_lifecycle(runner_class, implementation_id, prepare="_prepare_inputs")
            continue
        load_model = getattr(runner_class, "load_model", None)
        if callable(load_model) and not getattr(load_model, _CONSTRAINT_LOAD_PATCH_MARKER, False):
            runner_class.load_model = make_load_model_with_constraint_backend(
                load_model, max_batch=constraint_max_batch
            )

        platform_load_patcher = getattr(platform_module, "make_patched_load_model", None)
        if platform_load_patcher is not None and callable(
            getattr(runner_class, "load_model", None)
        ):
            runner_class.load_model = platform_load_patcher(runner_class.load_model)

        reload_patcher = getattr(platform_module, "make_patched_reload_weights", None)
        reload_weights = getattr(runner_class, "reload_weights", None)
        if reload_patcher is not None and callable(reload_weights):
            runner_class.reload_weights = reload_patcher(reload_weights)

        if workspace_load_patcher is not None and callable(
            getattr(runner_class, "load_model", None)
        ):
            runner_class.load_model = workspace_load_patcher(runner_class.load_model)

        shutdown = getattr(runner_class, "shutdown", None)
        if not getattr(shutdown, _CONSTRAINT_SHUTDOWN_PATCH_MARKER, False):
            runner_class.shutdown = make_shutdown_with_constraint_backend(
                shutdown if callable(shutdown) else None
            )

        platform_shutdown_patcher = getattr(platform_module, "make_patched_shutdown", None)
        if platform_shutdown_patcher is not None:
            runner_class.shutdown = platform_shutdown_patcher(runner_class.shutdown)

        runner_class.execute_model = make_patched_execute_model(
            runner_class.execute_model, hooks_type=cuda_hooks_type
        )
        update_states_patch = getattr(platform_module, "make_patched_update_states", None)
        if update_states_patch is not None:
            runner_class._update_states = update_states_patch(runner_class._update_states)
        runner_class._prepare_inputs = platform_module.make_patched_prepare_inputs(
            runner_class._prepare_inputs
        )
        # V1 stages own their results; ordinary inference retains native bookkeeping.
        runner_class._vllm_gr_native_bookkeeping = runner_class._bookkeeping_sync
        # Pure-prefill graph interception: replays captured prefill graphs for
        # prompt batches inside _model_forward (installed on both platforms;
        # the vLLM 0.30 runner classes expose this method).
        model_forward_patch = getattr(platform_module, "make_patched_model_forward", None)
        if model_forward_patch is not None and hasattr(runner_class, "_model_forward"):
            runner_class._model_forward = model_forward_patch(runner_class._model_forward)
        # Async V1 stages own sampling after execute_model returns.
        sample_tokens_patch = getattr(platform_module, "make_patched_sample_tokens", None)
        if sample_tokens_patch is not None and hasattr(runner_class, "sample_tokens"):
            runner_class.sample_tokens = sample_tokens_patch(runner_class.sample_tokens)

        # After native capture_model() finishes, run the startup dummy capture
        # of the beam-decode FULL graph (only when GRConfig enables it —
        # beam_max_width configured and not disabled; no-op otherwise).
        beam_capture = getattr(platform_module, "capture_beam_decode_graph", None)
        prefill_capture = getattr(platform_module, "capture_prefill_graphs", None)
        if (
            (beam_capture is not None or prefill_capture is not None)
            and hasattr(runner_class, "capture_model")
            and not getattr(runner_class.capture_model, _WORKER_CAPTURE_PATCH_MARKER, False)
        ):
            _orig_capture_model = runner_class.capture_model

            def make_patched_capture_model(
                _orig: Callable[..., Any],
                _beam_capture: Callable[..., Any] | None,
                _prefill_capture: Callable[..., Any] | None,
                hooks_type: CudaStageHooksType | None,
            ) -> Callable[..., Any]:
                @wraps(_orig)
                def patched_capture_model(self: Any, *args: Any, **kwargs: Any) -> Any:
                    res = _orig(self, *args, **kwargs)
                    from vllm_gr.v1.worker.gr_startup import capture_async_forward

                    if _get_gr_config(self).beam_max_width is not None:
                        capture_async_forward(self, hooks_type=hooks_type)
                    elif _beam_capture is not None:
                        try:
                            _beam_capture(self)
                        except Exception as error:
                            from vllm_gr.v1.worker.cudagraph_capture import (
                                is_recoverable_capture_error,
                            )

                            if not isinstance(
                                error, RuntimeError
                            ) or not is_recoverable_capture_error(error):
                                raise
                            logger.exception(
                                "vllm-gr: beam decode graph dummy capture failed; "
                                "beam decode will fall back to eager"
                            )
                    if _prefill_capture is not None:
                        try:
                            _prefill_capture(self)
                        except Exception as error:
                            from vllm_gr.v1.worker.cudagraph_capture import (
                                is_recoverable_capture_error,
                            )

                            if not isinstance(
                                error, RuntimeError
                            ) or not is_recoverable_capture_error(error):
                                raise
                            logger.exception(
                                "vllm-gr: prefill graph capture failed; "
                                "prefill will fall back to eager"
                            )
                    # All CUDA graphs (native + beam + prefill) are now captured
                    # and the serving-time heap is fully built.  Freeze it so the
                    # long-lived post-startup objects (KV caches, captured graph
                    # state, beam sessions) are not rescanned by gen-1/gen-2 GC
                    # during serving (measured 29-36ms collected=0 pauses).
                    freeze_gc_after_graph_capture(self)
                    return res

                return patched_capture_model

            patched = make_patched_capture_model(
                _orig_capture_model, beam_capture, prefill_capture, cuda_hooks_type
            )
            setattr(patched, _WORKER_CAPTURE_PATCH_MARKER, True)
            runner_class.capture_model = patched

        # Patch _determine_batch_execution_and_padding to force FULL CUDA Graph
        # for beam decode steps (if the platform module provides it).
        determine_patcher = getattr(
            platform_module, "make_patched_determine_batch_execution_and_padding", None
        )
        if determine_patcher is not None and hasattr(
            runner_class, "_determine_batch_execution_and_padding"
        ):
            if not getattr(
                runner_class._determine_batch_execution_and_padding,
                _WORKER_DETERMINE_PATCH_MARKER,
                False,
            ):
                runner_class._determine_batch_execution_and_padding = determine_patcher(
                    runner_class._determine_batch_execution_and_padding
                )

        if callable(getattr(runner_class, "load_model", None)):
            runner_class.load_model = bind_after_model_load(runner_class.load_model)
        if implementation_id is not None:
            install_runner_lifecycle(runner_class, implementation_id, prepare="_prepare_inputs")
        runner_class._patched_for_beam_pos_ids = True
        logger.info("%s hooks patched successfully.", runner_name)


def validate_model_runner_patch() -> str | None:
    runners = discover_active_worker_runners()
    if not runners:
        return "No supported active-platform ModelRunner is available"

    failures = [
        f"{runner_name} beam execution hooks are not installed"
        for runner_name, runner_class in runners
        if not is_active_worker_runner_patched(runner_class)
    ]
    return "; ".join(failures) or None
