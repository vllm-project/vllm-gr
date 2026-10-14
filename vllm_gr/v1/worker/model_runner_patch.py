# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Active-platform ModelRunner Beam patch dispatcher."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from types import ModuleType

from vllm.logger import init_logger

from vllm_gr.config import ConstraintTableBackendEnum
from vllm_gr.v1.worker.model_runner_common import (
    _CONSTRAINT_LOAD_PATCH_MARKER,
    _CONSTRAINT_SHUTDOWN_PATCH_MARKER,
    _WORKER_CAPTURE_PATCH_MARKER,
    _WORKER_DETERMINE_PATCH_MARKER,
    freeze_gc_after_graph_capture,
    freeze_gc_after_warmup,
    get_beam_graph_runtime_stats,
    is_worker_runner_patched,
    make_patched_execute_model,
)

logger = init_logger(__name__)

_GPU_PATCH_MODULE = "vllm_gr.v1.worker.gpu_model_runner_patch"
_NPU_PATCH_MODULE = "vllm_gr.v1.worker.npu_model_runner_patch"
_V2_GPU_PATCH_MODULE = "vllm_gr.v1.worker.gpu_model_runner_v2_patch"
_BEAM_GRAPH_STATS_HOOK_MARKER = "_vllm_gr_beam_graph_stats_hook_installed"


def close_constraint_backend(runner: object) -> None:
    """Detach and idempotently close the runner-owned canonical Backend."""

    backend = getattr(runner, "constraint_backend", None)
    # Detach first so no concurrent observer can discover a Backend while it is
    # being closed, and so repeated shutdown/load calls are naturally idempotent.
    runner.constraint_backend = None
    for processor in getattr(runner, "_gr_vocab_slice_processors", ()):
        processor._gr_vocab_slice = None
    runner._gr_vocab_slice_processors = ()
    if backend is not None:
        backend.close()


def _remove_model_bytecode_hooks(runner: object) -> None:
    """Release native compiler callbacks owned by this runner's model."""
    model = getattr(runner, "model", None)
    if model is None:
        return

    from torch._dynamo.convert_frame import _bytecode_hooks
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    # vLLM 0.22 registers bound methods but discards their RemovableHandles.
    # The global registry then keeps the model/weights alive after shutdown,
    # even after gc.unfreeze()/collect(). AOT loading can register duplicates.
    # CUDAGraphWrapper forwards modules() to the underlying model. Remove all
    # native hooks for those modules before native shutdown drops self.model;
    # hooks for other engines or unrelated callbacks must remain registered.
    module_ids = {id(module) for module in model.modules()}
    for key, hook in tuple(_bytecode_hooks.items()):
        if (
            id(getattr(hook, "__self__", None)) in module_ids
            and getattr(hook, "__func__", None) is TorchCompileWithNoGuardsWrapper.bytecode_hook
        ):
            _bytecode_hooks.pop(key, None)


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
    return vocab_size


def make_load_model_with_constraint_backend(original, *, max_batch: int = 1, close_runtime=None):
    """Load weights first, then atomically publish a ready canonical Backend."""

    if getattr(original, _CONSTRAINT_LOAD_PATCH_MARKER, False):
        return original

    @wraps(original)
    def load_model_with_constraint_backend(self, *args, **kwargs):
        from vllm_gr.config import GRConfig

        # A complete load replaces the model runtime. Release the previous
        # Backend before loading weights so a failed reload cannot leave stale
        # canonical tables reachable from this runner.
        if close_runtime is not None:
            close_runtime(self)
        close_constraint_backend(self)
        gr_config = GRConfig.from_vllm_config(getattr(self, "vllm_config", None))
        table_config = gr_config.constraint_table
        parallel = getattr(getattr(self, "vllm_config", None), "parallel_config", None)
        result = original(self, *args, **kwargs)

        backend = None
        try:
            if table_config is None or not table_config.enabled:
                return result

            max_rows = gr_config.beam_max_width
            if max_rows is None:
                raise ValueError(
                    "canonical constraint_table.enabled=true requires "
                    "beam_max_width to be configured"
                )
            if max_rows > table_config.max_top_k:
                raise ValueError(
                    "beam_max_width must be <= constraint_table.max_top_k "
                    f"for canonical constraint sampling, got {max_rows} > "
                    f"{table_config.max_top_k}"
                )

            from vllm_gr.constraints.loader import load_constraint_backend

            backend = load_constraint_backend(
                config=table_config,
                device=getattr(self, "device", None),
                vocab_size=_runner_vocab_size(self),
                dtype=getattr(getattr(self, "model_config", None), "dtype", None),
                # V1 retains its row bound; V2 opts into batched sampling.
                max_rows=max_batch * max_rows,
            )
            if backend is None:
                raise RuntimeError("enabled canonical constraint table loader returned no Backend")

            model_memory_usage = int(getattr(self, "model_memory_usage", 0))
            self.model_memory_usage = model_memory_usage + backend.info.device_memory_bytes

            # Publication is the atomic boundary. The loader returns an NPU
            # Backend only after readiness and returns an immediately usable CPU
            # Backend, so requests never observe a partially initialized object.
            self.constraint_backend = backend
            if getattr(backend.info, "backend", None) is ConstraintTableBackendEnum.CUDA:
                # Import GPU layers only after selecting the CUDA backend.
                from vllm.model_executor.layers.logits_processor import LogitsProcessor

                window = backend.vocab_slice
                if getattr(parallel, "tensor_parallel_size", 1) != 1:
                    # All ranks must retain the original full-logits collective path.
                    window = None
                    backend.vocab_slice = None
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
            return result
        except BaseException:
            if backend is not None:
                backend.close()
            self.constraint_backend = None
            raise

    setattr(load_model_with_constraint_backend, _CONSTRAINT_LOAD_PATCH_MARKER, True)
    return load_model_with_constraint_backend


def make_shutdown_with_constraint_backend(original=None, *, close_runtime=None):
    """Release model callbacks and Backend before native runner shutdown."""

    if original is not None and getattr(original, _CONSTRAINT_SHUTDOWN_PATCH_MARKER, False):
        return original

    if original is None:

        def shutdown_with_constraint_backend(self, *args, **kwargs):
            del args, kwargs
            _remove_model_bytecode_hooks(self)
            if close_runtime is not None:
                close_runtime(self)
            close_constraint_backend(self)

    else:

        @wraps(original)
        def shutdown_with_constraint_backend(self, *args, **kwargs):
            _remove_model_bytecode_hooks(self)
            if close_runtime is not None:
                close_runtime(self)
            close_constraint_backend(self)
            return original(self, *args, **kwargs)

    setattr(
        shutdown_with_constraint_backend,
        _CONSTRAINT_SHUTDOWN_PATCH_MARKER,
        True,
    )
    return shutdown_with_constraint_backend


def _active_platform_module_name() -> str:
    """Return the patch module for the platform selected by vLLM."""
    from vllm.platforms import current_platform

    if str(current_platform.device_type).lower() == "npu":
        return _NPU_PATCH_MODULE
    return _GPU_PATCH_MODULE


def _active_runner_module_name() -> str:
    """Select the platform first; only CUDA can opt into the V2 patches."""
    from vllm.platforms import current_platform

    platform_module = _active_platform_module_name()
    if str(current_platform.device_type).lower() == "cuda" and _use_v2_model_runner():
        return _V2_GPU_PATCH_MODULE
    return platform_module


def _use_v2_model_runner() -> bool:
    """Read the V2 opt-in at patch installation, never during execution."""
    from vllm.envs import VLLM_USE_V2_MODEL_RUNNER

    return bool(VLLM_USE_V2_MODEL_RUNNER)


def _get_active_platform_module() -> ModuleType:
    return import_module(_active_runner_module_name())


def discover_active_worker_runners() -> tuple[tuple[str, type], ...]:
    """Discover ModelRunner targets for only the active platform."""
    return _get_active_platform_module().discover_worker_runners()


def install_worker_utility_hooks(worker_cls: type) -> None:
    """Install named utility RPCs shared by CUDA and NPU Worker classes."""
    from vllm_gr.constraints.worker_inject import install_worker_resource_hook

    install_worker_resource_hook(worker_cls)
    if not getattr(worker_cls, _BEAM_GRAPH_STATS_HOOK_MARKER, False):
        # Utility RPC dispatch resolves named Worker methods. Keep these
        # importable functions instead of relying on callable serialization.
        worker_cls.get_gr_beam_graph_runtime_stats = get_beam_graph_runtime_stats  # type: ignore[attr-defined]
        worker_cls.gr_freeze_gc_after_warmup = freeze_gc_after_warmup  # type: ignore[attr-defined]
        setattr(worker_cls, _BEAM_GRAPH_STATS_HOOK_MARKER, True)


def _install_active_worker_resource_hook() -> None:
    """Mount named vllm-gr utility RPCs on the active CUDA Worker class."""

    if _active_platform_module_name() != _GPU_PATCH_MODULE:
        # NPUWorker installs this hook from its delayed init_device wrapper in
        # ascend_compat, after vllm-ascend has finished resolving its class.
        return
    from vllm.v1.worker.gpu_worker import Worker

    install_worker_utility_hooks(Worker)


def apply_worker_patches() -> None:
    """Install grouped-Beam hooks on active-platform ModelRunner targets."""
    from vllm_gr.v1.worker.gr_async_output import install_async_output_guard

    install_async_output_guard()
    logger.info("vllm-gr: apply_worker_patches called")
    # collective_rpc targets the Worker, not its ModelRunner.  Install this
    # even when all runner hooks were already present (idempotent) so canonical
    # Constraint Tables can be injected in both uniproc and multiproc engines.
    _install_active_worker_resource_hook()

    if _active_runner_module_name() == _V2_GPU_PATCH_MODULE:
        # V2 runner exposes ``prepare_inputs``/``sample``/``sample_tokens`` and
        # no ``_prepare_inputs``/``_bookkeeping_sync``; the V1 body below does
        # unguarded attribute access that V2 lacks, so dispatch to the
        # V2-specific installer instead.
        from vllm_gr.v1.worker import gpu_model_runner_v2_patch as v2

        v2._apply_v2_runner_patches()
        return

    platform_module = _get_active_platform_module()
    runners_to_patch = [
        (runner_name, runner_class)
        for runner_name, runner_class in platform_module.discover_worker_runners()
        if not is_worker_runner_patched(runner_class)
    ]

    for runner_name, runner_class in runners_to_patch:
        load_model = getattr(runner_class, "load_model", None)
        if callable(load_model) and not getattr(load_model, _CONSTRAINT_LOAD_PATCH_MARKER, False):
            runner_class.load_model = make_load_model_with_constraint_backend(load_model)

        workspace_load_patcher = getattr(
            platform_module,
            "make_load_model_with_beam_postprocess_workspace",
            None,
        )
        if workspace_load_patcher is not None and callable(
            getattr(runner_class, "load_model", None)
        ):
            runner_class.load_model = workspace_load_patcher(runner_class.load_model)

        shutdown = getattr(runner_class, "shutdown", None)
        if not getattr(shutdown, _CONSTRAINT_SHUTDOWN_PATCH_MARKER, False):
            runner_class.shutdown = make_shutdown_with_constraint_backend(
                shutdown if callable(shutdown) else None
            )

        runner_class.execute_model = make_patched_execute_model(runner_class.execute_model)
        update_states_patch = getattr(platform_module, "make_patched_update_states", None)
        if update_states_patch is not None:
            runner_class._update_states = update_states_patch(runner_class._update_states)
        runner_class._prepare_inputs = platform_module.make_patched_prepare_inputs(
            runner_class._prepare_inputs
        )
        runner_class._bookkeeping_sync = platform_module.make_patched_bookkeeping_sync(
            runner_class._bookkeeping_sync
        )
        # Pure-prefill graph interception: replays captured prefill graphs for
        # prompt batches inside _model_forward (installed on both platforms;
        # the runner classes expose _model_forward on vllm 0.22.1).
        model_forward_patch = getattr(platform_module, "make_patched_model_forward", None)
        if model_forward_patch is not None and hasattr(runner_class, "_model_forward"):
            runner_class._model_forward = model_forward_patch(runner_class._model_forward)
        # Constraint Tables raw-logits hook: only where the runner exposes
        # ``_sample`` (a vllm-ascend build may not).  A Constraint Table worker
        # without the hook fails at injection (worker_inject), never here.
        if hasattr(runner_class, "_sample"):
            runner_class._sample = platform_module.make_patched_sample(runner_class._sample)

        # Async scheduling: sampling (and the worker-side beam decision that
        # runs inside the patched ``_sample``) happens in ``sample_tokens``
        # after ``execute_model`` returns.  Attach the decision to the output
        # there.  NPU-only hook (the platform module provides it).
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

            def make_patched_capture_model(_orig, _beam_capture, _prefill_capture):
                @wraps(_orig)
                def patched_capture_model(self, *args, **kwargs):
                    res = _orig(self, *args, **kwargs)
                    from vllm_gr.config import BeamExecutionMode
                    from vllm_gr.v1.worker.gr_startup import capture_async_forward
                    from vllm_gr.v1.worker.model_runner_common import _get_gr_config

                    if _get_gr_config(self).beam_execution_mode is BeamExecutionMode.V1:
                        capture_async_forward(self)
                    elif _beam_capture is not None:
                        try:
                            _beam_capture(self)
                        except Exception:
                            logger.exception(
                                "vllm-gr: beam decode graph dummy capture failed; "
                                "beam decode will fall back to eager"
                            )
                    if _prefill_capture is not None:
                        try:
                            _prefill_capture(self)
                        except Exception:
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

            patched = make_patched_capture_model(_orig_capture_model, beam_capture, prefill_capture)
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

        runner_class._patched_for_beam_pos_ids = True
        logger.info("%s hooks patched successfully.", runner_name)
