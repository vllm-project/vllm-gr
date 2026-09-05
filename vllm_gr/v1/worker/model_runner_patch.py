# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Active-platform ModelRunner Beam patch dispatcher."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from types import ModuleType

from vllm.logger import init_logger

from vllm_gr.v1.worker.model_runner_common import (
    _WORKER_CAPTURE_PATCH_MARKER,
    _WORKER_DETERMINE_PATCH_MARKER,
    get_beam_graph_runtime_stats,
    is_worker_runner_patched,
    make_patched_execute_model,
)

logger = init_logger(__name__)

_GPU_PATCH_MODULE = "vllm_gr.v1.worker.gpu_model_runner_patch"
_NPU_PATCH_MODULE = "vllm_gr.v1.worker.npu_model_runner_patch"
_CONSTRAINT_LOAD_PATCH_MARKER = "_vllm_gr_constraint_load_patched"
_CONSTRAINT_SHUTDOWN_PATCH_MARKER = "_vllm_gr_constraint_shutdown_patched"
_BEAM_GRAPH_STATS_HOOK_MARKER = "_vllm_gr_beam_graph_stats_hook_installed"
_GR_STATE_RELEASE_HOOK_MARKER = "_vllm_gr_state_release_hook_installed"
_GR_BACKEND_INIT_PATCH_MARKER = "_vllm_gr_backend_init_patched"


def make_init_with_gr_device_backend(original, factory):
    """Select the active platform backend exactly once per runner instance."""

    if getattr(original, _GR_BACKEND_INIT_PATCH_MARKER, False):
        return original

    @wraps(original)
    def init_with_gr_device_backend(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if not getattr(self, "_gr_device_backend_initialized", False):
            self._gr_device_backend = factory(self)
            self._gr_device_backend_initialized = True

    setattr(init_with_gr_device_backend, _GR_BACKEND_INIT_PATCH_MARKER, True)
    return init_with_gr_device_backend


def install_gr_state_worker_hooks(worker_cls: type) -> int:
    """Mount safe named GR lifecycle RPCs on either V1 Worker backend."""

    if getattr(worker_cls, _GR_STATE_RELEASE_HOOK_MARKER, False):
        return 0
    from vllm_gr.v1.worker.gr_model_runner_state import (
        get_gr_memory_stats,
        get_gr_model_runner_state_stats,
        release_gr_model_runner_states,
    )

    worker_cls.release_gr_model_runner_states = release_gr_model_runner_states
    worker_cls.get_gr_model_runner_state_stats = get_gr_model_runner_state_stats
    platform_memory_stats = getattr(
        _get_active_platform_module(), "get_gr_memory_stats", get_gr_memory_stats
    )
    worker_cls.get_gr_memory_stats = platform_memory_stats
    setattr(worker_cls, _GR_STATE_RELEASE_HOOK_MARKER, True)
    return 1


def close_constraint_backend(runner: object) -> None:
    """Detach and idempotently close the runner-owned canonical Backend."""

    backend = getattr(runner, "constraint_backend", None)
    # Detach first so no concurrent observer can discover a Backend while it is
    # being closed, and so repeated shutdown/load calls are naturally idempotent.
    setattr(runner, "constraint_backend", None)
    if backend is not None:
        backend.close()


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


def make_load_model_with_constraint_backend(original):
    """Load weights first, then atomically publish a ready canonical Backend."""

    if getattr(original, _CONSTRAINT_LOAD_PATCH_MARKER, False):
        return original

    @wraps(original)
    def load_model_with_constraint_backend(self, *args, **kwargs):
        from vllm_gr.config import GRConfig

        # A complete load replaces the model runtime. Release the previous
        # Backend before loading weights so a failed reload cannot leave stale
        # canonical tables reachable from this runner.
        close_constraint_backend(self)
        result = original(self, *args, **kwargs)

        backend = None
        try:
            gr_config = GRConfig.from_vllm_config(getattr(self, "vllm_config", None))
            table_config = gr_config.constraint_table
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
                # One Backend call is made per request. Decode requests carry
                # exactly beam_width logits rows, so beam_max_width is the
                # canonical row-capacity bound.
                max_rows=max_rows,
            )
            if backend is None:
                raise RuntimeError("enabled canonical constraint table loader returned no Backend")

            model_memory_usage = int(getattr(self, "model_memory_usage", 0))
            self.model_memory_usage = model_memory_usage + backend.info.device_memory_bytes

            # Publication is the atomic boundary. The loader returns an NPU
            # Backend only after readiness and returns an immediately usable CPU
            # Backend, so requests never observe a partially initialized object.
            self.constraint_backend = backend
            return result
        except BaseException:
            if backend is not None:
                backend.close()
            self.constraint_backend = None
            raise

    setattr(load_model_with_constraint_backend, _CONSTRAINT_LOAD_PATCH_MARKER, True)
    return load_model_with_constraint_backend


def make_shutdown_with_constraint_backend(original=None):
    """Release GR runner resources before native runner shutdown."""

    if original is not None and getattr(original, _CONSTRAINT_SHUTDOWN_PATCH_MARKER, False):
        return original

    if original is None:

        def shutdown_with_constraint_backend(self, *args, **kwargs):
            del args, kwargs
            from vllm_gr.v1.worker.gr_model_runner_state import (
                release_gr_model_runner_states,
            )

            try:
                release_gr_model_runner_states(self)
            finally:
                close_constraint_backend(self)

    else:

        @wraps(original)
        def shutdown_with_constraint_backend(self, *args, **kwargs):
            from vllm_gr.v1.worker.gr_model_runner_state import (
                release_gr_model_runner_states,
            )

            cleanup_error: BaseException | None = None
            try:
                release_gr_model_runner_states(self)
            except BaseException as exc:
                cleanup_error = exc
            try:
                close_constraint_backend(self)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            result = original(self, *args, **kwargs)
            if cleanup_error is not None:
                raise cleanup_error
            return result

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


def _get_active_platform_module() -> ModuleType:
    return import_module(_active_platform_module_name())


def discover_active_worker_runners() -> tuple[tuple[str, type], ...]:
    """Discover ModelRunner targets for only the active platform."""
    return _get_active_platform_module().discover_worker_runners()


def install_active_worker_diagnostic_hook(worker_cls: type) -> int:
    """Install only the diagnostic supplied by the active backend adapter."""
    installer = getattr(_get_active_platform_module(), "install_worker_diagnostic_hook", None)
    if not callable(installer):
        # Profiling diagnostics are observational, never a runtime dependency.
        # The profiling harness calls the named RPC and fails its evidence gate
        # if a backend does not provide it; ordinary patch initialization and
        # third-party/fake adapters must retain their pre-diagnostic behavior.
        return 0
    return int(installer(worker_cls))


def _install_active_worker_resource_hook() -> None:
    """Mount named vllm-gr utility RPCs on the active CUDA Worker class."""

    if _active_platform_module_name() != _GPU_PATCH_MODULE:
        # NPUWorker installs this hook from its delayed init_device wrapper in
        # ascend_compat, after vllm-ascend has finished resolving its class.
        return
    from vllm.v1.worker.gpu_worker import Worker

    from vllm_gr.constraints.worker_inject import install_worker_resource_hook

    install_worker_resource_hook(Worker)
    install_active_worker_diagnostic_hook(Worker)
    install_gr_state_worker_hooks(Worker)
    if not getattr(Worker, _BEAM_GRAPH_STATS_HOOK_MARKER, False):
        # A named method is required here: vLLM's safe msgpack utility channel
        # intentionally rejects callable serialization unless insecure pickle
        # fallback is enabled.  The profiling contract must work under spawn
        # without weakening that policy.
        Worker.get_gr_beam_graph_runtime_stats = get_beam_graph_runtime_stats  # type: ignore[attr-defined]
        setattr(Worker, _BEAM_GRAPH_STATS_HOOK_MARKER, True)


def apply_worker_patches() -> None:
    """Install grouped-Beam hooks on active-platform ModelRunner targets."""
    logger.info("vllm-gr: apply_worker_patches called")
    # collective_rpc targets the Worker, not its ModelRunner.  Install this
    # even when all runner hooks were already present (idempotent) so canonical
    # Constraint Tables can be injected in both uniproc and multiproc engines.
    _install_active_worker_resource_hook()
    platform_module = _get_active_platform_module()
    runners_to_patch = [
        (runner_name, runner_class)
        for runner_name, runner_class in platform_module.discover_worker_runners()
        if not is_worker_runner_patched(runner_class)
    ]

    for runner_name, runner_class in runners_to_patch:
        backend_factory = getattr(platform_module, "create_gr_device_backend", None)
        if callable(backend_factory):
            runner_class.__init__ = make_init_with_gr_device_backend(
                runner_class.__init__, backend_factory
            )
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
            platform_shutdown_patch = getattr(platform_module, "make_patched_shutdown", None)
            if callable(platform_shutdown_patch):
                runner_class.shutdown = platform_shutdown_patch(runner_class.shutdown)

        platform_execute_patcher = getattr(
            platform_module, "make_patched_execute_model", None
        )
        if callable(platform_execute_patcher):
            runner_class.execute_model = platform_execute_patcher(
                runner_class.execute_model
            )
        else:
            runner_class.execute_model = make_patched_execute_model(
                runner_class.execute_model
            )
        runner_class._prepare_inputs = platform_module.make_patched_prepare_inputs(
            runner_class._prepare_inputs
        )
        input_prep_patch = getattr(platform_module, "make_patched_synchronize_input_prep", None)
        if callable(input_prep_patch) and hasattr(runner_class, "synchronize_input_prep"):
            runner_class.synchronize_input_prep = input_prep_patch(
                runner_class.synchronize_input_prep
            )
        runner_class._bookkeeping_sync = platform_module.make_patched_bookkeeping_sync(
            runner_class._bookkeeping_sync
        )
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
        if (
            beam_capture is not None
            and hasattr(runner_class, "capture_model")
            and not getattr(runner_class.capture_model, _WORKER_CAPTURE_PATCH_MARKER, False)
        ):
            _orig_capture_model = runner_class.capture_model

            def make_patched_capture_model(_orig, _beam_capture):
                @wraps(_orig)
                def patched_capture_model(self, *args, **kwargs):
                    res = _orig(self, *args, **kwargs)
                    try:
                        _beam_capture(self)
                    except Exception:
                        logger.exception(
                            "vllm-gr: beam decode graph dummy capture failed; "
                            "beam decode will fall back to eager"
                        )
                    return res

                return patched_capture_model

            patched = make_patched_capture_model(_orig_capture_model, beam_capture)
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
