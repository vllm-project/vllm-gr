# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

import os
from functools import wraps

from vllm.logger import init_logger

from vllm_gr.v1.engine.scheduler_metadata import build_beam_request_metadata
from vllm_gr.v1.engine.wire import register_engine_core_request_types
from vllm_gr.v1.worker import model_runner_common as _model_runner_common

logger = init_logger(__name__)

BEAM_PROC_ENGINECORE_ENV = "VLLM_GR_BEAM_PROC_ENGINECORE_ADDR"
BEAM_PROC_ENGINECORE_TX_ENV = "VLLM_GR_BEAM_PROC_ENGINECORE_TX_ADDR"
BEAM_PROC_ENGINECORE_RX_ENV = "VLLM_GR_BEAM_PROC_ENGINECORE_RX_ADDR"
BEAM_PROC_ENABLED_ENV = "VLLM_GR_BEAM_PROC_ENABLED"
_BEAM_STATE_INIT_MARKER = "_vllm_gr_initializes_beam_state"
_BEAMPROC_PROC_INIT_MARKER = "_vllm_gr_initializes_beamproc_io"
_BEAMPROC_BUSY_LOOP_MARKER = "_vllm_gr_ensures_beamproc_io"
_BEAMPROC_IO_START_ATTEMPTED_ATTR = "_vllm_gr_beamproc_io_start_attempted"
_SCHEDULER_PATCH_MARKER = "_vllm_gr_beam_scheduler_patch"

# ---------------------------------------------------------------------------
# EngineCore.__init__ wrapper
# ---------------------------------------------------------------------------


def make_patched_engine_core_init(original_init):
    """Wrap EngineCore.__init__ to add beam state."""

    if getattr(original_init, _BEAM_STATE_INIT_MARKER, False):
        return original_init

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        # Initialize before calling upstream. EngineCoreProc starts its input
        # thread from __init__, so beam state must exist throughout startup.
        from vllm_gr.v1.engine.core import _ensure_beam_state

        _ensure_beam_state(self)
        original_init(self, *args, **kwargs)
        self.beamproc_block_size = getattr(
            getattr(getattr(self, "vllm_config", None), "cache_config", None),
            "block_size",
            None,
        )

    setattr(patched_init, _BEAM_STATE_INIT_MARKER, True)
    return patched_init


def _ensure_beamproc_io_started(engine_core_proc) -> None:
    """Start BeamProc I/O at most once for an EngineCoreProc instance."""
    if getattr(engine_core_proc, _BEAMPROC_IO_START_ATTEMPTED_ATTR, False):
        return

    setattr(engine_core_proc, _BEAMPROC_IO_START_ATTEMPTED_ATTR, True)
    from vllm_gr.v1.engine.core import _start_beamproc_io_thread

    _start_beamproc_io_thread(engine_core_proc)


def make_patched_engine_core_proc_init(original_init):
    """Wrap EngineCoreProc.__init__ to add BeamProc I/O queues/thread."""
    if getattr(original_init, _BEAMPROC_PROC_INIT_MARKER, False):
        return original_init

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _ensure_beamproc_io_started(self)

    setattr(patched_init, _BEAMPROC_PROC_INIT_MARKER, True)

    return patched_init


def make_patched_engine_core_proc_run_busy_loop(original_run_busy_loop):
    """Ensure BeamProc I/O exists before EngineCore begins processing work.

    This is a fallback for spawned workers where EngineCoreProc.__init__ was
    resolved before the constructor wrapper was installed. Attribute lookup
    for run_busy_loop happens after construction, so this hook still observes
    the patched class in that case.
    """
    if getattr(original_run_busy_loop, _BEAMPROC_BUSY_LOOP_MARKER, False):
        return original_run_busy_loop

    @wraps(original_run_busy_loop)
    def patched_run_busy_loop(self, *args, **kwargs):
        _ensure_beamproc_io_started(self)
        try:
            return original_run_busy_loop(self, *args, **kwargs)
        finally:
            from vllm_gr.v1.engine.core import _stop_beamproc_native_io

            _stop_beamproc_native_io(self)

    setattr(patched_run_busy_loop, _BEAMPROC_BUSY_LOOP_MARKER, True)
    return patched_run_busy_loop


# ---------------------------------------------------------------------------
# Child-process patch applicator — called from patched run_engine_core
# ---------------------------------------------------------------------------


def apply_engine_core_request_patches() -> None:
    """Install grouped-beam dispatch and the optional BeamProc transport."""
    import vllm.v1.engine as engine_mod
    import vllm.v1.engine.core as upstream_core
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    from vllm_gr.v1.engine.core import (
        _cache_beam_request,
        _enqueue_beamproc_abort,
        _enqueue_beamproc_message,
        _handle_beam_fork,
        _handle_beam_request_step_update,
        make_patched_handle_client_request,
        make_patched_process_engine_step,
        make_patched_process_input_queue,
        process_input_sockets,
        process_output_sockets,
    )

    logger.debug(
        "EngineCore BeamProc configuration pid=%s patch_file=%s "
        "upstream_core_file=%s enabled=%r binary=%r tx=%r rx=%r",
        os.getpid(),
        __file__,
        upstream_core.__file__,
        os.getenv(BEAM_PROC_ENABLED_ENV),
        os.getenv("VLLM_GR_BEAM_PROC_BINARY"),
        os.getenv(BEAM_PROC_ENGINECORE_TX_ENV),
        os.getenv(BEAM_PROC_ENGINECORE_RX_ENV),
    )

    beam_proc_addr = os.getenv(BEAM_PROC_ENGINECORE_ENV)
    beam_proc_tx_addr = os.getenv(BEAM_PROC_ENGINECORE_TX_ENV, beam_proc_addr or "")
    beam_proc_rx_addr = os.getenv(BEAM_PROC_ENGINECORE_RX_ENV, beam_proc_addr or "")
    if beam_proc_tx_addr:
        engine_mod.BEAM_PROC_ENGINECORE_ZMQ_ADDR = beam_proc_tx_addr
        EngineCore.BEAM_PROC_ENGINECORE_ZMQ_ADDR = beam_proc_tx_addr
        EngineCoreProc.BEAM_PROC_ENGINECORE_ZMQ_ADDR = beam_proc_tx_addr
        engine_mod.BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR = beam_proc_tx_addr
        EngineCore.BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR = beam_proc_tx_addr
        EngineCoreProc.BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR = beam_proc_tx_addr
        logger.info("EngineCore beam-proc TX address configured: %s", beam_proc_tx_addr)
    if beam_proc_rx_addr:
        engine_mod.BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR = beam_proc_rx_addr
        EngineCore.BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR = beam_proc_rx_addr
        EngineCoreProc.BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR = beam_proc_rx_addr
        logger.info("EngineCore beam-proc RX address configured: %s", beam_proc_rx_addr)

    register_engine_core_request_types()

    # Wrap both constructors. Patching EngineCoreProc explicitly protects
    # against changes in upstream inheritance and plugin loading order.
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)
    EngineCoreProc.__init__ = make_patched_engine_core_init(EngineCoreProc.__init__)

    # Wrap EngineCoreProc.__init__ to add BeamProc I/O queues/thread
    EngineCoreProc.__init__ = make_patched_engine_core_proc_init(EngineCoreProc.__init__)
    EngineCoreProc.run_busy_loop = make_patched_engine_core_proc_run_busy_loop(
        EngineCoreProc.run_busy_loop
    )

    # Bind the upstream grouped update and the internal BeamProc plan handler.
    EngineCore._cache_beam_request = _cache_beam_request
    EngineCore._handle_beam_fork = _handle_beam_fork
    EngineCore._handle_beam_request_step_update = _handle_beam_request_step_update
    EngineCoreProc._cache_beam_request = _cache_beam_request
    EngineCoreProc._handle_beam_fork = _handle_beam_fork
    EngineCoreProc._handle_beam_request_step_update = _handle_beam_request_step_update
    EngineCoreProc._enqueue_beamproc_message = _enqueue_beamproc_message
    EngineCoreProc._enqueue_beamproc_abort = _enqueue_beamproc_abort

    # Keep BeamProc plan queue processing on the EngineCore main thread.
    EngineCoreProc._process_input_queue = make_patched_process_input_queue(
        EngineCoreProc._process_input_queue
    )
    EngineCoreProc._process_engine_step = make_patched_process_engine_step(
        EngineCoreProc._process_engine_step
    )
    EngineCoreProc._handle_client_request = make_patched_handle_client_request(
        EngineCoreProc._handle_client_request
    )

    # Replace EngineCoreProc.process_input_sockets
    EngineCoreProc.process_input_sockets = process_input_sockets
    EngineCoreProc.process_output_sockets = process_output_sockets

    EngineCore._patched_for_beam_requests = True
    EngineCoreProc._patched_for_beam_requests = True
    logger.debug("Engine-core request patches applied.")


def apply_engine_core_child_patches() -> None:
    """Compatibility helper that installs all EngineCore-side capabilities.

    This is called by the process-entry wrapper in a freshly spawned
    interpreter. Each capability has its own idempotent applicator so a
    partially initialized process can be checked and repaired explicitly.
    """
    from vllm_gr.v1.worker.model_runner_patch import apply_worker_patches

    apply_engine_core_request_patches()
    apply_scheduler_patch()
    apply_worker_patches()


# Re-entry guard to detect recursive calls
_run_engine_core_active = False


def run_engine_core(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    """Wrapper for EngineCoreProc.run_engine_core that applies patches
    before the engine starts.  This is a module-level function so that
    multiprocessing 'spawn' can pickle it by qualified name.

    In the spawned child process, all modules are freshly imported, so
    EngineCoreProc.run_engine_core is the original unpatched version.
    We apply our patches first, then delegate to the original.
    """
    global _run_engine_core_active

    # Re-entry guard: detect if we're already inside run_engine_core
    if _run_engine_core_active:
        raise RuntimeError(
            "run_engine_core re-entry detected. This indicates a recursive "
            "call, likely due to patching being applied multiple times. "
            "Check that patch_batch_and_fork() is only called once."
        )

    from vllm.v1.engine.core import EngineCoreProc

    # Guard against recursion: if vllm_gr/init.py was auto-imported in
    # the child process, run_patch() will have replaced
    # EngineCoreProc.run_engine_core with *this* function again.
    # Use the saved original stashed by patch_batch_and_fork() if
    # available; otherwise fall back to the current class attribute
    # (which is the unpatched original in a fresh child interpreter).
    original_run = getattr(run_engine_core, "_original_run_engine_core", None)

    # Validate the saved original
    if original_run is not None and original_run is not run_engine_core:
        # We have a saved original that isn't ourselves - validate it
        if not callable(original_run):
            raise RuntimeError(
                f"run_engine_core._original_run_engine_core is not callable: "
                f"type={type(original_run)}"
            )
    else:
        # No saved original (or it points to ourselves, which is invalid)
        # Fall back to the current class attribute (should be the original in a fresh process)
        if original_run is run_engine_core:
            logger.warning(
                "run_engine_core._original_run_engine_core points to itself. "
                "This may indicate corrupted patch state from a previous run. "
                "Falling back to EngineCoreProc.run_engine_core."
            )
        original_run = EngineCoreProc.run_engine_core

    # Final validation: ensure we didn't end up with ourselves
    if original_run is run_engine_core:
        raise RuntimeError(
            "run_engine_core recursion detected: EngineCoreProc.run_engine_core "
            "points to this wrapper. This indicates that patching was applied "
            "but the original was not saved, or the child process imported "
            "vllm_gr before starting."
        )

    # Validate that original_run looks like a real function from vllm
    # (Skip validation for test mocks)
    if hasattr(original_run, "__module__"):
        original_module = original_run.__module__
        expected_module = "vllm.v1.engine.core"
        # Skip warning for unittest mocks (used in tests)
        if original_module not in (expected_module, "unittest.mock", "mock"):
            logger.warning(
                "Unexpected module for original run_engine_core: %s (expected %s). "
                "Proceeding anyway, but this may indicate an issue.",
                original_module,
                expected_module,
            )

    # Set re-entry guard and apply patches
    _run_engine_core_active = True
    try:
        # Use the shared initialization mechanism with an EngineCore-only
        # plan. In a spawned child this repairs targets that became importable
        # only after the frontend initialized without importing API modules.
        from vllm_gr.patch import initialize_engine_core_beam_patches

        initialize_engine_core_beam_patches()
        original_run(
            *args,
            dp_rank=dp_rank,
            local_dp_rank=local_dp_rank,
            **kwargs,
        )
    finally:
        _run_engine_core_active = False


def make_patch_get_computed_blocks(
    original_get_computed_blocks, create_kv_cache_blocks, block_size
):
    """Helper to create patched get_computed_blocks function for beam decode requests."""

    def patch_get_computed_blocks(req):
        bn, tn = original_get_computed_blocks(req)
        if req and getattr(req, "is_beam_decode", False):
            pl = getattr(req, "prefix_len", 0)
            if tn > pl:
                # Truncate cached computed blocks to match prefix_len
                num_blocks = (pl + block_size - 1) // block_size
                sliced_blocks = tuple(group[:num_blocks] for group in bn.blocks)
                return create_kv_cache_blocks(sliced_blocks), pl
        return bn, tn

    return patch_get_computed_blocks


def apply_scheduler_patch():
    """Monkey-patch Scheduler.schedule to propagate custom attributes via SchedulerOutput."""
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, False):
        return

    _original_schedule = Scheduler.schedule

    @wraps(_original_schedule)
    def patched_schedule(self):
        _original_get_computed_blocks = self.kv_cache_manager.get_computed_blocks
        _original_cache_blocks = self.kv_cache_manager.coordinator.cache_blocks
        block_size = self.cache_config.block_size

        patch_get_computed_blocks = make_patch_get_computed_blocks(
            _original_get_computed_blocks,
            self.kv_cache_manager.create_kv_cache_blocks,
            block_size,
        )

        def patch_cache_blocks(request, num_computed_tokens):
            if (
                request
                and not getattr(request, "is_beam_decode", False)
                and hasattr(request, "block_hashes")
                and request.block_hashes
            ):
                num_prompt_tokens = getattr(
                    request, "num_prompt_tokens", getattr(request, "num_tokens", None)
                )
                if num_prompt_tokens is not None and num_computed_tokens >= num_prompt_tokens:
                    chunk_padded_target = (
                        (num_computed_tokens + block_size - 1) // block_size * block_size
                    )
                    num_hashes = len(request.block_hashes)
                    padded_target = min(chunk_padded_target, num_hashes * block_size)
                    if (
                        num_computed_tokens < padded_target
                        and num_computed_tokens % block_size != 0
                    ):
                        num_computed_tokens = padded_target
            return _original_cache_blocks(request, num_computed_tokens)

        self.kv_cache_manager.get_computed_blocks = patch_get_computed_blocks
        self.kv_cache_manager.coordinator.cache_blocks = patch_cache_blocks
        try:
            output = _original_schedule(self)

            # Propagate beam request attributes from Request objects to SchedulerOutput
            beam_data = {}
            if hasattr(output, "num_scheduled_tokens"):
                for req_id, num_scheduled_tokens in output.num_scheduled_tokens.items():
                    req = self.requests.get(req_id)
                    if req:
                        metadata = build_beam_request_metadata(req, num_scheduled_tokens)
                        if metadata is not None:
                            beam_data[req_id] = metadata
            output.beam_data = beam_data
            return output
        finally:
            self.kv_cache_manager.get_computed_blocks = _original_get_computed_blocks
            self.kv_cache_manager.coordinator.cache_blocks = _original_cache_blocks

    Scheduler.schedule = patched_schedule
    setattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, True)
    Scheduler._patched_for_beam_requests = True
    logger.debug("Scheduler.schedule patched for beam-requests propagation.")


# Preserve the historical import path while using upstream's extracted,
# device-neutral model-runner implementation.
BeamReqIdsProxy = _model_runner_common.BeamReqIdsProxy
Mock1DArray = _model_runner_common.Mock1DArray
Mock2DArray = _model_runner_common.Mock2DArray
InputBatchProxy = _model_runner_common.InputBatchProxy
_get_positions_buffers = _model_runner_common._get_positions_buffers
_write_position_segments = _model_runner_common._write_position_segments
_process_beam_decode_request = _model_runner_common._process_beam_decode_request
_remux_logprobs_tensors = _model_runner_common._remux_logprobs_tensors
_beam_remap_inputs = _model_runner_common._beam_remap_inputs
_beam_collapse_bookkeeping = _model_runner_common._beam_collapse_bookkeeping
make_patched_execute_model = _model_runner_common.make_patched_execute_model
is_worker_runner_patched = _model_runner_common.is_worker_runner_patched
