# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for AsyncMPClient / AsyncLLM to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_gr.v1.engine.async_llm import (
    _add_requests_batch_fn,
    beam_request_step_update_fn,
    cancel_beam_controller_fn,
    prepare_request_fn,
    register_beam_output_fn,
    run_beam_controller_fn,
)
from vllm_gr.v1.engine.core_client import (
    add_requests_async,
    beam_request_step_update_async,
    cancel_beam_controller_async,
    ensure_beam_constraint_resource_async,
    run_beam_controller_async,
)

logger = init_logger(__name__)


def sync_mp_add_requests(self, requests, force_batch=False):
    from vllm.v1.engine import EngineCoreRequestType

    if not requests:
        return
    if len(requests) == 1 and not force_batch:
        return self.add_request(requests[0])
    if getattr(self, "is_dp", False):
        self.engines_running = True
    self._send_input(EngineCoreRequestType.ADD_BATCH, requests)


def sync_mp_beam_request_step_update(self, update):
    from vllm.v1.engine import EngineCoreRequestType

    self._send_input(EngineCoreRequestType.BEAM_REQUEST_STEP_UPDATE, update)


def inproc_add_requests(self, requests, force_batch=False):
    from vllm.v1.engine import EngineCoreRequestType

    if not requests:
        return
    if len(requests) == 1 and not force_batch:
        return self.add_request(requests[0])
    for req in requests:
        try:
            preprocessed = self.engine_core.preprocess_add_request(req)
        except Exception:
            self.engine_core._handle_request_preproc_error(req)
            continue
        self.engine_core._cache_beam_request(preprocessed[0])
        self.engine_core.input_queue.put_nowait((EngineCoreRequestType.ADD, preprocessed))


def inproc_beam_request_step_update(self, update):
    self.engine_core._handle_beam_request_step_update(update)


# ---------------------------------------------------------------------------
# apply_batch_fork_patches — wire ADD_BATCH / BEAM_REQUEST_STEP_UPDATE into client classes
# ---------------------------------------------------------------------------


def apply_request_type_patches() -> None:
    """Register the GR request types used on the EngineCore wire protocol."""
    from vllm_gr.v1.engine.wire import register_engine_core_request_types

    register_engine_core_request_types()


def apply_engine_client_patches() -> None:
    """Add batch and beam-update methods to EngineCore clients."""
    from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient

    AsyncMPClient.add_requests_async = add_requests_async
    AsyncMPClient.beam_request_step_update_async = beam_request_step_update_async
    AsyncMPClient.run_beam_controller_async = run_beam_controller_async
    AsyncMPClient.cancel_beam_controller_async = cancel_beam_controller_async
    AsyncMPClient.ensure_beam_constraint_resource_async = ensure_beam_constraint_resource_async
    SyncMPClient.add_requests = sync_mp_add_requests
    SyncMPClient.beam_request_step_update = sync_mp_beam_request_step_update
    InprocClient.add_requests = inproc_add_requests
    InprocClient.beam_request_step_update = inproc_beam_request_step_update


def apply_async_llm_patches() -> None:
    """Add grouped request orchestration methods to AsyncLLM."""
    from vllm.v1.engine.async_llm import AsyncLLM

    AsyncLLM.prepare_request = prepare_request_fn
    AsyncLLM._add_requests_batch = _add_requests_batch_fn
    AsyncLLM.register_beam_output = register_beam_output_fn
    AsyncLLM.beam_request_step_update = beam_request_step_update_fn
    AsyncLLM.run_beam_controller = run_beam_controller_fn
    AsyncLLM.cancel_beam_controller = cancel_beam_controller_fn


def apply_batch_fork_patches() -> None:
    """Compatibility helper that installs the complete grouped-beam pipeline."""
    apply_request_type_patches()
    apply_engine_client_patches()
    apply_async_llm_patches()

    # The aggregate helper is retained for callers outside the registry and
    # for InprocClient, where EngineCore runs in the frontend process.
    from vllm_gr.v1.engine.engine_core_patch import apply_engine_core_child_patches

    apply_engine_core_child_patches()
