# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for AsyncMPClient / AsyncLLM to support
ADD_BATCH and BEAM_REQUEST_STEP_UPDATE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_gr.v1.engine.async_llm import (
    _add_requests_batch_fn,
    beam_request_step_update_fn,
    beam_session_template_fn,
    make_patched_process_outputs,
    prepare_request_fn,
    register_beam_output_fn,
    submit_session_fn,
    track_beam_session_fn,
)
from vllm_gr.v1.engine.core_client import (
    add_requests_async,
    beam_request_step_update_async,
    beam_session_template_async,
    submit_gr_session_async,
    submit_session_async,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    pass


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


def sync_mp_beam_request_step_update_batch(self, updates):
    from vllm.v1.engine import EngineCoreRequestType

    if not updates:
        return
    self._send_input(EngineCoreRequestType.BEAM_STEP_BATCH, updates)


def sync_mp_beam_session_template(self, template):
    from vllm.v1.engine import EngineCoreRequestType

    self._send_input(EngineCoreRequestType.BEAM_SESSION_TEMPLATE, template)


def sync_mp_submit_session(self, requests, template):
    from vllm.v1.engine import EngineCoreRequestType

    if not requests:
        return
    self._send_input(EngineCoreRequestType.SUBMIT_SESSION, (requests, template))


def sync_mp_submit_gr_session(self, batch_request):
    from vllm.v1.engine import EngineCoreRequestType

    request = batch_request.items[0].request
    request.client_index = getattr(self, "client_index", 0)
    if hasattr(self, "current_wave"):
        request.current_wave = self.current_wave
    if getattr(self, "is_dp", False):
        self.engines_running = True
    self._send_input(EngineCoreRequestType.SUBMIT_GR_SESSION, batch_request)


def inproc_add_requests(self, requests, force_batch=False):
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
        self.engine_core.add_request(*preprocessed)


def inproc_beam_request_step_update(self, update):
    self.engine_core._handle_beam_request_step_update(update)


def inproc_beam_request_step_update_batch(self, updates):
    if not updates:
        return

    from vllm.v1.engine import EngineCoreRequestType

    from vllm_gr.v1.engine.core import dispatch_gr_engine_core_input

    dispatch_gr_engine_core_input(
        self.engine_core, EngineCoreRequestType.BEAM_STEP_BATCH, updates
    )


def inproc_beam_session_template(self, template):
    self.engine_core._handle_beam_session_template(template)


def inproc_submit_session(self, requests, template):
    if not requests:
        return
    self.engine_core._handle_submit_session(requests, template)


def inproc_submit_gr_session(self, batch_request):
    self.engine_core._handle_submit_gr_session(batch_request)


_INPROC_OUTPUT_MARKER = "_vllm_gr_inproc_self_advance"


def make_patched_inproc_get_output(original):
    """Run engine-driven chaining at the in-process step boundary.

    ``InprocClient.get_output`` calls ``EngineCore.step_fn()`` directly, so
    the EngineCoreProc ``_process_engine_step`` hook never runs.  Without
    this wrap, ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` silently skipped
    chaining and fell back to frontend-driven updates.
    """
    if getattr(original, _INPROC_OUTPUT_MARKER, False):
        return original

    from functools import wraps

    @wraps(original)
    def patched_get_output(self):
        from vllm.v1.engine import EngineCoreOutputs

        from vllm_gr.v1.engine.core import _beam_self_advance

        outputs, model_executed = self.engine_core.step_fn()
        if outputs:
            try:
                _beam_self_advance(self.engine_core, outputs)
            except Exception:
                logger.exception(
                    "vllm-gr: in-process beam self-advance failed; falling back "
                    "to the frontend step-update loop for this step"
                )
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

    setattr(patched_get_output, _INPROC_OUTPUT_MARKER, True)
    return patched_get_output


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
    AsyncMPClient.beam_session_template_async = beam_session_template_async
    AsyncMPClient.submit_session_async = submit_session_async
    AsyncMPClient.submit_gr_session_async = submit_gr_session_async
    SyncMPClient.add_requests = sync_mp_add_requests
    SyncMPClient.beam_request_step_update = sync_mp_beam_request_step_update
    SyncMPClient.beam_session_template = sync_mp_beam_session_template
    SyncMPClient.beam_request_step_update_batch = (
        sync_mp_beam_request_step_update_batch
    )
    SyncMPClient.submit_session = sync_mp_submit_session
    SyncMPClient.submit_gr_session = sync_mp_submit_gr_session
    InprocClient.add_requests = inproc_add_requests
    InprocClient.beam_request_step_update = inproc_beam_request_step_update
    InprocClient.beam_session_template = inproc_beam_session_template
    InprocClient.beam_request_step_update_batch = (
        inproc_beam_request_step_update_batch
    )
    InprocClient.submit_session = inproc_submit_session
    InprocClient.submit_gr_session = inproc_submit_gr_session
    InprocClient.get_output = make_patched_inproc_get_output(InprocClient.get_output)


def apply_async_llm_patches() -> None:
    """Add grouped request orchestration methods to AsyncLLM."""
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.engine.output_processor import OutputProcessor

    AsyncLLM.prepare_request = prepare_request_fn
    AsyncLLM._add_requests_batch = _add_requests_batch_fn
    AsyncLLM.register_beam_output = register_beam_output_fn
    AsyncLLM.beam_request_step_update = beam_request_step_update_fn
    AsyncLLM.beam_session_template = beam_session_template_fn
    AsyncLLM.submit_session = submit_session_fn
    AsyncLLM.track_beam_session = track_beam_session_fn
    OutputProcessor.process_outputs = make_patched_process_outputs(OutputProcessor.process_outputs)
    from vllm_gr.v1.engine.gr_session_client import install_gr_frontend_methods

    install_gr_frontend_methods(LLMEngine, AsyncLLM)


def apply_batch_fork_patches() -> None:
    """Compatibility helper that installs the complete grouped-beam pipeline."""
    apply_request_type_patches()
    apply_engine_client_patches()
    apply_async_llm_patches()

    # The aggregate helper is retained for callers outside the registry and
    # for InprocClient, where EngineCore runs in the frontend process.
    from vllm_gr.v1.engine.engine_core_patch import apply_engine_core_child_patches

    apply_engine_core_child_patches()
