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
    prepare_request_fn,
    register_beam_output_fn,
)
from vllm_gr.v1.engine.core_client import (
    add_requests_async,
    beam_request_step_update_async,
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
    if self.is_dp:
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


def apply_batch_fork_patches():
    """Monkey-patch AsyncMPClient and AsyncLLM with ADD_BATCH/BEAM_REQUEST_STEP_UPDATE
    methods. Must be called after the enum members exist (i.e. after
    _add_enum_member has been called for ADD_BATCH and BEAM_REQUEST_STEP_UPDATE in the
    parent process)."""
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient

    from vllm_gr.v1.engine.engine_core_patch import _add_enum_member

    # Ensure enum members exist in parent process
    _add_enum_member("ADD_BATCH", b"\x05")
    _add_enum_member("BEAM_REQUEST_STEP_UPDATE", b"\x06")

    # Patch AsyncMPClient
    AsyncMPClient.add_requests_async = add_requests_async
    AsyncMPClient.beam_request_step_update_async = beam_request_step_update_async

    # Patch AsyncLLM
    AsyncLLM.prepare_request = prepare_request_fn
    AsyncLLM._add_requests_batch = _add_requests_batch_fn
    AsyncLLM.register_beam_output = register_beam_output_fn
    AsyncLLM.beam_request_step_update = beam_request_step_update_fn

    # Patch SyncMPClient
    SyncMPClient.add_requests = sync_mp_add_requests
    SyncMPClient.beam_request_step_update = sync_mp_beam_request_step_update

    # Patch InprocClient
    InprocClient.add_requests = inproc_add_requests
    InprocClient.beam_request_step_update = inproc_beam_request_step_update
