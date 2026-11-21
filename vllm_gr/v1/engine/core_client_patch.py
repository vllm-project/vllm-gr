# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Install atomic GR session transport and frontend lifecycle adapters."""

from vllm_gr.v1.engine.beam_session_client import (
    _FRONTEND_NATIVE,
    async_llm_add_request,
    llm_engine_add_request,
)
from vllm_gr.v1.engine.core_client import (
    _INPROC_NATIVE,
    inproc_abort_requests,
    inproc_add_request,
)
from vllm_gr.v1.engine.core_client import inproc_engine_failure as inproc_engine_failure
from vllm_gr.v1.engine.core_client import (
    inproc_get_output,
    inproc_shutdown,
    inproc_submit_gr_session,
    submit_gr_session_async,
    sync_mp_submit_gr_session,
)
from vllm_gr.v1.engine.output_processor import make_patched_process_outputs


def apply_request_type_patches() -> None:
    from vllm_gr.v1.engine.wire import register_engine_core_request_types

    register_engine_core_request_types()


def apply_engine_client_patches() -> None:
    """Install V1 submission on every supported EngineCore transport."""
    from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient

    AsyncMPClient.submit_gr_session_async = submit_gr_session_async
    SyncMPClient.submit_gr_session = sync_mp_submit_gr_session
    InprocClient.submit_gr_session = inproc_submit_gr_session
    for name, patched in (
        ("get_output", inproc_get_output),
        ("add_request", inproc_add_request),
        ("abort_requests", inproc_abort_requests),
        ("shutdown", inproc_shutdown),
    ):
        native = getattr(InprocClient, name)
        if native is not patched:
            _INPROC_NATIVE[name] = native
            setattr(InprocClient, name, patched)


def apply_async_llm_patches() -> None:
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.engine.output_processor import OutputProcessor

    from vllm_gr.v1.engine.beam_session_client import install_gr_frontend_methods

    OutputProcessor.process_outputs = make_patched_process_outputs(OutputProcessor.process_outputs)
    install_gr_frontend_methods(LLMEngine, AsyncLLM)
    for owner, name, key, patched in (
        (LLMEngine, "add_request", "llm_engine_add_request", llm_engine_add_request),
        (AsyncLLM, "_add_request", "async_llm_add_request", async_llm_add_request),
    ):
        native = getattr(owner, name)
        if native is not patched:
            _FRONTEND_NATIVE[key] = native
            setattr(owner, name, patched)


def apply_gr_session_patches() -> None:
    """Install GR transport in the frontend, including in-process engines."""
    apply_request_type_patches()
    apply_engine_client_patches()
    apply_async_llm_patches()

    from vllm_gr.v1.engine.engine_core_patch import apply_engine_core_child_patches

    apply_engine_core_child_patches()


def validate_engine_clients_patch() -> str | None:
    from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient

    valid = (
        getattr(AsyncMPClient, "submit_gr_session_async", None) is submit_gr_session_async
        and getattr(SyncMPClient, "submit_gr_session", None) is sync_mp_submit_gr_session
        and getattr(InprocClient, "submit_gr_session", None) is inproc_submit_gr_session
        and getattr(InprocClient, "get_output", None) is inproc_get_output
        and getattr(InprocClient, "add_request", None) is inproc_add_request
        and getattr(InprocClient, "abort_requests", None) is inproc_abort_requests
        and getattr(InprocClient, "shutdown", None) is inproc_shutdown
    )
    return None if valid else "EngineCore client methods are not fully installed"


def validate_async_llm_patch() -> str | None:
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.engine.output_processor import OutputProcessor

    from vllm_gr.v1.engine import beam_session_client
    from vllm_gr.v1.engine.output_processor import _PROCESS_OUTPUTS_PATCH_MARKER

    valid = (
        bool(getattr(OutputProcessor.process_outputs, _PROCESS_OUTPUTS_PATCH_MARKER, False))
        and getattr(LLMEngine, "add_request", None) is llm_engine_add_request
        and getattr(AsyncLLM, "_add_request", None) is async_llm_add_request
    )
    for cls, suffix in ((LLMEngine, "sync"), (AsyncLLM, "async")):
        for operation in ("submit", "wait", "abort", "release"):
            name = f"{operation}_gr_session"
            valid = valid and getattr(cls, name, None) is getattr(
                beam_session_client, f"{name}_{suffix}"
            )
    return None if valid else "GR frontend lifecycle methods are not fully installed"
