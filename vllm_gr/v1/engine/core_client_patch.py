# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend helpers needed by the persistent GR request path."""

from vllm_gr.v1.engine.async_llm import _add_requests_batch_fn, prepare_request_fn


def apply_async_llm_patches() -> None:
    """Mount request preparation while keeping vLLM's native ADD protocol."""
    from vllm.v1.engine.async_llm import AsyncLLM

    AsyncLLM.prepare_request = prepare_request_fn
    AsyncLLM._add_requests_batch = _add_requests_batch_fn


def apply_persistent_runtime_patches() -> None:
    """Install the persistent frontend and EngineCore lifecycle hooks."""
    apply_async_llm_patches()

    from vllm_gr.v1.engine.engine_core_patch import apply_engine_core_child_patches

    apply_engine_core_child_patches()
