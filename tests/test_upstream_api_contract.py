# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contract checks for the private upstream APIs patched by vllm-gr."""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable, Sequence

import pytest


def _assert_parameter_prefix(
    target: Callable[..., object],
    expected: Sequence[str],
) -> None:
    parameters = list(inspect.signature(inspect.unwrap(target)).parameters)
    assert parameters[: len(expected)] == list(expected)


def test_vllm_core_patch_contract() -> None:
    from vllm.v1.attention.selector import _cached_get_attn_backend
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    _assert_parameter_prefix(
        _cached_get_attn_backend,
        ("backend", "attn_selector_config", "num_heads"),
    )
    _assert_parameter_prefix(EngineCore.__init__, ("self", "vllm_config", "executor_class"))
    assert callable(EngineCoreProc.run_engine_core)
    assert {"dp_rank", "local_dp_rank"}.issubset(
        inspect.signature(EngineCoreProc.run_engine_core).parameters
    )
    _assert_parameter_prefix(Scheduler.schedule, ("self",))


def test_vllm_gpu_model_runner_patch_contract() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _assert_parameter_prefix(
        GPUModelRunner.execute_model,
        ("self", "scheduler_output", "intermediate_tensors"),
    )
    _assert_parameter_prefix(
        GPUModelRunner._prepare_inputs,
        ("self", "scheduler_output", "num_scheduled_tokens"),
    )
    _assert_parameter_prefix(
        GPUModelRunner._bookkeeping_sync,
        (
            "self",
            "scheduler_output",
            "sampler_output",
            "logits",
            "hidden_states",
            "num_scheduled_tokens",
        ),
    )


def test_vllm_serving_and_client_patch_contract() -> None:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.engine.serving import OpenAIServing
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.logprobs import FlatLogprobs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient

    _assert_parameter_prefix(EngineArgs.add_cli_args, ("parser",))
    assert callable(EngineArgs.create_engine_config)
    assert callable(EngineArgs.from_cli_args)
    _assert_parameter_prefix(
        OpenAIServingModels.__init__,
        ("self", "engine_client", "base_model_paths", "lora_modules"),
    )
    assert callable(FlatLogprobs.append)
    assert isinstance(OpenAIServing, type)
    assert isinstance(ChatCompletionRequest, type)
    assert isinstance(AsyncLLM, type)
    assert callable(AsyncMPClient._send_input)
    assert callable(SyncMPClient._send_input)
    assert callable(SyncMPClient.add_request)
    assert callable(InprocClient.add_request)


def test_chat_beam_params_only_return_requested_logprobs() -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    from vllm_gr.entrypoints.openai.protocol import to_beam_search_params

    request = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "prompt"}],
        n=2,
        use_beam_search=True,
        logprobs=False,
    )
    assert to_beam_search_params(request, 5, {}).return_logprobs is False

    request.logprobs = True
    assert to_beam_search_params(request, 5, {}).return_logprobs is True


def test_vllm_ascend_model_runner_patch_contract() -> None:
    if importlib.util.find_spec("vllm_ascend") is None:
        pytest.skip("vllm-ascend is not installed")

    # Import through the worker entry point so vllm-ascend initializes its ops
    # package before loading model_runner_v1. A direct model_runner_v1 import
    # can otherwise enter the device_op/ops circular-import path.
    from vllm_ascend.worker.worker import NPUModelRunner, NPUWorker

    _assert_parameter_prefix(NPUWorker.init_device, ("self",))
    _assert_parameter_prefix(
        NPUModelRunner.execute_model,
        ("self", "scheduler_output", "intermediate_tensors"),
    )
    _assert_parameter_prefix(
        NPUModelRunner._prepare_inputs,
        ("self", "scheduler_output", "num_scheduled_tokens"),
    )
    _assert_parameter_prefix(
        NPUModelRunner._bookkeeping_sync,
        (
            "self",
            "scheduler_output",
            "sampler_output",
            "logits",
            "hidden_states",
            "num_scheduled_tokens",
            "spec_decode_metadata",
        ),
    )
