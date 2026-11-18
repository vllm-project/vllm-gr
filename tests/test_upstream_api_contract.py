# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="no-untyped-def"

"""Contract checks for the private upstream APIs patched by vllm-gr."""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable, Sequence

import pytest

from tools.hardware_marks import hardware_test


def _innermost(target: Callable[..., object]) -> Callable[..., object]:
    """Follow wrapper chains down to the function that owns the real signature.

    ``inspect.unwrap`` only follows ``__wrapped__``, which well-behaved wrappers
    set via ``functools.wraps``. Some third-party wrappers installed at import
    time do not: vllm-ascend's profiling-chunk patch replaces
    ``EngineCore.__init__`` with a plain closing function
    (``def _patched_engine_core_init(self, *args, **kwargs)``) that only closes
    over ``original_init``. Importing ``vllm.entrypoints.launchers.cli_args``
    installs it as a module-level side effect, so whether these contract checks
    see the upstream signature depends on what ran earlier in the process.
    Resolve that shape too, so the contract stays order-independent and still
    describes the upstream API vllm-gr patches.
    """
    current = target
    for _ in range(10):
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            current = wrapped
            continue
        try:
            nonlocals = inspect.getclosurevars(current).nonlocals
        except TypeError:
            break
        originals = [
            value
            for name, value in nonlocals.items()
            if name.startswith("original") and callable(value)
        ]
        if len(originals) != 1:
            break
        current = originals[0]
    return current


def _assert_parameter_prefix(
    target: Callable[..., object],
    expected: Sequence[str],
) -> None:
    parameters = list(inspect.signature(_innermost(target)).parameters)
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
    from vllm.entrypoints.generate.base.serving import GenerateBaseServing
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
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
    assert isinstance(GenerateBaseServing, type)
    assert isinstance(ChatCompletionRequest, type)
    assert isinstance(CompletionRequest, type)
    assert isinstance(AsyncLLM, type)
    assert callable(AsyncMPClient._send_input)
    assert callable(SyncMPClient._send_input)
    assert callable(SyncMPClient.add_request)
    assert callable(InprocClient.add_request)


def test_sampling_patch_covers_both_openai_prompt_shapes(monkeypatch) -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    from vllm_gr.entrypoints.openai.beam_search_patch import patch_sampling
    from vllm_gr.entrypoints.openai.protocol import to_beam_search_params

    monkeypatch.setattr(ChatCompletionRequest, "to_beam_search_params", None)
    monkeypatch.setattr(CompletionRequest, "to_beam_search_params", None)
    patch_sampling()

    assert ChatCompletionRequest.to_beam_search_params is to_beam_search_params
    assert CompletionRequest.to_beam_search_params is to_beam_search_params


@hardware_test(res={"npu": "A3"}, num_cards=1)  # type: ignore[misc]
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


pytestmark = pytest.mark.cpu_test
