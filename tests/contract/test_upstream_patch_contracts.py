# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors

"""Contract checks for the private upstream APIs patched by vllm-gr."""

from __future__ import annotations

import importlib.util
import inspect
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import yaml


def _assert_parameters(
    target: Callable[..., object],
    expected: Sequence[str],
    *,
    exact: bool = False,
) -> None:
    """Check the unwrapped prefix and, when required, the exposed full signature."""
    parameters = list(inspect.signature(inspect.unwrap(target)).parameters)
    assert parameters[: len(expected)] == list(expected)
    if exact:
        assert list(inspect.signature(target).parameters) == list(expected)


def test_patch_inventories_are_aligned_and_complete() -> None:
    inventory = yaml.safe_load(Path("docs/patch_inventory.yaml").read_text(encoding="utf-8"))
    patches = inventory["patches"]
    patch_ids = [patch["id"] for patch in patches]
    required_fields = {
        "owner",
        "installer",
        "activation",
        "validator_or_test",
        "process_roles",
        "removal_condition",
    }

    assert len(patch_ids) == len(set(patch_ids))
    assert all(re.fullmatch(r"GR-PATCH-\d{3}", patch_id) for patch_id in patch_ids)
    assert all(required_fields <= patch.keys() for patch in patches)

    runtime_inventory = Path("docs/runtime_patch_inventory.md").read_text(encoding="utf-8")
    runtime_ids = re.findall(r"^### (GR-PATCH-\d{3})", runtime_inventory, re.MULTILINE)
    assert runtime_ids == patch_ids


def test_engine_core_method_contracts() -> None:
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    _assert_parameters(EngineCore.__init__, ("self", "vllm_config", "executor_class"))
    assert hasattr(EngineCoreProc, "process_input_sockets")
    assert hasattr(EngineCoreProc, "run_engine_core")
    assert callable(EngineCoreProc.run_engine_core)

    process_params = list(inspect.signature(EngineCoreProc.process_input_sockets).parameters)
    assert process_params[0] == "self"

    run_signature = inspect.signature(EngineCoreProc.run_engine_core)
    assert {"dp_rank", "local_dp_rank"}.issubset(run_signature.parameters)
    assert run_signature.parameters["dp_rank"].default == 0
    assert run_signature.parameters["local_dp_rank"].default == 0


def test_scheduler_contracts() -> None:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler

    _assert_parameters(Scheduler.schedule, ("self",), exact=True)
    assert not hasattr(SchedulerOutput, "__slots__")


def test_engine_args_descriptor_contracts() -> None:
    from vllm.engine.arg_utils import EngineArgs

    _assert_parameters(EngineArgs.add_cli_args, ("parser",))
    assert callable(EngineArgs.create_engine_config)
    assert callable(EngineArgs.from_cli_args)
    assert isinstance(inspect.getattr_static(EngineArgs, "add_cli_args"), staticmethod)
    assert isinstance(inspect.getattr_static(EngineArgs, "from_cli_args"), classmethod)
    assert inspect.isfunction(inspect.getattr_static(EngineArgs, "create_engine_config"))


def test_attention_selector_contracts() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    from vllm.v1.attention.selector import _cached_get_attn_backend

    _assert_parameters(
        _cached_get_attn_backend,
        ("backend", "attn_selector_config", "num_heads"),
        exact=True,
    )
    assert AttentionBackendEnum.CUSTOM is not None
    assert callable(register_backend)


def test_gpu_model_runner_private_method_contracts() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _assert_parameters(
        GPUModelRunner.execute_model,
        ("self", "scheduler_output", "intermediate_tensors"),
    )
    _assert_parameters(
        GPUModelRunner._prepare_inputs,
        ("self", "scheduler_output", "num_scheduled_tokens"),
        exact=True,
    )
    _assert_parameters(
        GPUModelRunner._bookkeeping_sync,
        (
            "self",
            "scheduler_output",
            "sampler_output",
            "logits",
            "hidden_states",
            "num_scheduled_tokens",
        ),
        exact=True,
    )


def test_vllm_serving_and_client_patch_contract() -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.engine.serving import OpenAIServing
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.logprobs import FlatLogprobs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient

    _assert_parameters(
        OpenAIServingModels.__init__,
        ("self", "engine_client", "base_model_paths", "lora_modules"),
    )
    assert callable(FlatLogprobs.append)
    assert isinstance(OpenAIServing, type)
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


def test_vllm_ascend_model_runner_patch_contract() -> None:
    if importlib.util.find_spec("vllm_ascend") is None:
        pytest.skip("vllm-ascend is not installed")

    # Import through the worker entry point so vllm-ascend initializes its ops
    # package before loading model_runner_v1. A direct model_runner_v1 import
    # can otherwise enter the device_op/ops circular-import path.
    from vllm_ascend.worker.worker import NPUModelRunner, NPUWorker

    _assert_parameters(NPUWorker.init_device, ("self",))
    _assert_parameters(
        NPUModelRunner.execute_model,
        ("self", "scheduler_output", "intermediate_tensors"),
    )
    _assert_parameters(
        NPUModelRunner._prepare_inputs,
        ("self", "scheduler_output", "num_scheduled_tokens"),
    )
    _assert_parameters(
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
