# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors

from __future__ import annotations

import inspect
import re
from pathlib import Path

import yaml


def _parameter_names(callable_obj: object) -> list[str]:
    return list(inspect.signature(callable_obj).parameters)


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
    from vllm.v1.engine.core import EngineCoreProc

    assert hasattr(EngineCoreProc, "process_input_sockets")
    assert hasattr(EngineCoreProc, "run_engine_core")

    process_params = _parameter_names(EngineCoreProc.process_input_sockets)
    assert process_params[0] == "self"

    run_signature = inspect.signature(EngineCoreProc.run_engine_core)
    assert run_signature.parameters["dp_rank"].default == 0
    assert run_signature.parameters["local_dp_rank"].default == 0


def test_scheduler_contracts() -> None:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler

    assert _parameter_names(Scheduler.schedule) == ["self"]
    assert not hasattr(SchedulerOutput, "__slots__")


def test_engine_args_descriptor_contracts() -> None:
    from vllm.engine.arg_utils import EngineArgs

    assert isinstance(inspect.getattr_static(EngineArgs, "add_cli_args"), staticmethod)
    assert isinstance(inspect.getattr_static(EngineArgs, "from_cli_args"), classmethod)
    assert inspect.isfunction(inspect.getattr_static(EngineArgs, "create_engine_config"))


def test_attention_selector_contracts() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    from vllm.v1.attention.selector import _cached_get_attn_backend

    assert _parameter_names(_cached_get_attn_backend) == [
        "backend",
        "attn_selector_config",
        "num_heads",
    ]
    assert AttentionBackendEnum.CUSTOM is not None
    assert callable(register_backend)


def test_gpu_model_runner_private_method_contracts() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    assert _parameter_names(GPUModelRunner._prepare_inputs) == [
        "self",
        "scheduler_output",
        "num_scheduled_tokens",
    ]
    assert _parameter_names(GPUModelRunner._bookkeeping_sync) == [
        "self",
        "scheduler_output",
        "sampler_output",
        "logits",
        "hidden_states",
        "num_scheduled_tokens",
    ]


def test_chat_completion_stream_generator_contracts() -> None:
    import ast
    import textwrap

    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    orig_fn = inspect.unwrap(OpenAIServingChat.chat_completion_stream_generator)
    assert _parameter_names(orig_fn) == [
        "self",
        "request",
        "result_generator",
        "request_id",
        "model_name",
        "conversation",
        "tokenizer",
        "request_metadata",
        "reasoning_parser",
        "chat_template_kwargs",
    ]

    tree = ast.parse(textwrap.dedent(inspect.getsource(orig_fn)))
    request_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "request"
    }
    self_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }

    # 1-to-1 checklist mapping every upstream AST attribute read to the
    # corresponding gate condition or fast-path implementation in
    # vllm_gr/entrypoints/openai/beam_search_patch.py.
    expected_request_attr_coverage = {
        "_grammar_from_tool_parser": "gated: not getattr(request, '_grammar_from_tool_parser')",
        "echo": "gated: not request.echo",
        "include_reasoning": "inert when use_harmony=False and reasoning_parser is None",
        "logprobs": "gated: not (request.logprobs or request.top_logprobs)",
        "n": "handled in _beam_stream_fast: num_choices = request.n or 1",
        "return_prompt_text": "gated: not (request.return_token_ids or request.return_prompt_text)",
        "return_token_ids": "gated: not (request.return_token_ids or request.return_prompt_text)",
        "return_tokens_as_token_ids": "inert when request.logprobs is gated out",
        "stream_options": "include_usage in _beam_stream_fast; continuous_usage_stats gated",
        "tool_choice": "gated: tool_choice not in ('auto', 'required') and tool_parser is None",
        "tools": "gated: not request.tools",
        "top_logprobs": "gated: not (request.logprobs or request.top_logprobs)",
    }
    expected_self_attr_coverage = {
        "_convert_generation_error_to_streaming_response": "handled in except GenerationError",
        "_create_chat_logprobs": "inert when request.logprobs is gated out",
        "_create_remaining_args_delta": "inert when tool_parser is None",
        "_raise_if_error": "handled in _beam_stream_fast per output",
        "_should_check_for_unstreamed_tool_arg_tokens": "inert when tool_parser is None",
        "_should_stream_with_auto_tool_parsing": "inert when tool_parser is None",
        "create_streaming_error_response": "handled in _beam_stream_fast except Exception",
        "enable_force_include_usage": "handled in _beam_stream_fast via should_include_usage",
        "enable_log_deltas": "inert when enable_log_outputs and request_logger are gated",
        "enable_log_outputs": "gated in _server_stream_fast_path_eligible",
        "enable_prompt_tokens_details": "handled in _beam_stream_fast usage tail",
        "get_chat_request_role": "handled in _beam_stream_fast role chunk template",
        "parser_cls": "gated in _server_stream_fast_path_eligible",
        "request_logger": "gated in _server_stream_fast_path_eligible",
        "system_fingerprint": "handled in _beam_stream_fast content and usage chunks",
        "tool_call_id_type": "inert when parser_cls is None and tool_parser is None",
        "use_harmony": "gated in _server_stream_fast_path_eligible",
    }

    assert request_attrs == set(expected_request_attr_coverage)
    assert self_attrs == set(expected_self_attr_coverage)
