# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest
import regex as re
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


def _assert_return_layout(
    source: str, class_name: str, method_name: str, fields: tuple[str, ...]
) -> None:
    """Review-sensitive source contract, not a general return-type inference tool."""
    label = f"{class_name}.{method_name}: native return contract changed; review GR adaptation"
    classes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1, label
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    assert len(methods) == 1, label
    method = methods[0]
    annotation = method.returns
    assert isinstance(annotation, ast.Subscript), label
    assert isinstance(annotation.value, ast.Name) and annotation.value.id == "tuple", label
    assert isinstance(annotation.slice, ast.Tuple), label
    assert len(annotation.slice.elts) == len(fields), label
    assert not any(
        isinstance(item, ast.Constant) and item.value is Ellipsis for item in annotation.slice.elts
    ), label

    returns = []
    pending = list(method.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Return):
            returns.append(node.value)
        else:
            pending.extend(ast.iter_child_nodes(node))
    assert len(returns) == 1 and isinstance(returns[0], ast.Tuple), label
    actual = tuple(ast.unparse(item) for item in returns[0].elts)
    assert actual == fields, f"{label}: expected {fields}, got {actual}"


def _native_runner_source(package: str, relative_path: str, *, required: bool = True) -> str:
    # Resolve the package root only: importing NPUModelRunner initializes device
    # libraries and can enter Ascend's worker/ops circular import path.
    spec = importlib.util.find_spec(package)
    if spec is None:
        if required:
            pytest.fail(f"{package} must be installed for native return-contract checks")
        pytest.skip(f"{package} is not installed; NPU return contracts not validated")
    assert spec.submodule_search_locations, f"{package}: cannot locate native package sources"
    for root in spec.submodule_search_locations:
        path = Path(root) / relative_path
        if path.is_file():
            return path.read_text(encoding="utf-8")
    pytest.fail(f"{package}/{relative_path}: native source missing; review return contracts")


def test_gpu_prepare_native_return_layout() -> None:
    _assert_return_layout(
        _native_runner_source("vllm", "v1/worker/gpu_model_runner.py"),
        "GPUModelRunner",
        "_prepare_inputs",
        ("logits_indices", "spec_decode_metadata"),
    )


@pytest.mark.parametrize(
    ("method", "fields"),
    [
        (
            "_prepare_inputs",
            (
                "logits_indices",
                "spec_decode_metadata",
                "total_num_scheduled_tokens",
                "num_scheduled_tokens_compressed_list",
            ),
        ),
        (
            "_bookkeeping_sync",
            (
                "logprobs_lists",
                "valid_sampled_token_ids",
                "prompt_logprobs_dict",
                "req_ids_output_copy",
                "req_id_to_index_output_copy",
                "invalid_req_indices",
            ),
        ),
    ],
)
def test_npu_native_return_layout(
    request: pytest.FixtureRequest, method: str, fields: tuple[str, ...]
) -> None:
    source = _native_runner_source(
        "vllm_ascend",
        "worker/model_runner_v1.py",
        required=request.config.getoption("--require-npu-contracts"),
    )
    _assert_return_layout(source, "NPUModelRunner", method, fields)


@pytest.mark.parametrize("returned", ["second, first", "prefix, first, second", "first,"])
def test_native_return_contract_detects_reordered_inserted_and_removed_fields(
    returned: str,
) -> None:
    source = f"class Runner:\n    def prepare(self) -> tuple[object, object]:\n        return ({returned})\n"
    with pytest.raises(AssertionError, match="native return contract changed"):
        _assert_return_layout(source, "Runner", "prepare", ("first", "second"))


def test_native_return_contract_ignores_nested_scopes() -> None:
    source = """
class Runner:
    def prepare(self) -> tuple[object, object]:
        def helper():
            return second, first
        class Nested:
            def prepare(self):
                return second, first
        return first, second
"""
    _assert_return_layout(source, "Runner", "prepare", ("first", "second"))
