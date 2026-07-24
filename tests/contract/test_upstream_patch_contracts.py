# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors

from __future__ import annotations

import inspect
import re
from pathlib import Path


def _parameter_names(callable_obj: object) -> list[str]:
    return list(inspect.signature(callable_obj).parameters)


def test_patch_inventory_has_expected_ids() -> None:
    inventory = Path("docs/patch_inventory.yaml").read_text(encoding="utf-8")
    patch_ids = re.findall(r"^\s*- id: (GR-PATCH-\d{3})$", inventory, flags=re.MULTILINE)

    assert len(patch_ids) == len(set(patch_ids))
    assert patch_ids == [f"GR-PATCH-{index:03d}" for index in range(15)]


def test_engine_wire_values_do_not_conflict() -> None:
    from vllm.v1.engine import EngineCoreRequestType

    expected_gr_values = {
        b"\x06": "ADD_BATCH",
        b"\x07": "BEAM_REQUEST_STEP_UPDATE",
    }
    for value, expected_name in expected_gr_values.items():
        owners = [
            name
            for name, member in EngineCoreRequestType.__members__.items()
            if member.value == value
        ]
        assert owners in ([], [expected_name])


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
