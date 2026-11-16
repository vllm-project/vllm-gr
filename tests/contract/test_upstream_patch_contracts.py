# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: ignore-errors

from __future__ import annotations

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


@pytest.mark.l1
def test_core_hot_update_utility_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    from concurrent.futures import Future
    from unittest.mock import Mock

    from vllm.v1.engine.core import EngineCoreProc

    from vllm_gr.v1.engine import hot_update

    assert _parameter_names(EngineCoreProc.pause_scheduler) == ["self", "mode", "clear_cache"]
    assert _parameter_names(EngineCoreProc.resume_scheduler) == ["self"]

    class Core(EngineCoreProc):
        pass

    native_pause, native_resume = Core.pause_scheduler, Core.resume_scheduler
    hot_update.install_core_hot_update_hooks(Core)
    installed = {
        name: getattr(Core, name) for name in ("service_status", "pause", "resume", "reload_trie")
    }
    hot_update.install_core_hot_update_hooks(Core)
    assert all(getattr(Core, name) is method for name, method in installed.items())
    assert Core.pause_scheduler is native_pause and Core.resume_scheduler is native_resume

    core = object.__new__(Core)
    pending = Future()
    native_pause_call = Mock(return_value=pending)
    native_resume_call = Mock()
    monkeypatch.setattr(core, "pause_scheduler", native_pause_call)
    monkeypatch.setattr(core, "resume_scheduler", native_resume_call)
    assert getattr(core, "pause")() is pending
    native_pause_call.assert_called_once_with(mode="wait", clear_cache=True)
    assert not pending.done()
    pending.set_result(None)
    assert getattr(core, "resume")() == {"ok": True}
    native_resume_call.assert_called_once_with()


def test_scheduler_contracts() -> None:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler

    assert _parameter_names(Scheduler.schedule) == ["self", "throttle_prefills"]
    assert inspect.signature(Scheduler.schedule).parameters["throttle_prefills"].default is False
    assert not hasattr(SchedulerOutput, "__slots__")


def test_engine_args_descriptor_contracts() -> None:
    from vllm.engine.arg_utils import EngineArgs

    from vllm_gr.engine.arg_utils_gr import patch_add_cli_args

    patch_add_cli_args()
    assert isinstance(inspect.getattr_static(EngineArgs, "add_cli_args"), staticmethod)
    assert EngineArgs.add_cli_args is object.__new__(EngineArgs).add_cli_args
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


pytestmark = pytest.mark.cpu_test
