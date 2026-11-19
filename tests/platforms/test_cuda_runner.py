# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA workers install the generation selected by the engine configuration."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.config import GRConfig
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.cuda import provider as cuda
from vllm_gr.platforms.cuda import runner
from vllm_gr.platforms.plan import BACKEND_PLAN_KEY
from vllm_gr.platforms.planning import normalize_backend_request
from vllm_gr.platforms.services import LegacyPlanningProvider
from vllm_gr.v1.worker.runner_lifecycle import (
    NativeRunnerAdapter,
    has_runner_lifecycle,
    install_runner_lifecycle,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture
def workers(monkeypatch: pytest.MonkeyPatch) -> Any:
    class NativeRunner:
        def execute_model(self) -> str:
            return "executed"

        def prepare_inputs(self) -> None:
            pass

    installers = {}
    classes = {}
    for use_v2, name in ((False, "v1"), (True, "v2")):
        identity = runner.runner_id(use_v2)
        native = type(name, (NativeRunner,), {})
        classes[use_v2] = native
        installers[identity] = SimpleNamespace(
            discover_worker_runners=lambda native=native: ((native.__name__, native),),
            is_patched=lambda cls, identity=identity: has_runner_lifecycle(cls, identity),
            install=Mock(
                side_effect=lambda native=native,
                identity=identity,
                name=name: install_runner_lifecycle(
                    native,
                    identity,
                    prepare="prepare_inputs",
                    stage_factory=lambda **kwargs: SimpleNamespace(
                        implementation_id="cuda." + name
                    ),
                )
            ),
        )

    class Worker:
        def __init__(self, config: Any):
            self.vllm_config = config
            self.use_v2_model_runner = config.use_v2_model_runner

        def init_device(self) -> Any:
            return classes[self.use_v2_model_runner]()

    imports = Mock(side_effect=installers.__getitem__)
    monkeypatch.setattr(runner, "import_module", imports)
    monkeypatch.setattr(
        runner, "discover_worker_runners", lambda: (("CUDA Worker.init_device", Worker),)
    )
    monkeypatch.setattr(cuda, "install_worker_hooks", Mock())
    hooks = CUDA_BACKEND.get_model_runner_hooks()
    hooks.install()
    patched = Worker.init_device
    hooks.install()
    assert Worker.init_device is patched and hooks.is_patched(Worker)
    imports.assert_not_called()
    return Worker, installers


def admit(config: Any) -> Any:
    GRConfig().write_to_vllm_config(config)
    plan = CUDA_BACKEND.resolve(normalize_backend_request(config, CUDA_BACKEND))
    config.additional_config[BACKEND_PLAN_KEY] = plan.to_wire()
    return plan


def test_each_worker_uses_its_admitted_runner_and_stage(workers: Any) -> None:
    Worker, installers = workers
    for use_v2 in (False, True, True, False):
        config = SimpleNamespace(additional_config={}, use_v2_model_runner=use_v2)
        plan = admit(config)
        native = Worker(config).init_device()
        adapter = CUDA_BACKEND.create_model_runner_adapter(plan=plan, native_runner=native)
        assert isinstance(adapter, NativeRunnerAdapter) and adapter.stage is not None
        assert adapter.implementation_id == plan.choices.to_dict()["runner"]
        assert adapter.stage.implementation_id == plan.choices.to_dict()["stage"]
        assert adapter.execute_model() == "executed"
    assert all(hooks.install.call_count == 1 for hooks in installers.values())


def test_watermark_override_selects_v2_with_the_v1_environment(
    workers: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm import envs
    from vllm.config import VllmConfig

    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)

    class Config(SimpleNamespace):
        use_v2_model_runner = VllmConfig.use_v2_model_runner

    config = Config(
        additional_config={},
        attention_config=SimpleNamespace(hisparse_config=None),
        watermark_config=object(),
    )
    plan = admit(config)
    Worker, installers = workers
    native = Worker(config).init_device()
    adapter = CUDA_BACKEND.create_model_runner_adapter(plan=plan, native_runner=native)
    assert isinstance(adapter, NativeRunnerAdapter) and adapter.stage is not None
    assert plan.request.native.to_dict()["use_v2_model_runner"] is True
    assert adapter.implementation_id == runner.V2_RUNNER
    assert adapter.stage.implementation_id == plan.choices.to_dict()["stage"] == "cuda.v2"
    installers[runner.V1_RUNNER].install.assert_not_called()


def test_mismatch_and_incomplete_install_fail_before_runner_construction(workers: Any) -> None:
    Worker, installers = workers
    config = SimpleNamespace(additional_config={}, use_v2_model_runner=True)
    admit(config)
    worker = Worker(config)
    worker.use_v2_model_runner = False
    with pytest.raises(RuntimeError, match="differs from the admitted"):
        worker.init_device()
    installers[runner.V1_RUNNER].install.assert_not_called()
    worker.use_v2_model_runner = True
    hooks = installers[runner.V2_RUNNER]
    install = hooks.install
    hooks.install = Mock()
    with pytest.raises(RuntimeError, match="Failed to install selected"):
        worker.init_device()
    hooks.install = install
    native = worker.init_device()
    assert hooks.is_patched(type(native))


def test_legacy_provider_can_delegate_to_the_cuda_installer(workers: Any) -> None:
    Worker, _ = workers
    config = SimpleNamespace(additional_config={}, use_v2_model_runner=True)
    GRConfig().write_to_vllm_config(config)
    provider = LegacyPlanningProvider(CUDA_BACKEND)
    plan = provider.resolve(normalize_backend_request(config, CUDA_BACKEND))
    config.additional_config[BACKEND_PLAN_KEY] = plan.to_wire()
    native = Worker(config).init_device()
    adapter = provider.create_model_runner_adapter(plan=plan, native_runner=native)
    assert adapter.implementation_id == runner.V2_RUNNER


def test_installer_module_can_move_without_changing_a_plan_or_readiness(
    workers: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    Worker, installers = workers
    config = SimpleNamespace(additional_config={}, use_v2_model_runner=True)
    plan = admit(config)
    new_path = "example.version_2.installer"
    installers[new_path] = installers.pop(runner.V2_RUNNER)
    monkeypatch.setitem(runner._INSTALLER_MODULES, runner.V2_RUNNER, new_path)
    native = Worker(config).init_device()
    adapter = CUDA_BACKEND.create_model_runner_adapter(plan=plan, native_runner=native)
    assert isinstance(adapter, NativeRunnerAdapter) and adapter.stage is not None
    assert runner.is_patched(type(native))
    assert adapter.implementation_id == plan.choices.to_dict()["runner"] == runner.V2_RUNNER
    assert adapter.stage.implementation_id == "cuda.v2"
