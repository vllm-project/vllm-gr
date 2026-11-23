# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA workers install the generation selected by the engine configuration."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.helpers.vllm_subprocess import run_vllm_script
from vllm_gr.config import GRConfig
from vllm_gr.config.backend_plan import BACKEND_PLAN_KEY
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.planning import normalize_backend_request
from vllm_gr.platforms.services import LegacyPlanningProvider
from vllm_gr.v1.worker.cuda import runner_installer as runner
from vllm_gr.v1.worker.runner_lifecycle import (
    NativeRunnerAdapter,
    has_runner_lifecycle,
    install_runner_lifecycle,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture  # type: ignore[misc]
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
    monkeypatch.setattr(runner, "install_worker_hooks", Mock())
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
        assert isinstance(adapter, NativeRunnerAdapter)
        assert adapter.implementation_id == plan.choices.to_dict()["runner"]
        assert adapter.execute_model() == "executed"
    assert all(hooks.install.call_count == 1 for hooks in installers.values())


def test_cuda_startup_preserves_native_runner_fallback() -> None:
    # Isolate native config from vllm-ascend monkey patches in other tests.
    run_vllm_script(
        "from tests.platforms.test_cuda_runner import _assert_native_runner_fallback\n"
        "_assert_native_runner_fallback()"
    )


def _assert_native_runner_fallback() -> None:
    import os
    from unittest.mock import patch

    from vllm import platforms
    from vllm.config import vllm as native

    from vllm_gr.engine.startup import configure_model_runner_default

    class Config(SimpleNamespace):
        use_v2_model_runner = native.VllmConfig.use_v2_model_runner
        _get_v2_model_runner_unsupported_features = (
            native.VllmConfig._get_v2_model_runner_unsupported_features
        )

    config = Config(
        attention_config=SimpleNamespace(hisparse_config=None),
        model_config=SimpleNamespace(logits_processors=[]),
        speculative_config=None,
        compilation_config=SimpleNamespace(
            mode=native.CompilationMode.NONE, pass_config=SimpleNamespace(enable_sp=False)
        ),
        parallel_config=SimpleNamespace(
            distributed_executor_backend=None, use_ubatching=False, enable_elastic_ep=False
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner_flag = "VLLM_USE_V2_MODEL_RUNNER"
    with (
        patch.dict(os.environ),
        patch.object(native, "HAS_TRITON", True),
        patch.object(platforms, "current_platform", SimpleNamespace(is_rocm=lambda: False)),
        patch("importlib.metadata.entry_points", return_value=()),
        patch(
            "vllm_gr.platforms.identity.current_platform_identity",
            return_value=SimpleNamespace(platform_enum="CUDA", device_type="cuda"),
        ),
    ):
        os.environ.pop(runner_flag, None)
        configure_model_runner_default()
        assert config.use_v2_model_runner is True
        config.model_config.logits_processors = ["example.CustomLogitsProcessor"]
        assert config.use_v2_model_runner is False

        os.environ[runner_flag] = "1"
        configure_model_runner_default()
        assert config.use_v2_model_runner is True
        with pytest.raises(ValueError, match="custom logits processors"):
            native.VllmConfig._validate_v2_model_runner(config)

        os.environ[runner_flag] = "0"
        configure_model_runner_default()
        config.model_config.logits_processors = []
        assert config.use_v2_model_runner is False


def test_watermark_override_selects_v2_with_the_v1_environment(
    workers: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)

    class Config(SimpleNamespace):
        # Upstream VllmConfig semantics under test: a watermark forces
        # ModelRunner V2 even with VLLM_USE_V2_MODEL_RUNNER=0. Declared
        # inline instead of binding VllmConfig.use_v2_model_runner, which
        # vllm-ascend replaces with an env-only implementation when the
        # Ascend platform activates; earlier tests trigger that activation,
        # which made this cpu-tier test host- and order-dependent.
        @property
        def use_v2_model_runner(self) -> bool:
            if getattr(self, "watermark_config", None) is not None:
                return True
            return bool(envs.VLLM_USE_V2_MODEL_RUNNER)

    config = Config(
        additional_config={},
        attention_config=SimpleNamespace(hisparse_config=None),
        watermark_config=object(),
    )
    plan = admit(config)
    Worker, installers = workers
    native = Worker(config).init_device()
    adapter = CUDA_BACKEND.create_model_runner_adapter(plan=plan, native_runner=native)
    assert isinstance(adapter, NativeRunnerAdapter)
    assert plan.request.native.to_dict()["use_v2_model_runner"] is True
    assert adapter.implementation_id == runner.V2_RUNNER
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
    assert isinstance(adapter, NativeRunnerAdapter)
    assert runner.is_patched(type(native))
    assert adapter.implementation_id == plan.choices.to_dict()["runner"] == runner.V2_RUNNER
