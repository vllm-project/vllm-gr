# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Install CUDA runner hooks when the native Worker has resolved its generation."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from threading import RLock
from typing import Any, cast

from vllm_gr.config.backend_plan import read_backend_plan
from vllm_gr.platforms.interface import ModelRunnerInstaller

# These IDs are already serialized in admitted plans and reported in readiness.
V1_RUNNER = "vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch"
V2_RUNNER = "vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_patch"
# Deployment paths can move independently of the historical plan values above.
_INSTALLER_MODULES = {
    V1_RUNNER: "vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch",
    V2_RUNNER: "vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_patch",
}
_INSTALL_LOCK = RLock()
_WORKER_MARKER = "_gr_cuda_runner_selection"


def runner_id(use_v2: bool) -> str:
    return V2_RUNNER if use_v2 else V1_RUNNER


def _installer(implementation_id: str) -> ModelRunnerInstaller:
    return cast(ModelRunnerInstaller, import_module(_INSTALLER_MODULES[implementation_id]))


def discover_worker_runners() -> tuple[tuple[str, type[Any]], ...]:
    # Initial process verification checks the entry point that constructs the
    # selected runner. Its concrete class is verified inside that entry point.
    from vllm.v1.worker.gpu_worker import Worker

    return (("CUDA Worker.init_device", Worker),)


def is_patched(worker_class: type[Any]) -> bool:
    implementation_id = getattr(worker_class, "_gr_runner_implementation_id", None)
    if implementation_id in _INSTALLER_MODULES:
        return _installer(implementation_id).is_patched(worker_class)
    return bool(getattr(getattr(worker_class, "init_device", None), _WORKER_MARKER, False))


def install() -> None:
    with _INSTALL_LOCK:
        install_worker_hooks()
        ((_, worker_class),) = discover_worker_runners()
        if is_patched(worker_class):
            return
        original = worker_class.init_device

        @wraps(original)
        def init_device(worker: Any, *args: Any, **kwargs: Any) -> Any:
            # Worker.__init__ stores VllmConfig.use_v2_model_runner, including
            # upstream overrides such as watermarking. init_device branches on
            # this exact value when constructing the native runner.
            selected = runner_id(worker.use_v2_model_runner)
            plan = read_backend_plan(worker.vllm_config)
            if plan is not None:
                admitted = plan.choices.to_dict().get("runner")
                if admitted == "legacy-provider-hooks":
                    # API-v1 adapters delegate to this installer without a
                    # runner implementation ID. The normalized request still
                    # records the generation selected by the engine.
                    admitted = runner_id(plan.request.native.to_dict()["use_v2_model_runner"])
                if admitted != selected:
                    raise RuntimeError(
                        "CUDA Worker generation differs from the admitted backend plan"
                    )
            installer = _installer(selected)
            with _INSTALL_LOCK:
                runners = installer.discover_worker_runners()
                if not runners:
                    raise RuntimeError(f"Selected CUDA runner {selected!r} is unavailable")
                if not all(installer.is_patched(cls) for _, cls in runners):
                    installer.install()
                    if not all(installer.is_patched(cls) for _, cls in runners):
                        raise RuntimeError(f"Failed to install selected CUDA runner {selected!r}")
            return original(worker, *args, **kwargs)

        setattr(init_device, _WORKER_MARKER, True)
        worker_class.init_device = init_device


def install_worker_utilities() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    from vllm_gr.v1.worker.model_runner_patch import install_worker_utility_hooks

    install_worker_utility_hooks(Worker)


def install_worker_hooks() -> None:
    """Both CUDA runner generations use the same Worker and NumPy staging."""
    from vllm_gr.v1.worker.cuda.numpy_staging import install_numpy_staging

    install_worker_utilities()
    install_numpy_staging()
