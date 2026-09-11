# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply the verified mapping to CI vLLM processes without importing vLLM early."""

import os

if os.environ.get("CI_CUDA_MAPPING_FILE"):
    import importlib.abc
    import importlib.machinery
    import sys
    from functools import wraps
    from types import ModuleType
    from typing import Any

    import cuda_devices as devices

    def install(module: ModuleType) -> None:
        platform = getattr(module, "NvmlCudaPlatform", None)
        if platform is None:
            raise devices.DeviceError("Verified CI mapping requires vLLM NvmlCudaPlatform")

        def physical_id(cls: Any, device_id: int = 0) -> int:
            gpus = devices.verified_devices()
            if type(device_id) is not int or not 0 <= device_id < len(gpus):
                raise devices.DeviceError(
                    f"Logical CUDA device outside verified allocation: {device_id}"
                )
            return int(gpus[device_id]["physical_index"])

        # CUDA_VISIBLE_DEVICES stays numeric and unchanged. Only the NVML platform's
        # logical-to-physical conversion uses the independently verified UUID mapping.
        platform.device_id_to_physical_device_id = classmethod(physical_id)
        connectivity = getattr(platform, "is_fully_connected", None)
        if connectivity is not None:

            def is_fully_connected(cls: Any, physical_device_ids: list[int]) -> bool:
                # vLLM's CUDA all-reduce callers pass ordinals taken from
                # CUDA_VISIBLE_DEVICES, despite naming the argument "physical".
                targets = {
                    gpu["index"]: int(gpu["physical_index"]) for gpu in devices.verified_devices()
                }
                if any(index not in targets for index in physical_device_ids):
                    raise devices.DeviceError("NVLink query outside verified CUDA allocation")
                return bool(connectivity([targets[index] for index in physical_device_ids]))

            platform.is_fully_connected = classmethod(is_fully_connected)

    def verify_initialized_device() -> None:
        """Fail before EngineCore's memory check if its already initialized CUDA target changed."""
        cuda = getattr(sys.modules.get("torch"), "cuda", None)
        if cuda is None or not cuda.is_initialized():
            raise devices.DeviceError("EngineCore device validation requires initialized CUDA")
        gpus = devices.verified_devices()
        logical_id = cuda.current_device()
        if type(logical_id) is not int or not 0 <= logical_id < len(gpus):
            raise devices.DeviceError(
                f"EngineCore CUDA device outside verified allocation: {logical_id}"
            )
        gpu = gpus[logical_id]
        actual = str(cuda.get_device_properties(logical_id).uuid)
        if devices.normalize_uuid(actual) != devices.normalize_uuid(gpu["uuid"]):
            raise devices.DeviceError(
                f"EngineCore CUDA UUID mismatch: expected {gpu['uuid']}, actual {actual}"
            )

    class _MappingLoader(importlib.abc.Loader):
        def __init__(self, original: Any) -> None:
            self.original = original

        def create_module(self, spec: Any) -> Any:
            return self.original.create_module(spec)

        def exec_module(self, module: ModuleType) -> None:
            self.original.exec_module(module)
            if module.__name__ == "vllm.platforms.cuda":
                install(module)
            else:
                original = module.request_memory  # type: ignore[attr-defined]

                @wraps(original)
                def request_memory(*args: Any, **kwargs: Any) -> Any:
                    verify_initialized_device()
                    return original(*args, **kwargs)

                module.request_memory = request_memory  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> Any:
            return getattr(self.original, name)

    class _MappingFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            if fullname not in {"vllm.platforms.cuda", "vllm.v1.worker.gpu_worker"}:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _MappingLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _MappingFinder())
