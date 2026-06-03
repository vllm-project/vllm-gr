# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _purge_modules(*prefixes: str) -> dict[str, types.ModuleType]:
    saved: dict[str, types.ModuleType] = {}
    for name, module in list(sys.modules.items()):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            saved[name] = module
            sys.modules.pop(name, None)
    return saved


def _restore_modules(saved: dict[str, types.ModuleType]) -> None:
    for name, module in saved.items():
        sys.modules[name] = module


def _install_stub_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


class TestYuanrongMonkeyPatch(unittest.TestCase):
    backend_module_name = (
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend"
    )
    pool_worker_module_name = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker"

    def _install_fake_runtime(
        self,
        *,
        device_type: str,
        with_pool_worker: bool,
    ) -> dict[str, types.ModuleType]:
        saved = _purge_modules(
            "vllm",
            "vllm_ascend",
            "vllm_gr.distributed",
            "vllm_gr.patch",
        )

        fake_vllm = types.ModuleType("vllm")
        fake_platforms = types.ModuleType("vllm.platforms")
        fake_config = types.ModuleType("vllm.config")
        fake_logger = types.ModuleType("vllm.logger")
        fake_utils = types.ModuleType("vllm.utils")
        fake_network_utils = types.ModuleType("vllm.utils.network_utils")

        fake_platforms.current_platform = types.SimpleNamespace(device_type=device_type)

        class ParallelConfig:
            def __init__(self, rank: int = 0) -> None:
                self.rank = rank

        class Backend:
            pass

        fake_config.ParallelConfig = ParallelConfig
        fake_logger.logger = types.SimpleNamespace(
            debug=lambda *_args, **_kwargs: None,
            info=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None,
        )
        fake_network_utils.split_host_port = lambda value: tuple(value.rsplit(":", 1))

        sys.modules.update(
            {
                "vllm": fake_vllm,
                "vllm.platforms": fake_platforms,
                "vllm.config": fake_config,
                "vllm.logger": fake_logger,
                "vllm.utils": fake_utils,
                "vllm.utils.network_utils": fake_network_utils,
            }
        )

        _install_stub_package("vllm_ascend")
        _install_stub_package("vllm_ascend.distributed")
        _install_stub_package("vllm_ascend.distributed.kv_transfer")
        _install_stub_package("vllm_ascend.distributed.kv_transfer.kv_pool")
        ascend_store_pkg = _install_stub_package(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"
        )
        backend_pkg = _install_stub_package(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend"
        )

        backend_module = types.ModuleType(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend"
        )
        backend_module.Backend = Backend
        sys.modules[backend_module.__name__] = backend_module
        backend_pkg.backend = backend_module

        if with_pool_worker:
            pool_worker = types.ModuleType(self.pool_worker_module_name)
            pool_worker.backend_map = {
                "memcache": object(),
            }
            sys.modules[self.pool_worker_module_name] = pool_worker
            setattr(ascend_store_pkg, "pool_worker", pool_worker)

        return saved

    def test_patch_skips_non_npu_platform(self) -> None:
        saved = self._install_fake_runtime(device_type="cuda", with_pool_worker=True)
        try:
            module = importlib.import_module("vllm_gr.patch")
            patch_count = module.re_apply_patches()

            self.assertGreaterEqual(patch_count, 0)
            self.assertNotIn(self.backend_module_name, sys.modules)
            pool_worker = sys.modules[self.pool_worker_module_name]
            self.assertNotIn("yuanrong", pool_worker.backend_map)
        finally:
            _purge_modules("vllm", "vllm_ascend", "vllm_gr.distributed", "vllm_gr.patch")
            _restore_modules(saved)

    def test_patch_registers_yuanrong_backend_on_npu(self) -> None:
        saved = self._install_fake_runtime(device_type="npu", with_pool_worker=True)
        try:
            module = importlib.import_module("vllm_gr.patch")
            patch_count = module.re_apply_patches()

            self.assertGreater(patch_count, 0)
            pool_worker = sys.modules[self.pool_worker_module_name]
            backend_cls = pool_worker.backend_map["yuanrong"]

            self.assertEqual(backend_cls.__name__, "YuanrongBackend")
            self.assertEqual(backend_cls.__module__, self.backend_module_name)

            synthetic_module = importlib.import_module(self.backend_module_name)
            self.assertIs(synthetic_module.YuanrongBackend, backend_cls)
            self.assertIs(pool_worker.YuanrongBackend, backend_cls)
        finally:
            _purge_modules("vllm", "vllm_ascend", "vllm_gr.distributed", "vllm_gr.patch")
            _restore_modules(saved)

    def test_patch_is_idempotent(self) -> None:
        saved = self._install_fake_runtime(device_type="npu", with_pool_worker=True)
        try:
            module = importlib.import_module("vllm_gr.patch")
            first = module.apply_patches_if_needed()
            second = module.apply_patches_if_needed()

            self.assertGreater(first, 0)
            self.assertEqual(second, 0)
        finally:
            _purge_modules("vllm", "vllm_ascend", "vllm_gr.distributed", "vllm_gr.patch")
            _restore_modules(saved)

    def test_patch_skips_cleanly_when_vllm_ascend_pool_worker_missing(self) -> None:
        saved = self._install_fake_runtime(device_type="npu", with_pool_worker=False)
        try:
            module = importlib.import_module("vllm_gr.patch")
            patch_count = module.re_apply_patches()

            self.assertGreaterEqual(patch_count, 0)
            self.assertNotIn(self.backend_module_name, sys.modules)
        finally:
            _purge_modules("vllm", "vllm_ascend", "vllm_gr.distributed", "vllm_gr.patch")
            _restore_modules(saved)

    def test_mooncake_import_shim_allows_non_mooncake_backends_to_load(self) -> None:
        saved = self._install_fake_runtime(device_type="npu", with_pool_worker=False)
        try:
            module = importlib.import_module("vllm_gr.patch")
            original_import_module = module.importlib.import_module

            def _fake_import(name: str, package=None) -> None:
                if name == module._ASCEND_MOONCAKE_MODULE:
                    raise ModuleNotFoundError("No module named 'mooncake'", name="mooncake")
                return original_import_module(name, package)

            with mock.patch.object(module.importlib, "import_module", side_effect=_fake_import):
                patch_count = module._install_mooncake_backend_shim_if_needed()

            self.assertEqual(patch_count, 1)
            shim_module = importlib.import_module(module._ASCEND_MOONCAKE_MODULE)
            self.assertTrue(hasattr(shim_module, "MooncakeBackend"))
            self.assertEqual(shim_module.MooncakeBackend.__name__, "MooncakeBackend")
        finally:
            _purge_modules("vllm", "vllm_ascend", "vllm_gr.distributed", "vllm_gr.patch")
            _restore_modules(saved)
