# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from pathlib import Path


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


class _FakeBlob:
    def __init__(self, addr: int, size: int):
        self.addr = addr
        self.size = size


class _FakeDeviceBlobList:
    def __init__(self, dev_idx: int, blob_list: list[_FakeBlob]):
        self.dev_idx = dev_idx
        self.blob_list = blob_list


class _FakeSetParam:
    def __init__(self):
        self.write_mode = None


class _FakeWriteMode:
    NONE_L2_CACHE_EVICT = "NONE_L2_CACHE_EVICT"


class _FakeHeteroClient:
    instances: list["_FakeHeteroClient"] = []

    def __init__(self, *_args, **_kwargs):
        self.exist_calls: list[list[str]] = []
        self.put_calls: list[tuple[list[str], list[_FakeDeviceBlobList]]] = []
        self.get_calls: list[tuple[list[str], list[_FakeDeviceBlobList]]] = []
        type(self).instances.append(self)

    def init(self):
        return None

    def exist(self, keys: list[str]):
        self.exist_calls.append(list(keys))
        return [True] * len(keys)

    def mset_d2h(self, keys: list[str], blob_lists: list[_FakeDeviceBlobList], _set_param):
        self.put_calls.append((list(keys), list(blob_lists)))

    def mget_h2d(self, keys: list[str], blob_lists: list[_FakeDeviceBlobList], _timeout_ms: int):
        self.get_calls.append((list(keys), list(blob_lists)))
        return []


class _FakeLogger:
    def __init__(self):
        self.info_calls: list[tuple[object, ...]] = []
        self.warning_calls: list[tuple[object, ...]] = []
        self.error_calls: list[tuple[object, ...]] = []

    def info(self, *args, **_kwargs):
        self.info_calls.append(args)

    def warning(self, *args, **_kwargs):
        self.warning_calls.append(args)

    def error(self, *args, **_kwargs):
        self.error_calls.append(args)


class TestYuanrongBackend(unittest.TestCase):
    def setUp(self):
        self.saved_modules = _purge_modules(
            "vllm",
            "vllm_ascend",
            "yr",
            "vllm_gr.distributed",
        )
        self.saved_env = {
            "DS_WORKER_ADDR": os.environ.get("DS_WORKER_ADDR"),
        }

        fake_vllm = types.ModuleType("vllm")
        fake_config = types.ModuleType("vllm.config")
        fake_logger = types.ModuleType("vllm.logger")
        fake_utils = types.ModuleType("vllm.utils")
        fake_network_utils = types.ModuleType("vllm.utils.network_utils")
        fake_vllm_ascend = types.ModuleType("vllm_ascend")
        fake_vllm_ascend.__path__ = []
        fake_distributed = types.ModuleType("vllm_ascend.distributed")
        fake_distributed.__path__ = []
        fake_kv_transfer = types.ModuleType("vllm_ascend.distributed.kv_transfer")
        fake_kv_transfer.__path__ = []
        fake_kv_pool = types.ModuleType("vllm_ascend.distributed.kv_transfer.kv_pool")
        fake_kv_pool.__path__ = []
        fake_ascend_store = types.ModuleType(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"
        )
        fake_ascend_store.__path__ = []
        fake_backend_pkg = types.ModuleType(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend"
        )
        fake_backend_pkg.__path__ = []
        fake_backend_module = types.ModuleType(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend"
        )

        class ParallelConfig:
            def __init__(self, rank: int = 0):
                self.rank = rank

        class Backend:
            pass

        self.fake_logger = _FakeLogger()
        fake_config.ParallelConfig = ParallelConfig
        fake_backend_module.Backend = Backend
        fake_logger.logger = self.fake_logger
        fake_network_utils.split_host_port = lambda value: tuple(value.rsplit(":", 1))

        fake_yr = types.ModuleType("yr")
        fake_yr.__path__ = []
        fake_yr_datasystem = types.ModuleType("yr.datasystem")
        fake_yr_datasystem.__path__ = []
        fake_hetero = types.ModuleType("yr.datasystem.hetero_client")
        fake_kv = types.ModuleType("yr.datasystem.kv_client")
        fake_object = types.ModuleType("yr.datasystem.object_client")

        fake_hetero.Blob = _FakeBlob
        fake_hetero.DeviceBlobList = _FakeDeviceBlobList
        fake_hetero.HeteroClient = _FakeHeteroClient
        fake_kv.SetParam = _FakeSetParam
        fake_object.WriteMode = _FakeWriteMode

        sys.modules.update(
            {
                "vllm": fake_vllm,
                "vllm.config": fake_config,
                "vllm.logger": fake_logger,
                "vllm.utils": fake_utils,
                "vllm.utils.network_utils": fake_network_utils,
                "vllm_ascend": fake_vllm_ascend,
                "vllm_ascend.distributed": fake_distributed,
                "vllm_ascend.distributed.kv_transfer": fake_kv_transfer,
                "vllm_ascend.distributed.kv_transfer.kv_pool": fake_kv_pool,
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store": fake_ascend_store,
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend": fake_backend_pkg,
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend": fake_backend_module,
                "yr": fake_yr,
                "yr.datasystem": fake_yr_datasystem,
                "yr.datasystem.hetero_client": fake_hetero,
                "yr.datasystem.kv_client": fake_kv,
                "yr.datasystem.object_client": fake_object,
            }
        )
        os.environ["DS_WORKER_ADDR"] = "127.0.0.1:31501"
        _FakeHeteroClient.instances.clear()

    def tearDown(self):
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _purge_modules("vllm", "vllm_ascend", "yr", "vllm_gr.distributed")
        _restore_modules(self.saved_modules)

    def _build_backend(self):
        module = importlib.import_module(
            "vllm_gr.distributed.kv_transfer.kv_pool.backend.yuanrong_backend"
        )
        backend = module.YuanrongBackend(sys.modules["vllm.config"].ParallelConfig(rank=0))
        backend._ds_device_id = 0
        return backend

    def test_put_passes_through_single_batch(self):
        backend = self._build_backend()
        client = _FakeHeteroClient.instances[-1]

        backend.put(
            keys=["k1", "k2", "k3", "k4", "k5"],
            addrs=[[11, 12], [21, 22], [31, 32], [41, 42], [51, 52]],
            sizes=[[100, 200], [100, 200], [100, 200], [100, 200], [100, 200]],
        )

        self.assertEqual(
            [call[0] for call in client.put_calls],
            [["k1", "k2", "k3", "k4", "k5"]],
        )
        self.assertEqual(len(self.fake_logger.info_calls), 1)
        self.assertEqual(
            self.fake_logger.info_calls[0],
            (
                "Calling Yuanrong mset_d2h with key_count=%d, blob_count=%d, "
                "total_size=%s, per_key_mb=%s",
                5,
                10,
                "0.00 MB",
                "['0.00 MB', '0.00 MB', '0.00 MB', '0.00 MB', '0.00 MB']",
            ),
        )

    def test_get_passes_through_single_batch(self):
        backend = self._build_backend()
        client = _FakeHeteroClient.instances[-1]

        backend.get(
            keys=["k1", "k2", "k3"],
            addrs=[[11], [21], [31]],
            sizes=[[300], [300], [300]],
        )

        self.assertEqual(
            [call[0] for call in client.get_calls],
            [["k1", "k2", "k3"]],
        )
        self.assertEqual(len(self.fake_logger.info_calls), 1)
        self.assertEqual(
            self.fake_logger.info_calls[0],
            (
                "Calling Yuanrong mget_h2d with key_count=%d, blob_count=%d, "
                "total_size=%s, per_key_mb=%s",
                3,
                3,
                "0.00 MB",
                "['0.00 MB', '0.00 MB', '0.00 MB']",
            ),
        )

    def test_exists_uses_full_key_list_once(self):
        backend = self._build_backend()
        client = _FakeHeteroClient.instances[-1]

        result = backend.exists(["k1", "k2", "k3"])

        self.assertEqual(result, [1, 1, 1])
        self.assertEqual(client.exist_calls, [["k1", "k2", "k3"]])


if __name__ == "__main__":
    unittest.main()
