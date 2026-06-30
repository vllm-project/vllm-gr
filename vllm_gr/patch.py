# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Monkey-patch mechanism for vllm-GR.

Replaces vLLM's core types with GR-extended variants in all loaded modules:
  - EngineCoreRequest -> GREngineCoreRequest
  - Request -> GRRequest
  - NewRequestData -> GRNewRequestData

Activation:
  Primary - vLLM plugin system.  ``vllm_gr.plugin:register`` is registered
  under ``[project.entry-points."vllm.general_plugins"]`` in pyproject.toml.
  ``load_general_plugins()`` calls it in every process (main, EngineCore
  subprocess, Worker subprocess) *before* any MsgpackDecoder is created.

  Explicit main-process init - ``vllm_gr.plugin.initialize_runtime()`` is
  called by ``GREngineArgs.create_engine_config()`` before engine creation.

  Defense-in-depth - ``GRWorker.init_device()`` calls ``re_apply_patches()``
  to cover modules loaded after the initial plugin execution.

Safety features:
  - Source modules (vllm.v1.request, etc.) are patched first so that any
    future ``from X import Y`` in late-loaded modules picks up the GR variant.
  - verify_patches() checks the critical call-sites and logs actionable
    warnings when a patch is missing.
"""

import importlib
import logging
import math
import sys
import threading
import types
from typing import Any

from vllm_gr.ops.npu.split_qkv_rmsnorm_rope import apply_split_qkv_rmsnorm_rope_patch

logger = logging.getLogger("vllm_gr.patch")

_OriginalEngineCoreRequest = None
_OriginalRequest = None
_OriginalNewRequestData = None

_PATCHES_APPLIED = False
_PATCH_COUNT = 0
_YUANRONG_PATCHES_APPLIED = False

try:
    from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
except ImportError:
    logger.debug("vllm_gr: vLLM EngineCoreRequest not found")

try:
    from vllm.v1.request import Request as _OriginalRequest
except ImportError:
    logger.debug("vllm_gr: vLLM Request not found")

try:
    from vllm.v1.core.sched.output import NewRequestData as _OriginalNewRequestData
except ImportError:
    logger.debug("vllm_gr: vLLM NewRequestData not found")

# Source modules whose exported names are re-imported by consumer modules.
# Patching these first ensures late-loaded consumers get the GR variants.
_SOURCE_MODULES = {
    "EngineCoreRequest": "vllm.v1.engine",
    "Request": "vllm.v1.request",
    "NewRequestData": "vllm.v1.core.sched.output",
}

# Consumer modules that call from_engine_core_request / from_request.
_CRITICAL_CONSUMERS = {
    "EngineCoreRequest": ["vllm.v1.engine", "vllm.v1.engine.core"],
    "Request": ["vllm.v1.request", "vllm.v1.engine.core"],
    "NewRequestData": [
        "vllm.v1.core.sched.output",
        "vllm.v1.core.sched.scheduler",
    ],
}


_ASCEND_BACKEND_PACKAGE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend"
_ASCEND_POOL_WORKER_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker"
_ASCEND_CONFIG_DATA_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data"
_ASCEND_MOONCAKE_MODULE = f"{_ASCEND_BACKEND_PACKAGE}.mooncake_backend"
_ASCEND_YUANRONG_MODULE = f"{_ASCEND_BACKEND_PACKAGE}.yuanrong_backend"
_LOCAL_YUANRONG_BACKEND_MODULE = "vllm_gr.distributed.kv_transfer.kv_pool.backend.yuanrong_backend"


def _is_npu_platform() -> bool:
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


def _apply_gr_core_patches(force: bool = False) -> int:
    """Replace vLLM core types with GR-extended versions in loaded modules.

    Patches source modules first (so future imports inherit the replacement),
    then patches all already-loaded consumer modules.

    Args:
        force: Re-scan sys.modules even if patches were already applied.

    Returns:
        Number of module-level replacements performed.
    """
    global _PATCHES_APPLIED, _PATCH_COUNT

    if _PATCHES_APPLIED and not force:
        return _PATCH_COUNT

    if _OriginalEngineCoreRequest is None and _OriginalRequest is None:
        logger.warning("vllm_gr: vLLM core patch targets not found, skipping GR core patches")
        _PATCHES_APPLIED = True
        return 0

    from vllm_gr.core.sched.output import GRNewRequestData
    from vllm_gr.engine import GREngineCoreRequest
    from vllm_gr.request import GRRequest

    replacements = [
        (_OriginalEngineCoreRequest, GREngineCoreRequest, "EngineCoreRequest"),
        (_OriginalRequest, GRRequest, "Request"),
        (_OriginalNewRequestData, GRNewRequestData, "NewRequestData"),
    ]

    patched_count = 0
    patched_modules = []

    # Phase 1: patch source modules so future `from X import Y` picks up GR.
    for original, replacement, attr_name in replacements:
        if original is None:
            continue
        src_mod_name = _SOURCE_MODULES.get(attr_name)
        src_mod = sys.modules.get(src_mod_name) if src_mod_name else None
        if src_mod is not None and getattr(src_mod, attr_name, None) is original:
            setattr(src_mod, attr_name, replacement)
            patched_count += 1
            patched_modules.append(f"{src_mod_name}.{attr_name}")

    # Phase 2: patch all loaded consumer modules.
    for module_name, module in list(sys.modules.items()):
        if module is None or "vllm" not in module_name:
            continue
        for original, replacement, attr_name in replacements:
            if original is None:
                continue
            if getattr(module, attr_name, None) is original:
                setattr(module, attr_name, replacement)
                patched_count += 1
                patched_modules.append(f"{module_name}.{attr_name}")

    _PATCHES_APPLIED = True
    _PATCH_COUNT += patched_count

    if patched_count > 0:
        logger.info(
            "vllm_gr: Applied %d monkey patches: %s",
            patched_count,
            patched_modules,
        )

    return patched_count


def _build_yuanrong_backend_class() -> type[Any]:
    module = importlib.import_module(_LOCAL_YUANRONG_BACKEND_MODULE)
    backend_cls = getattr(module, "YuanrongBackend")
    backend_cls.__module__ = _ASCEND_YUANRONG_MODULE
    return backend_cls


def _install_mooncake_backend_shim_if_needed() -> int:
    existing = sys.modules.get(_ASCEND_MOONCAKE_MODULE)
    if existing is not None:
        return 0

    try:
        importlib.import_module(_ASCEND_MOONCAKE_MODULE)
        return 0
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) not in {
            "mooncake",
            "mooncake.store",
            _ASCEND_MOONCAKE_MODULE,
        }:
            raise
    except ImportError as exc:
        if "mooncake" not in str(exc).lower():
            raise

    backend_module = importlib.import_module(f"{_ASCEND_BACKEND_PACKAGE}.backend")
    backend_base = getattr(backend_module, "Backend")

    class MooncakeBackend(backend_base):
        def __init__(self, *_args, **_kwargs):
            raise ModuleNotFoundError(
                "Mooncake backend dependencies are not installed. "
                "Install the mooncake runtime package to use backend='mooncake'."
            )

    MooncakeBackend.__name__ = "MooncakeBackend"
    MooncakeBackend.__qualname__ = "MooncakeBackend"
    MooncakeBackend.__module__ = _ASCEND_MOONCAKE_MODULE

    module = types.ModuleType(_ASCEND_MOONCAKE_MODULE)
    module.__package__ = _ASCEND_BACKEND_PACKAGE
    module.MooncakeBackend = MooncakeBackend
    module.__all__ = ["MooncakeBackend"]
    sys.modules[_ASCEND_MOONCAKE_MODULE] = module

    backend_pkg = sys.modules.get(_ASCEND_BACKEND_PACKAGE)
    if backend_pkg is not None:
        setattr(backend_pkg, "mooncake_backend", module)

    return 1


def _install_yuanrong_module(backend_cls: type[Any]) -> int:
    existing = sys.modules.get(_ASCEND_YUANRONG_MODULE)
    if existing is not None and getattr(existing, "YuanrongBackend", None) is backend_cls:
        return 0

    module = existing if existing is not None else types.ModuleType(_ASCEND_YUANRONG_MODULE)
    module.__package__ = _ASCEND_BACKEND_PACKAGE
    module.YuanrongBackend = backend_cls
    module.__all__ = ["YuanrongBackend"]
    sys.modules[_ASCEND_YUANRONG_MODULE] = module

    backend_pkg = sys.modules.get(_ASCEND_BACKEND_PACKAGE)
    if backend_pkg is not None:
        setattr(backend_pkg, "yuanrong_backend", module)

    return 1


def _patch_yuanrong_backend_map(backend_cls: type[Any]) -> int:
    pool_worker = sys.modules.get(_ASCEND_POOL_WORKER_MODULE)
    if pool_worker is None:
        raise RuntimeError(
            "_patch_yuanrong_backend_map requires the pool_worker module to "
            "be loaded before patching"
        )
    backend_map = getattr(pool_worker, "backend_map", None)
    if not isinstance(backend_map, dict):
        logger.warning(
            "Skipping Yuanrong backend monkey-patch because %s.backend_map is "
            "missing or not a dict",
            _ASCEND_POOL_WORKER_MODULE,
        )
        return 0

    changes = 0
    if backend_map.get("yuanrong") is not backend_cls:
        backend_map["yuanrong"] = backend_cls
        changes += 1
    if getattr(pool_worker, "YuanrongBackend", None) is not backend_cls:
        setattr(pool_worker, "YuanrongBackend", backend_cls)
        changes += 1
    return changes


def _apply_yuanrong_npu_patch(force: bool = False) -> int:
    global _YUANRONG_PATCHES_APPLIED

    if _YUANRONG_PATCHES_APPLIED and not force:
        return 0

    if not _is_npu_platform():
        return 0

    shim_count = 0
    try:
        shim_count = _install_mooncake_backend_shim_if_needed()
        importlib.import_module(_ASCEND_POOL_WORKER_MODULE)
    except (ModuleNotFoundError, ImportError) as exc:
        logger.debug(
            "Skipping Yuanrong backend monkey-patch because %s is unavailable: %s",
            _ASCEND_POOL_WORKER_MODULE,
            exc,
        )
        return 0

    try:
        backend_cls = _build_yuanrong_backend_class()
    except (ModuleNotFoundError, ImportError, AttributeError) as exc:
        logger.debug(
            "Skipping Yuanrong backend monkey-patch because dependencies are unavailable: %s",
            exc,
        )
        return 0
    except Exception:
        logger.exception("Failed to build Yuanrong backend monkey-patch")
        raise

    patch_count = shim_count
    patch_count += _install_yuanrong_module(backend_cls)
    patch_count += _patch_yuanrong_backend_map(backend_cls)
    _YUANRONG_PATCHES_APPLIED = True

    if patch_count > 0:
        logger.info(
            "vllm_gr: Applied %d Yuanrong NPU monkey patch(es)",
            patch_count,
        )
    return patch_count


_KV_MERGE_PATCHES_APPLIED = False
_GR_BEAM_PATCHES_APPLIED = False


def _apply_gr_beam_patches(force: bool = False) -> int:
    """Apply beam-search related patches from vllm-gr.

    These patches are independent from the GR core type replacement patches
    and can be safely applied alongside them.
    """
    global _GR_BEAM_PATCHES_APPLIED

    if _GR_BEAM_PATCHES_APPLIED and not force:
        return 0

    patch_count = 0

    try:
        from vllm_gr.logprobs_patch import patch_flat_logprobs

        patch_flat_logprobs()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_flat_logprobs")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_flat_logprobs: %s", exc)

    try:
        from vllm_gr.entrypoints.openai.beam_search_patch import (
            patch_beam_search,
            patch_sampling,
        )

        patch_beam_search()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_beam_search")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_beam_search: %s", exc)

    try:
        from vllm_gr.entrypoints.openai.beam_search_patch import patch_sampling

        patch_sampling()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_sampling")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_sampling: %s", exc)

    try:
        from vllm_gr.engine.arg_utils_gr import patch_add_cli_args

        patch_add_cli_args()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_add_cli_args")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_add_cli_args: %s", exc)

    try:
        from vllm_gr.entrypoints.openai.serving_models import (
            patch_OpenAIServingModels_init,
        )

        patch_OpenAIServingModels_init()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_OpenAIServingModels_init")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_OpenAIServingModels_init: %s", exc)

    try:
        from vllm_gr.entrypoints.openai.beam_search_patch import patch_batch_and_fork

        patch_batch_and_fork()
        patch_count += 1
        logger.debug("vllm_gr: Applied patch_batch_and_fork")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping patch_batch_and_fork: %s", exc)

    # Optional: LMCache adapter patch
    try:
        import importlib.util
        from pathlib import Path

        adapter_path = (
            Path(__file__).resolve().parent
            / "3rd_party"
            / "lmcache"
            / "integration"
            / "vllm"
            / "vllm_v1_adapter.py"
        )
        if adapter_path.exists():
            import os
            import sys

            should_patch = False
            if os.getenv("VLLM_GR_LMCACHE_PATCH", "").lower() in (
                "1",
                "true",
                "yes",
                "y",
            ):
                should_patch = True
            elif any("LMCacheConnectorV1" in arg for arg in sys.argv):
                should_patch = True

            if should_patch:
                spec = importlib.util.spec_from_file_location(
                    "vllm_gr_3rd_party_lmcache_integration_vllm_v1_adapter",
                    adapter_path,
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.patch_installed_lmcache_adapter()
                patch_count += 1
                logger.debug("vllm_gr: Applied LMCache adapter patch")
    except Exception as exc:
        logger.debug("vllm_gr: Skipping LMCache adapter patch: %s", exc)

    _GR_BEAM_PATCHES_APPLIED = True

    if patch_count > 0:
        logger.info(
            "vllm_gr: Applied %d beam-search monkey patch(es)",
            patch_count,
        )
    return patch_count


def patch_lmcache_adapter():
    """Apply LMCache adapter patch.

    Public helper used by engine_core_patch.py in child processes where
    only the LMCache monkey patch must be installed, without re-running
    the full patch sequence.
    """
    import importlib.util
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parent
        / "3rd_party"
        / "lmcache"
        / "integration"
        / "vllm"
        / "vllm_v1_adapter.py"
    )
    if not module_path.exists():
        return
    spec = importlib.util.spec_from_file_location(
        "vllm_gr_3rd_party_lmcache_integration_vllm_v1_adapter",
        module_path,
    )
    if spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_installed_lmcache_adapter()


def _apply_kv_merge_patch(force: bool = False) -> int:
    """Patch vllm-ascend to support K+V merged KV cache tensors.

    This patch adds support for HSTU-style KV cache where K and V are concatenated
    into a single tensor per layer (kv_cache_per_layer=1), instead of the default
    separate K and V tensors (kv_cache_per_layer=2).

    The patch modifies:
    1. ChunkedTokenDatabase: adds kv_cache_per_layer attribute and set method
    2. ChunkedTokenDatabase.prepare_value_layer: handles merged tensor case
    3. KVPoolWorker.register_kv_caches: auto-detects merged vs separate tensors

    This is needed for backward compatibility with older vllm-ascend versions
    that don't have the "support kv hebing" commit.
    """
    global _KV_MERGE_PATCHES_APPLIED

    if _KV_MERGE_PATCHES_APPLIED and not force:
        return 0

    if not _is_npu_platform():
        return 0

    try:
        config_data_mod = importlib.import_module(_ASCEND_CONFIG_DATA_MODULE)
        pool_worker_mod = importlib.import_module(_ASCEND_POOL_WORKER_MODULE)
    except (ModuleNotFoundError, ImportError) as exc:
        logger.debug(
            "Skipping KV merge monkey-patch because ascend modules are unavailable: %s",
            exc,
        )
        return 0

    patch_count = 0
    ChunkedTokenDatabase = getattr(config_data_mod, "ChunkedTokenDatabase", None)
    KVPoolWorker = getattr(pool_worker_mod, "KVPoolWorker", None)

    if ChunkedTokenDatabase is None or KVPoolWorker is None:
        logger.debug(
            "Skipping KV merge monkey-patch: ChunkedTokenDatabase or KVPoolWorker not found"
        )
        return 0

    if not hasattr(ChunkedTokenDatabase, "kv_cache_per_layer"):
        original_init = ChunkedTokenDatabase.__init__

        def patched_init(self, metadata, block_size, use_mla, partitions):
            original_init(self, metadata, block_size, use_mla, partitions)
            self.kv_cache_per_layer = 2

        ChunkedTokenDatabase.__init__ = patched_init
        patch_count += 1
        logger.debug("vllm_gr: Patched ChunkedTokenDatabase.__init__ to add kv_cache_per_layer")

    if not hasattr(ChunkedTokenDatabase, "set_kv_cache_per_layer"):

        def set_kv_cache_per_layer(self, kv_cache_per_layer: int):
            self.kv_cache_per_layer = kv_cache_per_layer

        ChunkedTokenDatabase.set_kv_cache_per_layer = set_kv_cache_per_layer
        patch_count += 1
        logger.debug("vllm_gr: Added ChunkedTokenDatabase.set_kv_cache_per_layer method")

    original_prepare_value_layer = getattr(ChunkedTokenDatabase, "prepare_value_layer", None)
    if original_prepare_value_layer is not None:

        def patched_prepare_value_layer(
            self, start: int, end: int, block_ids: list[int], layer_id: int
        ):
            block_id = block_ids[start // self.block_size]
            kv_cache_per_layer = getattr(self, "kv_cache_per_layer", 2)
            if kv_cache_per_layer == 1:
                addr = self.kv_caches_base_addr[layer_id] + block_id * self.block_len[0]
                length = int(self.block_len[0] / self.block_size * (end - start))
                return [addr], [length]
            return original_prepare_value_layer(self, start, end, block_ids, layer_id)

        ChunkedTokenDatabase.prepare_value_layer = patched_prepare_value_layer
        patch_count += 1
        logger.debug(
            "vllm_gr: Patched ChunkedTokenDatabase.prepare_value_layer for kv_cache_per_layer=1"
        )

    original_prepare_value = getattr(ChunkedTokenDatabase, "prepare_value", None)
    if original_prepare_value is not None:

        def patched_prepare_value(self, start: int, end: int, block_ids: list[int]):
            addr_list = []
            size_list = []
            block_id = block_ids[start // self.block_size]
            for index, base_addr in enumerate(self.kv_caches_base_addr):
                block_len = (
                    self.block_len[index % len(self.block_len)]
                    if self.use_mla
                    else self.block_len[0]
                )
                addr = base_addr + block_id * block_len
                length = int(block_len / self.block_size * (end - start))
                addr_list.append(addr)
                size_list.append(length)
            return addr_list, size_list, block_id

        ChunkedTokenDatabase.prepare_value = patched_prepare_value
        patch_count += 1
        logger.debug(
            "vllm_gr: Patched ChunkedTokenDatabase.prepare_value for block_len indexing fix"
        )

    original_register_kv_caches = getattr(KVPoolWorker, "register_kv_caches", None)
    if original_register_kv_caches is not None:

        def patched_register_kv_caches(self, kv_caches: dict):
            _, first_kv_cache_value = next(iter(kv_caches.items()))
            is_merged = isinstance(first_kv_cache_value, object) and not isinstance(
                first_kv_cache_value, (list, tuple)
            )

            if is_merged:
                self.kv_cache_per_layer = 1
                self.token_database.kv_cache_per_layer = 1

                self.kv_caches = kv_caches
                self.kv_caches_base_addr = []
                ptrs = []
                lengths = []

                first_kv_cache = first_kv_cache_value
                self.num_blocks = first_kv_cache.shape[0]
                kv_elem_size = first_kv_cache.element_size()
                block_rank = 3
                block_shape = first_kv_cache.shape[-block_rank:]
                self.block_len = [kv_elem_size * math.prod(block_shape)]

                logger.info(
                    "Registering KV_Caches (merged). use_mla: %s, kv_cache_per_layer: %d, shape %s",
                    self.use_mla,
                    self.kv_cache_per_layer,
                    first_kv_cache.shape,
                )

                for layer_name, cache_or_caches in kv_caches.items():
                    base_addr = cache_or_caches.data_ptr()
                    self.kv_caches_base_addr.append(base_addr)
                    region_len = self.num_blocks * self.block_len[0]
                    ptrs.append(base_addr)
                    lengths.append(region_len)

                logger.info(
                    "kv_caches_base_addr count: %d, num_layers: %d, kv_cache_per_layer: %d",
                    len(self.kv_caches_base_addr),
                    self.num_layers,
                    self.kv_cache_per_layer,
                )

                self.m_store.register_buffer(ptrs, lengths)
                self.token_database.set_kv_caches_base_addr(self.kv_caches_base_addr)
                self.token_database.set_block_len(self.block_len)
                if hasattr(self.token_database, "set_kv_cache_per_layer"):
                    self.token_database.set_kv_cache_per_layer(self.kv_cache_per_layer)

                if self.use_layerwise:
                    self.get_event = threading.Event()
                    if self.kv_role in ["kv_producer", "kv_both"]:
                        ready_event_sending = threading.Event()
                        self.kv_send_thread = pool_worker_mod.KVCacheStoreLayerSendingThread(
                            self.m_store,
                            self.token_database,
                            self.block_size,
                            self.tp_rank,
                            self.dcp_size,
                            self.put_step,
                            ready_event_sending,
                            self.num_layers,
                        )
                        self.kv_send_thread.start()
                    ready_event = threading.Event()
                    self.kv_recv_thread = pool_worker_mod.KVCacheStoreLayerRecvingThread(
                        self.m_store,
                        self.token_database,
                        self.block_size,
                        self.tp_rank,
                        self.dcp_size,
                        ready_event,
                        self.get_event,
                    )
                    self.kv_recv_thread.start()
                    ready_event.wait()
                else:
                    if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put:
                        ready_event_sending = threading.Event()
                        self.kv_send_thread = pool_worker_mod.KVCacheStoreSendingThread(
                            self.m_store,
                            self.token_database,
                            self.block_size,
                            self.tp_rank,
                            self.dcp_size,
                            self.put_step,
                            self.kv_role,
                            ready_event_sending,
                        )
                        self.kv_send_thread.start()
                        ready_event_sending.wait()
                    if self.load_async:
                        ready_event = threading.Event()
                        self.kv_recv_thread = pool_worker_mod.KVCacheStoreRecvingThread(
                            self.m_store,
                            self.token_database,
                            self.block_size,
                            self.tp_rank,
                            self.dcp_size,
                            ready_event,
                        )
                        self.kv_recv_thread.start()
                        ready_event.wait()
            else:
                self.kv_cache_per_layer = 2
                if hasattr(self.token_database, "set_kv_cache_per_layer"):
                    self.token_database.set_kv_cache_per_layer(2)
                original_register_kv_caches(self, kv_caches)

        KVPoolWorker.register_kv_caches = patched_register_kv_caches
        patch_count += 1
        logger.debug(
            "vllm_gr: Patched KVPoolWorker.register_kv_caches for kv_cache_per_layer detection"
        )

    _KV_MERGE_PATCHES_APPLIED = True

    if patch_count > 0:
        logger.info(
            "vllm_gr: Applied %d KV merge monkey patch(es) for HSTU-style KV cache",
            patch_count,
        )
    return patch_count


def _apply_patches(force: bool = False) -> int:
    """Apply all runtime patches needed by vllm-GR."""
    core_count = _apply_gr_core_patches(force=force)
    yuanrong_count = _apply_yuanrong_npu_patch(force=force)
    qkv_rmsnorm_rope_count = apply_split_qkv_rmsnorm_rope_patch(force=force)
    kv_merge_count = _apply_kv_merge_patch(force=force)
    beam_count = _apply_gr_beam_patches(force=force)
    return core_count + yuanrong_count + qkv_rmsnorm_rope_count + kv_merge_count + beam_count


def verify_patches() -> list[str]:
    """Check that critical call-sites have been patched.

    Returns a list of warning messages for any unpatched sites.
    Safe to call at any time; does not modify anything.
    """
    from vllm_gr.core.sched.output import GRNewRequestData
    from vllm_gr.engine import GREngineCoreRequest
    from vllm_gr.request import GRRequest

    expected = {
        "EngineCoreRequest": GREngineCoreRequest,
        "Request": GRRequest,
        "NewRequestData": GRNewRequestData,
    }

    warnings = []
    for attr_name, expected_cls in expected.items():
        for mod_name in _CRITICAL_CONSUMERS.get(attr_name, []):
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            actual = getattr(mod, attr_name, None)
            if actual is not None and actual is not expected_cls:
                msg = (
                    f"vllm_gr: {mod_name}.{attr_name} is {actual.__name__}, "
                    f"expected {expected_cls.__name__}. "
                    f"gr_features may be lost for requests going through this module."
                )
                warnings.append(msg)
                logger.warning(msg)

    if not warnings:
        logger.debug("vllm_gr: All critical patches verified OK")

    return warnings


def re_apply_patches() -> int:
    """Re-scan sys.modules and patch any newly-loaded vLLM modules.

    Also runs verify_patches() and logs warnings for any gaps.
    """
    count = _apply_patches(force=True)
    if _OriginalEngineCoreRequest is not None or _OriginalRequest is not None:
        verify_patches()
    return count


def get_original_request():
    return _OriginalRequest


def get_original_engine_core_request():
    return _OriginalEngineCoreRequest


def is_patched() -> bool:
    return _PATCHES_APPLIED


def get_patch_count() -> int:
    return _PATCH_COUNT


def apply_patches_if_needed() -> int:
    """Apply patches if not yet applied. Safe to call multiple times."""
    return _apply_patches()
