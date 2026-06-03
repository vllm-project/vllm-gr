# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Yuanrong KV pool backend implementation for Ascend/NPU runtimes."""

import hashlib
import os
from typing import Any

import regex as re
import torch
from vllm.config import ParallelConfig
from vllm.logger import logger
from vllm.utils.network_utils import split_host_port
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import (
    Backend,
)


class YuanrongBackend(Backend):
    """Backend implementation mirrored from the upstream vllm-ascend patch."""

    _DS_KEY_MAX_LEN = 255
    _DS_KEY_ALLOWED_PATTERN = re.compile(r"^[a-zA-Z0-9\-_!@#%\^\*\(\)\+\=\:;]+$")
    _DS_KEY_INVALID_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\-_!@#%\^\*\(\)\+\=\:;]")
    _DS_KEY_HASH_SUFFIX_LEN = 16
    _TRANSFER_PREVIEW_LIMIT = 8

    def __init__(self, parallel_config: ParallelConfig):
        try:
            from yr.datasystem.hetero_client import (  # type: ignore
                Blob,
                DeviceBlobList,
                HeteroClient,
            )
            from yr.datasystem.kv_client import SetParam  # type: ignore
            from yr.datasystem.object_client import WriteMode  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Please install openyuanrong-datasystem to use the yuanrong datasystem backend."
            ) from exc

        worker_addr = os.getenv("DS_WORKER_ADDR", "")
        host, port = split_host_port(worker_addr)
        enable_exclusive_connection = bool(int(os.getenv("DS_ENABLE_EXCLUSIVE_CONNECTION", "0")))
        enable_remote_h2d = bool(int(os.getenv("DS_ENABLE_REMOTE_H2D", "0")))
        self._hetero_client = HeteroClient(
            host,
            int(port),
            enable_exclusive_connection=enable_exclusive_connection,
            enable_remote_h2d=enable_remote_h2d,
        )
        self._hetero_client.init()
        self.rank = parallel_config.rank
        self._ds_device_id: int | None = None
        self._ds_blob_cls = Blob
        self._ds_blob_list_cls = DeviceBlobList
        self._ds_set_param = SetParam()
        self._ds_set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT

    def _normalize_ds_keys(self, keys: list[str]) -> list[str]:
        normalized: list[str] = []
        for key in keys:
            if len(key) <= self._DS_KEY_MAX_LEN and self._DS_KEY_ALLOWED_PATTERN.match(key):
                normalized.append(key)
                continue

            sanitized = self._DS_KEY_INVALID_CHAR_PATTERN.sub("_", key)
            hash_digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            suffix = f"__{hash_digest[: self._DS_KEY_HASH_SUFFIX_LEN]}"
            max_prefix_len = self._DS_KEY_MAX_LEN - len(suffix)
            normalized.append(sanitized[:max_prefix_len] + suffix)
        return normalized

    def set_device(self):
        device = torch.device(f"npu:{self.rank}")
        torch.npu.set_device(device)
        self._ds_device_id = int(torch.npu.current_device())

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        # Yuanrong uses explicit device pointers in Blob lists, so there is no
        # backend-side buffer registration step to perform here.
        del ptrs, lengths

    def _ds_make_blob_list(self, addrs: list[int], sizes: list[int]) -> Any:
        if len(addrs) != len(sizes):
            raise ValueError("Address list and size list length mismatch.")
        if self._ds_device_id is None:
            logger.error(
                "Device id is not set. Call set_device() before using the yuanrong backend."
            )
            raise RuntimeError("Yuanrong backend device id is not initialized.")
        blobs = [self._ds_blob_cls(addr, size) for addr, size in zip(addrs, sizes)]
        return self._ds_blob_list_cls(self._ds_device_id, blobs)

    def _summarize_transfer(self, sizes: list[list[int]]) -> tuple[int, int, list[int]]:
        per_key_bytes = [sum(size_list) for size_list in sizes]
        blob_count = sum(len(size_list) for size_list in sizes)
        total_bytes = sum(per_key_bytes)
        return blob_count, total_bytes, per_key_bytes

    def _format_mb(self, size_bytes: int) -> str:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

    def _preview_per_key_mb(self, per_key_bytes: list[int]) -> str:
        per_key_mb = [self._format_mb(size_bytes) for size_bytes in per_key_bytes]
        if len(per_key_mb) <= self._TRANSFER_PREVIEW_LIMIT:
            return str(per_key_mb)
        preview = per_key_mb[: self._TRANSFER_PREVIEW_LIMIT]
        remaining = len(per_key_bytes) - self._TRANSFER_PREVIEW_LIMIT
        return f"{preview} ... ({remaining} more)"

    def exists(self, keys: list[str]) -> list[int]:
        if len(keys) == 0:
            logger.warning("No keys provided.")
            return []
        normalized_keys = self._normalize_ds_keys(keys)
        try:
            exists = self._hetero_client.exist(normalized_keys)  # type: ignore[union-attr]
            return [1 if value else 0 for value in exists]
        except Exception as exc:
            logger.error("Failed to check keys %s: %s", normalized_keys, exc)
            return [0] * len(normalized_keys)

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        normalized_keys = self._normalize_ds_keys(keys)
        try:
            blob_count, total_bytes, per_key_bytes = self._summarize_transfer(sizes)
            logger.info(
                "Calling Yuanrong mget_h2d with key_count=%d, blob_count=%d, "
                "total_size=%s, per_key_mb=%s",
                len(normalized_keys),
                blob_count,
                self._format_mb(total_bytes),
                self._preview_per_key_mb(per_key_bytes),
            )
            blob_lists = [
                self._ds_make_blob_list(addr_list, size_list)
                for addr_list, size_list in zip(addrs, sizes)
            ]
            failed_keys = self._hetero_client.mget_h2d(  # type: ignore[union-attr]
                normalized_keys,
                blob_lists,
                0,
            )
            for key in failed_keys:
                logger.error("Failed to get key %s", key)
        except Exception as exc:
            logger.error("Failed to get keys %s: %s", normalized_keys, exc)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        normalized_keys = self._normalize_ds_keys(keys)
        try:
            blob_count, total_bytes, per_key_bytes = self._summarize_transfer(sizes)
            logger.info(
                "Calling Yuanrong mset_d2h with key_count=%d, blob_count=%d, "
                "total_size=%s, per_key_mb=%s",
                len(normalized_keys),
                blob_count,
                self._format_mb(total_bytes),
                self._preview_per_key_mb(per_key_bytes),
            )
            blob_lists = [
                self._ds_make_blob_list(addr_list, size_list)
                for addr_list, size_list in zip(addrs, sizes)
            ]
            self._hetero_client.mset_d2h(  # type: ignore[union-attr]
                normalized_keys,
                blob_lists,
                self._ds_set_param,
            )
        except Exception as exc:
            logger.error("Failed to put keys %s: %s", normalized_keys, exc)
