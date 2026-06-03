# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility helpers for backend detection and safe operator dispatch."""

from vllm_gr.utils.hstu_runtime import (
    HSTURuntimeDevices,
    HSTUSparseTransferPlan,
    get_hstu_autocast_context,
    move_hstu_sparse_inputs_to_device,
    move_hstu_sparse_outputs_to_device,
    resolve_hstu_dense_device,
    resolve_hstu_embedding_device,
    resolve_hstu_runtime_devices,
    resolve_hstu_sparse_transfer_plan,
)
from vllm_gr.utils.npu_adapter import (
    DeviceAdapter,
    DeviceType,
    check_fbgemm_available,
    check_recsdk_available,
    check_torchrec_available,
    get_default_device,
    get_device_adapter,
    get_device_backend,
    get_device_type,
    is_cuda_available,
    is_cuda_device,
    is_npu_available,
    is_npu_device,
    release_accelerator_memory,
    reset_device_adapter,
)
from vllm_gr.utils.reliable_ops import ReliableOps

__all__ = [
    "DeviceAdapter",
    "DeviceType",
    "HSTURuntimeDevices",
    "HSTUSparseTransferPlan",
    "ReliableOps",
    "check_fbgemm_available",
    "check_recsdk_available",
    "check_torchrec_available",
    "get_device_adapter",
    "get_default_device",
    "get_device_backend",
    "get_device_type",
    "get_hstu_autocast_context",
    "is_cuda_available",
    "is_cuda_device",
    "is_npu_available",
    "is_npu_device",
    "move_hstu_sparse_inputs_to_device",
    "move_hstu_sparse_outputs_to_device",
    "release_accelerator_memory",
    "reset_device_adapter",
    "resolve_hstu_dense_device",
    "resolve_hstu_embedding_device",
    "resolve_hstu_runtime_devices",
    "resolve_hstu_sparse_transfer_plan",
]
