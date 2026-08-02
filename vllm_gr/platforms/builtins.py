# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration of backend providers shipped with vllm-gr."""


def register_builtin_backends() -> None:
    from .cuda import CUDA_BACKEND
    from .npu import NPU_BACKEND
    from .registry import register_backend

    register_backend(CUDA_BACKEND, source="builtin:cuda")
    register_backend(NPU_BACKEND, source="builtin:npu")
