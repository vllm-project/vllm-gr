# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vllm-gr operator packages.

After the HSTU / DLRM-v3 / FuXi runtime was removed, the paged-KV attention
ops and jagged-tensor helpers that served it are gone. The only remaining
operator package is :mod:`vllm_gr.ops.npu`, which hosts the NPU (Ascend)
custom-op patches applied by ``vllm_gr.plugin`` / ``vllm_gr.patch``.
"""

from vllm_gr.ops.npu.split_qkv_rmsnorm_rope import (
    apply_split_qkv_rmsnorm_rope_patch,
)

__all__ = [
    "apply_split_qkv_rmsnorm_rope_patch",
]
