# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reserved NPU operator namespace.

Ascend fused operators, including split-QKV-RMSNorm-RoPE, are registered by
vLLM-Ascend 0.22.1rc1.
"""

from vllm_gr.ops.npu.beam_ops import (
    beam_search_group,
    beam_search_rec_final_select,
    cache_unshared_kv,
)
from vllm_gr.ops.npu.beam_ops import is_available as beam_ops_npu_available
from vllm_gr.ops.npu.beam_ops import (
    select_unshared_kv,
    x_attention,
)
from vllm_gr.ops.npu.rec_constrained_topk import (
    is_available as rec_constrained_topk_npu_available,
)
from vllm_gr.ops.npu.rec_constrained_topk import (
    rec_constrained_topk_npu,
    resolve_rec_constrained_topk_npu,
)

__all__ = [
    "beam_ops_npu_available",
    "beam_search_group",
    "beam_search_rec_final_select",
    "cache_unshared_kv",
    "rec_constrained_topk_npu",
    "rec_constrained_topk_npu_available",
    "resolve_rec_constrained_topk_npu",
    "select_unshared_kv",
    "x_attention",
]
