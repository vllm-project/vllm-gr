# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""NPU operator namespace.

Ascend fused operators are supplied by xllm-ops built for the CANN version
in the vendor image. The xllm parity and NPU acceptance gates validate their
registration contract.
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
