# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility exports for the NPU stage-width implementation."""

from vllm_gr.v1.beam.dynamic_beam import DynamicBeamState as NPUDynamicBeamState
from vllm_gr.v1.beam.dynamic_beam import get_stage_pools as get_npu_stage_pools
from vllm_gr.v1.beam.dynamic_beam import (
    initialize_packed_stage_pools as initialize_npu_packed_stage_pools,
)
from vllm_gr.v1.beam.dynamic_beam import prune_stage_pools as prune_npu_stage_pools
from vllm_gr.v1.beam.dynamic_beam import reorder_stage_kv as reorder_npu_stage_kv

__all__ = [
    "NPUDynamicBeamState",
    "get_npu_stage_pools",
    "initialize_npu_packed_stage_pools",
    "prune_npu_stage_pools",
    "reorder_npu_stage_kv",
]
