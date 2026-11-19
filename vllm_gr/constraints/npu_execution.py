# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Ascend execution for the legacy compiled constraint tables."""

from __future__ import annotations

import torch

from vllm_gr.constraints.execution import _ConstrainedTopKExecutionBase


class NPUConstrainedTopKExecution(_ConstrainedTopKExecutionBase):
    """NPU constrained top-k backed by the fused ``custom_ops_lib`` operator.

    Calls ``rec_constrained_topk_npu`` directly and fails explicitly when the
    custom op is unavailable; it never silently falls back to Torch.
    """

    implementation_id = "vllm_gr.constraints.execution.NPUConstrainedTopKExecution"

    def is_available(self) -> bool:
        from vllm_gr.ops.npu.rec_constrained_topk import is_available

        return is_available()

    def _default_device(self) -> torch.device:
        try:
            return torch.device("npu")
        except RuntimeError:
            # Torch builds without the Ascend extension reject the NPU device type.
            return super()._default_device()

    def _call_op(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_gr.ops.npu.rec_constrained_topk import rec_constrained_topk_npu

        return rec_constrained_topk_npu(
            logits,
            sequence_group,
            self._tables["first_token_ids"],
            self._tables["prefix1_offsets"],
            self._tables["prefix1_values"],
            self._tables["prefix1_pair_keys"],
            self._tables["prefix2_value_offsets"],
            self._tables["prefix2_values"],
            temperatures,
            current_step,
            top_k,
            self._max_degrees["max_prefix1_degree"],
            self._max_degrees["max_prefix2_degree"],
        )
