# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GRNewRequestData — extends NewRequestData to carry GR features from
the Scheduler to the ModelRunner via SchedulerOutput.
"""

from dataclasses import dataclass

from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request


@dataclass
class GRNewRequestData(NewRequestData):
    """NewRequestData with an additional gr_features field."""

    gr_features: bytes | None = None

    @property
    def gr_payload(self):
        """Alias for gr_features — adapter-level generic name."""
        return self.gr_features

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "GRNewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            prefill_token_ids=prefill_token_ids,
            gr_features=getattr(request, "gr_features", None),
        )
