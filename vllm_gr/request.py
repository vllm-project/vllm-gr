# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GRRequest — extends vLLM's Request to carry GR features through the
EngineCore → Scheduler → ModelRunner pipeline.
"""

from typing import Any, Optional
from collections.abc import Callable

from vllm.v1.request import Request

from vllm_gr.engine import GRFeaturesPayload, GREngineCoreRequest


class GRRequest(Request):
    """Request subclass that carries gr_features alongside standard fields.

    When monkey-patched via patch.py, vLLM's EngineCore creates GRRequest
    instances, so gr_features flows through Scheduler → NewRequestData →
    ModelRunner automatically.
    """

    def __init__(
        self,
        gr_features: Optional[GRFeaturesPayload] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gr_features: Optional[GRFeaturesPayload] = gr_features

    @property
    def gr_payload(self):
        """Alias for gr_features — adapter-level generic name."""
        return self.gr_features

    @classmethod
    def from_engine_core_request(
        cls,
        request: GREngineCoreRequest,
        block_hasher: Optional[Callable] = None,
    ) -> "GRRequest":
        return cls(
            gr_features=getattr(request, "gr_features", None),
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            client_index=getattr(request, "client_index", 0),
            arrival_time=request.arrival_time,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            mm_features=request.mm_features,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
        )


def is_gr_request(request: Any) -> bool:
    """Check if the request is a GRRequest with GR features."""
    return isinstance(request, GRRequest) and request.gr_features is not None
