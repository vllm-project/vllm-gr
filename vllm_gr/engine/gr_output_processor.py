# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GROutputProcessor — extends vLLM's OutputProcessor for recommendation output.

Introduces GRRequestState (extending RequestState) to carry per-request GR
metadata and override make_request_output() to convert raw pooling scores
into candidate-score GRRequestOutput objects via the injected adapter.
"""

from typing import Any

import numpy as np
import torch
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor as VLLMOutputProcessor
from vllm.v1.engine.output_processor import (
    RequestOutputCollector,
    RequestState,
)
from vllm.v1.engine.parallel_sampling import ParentRequest

from vllm_gr.engine.gr_features_codec import decode_gr_features_payload
from vllm_gr.outputs import GRRequestOutput


class GRRequestState(RequestState):
    """RequestState extended with GR metadata stored as a single dict."""

    def __init__(self, *args, **kwargs):
        is_gr_request = kwargs.pop("is_gr_request", False)
        gr_request_meta = dict(kwargs.pop("gr_request_meta", {}) or {})

        super().__init__(*args, **kwargs)

        self.is_gr_request: bool = is_gr_request
        self.gr_request_meta: dict = gr_request_meta or {}

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
        adapter=None,
    ) -> "GRRequestState":
        from vllm.v1.engine.detokenizer import IncrementalDetokenizer
        from vllm.v1.engine.logprobs import LogprobsProcessor

        gr_features = getattr(request, "gr_features", None)
        gr_request_meta = {}
        if gr_features is not None and adapter is not None:
            payload = decode_gr_features_payload(gr_features, adapter.payload_type)
            gr_request_meta = adapter.extract_request_meta(payload)
            gr_request_meta["request_id"] = request.external_req_id

        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer, request=request
            )
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer, request=request
            )
            max_tokens_param = sampling_params.max_tokens
            top_p = sampling_params.top_p
            n = sampling_params.n
            temperature = sampling_params.temperature
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            top_p = None
            n = None
            temperature = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind

        assert request.external_req_id is not None
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            parent_req=parent_req,
            request_index=request_index,
            lora_request=request.lora_request,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            top_p=top_p,
            n=n,
            temperature=temperature,
            arrival_time=request.arrival_time,
            queue=queue,
            log_stats=log_stats,
            stream_interval=stream_interval,
            is_gr_request=gr_features is not None,
            gr_request_meta=gr_request_meta,
        )

    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        routed_experts: np.ndarray | None = None,
    ) -> RequestOutput | PoolingRequestOutput | GRRequestOutput | None:
        """For GR requests, delegate to adapter.process_output."""

        if self.is_gr_request:
            return self._make_gr_output(pooling_output, finish_reason)

        return super().make_request_output(
            new_token_ids,
            pooling_output,
            finish_reason,
            stop_reason,
            kv_transfer_params,
            routed_experts,
        )

    def _make_gr_output(
        self,
        pooling_output: torch.Tensor,
        finish_reason: FinishReason | None,
    ) -> GRRequestOutput:
        """Build a GRRequestOutput via the adapter stored in meta."""
        adapter = self.gr_request_meta.get("_adapter")
        meta = dict(self.gr_request_meta)
        meta["finish_reason"] = finish_reason
        meta.pop("_adapter", None)
        return adapter.process_output(pooling_output, meta)


class GROutputProcessor(VLLMOutputProcessor):
    """OutputProcessor that uses GRRequestState for recommendation requests.

    Overrides add_request() to instantiate GRRequestState (instead of the
    base RequestState). The parent's process_outputs() loop then naturally
    calls GRRequestState.make_request_output(), which produces
    GRRequestOutput for GR requests without any additional overrides.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        log_stats: bool,
        stream_interval: int = 1,
        *,
        adapter=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            log_stats=log_stats,
            stream_interval=stream_interval,
        )
        self._adapter = adapter
        self.request_states: dict[str, GRRequestState] = {}
        self.total_prefix_cache_requests: int = 0
        self.total_prefix_cache_queries: int = 0
        self.total_prefix_cache_hits: int = 0
        self.total_connector_prefix_cache_requests: int = 0
        self.total_connector_prefix_cache_queries: int = 0
        self.total_connector_prefix_cache_hits: int = 0

    @property
    def num_pending_requests(self) -> int:
        return len(self.request_states)

    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        request_id = request.request_id
        if request_id in self.request_states:
            raise ValueError(f"Request id {request_id} already running.")

        req_state = GRRequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=self.stream_interval,
            adapter=self._adapter,
        )

        # Inject adapter reference for output processing
        if req_state.is_gr_request and self._adapter is not None:
            req_state.gr_request_meta["_adapter"] = self._adapter

        if self._requests_drained.is_set():
            self._requests_drained.clear()
        self.request_states[request_id] = req_state
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

        self.external_req_ids[req_state.external_req_id].append(request_id)

    def update_scheduler_stats(self, scheduler_stats: Any | None):
        super().update_scheduler_stats(scheduler_stats)
        if scheduler_stats is None:
            return

        prefix_cache_stats = getattr(scheduler_stats, "prefix_cache_stats", None)
        if prefix_cache_stats is not None:
            self.total_prefix_cache_requests += int(prefix_cache_stats.requests)
            self.total_prefix_cache_queries += int(prefix_cache_stats.queries)
            self.total_prefix_cache_hits += int(prefix_cache_stats.hits)

        connector_stats = getattr(scheduler_stats, "connector_prefix_cache_stats", None)
        if connector_stats is not None:
            self.total_connector_prefix_cache_requests += int(connector_stats.requests)
            self.total_connector_prefix_cache_queries += int(connector_stats.queries)
            self.total_connector_prefix_cache_hits += int(connector_stats.hits)

    def get_prefix_cache_counters(self) -> dict[str, int]:
        return {
            "requests": self.total_prefix_cache_requests,
            "queries": self.total_prefix_cache_queries,
            "hits": self.total_prefix_cache_hits,
            "connector_requests": self.total_connector_prefix_cache_requests,
            "connector_queries": self.total_connector_prefix_cache_queries,
            "connector_hits": self.total_connector_prefix_cache_hits,
        }
