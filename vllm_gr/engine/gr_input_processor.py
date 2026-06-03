# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GRInputProcessor — extends vLLM's InputProcessor to handle GRPrompt.

Delegates feature extraction, validation, payload construction, and
prefix-cache building to a ``GRModelAdapter`` instance, making the
processor model-family-agnostic.
"""

import logging
import time
from collections.abc import Mapping
from typing import Any

from vllm.config import VllmConfig
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor

from vllm_gr.adapters.capabilities import GRPrefixCacheSupport
from vllm_gr.engine import GREngineCoreRequest
from vllm_gr.inputs.data import GRPrompt, is_gr_prompt

logger = logging.getLogger("vllm_gr.input_processor")


class GRInputProcessor(InputProcessor):
    """Input processor for the GR-only engine contract.

    For GRPrompt inputs, delegates all feature extraction, validation,
    payload construction, and prefix-cache token building to the injected
    ``adapter``.  The processor itself is model-family-agnostic.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        tokenizer: TokenizerLike | None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        *,
        adapter=None,
    ):
        super().__init__(vllm_config, tokenizer, mm_registry)
        self._adapter = adapter

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> EngineCoreRequest:
        if not is_gr_prompt(prompt):
            raise TypeError(
                f"GRInputProcessor only supports GRPrompt inputs; got {type(prompt).__name__}"
            )

        return self._process_gr_prompt(
            request_id=request_id,
            prompt=prompt,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )

    def _process_gr_prompt(
        self,
        request_id: str,
        prompt: GRPrompt,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> GREngineCoreRequest:
        adapter = self._adapter

        self._validate_lora(lora_request)
        if arrival_time is None:
            arrival_time = time.time()

        if not isinstance(params, PoolingParams):
            raise TypeError(
                "GRInputProcessor only accepts PoolingParams for GR prompts; "
                f"got {type(params).__name__}"
            )

        # Step 1: Parse and validate
        adapter.validate_prompt(prompt)
        parsed = adapter.parse_prompt(prompt)

        # Step 2: Build payload and encode for cross-process transport
        from vllm_gr.engine.gr_features_codec import encode_gr_features_payload

        payload = adapter.build_payload(parsed)
        gr_features = encode_gr_features_payload(payload)

        # Step 3: Build prefix-cache token IDs and salt (if supported)
        prompt_token_ids = None
        cache_salt = None
        if isinstance(adapter, GRPrefixCacheSupport):
            prompt_token_ids = adapter.build_prompt_token_ids(parsed)
            cache_salt = adapter.build_cache_salt(parsed)

        if prompt_token_ids is None:
            prompt_token_ids = [0]

        request = GREngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=None,
            sampling_params=None,
            pooling_params=params,
            eos_token_id=None,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=cache_salt,
            data_parallel_rank=data_parallel_rank,
            priority=priority,
            trace_headers=trace_headers,
            gr_features=gr_features,
        )

        logger.debug(
            "Processed GR request %s: uih_keys=%s, history_len=%d, "
            "cand_keys=%s, num_candidates=%d, schema=%s",
            request_id,
            parsed.raw_features.get("uih_keys"),
            parsed.metadata.get("history_len"),
            parsed.raw_features.get("cand_keys"),
            parsed.metadata.get("num_candidates"),
            parsed.schema_fingerprint,
        )

        return request
