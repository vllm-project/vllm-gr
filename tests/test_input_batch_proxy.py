# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test to verify InputBatchProxy sampling_metadata immutability."""

from typing import List, Optional

import torch

from vllm_gr.v1.worker.model_runner_common import InputBatchProxy


class DummySamplingMetadata:
    def __init__(self, temp_tensor: torch.Tensor) -> None:
        self.temperature: torch.Tensor = temp_tensor
        self.top_p: Optional[torch.Tensor] = None
        self.top_k: Optional[torch.Tensor] = None
        self.min_p: Optional[torch.Tensor] = None
        self.frequency_penalties: Optional[torch.Tensor] = None
        self.presence_penalties: Optional[torch.Tensor] = None
        self.repetition_penalties: Optional[torch.Tensor] = None
        self.allowed_token_ids_mask: Optional[torch.Tensor] = None
        self.prompt_token_ids: Optional[torch.Tensor] = None


class DummyInputBatch:
    def __init__(self, req_ids: List[str], sampling_metadata: DummySamplingMetadata) -> None:
        self.req_ids: List[str] = req_ids
        self.sampling_metadata: DummySamplingMetadata = sampling_metadata


def test_input_batch_proxy_does_not_mutate_underlying_sampling_metadata() -> None:
    # Setup batch of 100 requests
    req_ids = [f"req_{i}" for i in range(100)]
    orig_temp = torch.full((100,), 0.6)
    orig_s_meta = DummySamplingMetadata(orig_temp.clone())
    input_batch = DummyInputBatch(req_ids, orig_s_meta)

    # Logits map has only 28 active rows (e.g. chunked prefill step)
    logits_map = list(range(28))

    proxy = InputBatchProxy(
        obj=input_batch,
        req_map=req_ids,
        idx_map=list(range(100)),
        logits_map=logits_map,
    )

    # Access sampling_metadata via proxy
    proxied_s_meta = proxy.sampling_metadata

    # 1. Proxied metadata should be sliced to 28 elements
    assert proxied_s_meta is not orig_s_meta
    assert proxied_s_meta.temperature.shape[0] == 28
    assert torch.allclose(proxied_s_meta.temperature, torch.full((28,), 0.6))

    # 2. Original input_batch.sampling_metadata MUST remain intact (100 elements)
    assert input_batch.sampling_metadata.temperature.shape[0] == 100
    assert torch.allclose(input_batch.sampling_metadata.temperature, torch.full((100,), 0.6))

    # 3. Next step with 100 active rows (e.g. full decode step)
    logits_map_full = list(range(100))
    proxy_full = InputBatchProxy(
        obj=input_batch,
        req_map=req_ids,
        idx_map=list(range(100)),
        logits_map=logits_map_full,
    )
    proxied_s_meta_full = proxy_full.sampling_metadata
    assert proxied_s_meta_full.temperature.shape[0] == 100
