# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test to verify InputBatchProxy sampling_metadata immutability."""

from typing import Any, Dict, List, Optional

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


def test_input_batch_proxy_remaps_non_tensor_and_penalty_state() -> None:
    req_ids = ["req_0", "req_1", "req_2"]
    gen1 = torch.Generator().manual_seed(42)

    s_meta = DummySamplingMetadata(torch.tensor([0.5, 0.7, 0.9]))
    setattr(s_meta, "generators", {1: gen1})
    setattr(s_meta, "output_token_ids", [[10, 11], [20, 21], [30, 31]])
    setattr(s_meta, "bad_words_token_ids", {1: [[100, 101]]})
    s_meta.frequency_penalties = torch.tensor([0.1, 0.2, 0.3])

    input_batch = DummyInputBatch(req_ids, s_meta)

    # Project row 1 to row 0
    logits_map = [1]
    proxy = InputBatchProxy(
        obj=input_batch,
        req_map=req_ids,
        idx_map=[0, 1, 2],
        logits_map=logits_map,
    )

    p_meta = proxy.sampling_metadata
    assert p_meta is not s_meta

    # Check generator remapped key 1 -> 0
    p_gens: Dict[int, Any] = getattr(p_meta, "generators")
    assert 0 in p_gens
    assert p_gens[0] is gen1
    assert 1 not in p_gens

    # Check output_token_ids sliced
    p_outs: List[List[int]] = getattr(p_meta, "output_token_ids")
    assert p_outs == [[20, 21]]

    # Check bad_words_token_ids remapped key 1 -> 0
    p_bads: Dict[int, List[List[int]]] = getattr(p_meta, "bad_words_token_ids")
    assert p_bads == {0: [[100, 101]]}

    # Check frequency_penalties sliced
    assert p_meta.frequency_penalties is not None
    assert torch.allclose(p_meta.frequency_penalties, torch.tensor([0.2]))

    # Check original input_batch.sampling_metadata unchanged
    assert getattr(s_meta, "generators") == {1: gen1}
    assert getattr(s_meta, "output_token_ids") == [[10, 11], [20, 21], [30, 31]]
    assert getattr(s_meta, "bad_words_token_ids") == {1: [[100, 101]]}


def test_input_batch_proxy_equal_length_non_identity_remap() -> None:
    # Scenario: 2 logical requests, but logits_map=[0, 0] (length 2, equal to orig_req_len, but non-identity!)
    req_ids = ["req_0", "req_1"]
    gen0 = torch.Generator().manual_seed(100)
    gen1 = torch.Generator().manual_seed(200)

    s_meta = DummySamplingMetadata(torch.tensor([0.5, 0.9]))
    setattr(s_meta, "generators", {0: gen0, 1: gen1})
    setattr(s_meta, "output_token_ids", [[10, 11], [20, 21]])

    input_batch = DummyInputBatch(req_ids, s_meta)

    # logits_map=[0, 0] has length 2 (same as orig_req_len=2), but is non-identity
    logits_map = [0, 0]
    proxy = InputBatchProxy(
        obj=input_batch,
        req_map=req_ids,
        idx_map=[0, 1],
        logits_map=logits_map,
    )

    p_meta = proxy.sampling_metadata
    assert p_meta is not s_meta

    # Both beam rows MUST receive request 0's temperature (0.5), NOT request 1's temperature (0.9)
    assert torch.allclose(p_meta.temperature, torch.tensor([0.5, 0.5]))

    # Both beam rows MUST receive request 0's generator
    p_gens: Dict[int, Any] = getattr(p_meta, "generators")
    assert p_gens == {0: gen0, 1: gen0}

    # Both beam rows MUST receive request 0's output_token_ids
    p_outs: List[List[int]] = getattr(p_meta, "output_token_ids")
    assert p_outs == [[10, 11], [10, 11]]


def test_input_batch_proxy_remaps_logitsprocs() -> None:
    class DummyLogitsProcessor:
        def __init__(self) -> None:
            self.req_biases = torch.tensor([1.0, 2.0, 3.0])
            self.req_masks = {0: [10], 2: [30]}

    class DummyLogitsProcessorsHolder:
        def __init__(self, lps: List[Any]) -> None:
            self.argmax_invariant = lps
            self.non_argmax_invariant: List[Any] = []

    req_ids = ["req_0", "req_1", "req_2"]
    s_meta = DummySamplingMetadata(torch.tensor([0.1, 0.2, 0.3]))
    lp = DummyLogitsProcessor()
    setattr(s_meta, "logitsprocs", DummyLogitsProcessorsHolder([lp]))

    input_batch = DummyInputBatch(req_ids, s_meta)

    # logits_map=[2, 0]
    proxy = InputBatchProxy(
        obj=input_batch,
        req_map=req_ids,
        idx_map=[0, 1, 2],
        logits_map=[2, 0],
    )

    p_meta = proxy.sampling_metadata
    assert p_meta is not s_meta
    assert p_meta.logitsprocs is not getattr(s_meta, "logitsprocs")

    remapped_lp = p_meta.logitsprocs.argmax_invariant[0]
    assert torch.allclose(remapped_lp.req_biases, torch.tensor([3.0, 1.0]))
    assert remapped_lp.req_masks == {0: [30], 1: [10]}
