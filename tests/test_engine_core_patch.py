# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for InputBatchProxy sampling_metadata isolation (no GPU required).

Pins the P1 fix for review issue #3: reading ``sampling_metadata`` through the
proxy must return a beam-row *projection* and must never mutate the original
upstream ``SamplingMetadata`` (which is shared with the underlying InputBatch
and survives across requests/steps).
"""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

from types import SimpleNamespace

import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_gr.v1.engine.engine_core_patch import InputBatchProxy

# SamplingMetadata per-request tensor fields the proxy projects. ``min_p`` is
# intentionally absent from upstream SamplingMetadata in vLLM 0.14.1; the proxy
# tolerates it via getattr(..., None) and skips it.
PROJECTED_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalties",
    "presence_penalties",
    "repetition_penalties",
)


def _make_metadata(num_reqs: int = 2) -> SamplingMetadata:
    """Build a real SamplingMetadata with per-request tensors of shape [num_reqs]."""
    idx = torch.arange(num_reqs, dtype=torch.float32)
    return SamplingMetadata(
        temperature=idx + 0.5,
        all_greedy=False,
        all_random=True,
        top_p=idx * 0.1 + 0.7,
        top_k=(idx + 1).to(torch.int64),
        generators={},
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=idx,
        presence_penalties=idx * 0.5,
        repetition_penalties=idx * 0.25,
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _make_input_batch(metadata: SamplingMetadata, req_ids: list[str]) -> SimpleNamespace:
    """A minimal InputBatch stand-in exposing only what the proxy reads."""
    return SimpleNamespace(sampling_metadata=metadata, req_ids=req_ids)


def _snapshot_tensors(meta: SamplingMetadata) -> dict[str, torch.Tensor]:
    return {f: getattr(meta, f) for f in PROJECTED_FIELDS}


class TestInputBatchProxySamplingMetadata:
    def test_projection_correct_for_repeated_index(self) -> None:
        metadata = _make_metadata(num_reqs=2)
        batch = _make_input_batch(metadata, req_ids=["r0", "r1"])
        # logits_map = [1, 0, 1] → 2 requests expanded to 3 beam rows.
        proxy = InputBatchProxy(batch, None, None, [1, 0, 1])

        projected = proxy.sampling_metadata
        for field in PROJECTED_FIELDS:
            orig = getattr(metadata, field)
            got = getattr(projected, field)
            assert got.shape[0] == 3, f"{field}: expected 3 rows, got {got.shape}"
            assert torch.equal(got, orig[[1, 0, 1]]), f"{field}: values not remapped"

    def test_original_metadata_is_never_mutated(self) -> None:
        metadata = _make_metadata(num_reqs=2)
        batch = _make_input_batch(metadata, req_ids=["r0", "r1"])
        before = _snapshot_tensors(metadata)

        proxy = InputBatchProxy(batch, None, None, [1, 0, 1])
        _ = proxy.sampling_metadata  # triggers projection
        _ = proxy.sampling_metadata  # second access must also leave original intact

        for field in PROJECTED_FIELDS:
            assert getattr(metadata, field) is before[field], f"{field}: identity changed"
            assert getattr(metadata, field).shape == before[field].shape, f"{field}: shape changed"
            assert torch.equal(getattr(metadata, field), before[field]), f"{field}: values changed"
            assert getattr(metadata, field).shape[0] == 2, f"{field}: original expanded"

    def test_projected_returned_object_is_not_the_original(self) -> None:
        metadata = _make_metadata(num_reqs=2)
        batch = _make_input_batch(metadata, req_ids=["r0", "r1"])
        proxy = InputBatchProxy(batch, None, None, [1, 0, 1])

        projected = proxy.sampling_metadata
        assert projected is not metadata, "projection must return a distinct object"

    def test_unprojectable_fields_keep_original_value(self) -> None:
        metadata = _make_metadata(num_reqs=2)
        # Force top_p to None and frequency_penalties to an empty tensor; these
        # must be carried through unchanged (not projected, not dropped).
        metadata.top_p = None
        metadata.frequency_penalties = torch.empty(0)
        batch = _make_input_batch(metadata, req_ids=["r0", "r1"])
        proxy = InputBatchProxy(batch, None, None, [1, 0, 1])

        projected = proxy.sampling_metadata
        assert projected.top_p is None
        # Empty tensor: numel()==0, so it is left as-is (same object).
        assert projected.frequency_penalties is metadata.frequency_penalties

        # A field whose first dim does not match orig_req_len is also left alone.
        metadata.repetition_penalties = torch.zeros(7)
        proxy2 = InputBatchProxy(batch, None, None, [1, 0, 1])
        projected2 = proxy2.sampling_metadata
        assert projected2.repetition_penalties is metadata.repetition_penalties

    def test_short_circuit_returns_original_object(self) -> None:
        metadata = _make_metadata(num_reqs=2)
        batch = _make_input_batch(metadata, req_ids=["r0", "r1"])

        # No logits_map → original.
        proxy_no_map = InputBatchProxy(batch, None, None, None)
        assert proxy_no_map.sampling_metadata is metadata

        # logits_map length == orig_req_len → no expansion → original (identity).
        proxy_same_len = InputBatchProxy(batch, None, None, [0, 1])
        assert proxy_same_len.sampling_metadata is metadata

        # No metadata at all.
        empty_batch = SimpleNamespace(sampling_metadata=None, req_ids=["r0"])
        proxy_empty = InputBatchProxy(empty_batch, None, None, [0])
        assert proxy_empty.sampling_metadata is None

    def test_two_different_logits_maps_do_not_pollute_each_other(self) -> None:
        metadata = _make_metadata(num_reqs=2)
        batch = _make_input_batch(metadata, req_ids=["r0", "r1"])

        proxy_a = InputBatchProxy(batch, None, None, [1, 0, 1])
        proj_a = proxy_a.sampling_metadata

        proxy_b = InputBatchProxy(batch, None, None, [0, 1])
        proj_b = proxy_b.sampling_metadata

        # Each projection reflects its own logits_map, built from the untouched
        # original.
        assert torch.equal(proj_a.temperature, metadata.temperature[[1, 0, 1]])
        assert torch.equal(proj_b.temperature, metadata.temperature[[0, 1]])
        # Original still pristine.
        assert metadata.temperature.shape[0] == 2

    def test_beam_then_ordinary_request_reads_original_metadata(self) -> None:
        # Simulates the production lifecycle: the proxy is installed for a beam
        # step (read happens here), then uninstalled by the finally block; the
        # next ordinary request reads the InputBatch directly and must see the
        # unprojected original metadata — the exact P1 leak this PR fixes.
        metadata = _make_metadata(num_reqs=2)
        batch = _make_input_batch(metadata, req_ids=["r0", "r1"])

        proxy = InputBatchProxy(batch, None, None, [1, 0, 1])
        _ = proxy.sampling_metadata

        # Proxy "uninstalled" — ordinary request reads the batch directly.
        assert batch.sampling_metadata is metadata
        for field in PROJECTED_FIELDS:
            assert getattr(batch.sampling_metadata, field).shape[0] == 2, (
                f"{field}: leaked beam-row shape into the original InputBatch"
            )
