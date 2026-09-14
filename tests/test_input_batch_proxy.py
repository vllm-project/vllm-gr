# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""InputBatchProxy must not leak remapped sampling state across requests."""

from __future__ import annotations

import torch

from vllm_gr.v1.worker.model_runner_common import InputBatchProxy


class _SamplingMetadata:
    def __init__(self, temperature: torch.Tensor) -> None:
        self.temperature = temperature
        self.top_p = None
        self.top_k = None
        self.min_p = None
        self.frequency_penalties = None
        self.presence_penalties = None
        self.repetition_penalties = None


class _InputBatch:
    def __init__(self, req_ids: list[str], sampling_metadata: _SamplingMetadata) -> None:
        self.req_ids = req_ids
        self.sampling_metadata = sampling_metadata


def _proxy(batch: _InputBatch, logits_map: list[int]) -> InputBatchProxy:
    return InputBatchProxy(
        obj=batch,
        req_map=batch.req_ids,
        idx_map=list(range(len(batch.req_ids))),
        logits_map=logits_map,
    )


def test_sampling_metadata_remap_is_copy_on_write() -> None:
    source = _SamplingMetadata(torch.tensor([0.5, 0.9]))
    batch = _InputBatch(["a", "b"], source)

    remapped = _proxy(batch, [0, 0, 1, 1]).sampling_metadata

    assert remapped is not source
    assert torch.equal(remapped.temperature, torch.tensor([0.5, 0.5, 0.9, 0.9]))
    assert torch.equal(source.temperature, torch.tensor([0.5, 0.9]))


def test_equal_length_non_identity_layout_is_still_remapped() -> None:
    source = _SamplingMetadata(torch.tensor([0.5, 0.9]))
    batch = _InputBatch(["a", "b"], source)

    remapped = _proxy(batch, [1, 0]).sampling_metadata

    assert torch.equal(remapped.temperature, torch.tensor([0.9, 0.5]))
    assert torch.equal(source.temperature, torch.tensor([0.5, 0.9]))
