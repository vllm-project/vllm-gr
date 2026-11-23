# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="no-untyped-def"
"""Tests for NPU ModelRunner Beam input views and indices."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker.npu import npu_model_runner_inputs as inputs


def test_position_helpers_support_buffer_forms() -> None:
    device = torch.full((8,), -1, dtype=torch.long)
    host = torch.full((8,), -1, dtype=torch.long)
    buffered = SimpleNamespace(positions=SimpleNamespace(gpu=device, cpu=host))
    plain = SimpleNamespace(positions=device, _positions_cpu_buf=host)
    assert inputs._get_positions_buffers(buffered) == (device, host)
    assert inputs._get_positions_buffers(plain) == (device, host)


def test_input_batch_proxy_remaps_decode_rows() -> None:
    sampling_metadata = SimpleNamespace(temperature=torch.tensor([10, 20]))
    input_batch = SimpleNamespace(
        req_ids=["first", "second"],
        num_tokens_no_spec=[10, 20],
        num_tokens=[11, 21],
        token_ids_cpu=torch.arange(8).view(2, 4),
        sampling_metadata=sampling_metadata,
        passthrough="original",
    )
    proxy = inputs.InputBatchProxy(
        input_batch,
        ["first", "first", "second", "second"],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        device=torch.device("cpu"),
        req_ids_key=("first", "second"),
    )

    assert proxy.req_ids.copy() == ["first", "first", "second", "second"]
    assert proxy.sampling_metadata.temperature.tolist() == [10, 10, 20, 20]
    assert proxy.num_tokens_no_spec[2] == 20
    proxy.num_tokens[2] = 99
    assert input_batch.num_tokens[1] == 99
    proxy.is_token_ids[3, 0] = 77
    assert input_batch.token_ids_cpu[1, 0].item() == 77
    proxy.passthrough = "updated"
    assert input_batch.passthrough == "updated"


def test_sampling_metadata_remap_is_cached_within_proxy() -> None:
    sampling_metadata = SimpleNamespace(temperature=torch.tensor([20]))
    input_batch = SimpleNamespace(req_ids=["beam"], sampling_metadata=sampling_metadata)
    proxy = inputs.InputBatchProxy(
        input_batch, ["beam", "beam"], [0, 0], [0, 0], device=torch.device("cpu")
    )

    assert proxy.sampling_metadata is sampling_metadata
    assert sampling_metadata.temperature.tolist() == [20, 20]

    sampling_metadata.temperature = torch.tensor([30])
    assert proxy.sampling_metadata is sampling_metadata
    assert sampling_metadata.temperature.tolist() == [30]


def test_sampling_metadata_cache_invalidated_by_rebuild_and_assignment() -> None:
    """A metadata swap (refresh_metadata / setattr) forces a fresh remap."""
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([10]),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    input_batch = SimpleNamespace(req_ids=["beam"])
    proxy = inputs.InputBatchProxy(
        input_batch, ["beam", "beam"], [0, 0], [0, 0], device=torch.device("cpu")
    )
    input_batch.sampling_metadata = sampling_metadata

    assert proxy.sampling_metadata is sampling_metadata
    assert sampling_metadata.temperature.tolist() == [10, 10]

    # Mid-step rebuild (refresh_metadata): a new metadata object with fresh
    # per-request tensors is picked up and remapped despite the cache.
    rebuilt = SimpleNamespace(
        temperature=torch.tensor([7]),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    input_batch.sampling_metadata = rebuilt
    assert proxy.sampling_metadata is rebuilt
    assert rebuilt.temperature.tolist() == [7, 7]

    # Assignment through the proxy forwards to the underlying batch AND
    # invalidates the cache, so the next read remaps the new object.
    replaced = SimpleNamespace(
        temperature=torch.tensor([5]),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    proxy.sampling_metadata = replaced
    assert input_batch.sampling_metadata is replaced
    assert proxy.sampling_metadata is replaced
    assert replaced.temperature.tolist() == [5, 5]


def test_sampling_metadata_batched_gather_matches_per_attr_index() -> None:
    """Mixed-dtype sampling attrs are all expanded by the shared logits map.

    The batched gather must produce exactly what the legacy per-attribute
    ``val[idx_tensor]`` produced, for both float and int attrs in one pass.
    """
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([0.5, 1.5], dtype=torch.float32),
        top_p=torch.tensor([0.9, 0.8], dtype=torch.float32),
        top_k=torch.tensor([32, 16], dtype=torch.int32),
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    input_batch = SimpleNamespace(req_ids=["a", "b"], sampling_metadata=sampling_metadata)
    proxy = inputs.InputBatchProxy(
        input_batch, ["a", "b"], [0, 0, 1], [0, 0, 1], device=torch.device("cpu")
    )

    meta = proxy.sampling_metadata
    assert meta is sampling_metadata
    assert meta.temperature.tolist() == [0.5, 0.5, 1.5]
    assert meta.top_p.dtype is torch.float32
    assert torch.equal(meta.top_p, torch.tensor([0.9, 0.9, 0.8], dtype=torch.float32))
    assert meta.top_k.tolist() == [32, 32, 16]


def test_sampling_metadata_batched_gather_mixed_presence() -> None:
    """Only present attrs are remapped; missing ones stay None."""
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([0.5], dtype=torch.float32),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    input_batch = SimpleNamespace(req_ids=["a"], sampling_metadata=sampling_metadata)
    proxy = inputs.InputBatchProxy(
        input_batch, ["a", "a"], [0, 0], [0, 0], device=torch.device("cpu")
    )

    meta = proxy.sampling_metadata
    assert meta.temperature.tolist() == [0.5, 0.5]
    assert meta.top_p is None
    assert meta.top_k is None


def test_logits_idx_tensor_shared_cache_reused_across_proxies() -> None:
    """Consecutive proxies with the same logits_map share one idx tensor.

    Per-frame InputBatchProxy instances are rebuilt every decode step, so the
    instance-level cache would rebuild the host->device logits-row index tensor
    each frame.  The runner-owned shared cache dedups that for the steady-state
    (unchanged beam geometry) case.
    """
    shared: dict[Any, Any] = {}
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([0.5], dtype=torch.float32),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )

    def make_proxy() -> inputs.InputBatchProxy:
        input_batch = SimpleNamespace(req_ids=["a"], sampling_metadata=sampling_metadata)
        return inputs.InputBatchProxy(
            input_batch,
            ["a", "a"],
            [0, 0],
            [0, 0],
            device=torch.device("cpu"),
            shared_idx_cache=shared,
        )

    p1 = make_proxy()
    meta = p1.sampling_metadata
    assert meta.temperature.tolist() == [0.5, 0.5]
    assert len(shared) == 1

    # Second frame: identical logits_map -> same cached idx tensor, no rebuild.
    p2 = make_proxy()
    meta2 = p2.sampling_metadata
    assert meta2.temperature.tolist() == [0.5, 0.5]
    assert len(shared) == 1

    # A different logits_map replaces the single-slot entry.
    fresh_meta = SimpleNamespace(
        temperature=torch.tensor([1.0, 2.0], dtype=torch.float32),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    p3 = inputs.InputBatchProxy(
        SimpleNamespace(req_ids=["a", "b"], sampling_metadata=fresh_meta),
        ["a", "a", "b"],
        [0, 0, 1],
        [0, 0, 1],
        device=torch.device("cpu"),
        shared_idx_cache=shared,
    )
    meta3 = p3.sampling_metadata
    assert meta3.temperature.tolist() == [1.0, 1.0, 2.0]
    assert len(shared) == 1


def test_sampling_remap_shared_cache_skips_gather_on_hit() -> None:
    """A second frame's proxy reuses the beam-row expansion of sampling attrs.

    Per-frame ``refresh_metadata`` builds a NEW SamplingMetadata object with
    fresh per-request tensors, so the instance-level identity cache always
    misses.  The runner-owned shared cache keys on (device, orig_req_len,
    logits_map): an unchanged beam geometry hits and attaches the previously
    gathered beam-row tensors without re-running stack/index_select/unbind.
    """
    shared_idx: dict[Any, Any] = {}
    shared_sampling: dict[Any, Any] = {}

    def make_proxy(temp_value: float, req_ids: list[str]) -> inputs.InputBatchProxy:
        meta = SimpleNamespace(
            temperature=torch.tensor([temp_value], dtype=torch.float32),
            top_p=torch.tensor([0.9], dtype=torch.float32),
            top_k=None,
            min_p=None,
            frequency_penalties=None,
            presence_penalties=None,
            repetition_penalties=None,
        )
        batch = SimpleNamespace(req_ids=req_ids, sampling_metadata=meta)
        return inputs.InputBatchProxy(
            batch,
            ["a", "a"],
            [0, 0],
            [0, 0],
            device=torch.device("cpu"),
            shared_idx_cache=shared_idx,
            shared_sampling_cache=shared_sampling,
            req_ids_key=tuple(req_ids),
        )

    # First frame (geometry miss): gather runs and populates the cache.
    p1 = make_proxy(0.5, ["a"])
    m1 = p1.sampling_metadata
    assert m1.temperature.tolist() == [0.5, 0.5]
    assert m1.top_p.tolist() == pytest.approx([0.9, 0.9])
    assert len(shared_sampling) == 1

    # Second frame: NEW metadata object (refresh_metadata rebuild) with the
    # same per-request VALUES — a hit attaches the cached beam-row tensors.
    p2 = make_proxy(0.5, ["a"])
    m2 = p2.sampling_metadata
    assert m2 is not m1
    assert m2.temperature.tolist() == [0.5, 0.5]
    assert m2.top_p.tolist() == pytest.approx([0.9, 0.9])
    # The attached tensors are the SAME objects (no second gather).
    assert m2.temperature is m1.temperature
    assert m2.top_p is m1.top_p
    assert len(shared_sampling) == 1


def test_sampling_remap_cache_isolated_across_requests() -> None:
    """A later request with the same layout must NOT inherit the previous
    request's gathered sampling tensors.

    The cache key includes the request ids: request B (same beam width, same
    logits_map, DIFFERENT temperature) misses request A's entry and gathers
    from its own per-request params.  Without the ids in the key, B would
    silently sample with A's temperature.
    """
    shared_idx: dict[Any, Any] = {}
    shared_sampling: dict[Any, Any] = {}

    def make_proxy(temp_value: float, req_id: str) -> inputs.InputBatchProxy:
        meta = SimpleNamespace(
            temperature=torch.tensor([temp_value], dtype=torch.float32),
            top_p=torch.tensor([0.9], dtype=torch.float32),
            top_k=None,
            min_p=None,
            frequency_penalties=None,
            presence_penalties=None,
            repetition_penalties=None,
        )
        batch = SimpleNamespace(req_ids=[req_id], sampling_metadata=meta)
        return inputs.InputBatchProxy(
            batch,
            [req_id, req_id],
            [0, 0],
            [0, 0],
            device=torch.device("cpu"),
            shared_idx_cache=shared_idx,
            shared_sampling_cache=shared_sampling,
            req_ids_key=(req_id,),
        )

    # Request A: temperature 0.5.
    pa = make_proxy(0.5, "req-a")
    ma = pa.sampling_metadata
    assert ma.temperature.tolist() == [0.5, 0.5]

    # Request B (sequential, same layout, temperature 1.5): must use ITS OWN
    # params — a geometry-only key would return A's [0.5, 0.5] here.
    pb = make_proxy(1.5, "req-b")
    mb = pb.sampling_metadata
    assert mb.temperature.tolist() == [1.5, 1.5]
    assert mb.temperature is not ma.temperature
    # Single-slot cache: B's entry replaced A's.
    assert len(shared_sampling) == 1

    # Request A returns (e.g. re-issued request id): re-gathers A's values.
    pa2 = make_proxy(0.5, "req-a")
    assert pa2.sampling_metadata.temperature.tolist() == [0.5, 0.5]


def test_materialize_logits_indices_reuses_runner_buffer() -> None:
    """The runner-owned buffer is rewritten in place across beam steps."""
    runner = SimpleNamespace(device=torch.device("cpu"))
    first = inputs._materialize_logits_indices(runner, [3, 5, 9])
    assert first.dtype is torch.int32
    assert first.tolist() == [3, 5, 9]

    second = inputs._materialize_logits_indices(runner, [10, 11])
    assert second.dtype is torch.int32
    assert second.tolist() == [10, 11]
    # Both views alias the same underlying storage (reused buffer).
    assert first.data_ptr() == second.data_ptr()


def test_materialize_logits_indices_contiguous_device_range() -> None:
    """A contiguous index list [0..n) is materialized device-side (no H2D)."""
    runner = SimpleNamespace(device=torch.device("cpu"))
    out = inputs._materialize_logits_indices(runner, [0, 1, 2, 3, 4])
    assert out.dtype is torch.int32
    assert out.tolist() == [0, 1, 2, 3, 4]

    # Non-contiguous index lists still follow the host copy path and share
    # the same backing storage.
    other = inputs._materialize_logits_indices(runner, [7, 9])
    assert other.tolist() == [7, 9]
    assert out.data_ptr() == other.data_ptr()


def test_is_contiguous_range() -> None:
    assert inputs._is_contiguous_range([]) is True
    assert inputs._is_contiguous_range([0]) is True
    assert inputs._is_contiguous_range([0, 1, 2, 3]) is True
    assert inputs._is_contiguous_range([0, 1, 3]) is False
    assert inputs._is_contiguous_range([1, 2, 3]) is False
    assert inputs._is_contiguous_range([0, 2, 3]) is False


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
