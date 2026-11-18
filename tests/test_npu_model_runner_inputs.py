# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="no-untyped-def"
"""Tests for NPU ModelRunner Beam input remapping."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker import npu_model_runner_inputs as inputs


def test_position_helpers_support_buffer_forms() -> None:
    device = torch.full((8,), -1, dtype=torch.long)
    host = torch.full((8,), -1, dtype=torch.long)
    buffered = SimpleNamespace(positions=SimpleNamespace(gpu=device, cpu=host))
    plain = SimpleNamespace(positions=device, _positions_cpu_buf=host)
    assert inputs._get_positions_buffers(buffered) == (device, host)
    assert inputs._get_positions_buffers(plain) == (device, host)


def test_beam_remap_requires_device_positions() -> None:
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["beam"], logprob_token_ids={}),
        positions=None,
        _positions_cpu_buf=None,
    )
    with pytest.raises(RuntimeError, match="position buffer"):
        inputs._beam_remap_inputs(runner, SimpleNamespace(beam_data={}), [1])


def test_incremental_beam_decode_fills_constant_position_ids() -> None:
    device = torch.full((6,), -1, dtype=torch.long)
    host = torch.full((6,), -1, dtype=torch.long)

    logits_indices: list[int] = []
    result = inputs._process_beam_decode_request(
        {
            "prefix_len": 10,
            "beam_width": 3,
            "beam_decode_steps": 4,
            "cache_len": 12,
            "decode_step": 2,
        },
        seq_len=3,
        token_offset=1,
        device_pos=device,
        cpu_pos=host,
        new_logits_indices=logits_indices,
    )

    assert result == (3, 3, 4, [0, 1, 2])
    assert logits_indices == [1, 2, 3]
    # Every beam consumes one token at position prefix_len + decode_step.
    assert device.tolist() == [-1, 12, 12, 12, -1, -1]
    assert torch.equal(device, host)


def test_mixed_batch_remap_installs_functional_proxy() -> None:
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([10, 20, 30, 40]),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    input_batch = SimpleNamespace(
        req_ids=["ordinary", "beam", None, "prefill"],
        logprob_token_ids={"stale": [1]},
        num_computed_tokens_cpu=[5, 0, 0, 2],
        num_prompt_tokens=[5, 0, 0, 4],
        num_tokens_no_spec=[10, 20, 30, 40],
        num_tokens=[11, 21, 31, 41],
        token_ids_cpu=torch.arange(16).view(4, 4),
        sampling_metadata=sampling_metadata,
        passthrough="original",
    )
    positions = SimpleNamespace(
        gpu=torch.full((12,), -1, dtype=torch.long),
        cpu=torch.full((12,), -1, dtype=torch.long),
    )
    runner = SimpleNamespace(input_batch=input_batch, positions=positions)
    scheduler_output = SimpleNamespace(
        beam_data={
            "beam": {
                "is_beam_decode": True,
                "prefix_len": 4,
                "beam_width": 2,
                "beam_decode_steps": 2,
                "cache_len": 4,
                "decode_step": 1,
            }
        }
    )

    logits_indices = inputs._beam_remap_inputs(runner, scheduler_output, [2, 2, 1, 3])

    assert logits_indices == [2, 3, 7]
    assert positions.gpu[2:4].tolist() == [5, 5]
    assert torch.equal(positions.gpu, positions.cpu)
    assert scheduler_output._gr_request_row_configs == [
        ("ordinary", 0, 0, []),
        ("beam", 1, 2, [0, 1]),
        ("prefill", 4, 1, []),
    ]

    proxy = runner.input_batch
    assert isinstance(proxy, inputs.InputBatchProxy)
    assert proxy.req_ids.copy() == ["ordinary", "beam", "beam", None, "prefill"]
    assert proxy.sampling_metadata.temperature.tolist() == [20, 20, 40]
    assert proxy.num_tokens_no_spec[2] == 20
    proxy.num_tokens[2] = 99
    assert input_batch.num_tokens[1] == 99
    proxy.is_token_ids[4, 0] = 77
    assert input_batch.token_ids_cpu[3, 0].item() == 77
    proxy.passthrough = "updated"
    assert input_batch.passthrough == "updated"
    assert input_batch.logprob_token_ids == {}


def test_incremental_layout_cache_short_circuits_remap() -> None:
    """A pure incremental-beam batch replays its cached row layout.

    Frame 2 with the same req_ids / token counts skips the per-beam Python
    loop: the four host lists and logits indices are reused from the cache,
    and only the step-dependent position ids are refreshed (decode_step 0
    writes prefix_len, decode_step 2 writes prefix_len + 2).
    """
    # Shared runner skeleton (the production runner persists across decode
    # steps, so the layout cache must live on it).
    shared_runner = SimpleNamespace()

    def make_frame(
        step: int, positions: SimpleNamespace
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        input_batch = SimpleNamespace(
            req_ids=["beam"],
            logprob_token_ids={"stale": [1]},
            num_computed_tokens_cpu=[0],
            num_prompt_tokens=[0],
            sampling_metadata=SimpleNamespace(
                temperature=torch.tensor([7.0]),
                top_p=None,
                top_k=None,
                min_p=None,
                frequency_penalties=None,
                presence_penalties=None,
                repetition_penalties=None,
            ),
        )
        shared_runner.input_batch = input_batch
        shared_runner.positions = positions
        scheduler_output = SimpleNamespace(
            beam_data={
                "beam": {
                    "is_beam_decode": True,
                    "prefix_len": 10,
                    "beam_width": 4,
                    "beam_decode_steps": 2,
                    "cache_len": 10,
                    "decode_step": step,
                }
            }
        )
        return shared_runner, scheduler_output

    # Frame 1 (decode_step=0): slow path populates the layout cache.
    pos1 = SimpleNamespace(
        gpu=torch.zeros((4,), dtype=torch.long),
        cpu=torch.zeros((4,), dtype=torch.long),
    )
    r1, so1 = make_frame(0, pos1)
    idx1 = inputs._beam_remap_inputs(r1, so1, [4])
    assert idx1 == [0, 1, 2, 3]
    assert pos1.gpu.tolist() == [10, 10, 10, 10]
    assert so1._gr_request_row_configs == [("beam", 0, 4, [0, 1, 2, 3])]

    # Frame 2 (decode_step=2): cache hit short-circuits the loop; the
    # reused layout lists are the SAME objects and positions refresh.
    pos2 = SimpleNamespace(
        gpu=torch.zeros((4,), dtype=torch.long),
        cpu=torch.zeros((4,), dtype=torch.long),
    )
    r2, so2 = make_frame(2, pos2)
    idx2 = inputs._beam_remap_inputs(r2, so2, [4])
    assert idx2 == [0, 1, 2, 3]
    assert pos2.gpu.tolist() == [12, 12, 12, 12]
    assert torch.equal(pos2.gpu, pos2.cpu)
    assert so2._gr_request_row_configs == [("beam", 0, 4, [0, 1, 2, 3])]
    assert so2._gr_request_row_configs[0] is so1._gr_request_row_configs[0]

    # A token count inconsistent with the Beam width is rejected, even after
    # this request has populated the layout cache.
    pos3 = SimpleNamespace(
        gpu=torch.zeros((5,), dtype=torch.long),
        cpu=torch.zeros((5,), dtype=torch.long),
    )
    r3, so3 = make_frame(2, pos3)
    with pytest.raises(ValueError, match="one query token per beam"):
        inputs._beam_remap_inputs(r3, so3, [5])

    # Reusing a request id for a different prompt must rebuild the valid
    # geometry instead of reusing stale positions or rejecting the new session.
    pos4 = SimpleNamespace(gpu=torch.zeros(4), cpu=torch.zeros(4))
    r4, so4 = make_frame(0, pos4)
    so4.beam_data["beam"].update(prefix_len=11, cache_len=11)
    assert inputs._beam_remap_inputs(r4, so4, [4]) == [0, 1, 2, 3]
    assert pos4.gpu.tolist() == [11] * 4
    assert so4._gr_request_row_configs[0] is not so2._gr_request_row_configs[0]


def test_sampling_metadata_remap_is_cached_and_warmed_at_remap() -> None:
    """The remap runs once at proxy install; repeated reads hit the cache."""
    sampling_metadata = SimpleNamespace(
        temperature=torch.tensor([20]),
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
    )
    input_batch = SimpleNamespace(
        req_ids=["beam"],
        logprob_token_ids={},
        sampling_metadata=sampling_metadata,
    )
    runner = SimpleNamespace(
        input_batch=input_batch,
        positions=SimpleNamespace(
            gpu=torch.zeros(8, dtype=torch.long),
            cpu=torch.zeros(8, dtype=torch.long),
        ),
    )
    scheduler_output = SimpleNamespace(
        beam_data={
            "beam": {
                "is_beam_decode": True,
                "prefix_len": 4,
                "beam_width": 2,
                "beam_decode_steps": 2,
                "cache_len": 4,
                "decode_step": 1,
            }
        }
    )

    inputs._beam_remap_inputs(runner, scheduler_output, [2])
    proxy = runner.input_batch

    # Eager warm: the metadata was remapped during _beam_remap_inputs
    # (prepare phase), not deferred to the first _sample access.  One beam
    # request with beam_width=2 → the 1-row temperature expands to 2 rows.
    assert sampling_metadata.temperature.tolist() == [20, 20]
    cache = object.__getattribute__(proxy, "_sampling_metadata_cache")
    assert cache is sampling_metadata
    idx_cache = object.__getattribute__(proxy, "_logits_idx_tensor_cache")
    assert idx_cache is not None and idx_cache[1].tolist() == [0, 0]

    # Cache hit: a fresh request-length tensor planted after the first read
    # must NOT be re-expanded (the remap loop is skipped entirely on the
    # second access).
    sampling_metadata.temperature = torch.tensor([30])
    assert proxy.sampling_metadata is sampling_metadata
    assert proxy.sampling_metadata.temperature.tolist() == [30]


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


@pytest.mark.parametrize("step", [None, -1, True])  # type: ignore[misc]
def test_beam_input_rejects_missing_or_invalid_step(step: Any) -> None:
    metadata = {"prefix_len": 4, "cache_len": 4, "beam_width": 2}
    if step is not None:
        metadata["decode_step"] = step
    positions = torch.full((4,), -1)
    logits: list[int] = []
    with pytest.raises(ValueError, match="decode_step"):
        inputs._process_beam_decode_request(metadata, 2, 0, positions, None, logits)
    assert positions.tolist() == [-1] * 4
    assert not logits


def test_beam_input_rejects_full_suffix_rows() -> None:
    metadata = {"prefix_len": 4, "cache_len": 4, "beam_width": 2, "decode_step": 1}
    with pytest.raises(ValueError, match="one query token per beam"):
        inputs._process_beam_decode_request(metadata, 4, 0, None, None, [])
