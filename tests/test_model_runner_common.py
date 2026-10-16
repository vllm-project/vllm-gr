# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="no-untyped-def"
"""Focused tests for device-neutral ModelRunner Beam helpers."""

from __future__ import annotations

import gc
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.worker import model_runner_common as common


def test_position_helpers_support_buffer_forms_and_partial_beam() -> None:
    device = torch.full((8,), -1, dtype=torch.long)
    host = torch.full((8,), -1, dtype=torch.long)

    buffered_runner = SimpleNamespace(positions=SimpleNamespace(gpu=device, cpu=host))
    assert common._get_positions_buffers(buffered_runner) == (device, host)

    plain_runner = SimpleNamespace(positions=device, _positions_cpu_buf=host)
    assert common._get_positions_buffers(plain_runner) == (device, host)

    logits_indices: list[int] = []
    result = common._process_beam_decode_request(
        {
            "prefix_len": 10,
            "beam_width": 2,
            "beam_decode_steps": 4,
            "cache_len": 12,
        },
        seq_len=3,
        token_offset=1,
        gpu_pos=device,
        cpu_pos=host,
        new_logits_indices=logits_indices,
    )

    assert result == (2, 1, 4, [0])
    assert logits_indices == [2]
    assert device.tolist() == [-1, 12, 13, 10, -1, -1, -1, -1]
    assert torch.equal(device, host)

    common._write_position_segments(0, [], [], device, host)
    assert torch.equal(device, host)


def test_incremental_beam_decode_fills_constant_position_ids() -> None:
    device = torch.full((6,), -1, dtype=torch.long)
    host = torch.full((6,), -1, dtype=torch.long)

    logits_indices: list[int] = []
    result = common._process_beam_decode_request(
        {
            "prefix_len": 10,
            "beam_width": 3,
            "beam_decode_steps": 4,
            "cache_len": 12,
            "decode_step": 2,
        },
        seq_len=3,
        token_offset=1,
        gpu_pos=device,
        cpu_pos=host,
        new_logits_indices=logits_indices,
        is_incremental=True,
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
            }
        }
    )

    logits_indices = common._beam_remap_inputs(runner, scheduler_output, [2, 4, 1, 3])

    assert logits_indices == [3, 5, 9]
    assert positions.gpu[2:6].tolist() == [4, 5, 4, 5]
    assert torch.equal(positions.gpu, positions.cpu)
    assert scheduler_output._gr_request_row_configs == [
        ("ordinary", 0, 0, []),
        ("beam", 1, 2, [0, 1]),
        ("prefill", 4, 1, []),
    ]

    proxy = runner.input_batch
    assert isinstance(proxy, common.InputBatchProxy)
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
    idx1 = common._beam_remap_inputs(r1, so1, [4])
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
    idx2 = common._beam_remap_inputs(r2, so2, [4])
    assert idx2 == [0, 1, 2, 3]
    assert pos2.gpu.tolist() == [12, 12, 12, 12]
    assert torch.equal(pos2.gpu, pos2.cpu)
    assert so2._gr_request_row_configs == [("beam", 0, 4, [0, 1, 2, 3])]
    assert so2._gr_request_row_configs[0] is so1._gr_request_row_configs[0]

    # A different token count (different layout key) must NOT hit the cache:
    # the slow path reruns and builds a fresh (equal-valued) layout.  The row
    # count still tracks beam_width (each beam contributes one token), so the
    # logits list keeps 4 entries; only the per-frame token offsets differ.
    pos3 = SimpleNamespace(
        gpu=torch.zeros((5,), dtype=torch.long),
        cpu=torch.zeros((5,), dtype=torch.long),
    )
    r3, so3 = make_frame(2, pos3)
    idx3 = common._beam_remap_inputs(r3, so3, [5])
    assert idx3 == [0, 1, 2, 3]
    assert so3._gr_request_row_configs[0] is not so2._gr_request_row_configs[0]


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
            }
        }
    )

    common._beam_remap_inputs(runner, scheduler_output, [4])
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
    proxy = common.InputBatchProxy(input_batch, ["beam", "beam"], [0, 0], [0, 0])
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
    proxy = common.InputBatchProxy(input_batch, ["a", "b"], [0, 0, 1], [0, 0, 1])

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
    proxy = common.InputBatchProxy(input_batch, ["a", "a"], [0, 0], [0, 0])

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

    def make_proxy() -> common.InputBatchProxy:
        input_batch = SimpleNamespace(req_ids=["a"], sampling_metadata=sampling_metadata)
        return common.InputBatchProxy(
            input_batch, ["a", "a"], [0, 0], [0, 0], shared_idx_cache=shared
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
    p3 = common.InputBatchProxy(
        SimpleNamespace(req_ids=["a", "b"], sampling_metadata=fresh_meta),
        ["a", "a", "b"],
        [0, 0, 1],
        [0, 0, 1],
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

    def make_proxy(temp_value: float, req_ids: list[str]) -> common.InputBatchProxy:
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
        return common.InputBatchProxy(
            batch,
            ["a", "a"],
            [0, 0],
            [0, 0],
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

    def make_proxy(temp_value: float, req_id: str) -> common.InputBatchProxy:
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
        return common.InputBatchProxy(
            batch,
            [req_id, req_id],
            [0, 0],
            [0, 0],
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
    first = common._materialize_logits_indices(runner, [3, 5, 9])
    assert first.dtype is torch.int32
    assert first.tolist() == [3, 5, 9]

    second = common._materialize_logits_indices(runner, [10, 11])
    assert second.dtype is torch.int32
    assert second.tolist() == [10, 11]
    # Both views alias the same underlying storage (reused buffer).
    assert first.data_ptr() == second.data_ptr()


def test_materialize_logits_indices_contiguous_device_range() -> None:
    """A contiguous index list [0..n) is materialized device-side (no H2D)."""
    runner = SimpleNamespace(device=torch.device("cpu"))
    out = common._materialize_logits_indices(runner, [0, 1, 2, 3, 4])
    assert out.dtype is torch.int32
    assert out.tolist() == [0, 1, 2, 3, 4]

    # Non-contiguous index lists still follow the host copy path and share
    # the same backing storage.
    other = common._materialize_logits_indices(runner, [7, 9])
    assert other.tolist() == [7, 9]
    assert out.data_ptr() == other.data_ptr()


def test_is_contiguous_range() -> None:
    assert common._is_contiguous_range([]) is True
    assert common._is_contiguous_range([0]) is True
    assert common._is_contiguous_range([0, 1, 2, 3]) is True
    assert common._is_contiguous_range([0, 1, 3]) is False
    assert common._is_contiguous_range([1, 2, 3]) is False
    assert common._is_contiguous_range([0, 2, 3]) is False


def test_logprob_remux_preserves_flattening_and_placeholder_rows() -> None:
    st_ids = torch.tensor([[10], [20], [30]])
    logprobs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    token_ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
    ranks = torch.tensor([7, 8, 9])

    final_st, final_lps, final_ids, final_ranks = common._remux_logprobs_tensors(
        [
            ("ordinary", 0, 1, []),
            ("beam", 1, 2, [0, 1]),
            ("prefill", 3, 0, []),
        ],
        max_chained_dim=4,
        target_K=2,
        orig_lps=logprobs,
        orig_lp_ids=token_ids,
        orig_ranks=ranks,
        st_ids=st_ids,
    )

    assert final_st.tolist() == [[10], [20], [-1]]
    assert final_lps[:2].tolist() == [
        [1.0, 2.0, float("-inf"), float("-inf")],
        [3.0, 4.0, 5.0, 6.0],
    ]
    assert torch.isneginf(final_lps[2]).all()
    assert final_ids.tolist() == [[1, 2, 0, 0], [3, 4, 5, 6], [0, 0, 0, 0]]
    assert final_ranks.tolist() == [[7, 0, 0, 0], [8, 9, 0, 0], [0, 0, 0, 0]]


def test_beam_collapse_accumulates_partial_rows_and_clears_cache() -> None:
    runner = SimpleNamespace(_beam_result_host_ready=True)
    beam_data = {"beam": {"is_beam_decode": True, "beam_width": 2}}

    def sampler(values: list[float], token_ids: list[int], rank: int) -> SimpleNamespace:
        return SimpleNamespace(
            sampled_token_ids=torch.tensor([[token_ids[0]]]),
            logprobs_tensors=SimpleNamespace(
                logprobs=torch.tensor([values]),
                logprob_token_ids=torch.tensor([token_ids]),
                selected_token_ranks=torch.tensor([rank]),
            ),
        )

    first = sampler([0.1, 0.2], [1, 2], 3)
    common._beam_collapse_bookkeeping(
        runner,
        object(),
        [("beam", 0, 1, [0])],
        beam_data,
        first,
    )
    assert runner._beam_logprobs_cache["beam"]["completed_beams"] == {0}
    assert first.logprobs_tensors.logprobs.shape == (1, 4)

    second = sampler([0.3, 0.4], [3, 4], 5)
    collapsed = common._beam_collapse_bookkeeping(
        runner,
        object(),
        [("beam", 0, 1, [1])],
        beam_data,
        second,
    )

    assert collapsed == (["beam"], {"beam": 0}, 1)
    assert torch.allclose(
        second.logprobs_tensors.logprobs,
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
    )
    assert second.logprobs_tensors.logprob_token_ids.tolist() == [[1, 2, 3, 4]]
    assert "beam" not in runner._beam_logprobs_cache


def test_worker_decision_reduces_sampler_payload_before_host_copy() -> None:
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    width = 4
    selected_ids = torch.tensor([[41, 42, 43, 44]], dtype=torch.int32)
    selected_scores = torch.tensor([[-0.1, -0.2, -0.3, -0.4]], dtype=torch.float32)
    runner = SimpleNamespace(
        _beam_decision_tensors={
            "out_token_ids": selected_ids,
            "out_token_index": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            "out_log_probs": selected_scores,
        }
    )
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.arange(width, dtype=torch.int32).reshape(width, 1),
        logprobs_tensors=LogprobsTensors(
            logprob_token_ids=torch.arange(width * width, dtype=torch.int32).reshape(width, width),
            logprobs=torch.zeros((width, width), dtype=torch.float32),
            selected_token_ranks=torch.arange(width, dtype=torch.int64),
        ),
    )

    assert common._reduce_sampler_output_for_worker_decision(runner, sampler_output)
    assert sampler_output.sampled_token_ids.tolist() == [[41]]
    assert sampler_output.logprobs_tensors.logprob_token_ids is selected_ids
    assert sampler_output.logprobs_tensors.logprobs is selected_scores
    assert sampler_output.logprobs_tensors.selected_token_ranks.tolist() == [0]
    assert sampler_output._gr_beam_result_transport is True

    collapsed = common._beam_collapse_bookkeeping(
        runner,
        object(),
        [("beam", 0, width, list(range(width)))],
        {"beam": {"is_beam_decode": True, "beam_width": width}},
        sampler_output,
    )
    assert collapsed == (["beam"], {"beam": 0}, 1)
    # The later collapse must preserve O(W), never recreate W-by-W.
    assert sampler_output.logprobs_tensors.logprobs.shape == (1, width)


def test_seed_worker_decision_uses_frontend_parent_state(monkeypatch) -> None:
    captured = {}

    def fake_init(session, sequences, scores):
        captured["session"] = session
        captured["sequences"] = sequences
        captured["scores"] = scores

    monkeypatch.setattr(
        "vllm_gr.v1.beam.beam_decision.init_beam_decision_from_parents",
        fake_init,
    )
    session = object()
    common._seed_beam_decision_from_metadata(
        session,
        {
            "beam_suffix": [[11], [22]],
            "parent_cumulative_logprobs": [-0.5, -1.75],
        },
    )

    assert captured == {
        "session": session,
        "sequences": [[11], [22]],
        "scores": [-0.5, -1.75],
    }


def test_seed_worker_decision_rejects_scores_without_suffixes() -> None:
    with pytest.raises(RuntimeError, match="parent scores without parent beam suffixes"):
        common._seed_beam_decision_from_metadata(object(), {"parent_cumulative_logprobs": [-0.5]})


def test_execute_wrapper_is_idempotent_and_restores_outer_context() -> None:
    class ExpectedError(RuntimeError):
        pass

    scheduler_output = SimpleNamespace(beam_data={"beam": {"is_beam_decode": True}})
    runner = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))
    should_raise = False

    def execute_model(self: Any, output: Any) -> str:
        assert self._beam_result_host_ready is False
        assert BEAM_DATA_VAR.get() is self._current_beam_info
        assert self._current_beam_info == {
            "beam_data": output.beam_data,
            "req_ids": [],
        }
        # The Constraint Tables _sample hook reads the scheduler output to find
        # _gr_request_row_configs; it must be published for the whole run.
        assert self._current_scheduler_output is output
        if should_raise:
            raise ExpectedError
        return "result"

    wrapped = common.make_patched_execute_model(execute_model)
    assert common.make_patched_execute_model(wrapped) is wrapped

    outer = {"level": "outer"}
    token = BEAM_DATA_VAR.set(outer)
    try:
        assert wrapped(runner, scheduler_output) == "result"
        assert BEAM_DATA_VAR.get() is outer

        should_raise = True
        with pytest.raises(ExpectedError):
            wrapped(runner, scheduler_output)
        assert BEAM_DATA_VAR.get() is outer
        should_raise = False
    finally:
        BEAM_DATA_VAR.reset(token)

    # The beam context deliberately OUTLIVES execute_model: with async
    # scheduling, sample_tokens (constraint hook + worker beam decision)
    # runs AFTER execute_model returns and still needs the published
    # context — including on the exception path (only the ContextVar is
    # reset in the finally block).
    assert runner._current_beam_data is scheduler_output.beam_data
    assert runner._current_scheduler_output is scheduler_output
    assert runner._current_beam_info == {
        "beam_data": scheduler_output.beam_data,
        "req_ids": [],
    }

    # A subsequent non-beam step overwrites the stale context at entry.
    plain_output = SimpleNamespace(beam_data={})
    assert wrapped(runner, plain_output) == "result"
    assert runner._current_beam_data == {}
    assert runner._current_scheduler_output is plain_output
    assert runner._current_beam_info == {"beam_data": {}, "req_ids": []}


@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_beam_context_pool_follows_model_activation_dtype(
    model_dtype: torch.dtype,
) -> None:
    parallel_config = object()
    model_config = SimpleNamespace(
        dtype=model_dtype,
        get_num_kv_heads=lambda config: 2,
        get_head_size=lambda: 8,
        get_num_layers=lambda config: 2,
    )
    runner = SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        device=torch.device("cpu"),
        # A quantized paged prefix cache must not determine the suffix pool dtype.
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        # Provide a GRConfig with beam_max_width set: beam decode requests
        # without a configured pool raise in _get_or_create_beam_session.
        _vllm_gr_config=SimpleNamespace(
            beam_max_width=128, beam_max_decode_steps=3, enable_short_suffix_attention=False
        ),
    )
    beam_data = {
        "beam": {
            "is_beam_decode": True,
            "decode_step": 0,
            "beam_decode_steps": 1,
            "beam_width": 2,
            "prefix_len": 4,
            "max_decode_steps": 3,
        }
    }

    session = common._get_or_create_beam_session(
        runner,
        beam_data,
        SimpleNamespace(beam_finished_session_ids=[]),
    )

    assert session is not None
    assert runner._beam_context.dtype == model_dtype
    assert runner._beam_context.suffix_k_buf.dtype == model_dtype
    assert runner._beam_context.suffix_v_buf.dtype == model_dtype
    assert all(pool.dtype == model_dtype for pool in runner._beam_context.pools)


def test_beam_session_raises_when_beam_max_width_unconfigured() -> None:
    """Beam decode without a configured pool must fail loudly.

    The scheduler freezes the paged cache at the prefix for beam decode
    requests, so running them without the beam KV pool would silently
    compute wrong attention; a RuntimeError pointing at --beam-max-width
    is the only safe outcome.
    """
    runner = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(beam_max_width=None, beam_max_decode_steps=3),
    )
    beam_data = {
        "beam": {
            "is_beam_decode": True,
            "decode_step": 0,
            "beam_decode_steps": 1,
            "beam_width": 2,
            "prefix_len": 4,
            "max_decode_steps": 3,
        }
    }

    with pytest.raises(RuntimeError, match="beam_max_width"):
        common._get_or_create_beam_session(
            runner,
            beam_data,
            SimpleNamespace(beam_finished_session_ids=[]),
        )


def test_beam_session_rejects_concurrent_incremental_requests() -> None:
    """Two concurrent incremental beam-decode requests must be rejected loudly.

    The shared beam context is created with max_batch=1 (pool slots, suffix
    buffers and captured FULL graphs all bake a single session).  Without
    this guard the second request's registry allocate would fail mid-batch
    with the generic "No free beam session slots" error — and the stale-
    session retry cannot help because BOTH ids are active.  An explicit
    concurrency error is the only safe outcome.
    """
    runner = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(
            beam_max_width=128, beam_max_decode_steps=3, enable_short_suffix_attention=False
        ),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            get_num_kv_heads=lambda config: 2,
            get_head_size=lambda: 8,
            get_num_layers=lambda config: 1,
        ),
        parallel_config=object(),
        device=torch.device("cpu"),
    )
    beam_data = {
        "beam-a": {
            "is_beam_decode": True,
            "decode_step": 0,
            "beam_decode_steps": 1,
            "beam_width": 2,
            "prefix_len": 4,
            "max_decode_steps": 3,
        },
        "beam-b": {
            "is_beam_decode": True,
            "decode_step": 0,
            "beam_decode_steps": 1,
            "beam_width": 2,
            "prefix_len": 4,
            "max_decode_steps": 3,
        },
    }

    with pytest.raises(RuntimeError, match="concurrent incremental beam decode"):
        common._get_or_create_beam_session(
            runner,
            beam_data,
            SimpleNamespace(beam_finished_session_ids=[]),
        )


# ---------------------------------------------------------------------------
# M2: inlined sample_tokens (merge the second collective_rpc into
# execute_model; upstream core.py None short-circuit does the skipping)
# ---------------------------------------------------------------------------


def _m2_runner(
    *,
    beam_engine_driven: bool = True,
    beam_worker_decision: bool = True,
    use_async_scheduling: bool = False,
    pipeline_parallel_size: int = 1,
    sample_tokens_return: Any = "mro",
    sample_tokens_side_effect: Any = None,
    state: Any = "state",
) -> SimpleNamespace:
    """Runner stub exercising the M2 gate + inline path in isolation."""
    calls: list[Any] = []

    def sample_tokens(grammar_output: Any) -> Any:
        calls.append(grammar_output)
        if sample_tokens_side_effect is not None:
            raise sample_tokens_side_effect
        return sample_tokens_return

    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config={}),
        _vllm_gr_config=SimpleNamespace(
            beam_engine_driven=beam_engine_driven,
            beam_worker_decision=beam_worker_decision,
        ),
        use_async_scheduling=use_async_scheduling,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pipeline_parallel_size),
        sample_tokens=sample_tokens,
        execute_model_state=state,
        _sample_tokens_calls=calls,
    )
    return runner


def test_should_inline_sample_tokens_gate() -> None:
    """Gate opens only with both opt-ins, sync scheduling, and PP=1."""
    assert common._should_inline_sample_tokens(_m2_runner()) is True

    assert common._should_inline_sample_tokens(_m2_runner(beam_engine_driven=False)) is False
    assert common._should_inline_sample_tokens(_m2_runner(beam_worker_decision=False)) is False
    assert common._should_inline_sample_tokens(_m2_runner(use_async_scheduling=True)) is False
    assert common._should_inline_sample_tokens(_m2_runner(pipeline_parallel_size=2)) is False
    # Missing attributes read as their safe defaults (sync / PP=1).
    minimal = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(beam_engine_driven=True, beam_worker_decision=True)
    )
    assert common._should_inline_sample_tokens(minimal) is True


def test_should_inline_sample_tokens_rejects_mixed_or_guided_rows() -> None:
    """M2 is per-request: ordinary, mixed, and structured rows stay two-RPC."""
    runner = _m2_runner()
    beam_sched = SimpleNamespace(
        beam_data={"b": {"is_beam_decode": True}},
        num_scheduled_tokens={"b": 1},
        pending_structured_output_tokens=False,
        scheduled_new_reqs=[],
    )
    assert common._should_inline_sample_tokens(runner, beam_sched) is True

    mixed = SimpleNamespace(
        beam_data={"b": {"is_beam_decode": True}},
        num_scheduled_tokens={"b": 1, "plain": 4},
        pending_structured_output_tokens=False,
        scheduled_new_reqs=[],
    )
    assert common._should_inline_sample_tokens(runner, mixed) is False

    guided = SimpleNamespace(
        beam_data={"b": {"is_beam_decode": True}},
        num_scheduled_tokens={"b": 1},
        pending_structured_output_tokens=True,
        scheduled_new_reqs=[],
    )
    assert common._should_inline_sample_tokens(runner, guided) is False

    structured_req = SimpleNamespace(
        sampling_params=SimpleNamespace(structured_outputs=object(), guided_decoding=None)
    )
    structured_batch = SimpleNamespace(
        beam_data={"b": {"is_beam_decode": True}},
        num_scheduled_tokens={"b": 1},
        pending_structured_output_tokens=False,
        scheduled_new_reqs=[structured_req],
    )
    assert common._should_inline_sample_tokens(runner, structured_batch) is False

    ordinary = SimpleNamespace(
        beam_data={},
        num_scheduled_tokens={"plain": 8},
        pending_structured_output_tokens=False,
        scheduled_new_reqs=[],
    )
    assert common._should_inline_sample_tokens(runner, ordinary) is False


def test_inline_sample_tokens_runs_and_clears_state() -> None:
    """grammar_output=None matches the engine's own second-RPC call."""
    runner = _m2_runner()

    out = common._inline_sample_tokens(runner)

    assert out == "mro"
    assert runner._sample_tokens_calls == [None]
    assert runner.execute_model_state is None


def test_inline_sample_tokens_clears_state_on_failure() -> None:
    runner = _m2_runner(sample_tokens_side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        common._inline_sample_tokens(runner)

    assert runner.execute_model_state is None


def _orig_execute_model_none(self: Any, scheduler_output: Any) -> None:
    """Upstream async-scheduling-style execute_model: stashes state, returns None."""
    self.execute_model_state = "state"
    return None


def test_execute_wrapper_inlines_when_gated() -> None:
    """M2 on: execute_model returns the ModelRunnerOutput (None short-circuit
    upstream skips the second collective_rpc); beam context stays published
    for the in-frame sampling."""
    wrapped = common.make_patched_execute_model(_orig_execute_model_none)
    runner = _m2_runner()
    scheduler_output = SimpleNamespace(beam_data={"b": {"is_beam_decode": True}})

    out = wrapped(runner, scheduler_output)

    assert out == "mro"
    assert runner._sample_tokens_calls == [None]
    assert runner.execute_model_state is None
    assert runner._current_beam_info is not None
    # The in-frame sampling consumed the context (decision attach etc.); it is
    # NOT reset early — the next execute_model overwrites it at entry.
    assert runner._current_scheduler_output is scheduler_output


def test_execute_wrapper_falls_back_when_inline_fails() -> None:
    """Inline failure returns None: the engine's fallback second RPC runs and
    the two-RPC path is preserved verbatim."""
    wrapped = common.make_patched_execute_model(_orig_execute_model_none)
    runner = _m2_runner(sample_tokens_side_effect=RuntimeError("boom"))
    scheduler_output = SimpleNamespace(beam_data={"b": {"is_beam_decode": True}})

    out = wrapped(runner, scheduler_output)

    assert out is None
    assert runner._sample_tokens_calls == [None]
    assert runner.execute_model_state is None


def test_execute_wrapper_does_not_inline_when_gate_closed() -> None:
    """Gate closed (async scheduling): execute_model returns None and never
    calls sample_tokens — the engine issues the second RPC itself."""
    wrapped = common.make_patched_execute_model(_orig_execute_model_none)
    runner = _m2_runner(use_async_scheduling=True)
    scheduler_output = SimpleNamespace(beam_data={"b": {"is_beam_decode": True}})

    out = wrapped(runner, scheduler_output)

    assert out is None
    assert runner._sample_tokens_calls == []
    # State survives for the engine-initiated sample_tokens RPC.
    assert runner.execute_model_state == "state"


def test_execute_wrapper_prelaunches_before_async_sample_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CUDA helper runs after ExecuteModelState publication, while the
    engine-owned second sample_tokens RPC remains untouched."""

    from vllm_gr.v1.worker import constraint_hook

    events: list[str] = []
    scheduler_output = SimpleNamespace(beam_data={})

    def execute(self: Any, received: Any) -> None:
        assert received is scheduler_output
        self.execute_model_state = SimpleNamespace(
            scheduler_output=received,
            logits="logits",
        )
        # The CUDA prepare path ran inside execute_model and published this
        # step's descriptor (the wrapper's import/launch is gated on it).
        self._gr_prepared_cuda_prefill_constraint = object()
        events.append("execute")
        return None

    def launch(self: Any) -> bool:
        assert self.execute_model_state.logits == "logits"
        events.append("prelaunch")
        return True

    monkeypatch.setattr(constraint_hook, "launch_prepared_cuda_prefill_constraint", launch)
    runner = _m2_runner(use_async_scheduling=True, state=None)

    assert common.make_patched_execute_model(execute)(runner, scheduler_output) is None
    assert events == ["execute", "prelaunch"]
    assert runner._sample_tokens_calls == []
    assert runner.execute_model_state.logits == "logits"


def test_execute_wrapper_does_not_inline_non_none_output() -> None:
    """A runner that already returned a value (e.g. pooling) is untouched."""
    wrapped = common.make_patched_execute_model(lambda self, output: "direct")
    runner = _m2_runner()
    scheduler_output = SimpleNamespace(beam_data={"b": {"is_beam_decode": True}})

    assert wrapped(runner, scheduler_output) == "direct"
    assert runner._sample_tokens_calls == []


def test_execute_wrapper_does_not_inline_mixed_beam_plus_structured() -> None:
    wrapped = common.make_patched_execute_model(_orig_execute_model_none)
    runner = _m2_runner()
    scheduler_output = SimpleNamespace(
        beam_data={"b": {"is_beam_decode": True}},
        num_scheduled_tokens={"b": 1, "guided": 2},
        pending_structured_output_tokens=True,
        scheduled_new_reqs=[],
    )

    out = wrapped(runner, scheduler_output)

    assert out is None
    assert runner._sample_tokens_calls == []
    assert runner.execute_model_state == "state"


# ---------------------------------------------------------------------------
# Warmup GC freeze: quiescent boundary + RPC completion barrier
# ---------------------------------------------------------------------------


def _reset_warmup_gc_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the process-level freeze state per test."""
    monkeypatch.setattr(common, "_GC_WARMUP_FREEZE_DONE", False)
    monkeypatch.setattr(common, "_GC_WARMUP_FREEZE_COMPLETE", False)
    monkeypatch.setattr(common, "_WORKER_EXECUTE_DEPTH", 0)
    monkeypatch.setattr(common, "_GC_FREEZE_IN_PROGRESS", False)
    # Keep the quiescent fallback wait short in tests.
    monkeypatch.setattr(common, "_QUIESCENT_WAIT_TIMEOUT_S", 2.0)


def _freeze_daemon_alive() -> bool:
    return any(t.name == "vllm-gr-gc-freeze" for t in threading.enumerate())


def _wait_freeze_daemon_done(timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _freeze_daemon_alive():
            return True
        time.sleep(0.02)
    return False


def test_warmup_rpc_does_not_acknowledge_scheduled_pending_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review regression: frozen=True must imply a *completed* freeze.

    The automatic daemon claims the one-shot before its collection runs; an
    RPC racing that collector must not return frozen=True while the heap is
    still unfrozen (probe observed thresholds still [700, 10, 10]).
    """
    _reset_warmup_gc_state(monkeypatch)
    calls: list[bool] = []

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        calls.append(raise_thresholds)
        return [700, 100_000, 100_000]

    monkeypatch.setattr(common, "_freeze_serving_gc", fake_freeze)
    # The automatic daemon claimed the one-shot; collection still pending.
    monkeypatch.setattr(common, "_GC_WARMUP_FREEZE_DONE", True)

    result = common.freeze_gc_after_warmup(SimpleNamespace(model_runner=None))

    assert result["frozen"] is True
    assert "deduped" not in result
    # The RPC performed the freeze itself, so completion is guaranteed
    # before frozen=True was returned.
    assert calls == [True]
    assert common._GC_WARMUP_FREEZE_COMPLETE is True


def test_warmup_rpc_dedupes_only_completed_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_warmup_gc_state(monkeypatch)
    calls: list[bool] = []

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        calls.append(raise_thresholds)
        return [700, 100_000, 100_000]

    monkeypatch.setattr(common, "_freeze_serving_gc", fake_freeze)
    monkeypatch.setattr(common, "_GC_WARMUP_FREEZE_COMPLETE", True)

    result = common.freeze_gc_after_warmup(SimpleNamespace(model_runner=None))

    assert result["frozen"] is True
    assert result["deduped"] is True
    assert result["thresholds"] == list(gc.get_threshold())
    assert calls == []


def test_warmup_rpc_failure_rolls_back_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed acknowledged collector must not strand the one-shot."""
    _reset_warmup_gc_state(monkeypatch)

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        raise RuntimeError("collect failed")

    monkeypatch.setattr(common, "_freeze_serving_gc", fake_freeze)

    result = common.freeze_gc_after_warmup(SimpleNamespace(model_runner=None))

    assert result["frozen"] is False
    assert "error" in result
    assert common._GC_WARMUP_FREEZE_DONE is False
    assert common._GC_WARMUP_FREEZE_COMPLETE is False


def test_warmup_daemon_waits_for_quiescent_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review regression: no collection while execute_model is in flight.

    The teardown signal is consumed at the START of the next execute_model;
    the freeze daemon must wait for that step to finish before collecting
    (gc.collect holds the GIL and would stall the request).
    """
    _reset_warmup_gc_state(monkeypatch)
    started = threading.Event()

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        started.set()
        return [700, 100_000, 100_000]

    monkeypatch.setattr(common, "_freeze_serving_gc", fake_freeze)

    common._enter_worker_execute()
    try:
        assert common.schedule_warmup_gc_freeze() is True
        assert common.schedule_warmup_gc_freeze() is False
        assert not started.wait(0.2), "freeze ran while execute_model was in flight"
    finally:
        common._exit_worker_execute()

    assert started.wait(5.0), "freeze did not run at the quiescent boundary"
    assert common._GC_WARMUP_FREEZE_COMPLETE is True


def test_warmup_daemon_blocks_execute_reentry_until_freeze_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-depth observation and freeze form one admission boundary."""
    _reset_warmup_gc_state(monkeypatch)
    freeze_started = threading.Event()
    release_freeze = threading.Event()
    execute_entered = threading.Event()

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        freeze_started.set()
        assert release_freeze.wait(5.0)
        return [700, 100_000, 100_000]

    def enter_execute() -> None:
        common._enter_worker_execute()
        try:
            execute_entered.set()
        finally:
            common._exit_worker_execute()

    monkeypatch.setattr(common, "_freeze_serving_gc", fake_freeze)
    assert common.schedule_warmup_gc_freeze() is True
    assert freeze_started.wait(5.0)

    thread = threading.Thread(target=enter_execute)
    thread.start()
    assert not execute_entered.wait(0.2), "execute entered during the claimed GC boundary"

    release_freeze.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert execute_entered.is_set()
    assert _wait_freeze_daemon_done()
    assert common._GC_FREEZE_IN_PROGRESS is False


def test_warmup_daemon_skips_when_rpc_already_froze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduled daemon must not re-collect after an RPC freeze."""
    _reset_warmup_gc_state(monkeypatch)
    calls: list[bool] = []

    def fake_freeze(*, raise_thresholds: bool) -> list[int]:
        calls.append(raise_thresholds)
        return [700, 100_000, 100_000]

    monkeypatch.setattr(common, "_freeze_serving_gc", fake_freeze)
    # Park the daemon behind the quiescence wait (a step is in flight), then
    # let the explicit RPC barrier complete a freeze while it waits.
    common._enter_worker_execute()
    try:
        assert common.schedule_warmup_gc_freeze() is True
        monkeypatch.setattr(common, "_GC_WARMUP_FREEZE_COMPLETE", True)
    finally:
        common._exit_worker_execute()

    assert _wait_freeze_daemon_done()
    assert calls == []


def test_execute_wrapper_tracks_inflight_depth_for_gc_freeze() -> None:
    depth_seen: dict[str, int] = {}

    def execute_model(self: Any, scheduler_output: Any) -> str:
        depth_seen["during"] = common._WORKER_EXECUTE_DEPTH
        return "result"

    wrapped = common.make_patched_execute_model(execute_model)
    runner = _m2_runner()
    scheduler_output = SimpleNamespace(beam_data={})

    assert common._WORKER_EXECUTE_DEPTH == 0
    assert wrapped(runner, scheduler_output) == "result"
    assert depth_seen["during"] == 1
    assert common._WORKER_EXECUTE_DEPTH == 0


def test_execute_wrapper_releases_depth_on_exception() -> None:
    class ExpectedError(Exception):
        pass

    def execute_model(self: Any, scheduler_output: Any) -> None:
        assert common._WORKER_EXECUTE_DEPTH == 1
        raise ExpectedError

    wrapped = common.make_patched_execute_model(execute_model)
    runner = _m2_runner()
    scheduler_output = SimpleNamespace(beam_data={})

    with pytest.raises(ExpectedError):
        wrapped(runner, scheduler_output)
    assert common._WORKER_EXECUTE_DEPTH == 0


# ---------------------------------------------------------------------------
# V2 ordinary frames remain native; no global dispatch context is needed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raises", [False, True])  # type: ignore[misc]
def test_v2_plain_frame_native_execution_and_gc_cleanup(raises: bool) -> None:
    runner = SimpleNamespace()
    output = SimpleNamespace(beam_data={})

    def native(self: Any, scheduled: Any) -> str:
        assert self is runner and scheduled is output
        assert common._WORKER_EXECUTE_DEPTH == 1
        if raises:
            raise RuntimeError("native failed")
        return "result"

    wrapped = common.make_patched_execute_model_v2(native)
    if raises:
        with pytest.raises(RuntimeError, match="native failed"):
            wrapped(runner, output)
    else:
        assert wrapped(runner, output) == "result"
    assert common._WORKER_EXECUTE_DEPTH == 0


def test_slim_v2_wrapper_fail_fast_on_legacy_beam_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refactor §4.3: the slim V2 execute_model wrapper must raise loudly on a
    legacy beam_data frame instead of silently running the (deleted) LEGACY
    stack under a runner whose patch layer no longer supports it."""
    scheduler_output = SimpleNamespace(beam_data={"beam": {"is_beam_decode": True}})
    runner = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))

    def execute_model(self: Any, output: Any) -> str:
        raise AssertionError("native execute_model must not run for a beam_data frame")

    wrapped = common.make_patched_execute_model_v2(execute_model)
    with pytest.raises(ValueError, match="legacy beam_data frames are not supported"):
        wrapped(runner, scheduler_output)


def test_slim_v2_wrapper_is_idempotent() -> None:
    def execute_model(self: Any, output: Any) -> str:
        return "result"

    wrapped = common.make_patched_execute_model_v2(execute_model)
    assert common.make_patched_execute_model_v2(wrapped) is wrapped
