# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Unit tests for BeamPrefillGraphRunner."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.helpers.gr_config import engine_config
from vllm_gr.v1.worker import prefill_graph_common
from vllm_gr.v1.worker.prefill_graph_runner import BeamPrefillGraphRunner


@pytest.fixture(autouse=True)  # type: ignore
def _force_non_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_gr.v1.worker.prefill_graph_runner as prefill_graph_runner_mod

    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: False)

    class MockCUDAProvider:
        device_type = "cuda"

    monkeypatch.setattr(prefill_graph_runner_mod, "current_platform", MockCUDAProvider())


class MockModelRunner:
    def __init__(self, **options: Any) -> None:
        self.device = torch.device("cpu")
        self.vllm_config: Any = engine_config(**options)


def test_match_bucket_standard() -> None:
    runner = MockModelRunner()
    graph_runner = BeamPrefillGraphRunner(runner, buckets=[512, 1024, 2048, 4096])

    # Requests use the smallest captured bucket that can hold them.
    assert graph_runner.match_bucket(400) == 512
    assert graph_runner.match_bucket(750) == 1024
    assert graph_runner.match_bucket(2000) == 2048
    assert graph_runner.match_bucket(4097) is None


def test_match_bucket_dense_default_range() -> None:
    graph_runner = BeamPrefillGraphRunner(MockModelRunner())

    # Without the bound override the grid is the plain step range: every total
    # maps to the same bucket it did before the small buckets existed, so a
    # 128-token or 17-token cache hit still replays in the graph it used to.
    assert graph_runner.match_bucket(1) == 128
    assert graph_runner.match_bucket(16) == 128
    assert graph_runner.match_bucket(17) == 128
    assert graph_runner.match_bucket(128) == 128
    assert graph_runner.match_bucket(129) == 256
    assert graph_runner.match_bucket(8192) == 8192
    assert graph_runner.match_bucket(8193) is None


def test_match_bucket_small_buckets_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    graph_runner = BeamPrefillGraphRunner(MockModelRunner(prefill_kv_bound=1024))

    # With the override set, an exact small bucket serves the cache-hit tails:
    # tail 1 (a separate gemv shape) gets bucket 1, tails 2..16 get bucket 16,
    # and everything above keeps the step range.
    assert graph_runner.match_bucket(1) == 1
    assert graph_runner.match_bucket(16) == 16
    assert graph_runner.match_bucket(17) == 128
    assert graph_runner.match_bucket(128) == 128
    assert graph_runner.match_bucket(129) == 256


def test_small_buckets_never_shrink_replay_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under one environment the opt-in grid admits at least the KV extent the
    step grid admits for the same scheduled total.

    A cache hit lands in the smallest bucket by scheduled tail while its KV
    length is the whole context, so moving such a step into a *smaller* bucket
    is only safe if the bound baked for that bucket is at least as wide as the
    bound the step grid baked for the same request. That is the regression the
    small buckets introduced when they were added to the default grid.
    """
    from vllm_gr.v1.worker.prefill_graph_common import (
        SMALL_PREFILL_BUCKETS,
    )

    # The grid the env leaves untouched, captured with the env unset.
    default_buckets = list(BeamPrefillGraphRunner(MockModelRunner()).buckets)

    # The opt-in grid under the bound the feature requires, compared against
    # the default grid under the *same* environment.
    opt_in_runner = BeamPrefillGraphRunner(MockModelRunner(prefill_kv_bound=1024))
    assert set(opt_in_runner.buckets) - set(default_buckets) == set(SMALL_PREFILL_BUCKETS)
    assert set(default_buckets) - set(opt_in_runner.buckets) == set()

    def coverage(bucket: int | None) -> int:
        """KV extent a step landing in ``bucket`` is admitted for."""
        if bucket is None:
            return 0
        return int(opt_in_runner._kv_capture_bound(bucket))

    for total in range(1, max(default_buckets) + 1):
        step_bucket = next((b for b in default_buckets if b >= total), None)
        assert coverage(opt_in_runner.match_bucket(total)) >= coverage(step_bucket)


def test_get_prefill_logit_indices() -> None:
    runner = MockModelRunner()
    graph_runner = BeamPrefillGraphRunner(runner)
    indices = graph_runner.get_prefill_logit_indices([200, 300])
    # 200 tokens -> end at index 199; + 300 tokens -> end at index 499
    assert indices == [199, 499]


def test_is_prefill_phase_batch() -> None:
    from types import SimpleNamespace

    import numpy as np

    from vllm_gr.v1.worker.prefill_graph_runner import is_prefill_phase_batch

    def make_batch(
        computed: list[int], prompt: list[int], **extra: Any
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        req_ids = [f"r{i}" for i in range(len(computed))]
        input_batch = SimpleNamespace(
            num_computed_tokens_cpu=np.array(computed),
            num_prompt_tokens=np.array(prompt),
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        )
        runner = SimpleNamespace(input_batch=input_batch)
        fields = {
            "beam_data": {},
            "scheduled_spec_decode_tokens": {},
            "scheduled_encoder_inputs": {},
            "num_scheduled_tokens": {req_id: 1 for req_id in req_ids},
        }
        fields.update(extra)
        return runner, SimpleNamespace(**fields)

    # First chunks (num_computed == 0 < num_prompt) -> True.
    runner, output = make_batch([0, 0], [128, 64])
    assert is_prefill_phase_batch(runner, output) is True

    # Cached / chunked prefill continuations -> True (cached prefill supported).
    runner, output = make_batch([30, 32], [128, 64])
    assert is_prefill_phase_batch(runner, output) is True

    # Ordinary decode mixed in (num_computed == num_prompt) -> False.
    runner, output = make_batch([128, 32], [128, 64])
    assert is_prefill_phase_batch(runner, output) is False

    # Beam decode -> False.
    runner, output = make_batch([0, 0], [128, 64], beam_data={"r1": {"is_beam_decode": True}})
    assert is_prefill_phase_batch(runner, output) is False

    # Spec decode / encoder batches -> False.
    runner, output = make_batch([0, 0], [128, 64], scheduled_spec_decode_tokens={"r1": [7]})
    assert is_prefill_phase_batch(runner, output) is False
    runner, output = make_batch([0, 0], [128, 64], scheduled_encoder_inputs={"r1": [0]})
    assert is_prefill_phase_batch(runner, output) is False

    # Empty batch -> False.
    runner, output = make_batch([], [])
    assert is_prefill_phase_batch(runner, output) is False

    # Non-dense row mapping: scheduled requests live on scattered rows, and a
    # stale decode row sits in position 0. Positional indexing would reject
    # this batch; the row mapping admits it (all scheduled rows are prefill).
    scrambled_input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([10, 100, 5]),
        num_prompt_tokens=np.array([20, 100, 10]),
        req_id_to_index={"rA": 2, "rB": 0},
    )
    scrambled_runner = SimpleNamespace(input_batch=scrambled_input_batch)
    scrambled_output = SimpleNamespace(
        beam_data={},
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_scheduled_tokens={"rA": 1, "rB": 1},
    )
    assert is_prefill_phase_batch(scrambled_runner, scrambled_output) is True

    # The same layout with a decode request scheduled on a non-positional row
    # is correctly rejected (positional indexing would have missed it).
    decode_scrambled_input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([5, 0, 100]),
        num_prompt_tokens=np.array([10, 20, 100]),
        req_id_to_index={"rA": 2, "rB": 1},
    )
    decode_scrambled_runner = SimpleNamespace(input_batch=decode_scrambled_input_batch)
    assert is_prefill_phase_batch(decode_scrambled_runner, scrambled_output) is False

    # A scheduled request missing from the row mapping fails the check.
    missing_row_input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([0]),
        num_prompt_tokens=np.array([10]),
        req_id_to_index={},
    )
    missing_row_runner = SimpleNamespace(input_batch=missing_row_input_batch)
    assert is_prefill_phase_batch(missing_row_runner, scrambled_output) is False

    # Missing input-batch state -> False.
    output = SimpleNamespace(
        beam_data={},
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_scheduled_tokens={"r1": 1},
    )
    assert is_prefill_phase_batch(SimpleNamespace(input_batch=None), output) is False


def test_is_prefill_graph_enabled() -> None:
    from vllm_gr.v1.worker.prefill_graph_runner import is_prefill_graph_enabled

    assert is_prefill_graph_enabled(MockModelRunner())
    assert not is_prefill_graph_enabled(MockModelRunner(prefill_graph_enabled=False))


def test_npu_gates_allow_prefill_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: True)
    from vllm_gr.v1.worker.prefill_graph_runner import (
        is_prefill_graph_enabled,
        is_prefill_phase_batch,
    )

    assert is_prefill_graph_enabled(MockModelRunner())
    assert not is_prefill_graph_enabled(MockModelRunner(prefill_graph_enabled=False))

    # Prefill-phase batches classify as prefill-phase on NPU.
    import numpy as np

    input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([0]),
        num_prompt_tokens=np.array([128]),
        req_id_to_index={"r1": 0},
    )
    runner = SimpleNamespace(input_batch=input_batch)
    output_std = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="r1")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        beam_data={},
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_scheduled_tokens={"r1": 1},
    )
    assert is_prefill_phase_batch(runner, output_std) is True


def test_multi_req_graph_buffers() -> None:
    runner = MockModelRunner()
    graph_runner = BeamPrefillGraphRunner(runner, buckets=[1024])
    buf = graph_runner.buffers[1024]

    # Initially empty
    assert not graph_runner.has_captured_graph(1024, num_reqs=1)
    assert not graph_runner.has_captured_graph(1024, num_reqs=4)

    # Set mock graphs for 1..8 requests
    for r in range(1, 9):
        buf.captured_graphs[r] = f"mock_graph_{r}"
        buf.output_bufs[r] = torch.zeros(1024)

    for r in range(1, 9):
        assert graph_runner.has_captured_graph(1024, num_reqs=r)
    # > 8 requests not captured
    assert not graph_runner.has_captured_graph(1024, num_reqs=9)


def test_platform_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_gr.v1.worker.prefill_graph_runner as prefill_graph_runner_mod
    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import GPUPrefillGraphRunner
    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner
    from vllm_gr.v1.worker.prefill_graph_runner import (
        BeamPrefillGraphRunner,
        get_active_prefill_graph_runner_class,
    )

    runner = MockModelRunner()

    # 1. Test NPU platform selection
    class MockNPUPlatform:
        device_type = "npu"

    monkeypatch.setattr(prefill_graph_runner_mod, "current_platform", MockNPUPlatform)
    assert get_active_prefill_graph_runner_class() is NPUPrefillGraphRunner
    npu_runner = BeamPrefillGraphRunner(runner, buckets=[512])
    assert isinstance(npu_runner, NPUPrefillGraphRunner)

    # 2. Test CUDA platform selection
    class MockCUDAProvider:
        device_type = "cuda"

    monkeypatch.setattr(prefill_graph_runner_mod, "current_platform", MockCUDAProvider)
    assert get_active_prefill_graph_runner_class() is GPUPrefillGraphRunner
    gpu_runner = BeamPrefillGraphRunner(runner, buckets=[512])
    assert isinstance(gpu_runner, GPUPrefillGraphRunner)


class MockNPUAttnBuilder:
    def __init__(self) -> None:
        self.num_heads_q = 8
        self.num_kv_heads = 8
        self.block_size = 16
        self.npu_causal_atten_mask = None


@pytest.mark.parametrize(  # type: ignore[misc]
    ("bucket", "expected_reqs", "expected_lens", "expected_cols"),
    [
        # Small buckets: fewer tokens than the dummy width must not produce
        # zero-length requests, and the block table must still cover seq_len.
        (1, 1, [1], 1),
        (4, 4, [1, 1, 1, 1], 1),
        (16, 8, [2] * 8, 1),
        (128, 8, [16] * 8, 1),
        # Uneven split across a block boundary: the longest request needs an
        # extra block-table column even though base_len fits in one.
        (129, 8, [17] + [16] * 7, 2),
    ],
)
def test_npu_dummy_metadata_per_bucket(
    bucket: int, expected_reqs: int, expected_lens: list[int], expected_cols: int
) -> None:
    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

    npu_runner = NPUPrefillGraphRunner(MockModelRunner(), buckets=[bucket])
    builder = MockNPUAttnBuilder()
    meta = npu_runner._build_dummy_metadata(bucket, builder, npu_runner.buffers[bucket])

    assert meta.num_reqs == expected_reqs
    assert meta.seq_lens_cpu == expected_lens
    assert meta.num_actual_tokens == bucket
    assert meta.query_start_loc_cpu[0] == 0
    assert meta.query_start_loc_cpu[-1] == bucket
    # Structural FIA input contract, checked for every bucket in the grid.
    assert all(length > 0 for length in meta.seq_lens_cpu)
    assert sum(meta.seq_lens_cpu) == bucket
    assert meta.block_table.shape[0] == meta.num_reqs
    assert meta.block_table.shape[1] == expected_cols
    assert meta.block_table.shape[1] * builder.block_size >= max(meta.seq_lens_cpu)


def test_build_update_lists_pad_absorption() -> None:
    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import build_update_lists

    # No padding: lists mirror the real per-request token counts.
    qlen_list, kvlen_list = build_update_lists([32, 32], 64)
    assert qlen_list == [32, 64]
    assert kvlen_list == [32, 32]

    # Padding absorbed by the last request (lists must sum to the bucket).
    qlen_list, kvlen_list = build_update_lists([48, 16], 128)
    assert qlen_list == [48, 128]
    assert kvlen_list == [48, 80]

    qlen_list, kvlen_list = build_update_lists([24, 20, 20], 64)
    assert qlen_list == [24, 44, 64]
    assert kvlen_list == [24, 20, 20]

    qlen_list, kvlen_list = build_update_lists([64], 128)
    assert qlen_list == [128]
    assert kvlen_list == [128]


def test_npu_dummy_metadata_and_bucket_registry() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

    runner = MockModelRunner()
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[128])
    builder = MockNPUAttnBuilder()
    buf = npu_runner.buffers[128]

    meta = npu_runner._build_dummy_metadata(128, builder, buf)
    assert meta.num_reqs == 8
    assert meta.num_actual_tokens == 128
    assert meta.is_prefill_graph_exec is True
    assert meta.query_start_loc_cpu[0] == 0
    assert meta.query_start_loc_cpu[-1] == 128
    assert len(meta.query_start_loc_cpu) == 9
    assert len(meta.seq_lens_cpu) == 8
    assert sum(meta.seq_lens_cpu) == 128
    assert meta.block_table.shape == (8, 1)
    assert meta.num_heads == 8 and meta.num_kv_heads == 8 and meta.block_size == 16

    # No graph captured yet.
    assert npu_runner.has_captured_graph(128, 3) is False

    # One graph per bucket covers every supported num_reqs.
    buf.captured_graphs[1] = object()
    npu_runner.entries[128] = SimpleNamespace()
    assert npu_runner.has_captured_graph(128, 1) is True
    assert npu_runner.has_captured_graph(128, 8) is True
    assert npu_runner.has_captured_graph(128, 9) is False
    assert npu_runner.has_captured_graph(256, 1) is False

    # Replay without a real NPU capture falls back to None (no torch.npu calls).
    assert (
        npu_runner.execute_model_prefill(
            bucket=128,
            input_ids=torch.zeros(64, dtype=torch.int32),
            positions=torch.zeros(64, dtype=torch.int64),
            slot_mapping=torch.zeros(64, dtype=torch.int64),
            num_scheduled_tokens=[32, 32],
        )
        is None
    )


def test_build_update_lists_from_metadata() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import (
        build_update_lists_from_metadata,
    )

    # Fresh prefill: metadata lists equal the token-count version.
    meta = SimpleNamespace(
        query_start_loc_cpu=[0, 30, 64],
        seq_lens_cpu=[30, 64],
        num_reqs=2,
        num_actual_tokens=64,
    )
    qlen_list, kvlen_list = build_update_lists_from_metadata(meta, [30, 34], 64)
    assert qlen_list == [30, 64]
    assert kvlen_list == [30, 64]

    # Padding absorbed by the last request (lists must sum to the bucket).
    qlen_list, kvlen_list = build_update_lists_from_metadata(meta, [30, 34], 128)
    assert qlen_list == [30, 128]
    assert kvlen_list == [30, 128]

    # Cached / chunked prefill: KV lens include already-computed tokens.
    meta = SimpleNamespace(
        query_start_loc_cpu=[0, 30, 64],
        seq_lens_cpu=[1024, 512],
        num_reqs=2,
        num_actual_tokens=64,
    )
    qlen_list, kvlen_list = build_update_lists_from_metadata(meta, [30, 34], 64)
    assert qlen_list == [30, 64]
    assert kvlen_list == [1024, 512]

    # Missing metadata lists fall back to the token-count version.
    meta = SimpleNamespace(
        query_start_loc_cpu=None, seq_lens_cpu=None, num_reqs=2, num_actual_tokens=64
    )
    qlen_list, kvlen_list = build_update_lists_from_metadata(meta, [48, 16], 64)
    assert qlen_list == [48, 64]
    assert kvlen_list == [48, 16]


def test_npu_capture_buffers_int32_slot_mapping() -> None:
    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

    runner = MockModelRunner()
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[128])
    buf = npu_runner.buffers[128]
    npu_runner._prepare_capture_buffers(buf, 128, runner)

    # NPU's ATB ReshapeAndCache op requires int32 slot mappings with valid
    # (non-negative) capture-time values; replay copies the real slots in.
    assert buf.slot_mapping_buf.dtype == torch.int32
    assert buf.slot_mapping_buf[0].item() == 0
    assert buf.slot_mapping_buf[-1].item() == 127
    assert buf.positions_buf.dtype == torch.int64
    assert buf.positions_buf[-1].item() == 127


def test_npu_shared_workspace_binding() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

    runner = MockModelRunner()
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[128, 512])

    # Before any capture establishes the shared workspace, entries keep None
    # and size their own workspace during capture.
    entry = SimpleNamespace(workspace=None)
    npu_runner._bind_entry_workspace(entry)
    assert entry.workspace is None

    # After the largest bucket establishes the shared workspace, smaller
    # buckets reuse it instead of sizing their own.
    npu_runner.shared_workspace = "ws"
    npu_runner._bind_entry_workspace(entry)
    assert entry.workspace == "ws"


def test_bucket_clamp_to_max_num_batched_tokens() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_runner import BeamPrefillGraphRunner

    # Runner exposing the canonical scheduler config field.
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        vllm_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
    )
    graph_runner = BeamPrefillGraphRunner(runner)
    assert max(graph_runner.buckets) <= 512
    assert 512 in graph_runner.buckets
    assert 8192 not in graph_runner.buckets

    # Runner without the config field: buckets unchanged (no-op).
    graph_runner = BeamPrefillGraphRunner(MockModelRunner(), buckets=[256, 1024])
    assert graph_runner.buckets == [256, 1024]


def test_platform_bucket_config() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        DEFAULT_PREFILL_BUCKETS,
        get_default_buckets,
    )

    assert get_default_buckets() == DEFAULT_PREFILL_BUCKETS
    for buckets, bound, expected in (
        ([256, 512, 768, 1024], None, [256, 512, 768, 1024]),
        ([256, 512, 768, 1024], 1024, [1, 16, 256, 512, 768, 1024]),
        ([8], 1024, [1, 8]),
    ):
        config = engine_config(prefill_token_buckets=buckets, prefill_kv_bound=bound)
        assert get_default_buckets(config) == expected


@pytest.mark.parametrize(  # type: ignore[misc]
    "options,match",
    [
        ({"prefill_token_buckets": [0]}, "positive integer"),
        ({"prefill_token_buckets": []}, "nonempty"),
        ({"prefill_token_buckets": [256, 1024, 512]}, "strictly increasing"),
        ({"beam_batch_sizes": [9]}, "1..8"),
    ],
)
def test_prefill_bucket_config_validation(options: Any, match: Any) -> None:
    with pytest.raises(ValueError, match=match):
        engine_config(**options)


def test_opt_in_graded_b4_prefill_graph_config() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        get_default_buckets,
        get_prefill_graph_capture_num_reqs,
    )

    buckets = [256, 512, 1024, 2048, 4096, 8192, 12288, 24576]
    config = engine_config(prefill_token_buckets=buckets, beam_batch_sizes=[4])
    assert get_default_buckets(config) == buckets
    assert get_prefill_graph_capture_num_reqs(config) == (4,)


@pytest.mark.parametrize("sizes", [[1, 2, 4], [3, 5, 6, 7, 8]])  # type: ignore[misc]
def test_prefill_capture_policy_and_explicit_wider_sizes(sizes: Any) -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        get_prefill_graph_capture_num_reqs,
    )

    assert get_prefill_graph_capture_num_reqs() == (1, 2, 4)
    assert get_prefill_graph_capture_num_reqs(engine_config(beam_batch_sizes=sizes)) == tuple(sizes)


def test_custom_b4_bucket_limit_uses_total_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

    runner = MockModelRunner()
    from dataclasses import replace

    from tests.helpers.gr_config import make_gr_config
    from vllm_gr.config.models import DeploymentOptions

    config = make_gr_config(beam_max_width=4)
    config.write_to_vllm_config(runner.vllm_config)
    setattr(runner, "scheduler_config", SimpleNamespace(max_num_batched_tokens=24))
    setattr(runner, "model_config", SimpleNamespace(max_model_len=8))

    assert BeamPrefillGraphRunner(runner, buckets=[4, 8, 12, 24]).buckets == [4, 8, 12, 24]
    replace(
        config, deployment=DeploymentOptions(prefill_token_buckets=(4, 8, 12, 24))
    ).write_to_vllm_config(runner.vllm_config)
    assert BeamPrefillGraphRunner(runner, buckets=[4, 8, 12, 24]).buckets == [4, 8, 12, 24]
    assert NPUPrefillGraphRunner(runner, buckets=[4, 8, 12, 24]).buckets == [4, 8]


@pytest.mark.l1  # type: ignore[misc]
def test_prefill_uses_configured_batch_sizes() -> None:
    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import (
        _capture_request_counts,
    )

    assert _capture_request_counts((4,), 3) == ()
    assert _capture_request_counts((4,), 4) == (4,)


def test_b4_capture_geometry_respects_per_request_model_limit() -> None:
    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import (
        _capture_dummy_positions,
        _capture_eligible_request_counts,
        _capture_query_len,
        _capture_request_counts,
        _capture_request_lengths,
    )
    from vllm_gr.v1.worker.prefill_graph_common import PREFILL_GRAPH_CAPTURE_NUM_REQS

    counts = _capture_request_counts(PREFILL_GRAPH_CAPTURE_NUM_REQS, 4)
    assert counts == (1, 2, 4)
    assert _capture_eligible_request_counts(32768, counts, 8192, True) == (4,)
    assert _capture_eligible_request_counts(8192, counts, 8192, True) == counts
    assert _capture_request_lengths(32768, 4, 8192, True) == [8192] * 4

    lengths = _capture_request_lengths(24, 4, 8, True)
    assert lengths == [6, 6, 6, 6]
    positions = _capture_dummy_positions(lengths, torch.device("cpu"))
    assert positions.tolist() == list(range(6)) * 4
    assert _capture_query_len(24, 8, True) == 8
    with pytest.raises(ValueError, match="per-request model limit"):
        _capture_request_lengths(36, 4, 8, True)


def test_npu_max_kvlen_from_model_config() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

    # Direct model_config attribute.
    runner: Any = SimpleNamespace(
        device=torch.device("cpu"),
        model_config=SimpleNamespace(max_model_len=40960),
    )
    assert NPUPrefillGraphRunner._get_max_kvlen(runner) == 40960

    # Fallback through vllm_config.
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        model_config=None,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192)),
    )
    assert NPUPrefillGraphRunner._get_max_kvlen(runner) == 8192

    # Missing config -> 0 (workspace sized from capture lengths).
    runner = MockModelRunner()
    assert NPUPrefillGraphRunner._get_max_kvlen(runner) == 0


def test_gpu_replay_falls_back_without_slot_mapping() -> None:
    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    runner = MockModelRunner()
    gpu_runner = GPUPrefillGraphRunner(runner, buckets=[512])
    buf = gpu_runner.buffers[512]
    buf.captured_graphs[2] = object()
    buf.output_bufs[2] = torch.zeros(512, 8, 64)

    # A missing slot mapping must never reach the replay (stale slots would
    # corrupt the KV cache); the runner falls back to eager.
    assert (
        gpu_runner.execute_model_prefill(
            bucket=512,
            input_ids=torch.zeros(64, dtype=torch.int32),
            positions=torch.zeros(64, dtype=torch.int64),
            slot_mapping=None,
            num_scheduled_tokens=[30, 34],
        )
        is None
    )
    assert gpu_runner.last_replay_reason == "missing_slot_mapping"


def test_gpu_multi_request_replay_refreshes_request_local_buffers() -> None:
    from types import SimpleNamespace

    import numpy as np

    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    request_ids = ("r0", "r1", "r2", "r3")
    token_counts = {"r0": 3, "r1": 4, "r2": 2, "r3": 5}
    total_tokens = sum(token_counts.values())
    live_query_start_loc = torch.tensor([0, 3, 7, 9, 14], dtype=torch.int32)
    live_seq_lens = torch.tensor([3, 14, 22, 35], dtype=torch.int32)
    live_block_table = torch.arange(16, dtype=torch.int32).reshape(4, 4)
    live_slots = torch.arange(total_tokens, dtype=torch.int64)

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            req_id_to_index={request_id: index for index, request_id in enumerate(request_ids)},
            num_computed_tokens_cpu=np.array([0, 10, 20, 30]),
            block_table=[
                SimpleNamespace(
                    block_table=SimpleNamespace(gpu=live_block_table),
                )
            ],
        ),
        query_start_loc=SimpleNamespace(gpu=live_query_start_loc),
        seq_lens=live_seq_lens,
    )
    graph_runner = GPUPrefillGraphRunner(runner, buckets=[16])
    buf = graph_runner.buffers[16]
    replayed: list[bool] = []

    class Graph:
        def replay(self) -> None:
            replayed.append(True)

    buf.captured_graphs[4] = Graph()
    buf.output_bufs[4] = torch.zeros(16, 8)
    buf.max_seq_len_bounds[4] = 64
    buf.block_table_buf = torch.zeros_like(live_block_table)

    input_ids = torch.arange(total_tokens, dtype=torch.int32)
    positions = torch.arange(total_tokens, dtype=torch.int64)
    output = graph_runner.execute_model_prefill(
        bucket=16,
        input_ids=input_ids,
        positions=positions,
        slot_mapping=live_slots,
        num_scheduled_tokens=token_counts,
    )

    assert output is buf.output_bufs[4]
    assert replayed == [True]
    assert graph_runner.last_replay_reason == "replay"
    assert torch.equal(buf.input_ids_buf[:total_tokens], input_ids)
    assert torch.all(buf.input_ids_buf[total_tokens:] == buf.pad_token_id)
    assert torch.equal(buf.positions_buf[:total_tokens], positions)
    assert torch.all(buf.positions_buf[total_tokens:] == 0)
    assert torch.equal(buf.slot_mapping_buf[:total_tokens], live_slots)
    assert torch.all(buf.slot_mapping_buf[total_tokens:] == -1)
    assert torch.equal(buf.query_start_loc_buf[:5], live_query_start_loc)
    assert torch.equal(buf.seq_lens_buf[:4], live_seq_lens)
    assert torch.equal(buf.block_table_buf, live_block_table)

    runner.input_batch.req_id_to_index = {
        "r0": 1,
        "r1": 0,
        "r2": 2,
        "r3": 3,
    }
    assert (
        graph_runner.execute_model_prefill(
            bucket=16,
            input_ids=input_ids,
            positions=positions,
            slot_mapping=live_slots,
            num_scheduled_tokens=token_counts,
        )
        is None
    )
    assert graph_runner.last_replay_reason == "request_order_mismatch"
    assert replayed == [True]

    assert (
        graph_runner.execute_model_prefill(
            bucket=16,
            input_ids=torch.zeros(17, dtype=torch.int32),
            positions=torch.zeros(17, dtype=torch.int64),
            slot_mapping=torch.zeros(17, dtype=torch.int64),
            num_scheduled_tokens={"r0": 5, "r1": 4, "r2": 4, "r3": 4},
        )
        is None
    )
    assert graph_runner.last_replay_reason == "token_count_exceeds_bucket"


def test_get_max_context_len() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import get_max_context_len

    runner: Any = SimpleNamespace(
        device=torch.device("cpu"),
        model_config=SimpleNamespace(max_model_len=40960),
    )
    assert get_max_context_len(runner) == 40960

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        model_config=None,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192)),
    )
    assert get_max_context_len(runner) == 8192

    assert get_max_context_len(MockModelRunner()) == 0


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("dcp_size", [1, 2, None])  # type: ignore[misc]
@pytest.mark.parametrize("as_dict", [False, True])  # type: ignore[misc]
def test_gpu_capture_preserves_only_valid_native_fa_schedules(
    dcp_size: int | None, as_dict: bool
) -> None:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    schedule = object()
    ordinary = object.__new__(FlashAttentionMetadata)
    ordinary.use_cascade = False
    ordinary.scheduler_metadata = schedule
    cascade = object.__new__(FlashAttentionMetadata)
    cascade.use_cascade = True
    cascade.scheduler_metadata = schedule
    gr_metadata = SimpleNamespace(use_cascade=False, scheduler_metadata=schedule)

    if as_dict:
        metas: Any = {"ordinary": ordinary, "cascade": cascade, "gr": gr_metadata}
        GPUPrefillGraphRunner._disable_stale_native_fa_scheduler(metas, dcp_size)
    else:
        for meta in (ordinary, cascade, gr_metadata):
            GPUPrefillGraphRunner._disable_stale_native_fa_scheduler(meta, dcp_size)

    assert ordinary.scheduler_metadata is (None if dcp_size == 1 else schedule)
    assert cascade.scheduler_metadata is schedule
    assert gr_metadata.scheduler_metadata is schedule


def test_gpu_max_seq_len_bound() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    # Per-layer dict: every layer's scalar is bound to the replay KV-length
    # bound (2x the bucket).
    layer_a = SimpleNamespace(max_seq_len=512)
    layer_b = SimpleNamespace(max_seq_len=512)
    attn_metadata = {"l0": layer_a, "l1": layer_b}
    GPUPrefillGraphRunner._apply_max_seq_len_bound(attn_metadata, 1024)
    assert layer_a.max_seq_len == 1024
    assert layer_b.max_seq_len == 1024

    # bound == 0 keeps the dummy value.
    layer_c = SimpleNamespace(max_seq_len=512)
    GPUPrefillGraphRunner._apply_max_seq_len_bound({"l0": layer_c}, 0)
    assert layer_c.max_seq_len == 512

    # Non-dict metadata is handled too.
    meta = SimpleNamespace(max_seq_len=512)
    GPUPrefillGraphRunner._apply_max_seq_len_bound(meta, 2048)
    assert meta.max_seq_len == 2048


def test_gpu_kv_capture_bound_matches_replay_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The value baked into the graph is the value the replay gate reads.

    With the override unset the small buckets keep ``2 * bucket`` (they are
    not in the grid at all), and the override only ever widens.
    """
    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    graph = GPUPrefillGraphRunner(MockModelRunner(), buckets=[16])
    assert graph._kv_capture_bound(1) == 2
    assert graph._kv_capture_bound(16) == 32
    assert graph._kv_capture_bound(128) == 256
    assert graph._kv_capture_bound(1024) == 2048

    graph = GPUPrefillGraphRunner(MockModelRunner(prefill_kv_bound=1024), buckets=[16])
    # Small buckets: the override is what makes them usable for cache hits.
    assert graph._kv_capture_bound(1) == 1024
    assert graph._kv_capture_bound(16) == 1024
    # Buckets whose default already covers their contexts keep it.
    assert graph._kv_capture_bound(1024) == 2048
    assert graph._kv_capture_bound(2048) == 4096


def test_gpu_bucket16_gate_uses_capture_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache hit in bucket 16 is admitted within the bound, rejected beyond it.

    This is the shape the small buckets exist for: 16 scheduled tokens with a
    KV length equal to the whole context.
    """
    from types import SimpleNamespace

    import numpy as np

    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    gpu_runner = GPUPrefillGraphRunner(MockModelRunner(prefill_kv_bound=1024), buckets=[16])
    buf = gpu_runner.buffers[16]
    buf.captured_graphs[1] = SimpleNamespace(replay=lambda: None)
    buf.output_bufs[1] = torch.zeros(16, 8)
    # What capture_graphs records for the gate; must be the same value it bakes
    # into the graph (asserted by the sibling test above).
    buf.max_seq_len_bounds[1] = gpu_runner._kv_capture_bound(16)
    assert buf.max_seq_len_bounds[1] == 1024

    def runner_for(computed: int) -> Any:
        return SimpleNamespace(
            device=torch.device("cpu"),
            input_batch=SimpleNamespace(
                req_id_to_index={"rA": 0},
                num_computed_tokens_cpu=np.array([computed]),
                block_table=[
                    SimpleNamespace(
                        block_table=SimpleNamespace(gpu=torch.zeros(1, 4, dtype=torch.int32))
                    )
                ],
            ),
            query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 16], dtype=torch.int32)),
            seq_lens=torch.tensor([computed + 16], dtype=torch.int32),
            kv_caches=[],
        )

    def replay(computed: int) -> Any:
        gpu_runner.model_runner = runner_for(computed)
        return gpu_runner.execute_model_prefill(
            bucket=16,
            input_ids=torch.zeros(16, dtype=torch.int32),
            positions=torch.zeros(16, dtype=torch.int64),
            slot_mapping=torch.zeros(16, dtype=torch.int64),
            num_scheduled_tokens={"rA": 16},
        )

    # kvlen 1008 (16 scheduled onto a 992-token cached prefix) is within the
    # capture bound, so the graph is replayed.
    assert replay(992) is buf.output_bufs[1]
    # Beyond it the step must fall back to eager: the captured scalar cannot
    # change at replay time, so attention would only see the first 1024 keys.
    assert replay(2000) is None


def test_gpu_replay_falls_back_when_kvlen_exceeds_bound() -> None:
    from types import SimpleNamespace

    import numpy as np

    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            req_id_to_index={"rA": 0},
            num_computed_tokens_cpu=np.array([1000]),
        ),
        query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 64], dtype=torch.int32)),
        seq_lens=torch.tensor([564], dtype=torch.int32),
    )
    gpu_runner = GPUPrefillGraphRunner(runner, buckets=[512])
    buf = gpu_runner.buffers[512]
    buf.captured_graphs[1] = object()
    buf.output_bufs[1] = torch.zeros(512, 8, 64)
    buf.max_seq_len_bounds[1] = 1024  # 2 x bucket

    # A cached request whose KV length (1000 + 64) exceeds the bound must
    # fall back to eager before touching the buffers.
    assert (
        gpu_runner.execute_model_prefill(
            bucket=512,
            input_ids=torch.zeros(64, dtype=torch.int32),
            positions=torch.zeros(64, dtype=torch.int64),
            slot_mapping=torch.zeros(64, dtype=torch.int64),
            num_scheduled_tokens={"rA": 64},
        )
        is None
    )

    # Within the bound, the replay proceeds (no-op fake graph).
    assert gpu_runner.last_replay_reason == "kv_length_exceeds_bound"

    class FakeGraph:
        def replay(self) -> None:
            pass

    runner.input_batch.num_computed_tokens_cpu = np.array([500])
    buf.captured_graphs[1] = FakeGraph()
    assert (
        gpu_runner.execute_model_prefill(
            bucket=512,
            input_ids=torch.zeros(64, dtype=torch.int32),
            positions=torch.zeros(64, dtype=torch.int64),
            slot_mapping=torch.zeros(64, dtype=torch.int64),
            num_scheduled_tokens={"rA": 64},
        )
        is buf.output_bufs[1]
    )
    assert gpu_runner.last_replay_reason == "replay"


@pytest.mark.parametrize("async_stage", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("native_padding", ["none", "bucket", "larger"])  # type: ignore[misc]
def test_gpu_padded_prefill_preserves_live_lengths_and_refreshes_static_buffers(
    monkeypatch: pytest.MonkeyPatch, async_stage: bool, native_padding: str
) -> None:
    """Both dispatch paths replay padded buffers with the real Attention extent."""
    from types import SimpleNamespace

    import numpy as np

    from vllm_gr.v1.beam.dispatch import GRStageKind
    from vllm_gr.v1.worker.cuda.v1 import gpu_model_runner_patch as patch
    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    live_slots = torch.arange(1152, dtype=torch.int64)
    live_table = torch.zeros((1, 72), dtype=torch.int32)
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        model_config=SimpleNamespace(enforce_eager=False),
        input_batch=SimpleNamespace(
            req_id_to_index={"r": 0},
            num_computed_tokens_cpu=np.array([0]),
            num_prompt_tokens=np.array([2048]),
            block_table=[
                SimpleNamespace(
                    slot_mapping=SimpleNamespace(gpu=live_slots),
                    block_table=SimpleNamespace(gpu=live_table),
                )
            ],
        ),
        query_start_loc=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32)),
        seq_lens=torch.zeros(1, dtype=torch.int32),
    )
    graph_runner = GPUPrefillGraphRunner(runner, buckets=[1024, 1152])
    runner._prefill_graph_runner = graph_runner
    monkeypatch.setattr(patch, "is_prefill_graph_enabled", lambda runner=None: True)
    monkeypatch.setattr(patch, "get_prefill_graph_runner", lambda _: graph_runner)
    replays: list[int] = []

    class Graph:
        def __init__(self, bucket: int) -> None:
            self.bucket = bucket

        def replay(self) -> None:
            replays.append(self.bucket)

    for bucket, buf in graph_runner.buffers.items():
        buf.captured_graphs[1] = Graph(bucket)
        buf.output_bufs[1] = torch.zeros(bucket, 8)
        buf.max_seq_len_bounds[1] = 2 * bucket
        buf.block_table_buf = torch.zeros_like(live_table)

    def pointers() -> list[int]:
        return [
            getattr(buf, name).data_ptr()
            for buf in graph_runner.buffers.values()
            for name in (
                "input_ids_buf",
                "positions_buf",
                "slot_mapping_buf",
                "query_start_loc_buf",
                "seq_lens_buf",
                "block_table_buf",
            )
        ]

    addresses = pointers()

    wrapped = patch.make_patched_model_forward(lambda *args, **kwargs: "fallback")
    for index, length in enumerate((900, 1024, 1025, 900)):
        scheduled = SimpleNamespace(
            num_scheduled_tokens={"r": length},
            total_num_scheduled_tokens=length,
        )
        if async_stage:
            scheduled.gr_stage_metadata = {
                "r": SimpleNamespace(
                    session_id="r",
                    dispatch_id=index,
                    stage_index=0,
                    kind=GRStageKind.PREFILL,
                    decode_step=None,
                    owner_generation=0,
                    request_ids=("r",),
                    num_prompt_tokens_by_request=(2048,),
                    num_prompt_tokens=2048,
                    beam_params=SimpleNamespace(beam_width=4),
                    produces_output=True,
                )
            }
        runner._current_scheduler_output = scheduled
        runner.query_start_loc.gpu[1] = length
        runner.seq_lens[0] = length
        live_table.fill_(index + 1)
        bucket = 1024 if length <= 1024 else 1152
        input_length = {"none": length, "bucket": bucket, "larger": bucket + 128}[native_padding]
        ids = torch.full((input_length,), index + 7, dtype=torch.int32)
        ids[length:].fill_(999)
        positions = torch.arange(input_length)
        buf = graph_runner.buffers[bucket]
        assert wrapped(runner, ids, positions) is buf.output_bufs[1]
        assert graph_runner.last_replay_reason == "replay"
        assert torch.equal(buf.input_ids_buf[:length], ids[:length])
        assert torch.all(buf.input_ids_buf[length:] == buf.pad_token_id)
        assert torch.equal(buf.positions_buf[:length], positions[:length])
        assert torch.all(buf.positions_buf[length:] == 0)
        assert torch.equal(buf.slot_mapping_buf[:length], live_slots[:length])
        assert torch.all(buf.slot_mapping_buf[length:] == -1)
        assert buf.query_start_loc_buf[:2].tolist() == [0, length]
        assert buf.seq_lens_buf[0].item() == length
        assert torch.equal(buf.block_table_buf, live_table)
        assert pointers() == addresses
    assert replays == [1024, 1024, 1152, 1024]
    if async_stage:
        fallback = patch.make_patched_model_forward(lambda *args, **kwargs: "fallback")
        assert fallback(runner, ids[:899], positions) == "fallback"
        assert fallback(runner, ids, positions[:899]) == "fallback"
        assert fallback(runner, ids[None, :], positions) == "fallback"
        assert replays == [1024, 1024, 1152, 1024]


def test_prefill_runner_init_propagates_allocation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static buffer allocation failures propagate to worker startup."""
    from vllm_gr.v1.worker.prefill_graph_runner import BeamPrefillGraphRunner

    def fail_full(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("no memory")

    monkeypatch.setattr(torch, "full", fail_full)
    with pytest.raises(RuntimeError, match="no memory"):
        BeamPrefillGraphRunner(MockModelRunner(), buckets=[512])


def test_fatal_prefill_capture_error_reaches_worker_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm_gr.v1.worker import gr_async_output, gr_startup, model_runner_patch
    from vllm_gr.v1.worker.cuda.v1 import gpu_model_runner_patch as patch
    from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1

    stream = SimpleNamespace(wait_stream=lambda _other: None)
    monkeypatch.setattr(patch, "is_prefill_graph_enabled", lambda runner=None: True)
    monkeypatch.setattr(patch.torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(patch.torch.cuda, "stream", lambda _side: nullcontext())
    monkeypatch.setattr(patch.torch.accelerator, "current_stream", lambda: stream)
    monkeypatch.setattr(patch.torch.accelerator, "synchronize", lambda: None)
    error = RuntimeError("operation failed due to a previous error during capture")

    def fail(_runner: Any) -> None:
        raise error

    monkeypatch.setattr(
        patch, "get_prefill_graph_runner", lambda _runner: SimpleNamespace(capture_graphs=fail)
    )
    runner = SimpleNamespace()
    with pytest.raises(RuntimeError) as raised:
        patch.capture_prefill_graphs(runner)
    assert raised.value is error
    assert not getattr(runner, "_prefill_graphs_captured", False)

    events: list[str] = []

    class Runner:
        _vllm_gr_config = SimpleNamespace(beam_max_width=128)

        def execute_model(self) -> None:
            pass

        def _prepare_inputs(self) -> None:
            pass

        def _bookkeeping_sync(self) -> None:
            pass

        def capture_model(self) -> None:
            events.append("native")

    platform = SimpleNamespace(
        discover_worker_runners=lambda: (("GPUModelRunner", Runner),),
        make_patched_prepare_inputs=lambda original: original,
        capture_prefill_graphs=patch.capture_prefill_graphs,
    )
    monkeypatch.setattr(gr_async_output, "install_async_output_guard", lambda: None)
    monkeypatch.setattr(model_runner_patch, "_install_active_worker_resource_hook", lambda: None)
    monkeypatch.setattr(
        model_runner_patch,
        "_active_runner_module_name",
        lambda: model_runner_patch._GPU_PATCH_MODULE,
    )
    # Pin the platform selector too: the installer binds the CUDA hooks class by
    # reading the real vLLM platform, so this test must not depend on the host
    # device (it fails on an NPU machine where the selector reports NPU).
    monkeypatch.setattr(
        model_runner_patch,
        "_active_platform_module_name",
        lambda: model_runner_patch._GPU_PATCH_MODULE,
    )
    monkeypatch.setattr(model_runner_patch, "_get_active_platform_module", lambda: platform)
    capture_async = Mock(return_value=None)
    monkeypatch.setattr(gr_startup, "capture_async_forward", capture_async)
    monkeypatch.setattr(
        model_runner_patch,
        "freeze_gc_after_graph_capture",
        lambda _runner: events.append("freeze"),
    )

    model_runner_patch.apply_worker_patches()
    worker = Runner()
    with pytest.raises(RuntimeError) as raised:
        worker.capture_model()
    capture_async.assert_called_once_with(worker, hooks_type=CudaStageHooksV1)
    assert raised.value is error
    assert events == ["native"]


def test_npu_bsnd_config_defaults_off() -> None:
    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

    runner = NPUPrefillGraphRunner(MockModelRunner(), buckets=[128])
    assert runner._BSND_CAPTURE_SPECS == ()


def test_bsnd_metadata_resources_are_instance_owned() -> None:
    from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadata

    tensor = torch.empty(0)
    first = BeamAttentionMetadata(0, 0, tensor, 0, tensor, tensor, tensor)
    second = BeamAttentionMetadata(0, 0, tensor, 0, tensor, tensor, tensor)
    assert first.prefill_fia_layout == second.prefill_fia_layout == "TND"
    assert first.bsnd_resources == second.bsnd_resources == {}
    first.bsnd_resources["workspaces"] = {"probe": object()}
    assert second.bsnd_resources == {}


def test_npu_bsnd_config_validates_runner_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

    runner = NPUPrefillGraphRunner(
        MockModelRunner(npu_bsnd_capture_specs="1024,4;512,4"), buckets=[512, 1024]
    )
    assert runner._BSND_CAPTURE_SPECS == ((512, 4), (1024, 4))
    with pytest.raises(ValueError, match="1024 is not in configured prefill buckets"):
        NPUPrefillGraphRunner(MockModelRunner(npu_bsnd_capture_specs="1024,4;512,4"), buckets=[512])


def test_npu_prefill_capture_skips_native_attention() -> None:
    from vllm_gr.v1.attention.backends.beam_attn_metadata import (
        BeamAttentionMetadataBuilder,
    )
    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

    class NativeBuilder:
        pass

    class Group:
        def __init__(self, builder: object) -> None:
            self.builder = builder

        def get_metadata_builder(self, _index: int) -> object:
            return self.builder

    beam_builder = object.__new__(BeamAttentionMetadataBuilder)
    beam_runner = SimpleNamespace(attn_groups=[[Group(beam_builder)]])
    native_runner = SimpleNamespace(attn_groups=[[Group(NativeBuilder())]])
    assert NPUPrefillGraphRunner._supports_beam_prefill_capture(beam_runner) is True
    assert NPUPrefillGraphRunner._supports_beam_prefill_capture(native_runner) is False


def test_npu_prefill_capture_skips_mixed_attention_groups() -> None:
    from vllm_gr.v1.attention.backends.beam_attn_metadata import (
        BeamAttentionMetadataBuilder,
    )
    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

    class Group:
        def __init__(self, builder: object) -> None:
            self.builder = builder

        def get_metadata_builder(self, _index: int) -> object:
            return self.builder

    beam_builder = object.__new__(BeamAttentionMetadataBuilder)
    runner = SimpleNamespace(attn_groups=[[Group(beam_builder), Group(object())]])
    assert NPUPrefillGraphRunner._supports_beam_prefill_capture(runner) is False


@pytest.mark.l1  # type: ignore[misc]
def test_scheduled_requests_are_fresh() -> None:
    """Dense-Prefill graphs replay only when no scheduled request has computed tokens."""
    import numpy as np

    def batch(computed: list[int], scheduled: Any) -> tuple[Any, Any]:
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                num_computed_tokens_cpu=np.array(computed),
                req_id_to_index={"rA": 2, "rB": 0},
            )
        )
        return runner, SimpleNamespace(num_scheduled_tokens=scheduled)

    fresh = prefill_graph_common.scheduled_requests_are_fresh
    # Scattered rows: the unscheduled row 1 carries context and is ignored.
    assert fresh(*batch([0, 300, 0], {"rA": 4, "rB": 4})) is True
    # A cached-Prefill suffix is a prefill-phase row but not fresh.
    assert fresh(*batch([0, 300, 32], {"rA": 4, "rB": 4})) is False
    assert fresh(*batch([0, 0, 0], {"rA": 4, "missing": 4})) is False
    assert fresh(*batch([0, 0, 0], {})) is False
    # Legacy positional list form.
    assert fresh(*batch([0, 0, 7], [4, 4])) is True
    assert fresh(*batch([0, 5, 0], [4, 4])) is False


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
def test_prefill_allocates_only_shapes_in_capture_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.worker import prefill_graph_common as common
    from vllm_gr.v1.worker.prefill_capture_plan import plan_prefill_capture

    allocated: list[int] = []
    monkeypatch.setattr(
        common, "PrefillBucketBuffers", lambda b, *args, **kwargs: allocated.append(b)
    )
    monkeypatch.setattr(common, "_get_block_table_cols", lambda runner: None)
    monkeypatch.setattr(common, "get_max_context_len", lambda runner: 512)
    monkeypatch.setattr(common.BasePrefillGraphRunner, "supports_multi_request_total_buckets", True)
    scheduler = SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=4)
    config = SimpleNamespace(
        additional_config={
            "vllm_gr": {
                "gr": {
                    "beam_width": [8, 4, 2],
                    "candidate_width": [8, 4, 2],
                    "max_decode_steps": 5,
                    "beam_batch_sizes": [1, 2, 4],
                }
            }
        }
    )
    config.scheduler_config = scheduler
    runner = SimpleNamespace(vllm_config=config, scheduler_config=scheduler, device="cpu")
    graph = common.BasePrefillGraphRunner(runner, list(range(128, 8193, 128)))
    assert allocated == list(graph.capture_plan) == list(range(128, 2049, 128))
    assert graph.capture_plan[512] == (1, 2, 4)
    assert graph.capture_plan[2048] == (4,)
    limited = plan_prefill_capture(
        [512, 1024, 2048],
        [1, 2, 4],
        max_num_seqs=2,
        max_num_batched_tokens=8192,
        max_model_len=512,
        total_token_buckets=True,
    )
    assert limited == {512: (1, 2), 1024: (2,)}
    per_context = plan_prefill_capture(
        [512, 1024],
        [1, 2],
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        max_model_len=512,
        total_token_buckets=False,
    )
    assert per_context == {512: (1, 2)}
