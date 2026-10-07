# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BeamPrefillGraphRunner."""

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

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
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.vllm_config = None
        self.model_config: Any = None


def test_match_bucket_excessive_padding_fallback() -> None:
    runner = MockModelRunner()

    # Set small max_pad_tokens=100 and max_pad_ratio=0.10 to test fallback
    graph_runner = BeamPrefillGraphRunner(
        runner,
        buckets=[1000, 10000],
        max_pad_ratio=0.10,
        max_pad_tokens=100,
    )

    # 5000 tokens trying to match 10000 (pad 5000 = 100% overhead > 10% limit) -> Fallback None
    assert graph_runner.match_bucket(5000) is None


def test_match_bucket_lut_consistency() -> None:
    """The O(1) LUT must agree with the reference policy for every token count.

    Uses explicit pad bounds so the test is independent of the dynamic
    threshold tiers (which are tuning constants and may vary).
    """
    runner = MockModelRunner()
    graph_runner = BeamPrefillGraphRunner(
        runner,
        buckets=[256, 512, 1024],
        max_pad_ratio=0.5,
        max_pad_tokens=256,
    )

    for total_tokens in range(1, 1025):
        assert graph_runner.match_bucket(total_tokens) == graph_runner._compute_match_bucket(
            total_tokens
        )
    # Beyond the largest bucket both paths fall back to None.
    assert graph_runner.match_bucket(5000) is None
    assert graph_runner._compute_match_bucket(5000) is None


def test_get_block_table_cols() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import _get_block_table_cols

    # Unconfigured runner (mocks): no columns derivable.
    assert _get_block_table_cols(MockModelRunner()) is None

    # Configured runner: ceil(max_model_len / block_size).
    runner = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=40960),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))]
        ),
    )
    assert _get_block_table_cols(runner) == 2560

    # Missing model length falls back to None.
    runner = SimpleNamespace(
        model_config=None,
        vllm_config=None,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))]
        ),
    )
    assert _get_block_table_cols(runner) is None


def test_match_bucket_dense_default_range() -> None:
    graph_runner = BeamPrefillGraphRunner(MockModelRunner())

    # Without the bound override the grid is the P2D bucket list: every total
    # maps to the same bucket it did before the small buckets existed, so a
    # 128-token or 17-token cache hit still replays in the graph it used to.
    assert graph_runner.match_bucket(1) == 128
    assert graph_runner.match_bucket(16) == 128
    assert graph_runner.match_bucket(17) == 128
    assert graph_runner.match_bucket(128) == 128
    assert graph_runner.match_bucket(129) == 256
    assert graph_runner.match_bucket(1024) == 1024
    assert graph_runner.match_bucket(1025) is None


def test_match_bucket_small_buckets_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.worker.prefill_graph_common import ENV_PREFILL_GRAPH_KV_BOUND
    from vllm_gr.v1.worker.prefill_graph_runner import (
        ENV_PREFILL_GRAPH_BUCKET_STEP,
        ENV_PREFILL_GRAPH_MAX_BUCKET,
    )

    # The small buckets join the grid only together with the dense step-range
    # override and the KV-bound opt-in (the P2D default grid stays unchanged).
    monkeypatch.setenv(ENV_PREFILL_GRAPH_BUCKET_STEP, "128")
    monkeypatch.setenv(ENV_PREFILL_GRAPH_MAX_BUCKET, "1024")
    monkeypatch.setenv(ENV_PREFILL_GRAPH_KV_BOUND, "1024")
    graph_runner = BeamPrefillGraphRunner(MockModelRunner())

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
    plain grid admits for the same scheduled total.

    A cache hit lands in the smallest bucket by scheduled tail while its KV
    length is the whole context, so moving such a step into a *smaller* bucket
    is only safe if the bound baked for that bucket is at least as wide as the
    bound the plain grid baked for the same request. That is the regression the
    small buckets introduced when they were added to the default grid.
    """
    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner
    from vllm_gr.v1.worker.prefill_graph_common import (
        ENV_PREFILL_GRAPH_KV_BOUND,
        SMALL_PREFILL_BUCKETS,
    )
    from vllm_gr.v1.worker.prefill_graph_runner import (
        ENV_PREFILL_GRAPH_BUCKET_STEP,
        ENV_PREFILL_GRAPH_MAX_BUCKET,
    )

    # The dense step-range grid with the env unset (the small buckets are
    # opt-in and stay out).
    monkeypatch.delenv(ENV_PREFILL_GRAPH_KV_BOUND, raising=False)
    monkeypatch.setenv(ENV_PREFILL_GRAPH_BUCKET_STEP, "128")
    monkeypatch.setenv(ENV_PREFILL_GRAPH_MAX_BUCKET, "8192")
    default_buckets = list(BeamPrefillGraphRunner(MockModelRunner()).buckets)

    # The opt-in grid under the bound the feature requires, compared against
    # the default grid under the *same* environment.
    monkeypatch.setenv(ENV_PREFILL_GRAPH_KV_BOUND, "1024")
    opt_in_runner = BeamPrefillGraphRunner(MockModelRunner())
    assert set(opt_in_runner.buckets) - set(default_buckets) == set(SMALL_PREFILL_BUCKETS)
    assert set(default_buckets) - set(opt_in_runner.buckets) == set()

    def coverage(bucket: int | None) -> int:
        """KV extent a step landing in ``bucket`` is admitted for."""
        if bucket is None:
            return 0
        return int(GPUPrefillGraphRunner._kv_capture_bound(bucket))

    for total in range(1, max(default_buckets) + 1):
        step_bucket = next((b for b in default_buckets if b >= total), None)
        assert coverage(opt_in_runner.match_p2d_bucket(total)) >= coverage(step_bucket)


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


def test_is_prefill_graph_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        PrefillGraphEnv,
        _read_env_flag,
    )
    from vllm_gr.v1.worker.prefill_graph_runner import (
        ENV_ENABLE_PREFILL_GRAPH,
        is_prefill_graph_enabled,
    )

    # Flag parsing (the class reads this once at import).
    monkeypatch.delenv(ENV_ENABLE_PREFILL_GRAPH, raising=False)
    assert _read_env_flag(ENV_ENABLE_PREFILL_GRAPH, default=True) is True
    assert _read_env_flag("VLLM_GR_OTHER_FLAG", default=False) is False

    monkeypatch.setenv(ENV_ENABLE_PREFILL_GRAPH, "0")
    assert _read_env_flag(ENV_ENABLE_PREFILL_GRAPH, default=True) is False
    monkeypatch.setenv(ENV_ENABLE_PREFILL_GRAPH, "false")
    assert _read_env_flag(ENV_ENABLE_PREFILL_GRAPH, default=True) is False
    monkeypatch.setenv(ENV_ENABLE_PREFILL_GRAPH, "1")
    assert _read_env_flag(ENV_ENABLE_PREFILL_GRAPH, default=True) is True
    monkeypatch.setenv(ENV_ENABLE_PREFILL_GRAPH, "true")
    assert _read_env_flag(ENV_ENABLE_PREFILL_GRAPH, default=True) is True

    # The production gate reflects the cached class variable.
    try:
        old = PrefillGraphEnv.enabled
        PrefillGraphEnv.enabled = True
        assert is_prefill_graph_enabled() is True
        PrefillGraphEnv.enabled = False
        assert is_prefill_graph_enabled() is False
    finally:
        PrefillGraphEnv.enabled = old


def test_npu_gates_allow_prefill_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: True)
    from vllm_gr.v1.worker.prefill_graph_runner import (
        is_prefill_graph_enabled,
        is_prefill_phase_batch,
    )

    # NPU prefill graphs follow the cached env flag (read once at import;
    # the test assigns the class variable directly).
    try:
        old = prefill_graph_common.PrefillGraphEnv.enabled
        prefill_graph_common.PrefillGraphEnv.enabled = True
        assert is_prefill_graph_enabled() is True

        prefill_graph_common.PrefillGraphEnv.enabled = False
        assert is_prefill_graph_enabled() is False

        # Enabled with the default (unset env) value.
        prefill_graph_common.PrefillGraphEnv.enabled = True
        assert is_prefill_graph_enabled() is True
    finally:
        prefill_graph_common.PrefillGraphEnv.enabled = old

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
    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner
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
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner

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
    from vllm_gr.v1.worker.npu_prefill_graph_runner import build_update_lists

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

    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner

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

    from vllm_gr.v1.worker.npu_prefill_graph_runner import (
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
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner

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

    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner

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
    assert 5192 not in graph_runner.buckets

    # Runner without the config field: buckets unchanged (no-op).
    graph_runner = BeamPrefillGraphRunner(MockModelRunner(), buckets=[256, 1024])
    assert graph_runner.buckets == [256, 1024]


def test_platform_bucket_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker.prefill_graph_runner import (
        DEFAULT_PREFILL_BUCKETS,
        ENV_PREFILL_GRAPH_BUCKET_STEP,
        ENV_PREFILL_GRAPH_KV_BOUND,
        ENV_PREFILL_GRAPH_MAX_BUCKET,
        NPU_DEFAULT_PREFILL_BUCKETS,
        get_default_buckets,
        get_p2d_buckets,
    )

    # Defaults are the unified P2D bucket list shared by both platforms.
    assert get_default_buckets() == get_p2d_buckets()

    # NPU keeps the legacy mixed list as a compatibility alias.
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: True)
    assert NPU_DEFAULT_PREFILL_BUCKETS == list(DEFAULT_PREFILL_BUCKETS)
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: False)

    # The legacy dense-bucket override still builds a dense range when set.
    monkeypatch.setenv(ENV_PREFILL_GRAPH_BUCKET_STEP, "256")
    monkeypatch.setenv(ENV_PREFILL_GRAPH_MAX_BUCKET, "1024")
    assert get_default_buckets() == [256, 512, 768, 1024]

    # The KV-bound override adds the small exact buckets on top of the dense
    # range (they serve cache-hit tails; their own 2 * bucket bound would
    # otherwise be tighter than the step grid's and drop replays).
    monkeypatch.setenv(ENV_PREFILL_GRAPH_KV_BOUND, "1024")
    assert get_default_buckets() == [1, 16, 256, 512, 768, 1024]

    # A deliberately tiny configured maximum still wins over the opt-in.
    monkeypatch.setenv(ENV_PREFILL_GRAPH_MAX_BUCKET, "8")
    monkeypatch.setenv(ENV_PREFILL_GRAPH_BUCKET_STEP, "8")
    assert get_default_buckets() == [1, 8]

    # Without the dense override the small buckets stay out of the P2D grid.
    monkeypatch.delenv(ENV_PREFILL_GRAPH_BUCKET_STEP)
    monkeypatch.delenv(ENV_PREFILL_GRAPH_MAX_BUCKET)
    assert get_default_buckets() == get_p2d_buckets()
    monkeypatch.delenv(ENV_PREFILL_GRAPH_KV_BOUND)


def test_prefill_bucket_config_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.worker.prefill_graph_runner import (
        ENV_PREFILL_GRAPH_BUCKET_STEP,
        ENV_PREFILL_GRAPH_MAX_BUCKET,
        get_default_buckets,
    )

    monkeypatch.setenv(ENV_PREFILL_GRAPH_BUCKET_STEP, "0")
    with pytest.raises(ValueError, match="positive integer"):
        get_default_buckets()

    monkeypatch.setenv(ENV_PREFILL_GRAPH_BUCKET_STEP, "128")
    monkeypatch.setenv(ENV_PREFILL_GRAPH_MAX_BUCKET, "1000")
    with pytest.raises(ValueError, match="must be divisible"):
        get_default_buckets()


def test_npu_max_kvlen_from_model_config() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner

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
    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

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


def test_gpu_max_seq_len_bound() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

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
    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner
    from vllm_gr.v1.worker.prefill_graph_common import ENV_PREFILL_GRAPH_KV_BOUND

    monkeypatch.delenv(ENV_PREFILL_GRAPH_KV_BOUND, raising=False)
    assert GPUPrefillGraphRunner._kv_capture_bound(1) == 2
    assert GPUPrefillGraphRunner._kv_capture_bound(16) == 32
    assert GPUPrefillGraphRunner._kv_capture_bound(128) == 256
    assert GPUPrefillGraphRunner._kv_capture_bound(1024) == 2048

    monkeypatch.setenv(ENV_PREFILL_GRAPH_KV_BOUND, "1024")
    # Small buckets: the override is what makes them usable for cache hits.
    assert GPUPrefillGraphRunner._kv_capture_bound(1) == 1024
    assert GPUPrefillGraphRunner._kv_capture_bound(16) == 1024
    # Buckets whose default already covers their contexts keep it.
    assert GPUPrefillGraphRunner._kv_capture_bound(1024) == 2048
    assert GPUPrefillGraphRunner._kv_capture_bound(2048) == 4096


def test_gpu_bucket16_gate_uses_capture_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache hit in bucket 16 is admitted within the bound, rejected beyond it.

    This is the shape the small buckets exist for: 16 scheduled tokens with a
    KV length equal to the whole context.
    """
    from types import SimpleNamespace

    import numpy as np

    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner
    from vllm_gr.v1.worker.prefill_graph_common import ENV_PREFILL_GRAPH_KV_BOUND

    monkeypatch.setenv(ENV_PREFILL_GRAPH_KV_BOUND, "1024")
    gpu_runner = GPUPrefillGraphRunner(MockModelRunner(), buckets=[16])
    buf = gpu_runner.buffers[16]
    buf.captured_graphs[1] = SimpleNamespace(replay=lambda: None)
    buf.output_bufs[1] = torch.zeros(16, 8)
    # What capture_graphs records for the gate; must be the same value it bakes
    # into the graph (asserted by the sibling test above).
    buf.max_seq_len_bounds[1] = GPUPrefillGraphRunner._kv_capture_bound(16)
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

    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

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


def test_prefill_runner_init_survives_allocation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static buffer allocation failures disable the runner instead of aborting startup."""
    from vllm_gr.v1.worker.prefill_graph_runner import BeamPrefillGraphRunner

    def fail_full(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("no memory")

    monkeypatch.setattr(torch, "full", fail_full)
    graph_runner = BeamPrefillGraphRunner(MockModelRunner(), buckets=[512])
    assert graph_runner.buckets == []
    assert graph_runner.buffers == {}
    assert graph_runner.has_captured_graph(512) is False


def test_use_dynamic_thresholds() -> None:
    from vllm_gr.v1.worker.prefill_graph_runner import BeamPrefillGraphRunner

    # Explicit padding thresholds disable dynamic matching.
    graph_runner = BeamPrefillGraphRunner(
        MockModelRunner(), buckets=[512], max_pad_ratio=0.1, max_pad_tokens=50
    )
    assert graph_runner.use_dynamic_thresholds is False

    # No explicit thresholds: dynamic thresholds are always used.
    graph_runner = BeamPrefillGraphRunner(MockModelRunner(), buckets=[512])
    assert graph_runner.use_dynamic_thresholds is True


def test_match_p2d_beam_width() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import match_p2d_beam_width

    def _beam_sp(logprobs: int) -> SimpleNamespace:
        # GR beam prefills: max_tokens=1 + flat_logprobs mark the request.
        return SimpleNamespace(logprobs=logprobs, max_tokens=1, flat_logprobs=True)

    # 1. From scheduled_new_reqs with beam-style sampling_params.logprobs
    sched_out_32 = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(sampling_params=_beam_sp(32))],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_beam_width(MockModelRunner(), sched_out_32) == 32

    sched_out_128 = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(sampling_params=_beam_sp(128))],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_beam_width(MockModelRunner(), sched_out_128) == 128

    # 2. Intermediate widths: 20 -> 32, 64 -> 128
    sched_out_20 = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(sampling_params=_beam_sp(20))],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_beam_width(MockModelRunner(), sched_out_20) == 32

    sched_out_64 = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(sampling_params=_beam_sp(64))],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_beam_width(MockModelRunner(), sched_out_64) == 128

    # 3. Width exceeding supported (256) -> None
    sched_out_256 = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(sampling_params=_beam_sp(256))],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_beam_width(MockModelRunner(), sched_out_256) is None

    # 4. Standard autoregressive requests requesting logprobs are NEVER
    #    promoted into P2D (no beam markers).
    for standard_sp in (
        SimpleNamespace(logprobs=5, max_tokens=100, flat_logprobs=False),
        SimpleNamespace(logprobs=5, max_tokens=1, flat_logprobs=False),
        SimpleNamespace(logprobs=5),
    ):
        sched_std = SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(sampling_params=standard_sp)],
            num_scheduled_tokens={"r1": 100},
        )
        assert match_p2d_beam_width(MockModelRunner(), sched_std) is None

    # 5. num_logprobs fallback requires a batch-level beam marker.
    runner_with_batch = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_logprobs={"r1": 32},
            req_ids=["r1"],
        )
    )
    sched_out_no_reqs = SimpleNamespace(
        scheduled_new_reqs=[],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_beam_width(runner_with_batch, sched_out_no_reqs) is None

    sched_marked = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                trace_headers={"vllm_gr_step_beam_widths": "32,32,32"},
                sampling_params=None,
            )
        ],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_beam_width(runner_with_batch, sched_marked) == 32


def test_has_captured_graph_distinguishes_p2d_and_prefill() -> None:
    """has_captured_graph without beam_width must NOT match P2D tuple keys."""
    from vllm_gr.v1.worker.prefill_graph_common import PrefillBucketBuffers

    buf = PrefillBucketBuffers(bucket_size=512, max_num_seqs=1, device=torch.device("cpu"))
    # Simulate P2D capture (tuples as keys)
    buf.captured_graphs[(1, 128)] = "dummy_p2d_graph_128"
    buf.captured_graphs[(1, 32)] = "dummy_p2d_graph_32"

    # With beam_width: matches corresponding P2D graph
    assert buf.has_captured_graph(1, beam_width=128) is True
    assert buf.has_captured_graph(1, beam_width=32) is True
    assert buf.has_captured_graph(1, beam_width=64) is False

    # Without beam_width (pure prefill query): must NOT match P2D graphs
    assert buf.has_captured_graph(1) is False
    assert buf.has_captured_graph(1, beam_width=None) is False

    # When pure prefill is actually captured (int key):
    buf.captured_graphs[1] = "dummy_prefill_graph"
    assert buf.has_captured_graph(1) is True


def test_is_p2d_eligible_gates_out_catalog() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import is_p2d_eligible

    sched_out = SimpleNamespace(
        num_scheduled_tokens={"req_1": 100},
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )
    runner_no_catalog = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req_1": 0},
            num_computed_tokens_cpu=[0],
            req_ids=["req_1"],
        ),
        vllm_config=SimpleNamespace(additional_config={}),
    )

    # 1. Without catalog: eligible
    assert (
        is_p2d_eligible(
            runner_no_catalog,
            sched_out,
            num_reqs=1,
            total_tokens=100,
            bucket=128,
            beam_width=128,
        )
        is True
    )

    # 2. With catalog_path on runner: ineligible
    runner_with_catalog = SimpleNamespace(
        catalog_path="/path/to/catalog.json",
        input_batch=runner_no_catalog.input_batch,
        vllm_config=runner_no_catalog.vllm_config,
    )
    assert (
        is_p2d_eligible(
            runner_with_catalog,
            sched_out,
            num_reqs=1,
            total_tokens=100,
            bucket=128,
            beam_width=128,
        )
        is False
    )

    # 3. With vllm_gr_catalog_path in vllm_config additional_config: ineligible
    runner_config_catalog = SimpleNamespace(
        input_batch=runner_no_catalog.input_batch,
        vllm_config=SimpleNamespace(
            additional_config={"vllm_gr_catalog_path": "/path/to/cat.json"}
        ),
    )
    assert (
        is_p2d_eligible(
            runner_config_catalog,
            sched_out,
            num_reqs=1,
            total_tokens=100,
            bucket=128,
            beam_width=128,
        )
        is False
    )

    # 4. With catalog on scheduler_output: ineligible
    sched_out_catalog = SimpleNamespace(
        catalog=object(),
        num_scheduled_tokens={"req_1": 100},
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )
    assert (
        is_p2d_eligible(
            runner_no_catalog,
            sched_out_catalog,
            num_reqs=1,
            total_tokens=100,
            bucket=128,
            beam_width=128,
        )
        is False
    )


def test_is_p2d_eligible_gates_out_non_three_step_requests() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import (
        P2D_TOTAL_STEPS_HEADER,
        is_p2d_eligible,
    )

    def make_sched_out(total_steps: str | None) -> SimpleNamespace:
        reqs = [
            SimpleNamespace(
                sampling_params=SimpleNamespace(beam_width=128),
                trace_headers=(
                    {P2D_TOTAL_STEPS_HEADER: total_steps} if total_steps is not None else None
                ),
            )
        ]
        return SimpleNamespace(
            num_scheduled_tokens={"req_1": 100},
            scheduled_new_reqs=reqs,
            scheduled_spec_decode_tokens=[],
            scheduled_encoder_inputs=[],
        )

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req_1": 0},
            num_computed_tokens_cpu=[0],
            req_ids=["req_1"],
        ),
        vllm_config=SimpleNamespace(additional_config={}),
    )
    kwargs = {"num_reqs": 1, "total_tokens": 100, "bucket": 128, "beam_width": 128}

    # 1. Exact three-step request: eligible.
    assert is_p2d_eligible(runner, make_sched_out("3"), **kwargs) is True

    # 2. Fewer steps would over-generate: ineligible.
    assert is_p2d_eligible(runner, make_sched_out("2"), **kwargs) is False

    # 3. More steps would be silently truncated: ineligible.
    assert is_p2d_eligible(runner, make_sched_out("5"), **kwargs) is False

    # 4. Missing stamped step count (legacy / non-GR callers): eligibility is
    #    unchanged so callers without the header keep today's behavior.
    assert is_p2d_eligible(runner, make_sched_out(None), **kwargs) is True

    # 5. Malformed stamped value: treated as absent, eligibility unchanged.
    sched_out_malformed = make_sched_out("three")
    sched_out_malformed.scheduled_new_reqs = [
        SimpleNamespace(
            sampling_params=SimpleNamespace(beam_width=128),
            trace_headers={P2D_TOTAL_STEPS_HEADER: "three"},
        )
    ]
    assert is_p2d_eligible(runner, sched_out_malformed, **kwargs) is True


def test_is_p2d_eligible_gates_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import (
        is_p2d_eligible,
        match_p2d_beam_width,
    )

    sched_out = SimpleNamespace(
        num_scheduled_tokens={"req_1": 100},
        scheduled_new_reqs=[SimpleNamespace(sampling_params=SimpleNamespace(beam_width=128))],
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req_1": 0},
            num_computed_tokens_cpu=[0],
            req_ids=["req_1"],
        ),
        vllm_config=SimpleNamespace(additional_config={}),
    )

    # 1. Non-NPU (GPU): eligible and resolves beam width
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: False)
    assert (
        is_p2d_eligible(
            runner,
            sched_out,
            num_reqs=1,
            total_tokens=100,
            bucket=128,
            beam_width=128,
        )
        is True
    )
    assert match_p2d_beam_width(runner, sched_out) == 128

    # 2. NPU platform: P2D is gated off and beam-width matching returns None.
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: True)
    assert (
        is_p2d_eligible(
            runner,
            sched_out,
            num_reqs=1,
            total_tokens=100,
            bucket=128,
            beam_width=128,
        )
        is False
    )
    assert match_p2d_beam_width(runner, sched_out) is None


def test_p2d_kill_switch_disables_all_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import (
        is_p2d_eligible,
        match_p2d_beam_width,
    )

    sched_out = SimpleNamespace(
        num_scheduled_tokens={"req_1": 100},
        scheduled_new_reqs=[SimpleNamespace(sampling_params=SimpleNamespace(beam_width=128))],
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req_1": 0},
            num_computed_tokens_cpu=[0],
            req_ids=["req_1"],
        ),
        vllm_config=SimpleNamespace(additional_config={}),
    )

    monkeypatch.setattr(prefill_graph_common, "P2DEnv", SimpleNamespace(disabled=True))
    assert match_p2d_beam_width(runner, sched_out) is None
    assert (
        is_p2d_eligible(
            runner,
            sched_out,
            num_reqs=1,
            total_tokens=100,
            bucket=128,
            beam_width=128,
        )
        is False
    )


def test_is_p2d_eligible_inspects_runner_private_attribute() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import is_p2d_eligible

    sched_out = SimpleNamespace(
        num_scheduled_tokens={"req_1": 100},
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )
    p_runner = SimpleNamespace(max_bucket=2048)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req_1": 0},
            num_computed_tokens_cpu=[0],
            req_ids=["req_1"],
        ),
        vllm_config=SimpleNamespace(additional_config={}),
        _prefill_graph_runner=p_runner,
    )

    # Within max_bucket (100 <= 2048)
    assert (
        is_p2d_eligible(
            runner,
            sched_out,
            num_reqs=1,
            total_tokens=100,
            bucket=None,
            beam_width=128,
        )
        is True
    )
    # Exceeds max_bucket (2049 > 2048) -> False
    assert (
        is_p2d_eligible(
            runner,
            sched_out,
            num_reqs=1,
            total_tokens=2049,
            bucket=None,
            beam_width=128,
        )
        is False
    )


def test_gpu_prefill_graph_runner_head_dim_mismatch_asserts() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    class DummyAttnImpl:
        head_size = 64  # Mismatch with hf_config head_dim = 128

    class DummyLayer:
        def __init__(self) -> None:
            self.self_attn = SimpleNamespace(impl=DummyAttnImpl())

    class DummyModel:
        def __init__(self) -> None:
            self.model = SimpleNamespace(layers=[DummyLayer()])

    runner = MockModelRunner()
    runner.model = DummyModel()  # type: ignore[attr-defined]
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=32000,
        )
    )

    with pytest.raises(AssertionError, match="head_dim.*impl.head_size"):
        GPUPrefillGraphRunner(runner, buckets=[128])


def test_gpu_prefill_graph_runner_beam_width_exceeds_max_beams_asserts() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    runner = MockModelRunner()
    gpu_runner = GPUPrefillGraphRunner(runner, buckets=[128])
    gpu_runner.p2d_pool = SimpleNamespace(max_beams=128)
    buf = gpu_runner.buffers[128]
    # Simulate captured graph exists
    buf.captured_graphs[(1, 256)] = SimpleNamespace(replay=lambda: None)

    with pytest.raises(AssertionError, match="exceeds p2d_pool.max_beams"):
        gpu_runner.execute_p2d(
            bucket=128,
            input_ids=torch.zeros(64, dtype=torch.int32),
            positions=torch.zeros(64, dtype=torch.int64),
            num_scheduled_tokens=[64],
            beam_width=256,
        )


def test_match_p2d_bucket_does_not_fall_back_on_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MockModelRunner()
    graph_runner = BeamPrefillGraphRunner(
        runner,
        buckets=[1000, 10000],
        max_pad_ratio=0.10,
        max_pad_tokens=100,
    )
    # Standard prefill falls back on 5000 -> 10000 (100% overhead > 10% limit)
    assert graph_runner.match_bucket(5000) is None
    # P2D intentionally stays in CUDA Graph to avoid falling back to slow eager beam search
    assert graph_runner.match_p2d_bucket(5000) == 10000
    assert graph_runner.match_p2d_bucket(950) == 1000

    # On NPU platform, P2D is gated off: bucket matching declines and the
    # base-class decode methods still signal "not implemented" for this
    # runner instance via their NotImplementedError contract.
    from types import SimpleNamespace

    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: True)
    assert graph_runner.match_p2d_bucket(5000) is None
    assert graph_runner.has_2d_graph() is False
    monkeypatch.setattr(prefill_graph_common, "P2DEnv", SimpleNamespace(disabled=True))
    assert graph_runner.match_p2d_bucket(5000) is None
    assert graph_runner.execute_p2d(1000, None, None, [100]) is None
    assert graph_runner.execute_eager_p2d(runner, None, None, [100]) is None
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: False)
    monkeypatch.setattr(prefill_graph_common, "P2DEnv", SimpleNamespace(disabled=False))
    assert graph_runner.match_p2d_bucket(5000) == 10000


def test_p2d_buffer_pool_cum_lps_float32() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import P2DBufferPool

    pool = P2DBufferPool(
        max_batch=1,
        max_bucket=128,
        max_beams=32,
        num_layers=1,
        device=torch.device("cpu"),
    )
    assert pool.cum_lps_0.dtype == torch.float32
    assert pool.cum_lps_1.dtype == torch.float32
    assert pool.cum_lps_2.dtype == torch.float32
    assert pool.final_cum_lps.dtype == torch.float32


def test_gpu_step0_select_topk() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        P2DBufferPool,
        gpu_step0_select_topk,
    )

    pool = P2DBufferPool(
        max_batch=1,
        max_bucket=128,
        max_beams=4,
        vocab_size=10,
        device=torch.device("cpu"),
    )
    logits = torch.zeros((1, 10), dtype=torch.float32)
    logits[0, 7] = 10.0
    logits[0, 3] = 8.0
    logits[0, 5] = 6.0
    logits[0, 1] = 4.0

    gpu_step0_select_topk(logits, num_reqs=1, beam_width=4, pool=pool)

    # Check top tokens
    assert pool.tokens_0[0, 0].item() == 7
    assert pool.tokens_0[0, 1].item() == 3
    assert pool.tokens_0[0, 2].item() == 5
    assert pool.tokens_0[0, 3].item() == 1

    # Check input_ids_1 populated
    assert pool.input_ids_1[0].item() == 7
    assert pool.input_ids_1[1].item() == 3
    assert pool.input_ids_1[2].item() == 5
    assert pool.input_ids_1[3].item() == 1

    # Check cum_lps_0 values are log_softmax values in float32
    expected_lps = torch.log_softmax(logits, dim=-1)
    for k in range(4):
        tok = pool.tokens_0[0, k].item()
        assert abs(pool.cum_lps_0[0, k].item() - expected_lps[0, tok].item()) < 1e-5


def test_beam_attn_p2d_attention_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.attention.backends import beam_attn

    dispatched: dict[str, Any] = {}

    class DummyExecutionModule:
        @staticmethod
        def p2d_attention(*args: Any, **kwargs: Any) -> None:
            dispatched["called"] = True
            dispatched["args"] = args
            dispatched["kwargs"] = kwargs

    monkeypatch.setattr(beam_attn, "_get_platform_execution_module", lambda: DummyExecutionModule)

    dummy_out = torch.zeros(1)
    dummy_query = torch.zeros(1)
    dummy_key = torch.zeros(1)
    dummy_val = torch.zeros(1)
    dummy_meta = SimpleNamespace(p2d_pool=None, p2d_pass=0)

    beam_attn.BeamAttentionImpl.p2d_attention(
        output=dummy_out,
        query=dummy_query,
        key=dummy_key,
        value=dummy_val,
        attn_metadata=dummy_meta,
        layer_idx=0,
        scale=1.0,
        num_heads=8,
        num_kv_heads=2,
        head_size=128,
        vllm_flash_attn_version=2,
    )
    assert dispatched.get("called") is True


def test_gpu_prefill_graph_runner_execute_eager_p2d() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    runner = MockModelRunner()
    gpu_runner = GPUPrefillGraphRunner(runner, buckets=[128])
    gpu_runner.p2d_pool = SimpleNamespace(
        max_beams=128,
        cu_seqlens_q_0=torch.zeros(3, dtype=torch.int32),
        cu_seqlens_k_prompt=torch.zeros(3, dtype=torch.int32),
        cu_seqlens_q_prefix=torch.zeros(3, dtype=torch.int32),
        positions_1=torch.zeros(128, dtype=torch.int64),
        positions_2=torch.zeros(128, dtype=torch.int64),
        logits_0=torch.zeros((1, 100), dtype=torch.bfloat16),
        tokens_0=torch.zeros((1, 128), dtype=torch.int64),
        cum_lps_0=torch.zeros((1, 128), dtype=torch.float32),
        input_ids_1=torch.zeros(128, dtype=torch.int32),
        final_tokens=torch.ones((1, 32, 3), dtype=torch.int64),
        final_cum_lps=torch.zeros((1, 32), dtype=torch.float32),
        valid_beam_counts=torch.zeros(1, dtype=torch.int32),
        step_scores=torch.zeros((1, 32, 3), dtype=torch.float32),
        block_table=torch.zeros((1, 4), dtype=torch.int32),
        seqused_k=torch.zeros(1, dtype=torch.int32),
        logits_indices_0=torch.zeros(2, dtype=torch.int64),
    )

    replayed = []
    gpu_runner.captured_2d_graphs[(1, 32)] = SimpleNamespace(replay=lambda: replayed.append(True))

    mock_runner = SimpleNamespace(
        vllm_config=None,
        model=lambda input_ids, positions: torch.zeros((input_ids.shape[0], 64)),
    )
    mock_runner.model.compute_logits = lambda hidden: torch.zeros((1, 100))

    res = gpu_runner.execute_eager_p2d(
        runner=mock_runner,
        input_ids=torch.zeros(50, dtype=torch.int32),
        positions=torch.zeros(50, dtype=torch.int64),
        num_scheduled_tokens=[50],
        beam_width=32,
        block_table=torch.zeros((1, 4), dtype=torch.int32),
        slot_mapping=torch.zeros(50, dtype=torch.int64),
    )
    assert res is not None
    tokens, lps, valid_counts, step_scores = res
    assert len(replayed) == 1
    assert tokens.shape == (1, 32, 3)
    assert lps.shape == (1, 32)
    assert valid_counts.shape == (1,)
    assert step_scores.shape == (1, 32, 3)


def test_p2d_option_b_config_and_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Option B: Monolithic capture bucket list and batch sizes are configurable,

    while BasePrefillGraphRunner capacity stays at 8192 (or scheduler max) for
    Eager-P + 2D. Also verifies env var overrides VLLM_GR_P2D_BUCKETS and
    VLLM_GR_P2D_BATCH_SIZES, and the universal VLLM_GR_DISABLE_P2D kill switch.
    """
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import (
        DEFAULT_P2D_BUCKETS,
        DEFAULT_P2D_SUPPORTED_NUM_REQS,
        BasePrefillGraphRunner,
        get_default_buckets,
        get_p2d_buckets,
        get_p2d_supported_num_reqs,
        is_p2d_eligible,
        is_p2d_enabled,
    )

    # 1. Defaults
    monkeypatch.delenv("VLLM_GR_P2D_BUCKETS", raising=False)
    monkeypatch.delenv("VLLM_GR_P2D_BATCH_SIZES", raising=False)
    monkeypatch.delenv("VLLM_GR_DISABLE_P2D", raising=False)
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: False)

    assert get_p2d_buckets() == DEFAULT_P2D_BUCKETS
    assert get_p2d_supported_num_reqs() == DEFAULT_P2D_SUPPORTED_NUM_REQS == [1, 2]
    assert get_default_buckets() == get_p2d_buckets()
    assert is_p2d_enabled() is True

    # 2. Env overrides
    monkeypatch.setenv("VLLM_GR_P2D_BUCKETS", "256, 512, 2048")
    monkeypatch.setenv("VLLM_GR_P2D_BATCH_SIZES", "1, 2, 4")
    assert get_p2d_buckets() == [256, 512, 2048]
    assert get_p2d_supported_num_reqs() == [1, 2, 4]
    assert get_default_buckets() == [256, 512, 2048]

    # Reset
    monkeypatch.delenv("VLLM_GR_P2D_BUCKETS", raising=False)
    monkeypatch.delenv("VLLM_GR_P2D_BATCH_SIZES", raising=False)

    # 3. P2D is gated off on NPU; the universal kill switch gates both platforms.
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: True)
    assert is_p2d_enabled() is False
    monkeypatch.setattr(prefill_graph_common, "P2DEnv", SimpleNamespace(disabled=True))
    assert is_p2d_enabled() is False
    monkeypatch.setattr(prefill_graph_common, "P2DEnv", SimpleNamespace(disabled=False))
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: False)
    assert is_p2d_enabled() is True

    # 4. BasePrefillGraphRunner capacity vs capture buckets
    mock_runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=128),
        device=torch.device("cpu"),
    )
    runner = BasePrefillGraphRunner(mock_runner)
    assert runner.buckets == get_p2d_buckets()
    assert runner.max_bucket == 8192

    # 5. is_p2d_eligible accepts prompts > 1024 tokens up to 8192
    sched_out = SimpleNamespace(
        num_scheduled_tokens={"req_1": 4096},
        scheduled_new_reqs=[SimpleNamespace(sampling_params=SimpleNamespace(beam_width=128))],
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )
    runner_ns = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req_1": 0},
            num_computed_tokens_cpu=[0],
            req_ids=["req_1"],
        ),
        vllm_config=SimpleNamespace(additional_config={}),
        _prefill_graph_runner=runner,
    )
    assert (
        is_p2d_eligible(
            runner=runner_ns,
            scheduler_output=sched_out,
            num_reqs=1,
            total_tokens=4096,
            bucket=None,
            beam_width=128,
        )
        is True
    )

    # Exceeds max_bucket (8193 > 8192) -> False
    assert (
        is_p2d_eligible(
            runner=runner_ns,
            scheduler_output=sched_out,
            num_reqs=1,
            total_tokens=8193,
            bucket=None,
            beam_width=128,
        )
        is False
    )


def test_run_decode_passes_shared_beam_math() -> None:
    """Validates the shared decode-pass beam math on CPU with a deterministic
    fake model: every row emits logits [vocab, vocab-1, ..., 1], so the
    hand-computable beam tree must match exactly."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import (
        P2DBufferPool,
        run_decode_passes,
        step0_select_topk,
    )

    vocab = 4
    beam_width = 2
    pool = P2DBufferPool(
        max_batch=1,
        max_bucket=16,
        max_beams=beam_width,
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        vocab_size=vocab,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    class FakeModel:
        def __call__(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            del positions
            hidden = torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)
            hidden[torch.arange(input_ids.shape[0]), input_ids.long() % 4] = 1.0
            return hidden

        def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
            n = hidden.shape[0]
            return (
                torch.arange(vocab, 0, -1, dtype=torch.float32)
                .unsqueeze(0)
                .expand(n, vocab)
                .clone()
            )

    model = FakeModel()
    runner = type("FakeRunner", (), {"model": model, "vllm_config": None})()

    logits_0 = model.compute_logits(torch.zeros(1, 4))
    pool.logits_0[:1].copy_(logits_0)
    step0_select_topk(logits_0=pool.logits_0, num_reqs=1, beam_width=beam_width, pool=pool)

    attn_meta_1 = SimpleNamespace()
    attn_meta_2 = SimpleNamespace()
    run_decode_passes(
        runner=runner,
        p2d_pool=pool,
        attn_meta_1=attn_meta_1,
        attn_meta_2=attn_meta_2,
        total_beam_tokens=beam_width,
        num_reqs=1,
        beam_width=beam_width,
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        forward_ctx_fn=lambda r, m, nt: nullcontext(),
    )

    lps = torch.log_softmax(torch.tensor([4.0, 3.0, 2.0, 1.0]), dim=-1)
    v0, v1 = lps[0].item(), lps[1].item()
    assert pool.final_tokens[0, 0].tolist() == [0, 0, 0]
    assert pool.final_tokens[0, 1].tolist() == [0, 0, 1]
    assert abs(pool.final_cum_lps[0, 0].item() - 3 * v0) < 1e-4
    assert abs(pool.final_cum_lps[0, 1].item() - (2 * v0 + v1)) < 1e-4


def test_gpu_p2d_rejects_missing_block_table_or_slot_mapping_in_paged_mode() -> None:
    """Verifies that execute_p2d and execute_eager_p2d return None when required paged KV tensors are missing."""
    from types import SimpleNamespace

    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    runner = MockModelRunner()
    gpu_runner = GPUPrefillGraphRunner(runner, buckets=[128])
    gpu_runner.p2d_pool = SimpleNamespace(
        max_beams=128,
        prompt_kv=None,  # paged KV mode
    )
    buf = gpu_runner.buffers[128]
    buf.captured_graphs[(1, 32)] = SimpleNamespace(replay=lambda: None)
    gpu_runner.captured_2d_graphs[(1, 32)] = SimpleNamespace(replay=lambda: None)

    # 1. execute_p2d without block_table or slot_mapping returns None
    res1 = gpu_runner.execute_p2d(
        bucket=128,
        input_ids=torch.zeros(64, dtype=torch.int32),
        positions=torch.zeros(64, dtype=torch.int64),
        num_scheduled_tokens=[64],
        beam_width=32,
        block_table=None,
        slot_mapping=None,
    )
    assert res1 is None

    # 2. execute_eager_p2d without block_table returns None
    mock_runner = SimpleNamespace(
        vllm_config=None,
        model=lambda input_ids, positions: torch.zeros((input_ids.shape[0], 64)),
    )
    res2 = gpu_runner.execute_eager_p2d(
        runner=mock_runner,
        input_ids=torch.zeros(64, dtype=torch.int32),
        positions=torch.zeros(64, dtype=torch.int64),
        num_scheduled_tokens=[64],
        beam_width=32,
        block_table=None,
    )
    assert res2 is None

    # 3. execute_eager_p2d with block_table present but slot_mapping missing returns None
    res3 = gpu_runner.execute_eager_p2d(
        runner=mock_runner,
        input_ids=torch.zeros(64, dtype=torch.int32),
        positions=torch.zeros(64, dtype=torch.int64),
        num_scheduled_tokens=[64],
        beam_width=32,
        block_table=torch.zeros((1, 10), dtype=torch.int32),
        slot_mapping=None,
    )
    assert res3 is None


def test_is_p2d_eligible_supports_computed_tokens_with_monolithic_bucket() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import is_p2d_eligible

    # Request 1: prompt has 512 tokens, 128 computed, 384 scheduled (128 + 384 == 512)
    sched_out = SimpleNamespace(
        num_scheduled_tokens={"req_1": 384},
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req_1": 0},
            num_computed_tokens_cpu=[128],
            num_prompt_tokens=[512],
            req_ids=["req_1"],
        ),
        vllm_config=SimpleNamespace(additional_config={}),
    )

    # Monolithic bucket=512 is eligible when computed > 0 and computed + sched == prompt
    assert (
        is_p2d_eligible(
            runner,
            sched_out,
            num_reqs=1,
            total_tokens=384,
            bucket=512,
            beam_width=128,
        )
        is True
    )

    # But if prompt is not completed (e.g. only 200 scheduled, 128 + 200 != 512), not eligible
    sched_out_incomplete = SimpleNamespace(
        num_scheduled_tokens={"req_1": 200},
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )
    assert (
        is_p2d_eligible(
            runner,
            sched_out_incomplete,
            num_reqs=1,
            total_tokens=200,
            bucket=512,
            beam_width=128,
        )
        is False
    )


def test_gpu_execute_p2d_handles_computed_tokens() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    runner = MockModelRunner()
    gpu_runner = GPUPrefillGraphRunner(runner, buckets=[512])

    p2d_pool = SimpleNamespace(
        max_beams=32,
        prompt_kv=None,
        input_ids_0=torch.zeros(512, dtype=torch.int32),
        positions_0=torch.zeros(512, dtype=torch.int64),
        logits_indices_0=torch.zeros(2, dtype=torch.int64),
        cu_seqlens_q_0=torch.zeros(3, dtype=torch.int32),
        cu_seqlens_k_prompt=torch.zeros(3, dtype=torch.int32),
        cu_seqlens_q_prefix=torch.zeros(3, dtype=torch.int32),
        positions_1=torch.zeros(64, dtype=torch.int64),
        positions_2=torch.zeros(64, dtype=torch.int64),
        seqused_k=torch.zeros(2, dtype=torch.int32),
        block_table=torch.zeros((2, 10), dtype=torch.int32),
        slot_mapping_0=torch.zeros(512, dtype=torch.int64),
        final_tokens=torch.zeros((2, 32, 3), dtype=torch.int64),
        final_cum_lps=torch.zeros((2, 32), dtype=torch.float32),
        valid_beam_counts=torch.zeros(2, dtype=torch.int32),
        step_scores=torch.zeros((2, 32, 3), dtype=torch.float32),
    )
    gpu_runner.p2d_pool = p2d_pool
    buf = gpu_runner.buffers[512]
    buf.captured_graphs[(1, 32)] = SimpleNamespace(replay=lambda: None)
    buf.captured_graphs[(2, 32)] = SimpleNamespace(replay=lambda: None)

    block_table = torch.zeros((2, 10), dtype=torch.int32)
    slot_mapping = torch.zeros(512, dtype=torch.int64)

    # 1. 1 request: scheduled=384, computed=128 -> T0=512
    res = gpu_runner.execute_p2d(
        bucket=512,
        input_ids=torch.zeros(384, dtype=torch.int32),
        positions=torch.zeros(384, dtype=torch.int64),
        num_scheduled_tokens=[384],
        beam_width=32,
        block_table=block_table,
        slot_mapping=slot_mapping,
        computed_tokens=[128],
    )
    assert res is not None
    assert p2d_pool.seqused_k[0].item() == 512
    assert p2d_pool.positions_1[0].item() == 512
    assert p2d_pool.positions_2[0].item() == 513
    assert p2d_pool.cu_seqlens_q_0[0].item() == 0
    assert p2d_pool.cu_seqlens_q_0[1].item() == 384

    # 2. 2 requests: req 0 (sched=100, comp=50 -> T0=150), req 1 (sched=200, comp=60 -> T1=260)
    res2 = gpu_runner.execute_p2d(
        bucket=512,
        input_ids=torch.zeros(300, dtype=torch.int32),
        positions=torch.zeros(300, dtype=torch.int64),
        num_scheduled_tokens=[100, 200],
        beam_width=32,
        block_table=block_table,
        slot_mapping=slot_mapping,
        computed_tokens=[50, 60],
    )
    assert res2 is not None
    assert p2d_pool.seqused_k[0].item() == 150
    assert p2d_pool.seqused_k[1].item() == 260
    assert p2d_pool.positions_1[0].item() == 150
    assert p2d_pool.positions_1[32].item() == 260
    assert p2d_pool.positions_2[0].item() == 151
    assert p2d_pool.positions_2[32].item() == 261
    assert p2d_pool.cu_seqlens_q_0[0].item() == 0
    assert p2d_pool.cu_seqlens_q_0[1].item() == 100
    assert p2d_pool.cu_seqlens_q_0[2].item() == 300


def test_compute_sid_slices_from_tokenizer() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        SIDSlices,
        compute_sid_slices_from_tokenizer,
        resolve_sid_slices_from_runner,
    )

    # 1. Valid tokenizer with added_tokens
    mock_tok = SimpleNamespace(
        get_added_vocab=lambda: (
            {f"<s_a_{i}>": 1000 + i for i in range(16)}
            | {f"<s_b_{i}>": 2000 + i for i in range(16)}
            | {f"<s_c_{i}>": 3000 + i for i in range(16)}
            | {"<|sid_begin|>": 4000, "<|sid_end|>": 4001}
        )
    )
    slices = compute_sid_slices_from_tokenizer(mock_tok)
    assert isinstance(slices, SIDSlices)
    assert slices.sid_a == (1000, 16)
    assert slices.sid_b == (2000, 16)
    assert slices.sid_c == (3000, 16)

    # 2. Missing one level (e.g. only 'a' and 'b') -> returns None
    mock_partial = SimpleNamespace(
        get_added_vocab=lambda: (
            {f"<s_a_{i}>": 1000 + i for i in range(16)}
            | {f"<s_b_{i}>": 2000 + i for i in range(16)}
        )
    )
    assert compute_sid_slices_from_tokenizer(mock_partial) is None

    # 3. None or empty tokenizer -> returns None
    assert compute_sid_slices_from_tokenizer(None) is None
    assert compute_sid_slices_from_tokenizer(SimpleNamespace(get_added_vocab=dict)) is None

    # 4. resolve_sid_slices_from_runner fallback
    assert resolve_sid_slices_from_runner(SimpleNamespace()) is None
    assert resolve_sid_slices_from_runner(SimpleNamespace(model_config=None)) is None
    assert (
        resolve_sid_slices_from_runner(
            SimpleNamespace(model_config=SimpleNamespace(tokenizer=None))
        )
        is None
    )


def test_step0_select_topk_with_sid_slices() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        P2DBufferPool,
        SIDSlices,
        step0_select_topk,
    )

    slices = SIDSlices(sid_a=(500, 100), sid_b=(1000, 100), sid_c=(1500, 100))
    pool = P2DBufferPool(
        max_batch=1,
        max_bucket=128,
        max_beams=4,
        vocab_size=2000,
        device=torch.device("cpu"),
        sid_slices=slices,
    )

    # Set logits_0: highest logit within sid_a range [500, 600) is at index 542
    pool.logits_0.zero_()
    # High value out-of-domain (should be ignored due to slicing)
    pool.logits_0[0, 10] = 999.0
    # High values inside sid_a range
    pool.logits_0[0, 542] = 100.0
    pool.logits_0[0, 510] = 90.0
    pool.logits_0[0, 580] = 80.0
    pool.logits_0[0, 501] = 70.0

    step0_select_topk(logits_0=pool.logits_0, num_reqs=1, beam_width=4, pool=pool)

    # Assert top-4 tokens fall strictly within [500, 600) and top is 542
    tokens = pool.tokens_0[0, :4].tolist()
    assert tokens == [542, 510, 580, 501]
    assert all(500 <= t < 600 for t in tokens)
    assert pool.input_ids_1[0].item() == 542


def test_run_decode_passes_with_sid_slices() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        P2DBufferPool,
        P2DMetadata,
        SIDSlices,
        run_decode_passes,
    )

    slices = SIDSlices(sid_a=(500, 100), sid_b=(1000, 100), sid_c=(1500, 100))
    pool = P2DBufferPool(
        max_batch=1,
        max_bucket=128,
        max_beams=2,
        num_layers=1,
        num_kv_heads=1,
        head_dim=16,
        vocab_size=2000,
        device=torch.device("cpu"),
        sid_slices=slices,
    )

    pool.cum_lps_0[0, 0] = 0.0
    pool.cum_lps_0[0, 1] = -0.5
    pool.tokens_0[0, 0] = 510
    pool.tokens_0[0, 1] = 520
    pool.input_ids_1[0] = 510
    pool.input_ids_1[1] = 520

    class MockModel:
        def __call__(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            return torch.zeros((input_ids.shape[0], 16))

        def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
            B = hidden.shape[0]
            logits = torch.zeros((B, 2000))
            # Out-of-domain distractor (should be ignored by slicing)
            logits[:, 50] = 9999.0
            # Valid in-slice tokens: sid_b in [1000, 1100), sid_c in [1500, 1600)
            logits[:, 1025] = 50.0
            logits[:, 1075] = 40.0
            logits[:, 1530] = 60.0
            logits[:, 1580] = 55.0
            return logits

    runner = SimpleNamespace(model=MockModel())

    @contextlib.contextmanager
    def mock_forward_ctx(r: Any, meta: Any, tokens: int) -> Iterator[None]:
        yield

    attn_meta_1 = P2DMetadata(
        p2d_pool=pool,
        p2d_pass=1,
        bucket=128,
        num_reqs=1,
        beam_width=2,
        total_prompt_tokens=128,
    )
    attn_meta_2 = P2DMetadata(
        p2d_pool=pool,
        p2d_pass=2,
        bucket=128,
        num_reqs=1,
        beam_width=2,
        total_prompt_tokens=128,
    )

    run_decode_passes(
        runner=runner,
        p2d_pool=pool,
        attn_meta_1=attn_meta_1,
        attn_meta_2=attn_meta_2,
        total_beam_tokens=2,
        num_reqs=1,
        beam_width=2,
        num_layers=1,
        num_kv_heads=1,
        head_dim=16,
        forward_ctx_fn=mock_forward_ctx,
    )

    # Tokens 1 must be within [1000, 1100)
    toks_1 = pool.tokens_1[0, :2].tolist()
    assert all(1000 <= t < 1100 for t in toks_1)

    # Tokens 2 must be within [1500, 1600)
    toks_2 = pool.tokens_2[0, :2].tolist()
    assert all(1500 <= t < 1600 for t in toks_2)

    # Final tokens across 3 steps: [step0 (sid_a), step1 (sid_b), step2 (sid_c)]
    for beam in range(2):
        assert 500 <= pool.final_tokens[0, beam, 0].item() < 600
        assert 1000 <= pool.final_tokens[0, beam, 1].item() < 1100
        assert 1500 <= pool.final_tokens[0, beam, 2].item() < 1600


def test_runners_initialize_sid_slices() -> None:
    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner
    from vllm_gr.v1.worker.prefill_graph_common import SIDSlices

    mock_tok = SimpleNamespace(
        get_added_vocab=lambda: (
            {f"<s_a_{i}>": 1000 + i for i in range(16)}
            | {f"<s_b_{i}>": 2000 + i for i in range(16)}
            | {f"<s_c_{i}>": 3000 + i for i in range(16)}
        )
    )

    # 1. Runner with no tokenizer -> sid_slices is None
    runner_no_tok = MockModelRunner()
    gpu_runner = GPUPrefillGraphRunner(runner_no_tok, buckets=[128])
    assert gpu_runner.sid_slices is None
    npu_runner = NPUPrefillGraphRunner(runner_no_tok, buckets=[128])
    assert npu_runner.sid_slices is None

    # 2. When tokenizer is patched/present
    with patch("vllm.tokenizers.get_tokenizer", return_value=mock_tok):
        runner_with_tok = MockModelRunner()
        runner_with_tok.model_config = SimpleNamespace(
            tokenizer="mock-model", trust_remote_code=False
        )
        gpu_runner_tok = GPUPrefillGraphRunner(runner_with_tok, buckets=[128])
        assert isinstance(gpu_runner_tok.sid_slices, SIDSlices)
        assert gpu_runner_tok.sid_slices.sid_a == (1000, 16)
        assert gpu_runner_tok.sid_slices.sid_b == (2000, 16)
        assert gpu_runner_tok.sid_slices.sid_c == (3000, 16)
        if gpu_runner_tok.p2d_pool is not None:
            assert gpu_runner_tok.p2d_pool.sid_slices == gpu_runner_tok.sid_slices

        npu_runner_tok = NPUPrefillGraphRunner(runner_with_tok, buckets=[128])
        assert isinstance(npu_runner_tok.sid_slices, SIDSlices)
        pool = npu_runner_tok._ensure_p2d_pool()
        assert pool.sid_slices == npu_runner_tok.sid_slices


def test_compute_sid_slices_merges_vocab_and_detects_non_contiguous() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        compute_sid_slices_from_tokenizer,
        resolve_sid_slices_from_runner,
    )

    # Merges get_vocab and get_added_vocab
    mock_tok = SimpleNamespace(
        get_vocab=lambda: {f"<s_a_{i}>": 100 + i for i in range(10)},
        get_added_vocab=lambda: (
            {f"<s_b_{i}>": 200 + i for i in range(10)} | {f"<s_c_{i}>": 300 + i for i in range(10)}
        ),
    )
    slices = compute_sid_slices_from_tokenizer(mock_tok)
    assert slices is not None
    assert slices.sid_a == (100, 10)
    assert slices.sid_b == (200, 10)
    assert slices.sid_c == (300, 10)

    # Non-contiguous SIDs return None
    non_contig_tok = SimpleNamespace(
        get_vocab=lambda: (
            {"<s_a_0>": 100, "<s_a_1>": 105}  # Gap!
            | {f"<s_b_{i}>": 200 + i for i in range(2)}
            | {f"<s_c_{i}>": 300 + i for i in range(2)}
        )
    )
    assert compute_sid_slices_from_tokenizer(non_contig_tok) is None

    # resolve_sid_slices_from_runner gracefully handles exceptions
    runner = SimpleNamespace(
        model_config=SimpleNamespace(tokenizer="error-tok", trust_remote_code=False)
    )
    with patch("vllm.tokenizers.get_tokenizer", side_effect=RuntimeError("tokenizer crash")):
        assert resolve_sid_slices_from_runner(runner) is None


def test_step0_and_decode_guards_fallback_when_slice_invalid() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        P2DBufferPool,
        P2DMetadata,
        SIDSlices,
        run_decode_passes,
        step0_select_topk,
    )

    # Slice size is 4, but beam_width is 8 -> size < beam_width fallback to full logits
    slices = SIDSlices(sid_a=(100, 4), sid_b=(200, 4), sid_c=(300, 4))
    pool = P2DBufferPool(
        max_batch=1,
        max_bucket=128,
        max_beams=8,
        num_layers=1,
        num_kv_heads=1,
        head_dim=16,
        vocab_size=500,
        device=torch.device("cpu"),
        sid_slices=slices,
    )

    logits_0 = torch.zeros(1, 500)
    # Highest logit is at index 10 (outside sid_a [100, 104))
    logits_0[0, 10] = 50.0
    step0_select_topk(logits_0=logits_0, num_reqs=1, beam_width=8, pool=pool)
    # Since beam_width (8) > slice_size (4), fallback to unsliced was used and no offset added
    assert pool.tokens_0[0, 0].item() == 10

    # Also test truncated logits where logits.shape[-1] < offset + size
    small_logits = torch.zeros(1, 50)  # vocab_size 50 < offset 100
    small_logits[0, 5] = 50.0
    step0_select_topk(logits_0=small_logits, num_reqs=1, beam_width=2, pool=pool)
    # Fallback to unsliced
    assert pool.tokens_0[0, 0].item() == 5

    # Test decode passes with beam_width > slice_size
    class MockModel:
        def __call__(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            return torch.zeros((input_ids.shape[0], 16))

        def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
            B = hidden.shape[0]
            logits = torch.zeros((B, 500))
            logits[:, 15] = 100.0  # Outside sid_b/c slices
            return logits

    runner = SimpleNamespace(model=MockModel())

    @contextlib.contextmanager
    def mock_forward_ctx(r: Any, meta: Any, tokens: int) -> Iterator[None]:
        yield

    attn_meta = P2DMetadata(
        p2d_pool=pool,
        p2d_pass=1,
        bucket=128,
        num_reqs=1,
        beam_width=8,
        total_prompt_tokens=128,
    )
    run_decode_passes(
        runner=runner,
        p2d_pool=pool,
        attn_meta_1=attn_meta,
        attn_meta_2=attn_meta,
        total_beam_tokens=8,
        num_reqs=1,
        beam_width=8,
        num_layers=1,
        num_kv_heads=1,
        head_dim=16,
        forward_ctx_fn=mock_forward_ctx,
    )
    # Unshifted token 15 chosen
    assert pool.tokens_1[0, 0].item() == 15
    assert pool.tokens_2[0, 0].item() == 15


def test_is_p2d_eligible_gates_out_sliced_logits() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import is_p2d_eligible

    runner = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        input_batch=SimpleNamespace(
            req_id_to_index={"r0": 0},
            num_computed_tokens_cpu=[0],
            num_prompt_tokens=[64],
        ),
    )
    sched_output = SimpleNamespace(
        scheduled_new_reqs=[],
        num_scheduled_tokens={"r0": 64},
        scheduled_spec_decode_tokens=None,
        scheduled_encoder_inputs=None,
    )

    with patch("vllm_gr.ops.sliced_logits.is_vocab_slice_enabled", return_value=True):
        assert (
            is_p2d_eligible(
                runner=runner,
                scheduler_output=sched_output,
                num_reqs=1,
                total_tokens=64,
                bucket=128,
                beam_width=4,
            )
            is False
        )

    with patch("vllm_gr.ops.sliced_logits.is_vocab_slice_enabled", return_value=False):
        assert (
            is_p2d_eligible(
                runner=runner,
                scheduler_output=sched_output,
                num_reqs=1,
                total_tokens=64,
                bucket=128,
                beam_width=4,
            )
            is True
        )


def test_monolithic_constants_and_shared_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        DEFAULT_P2D_BUCKETS,
        MONOLITHIC_MAX_PAD_RATIO,
        MONOLITHIC_SUPPORTED_BUCKETS,
        get_default_buckets,
    )

    assert DEFAULT_P2D_BUCKETS == [128, 256, 512, 768, 1024]
    assert MONOLITHIC_SUPPORTED_BUCKETS == (128, 256, 512, 768, 1024)
    assert MONOLITHIC_MAX_PAD_RATIO == 0.10

    # The shared bucket list applies on both platforms (no NPU-specific
    # default anymore); the env override still wins.
    monkeypatch.delenv("VLLM_GR_P2D_BUCKETS", raising=False)
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: True)
    assert get_default_buckets() == DEFAULT_P2D_BUCKETS
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: False)
    assert get_default_buckets() == DEFAULT_P2D_BUCKETS
    monkeypatch.setenv("VLLM_GR_P2D_BUCKETS", "256, 512")
    assert get_default_buckets() == [256, 512]


def test_base_runner_has_monolithic_graph_stub() -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import BasePrefillGraphRunner

    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=128),
        device=torch.device("cpu"),
    )
    base = BasePrefillGraphRunner(runner)
    assert base.has_monolithic_graph(512, 1, 128) is False
    assert base.has_monolithic_graph(128, 2, 32) is False


def test_get_p2d_supported_num_reqs_env_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from vllm_gr.v1.worker.prefill_graph_common import (
        P2DBufferPool,
        get_p2d_supported_num_reqs,
        is_p2d_eligible,
    )

    monkeypatch.setenv("VLLM_GR_P2D_BATCH_SIZES", "1,2,4")
    assert get_p2d_supported_num_reqs() == [1, 2, 4]

    # A 4-request pool sizes its beam buffers for 4 requests.
    pool = P2DBufferPool(
        max_batch=max(get_p2d_supported_num_reqs()),
        max_bucket=512,
        max_beams=32,
        num_layers=1,
        device=torch.device("cpu"),
    )
    assert pool.max_batch == 4
    assert pool.positions_1.shape[0] == 4 * 32

    # A 4-request batch passes eligibility when the env allows it.
    sched_out = SimpleNamespace(
        num_scheduled_tokens={"r1": 100, "r2": 100, "r3": 100, "r4": 100},
        scheduled_new_reqs=[SimpleNamespace(sampling_params=SimpleNamespace(beam_width=32))],
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )
    runner_ns = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"r1": 0, "r2": 1, "r3": 2, "r4": 3},
            num_computed_tokens_cpu=[0, 0, 0, 0],
            num_prompt_tokens=[100, 100, 100, 100],
            req_ids=["r1", "r2", "r3", "r4"],
        ),
        vllm_config=SimpleNamespace(additional_config={}),
        _prefill_graph_runner=SimpleNamespace(max_bucket=512),
    )
    assert (
        is_p2d_eligible(
            runner_ns,
            sched_out,
            num_reqs=4,
            total_tokens=400,
            bucket=None,
            beam_width=32,
        )
        is True
    )


def test_sliced_logits_with_sid_slices_p2d_equivalence() -> None:
    """Verifies that P2D functions identically with sliced vs full logits under SID slices."""
    from vllm_gr.v1.worker.prefill_graph_common import (
        P2DBufferPool,
        SIDSlices,
        _slice_logits_for_topk,
        is_p2d_eligible,
        step0_select_topk,
    )

    sid_slices = SIDSlices(sid_a=(1000, 100), sid_b=(1100, 100), sid_c=(1200, 100))
    window_off = 1000
    window_sz = 300

    # 1. Eligibility gate check
    runner_mock = SimpleNamespace(
        sid_slices=sid_slices,
        model_runner=SimpleNamespace(sid_slices=sid_slices),
        input_batch=SimpleNamespace(
            req_id_to_index={"r1": 0},
            num_computed_tokens_cpu=[0],
            num_prompt_tokens=[64],
            req_ids=["r1"],
        ),
        vllm_config=SimpleNamespace(additional_config={}),
        _prefill_graph_runner=SimpleNamespace(max_bucket=512, sid_slices=sid_slices),
    )
    sched_mock = SimpleNamespace(
        num_scheduled_tokens={"r1": 64},
        scheduled_new_reqs=[SimpleNamespace(sampling_params=SimpleNamespace(beam_width=4))],
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )

    with patch("vllm_gr.ops.sliced_logits.is_vocab_slice_enabled", return_value=True):
        assert (
            is_p2d_eligible(
                runner=runner_mock,
                scheduler_output=sched_mock,
                num_reqs=1,
                total_tokens=64,
                bucket=128,
                beam_width=4,
            )
            is True
        )

    # 2. Bitwise equivalence for step 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_reqs = 2
    beam_width = 8

    # Random slice data for the 300 tokens
    torch.manual_seed(42)
    slice_data = torch.randn((num_reqs, window_sz), dtype=torch.float32, device=device)

    # Full logits: 2000 cols with slice_data embedded at [1000:1300]
    logits_full = torch.full((num_reqs, 2000), -100.0, dtype=torch.float32, device=device)
    logits_full[:, window_off : window_off + window_sz].copy_(slice_data)

    logits_sliced = slice_data.clone()

    pool_full = P2DBufferPool(
        max_batch=num_reqs,
        max_bucket=64,
        max_beams=beam_width,
        vocab_size=2000,
        dtype=torch.float32,
        device=device,
        sid_slices=sid_slices,
    )
    pool_sliced = P2DBufferPool(
        max_batch=num_reqs,
        max_bucket=64,
        max_beams=beam_width,
        vocab_size=window_sz,
        dtype=torch.float32,
        device=device,
        sid_slices=sid_slices,
    )

    pool_full.logits_0[:num_reqs].copy_(logits_full)
    pool_sliced.logits_0[:num_reqs].copy_(logits_sliced)

    with patch("vllm_gr.ops.sliced_logits.is_vocab_slice_enabled", return_value=False):
        step0_select_topk(pool_full.logits_0, num_reqs, beam_width, pool_full)

    with (
        patch("vllm_gr.ops.sliced_logits.is_vocab_slice_enabled", return_value=True),
        patch(
            "vllm_gr.ops.sliced_logits.get_vocab_slice_config",
            return_value=(window_off, window_sz, None),
        ),
    ):
        step0_select_topk(pool_sliced.logits_0, num_reqs, beam_width, pool_sliced)

    assert torch.equal(
        pool_full.tokens_0[:num_reqs, :beam_width], pool_sliced.tokens_0[:num_reqs, :beam_width]
    )
    assert torch.allclose(
        pool_full.cum_lps_0[:num_reqs, :beam_width], pool_sliced.cum_lps_0[:num_reqs, :beam_width]
    )

    # 3. Equivalence for _slice_logits_for_topk (step 1 & 2)
    for sid_level in (sid_slices.sid_b, sid_slices.sid_c):
        with patch("vllm_gr.ops.sliced_logits.is_vocab_slice_enabled", return_value=False):
            res_full, off_full = _slice_logits_for_topk(logits_full, sid_level, beam_width)

        with (
            patch("vllm_gr.ops.sliced_logits.is_vocab_slice_enabled", return_value=True),
            patch(
                "vllm_gr.ops.sliced_logits.get_vocab_slice_config",
                return_value=(window_off, window_sz, None),
            ),
        ):
            res_sliced, off_sliced = _slice_logits_for_topk(logits_sliced, sid_level, beam_width)

        assert off_full == off_sliced == sid_level[0]
        assert torch.equal(res_full, res_sliced)


def test_npu_prefill_graph_runner_sliced_logits_pool_sizing() -> None:
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner

    runner = MockModelRunner()
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[512])

    with (
        patch("vllm_gr.ops.sliced_logits.is_vocab_slice_enabled", return_value=True),
        patch(
            "vllm_gr.ops.sliced_logits.get_vocab_slice_config", return_value=(151669, 24576, None)
        ),
    ):
        pool = npu_runner._ensure_p2d_pool()
        assert pool.vocab_size == 24576
        assert pool.logits_0.shape[-1] == 24576
        assert pool.logits_1.shape[-1] == 24576
        assert pool.logits_2.shape[-1] == 24576


def test_check_p2d_vocab_size_asserts_on_mismatch() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import check_p2d_vocab_size

    pool = SimpleNamespace(vocab_size=200)
    logits_mismatch = torch.zeros((1, 100))
    with pytest.raises(
        AssertionError,
        match="P2D test-error error: compute_logits returned width 100 but p2d_pool.vocab_size is 200",
    ):
        check_p2d_vocab_size(pool, logits_mismatch, "test-error")

    logits_match = torch.zeros((1, 200))
    # Should not raise
    check_p2d_vocab_size(pool, logits_match, "test-match")

    # Fallback to logits_0.shape[-1] if vocab_size is not set
    pool_no_vocab_attr = SimpleNamespace(logits_0=torch.zeros((1, 150)))
    check_p2d_vocab_size(pool_no_vocab_attr, torch.zeros((1, 150)), "test-logits-0")
    with pytest.raises(
        AssertionError, match="compute_logits returned width 100 but p2d_pool.vocab_size is 150"
    ):
        check_p2d_vocab_size(pool_no_vocab_attr, logits_mismatch, "test-logits-0-mismatch")


def test_sliced_logits_fallback_adds_win_off() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        P2DBufferPool,
        SIDSlices,
        _slice_logits_for_topk,
        step0_select_topk,
    )

    # When slicing is enabled with window [100, 150)
    window_off = 100
    window_sz = 50

    with (
        patch("vllm_gr.ops.sliced_logits.is_vocab_slice_enabled", return_value=True),
        patch(
            "vllm_gr.ops.sliced_logits.get_vocab_slice_config",
            return_value=(window_off, window_sz, None),
        ),
    ):
        logits = torch.randn((1, window_sz))

        # 1. sid_level outside window (e.g. [50, 60)) -> falls back to local window with offset = win_off
        out_slice, out_offset = _slice_logits_for_topk(logits, (50, 10), k=4)
        assert out_offset == window_off
        assert out_slice.shape == logits.shape

        # 2. sid_level fitting inside window (e.g. [110, 120)) -> uses sid slice with offset = 110
        in_slice, in_offset = _slice_logits_for_topk(logits, (110, 10), k=4)
        assert in_offset == 110
        assert in_slice.shape == (1, 10)

        # 3. step0_select_topk with sid_a outside window -> falls back to window with offset = win_off
        pool = P2DBufferPool(
            max_batch=1,
            max_bucket=16,
            max_beams=4,
            vocab_size=window_sz,
            dtype=torch.float32,
            device="cpu",
            sid_slices=SIDSlices(sid_a=(50, 10), sid_b=(110, 10), sid_c=(130, 10)),
        )
        pool.logits_0[:1].zero_()
        pool.logits_0[0, 5] = 10.0  # index 5 within window
        step0_select_topk(pool.logits_0, num_reqs=1, beam_width=1, pool=pool)
        # Expected token: 5 + win_off (100) = 105
        assert pool.tokens_0[0, 0].item() == 105


def test_is_p2d_eligible_gates_out_when_window_does_not_contain_sid() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import SIDSlices, is_p2d_eligible

    sched_out = SimpleNamespace(
        num_scheduled_tokens={"req_1": 10},
        scheduled_new_reqs=[SimpleNamespace(sampling_params=SimpleNamespace(beam_width=4))],
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req_1": 0},
            num_computed_tokens_cpu=[0],
            req_ids=["req_1"],
        ),
        vllm_config=SimpleNamespace(additional_config={}),
        sid_slices=SIDSlices(
            sid_a=(100, 10), sid_b=(120, 10), sid_c=(150, 10)
        ),  # SID range [100, 160)
        captured_monolithic_graphs={(1, 4): {16: object()}},
    )

    # Window [100, 140) does not cover up to sid_c end (160) -> should be ineligible
    with (
        patch("vllm_gr.ops.sliced_logits.is_vocab_slice_enabled", return_value=True),
        patch("vllm_gr.ops.sliced_logits.get_vocab_slice_config", return_value=(100, 40, None)),
    ):
        assert (
            is_p2d_eligible(runner, sched_out, num_reqs=1, total_tokens=10, bucket=16, beam_width=4)
            is False
        )

    # Window [100, 160) covers [100, 160) -> eligible
    with (
        patch("vllm_gr.ops.sliced_logits.is_vocab_slice_enabled", return_value=True),
        patch("vllm_gr.ops.sliced_logits.get_vocab_slice_config", return_value=(100, 60, None)),
    ):
        assert (
            is_p2d_eligible(runner, sched_out, num_reqs=1, total_tokens=10, bucket=16, beam_width=4)
            is True
        )


class FakeConstraintBackend:
    """Minimal capability-compatible stand-in for CUDAConstraintBackend.

    Emulates the constrained Top-K contract: writes global catalog token IDs
    and validity masks into caller-owned buffers. Candidate continuations are
    derived from the prefix so step-1/step-2 prefix construction is verifiable.
    """

    def __init__(
        self,
        first_tokens: list[int],
        degree_1: int = 3,
        degree_2: int = 3,
        max_top_k: int = 64,
        max_rows: int = 64,
    ) -> None:
        self.first_tokens = first_tokens
        self.degree_1 = degree_1
        self.degree_2 = degree_2
        self.is_ready = True
        self._invoke = None
        self.info = SimpleNamespace(max_top_k=max_top_k, max_rows=max_rows)
        self.calls: list[tuple[int, torch.Tensor]] = []

    def _candidates(self, step: int, prefixes: torch.Tensor, row: int) -> list[int]:
        if step == 0:
            return list(self.first_tokens)
        if step == 1:
            t0 = int(prefixes[row, 0])
            return [t0 * 10 + i for i in range(1, self.degree_1 + 1)]
        t0 = int(prefixes[row, 0])
        t1 = int(prefixes[row, 1])
        return [t0 * 100 + t1 * 10 + i for i in range(1, self.degree_2 + 1)]

    def select_out_unchecked(
        self,
        logits: Any,
        metadata: Any,
        *,
        out_token_ids: Any,
        out_logprobs: Any,
        out_valid_mask: Any,
        out_valid_counts: Any,
    ) -> None:
        rows, top_k = out_token_ids.shape
        prefixes = metadata.prefix_token_ids
        self.calls.append((metadata.current_step, prefixes.clone()))
        for r in range(rows):
            cand = self._candidates(metadata.current_step, prefixes, r)
            n = min(len(cand), top_k)
            out_valid_mask[r, :n] = True
            out_valid_mask[r, n:] = False
            out_token_ids[r, :n] = torch.tensor(cand[:n], dtype=torch.int32)
            out_logprobs[r, :n] = 1.0 + 0.1 * r
            out_logprobs[r, n:] = -1.0e20
            out_valid_counts[r] = n


def _make_constrained_pool(**kwargs: Any) -> Any:
    from vllm_gr.v1.worker.prefill_graph_common import P2DBufferPool

    return P2DBufferPool(
        max_batch=kwargs.get("max_batch", 2),
        max_bucket=kwargs.get("max_bucket", 64),
        max_beams=kwargs.get("max_beams", 8),
        vocab_size=kwargs.get("vocab_size", 256),
        dtype=torch.float32,
        device=torch.device("cpu"),
        constraint_backend=kwargs.get("backend"),
        num_layers=kwargs.get("num_layers", 2),
        num_kv_heads=kwargs.get("num_kv_heads", 1),
        head_dim=kwargs.get("head_dim", 8),
    )


def test_constrained_step_functions_end_to_end() -> None:
    """Constrained step0/1/2 run end-to-end with global IDs and no add-back."""
    from vllm_gr.v1.worker.prefill_graph_common import (
        step0_select_topk,
        step1_prune_and_kv_gather,
        step2_prune_and_backtrack,
    )

    backend = FakeConstraintBackend(first_tokens=[101, 102, 103, 104])
    pool = _make_constrained_pool(backend=backend)
    num_reqs = 2
    w0 = w1 = w2 = 4

    torch.manual_seed(7)
    pool.logits_0[:num_reqs].copy_(torch.randn(num_reqs, 256))

    step0_select_topk(pool.logits_0, num_reqs, w0, pool)
    assert backend.calls[0][0] == 0
    # Global catalog IDs land directly in tokens_0 (no offset add-back).
    assert pool.tokens_0[0, :w0].tolist() == [101, 102, 103, 104]
    assert torch.isfinite(pool.cum_lps_0[:num_reqs, :w0]).all()

    pool.logits_1[: num_reqs * w0].copy_(torch.randn(num_reqs * w0, 256))
    step1_prune_and_kv_gather(pool, num_reqs, w0, w1, num_layers=2, num_kv_heads=1, head_dim=8)
    assert all(call[0] == 1 for call in backend.calls[1 : 1 + num_reqs])
    # Per-request launch rows = w0; prefixes[:, 0] carries the step-0 roots.
    for r in range(num_reqs):
        assert backend.calls[1 + r][1][:, 0].tolist() == [101, 102, 103, 104]
    # Global tokens_1, parents within [0, w0), KV reordered into w1 rows.
    assert torch.all(pool.tokens_1[:num_reqs, :w1] > 100)
    assert pool.parents_1[:num_reqs, :w1].max() < w0
    assert pool.input_ids_2[: num_reqs * w1].tolist() == (
        pool.tokens_1[:num_reqs, :w1].reshape(-1).tolist()
    )

    pool.logits_2[: num_reqs * w1].copy_(torch.randn(num_reqs * w1, 256))
    step2_prune_and_backtrack(pool, num_reqs, w0, w1, w2)
    step2_calls = backend.calls[1 + num_reqs :]
    assert all(call[0] == 2 for call in step2_calls)
    # Step-2 prefix col 0 must be the gathered step-0 root, col 1 the step-1 token.
    for r in range(num_reqs):
        expect_t0 = pool.tokens_0[r, pool.parents_1[r, :w1]].tolist()
        expect_t1 = pool.tokens_1[r, :w1].tolist()
        got = step2_calls[r][1]
        assert got[:, 0].tolist() == expect_t0
        assert got[:, 1].tolist() == expect_t1
    # Backtracking consistency: final root/step-1 tokens match the parents.
    for r in range(num_reqs):
        for b in range(w2):
            p2 = int(pool.parents_2[r, b])
            assert pool.final_tokens[r, b, 0].item() == (
                pool.tokens_0[r, pool.parents_1[r, p2]].item()
            )
            assert pool.final_tokens[r, b, 1].item() == pool.tokens_1[r, p2].item()
    assert pool.valid_beam_counts[:num_reqs].tolist() == [w2, w2]

    # Per-token local scores telescope into the final cumulative, and the
    # step-0 score matches the root beam's cumulative at step 0.
    for r in range(num_reqs):
        for b in range(w2):
            scores = pool.step_scores[r, b].tolist()
            assert sum(scores) == pytest.approx(pool.final_cum_lps[r, b].item(), abs=1e-4)
            root = int(pool.parents_1[r, int(pool.parents_2[r, b])])
            assert scores[0] == pytest.approx(pool.cum_lps_0[r, root].item(), abs=1e-4)


def test_constrained_step0_invalid_roots_masked() -> None:
    """Fewer catalog roots than the beam width: extra slots are -inf."""
    from vllm_gr.v1.worker.prefill_graph_common import step0_select_topk

    backend = FakeConstraintBackend(first_tokens=[101, 102])
    pool = _make_constrained_pool(backend=backend)
    pool.logits_0[:1].copy_(torch.randn(1, 256))
    step0_select_topk(pool.logits_0, num_reqs=1, beam_width=4, pool=pool)

    assert torch.isfinite(pool.cum_lps_0[0, :2]).all()
    assert not torch.isfinite(pool.cum_lps_0[0, 2:4]).any()
    assert pool.tokens_0[0, :2].tolist() == [101, 102]


def test_constrained_truncation_valid_beam_counts() -> None:
    """A sparse catalog yields valid_beam_counts < w2 for bookkeeping."""
    from vllm_gr.v1.worker.prefill_graph_common import (
        step0_select_topk,
        step1_prune_and_kv_gather,
        step2_prune_and_backtrack,
    )

    # w0=2 roots, each with 2 step-1 continuations, each with 1 completion:
    # total legal triples = 4 >= w2? w2=4 needs 4; make step-2 degree 1 and
    # w1=2 -> 2 legal completions < w2=4.
    backend = FakeConstraintBackend(first_tokens=[101, 102], degree_1=2, degree_2=1)
    pool = _make_constrained_pool(backend=backend, num_layers=1, head_dim=4)
    num_reqs = 1
    w0, w1, w2 = 2, 2, 4

    pool.logits_0[:num_reqs].copy_(torch.randn(num_reqs, 256))
    step0_select_topk(pool.logits_0, num_reqs, w0, pool)
    pool.logits_1[: num_reqs * w0].copy_(torch.randn(num_reqs * w0, 256))
    step1_prune_and_kv_gather(pool, num_reqs, w0, w1, num_layers=1, num_kv_heads=1, head_dim=4)
    pool.logits_2[: num_reqs * w1].copy_(torch.randn(num_reqs * w1, 256))
    step2_prune_and_backtrack(pool, num_reqs, w0, w1, w2)

    # 2 parents x 1 completion = 2 legal triples, all of them finite.
    assert pool.valid_beam_counts[0].item() == 2
    assert torch.isfinite(pool.final_cum_lps[0, :2]).all()
    assert not torch.isfinite(pool.final_cum_lps[0, 2:w2]).any()


def test_constrained_flat_slabs_are_contiguous() -> None:
    """The per-call candidate views must satisfy the Triton contiguity ABI."""
    pool = _make_constrained_pool(backend=FakeConstraintBackend(first_tokens=[101]))
    rows, top_k = 3, 5
    view = pool.cand_tokens_flat[: rows * top_k].view(rows, top_k)
    assert view.is_contiguous()
    assert pool.temperatures.numel() == 1
    # Column-limited 2D slices of the raw flat are what must be avoided.
    assert pool.cand_tokens_flat[: rows * top_k].is_contiguous()


def test_is_p2d_eligible_catalog_backend_gate() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import is_p2d_eligible

    sched_out = SimpleNamespace(
        num_scheduled_tokens={"req_1": 10},
        scheduled_new_reqs=[SimpleNamespace(sampling_params=SimpleNamespace(beam_width=4))],
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )

    def make_runner(*, catalog_path: Any = None, backend: Any = None) -> Any:
        return SimpleNamespace(
            catalog_path=catalog_path,
            constraint_backend=backend,
            model_runner=SimpleNamespace(constraint_backend=backend),
            input_batch=SimpleNamespace(
                req_id_to_index={"req_1": 0},
                num_computed_tokens_cpu=[0],
                num_prompt_tokens=[10],
                req_ids=["req_1"],
            ),
            vllm_config=SimpleNamespace(additional_config={}),
            sid_slices=None,
            _prefill_graph_runner=SimpleNamespace(
                max_bucket=512,
                p2d_pool=SimpleNamespace(constraint_backend=backend),
            ),
        )

    # Catalog configured without a ready worker backend -> gated out.
    assert (
        is_p2d_eligible(
            make_runner(catalog_path="triples.json"),
            sched_out,
            num_reqs=1,
            total_tokens=10,
            bucket=16,
            beam_width=4,
        )
        is False
    )

    # Catalog + ready backend -> allowed (beam_width legacy path).
    backend = FakeConstraintBackend(first_tokens=[101])
    assert (
        is_p2d_eligible(
            make_runner(catalog_path="triples.json", backend=backend),
            sched_out,
            num_reqs=1,
            total_tokens=10,
            bucket=16,
            beam_width=4,
        )
        is True
    )

    # Catalog + ready backend but the P2D pool is not bound to it (e.g. a
    # backend injected after capture) -> gated out; unconstrained graphs must
    # never replay for catalog batches.
    unbound = make_runner(catalog_path="triples.json", backend=backend)
    unbound._prefill_graph_runner.p2d_pool = SimpleNamespace(constraint_backend=None)
    assert (
        is_p2d_eligible(
            unbound,
            sched_out,
            num_reqs=1,
            total_tokens=10,
            bucket=16,
            beam_width=4,
        )
        is False
    )


def test_is_p2d_eligible_backend_capacity_gates() -> None:
    from vllm_gr.v1.worker.prefill_graph_common import StepBeamWidths, is_p2d_eligible

    sched_out = SimpleNamespace(
        num_scheduled_tokens={f"req_{i}": 10 for i in range(2)},
        scheduled_new_reqs=[
            SimpleNamespace(sampling_params=SimpleNamespace(beam_width=4)) for _ in range(2)
        ],
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )

    def make_runner(backend: Any) -> Any:
        return SimpleNamespace(
            catalog_path="triples.json",
            constraint_backend=backend,
            model_runner=SimpleNamespace(constraint_backend=backend),
            input_batch=SimpleNamespace(
                req_id_to_index={f"req_{i}": i for i in range(2)},
                num_computed_tokens_cpu=[0, 0],
                num_prompt_tokens=[10, 10],
                req_ids=["req_0", "req_1"],
            ),
            vllm_config=SimpleNamespace(additional_config={}),
            sid_slices=None,
            _prefill_graph_runner=SimpleNamespace(
                max_bucket=512,
                p2d_pool=SimpleNamespace(constraint_backend=backend),
            ),
        )

    # max_w > max_top_k -> ineligible.
    small_top_k = FakeConstraintBackend(first_tokens=[101], max_top_k=4)
    assert (
        is_p2d_eligible(
            make_runner(small_top_k),
            sched_out,
            num_reqs=1,
            total_tokens=10,
            bucket=16,
            step_bws=StepBeamWidths(8, 8, 8),
        )
        is False
    )

    # num_reqs > max_rows -> ineligible (step 0 batches all requests).
    small_rows = FakeConstraintBackend(first_tokens=[101], max_rows=1)
    assert (
        is_p2d_eligible(
            make_runner(small_rows),
            sched_out,
            num_reqs=2,
            total_tokens=20,
            bucket=32,
            step_bws=StepBeamWidths(4, 4, 4),
        )
        is False
    )

    # Capacity fits -> eligible.
    ok = FakeConstraintBackend(first_tokens=[101], max_top_k=8, max_rows=8)
    assert (
        is_p2d_eligible(
            make_runner(ok),
            sched_out,
            num_reqs=2,
            total_tokens=20,
            bucket=32,
            step_bws=StepBeamWidths(4, 4, 4),
        )
        is True
    )


def test_p2d_bookkeeping_truncates_dead_beams() -> None:
    """Invalid beams past valid_beam_counts are marked -1 for the frontend."""
    import numpy as np

    from vllm_gr.v1.worker.model_runner_common import _build_p2d_bookkeeping_result

    final_tokens = torch.tensor(
        [[[101, 201, 301], [102, 202, 302], [103, 203, 303], [104, 204, 304]]],
        dtype=torch.int64,
    )
    final_cum_lps = torch.tensor([[0.1, 0.2, -float("inf"), -float("inf")]], dtype=torch.float32)
    valid_counts = torch.tensor([2], dtype=torch.int32)

    token_ids_cpu = torch.zeros((1, 12), dtype=torch.int64)
    is_token_ids = torch.zeros((1, 12), dtype=torch.bool)

    class FakeRequest:
        def __init__(self) -> None:
            self.output_token_ids: list[int] = []

    input_batch = SimpleNamespace(
        token_ids_cpu=token_ids_cpu,
        is_token_ids=is_token_ids,
        req_ids=["req_1"],
        req_id_to_index={"req_1": 0},
        num_tokens_no_spec=[0],
    )

    runner = SimpleNamespace(
        input_batch=input_batch,
        requests={"req_1": FakeRequest()},
        use_async_scheduling=False,
        model_config=SimpleNamespace(eos_token_id=2),
    )

    logprobs_lists, sampled, *_ = _build_p2d_bookkeeping_result(
        runner, final_tokens, final_cum_lps, valid_counts, sampler_output=None
    )

    row = logprobs_lists.logprob_token_ids[0]
    assert row[:6].tolist() == [101, 201, 301, 102, 202, 302]
    assert row[6:].tolist() == [-1] * 6
    assert sampled == [[101]]
    # Dead beams also carry -inf scores (defense in depth).
    assert np.all(np.isneginf(logprobs_lists.logprobs[0][6:]))

    # Zero valid beams: the engine continuation token clamps to eos.
    zero_counts = torch.tensor([0], dtype=torch.int32)
    logprobs_lists, sampled, *_ = _build_p2d_bookkeeping_result(
        runner, final_tokens, final_cum_lps, zero_counts, sampler_output=None
    )
    assert logprobs_lists.logprob_token_ids[0].tolist() == [-1] * 12
    assert sampled == [[2]]


def test_p2d_bookkeeping_preserves_per_step_scores() -> None:
    """The sentinel flat carries true per-token scores, not the repeated cumulative."""
    import numpy as np

    from vllm_gr.v1.worker.model_runner_common import _build_p2d_bookkeeping_result

    final_tokens = torch.tensor([[[101, 201, 301], [102, 202, 302]]], dtype=torch.int64)
    final_cum_lps = torch.tensor([[0.9, -float("inf")]], dtype=torch.float32)
    valid_counts = torch.tensor([1], dtype=torch.int32)
    step_scores = torch.tensor(
        [[[0.1, 0.3, 0.5], [-float("inf"), -float("inf"), -float("inf")]]],
        dtype=torch.float32,
    )

    token_ids_cpu = torch.zeros((1, 12), dtype=torch.int64)
    is_token_ids = torch.zeros((1, 12), dtype=torch.bool)

    class FakeRequest:
        def __init__(self) -> None:
            self.output_token_ids: list[int] = []

    input_batch = SimpleNamespace(
        token_ids_cpu=token_ids_cpu,
        is_token_ids=is_token_ids,
        req_ids=["req_1"],
        req_id_to_index={"req_1": 0},
        num_tokens_no_spec=[0],
    )
    runner = SimpleNamespace(
        input_batch=input_batch,
        requests={"req_1": FakeRequest()},
        use_async_scheduling=False,
        model_config=SimpleNamespace(eos_token_id=2),
    )

    logprobs_lists, sampled, *_ = _build_p2d_bookkeeping_result(
        runner, final_tokens, final_cum_lps, valid_counts, None, step_scores
    )

    assert logprobs_lists.logprob_token_ids[0].tolist() == [101, 201, 301, -1, -1, -1]
    assert logprobs_lists.logprobs[0][:3].tolist() == pytest.approx([0.1, 0.3, 0.5])
    assert np.all(np.isneginf(logprobs_lists.logprobs[0][3:]))
    assert sampled == [[101]]


def test_p2d_kill_switch_skips_gpu_pool_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kill switch prevents the multi-GB P2D pool allocation at startup."""
    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    monkeypatch.setattr(prefill_graph_common, "P2DEnv", SimpleNamespace(disabled=True))
    gpu_runner = GPUPrefillGraphRunner(MockModelRunner(), buckets=[128])
    assert gpu_runner.p2d_pool is None
    # The standard prefill bucket buffers remain allocated (not P2D-specific).
    assert len(gpu_runner.buffers) == 1


def test_p2d_kill_switch_skips_gpu_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kill switch skips every P2D capture while capture_graphs runs."""
    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    monkeypatch.setattr(prefill_graph_common, "P2DEnv", SimpleNamespace(disabled=True))
    gpu_runner = GPUPrefillGraphRunner(MockModelRunner(), buckets=[128])
    gpu_runner.capture_graphs(SimpleNamespace())
    assert gpu_runner.captured_2d_graphs == {}
    assert gpu_runner.captured_decode_step_graphs == {}
    assert gpu_runner.p2d_pool is None
    assert all(buf.captured_graphs == {} for buf in gpu_runner.buffers.values())
    assert gpu_runner.has_2d_graph(1, 128) is False
    assert gpu_runner.has_decode_step_graph(1, 1, 128) is False


def test_p2d_kill_switch_skips_npu_p2d_captures(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kill switch skips NPU P2D captures and never allocates the pool."""
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner

    monkeypatch.setattr(prefill_graph_common, "P2DEnv", SimpleNamespace(disabled=True))
    npu_runner = NPUPrefillGraphRunner(MockModelRunner(), buckets=[128])

    def fail_pool() -> None:
        pytest.fail("P2D pool must not be created with the kill switch active")

    monkeypatch.setattr(npu_runner, "_ensure_p2d_pool", fail_pool)
    monkeypatch.setattr(
        "vllm_gr.v1.worker.npu_prefill_graph_runner.is_npu_available", lambda: False
    )
    npu_runner.capture_graphs(SimpleNamespace())
    assert npu_runner.monolithic_graphs == {}
    assert npu_runner.captured_2d_graphs == {}
    assert npu_runner.captured_decode_step_graphs == {}
    assert npu_runner.p2d_pool is None
    assert npu_runner.has_2d_graph(1, 32) is False
    assert npu_runner.has_decode_step_graph(1, 1, 32) is False
    assert npu_runner.has_monolithic_graph(128, 1, 32) is False


def test_resolve_p2d_pool_kv_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.worker.prefill_graph_common import (
        ENV_P2D_POOL_KV_BOUND,
        PREFILL_GRAPH_MAX_BUCKET_DEFAULT,
        resolve_p2d_pool_kv_bound,
    )

    monkeypatch.delenv(ENV_P2D_POOL_KV_BOUND, raising=False)
    # Default: the model context minus a two-token margin.
    assert resolve_p2d_pool_kv_bound(32768) == 32766
    # No context available -> the legacy bucket default.
    assert resolve_p2d_pool_kv_bound(0) == PREFILL_GRAPH_MAX_BUCKET_DEFAULT

    # Env override, clamped to the context.
    monkeypatch.setenv(ENV_P2D_POOL_KV_BOUND, "8192")
    assert resolve_p2d_pool_kv_bound(32768) == 8192
    monkeypatch.setenv(ENV_P2D_POOL_KV_BOUND, "100000")
    assert resolve_p2d_pool_kv_bound(32768) == 32766

    for bad in ("abc", "0", "-5"):
        monkeypatch.setenv(ENV_P2D_POOL_KV_BOUND, bad)
        with pytest.raises(ValueError, match="positive integer"):
            resolve_p2d_pool_kv_bound(32768)


def test_gpu_pool_kv_bound_follows_context_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner
    from vllm_gr.v1.worker.prefill_graph_common import ENV_P2D_POOL_KV_BOUND

    def make_runner(max_model_len: int) -> GPUPrefillGraphRunner:
        return GPUPrefillGraphRunner(
            SimpleNamespace(
                device=torch.device("cpu"),
                vllm_config=None,
                model_config=SimpleNamespace(max_model_len=max_model_len),
                scheduler_config=SimpleNamespace(max_num_batched_tokens=32768),
            ),
            buckets=[128],
        )

    monkeypatch.delenv(ENV_P2D_POOL_KV_BOUND, raising=False)
    assert make_runner(32768)._ensure_p2d_pool().max_bucket == 32766

    monkeypatch.setenv(ENV_P2D_POOL_KV_BOUND, "8192")
    assert make_runner(32768)._ensure_p2d_pool().max_bucket == 8192


def test_is_p2d_eligible_rejects_gr_stage_metadata() -> None:
    """beam_search_v1 stage batches carry gr_stage_metadata and must not enter P2D."""
    from vllm_gr.v1.worker.prefill_graph_common import StepBeamWidths, is_p2d_eligible

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"r1": 0},
            num_computed_tokens_cpu=[0],
            num_prompt_tokens=[128],
            block_table=[
                SimpleNamespace(
                    block_table=SimpleNamespace(gpu=torch.zeros((1, 8), dtype=torch.int32))
                )
            ],
        )
    )
    sp = SimpleNamespace(logprobs=128, max_tokens=1, flat_logprobs=True, temperature=0.0)
    sched_without_stage = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="r1", sampling_params=sp)],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"r1": 128},
        beam_data={},
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        gr_stage_metadata={},
    )
    assert (
        is_p2d_eligible(
            runner,
            sched_without_stage,
            num_reqs=1,
            total_tokens=128,
            step_bws=StepBeamWidths(128, 128, 128),
        )
        is True
    )

    sched_with_stage = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="r1", sampling_params=sp)],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={"r1": 128},
        beam_data={},
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        gr_stage_metadata={"r1": SimpleNamespace(kind="prefill")},
    )
    assert (
        is_p2d_eligible(
            runner,
            sched_with_stage,
            num_reqs=1,
            total_tokens=128,
            step_bws=StepBeamWidths(128, 128, 128),
        )
        is False
    )
