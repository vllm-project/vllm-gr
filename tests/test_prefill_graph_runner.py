# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BeamPrefillGraphRunner."""

from typing import Any

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


def test_match_bucket_standard() -> None:
    runner = MockModelRunner()
    graph_runner = BeamPrefillGraphRunner(runner, buckets=[512, 1024, 2048, 4096])

    # 400 tokens -> fits 512 (pad 112 tokens = 28% <= 512 pad tokens threshold)
    assert graph_runner.match_bucket(400) == 512

    # 750 tokens -> fits 1024 (pad 274 tokens <= 512 pad tokens threshold)
    assert graph_runner.match_bucket(750) == 1024

    # 2000 tokens -> fits 2048 (pad 48 tokens = 2.4% <= 20% threshold)
    assert graph_runner.match_bucket(2000) == 2048


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


def test_populate_buffers() -> None:
    runner = MockModelRunner()
    graph_runner = BeamPrefillGraphRunner(runner, buckets=[1024])

    input_ids = torch.tensor(list(range(500)), dtype=torch.int32)
    positions = torch.tensor(list(range(500)), dtype=torch.int32)
    slot_mapping = torch.tensor(list(range(500)), dtype=torch.int64)

    buf = graph_runner.populate_buffers(
        bucket=1024,
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        num_scheduled_tokens=[250, 250],
    )

    assert buf.input_ids_buf[:500].equal(input_ids)
    assert (buf.input_ids_buf[500:] == 0).all()
    assert buf.positions_buf[:500].equal(positions)
    assert buf.slot_mapping_buf[:500].equal(slot_mapping)
    assert (buf.slot_mapping_buf[500:] == -1).all()


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
    from vllm_gr.v1.worker.prefill_graph_runner import (
        ENV_ENABLE_PREFILL_GRAPH,
        is_prefill_graph_enabled,
    )

    # Default is enabled
    monkeypatch.delenv(ENV_ENABLE_PREFILL_GRAPH, raising=False)
    assert is_prefill_graph_enabled() is True

    # Explicitly disabled via the env var
    monkeypatch.setenv(ENV_ENABLE_PREFILL_GRAPH, "0")
    assert is_prefill_graph_enabled() is False

    monkeypatch.setenv(ENV_ENABLE_PREFILL_GRAPH, "false")
    assert is_prefill_graph_enabled() is False

    # Explicitly enabled
    monkeypatch.setenv(ENV_ENABLE_PREFILL_GRAPH, "1")
    assert is_prefill_graph_enabled() is True

    monkeypatch.setenv(ENV_ENABLE_PREFILL_GRAPH, "true")
    assert is_prefill_graph_enabled() is True


def test_npu_gates_allow_prefill_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: True)
    from vllm_gr.v1.worker.prefill_graph_runner import (
        ENV_ENABLE_PREFILL_GRAPH,
        is_prefill_graph_enabled,
        is_prefill_phase_batch,
    )

    # NPU prefill graphs follow the env flag (task-group capture is supported).
    monkeypatch.setenv(ENV_ENABLE_PREFILL_GRAPH, "1")
    assert is_prefill_graph_enabled() is True

    monkeypatch.setenv(ENV_ENABLE_PREFILL_GRAPH, "0")
    assert is_prefill_graph_enabled() is False

    # Enabled with default (unset) env.
    monkeypatch.delenv(ENV_ENABLE_PREFILL_GRAPH, raising=False)
    assert is_prefill_graph_enabled() is True

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


def test_dynamic_thresholds() -> None:
    runner = MockModelRunner()
    graph_runner = BeamPrefillGraphRunner(
        runner,
        buckets=[16, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5192],
    )

    # 1. Short prompt (<= 256 tokens): generous padding allowed (up to 35% or 256 tokens)
    # 200 tokens -> fits 256 (pad 56 = 28% <= 35%)
    assert graph_runner.match_bucket(200) == 256

    # 2. Medium prompt (257..1024 tokens): up to 20% or 256 tokens
    # 850 tokens -> fits 1024 (pad 174 = 20.4% but <= 256 max_pad_tokens)
    assert graph_runner.match_bucket(850) == 1024

    # 3. Transition prompt (1025..2048 tokens): up to 5% or 100 tokens
    # 1500 tokens -> fits 1536 (pad 36 = 2.4% <= 5%)
    assert graph_runner.match_bucket(1500) == 1536
    # 1100 tokens trying to match 1536 (pad 436 tokens = 39.6% > 5% and > 100 tokens) -> Fallback None
    assert graph_runner.match_bucket(1100) is None

    # 4. Strictly compute-bound (> 2048 tokens): up to 2% or 64 tokens
    # 3550 tokens -> fits 3584 (pad 34 tokens <= 64 tokens)
    assert graph_runner.match_bucket(3550) == 3584
    # 3200 tokens trying to match 3584 (pad 384 tokens = 12% > 2% and > 64 tokens) -> Fallback None
    assert graph_runner.match_bucket(3200) is None


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
    assert 8192 not in graph_runner.buckets

    # Runner without the config field: buckets unchanged (no-op).
    graph_runner = BeamPrefillGraphRunner(MockModelRunner(), buckets=[256, 1024])
    assert graph_runner.buckets == [256, 1024]


def test_platform_bucket_and_threshold_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker.prefill_graph_runner import (
        DEFAULT_PREFILL_BUCKETS,
        ENV_PREFILL_GRAPH_BUCKET_STEP,
        ENV_PREFILL_GRAPH_MAX_BUCKET,
        NPU_DEFAULT_PREFILL_BUCKETS,
        PREFILL_GRAPH_BUCKET_STEP_DEFAULT,
        PREFILL_GRAPH_MAX_BUCKET_DEFAULT,
        get_default_buckets,
        get_dynamic_thresholds,
    )

    # GPU (default fixture forces non-NPU): configured default buckets and the
    # GPU threshold table. Derive the expected sequence so changing the default
    # bucket geometry does not require updating this test.
    expected_default_buckets = list(
        range(
            PREFILL_GRAPH_BUCKET_STEP_DEFAULT,
            PREFILL_GRAPH_MAX_BUCKET_DEFAULT + 1,
            PREFILL_GRAPH_BUCKET_STEP_DEFAULT,
        )
    )
    assert get_default_buckets() == DEFAULT_PREFILL_BUCKETS
    assert get_default_buckets() == expected_default_buckets
    assert get_dynamic_thresholds(256) == (0.35, 512)
    assert get_dynamic_thresholds(700) == (0.30, 300)
    assert get_dynamic_thresholds(1500) == (0.05, 100)
    assert get_dynamic_thresholds(3000) == (0.02, 64)

    # NPU: the default bucket geometry is shared with GPU; <=512 tier is the
    # same as GPU and the <=1024 padding threshold is tightened to (0.1, 10).
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: True)
    assert get_default_buckets() == NPU_DEFAULT_PREFILL_BUCKETS
    assert get_default_buckets() == expected_default_buckets
    assert get_dynamic_thresholds(256) == (0.35, 512)
    assert get_dynamic_thresholds(700) == (0.1, 10)
    assert get_dynamic_thresholds(1024) == (0.1, 10)

    monkeypatch.setenv(ENV_PREFILL_GRAPH_BUCKET_STEP, "256")
    monkeypatch.setenv(ENV_PREFILL_GRAPH_MAX_BUCKET, "1024")
    assert get_default_buckets() == [256, 512, 768, 1024]


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


def test_gpu_replay_falls_back_when_kvlen_exceeds_bound() -> None:
    from types import SimpleNamespace

    import numpy as np

    from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            req_id_to_index={"rA": 0, "rB": 1},
            num_computed_tokens_cpu=np.array([1000, 0]),
        ),
    )
    gpu_runner = GPUPrefillGraphRunner(runner, buckets=[512])
    buf = gpu_runner.buffers[512]
    buf.captured_graphs[2] = object()
    buf.output_bufs[2] = torch.zeros(512, 8, 64)
    buf.max_seq_len_bounds[2] = 1024  # 2 x bucket

    # A cached request whose KV length (1000 + 64) exceeds the bound must
    # fall back to eager before touching the buffers.
    assert (
        gpu_runner.execute_model_prefill(
            bucket=512,
            input_ids=torch.zeros(64, dtype=torch.int32),
            positions=torch.zeros(64, dtype=torch.int64),
            slot_mapping=torch.zeros(64, dtype=torch.int64),
            num_scheduled_tokens={"rA": 64, "rB": 30},
        )
        is None
    )

    # Within the bound, the replay proceeds (no-op fake graph).
    class FakeGraph:
        def replay(self) -> None:
            pass

    runner.input_batch.num_computed_tokens_cpu = np.array([500, 0])
    buf.captured_graphs[2] = FakeGraph()
    assert (
        gpu_runner.execute_model_prefill(
            bucket=512,
            input_ids=torch.zeros(94, dtype=torch.int32),
            positions=torch.zeros(94, dtype=torch.int64),
            slot_mapping=torch.zeros(94, dtype=torch.int64),
            num_scheduled_tokens={"rA": 64, "rB": 30},
        )
        is buf.output_bufs[2]
    )


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
