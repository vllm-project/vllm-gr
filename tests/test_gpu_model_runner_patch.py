# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for GPU ModelRunner wrappers."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.config import GRConfig
from vllm_gr.v1.engine.beam_search_v1 import GRStageKind
from vllm_gr.v1.worker import gpu_model_runner_patch as gpu_patch
from vllm_gr.v1.worker import model_runner_common as common


def _prefill_stage(*request_ids: str) -> SimpleNamespace:
    prompt_tokens = (4,) * len(request_ids)
    return SimpleNamespace(
        session_id=request_ids[0],
        dispatch_id=0,
        stage_index=0,
        kind=GRStageKind.PREFILL,
        decode_step=None,
        owner_generation=0,
        request_ids=request_ids,
        num_prompt_tokens_by_request=prompt_tokens,
        num_prompt_tokens=sum(prompt_tokens),
        beam_params=SimpleNamespace(beam_width=4),
        produces_output=True,
    )


def test_async_scheduling_decode_bypasses_prefill_graph_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = SimpleNamespace(
        gr_stage_metadata={"request": SimpleNamespace(kind=GRStageKind.DECODE)}
    )
    runner = SimpleNamespace(_current_scheduler_output=output)
    tokens, positions, hidden = object(), object(), object()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Asynchronous scheduling entered the legacy bucket Prefill graph")

    def original(self: Any, input_ids: Any, positions: Any) -> Any:
        assert self is runner and input_ids is tokens
        return hidden

    monkeypatch.setattr(gpu_patch, "is_prefill_graph_enabled", forbidden)
    assert gpu_patch.make_patched_model_forward(original)(runner, tokens, positions) is hidden


@pytest.mark.parametrize(  # type: ignore[misc]
    "bucket, captured, replay",
    [(4, True, True), (8, True, True), (4, False, False), (None, False, False)],
)
def test_async_scheduling_prefill_reuses_existing_padded_graph_without_runtime_capture(
    monkeypatch: pytest.MonkeyPatch,
    bucket: int | None,
    captured: bool,
    replay: bool,
) -> None:
    tokens, positions = torch.zeros(4, dtype=torch.int32), torch.arange(4)
    hidden, eager = object(), object()
    calls: list[str] = []
    output = SimpleNamespace(
        gr_stage_metadata={"request": _prefill_stage("request")},
        num_scheduled_tokens={"request": 4},
        total_num_scheduled_tokens=4,
    )

    def execute_model_prefill(**kwargs: Any) -> Any:
        assert kwargs["input_ids"] is tokens
        assert kwargs["positions"] is positions
        assert kwargs["bucket"] == bucket
        calls.append("replay")
        return hidden

    prefill = SimpleNamespace(
        match_bucket=lambda _: bucket,
        has_captured_graph=lambda *args: captured,
        execute_model_prefill=execute_model_prefill,
    )
    runner = SimpleNamespace(
        _current_scheduler_output=output,
        _prefill_graph_runner=prefill,
        model_config=SimpleNamespace(enforce_eager=False),
        input_batch=SimpleNamespace(
            block_table=[SimpleNamespace(slot_mapping=SimpleNamespace(gpu=positions))]
        ),
    )

    def original(self: Any, input_ids: Any, positions: Any) -> Any:
        assert input_ids is tokens
        calls.append("eager")
        return eager

    monkeypatch.setattr(gpu_patch, "is_prefill_graph_enabled", lambda: True)
    monkeypatch.setattr(gpu_patch, "is_prefill_phase_batch", lambda *args: True)
    result = gpu_patch.make_patched_model_forward(original)(runner, tokens, positions)
    assert result is (hidden if replay else eager)
    assert calls == (["replay"] if replay else ["eager"])


@pytest.mark.parametrize(  # type: ignore[misc]
    "invalid", ["ids", "positions", "inputs_embeds", "intermediate_tensors"]
)
def test_async_prefill_padding_preserves_input_guards(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    def forbidden(**kwargs: Any) -> None:
        raise AssertionError("Invalid inputs reached graph replay")

    prefill = SimpleNamespace(
        match_bucket=lambda _: 8,
        has_captured_graph=lambda *args: True,
        execute_model_prefill=forbidden,
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        _prefill_graph_runner=prefill,
        _current_scheduler_output=SimpleNamespace(
            gr_stage_metadata={"request": _prefill_stage("request")},
            num_scheduled_tokens={"request": 4},
            total_num_scheduled_tokens=4,
        ),
    )
    monkeypatch.setattr(gpu_patch, "is_prefill_graph_enabled", lambda: True)
    monkeypatch.setattr(gpu_patch, "is_prefill_phase_batch", lambda *args: True)
    monkeypatch.setattr(gpu_patch, "get_prefill_graph_runner", lambda _: prefill)
    ids, positions = torch.zeros(4, dtype=torch.int32), torch.arange(4)
    kwargs: dict[str, Any] = {}
    if invalid == "ids":
        ids = ids[:3]
    elif invalid == "positions":
        positions = positions[None, :]
    else:
        kwargs[invalid] = object()
    wrapped = gpu_patch.make_patched_model_forward(lambda *a, **k: "eager")
    assert wrapped(runner, ids, positions, **kwargs) == "eager"


@pytest.mark.parametrize("enforce_eager", [True, False])  # type: ignore[misc]
def test_async_scheduling_prefill_never_initializes_a_graph_runner_during_requests(
    monkeypatch: pytest.MonkeyPatch,
    enforce_eager: bool,
) -> None:
    runner = SimpleNamespace(
        _current_scheduler_output=SimpleNamespace(gr_stage_metadata={"r": _prefill_stage("r")}),
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("request allocated a graph runner")

    monkeypatch.setattr(gpu_patch, "is_prefill_graph_enabled", lambda: True)
    monkeypatch.setattr(gpu_patch, "get_prefill_graph_runner", forbidden)
    assert gpu_patch.make_patched_model_forward(lambda *args, **kwargs: "eager")(runner) == "eager"


def test_batch_session_prefill_selects_exact_four_request_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_ids = ("r0", "r1", "r2", "r3")
    scheduled_tokens = dict.fromkeys(request_ids, 1)
    tokens = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4)
    hidden = object()
    calls: list[str] = []

    def has_captured_graph(bucket: int, num_reqs: int) -> bool:
        calls.append(f"select:{bucket}:{num_reqs}")
        return bucket == 4 and num_reqs == 4

    def execute_model_prefill(**kwargs: Any) -> object:
        assert tuple(kwargs["num_scheduled_tokens"]) == request_ids
        calls.append("replay")
        return hidden

    prefill = SimpleNamespace(
        match_bucket=lambda total: total,
        has_captured_graph=has_captured_graph,
        execute_model_prefill=execute_model_prefill,
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        _prefill_graph_runner=prefill,
        _current_scheduler_output=SimpleNamespace(
            gr_stage_metadata={"session": _prefill_stage(*request_ids)},
            num_scheduled_tokens=scheduled_tokens,
            total_num_scheduled_tokens=4,
        ),
        input_batch=SimpleNamespace(
            block_table=[SimpleNamespace(slot_mapping=SimpleNamespace(gpu=positions))]
        ),
    )

    monkeypatch.setattr(gpu_patch, "is_prefill_graph_enabled", lambda: True)
    monkeypatch.setattr(gpu_patch, "is_prefill_phase_batch", lambda *args: True)
    result = gpu_patch.make_patched_model_forward(lambda *args, **kwargs: "eager")(
        runner, tokens, positions
    )

    assert result is hidden
    assert calls == ["select:4:4", "replay"]


@pytest.mark.parametrize(  # type: ignore[misc]
    "scheduled_ids",
    [
        ("r0", "r1", "r2"),
        ("r1", "r0", "r2", "r3"),
        ("r0", "r1", "r2", "other"),
    ],
)
def test_batch_session_prefill_rejects_partial_or_reordered_requests(
    monkeypatch: pytest.MonkeyPatch,
    scheduled_ids: tuple[str, ...],
) -> None:
    request_ids = ("r0", "r1", "r2", "r3")
    tokens = torch.arange(len(scheduled_ids), dtype=torch.int32)
    positions = torch.arange(len(scheduled_ids))
    runner = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        _prefill_graph_runner=SimpleNamespace(),
        _current_scheduler_output=SimpleNamespace(
            gr_stage_metadata={"session": _prefill_stage(*request_ids)},
            num_scheduled_tokens=dict.fromkeys(scheduled_ids, 1),
            total_num_scheduled_tokens=len(scheduled_ids),
        ),
    )
    eager_calls: list[bool] = []

    def eager(*args: Any, **kwargs: Any) -> str:
        eager_calls.append(True)
        return "eager"

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid Batch Session reached graph selection")

    monkeypatch.setattr(gpu_patch, "is_prefill_graph_enabled", lambda: True)
    monkeypatch.setattr(gpu_patch, "is_prefill_phase_batch", lambda *args: True)
    monkeypatch.setattr(gpu_patch, "get_prefill_graph_runner", forbidden)
    result = gpu_patch.make_patched_model_forward(eager)(runner, tokens, positions)

    assert result == "eager"
    assert eager_calls == [True]


def test_gpu_prepare_preserves_order_and_two_item_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    spec_metadata = object()
    original_logits = object()
    input_batch = SimpleNamespace(req_ids=["request"])
    runner = SimpleNamespace(
        input_batch=input_batch,
        _current_beam_info={"req_ids": []},
        device=torch.device("cpu"),
    )

    def original(
        self: Any,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
    ) -> tuple[object, object]:
        assert self._current_beam_info["req_ids"] == []
        events.append("original")
        return original_logits, spec_metadata

    wrapped = gpu_patch.make_patched_prepare_inputs(original)
    assert gpu_patch.make_patched_prepare_inputs(wrapped) is wrapped

    ordinary = wrapped(runner, SimpleNamespace(beam_data={}), [1])
    assert ordinary == (original_logits, spec_metadata)
    assert events == ["original"]

    assert runner._current_scheduler_output.beam_data == {}
    assert runner._current_beam_info["req_ids"] is input_batch.req_ids


# ---------------------------------------------------------------------------
# _beam_full_graph_ready
# ---------------------------------------------------------------------------


def test_beam_full_graph_ready_returns_true_for_captured_width() -> None:
    runner = SimpleNamespace(_beam_graph_keys={4, 128})
    assert common._beam_full_graph_ready(runner, 4) is True
    assert common._beam_full_graph_ready(runner, 128) is True


def test_beam_full_graph_ready_returns_false_for_uncaptured_width() -> None:
    runner = SimpleNamespace(_beam_graph_keys={4})
    assert common._beam_full_graph_ready(runner, 8) is False


def test_beam_full_graph_ready_returns_false_when_no_keys() -> None:
    runner = SimpleNamespace()
    assert common._beam_full_graph_ready(runner, 4) is False

    runner2 = SimpleNamespace(_beam_graph_keys=None)
    assert common._beam_full_graph_ready(runner2, 4) is False

    runner3 = SimpleNamespace(_beam_graph_keys=set())
    assert common._beam_full_graph_ready(runner3, 4) is False


def test_beam_graph_runtime_stats_reports_capture_and_dispatch_state() -> None:
    config = SimpleNamespace(beam_graph_enabled=True, beam_max_width=128)
    runner = SimpleNamespace(
        _vllm_gr_config=config,
        _beam_graph_keys={128},
        _beam_graph_full_dispatch_count=2,
    )
    worker = SimpleNamespace(model_runner=runner)

    stats = common.get_beam_graph_runtime_stats(worker)

    assert stats == {
        "runner": "SimpleNamespace",
        "beam_graph_enabled": True,
        "beam_max_width": 128,
        "captured_widths": [128],
        "full_dispatches": 2,
        "gr_prefill_dispatches": {},
    }


def test_worker_graph_stats_hook_is_named_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.worker.gpu_worker as gpu_worker

    import vllm_gr.v1.worker.model_runner_patch as dispatcher

    class FakeWorker:
        pass

    monkeypatch.setattr(gpu_worker, "Worker", FakeWorker)
    monkeypatch.setattr(
        dispatcher, "_active_platform_module_name", lambda: dispatcher._GPU_PATCH_MODULE
    )
    monkeypatch.setattr(
        "vllm_gr.constraints.worker_inject.install_worker_resource_hook",
        lambda worker_cls: 0,
    )

    dispatcher._install_active_worker_resource_hook()
    first = getattr(FakeWorker, "get_gr_beam_graph_runtime_stats")
    dispatcher._install_active_worker_resource_hook()

    assert first is common.get_beam_graph_runtime_stats
    assert getattr(FakeWorker, "get_gr_beam_graph_runtime_stats") is first


@pytest.mark.parametrize("device", ["cuda", "cpu", "privateuseone"])  # type: ignore[misc]
@pytest.mark.parametrize("enabled", [None, False, True])  # type: ignore[misc]
def test_create_beam_context_uses_grconfig_params(device: str, enabled: bool | None) -> None:
    """_create_beam_context must pass gr_cfg.max_width/max_decode_steps."""
    from unittest.mock import MagicMock, patch

    gr_cfg = GRConfig(beam_max_width=64, beam_max_decode_steps=10)
    if enabled is not None:
        gr_cfg = replace(gr_cfg, enable_short_suffix_attention=enabled)
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_kv_heads=lambda pc: 8,
            get_head_size=lambda: 128,
            get_num_layers=lambda pc: 2,
            dtype=torch.float16,
        ),
        parallel_config=SimpleNamespace(),
        device=torch.device(device),
    )

    with patch("vllm_gr.v1.beam.beam_attention_context.BeamAttentionContext") as MockCtx:
        mock_instance = MagicMock()
        MockCtx.return_value = mock_instance
        common._create_beam_context(runner, gr_cfg, max_batch=1)
        MockCtx.assert_called_once()
        call_kwargs = MockCtx.call_args[1]
        # gr_cfg params must flow through directly
        assert call_kwargs["max_beams"] == 64
        assert call_kwargs["max_decode_steps"] == 10
        assert call_kwargs["dtype"] is torch.float16
        assert call_kwargs["max_batch"] == 1
        assert call_kwargs["enable_short_suffix_attention"] is (
            device == "cuda" and enabled is True
        )


def test_create_beam_context_dtype_from_model_config() -> None:
    """dtype must come from model_config.dtype, not hardcoded."""
    from unittest.mock import MagicMock, patch

    gr_cfg = GRConfig(beam_max_width=32, beam_max_decode_steps=5)
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_kv_heads=lambda pc: 4,
            get_head_size=lambda: 96,
            get_num_layers=lambda pc: 1,
            dtype=torch.bfloat16,
        ),
        parallel_config=SimpleNamespace(),
        device=torch.device("cpu"),
    )

    with patch("vllm_gr.v1.beam.beam_attention_context.BeamAttentionContext") as MockCtx:
        MockCtx.return_value = MagicMock()
        common._create_beam_context(runner, gr_cfg)
        call_kwargs = MockCtx.call_args[1]
        assert call_kwargs["dtype"] is torch.bfloat16
        assert call_kwargs["enable_short_suffix_attention"] is False


# ---------------------------------------------------------------------------
# Beam graph readiness uses the current Worker's registry.
# ---------------------------------------------------------------------------


def test_beam_full_graph_ready_uses_worker_registry() -> None:
    graph = SimpleNamespace(signature=SimpleNamespace(num_model_rows=8), ready=True)
    runner = SimpleNamespace(_beam_decode_graph=graph)
    assert common._beam_full_graph_ready(runner, 8)
    assert not common._beam_full_graph_ready(runner, 4)
    runner._beam_decode_graph.ready = False
    assert not common._beam_full_graph_ready(runner, 8)


def test_beam_full_graph_ready_ignores_other_native_graphs() -> None:
    from unittest.mock import MagicMock, patch

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    wrapper = SimpleNamespace(
        runtime_mode=CUDAGraphMode.FULL,
        concrete_cudagraph_entries={BatchDescriptor(num_tokens=4, uniform=True): MagicMock()},
    )
    with patch("vllm.compilation.cuda_graph.CUDAGraphWrapper._all_instances", [wrapper]):
        assert not common._beam_full_graph_ready(SimpleNamespace(), 4)


# ---------------------------------------------------------------------------
# Persistent postprocess workspace load hook
# ---------------------------------------------------------------------------


def test_postprocess_workspace_load_wrapper_initializes_after_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def original(runner: Any, marker: Any) -> str:
        calls.append(("load", marker))
        return "loaded"

    def initialize(runner: Any) -> None:
        calls.append(("workspace", runner))

    monkeypatch.setattr(gpu_patch, "initialize_beam_postprocess_workspaces", initialize)
    wrapped = gpu_patch.make_load_model_with_beam_postprocess_workspace(original)
    runner = SimpleNamespace()

    assert wrapped(runner, "weights") == "loaded"
    assert calls == [("load", "weights"), ("workspace", runner)]
    assert gpu_patch.make_load_model_with_beam_postprocess_workspace(wrapped) is wrapped


def test_postprocess_workspace_load_wrapper_accounts_device_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_pool = SimpleNamespace(device_memory_bytes=37)

    def original(runner: Any) -> str:
        # Native load_model recomputes this value on every model load.
        runner.model_memory_usage = 100
        return "loaded"

    monkeypatch.setattr(
        gpu_patch,
        "initialize_beam_postprocess_workspaces",
        lambda runner: workspace_pool,
    )

    wrapped = gpu_patch.make_load_model_with_beam_postprocess_workspace(original)
    runner = SimpleNamespace()

    assert wrapped(runner) == "loaded"
    assert runner.model_memory_usage == 137
    # Reusing an existing pool after reload must be accounted once relative
    # to the freshly measured model memory, rather than accumulated twice.
    assert wrapped(runner) == "loaded"
    assert runner.model_memory_usage == 137


def test_postprocess_workspace_initializer_is_noop_off_cuda() -> None:
    runner = SimpleNamespace(device=torch.device("cpu"))
    assert gpu_patch.initialize_beam_postprocess_workspaces(runner) is None
    assert not hasattr(runner, "_beam_postprocess_workspaces")


def test_postprocess_workspace_initializer_matches_v1_capacity_and_reuses_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    from vllm_gr.config import SUPPORTED_BEAM_BATCH_SIZES, BeamExecutionMode, GRConfig
    from vllm_gr.v1.beam.beam_decision import (
        BeamPostprocessWorkspace,
        BeamPostprocessWorkspaceKey,
    )

    capacity = max(SUPPORTED_BEAM_BATCH_SIZES)
    key = BeamPostprocessWorkspaceKey(4, 4, torch.float16)
    workspace = SimpleNamespace(
        key=key,
        max_batch=capacity,
        max_decode_steps=3,
        device_memory_bytes=0,
        pinned_memory_bytes=0,
    )
    allocate = Mock(return_value=workspace)
    monkeypatch.setattr(BeamPostprocessWorkspace, "allocate", allocate)
    runner = SimpleNamespace(
        device=torch.device("cuda"),
        model_config=SimpleNamespace(dtype=torch.float16),
        _vllm_gr_config=GRConfig(
            beam_execution_mode=BeamExecutionMode.V1,
            beam_worker_decision=True,
            beam_max_width=4,
            beam_max_decode_steps=3,
        ),
    )

    pool = gpu_patch.initialize_beam_postprocess_workspaces(runner)
    assert pool.require(key) is workspace
    allocate.assert_called_once_with(
        max_batch=capacity,
        beam_width=4,
        top_k=4,
        max_decode_steps=3,
        device=runner.device,
        logits_dtype=torch.float16,
        pin_host=True,
    )
    runner._beam_context = SimpleNamespace(max_batch=capacity)
    assert gpu_patch.initialize_beam_postprocess_workspaces(runner) is pool
    assert allocate.call_count == 1

    runner._beam_context.max_batch = capacity - 1
    with pytest.raises(RuntimeError, match="geometry changed"):
        gpu_patch.initialize_beam_postprocess_workspaces(runner)
    assert allocate.call_count == 1
