# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Core contracts for GPU ModelRunner wrappers."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.config import GRConfig
from vllm_gr.v1.engine.beam_search_v1 import GRStageKind
from vllm_gr.v1.worker import gr_context as _gr_context
from vllm_gr.v1.worker.cuda.v1 import gpu_model_runner_patch as gpu_patch


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
    "invalid",
    ["ids", "positions", "mismatched_padding", "inputs_embeds", "intermediate_tensors"],
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
    elif invalid == "mismatched_padding":
        ids = torch.zeros(8, dtype=torch.int32)
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


@pytest.mark.parametrize("padding", [0, 4])  # type: ignore[misc]
def test_batch_session_prefill_selects_exact_four_request_graph(
    monkeypatch: pytest.MonkeyPatch,
    padding: int,
) -> None:
    request_ids = ("r0", "r1", "r2", "r3")
    scheduled_tokens = dict.fromkeys(request_ids, 1)
    tokens = torch.arange(4 + padding, dtype=torch.int32)
    positions = torch.arange(4 + padding)
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


def test_gpu_prepare_preserves_order_and_three_item_contract(
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
    ) -> tuple[object, object, int]:
        assert self._current_beam_info["req_ids"] == []
        events.append("original")
        return original_logits, spec_metadata, 5

    wrapped = gpu_patch.make_patched_prepare_inputs(original)
    assert gpu_patch.make_patched_prepare_inputs(wrapped) is wrapped

    ordinary = wrapped(runner, SimpleNamespace(beam_data={}), [1])
    assert ordinary == (original_logits, spec_metadata, 5)
    assert events == ["original"]

    assert runner._current_scheduler_output.beam_data == {}
    assert runner._current_beam_info["req_ids"] is input_batch.req_ids


# ---------------------------------------------------------------------------
# _beam_full_graph_ready
# ---------------------------------------------------------------------------


def test_beam_full_graph_ready_returns_true_for_captured_width() -> None:
    runner = SimpleNamespace(_beam_graph_keys={4, 128})
    assert gpu_patch._beam_full_graph_ready(runner, 4) is True
    assert gpu_patch._beam_full_graph_ready(runner, 128) is True


def test_beam_full_graph_ready_returns_false_for_uncaptured_width() -> None:
    runner = SimpleNamespace(_beam_graph_keys={4})
    assert gpu_patch._beam_full_graph_ready(runner, 8) is False


def test_beam_full_graph_ready_returns_false_when_no_keys() -> None:
    runner = SimpleNamespace()
    assert gpu_patch._beam_full_graph_ready(runner, 4) is False

    runner2 = SimpleNamespace(_beam_graph_keys=None)
    assert gpu_patch._beam_full_graph_ready(runner2, 4) is False

    runner3 = SimpleNamespace(_beam_graph_keys=set())
    assert gpu_patch._beam_full_graph_ready(runner3, 4) is False


# ---------------------------------------------------------------------------
# patched_determine — FULL grant / session guard / mixed-batch guard
# ---------------------------------------------------------------------------


def _make_determine_wrapper(beam_info: Any, native_result: Any) -> Any:
    """Build a patched_determine with a stubbed native _orig_determine."""

    def _orig_determine(self: Any, *args: Any, **kwargs: Any) -> Any:
        return native_result

    wrapped = gpu_patch.make_patched_determine_batch_execution_and_padding(_orig_determine)
    return wrapped


def test_patched_determine_no_beam_info_passthrough() -> None:
    """When _current_beam_info is None, the native result is returned unchanged."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=16))
    runner = SimpleNamespace(_current_beam_info=None)
    wrapped = _make_determine_wrapper(None, native)
    result = wrapped(runner, num_tokens=16)
    assert result[0] == CUDAGraphMode.NONE
    assert result[1].num_tokens == 16


def test_patched_determine_session_missing_downgrades_full() -> None:
    """If beam_data has beam-decode entries but no beam_session, FULL -> NONE."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    beam_info: dict[str, Any] = {
        "beam_data": {"req-1": {"is_beam_decode": True, "decode_step": 0}},
        "req_ids": ["req-1"],
        # No "beam_session" key
    }
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=4, uniform=True))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
    )

    import vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch as mod

    original_fn = mod._beam_graph_enabled_for_run

    def _stub_enabled(r: Any) -> bool:
        return True

    mod._beam_graph_enabled_for_run = _stub_enabled
    try:
        wrapped = _make_determine_wrapper(beam_info, native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        mod._beam_graph_enabled_for_run = original_fn


def test_patched_determine_mixed_batch_downgrades_full() -> None:
    """A mixed batch (beam + non-beam) must not get FULL even with a session."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True)
    beam_info: dict[str, Any] = {
        "beam_data": {
            "beam-1": {"is_beam_decode": True, "decode_step": 0},
            "std-1": {},  # non-beam request
        },
        "req_ids": ["beam-1", "std-1"],
        "beam_session": session,
    }
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=8, uniform=True))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={8},
    )

    import vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch as mod

    original_fn = mod._beam_graph_enabled_for_run

    def _stub_enabled(r: Any) -> bool:
        return True

    mod._beam_graph_enabled_for_run = _stub_enabled
    try:
        wrapped = _make_determine_wrapper(beam_info, native)
        result = wrapped(runner, num_tokens=8)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        mod._beam_graph_enabled_for_run = original_fn


def test_patched_determine_pure_beam_batch_grants_full() -> None:
    """A pure beam batch with session + matching width should get FULL."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True)
    beam_info: dict[str, Any] = {
        "beam_data": {"beam-1": {"is_beam_decode": True, "decode_step": 0}},
        "req_ids": ["beam-1"],
        "beam_session": session,
    }
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=4))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
    )

    import vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch as mod

    original_fn = mod._beam_graph_enabled_for_run

    def _stub_enabled(r: Any) -> bool:
        return True

    mod._beam_graph_enabled_for_run = _stub_enabled
    try:
        wrapped = _make_determine_wrapper(beam_info, native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.FULL
        assert result[1].num_tokens == 4
        assert result[1].uniform is True
        assert runner._beam_graph_full_dispatch_count == 1
        wrapped(runner, num_tokens=4)
        assert runner._beam_graph_full_dispatch_count == 2
    finally:
        mod._beam_graph_enabled_for_run = original_fn


def test_beam_graph_runtime_stats_reports_capture_and_dispatch_state() -> None:
    config = SimpleNamespace(beam_graph_enabled=True, beam_max_width=128)
    runner = SimpleNamespace(
        _vllm_gr_config=config,
        _beam_graph_keys={128},
        _beam_graph_full_dispatch_count=2,
    )
    worker = SimpleNamespace(model_runner=runner)

    stats = _gr_context.get_beam_graph_runtime_stats(worker)

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

    from vllm_gr.platforms.cuda import provider as cuda

    class FakeWorker:
        pass

    monkeypatch.setattr(gpu_worker, "Worker", FakeWorker)
    monkeypatch.setattr(
        "vllm_gr.constraints.worker_inject.install_worker_resource_hook",
        lambda worker_cls: 0,
    )

    cuda.install_worker_utilities()
    first = getattr(FakeWorker, "get_gr_beam_graph_runtime_stats")
    cuda.install_worker_utilities()

    assert first is _gr_context.get_beam_graph_runtime_stats
    assert getattr(FakeWorker, "get_gr_beam_graph_runtime_stats") is first


def test_patched_determine_active_lora_downgrades_full() -> None:
    """Active-LoRA beam batches must run eager: the beam graph is captured
    only for the no-LoRA descriptor, and the FULL rewrite would drop
    has_lora/num_active_loras (silently routing to the LoRA-free graph)."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True)
    beam_info: dict[str, Any] = {
        "beam_data": {"beam-1": {"is_beam_decode": True, "decode_step": 0}},
        "req_ids": ["beam-1"],
        "beam_session": session,
    }
    native = (CUDAGraphMode.FULL, BatchDescriptor(num_tokens=4, uniform=True, has_lora=True))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
    )

    import vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch as mod

    original_fn = mod._beam_graph_enabled_for_run

    def _stub_enabled(r: Any) -> bool:
        return True

    mod._beam_graph_enabled_for_run = _stub_enabled
    try:
        wrapped = _make_determine_wrapper(beam_info, native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        mod._beam_graph_enabled_for_run = original_fn


def test_patched_determine_graph_disabled_no_full() -> None:
    """If beam_graph_enabled is False, FULL is not granted even for pure batch."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True)
    beam_info: dict[str, Any] = {
        "beam_data": {"beam-1": {"is_beam_decode": True, "decode_step": 0}},
        "req_ids": ["beam-1"],
        "beam_session": session,
    }
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=4))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},
    )

    import vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch as mod

    original_fn = mod._beam_graph_enabled_for_run

    def _stub_disabled(r: Any) -> bool:
        return False

    mod._beam_graph_enabled_for_run = _stub_disabled
    try:
        wrapped = _make_determine_wrapper(beam_info, native)
        result = wrapped(runner, num_tokens=4)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        mod._beam_graph_enabled_for_run = original_fn


def test_patched_determine_width_mismatch_no_full() -> None:
    """If captured width != real_num_tokens, FULL is not granted."""
    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    session = SimpleNamespace(is_active=True)
    beam_info: dict[str, Any] = {
        "beam_data": {"beam-1": {"is_beam_decode": True, "decode_step": 0}},
        "req_ids": ["beam-1"],
        "beam_session": session,
    }
    native = (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=8))
    runner = SimpleNamespace(
        _current_beam_info=beam_info,
        _beam_graph_keys={4},  # captured at 4, but batch is 8 tokens
    )

    import vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch as mod

    original_fn = mod._beam_graph_enabled_for_run

    def _stub_enabled(r: Any) -> bool:
        return True

    mod._beam_graph_enabled_for_run = _stub_enabled
    try:
        wrapped = _make_determine_wrapper(beam_info, native)
        result = wrapped(runner, num_tokens=8)
        assert result[0] == CUDAGraphMode.NONE
    finally:
        mod._beam_graph_enabled_for_run = original_fn


# ---------------------------------------------------------------------------
# capture_beam_decode_graph — entry guards
# ---------------------------------------------------------------------------


def test_capture_beam_decode_graph_disabled_returns_false() -> None:
    """When beam_graph_active is False, capture returns False immediately."""
    gr_cfg = SimpleNamespace(beam_graph_active=False)

    original_fn = _gr_context._get_gr_config

    def _stub_config(r: Any) -> Any:
        return gr_cfg

    _gr_context._get_gr_config = _stub_config
    try:
        runner = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False)
        )
        result = gpu_patch.capture_beam_decode_graph(runner)
        assert result is False
    finally:
        _gr_context._get_gr_config = original_fn


def test_capture_beam_decode_graph_tp_gt1_returns_false() -> None:
    """With TP > 1, capture must refuse and return False."""
    gr_cfg = SimpleNamespace(
        beam_graph_active=True,
        beam_max_width=4,
        beam_max_decode_steps=3,
    )

    original_fn = _gr_context._get_gr_config

    def _stub_config(r: Any) -> Any:
        return gr_cfg

    _gr_context._get_gr_config = _stub_config
    try:
        runner = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            parallel_config=SimpleNamespace(tensor_parallel_size=2, use_ubatching=False),
        )
        result = gpu_patch.capture_beam_decode_graph(runner)
        assert result is False
    finally:
        _gr_context._get_gr_config = original_fn


def test_capture_beam_decode_graph_ubatching_returns_false() -> None:
    """With DP ubatching, capture must refuse and return False."""
    gr_cfg = SimpleNamespace(
        beam_graph_active=True,
        beam_max_width=4,
        beam_max_decode_steps=3,
    )

    original_fn = _gr_context._get_gr_config

    def _stub_config(r: Any) -> Any:
        return gr_cfg

    _gr_context._get_gr_config = _stub_config
    try:
        runner = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=True),
        )
        result = gpu_patch.capture_beam_decode_graph(runner)
        assert result is False
    finally:
        _gr_context._get_gr_config = original_fn


def test_capture_beam_decode_graph_forwards_run_under_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextlib

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor

    from vllm_gr.v1.worker.cuda import beam_decode_graph as shared

    monkeypatch.setattr(
        _gr_context,
        "_get_gr_config",
        lambda _: SimpleNamespace(
            beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3
        ),
    )
    monkeypatch.setattr(shared, "set_cudagraph_capturing_enabled", lambda _: None)
    stream = SimpleNamespace(wait_stream=lambda _: None, synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kw: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 123))
    context = object()
    native_model = object()
    runner = SimpleNamespace(
        device="cuda",
        model_config=SimpleNamespace(enforce_eager=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
        _beam_sessions={},
        _beam_context=context,
        model=native_model,
    )
    session = SimpleNamespace(end_session=lambda: None)
    monkeypatch.setattr(
        gpu_patch,
        "_capture_create_dummy_context",
        lambda *a: (context, session, object(), object()),
    )
    graph = object.__new__(shared.BeamDecodeGraph)
    graph.worker, graph.width = runner, 4
    graph.signature = shared.BeamDecodeGraphSignature(1, 4, 4)
    graph._bindings = {}
    graph.capture_inputs = graph.capture_attention = None
    graph.descriptor = BatchDescriptor(num_tokens=4, uniform=True)
    entries: dict[Any, Any] = {}
    graph.wrapper = SimpleNamespace(concrete_cudagraph_entries=entries, clear_graphs=entries.clear)
    graph._registries = {graph.signature: entries}
    # R1 install/uninstall state (normally set in __init__, bypassed by
    # object.__new__): V1's capture() installs the wrapper persistently, so
    # the ownership pair must be present and consistent.
    graph.original_model = native_model
    graph.installed = False
    monkeypatch.setattr(shared, "get_beam_decode_graph", lambda _: graph)
    modes = []

    def forward(*args: Any) -> None:
        assert torch.is_inference_mode_enabled()
        modes.append(args[-1])
        if args[-1]:
            graph.wrapper.concrete_cudagraph_entries[graph.descriptor] = SimpleNamespace(
                cudagraph=object()
            )

    monkeypatch.setattr(gpu_patch, "_dummy_beam_forward", forward)
    assert gpu_patch.capture_beam_decode_graph(runner)
    assert modes == [False, False, True]
    assert graph.wrapper.graph_pool == (0, 123)
    assert runner._beam_graph_keys == {4}
    # V1 persistent install (R1): after capture the wrapper owns worker.model
    # (the native FULL replay path routes through it for the engine lifetime).
    assert runner.model is graph.wrapper
    assert graph.installed is True


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
        gpu_patch._create_beam_context(runner, gr_cfg, max_batch=1)
        MockCtx.assert_called_once()
        call_kwargs = MockCtx.call_args[1]
        # gr_cfg params must flow through directly
        assert call_kwargs["max_beams"] == 64
        assert call_kwargs["max_decode_steps"] == 10
        assert call_kwargs["dtype"] is torch.float16
        assert call_kwargs["max_batch"] == 1
        assert call_kwargs["enable_short_suffix_attention"] is (
            device == "cuda" and enabled is not False
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
        gpu_patch._create_beam_context(runner, gr_cfg)
        call_kwargs = MockCtx.call_args[1]
        assert call_kwargs["dtype"] is torch.bfloat16
        assert call_kwargs["enable_short_suffix_attention"] is False


# ---------------------------------------------------------------------------
# _capture_create_dummy_context — dummy session setup
# ---------------------------------------------------------------------------


def test_capture_create_dummy_context_creates_session_and_beam_info() -> None:
    """_capture_create_dummy_context must create a session with correct width/prefix."""
    from unittest.mock import MagicMock, patch

    gr_cfg = SimpleNamespace(beam_max_width=128, beam_max_decode_steps=5)

    with patch(
        "vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch._create_beam_context"
    ) as mock_create:
        mock_context = MagicMock()
        mock_context.registry.allocate.return_value = 0
        mock_create.return_value = mock_context

        with patch("vllm_gr.v1.beam.beam_search_session.BeamSearchSession") as MockSession:
            mock_session = MagicMock()
            MockSession.return_value = mock_session

            runner = SimpleNamespace(
                model_config=SimpleNamespace(),
                parallel_config=SimpleNamespace(),
                device=torch.device("cpu"),
            )

            _context, _session, beam_info, batch_descriptor = (
                gpu_patch._capture_create_dummy_context(
                    runner, gr_cfg, width=4, steps_upper=5, prefix_len=1024
                )
            )

            # Session created with correct params
            MockSession.assert_called_once_with(
                session_id=gpu_patch._BEAM_DUMMY_REQ_ID, slot=0, context=mock_context
            )
            mock_session.begin_session.assert_called_once_with(4, 1024)

            # beam_info contains the dummy request
            assert gpu_patch._BEAM_DUMMY_REQ_ID in beam_info["beam_data"]
            dummy_mdata = beam_info["beam_data"][gpu_patch._BEAM_DUMMY_REQ_ID]
            assert dummy_mdata["beam_width"] == 4
            assert dummy_mdata["prefix_len"] == 1024
            assert dummy_mdata["is_beam_decode"] is True

            # beam_session linked
            assert beam_info["beam_session"] is mock_session

            # batch_descriptor matches width
            from vllm.v1.cudagraph_dispatcher import BatchDescriptor

            assert batch_descriptor == BatchDescriptor(num_tokens=4, uniform=True)


# ---------------------------------------------------------------------------
# Beam graph readiness uses the current Worker's registry.
# ---------------------------------------------------------------------------


def test_beam_full_graph_ready_uses_worker_registry() -> None:
    graph = SimpleNamespace(signature=SimpleNamespace(num_model_rows=8), ready=True)
    runner = SimpleNamespace(_beam_decode_graph=graph)
    assert _gr_context._beam_full_graph_ready(runner, 8)
    assert not _gr_context._beam_full_graph_ready(runner, 4)
    runner._beam_decode_graph.ready = False
    assert not _gr_context._beam_full_graph_ready(runner, 8)


def test_beam_full_graph_ready_ignores_other_native_graphs() -> None:
    from unittest.mock import MagicMock, patch

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    wrapper = SimpleNamespace(
        runtime_mode=CUDAGraphMode.FULL,
        concrete_cudagraph_entries={BatchDescriptor(num_tokens=4, uniform=True): MagicMock()},
    )
    with patch("vllm.compilation.cuda_graph.CUDAGraphWrapper._all_instances", [wrapper]):
        assert not _gr_context._beam_full_graph_ready(SimpleNamespace(), 4)


pytestmark = pytest.mark.cpu_test
