# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for GPU ModelRunner wrappers."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker import gpu_model_runner_patch as gpu_patch
from vllm_gr.v1.worker import model_runner_common as common


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

    def remap(
        self: Any,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
    ) -> list[int]:
        assert self._current_beam_info["req_ids"] is input_batch.req_ids
        events.append("remap")
        return [2, 4]

    monkeypatch.setattr(gpu_patch, "_beam_remap_inputs", remap)
    wrapped = gpu_patch.make_patched_prepare_inputs(original)
    assert gpu_patch.make_patched_prepare_inputs(wrapped) is wrapped

    ordinary = wrapped(runner, SimpleNamespace(beam_data={}), [1])
    assert ordinary == (original_logits, spec_metadata)
    assert events == ["original"]

    runner._current_beam_info["req_ids"] = []
    logits, returned_spec = wrapped(
        runner,
        SimpleNamespace(beam_data={"request": {}}),
        [1],
    )
    assert events == ["original", "original", "remap"]
    assert logits.dtype is torch.int32
    assert logits.tolist() == [2, 4]
    assert returned_spec is spec_metadata


def test_gpu_bookkeeping_preserves_seven_items_and_restores_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_batch = SimpleNamespace()
    proxy = SimpleNamespace(_obj=original_batch)
    runner = SimpleNamespace(
        input_batch=original_batch,
        _beam_logprobs_cache={"finished": {}, "active": {}},
    )
    ordinary_result = (1, ["lp"], ["valid"], object(), ["request"], {"request": 0}, [3])
    sampler_output = object()

    def original(
        self: Any,
        scheduler_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        if getattr(scheduler_output, "fail", False):
            raise RuntimeError("boom")
        return ordinary_result

    def collapse(
        self: Any,
        orig_batch: Any,
        configs: Any,
        beam_data: Any,
        sampler: Any,
    ) -> tuple[list[str], dict[str, int], int]:
        assert orig_batch is original_batch
        assert sampler is sampler_output
        return ["beam"], {"beam": 0}, 1

    monkeypatch.setattr(gpu_patch, "_beam_collapse_bookkeeping", collapse)
    wrapped = gpu_patch.make_patched_bookkeeping_sync(original)

    assert wrapped(runner, SimpleNamespace(beam_data={}), sampler_output) is ordinary_result

    beam_output = SimpleNamespace(
        beam_data={"beam": {}},
        _gr_request_row_configs=[("beam", 0, 1, [0])],
        finished_req_ids={"finished"},
    )
    runner.input_batch = proxy
    result = wrapped(runner, beam_output, sampler_output)
    assert len(result) == 7
    assert result[:3] == (1, [[]], [[]])
    assert result[4:6] == (["beam"], {"beam": 0})
    assert runner.input_batch is original_batch
    assert "finished" not in runner._beam_logprobs_cache

    runner.input_batch = proxy
    with pytest.raises(RuntimeError, match="boom"):
        wrapped(
            runner,
            SimpleNamespace(
                beam_data={"beam": {}},
                _gr_request_row_configs=[],
                fail=True,
            ),
            sampler_output,
        )
    assert runner.input_batch is original_batch


def test_gpu_bookkeeping_materializes_sync_legacy_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    original_batch = SimpleNamespace()
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(_obj=original_batch),
        use_async_scheduling=False,
    )
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[11]], dtype=torch.int32),
        logprobs_tensors=LogprobsTensors(
            logprob_token_ids=torch.tensor([[11, 12]], dtype=torch.int32),
            logprobs=torch.tensor([[-0.1, -0.2]], dtype=torch.float32),
            selected_token_ranks=torch.tensor([0], dtype=torch.int64),
        ),
    )
    scheduler_output = SimpleNamespace(
        beam_data={"beam": {}},
        _gr_request_row_configs=[("beam", 0, 1, [0])],
        finished_req_ids=set(),
    )

    def original(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return (0, object(), [[99]], {}, ["physical"], {"physical": 0}, [])

    monkeypatch.setattr(
        gpu_patch,
        "_beam_collapse_bookkeeping",
        lambda *args, **kwargs: (["beam"], {"beam": 0}, 1),
    )
    result = gpu_patch.make_patched_bookkeeping_sync(original)(
        runner,
        scheduler_output,
        sampler_output,
    )

    assert result[1].slice_request(0, 1).logprob_token_ids.tolist() == [[11, 12]]
    assert result[2] == [[11]]
    assert result[4:6] == (["beam"], {"beam": 0})
    assert runner.input_batch is original_batch


def test_gpu_bookkeeping_collapses_legacy_sampled_rows_before_sync_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    original_batch = SimpleNamespace()
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(_obj=original_batch),
        use_async_scheduling=False,
    )
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[11], [12], [13], [14]], dtype=torch.int32),
        logprobs_tensors=LogprobsTensors(
            logprob_token_ids=torch.arange(16, dtype=torch.int32).reshape(4, 4),
            logprobs=torch.zeros((4, 4), dtype=torch.float32),
            selected_token_ranks=torch.zeros(4, dtype=torch.int64),
        ),
    )
    scheduler_output = SimpleNamespace(
        beam_data={"beam": {}},
        _gr_request_row_configs=[("beam", 0, 4, [0, 1, 2, 3])],
        finished_req_ids=set(),
    )

    def original(self: Any, scheduler: Any, received: Any, *args: Any) -> tuple[Any, ...]:
        assert received.sampled_token_ids.tolist() == [[11]]
        assert received.logprobs_tensors.logprobs.shape == (4, 4)
        return (0, received.logprobs_tensors.tolists(), [[11]], {}, ["beam"], {"beam": 0}, [])

    monkeypatch.setattr(
        gpu_patch,
        "_beam_collapse_bookkeeping",
        lambda *args, **kwargs: (["beam"], {"beam": 0}, 1),
    )
    result = gpu_patch.make_patched_bookkeeping_sync(original)(
        runner,
        scheduler_output,
        sampler_output,
    )

    assert result[2] == [[11]]
    assert runner.input_batch is original_batch


def test_gpu_bookkeeping_reduces_worker_decision_before_sync_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    width = 4
    original_batch = SimpleNamespace()
    selected_ids = torch.tensor([[21, 22, 23, 24]], dtype=torch.int32)
    selected_scores = torch.tensor([[-0.1, -0.2, -0.3, -0.4]], dtype=torch.float32)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(_obj=original_batch),
        use_async_scheduling=False,
        _beam_decision_tensors={
            "out_token_ids": selected_ids,
            "out_token_index": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            "out_log_probs": selected_scores,
        },
    )
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.arange(width, dtype=torch.int32).reshape(width, 1),
        logprobs_tensors=LogprobsTensors(
            logprob_token_ids=torch.arange(width * width, dtype=torch.int32).reshape(width, width),
            logprobs=torch.zeros((width, width), dtype=torch.float32),
            selected_token_ranks=torch.zeros(width, dtype=torch.int64),
        ),
    )
    scheduler_output = SimpleNamespace(
        beam_data={"beam": {"is_beam_decode": True, "beam_width": width}},
        _gr_request_row_configs=[("beam", 0, width, list(range(width)))],
        finished_req_ids=set(),
    )

    def original(self: Any, scheduler: Any, received: Any, *args: Any) -> tuple[Any, ...]:
        # This is the critical ordering contract: upstream D2H bookkeeping
        # must see O(W), not the original O(W^2) candidate matrix.
        assert received.sampled_token_ids.shape == (1, 1)
        assert received.logprobs_tensors.logprobs.shape == (1, width)
        return (
            0,
            received.logprobs_tensors.tolists(),
            received.sampled_token_ids.tolist(),
            {},
            ["beam"],
            {"beam": 0},
            [],
        )

    result = gpu_patch.make_patched_bookkeeping_sync(original)(
        runner,
        scheduler_output,
        sampler_output,
    )

    assert result[1].slice_request(0, 1).logprob_token_ids.tolist() == [[21, 22, 23, 24]]
    assert result[2] == [[21]]
    assert runner.input_batch is original_batch


def test_gpu_bookkeeping_reuses_host_ready_prefill_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    width = 4
    original_batch = SimpleNamespace()
    host_ids = torch.tensor([[31, 32, 33, 34]], dtype=torch.int32)
    host_scores = torch.tensor([[-0.5, -1.5, -2.5, -3.5]], dtype=torch.float32)
    host_ranks = torch.zeros(1, dtype=torch.int32)

    def unexpected_copy(value: torch.Tensor) -> list[list[int]]:
        raise AssertionError("host-ready prefill must not call the CUDA copy helper")

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(_obj=original_batch),
        use_async_scheduling=False,
        _beam_result_host_ready=True,
        _to_list=unexpected_copy,
        _beam_decision_tensors={
            "out_token_ids": torch.zeros((1, width), dtype=torch.int32),
            "out_log_probs": torch.zeros((1, width), dtype=torch.float32),
            "_host_token_ids": host_ids,
            "_host_log_probs": host_scores,
            "_host_selected_token_ranks": host_ranks,
        },
    )
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.zeros((1, 1), dtype=torch.int32),
        logprobs_tensors=LogprobsTensors(
            logprob_token_ids=torch.zeros((1, width), dtype=torch.int32),
            logprobs=torch.zeros((1, width), dtype=torch.float32),
            selected_token_ranks=torch.zeros(1, dtype=torch.int64),
        ),
    )
    scheduler_output = SimpleNamespace(
        beam_data={"beam": {"is_constraint_prefill": True, "beam_width": width}},
        _gr_request_row_configs=[("beam", 0, 1, [0])],
        finished_req_ids=set(),
    )

    def original(self: Any, scheduler: Any, received: Any, *args: Any) -> tuple[Any, ...]:
        assert received.sampled_token_ids.device.type == "cpu"
        assert received.logprobs_tensors.logprobs.device.type == "cpu"
        return (
            0,
            received.logprobs_tensors.tolists(),
            self._to_list(received.sampled_token_ids),
            {},
            ["beam"],
            {"beam": 0},
            [],
        )

    monkeypatch.setattr(
        gpu_patch,
        "_beam_collapse_bookkeeping",
        lambda *args, **kwargs: (["beam"], {"beam": 0}, 1),
    )
    result = gpu_patch.make_patched_bookkeeping_sync(original)(
        runner,
        scheduler_output,
        sampler_output,
    )

    assert result[2] == [[31]]
    assert result[1].slice_request(0, 1).logprob_token_ids.tolist() == [[31, 32, 33, 34]]
    assert runner._to_list is unexpected_copy


def test_gpu_sample_patch_runs_original_then_constraint_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy branch must still sample first, then override rows."""
    from vllm_gr.v1.worker.model_runner_common import _WORKER_SAMPLE_PATCH_MARKER

    events: list[str] = []
    sample_output = object()

    def original(self: Any, logits: Any, spec_decode_metadata: Any = None) -> Any:
        assert logits == "logits"
        events.append("sample")
        return sample_output

    def override(self: Any, logits: Any, sampler_output: Any) -> None:
        assert self is runner
        assert logits == "logits"
        assert sampler_output is sample_output
        events.append("override")

    monkeypatch.setattr(gpu_patch, "apply_constrained_sample_override", override)
    wrapped = gpu_patch.make_patched_sample_gpu(original)
    assert getattr(wrapped, _WORKER_SAMPLE_PATCH_MARKER, False)
    assert gpu_patch.make_patched_sample_gpu(wrapped) is wrapped

    runner = SimpleNamespace()
    assert wrapped(runner, "logits", "meta") is sample_output
    assert events == ["sample", "override"]


def test_gpu_sample_patch_returns_canonical_output_before_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    direct_output = object()

    def direct(self: Any, logits: Any, spec_decode_metadata: Any) -> Any:
        assert logits == "logits"
        assert spec_decode_metadata == "meta"
        events.append("direct")
        return direct_output

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"ordinary sampler or legacy override ran: {args} {kwargs}")

    def decide(self: Any, sampler_output: Any) -> None:
        assert sampler_output is direct_output
        events.append("decision")

    monkeypatch.setattr(gpu_patch, "try_sample_with_constraint_backend", direct)
    monkeypatch.setattr(gpu_patch, "apply_constrained_sample_override", forbidden)
    monkeypatch.setattr(gpu_patch, "maybe_run_worker_beam_decision", decide)
    wrapped = gpu_patch.make_patched_sample_gpu(forbidden)

    assert wrapped(SimpleNamespace(), "logits", "meta") is direct_output
    assert events == ["direct", "decision"]


def test_gpu_sample_patch_consumes_prelaunched_prefill_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    prelaunched = object()

    def consume(self: Any, logits: Any, spec: Any) -> Any:
        assert logits == "logits"
        assert spec == "meta"
        events.append("consume")
        return prelaunched

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"prelaunched prefill sampled twice: {args} {kwargs}")

    def decide(self: Any, sampler_output: Any) -> None:
        assert sampler_output is prelaunched
        events.append("decision")

    monkeypatch.setattr(gpu_patch, "consume_prelaunched_cuda_prefill", consume)
    monkeypatch.setattr(gpu_patch, "try_sample_with_constraint_backend", forbidden)
    monkeypatch.setattr(gpu_patch, "apply_constrained_sample_override", forbidden)
    monkeypatch.setattr(gpu_patch, "maybe_run_worker_beam_decision", decide)

    wrapped = gpu_patch.make_patched_sample_gpu(forbidden)
    assert wrapped(SimpleNamespace(), "logits", "meta") is prelaunched
    assert events == ["consume", "decision"]


def test_compute_logits_hook_prelaunches_on_returned_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker import constraint_hook

    events: list[Any] = []
    logits = object()

    class Model:
        def compute_logits(self, hidden: Any) -> Any:
            events.append(("lm_head", hidden))
            return logits

    runner = SimpleNamespace(model=Model())

    def launch(received_runner: Any, received_logits: Any) -> bool:
        assert received_runner is runner
        events.append(("prelaunch", received_logits))
        return True

    monkeypatch.setattr(constraint_hook, "launch_prepared_cuda_prefill_constraint", launch)
    assert gpu_patch._install_cuda_prefill_compute_logits_hook(runner) is True
    assert gpu_patch._install_cuda_prefill_compute_logits_hook(runner) is True
    assert runner.model.compute_logits("hidden") is logits
    assert events == [("lm_head", "hidden"), ("prelaunch", logits)]


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

    import vllm_gr.v1.worker.gpu_model_runner_patch as mod

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

    import vllm_gr.v1.worker.gpu_model_runner_patch as mod

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

    import vllm_gr.v1.worker.gpu_model_runner_patch as mod

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

    stats = common.get_beam_graph_runtime_stats(worker)

    assert stats == {
        "runner": "SimpleNamespace",
        "beam_graph_enabled": True,
        "beam_max_width": 128,
        "captured_widths": [128],
        "full_dispatches": 2,
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
    assert getattr(FakeWorker, "get_gr_profile_runtime_stats") is (
        gpu_patch.get_gr_profile_runtime_stats
    )


def test_gpu_named_profile_diagnostic_returns_runner_and_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRunner:
        device = torch.device("cuda:0")
        _vllm_gr_config = SimpleNamespace(beam_graph_enabled=True, beam_max_width=128)
        _beam_graph_keys = {128}
        _beam_graph_full_dispatch_count = 7

    fake_properties = SimpleNamespace(name="L20")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: fake_properties)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 11)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 12)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 13)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 14)

    result = gpu_patch.get_gr_profile_runtime_stats(SimpleNamespace(model_runner=FakeRunner()))
    assert result["backend"] == "gpu"
    assert result["runner_class"].endswith(".FakeRunner")
    assert result["device_name"] == "L20"
    assert result["memory"] == {
        "allocated_bytes": 11,
        "reserved_bytes": 12,
        "peak_allocated_bytes": 13,
        "peak_reserved_bytes": 14,
    }
    assert result["beam_graph"]["captured_widths"] == [128]
    assert result["beam_graph"]["full_dispatches"] == 7


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

    import vllm_gr.v1.worker.gpu_model_runner_patch as mod

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

    import vllm_gr.v1.worker.gpu_model_runner_patch as mod

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

    import vllm_gr.v1.worker.gpu_model_runner_patch as mod

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

    import vllm_gr.v1.worker.model_runner_common as common

    original_fn = common._get_gr_config

    def _stub_config(r: Any) -> Any:
        return gr_cfg

    common._get_gr_config = _stub_config
    try:
        runner = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False)
        )
        result = gpu_patch.capture_beam_decode_graph(runner)
        assert result is False
    finally:
        common._get_gr_config = original_fn


def test_capture_beam_decode_graph_tp_gt1_returns_false() -> None:
    """With TP > 1, capture must refuse and return False."""
    gr_cfg = SimpleNamespace(
        beam_graph_active=True,
        beam_max_width=4,
        beam_max_decode_steps=3,
    )

    import vllm_gr.v1.worker.model_runner_common as common

    original_fn = common._get_gr_config

    def _stub_config(r: Any) -> Any:
        return gr_cfg

    common._get_gr_config = _stub_config
    try:
        runner = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2, use_ubatching=False),
        )
        result = gpu_patch.capture_beam_decode_graph(runner)
        assert result is False
    finally:
        common._get_gr_config = original_fn


def test_capture_beam_decode_graph_ubatching_returns_false() -> None:
    """With DP ubatching, capture must refuse and return False."""
    gr_cfg = SimpleNamespace(
        beam_graph_active=True,
        beam_max_width=4,
        beam_max_decode_steps=3,
    )

    import vllm_gr.v1.worker.model_runner_common as common

    original_fn = common._get_gr_config

    def _stub_config(r: Any) -> Any:
        return gr_cfg

    common._get_gr_config = _stub_config
    try:
        runner = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=True),
        )
        result = gpu_patch.capture_beam_decode_graph(runner)
        assert result is False
    finally:
        common._get_gr_config = original_fn


def test_capture_beam_decode_graph_forwards_run_under_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warmup + capture forwards must run inside torch.inference_mode().

    The CUDA graph pool holds inference tensors created by the native
    ``_dummy_run`` (decorated with ``@torch.inference_mode()``).  Running the
    capture forward outside inference mode makes ``capture_begin``'s in-place
    pool updates raise "Inplace update to inference tensor outside
    InferenceMode is not allowed".
    """
    import contextlib

    import vllm.compilation.monitor as monitor

    import vllm_gr.v1.worker.model_runner_common as common

    monkeypatch.setattr(
        common,
        "_get_gr_config",
        lambda r: SimpleNamespace(
            beam_graph_active=True, beam_max_width=4, beam_max_decode_steps=3
        ),
    )
    monkeypatch.setattr(monitor, "set_cudagraph_capturing_enabled", lambda *_: None)

    side_stream = SimpleNamespace(wait_stream=lambda *_: None)
    fake_cuda = SimpleNamespace(
        Stream=lambda: side_stream, stream=lambda _s: contextlib.nullcontext()
    )
    fake_accelerator = SimpleNamespace(
        current_stream=lambda: SimpleNamespace(wait_stream=lambda *_: None),
        synchronize=lambda: None,
    )
    monkeypatch.setattr(torch, "cuda", fake_cuda, raising=False)
    monkeypatch.setattr(torch, "accelerator", fake_accelerator, raising=False)

    session = SimpleNamespace(end_session=lambda: None)
    monkeypatch.setattr(
        gpu_patch,
        "_capture_create_dummy_context",
        lambda *args, **kwargs: (object(), session, object(), object()),
    )

    inference_flags: list[bool] = []

    def fake_forward(*args: Any, **kwargs: Any) -> None:
        inference_flags.append(torch.is_inference_mode_enabled())

    monkeypatch.setattr(gpu_patch, "_dummy_beam_forward", fake_forward)
    monkeypatch.setattr(gpu_patch, "_verify_and_record_captured_graph", lambda r, width: True)

    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False),
        _beam_sessions={},
        _current_beam_info=None,
    )
    assert gpu_patch.capture_beam_decode_graph(runner) is True
    assert len(inference_flags) == 3  # num_warmups (2) + capture forward
    assert all(inference_flags), f"capture forwards ran outside inference_mode: {inference_flags}"


# ---------------------------------------------------------------------------
# _create_beam_context — shared context factory
# ---------------------------------------------------------------------------


def test_create_beam_context_uses_grconfig_params() -> None:
    """_create_beam_context must pass gr_cfg.max_width/max_decode_steps."""
    from unittest.mock import MagicMock, patch

    gr_cfg = SimpleNamespace(beam_max_width=64, beam_max_decode_steps=10)
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_kv_heads=lambda pc: 8,
            get_head_size=lambda: 128,
            get_num_layers=lambda pc: 2,
            dtype=torch.float16,
        ),
        parallel_config=SimpleNamespace(),
        device=torch.device("cpu"),
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


def test_create_beam_context_dtype_from_model_config() -> None:
    """dtype must come from model_config.dtype, not hardcoded."""
    from unittest.mock import MagicMock, patch

    gr_cfg = SimpleNamespace(beam_max_width=32, beam_max_decode_steps=5)
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


# ---------------------------------------------------------------------------
# _capture_create_dummy_context — dummy session setup
# ---------------------------------------------------------------------------


def test_capture_create_dummy_context_creates_session_and_beam_info() -> None:
    """_capture_create_dummy_context must create a session with correct width/prefix."""
    from unittest.mock import MagicMock, patch

    gr_cfg = SimpleNamespace(beam_max_width=128, beam_max_decode_steps=5)

    with patch("vllm_gr.v1.worker.gpu_model_runner_patch._create_beam_context") as mock_create:
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
# _verify_and_record_captured_graph — graph verification + key recording
# ---------------------------------------------------------------------------


def test_verify_and_record_captured_graph_records_on_match() -> None:
    """When a matching CUDAGraphWrapper entry exists, record the width."""
    from unittest.mock import MagicMock, patch

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    mock_entry = MagicMock()
    mock_entry.cudagraph = MagicMock()  # not None → captured

    mock_wrapper = MagicMock()
    mock_wrapper.runtime_mode = CUDAGraphMode.FULL
    mock_wrapper.concrete_cudagraph_entries = {
        BatchDescriptor(num_tokens=4, uniform=True): mock_entry
    }

    runner = SimpleNamespace()

    with patch("vllm.compilation.cuda_graph.CUDAGraphWrapper._all_instances", [mock_wrapper]):
        result = gpu_patch._verify_and_record_captured_graph(runner, width=4)

    assert result is True
    assert 4 in runner._beam_graph_keys


def test_verify_and_record_captured_graph_no_match() -> None:
    """When no matching entry exists, return False and don't record."""
    from unittest.mock import MagicMock, patch

    mock_wrapper = MagicMock()
    mock_wrapper.runtime_mode = MagicMock()  # not FULL
    mock_wrapper.concrete_cudagraph_entries = {}

    runner = SimpleNamespace()

    with patch("vllm.compilation.cuda_graph.CUDAGraphWrapper._all_instances", [mock_wrapper]):
        result = gpu_patch._verify_and_record_captured_graph(runner, width=4)

    assert result is False
    assert not hasattr(runner, "_beam_graph_keys")


def test_verify_and_record_captured_graph_appends_to_existing_keys() -> None:
    """If _beam_graph_keys already has entries, append (don't replace)."""
    from unittest.mock import MagicMock, patch

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    mock_entry = MagicMock()
    mock_entry.cudagraph = MagicMock()

    mock_wrapper = MagicMock()
    mock_wrapper.runtime_mode = CUDAGraphMode.FULL
    mock_wrapper.concrete_cudagraph_entries = {
        BatchDescriptor(num_tokens=8, uniform=True): mock_entry
    }

    runner = SimpleNamespace(_beam_graph_keys={4})

    with patch("vllm.compilation.cuda_graph.CUDAGraphWrapper._all_instances", [mock_wrapper]):
        result = gpu_patch._verify_and_record_captured_graph(runner, width=8)

    assert result is True
    assert runner._beam_graph_keys == {4, 8}


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
    monkeypatch.setattr(
        gpu_patch,
        "_install_cuda_prefill_compute_logits_hook",
        lambda runner: False,
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
