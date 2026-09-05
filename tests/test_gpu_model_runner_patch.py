# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for GPU ModelRunner wrappers."""

from contextlib import contextmanager
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker import gpu_model_runner_patch as gpu_patch
from vllm_gr.v1.worker import model_runner_common as common
from vllm_gr.v1.worker.gr_model_runner_state import GRDeviceBackend


def test_gpu_backend_factory_returns_v1_protocol_implementation() -> None:
    backend = gpu_patch.create_gr_device_backend(SimpleNamespace())
    assert isinstance(backend, GRDeviceBackend)


@pytest.mark.parametrize("stage", ["ordinary", "prefill", "decode"])
def test_execute_failure_quarantines_before_any_later_forward(monkeypatch, stage):
    calls = []

    class Transport:
        error = None
        retained = object()

        def check_live(self):
            if self.error is not None:
                raise RuntimeError("quarantined")

        def quarantine(self, error):
            self.error = error

    transport = Transport()
    retained = transport.retained
    runner = SimpleNamespace(_gr_cuda_output_transport=transport)
    metadata = {"persistent_runner_state": True, "beam_width": 4}
    metadata["is_beam_decode" if stage == "decode" else "is_constraint_prefill"] = True
    output = SimpleNamespace(
        beam_data={} if stage == "ordinary" else {"gr": metadata},
        num_scheduled_tokens={"gr": 1}, total_num_scheduled_tokens=1,
        gr_outputs={"gr": SimpleNamespace(logical_token_count=1, physical_row_count=4)},
    )

    def original(*_args):
        calls.append("forward")
        raise ValueError("forward failed")

    monkeypatch.setattr(gpu_patch, "_make_common_patched_execute_model", lambda fn: fn)
    monkeypatch.setattr(gpu_patch, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(gpu_patch, "has_ec_transfer", lambda: False)
    execute = gpu_patch.make_patched_execute_model_gpu(original)
    with pytest.raises(ValueError, match="forward failed"):
        execute(runner, output)
    with pytest.raises(RuntimeError, match="quarantined"):
        execute(runner, output)
    assert calls == ["forward"] and transport.retained is retained
    assert output.total_num_scheduled_tokens == 1


@pytest.mark.parametrize("failure", [False, True])
def test_gpu_shutdown_drains_before_common_release(failure):
    calls = []

    def drain():
        calls.append("drain")
        if failure:
            raise RuntimeError("drain failed")

    shutdown = gpu_patch.make_patched_shutdown(lambda _runner: calls.append("release"))
    runner = SimpleNamespace(_gr_cuda_output_transport=SimpleNamespace(shutdown=drain))
    if failure:
        with pytest.raises(RuntimeError, match="drain failed"):
            shutdown(runner)
        assert calls == ["drain"]
    else:
        shutdown(runner)
        assert calls == ["drain", "release"]


def _input_prep_fixture(*, ready=True, active=True, failure=None):
    calls = []

    class Event:
        def query(self):
            calls.append("query")
            if failure == "query":
                raise RuntimeError("query failed")
            return ready

        def synchronize(self):
            calls.append("synchronize")
            if failure == "synchronize":
                raise RuntimeError("synchronize failed")

        def record(self):
            calls.append("record")

    @contextmanager
    def original(self):
        calls.append("original")
        if self.prepare_inputs_event is None:
            yield
            return
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    output = SimpleNamespace(beam_data={"gr": {"persistent_runner_state": True}})
    if active:
        output._gr_cuda_logical_geometry = ("gr", 1, 128)
    runner = SimpleNamespace(
        prepare_inputs_event=Event(), _current_scheduler_output=output
    )
    return runner, original, calls


@pytest.mark.parametrize("ready", [True, False])
@pytest.mark.parametrize("body_error", [True, False])
def test_input_prep_keeps_complete_guard_and_error_record(ready, body_error):
    runner, original, calls = _input_prep_fixture(ready=ready)
    event = runner.prepare_inputs_event
    patched = gpu_patch.make_patched_synchronize_input_prep(original)
    assert gpu_patch.make_patched_synchronize_input_prep(patched) is patched

    def prepare():
        with patched(runner):
            # The guarded region starts before _update_states, not only at
            # _prepare_inputs. Native zeroing/metadata/preprocess stay inside.
            calls.extend(["update_states", "prepare_inputs", "metadata", "preprocess"])
            if body_error:
                raise ValueError("preparation failed")

    if body_error:
        with pytest.raises(ValueError, match="preparation failed"):
            prepare()
    else:
        prepare()
    assert calls == (
        (["query"] if ready else ["query", "original", "synchronize"])
        + ["update_states", "prepare_inputs", "metadata", "preprocess", "record"]
    )
    assert runner.prepare_inputs_event is event
    assert getattr(runner, "_gr_input_prep_ready_skips", 0) == int(ready)
    assert getattr(runner, "_gr_input_prep_waits", 0) == int(not ready)


@pytest.mark.parametrize("failure", ["query", "synchronize"])
def test_input_prep_failed_completion_never_grants_host_write(failure):
    runner, original, calls = _input_prep_fixture(ready=False, failure=failure)
    patched = gpu_patch.make_patched_synchronize_input_prep(original)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        with patched(runner):
            pytest.fail("failed completion must not release DMA ownership")
    assert calls == (
        ["query"] if failure == "query" else ["query", "original", "synchronize"]
    )


@pytest.mark.parametrize("mode", ["ordinary", "prefill", "dummy", "no_event"])
def test_input_prep_non_decode_delegates_without_query(mode):
    runner, original, calls = _input_prep_fixture(active=mode == "no_event")
    if mode == "ordinary":
        runner._current_scheduler_output.beam_data = {}
    elif mode == "dummy":
        # Deliberately leave stale persistent beam metadata: the scoped
        # physical-geometry marker, not this metadata, controls the fast path.
        runner._current_beam_info = {"beam_data": {"gr": {"is_beam_decode": True}}}
    elif mode == "no_event":
        runner.prepare_inputs_event = None
    with gpu_patch.make_patched_synchronize_input_prep(original)(runner):
        calls.append("body")
    assert calls == (
        ["original", "body"]
        if mode == "no_event"
        else ["original", "synchronize", "body", "record"]
    )
    assert not hasattr(runner, "_gr_input_prep_ready_skips")
    assert not hasattr(runner, "_gr_input_prep_waits")


@pytest.mark.parametrize("execute_error", [True, False])
def test_input_prep_execute_scope_and_mode_handoffs(monkeypatch, execute_error):
    runner, original_prep, calls = _input_prep_fixture(active=False)
    event = runner.prepare_inputs_event
    guarded = gpu_patch.make_patched_synchronize_input_prep(original_prep)
    output = SimpleNamespace(
        num_scheduled_tokens={"gr": 1}, total_num_scheduled_tokens=1,
        beam_data={"gr": {
            "persistent_runner_state": True, "is_beam_decode": True, "beam_width": 4,
        }},
        gr_outputs={"gr": SimpleNamespace(logical_token_count=1, physical_row_count=4)},
    )

    def common_patcher(original_execute):
        def common_execute(self, scheduler_output):
            self._current_scheduler_output = scheduler_output
            return original_execute(self, scheduler_output)
        return common_execute

    monkeypatch.setattr(gpu_patch, "_make_common_patched_execute_model", common_patcher)
    monkeypatch.setattr(gpu_patch, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(gpu_patch, "has_ec_transfer", lambda: False)

    def original_execute(self, scheduler_output):
        with guarded(self):
            calls.append("decode")
            if execute_error:
                raise RuntimeError("execute failed")

    with guarded(runner):
        calls.append("prefill")
    execute = gpu_patch.make_patched_execute_model_gpu(original_execute)
    if execute_error:
        with pytest.raises(RuntimeError, match="execute failed"):
            execute(runner, output)
    else:
        execute(runner, output)
    assert runner._current_scheduler_output is output  # intentionally retained upstream context
    assert not hasattr(output, "_gr_cuda_logical_geometry")
    with guarded(runner):
        calls.append("later_dummy")
    assert runner.prepare_inputs_event is event
    assert calls == [
        "original", "synchronize", "prefill", "record",
        "query", "decode", "record",
        "original", "synchronize", "later_dummy", "record",
    ]


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


def test_gpu_prepare_delegates_persistent_device_writes_to_backend_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class State:
        next_token_ids = torch.tensor([10, 11])

        def prepare_decode_inputs(self, runner, output, counts, metadata) -> None:
            assert runner.input_ids.gpu.tolist() == [9, 9, 9, 9]
            assert output is scheduler_output
            assert counts == [2, 2]
            assert metadata["beam_width"] == 2
            events.append("backend-prepare")

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["plain", "gr-0"]),
        input_ids=SimpleNamespace(gpu=torch.full((4,), -1, dtype=torch.int32)),
        _gr_model_runner_states={"gr-0": State()},
        _current_beam_info={"req_ids": []},
        device=torch.device("cpu"),
    )
    scheduler_output = SimpleNamespace(
        beam_data={
            "gr-0": {
                "persistent_request": True,
                "persistent_runner_state": True,
                "is_beam_decode": True,
                "beam_width": 2,
            },
        }
    )
    monkeypatch.setattr(gpu_patch, "_beam_remap_inputs", lambda *_args: [])
    monkeypatch.setattr(gpu_patch, "prepare_cuda_prefill_constraint", lambda *_args: None)

    def original(*_args):
        events.append("original")
        runner.input_ids.gpu.fill_(9)
        return None, None

    wrapped = gpu_patch.make_patched_prepare_inputs(original)
    wrapped(runner, scheduler_output, [2, 2])

    assert events == ["original", "backend-prepare"]
    assert runner.input_ids.gpu.tolist() == [9, 9, 9, 9]


@pytest.mark.parametrize("raises", [False, True])
def test_gpu_execute_publishes_physical_geometry_only_inside_worker(
    monkeypatch: pytest.MonkeyPatch,
    raises: bool,
) -> None:
    observations: list[tuple[int, int, tuple[str, int, int]]] = []

    def original(self: Any, scheduler_output: Any) -> str:
        del self
        observations.append(
            (
                scheduler_output.num_scheduled_tokens["gr"],
                scheduler_output.total_num_scheduled_tokens,
                scheduler_output._gr_cuda_logical_geometry,
            )
        )
        if raises:
            raise RuntimeError("worker failed")
        return "ok"

    monkeypatch.setattr(
        gpu_patch,
        "_make_common_patched_execute_model",
        lambda received: received,
    )
    monkeypatch.setattr(gpu_patch, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(gpu_patch, "has_ec_transfer", lambda: False)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"gr": 1},
        total_num_scheduled_tokens=1,
        beam_data={
            "gr": {
                "persistent_runner_state": True,
                "is_beam_decode": True,
                "beam_width": 4,
            }
        },
        gr_outputs={
            "gr": SimpleNamespace(logical_token_count=1, physical_row_count=4)
        },
    )
    wrapped = gpu_patch.make_patched_execute_model_gpu(original)
    if raises:
        with pytest.raises(RuntimeError, match="worker failed"):
            wrapped(SimpleNamespace(), scheduler_output)
    else:
        assert wrapped(SimpleNamespace(), scheduler_output) == "ok"

    assert observations == [(4, 4, ("gr", 1, 4))]
    assert scheduler_output.num_scheduled_tokens == {"gr": 1}
    assert scheduler_output.total_num_scheduled_tokens == 1
    assert not hasattr(scheduler_output, "_gr_cuda_logical_geometry")


@pytest.mark.parametrize(
    ("kv_transfer", "ec_transfer", "message"),
    [
        (True, False, "active KV connector"),
        (False, True, "active EC transfer"),
    ],
)
def test_gpu_execute_rejects_persistent_transfer_modes_before_common_execute(
    monkeypatch: pytest.MonkeyPatch,
    kv_transfer: bool,
    ec_transfer: bool,
    message: str,
) -> None:
    calls: list[str] = []

    def original(*_args: Any) -> None:
        calls.append("common")

    monkeypatch.setattr(
        gpu_patch,
        "_make_common_patched_execute_model",
        lambda received: received,
    )
    monkeypatch.setattr(gpu_patch, "has_kv_transfer_group", lambda: kv_transfer)
    monkeypatch.setattr(gpu_patch, "has_ec_transfer", lambda: ec_transfer)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"gr": 1},
        total_num_scheduled_tokens=1,
        beam_data={
            "gr": {
                "persistent_runner_state": True,
                "is_beam_decode": True,
                "beam_width": 4,
            }
        },
        gr_outputs={
            "gr": SimpleNamespace(logical_token_count=1, physical_row_count=4)
        },
    )

    with pytest.raises(RuntimeError, match=message):
        gpu_patch.make_patched_execute_model_gpu(original)(
            SimpleNamespace(),
            scheduler_output,
        )

    assert calls == []
    assert scheduler_output.num_scheduled_tokens == {"gr": 1}
    assert scheduler_output.total_num_scheduled_tokens == 1
    assert not hasattr(scheduler_output, "_gr_cuda_logical_geometry")


def test_gpu_execute_transfer_modes_do_not_affect_ordinary_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_patch,
        "_make_common_patched_execute_model",
        lambda received: received,
    )
    monkeypatch.setattr(
        gpu_patch,
        "has_kv_transfer_group",
        lambda: (_ for _ in ()).throw(AssertionError("must not query KV transfer")),
    )
    monkeypatch.setattr(
        gpu_patch,
        "has_ec_transfer",
        lambda: (_ for _ in ()).throw(AssertionError("must not query EC transfer")),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"ordinary": 1},
        total_num_scheduled_tokens=1,
        beam_data={},
    )

    result = gpu_patch.make_patched_execute_model_gpu(
        lambda *_args: "ordinary"
    )(SimpleNamespace(), scheduler_output)

    assert result == "ordinary"


def test_gpu_execute_rejects_mixed_persistent_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_patch,
        "_make_common_patched_execute_model",
        lambda received: received,
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"gr": 1, "ordinary": 1},
        total_num_scheduled_tokens=2,
        beam_data={
            "gr": {
                "persistent_runner_state": True,
                "is_beam_decode": True,
                "beam_width": 4,
            }
        },
        gr_outputs={
            "gr": SimpleNamespace(logical_token_count=1, physical_row_count=4)
        },
    )
    wrapped = gpu_patch.make_patched_execute_model_gpu(lambda *_args: None)
    with pytest.raises(RuntimeError, match="isolated B=1"):
        wrapped(SimpleNamespace(), scheduler_output)


def test_cuda_backend_control_write_requires_active_lease() -> None:
    workspace = SimpleNamespace(
        gr_control_output_0=torch.empty((1, 1), dtype=torch.int32),
        gr_control_output_1=torch.empty((1, 1), dtype=torch.int32),
    )
    with pytest.raises(RuntimeError, match="active persistent sampling lease"):
        gpu_patch.CudaGRDeviceBackend._control_output(workspace, 1, 7)


@pytest.fixture
def isolated_control_transport(monkeypatch):
    """Only algorithm-isolated CPU tests stub the separately tested transport."""
    from vllm_gr.v1.worker import cuda_async_output

    def fill(workspace, device_step, token_id):
        slot = getattr(workspace, f"gr_control_output_{device_step % 2}")
        slot.fill_(token_id)
        return slot

    monkeypatch.setattr(cuda_async_output, "fill_control", fill)


def test_cuda_backend_bootstrap_initializes_authoritative_decision_without_d2h(
    monkeypatch: pytest.MonkeyPatch,
    isolated_control_transport,
) -> None:
    from vllm_gr.v1.beam import beam_decision

    workspace = beam_decision.BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=2,
        top_k=2,
        max_decode_steps=3,
        device="cpu",
        logits_dtype=torch.float16,
        pin_host=False,
    )
    decision_state = SimpleNamespace(
        sequence=torch.zeros((1, 2, 3), dtype=torch.int32),
        beam_scores=torch.zeros((1, 2), dtype=torch.float32),
        worker_decode_steps=0,
    )
    state = SimpleNamespace(
        beam_width=2,
        max_tokens=3,
        finish_token_id=9,
        continue_token_id=7,
        session=SimpleNamespace(context=SimpleNamespace(decision=decision_state)),
        backend_state=None,
        prefix_binding=None,
    )
    sampler_output = SimpleNamespace(_gr_deferred_prefill_status=False)
    token_ids = torch.tensor([[11, 12]], dtype=torch.int32)
    scores = torch.tensor([[-0.1, -0.2]], dtype=torch.float32)
    backend = gpu_patch.CudaGRDeviceBackend()
    monkeypatch.setattr(backend, "_require_workspace", lambda *_args: workspace)
    monkeypatch.setattr(backend, "_extract_prefix_binding", lambda *_args: (object(),))
    monkeypatch.setattr(
        beam_decision,
        "_run_cuda_prefill_decision",
        lambda *_args: {"out_token_ids": token_ids, "out_log_probs": scores},
    )

    step = backend.bootstrap_from_prefill(
        SimpleNamespace(), state, sampler_output, {"generation_index": 0}
    )

    assert decision_state.worker_decode_steps == 1
    assert decision_state.sequence.tolist() == [[[11, 0, 0], [12, 0, 0]]]
    torch.testing.assert_close(decision_state.beam_scores, scores)
    assert step.next_token_ids is token_ids
    assert step.beam_scores.data_ptr() == decision_state.beam_scores.data_ptr()
    assert step.control_output is workspace.gr_control_output_1
    assert not any(
        marker in inspect.getsource(gpu_patch.CudaGRDeviceBackend.bootstrap_from_prefill)
        for marker in (".cpu(", ".tolist(", ".item(", "synchronize(")
    )


def test_cuda_persistent_bootstrap_then_advance_preserves_parent_scores_once(
    monkeypatch: pytest.MonkeyPatch,
    isolated_control_transport,
) -> None:
    """First decode ranks prefill parents plus one local score exactly once.

    This is the adversarial precision case from 89f4178 expressed through the
    new persistent state/backend boundary.  Parent 1 owns the best local
    decode candidate, but its much worse prefill score means parent 0 must
    still win globally.  Initializing from the decode matrix would instead
    drop the prefill scores and count the local score twice.
    """
    from vllm_gr.v1.beam import BeamAttentionContext, BeamSearchSession, beam_decision
    from vllm_gr.v1.worker.gr_model_runner_state import (
        GR_FINAL_OUTPUT_FORMAT,
        GRModelRunnerState,
        unpack_gr_final_output,
    )

    workspace = beam_decision.BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=2,
        top_k=2,
        max_decode_steps=2,
        device="cpu",
        logits_dtype=torch.float16,
        pin_host=False,
    )
    context = BeamAttentionContext(
        max_batch=1,
        max_beams=2,
        num_kv_heads=1,
        max_decode_steps=2,
        head_dim=2,
        num_layers=1,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    slot = context.registry.allocate("gr", beam_width=2, prefix_len=8)
    session = BeamSearchSession(session_id="gr", slot=slot, context=context)
    session.begin_session(beam_width=2, prefix_len=8)
    backend = gpu_patch.CudaGRDeviceBackend()
    state = GRModelRunnerState(
        request_id="gr",
        session=session,
        backend=backend,
        beam_width=2,
        max_tokens=2,
        continue_token_id=7,
        finish_token_id=9,
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(req_ids=["gr"]),
        input_ids=SimpleNamespace(gpu=torch.full((2,), -1, dtype=torch.int32)),
        _current_beam_info={
            "beam_data": {"gr": {"is_beam_decode": True, "parent_beam_ids": None}}
        },
    )
    prefill_tokens = torch.tensor([[10, 20]], dtype=torch.int32)
    prefill_scores = torch.tensor([[-0.1, -5.0]], dtype=torch.float32)
    monkeypatch.setattr(backend, "_require_workspace", lambda *_args: workspace)
    monkeypatch.setattr(
        backend,
        "_extract_prefix_binding",
        lambda *_args: (torch.tensor([31], dtype=torch.int32),),
    )
    monkeypatch.setattr(
        beam_decision,
        "_run_cuda_prefill_decision",
        lambda *_args: {
            "out_token_ids": prefill_tokens,
            "out_log_probs": prefill_scores,
        },
    )

    state.start_prefill()
    state.bootstrap_from_prefill(
        runner,
        SimpleNamespace(_gr_deferred_prefill_status=False),
        {"generation_index": 0},
    )
    scheduler_output = SimpleNamespace(
        gr_outputs={
            "gr": SimpleNamespace(logical_token_count=1, physical_row_count=2)
        }
    )
    decode_metadata = {"generation_index": 1, "result_width": 2}
    state.prepare_decode_inputs(runner, scheduler_output, [2], decode_metadata)
    sampler_output = SimpleNamespace(
        logprobs_tensors=SimpleNamespace(
            logprob_token_ids=torch.tensor(
                [[11, 12], [21, 22]], dtype=torch.int32
            ),
            logprobs=torch.tensor(
                [[-4.0, -6.0], [-0.1, -1.0]], dtype=torch.float32
            ),
        )
    )

    assert state.advance_beam(runner, sampler_output, decode_metadata) is True

    staging, payload_words, width = backend._pack_final_result(state, workspace)
    payload = staging[workspace.max_batch : workspace.max_batch + payload_words]
    decoded = unpack_gr_final_output(
        {
            "buffer_format": GR_FINAL_OUTPUT_FORMAT,
            "shape": [width, state.step],
            "buffer": payload.numpy().tobytes(),
        }
    )
    assert decoded is not None
    sequences, parents, per_token_logprobs, cumulative_scores = decoded
    assert sequences == [[10, 11], [20, 21]]
    assert parents == [[0, 0], [0, 1]]
    torch.testing.assert_close(
        torch.tensor(per_token_logprobs),
        torch.tensor([[-0.1, -4.0], [-5.0, -0.1]]),
    )
    assert cumulative_scores == pytest.approx([-4.1, -5.1])


def test_cuda_backend_does_not_reorder_whole_history_per_step() -> None:
    source = inspect.getsource(gpu_patch.CudaGRDeviceBackend.advance_beam)
    assert source.count("torch.index_select") == 1
    assert "gr_parent_history[0, :result_width, state.step]" in source
    assert "gr_logprob_history[0, :result_width, state.step]" in source


def test_cuda_backend_extracts_stable_view_from_vllm_multigroup_block_table() -> None:
    from vllm.v1.worker.block_table import MultiGroupBlockTable

    block_table = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=128,
        max_num_batched_tokens=32,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16],
        kernel_block_sizes=[16],
    )
    block_table.add_row(([31, 32, 33],), 0)
    block_table.commit_block_table(1)
    input_batch = SimpleNamespace(
        req_id_to_index={"gr": 0},
        block_table=block_table,
        num_reqs=1,
    )
    runner = SimpleNamespace(input_batch=input_batch)
    state = SimpleNamespace(
        request_id="gr",
        session=SimpleNamespace(_state=SimpleNamespace(prefix_len=31)),
    )

    binding = gpu_patch.CudaGRDeviceBackend._extract_prefix_binding(runner, state)

    assert len(binding) == 1
    assert binding[0].tolist() == [31, 32]
    assert binding[0].data_ptr() == block_table[0].get_device_tensor(1)[0].data_ptr()
    binding_ptr = binding[0].data_ptr()

    # A later native block-table commit updates the same fixed storage; it
    # neither reallocates nor invalidates the prefix-only row view retained by
    # the persistent GR state.
    block_table.append_row(([34],), 0)
    block_table.commit_block_table(1)
    assert binding[0].data_ptr() == binding_ptr
    assert binding[0].tolist() == [31, 32]


def test_cuda_backend_rejects_multiple_vllm_kv_cache_groups() -> None:
    from vllm.v1.worker.block_table import MultiGroupBlockTable

    block_table = MultiGroupBlockTable(
        max_num_reqs=1,
        max_model_len=128,
        max_num_batched_tokens=32,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16, 32],
        kernel_block_sizes=[16, 32],
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"gr": 0},
            block_table=block_table,
            num_reqs=1,
        )
    )
    state = SimpleNamespace(
        request_id="gr",
        session=SimpleNamespace(_state=SimpleNamespace(prefix_len=31)),
    )

    with pytest.raises(RuntimeError, match="one native KV-cache group"):
        gpu_patch.CudaGRDeviceBackend._extract_prefix_binding(runner, state)


def test_cuda_backend_prepare_writes_stable_input_and_publishes_prefix() -> None:
    workspace = SimpleNamespace()
    prefix = (torch.tensor([3, 4], dtype=torch.int32),)
    context = SimpleNamespace(decode_step_buf=torch.full((1,), -1, dtype=torch.int32))
    session = SimpleNamespace(decode_step=1, context=context)
    state = SimpleNamespace(
        request_id="gr",
        beam_width=2,
        step=2,
        next_token_ids=torch.tensor([[41, 42]], dtype=torch.int32),
        session=session,
        prefix_binding=prefix,
        backend_state=gpu_patch._CudaGRRequestState(workspace=workspace),
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["gr"]),
        input_ids=SimpleNamespace(gpu=torch.full((4,), -1, dtype=torch.int32)),
    )
    sidecar = SimpleNamespace(logical_token_count=1, physical_row_count=2)
    output = SimpleNamespace(gr_outputs={"gr": sidecar})

    gpu_patch.CudaGRDeviceBackend().prepare_decode_inputs(
        runner, state, output, [2], {"generation_index": 2}
    )

    assert runner.input_ids.gpu.tolist() == [41, 42, -1, -1]
    assert context.decode_step_buf.tolist() == [1]
    assert context._gr_persistent_prefix_binding is prefix

    state.step = 3
    session.decode_step = 2
    state.next_token_ids = torch.tensor([[51, 52]], dtype=torch.int32)
    gpu_patch.CudaGRDeviceBackend().prepare_decode_inputs(
        runner, state, output, [2], {"generation_index": 3}
    )
    assert runner.input_ids.gpu.tolist() == [51, 52, -1, -1]
    assert context._gr_persistent_prefix_binding is prefix


def test_cuda_backend_release_clears_all_prefix_binding_owners() -> None:
    prefix = object()
    backend_state = gpu_patch._CudaGRRequestState(
        workspace=object(),
        prefix_binding=prefix,
        pending_reorder_parents=object(),
        pending_reorder_prefix=object(),
    )
    context = SimpleNamespace(_gr_persistent_prefix_binding=prefix)
    state = SimpleNamespace(
        backend_state=backend_state,
        prefix_binding=prefix,
        session=SimpleNamespace(context=context),
    )

    gpu_patch.CudaGRDeviceBackend().release(state)

    assert state.prefix_binding is None
    assert backend_state.prefix_binding is None
    assert backend_state.pending_reorder_parents is None
    assert backend_state.pending_reorder_prefix is None
    assert context._gr_persistent_prefix_binding is None
    assert backend_state.released is True


def test_cuda_backend_advance_reorders_once_and_writes_one_raw_column(
    monkeypatch: pytest.MonkeyPatch,
    isolated_control_transport,
) -> None:
    from vllm_gr.v1.beam import beam_decision

    workspace = beam_decision.BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=2,
        top_k=2,
        max_decode_steps=3,
        device="cpu",
        logits_dtype=torch.float16,
        pin_host=False,
    )
    decision_state = SimpleNamespace(
        sequence=torch.zeros((1, 2, 3), dtype=torch.int32),
        beam_scores=torch.tensor([[-0.1, -0.2]], dtype=torch.float32),
        worker_decode_steps=1,
    )
    session_state = SimpleNamespace(is_last_step=False)
    session = SimpleNamespace(
        context=SimpleNamespace(decision=decision_state),
        _state=session_state,
    )
    state = SimpleNamespace(
        step=1,
        device_step=1,
        beam_width=2,
        max_tokens=3,
        continue_token_id=7,
        finish_token_id=9,
        session=session,
        backend_state=gpu_patch._CudaGRRequestState(workspace=workspace),
    )
    tokens = torch.tensor([[20, 21]], dtype=torch.int32)
    parents = torch.tensor([[1, 0]], dtype=torch.int32)
    scores = torch.tensor([[-0.5, -0.6]], dtype=torch.float32)
    prefix = torch.tensor([[1, 2]], dtype=torch.int32)

    def decide(*_args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        assert kwargs["reorder_kv"] is False
        decision_state.beam_scores.copy_(scores)
        decision_state.worker_decode_steps = 2
        return {
            "out_token_ids": tokens,
            "out_token_index": parents,
            "out_log_probs": scores,
            "_gr_reorder_parents": parents,
            "_gr_reorder_prefix": prefix,
        }

    reorder_calls: list[tuple[Any, Any]] = []
    monkeypatch.setattr(beam_decision, "run_beam_decision", decide)
    monkeypatch.setattr(
        beam_decision,
        "_reorder_kv_same_step",
        lambda received_session, received_parent, received_prefix: reorder_calls.append(
            (received_parent, received_prefix)
        ),
    )
    backend = gpu_patch.CudaGRDeviceBackend()
    result = backend.advance_beam(
        SimpleNamespace(), state, SimpleNamespace(), {"result_width": 2}, is_last_step=False
    )
    backend.reorder_kv(SimpleNamespace(), state, result, is_last_step=False)

    assert len(reorder_calls) == 1
    assert reorder_calls[0][0].data_ptr() == parents.data_ptr()
    assert reorder_calls[0][1].data_ptr() == prefix.data_ptr()
    assert workspace.gr_parent_history[0, :, 1].tolist() == [1, 0]
    torch.testing.assert_close(
        workspace.gr_logprob_history[0, :, 1],
        torch.tensor([-0.3, -0.5]),
    )
    assert result.control_output is workspace.gr_control_output_0


def test_cuda_backend_terminal_advance_does_not_reorder(
    monkeypatch: pytest.MonkeyPatch,
    isolated_control_transport,
) -> None:
    from vllm_gr.v1.beam import beam_decision

    workspace = beam_decision.BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=2,
        top_k=2,
        max_decode_steps=2,
        device="cpu",
        logits_dtype=torch.float16,
        pin_host=False,
    )
    scores = torch.tensor([[-0.1, -0.2]], dtype=torch.float32)
    decision_state = SimpleNamespace(
        sequence=torch.zeros((1, 2, 2), dtype=torch.int32),
        beam_scores=scores.clone(),
        worker_decode_steps=1,
    )
    state = SimpleNamespace(
        step=1,
        device_step=1,
        beam_width=2,
        max_tokens=2,
        continue_token_id=7,
        finish_token_id=9,
        session=SimpleNamespace(
            context=SimpleNamespace(decision=decision_state),
            _state=SimpleNamespace(is_last_step=False),
        ),
        backend_state=gpu_patch._CudaGRRequestState(workspace=workspace),
    )
    final_scores = torch.tensor([[-0.4]], dtype=torch.float32)

    def decide(*_args: Any, **_kwargs: Any) -> dict[str, torch.Tensor]:
        decision_state.beam_scores[:, :1].copy_(final_scores)
        decision_state.worker_decode_steps = 2
        return {
            "out_token_ids": torch.tensor([[22]], dtype=torch.int32),
            "out_token_index": torch.tensor([[1]], dtype=torch.int32),
            "out_log_probs": final_scores,
        }

    monkeypatch.setattr(beam_decision, "run_beam_decision", decide)
    backend = gpu_patch.CudaGRDeviceBackend()
    result = backend.advance_beam(
        SimpleNamespace(), state, SimpleNamespace(), {"result_width": 1}, is_last_step=True
    )
    backend.reorder_kv(SimpleNamespace(), state, result, is_last_step=True)

    assert result.next_token_ids.shape == (1, 1)
    assert result.control_output is workspace.gr_control_output_0


def test_cuda_backend_final_pack_backtraces_raw_history() -> None:
    from vllm_gr.v1.beam.beam_decision import BeamPostprocessWorkspace
    from vllm_gr.v1.worker.gr_model_runner_state import (
        GR_FINAL_OUTPUT_FORMAT,
        unpack_gr_final_output,
    )

    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=2,
        top_k=2,
        max_decode_steps=3,
        device="cpu",
        logits_dtype=torch.float16,
        pin_host=False,
    )
    workspace.gr_invalid_status.zero_()
    workspace.gr_parent_history[0] = torch.tensor(
        [[0, 1, 1], [0, 0, 0]], dtype=torch.int32
    )
    workspace.gr_logprob_history[0] = torch.tensor(
        [[-0.1, -0.3, -0.5], [-0.2, -0.4, -0.6]], dtype=torch.float32
    )
    sequence = torch.tensor([[[10, 20, 30], [11, 21, 31]]], dtype=torch.int32)
    scores = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
    state = SimpleNamespace(
        next_token_ids=torch.empty((1, 2), dtype=torch.int32),
        step=3,
        beam_scores=scores,
        session=SimpleNamespace(
            context=SimpleNamespace(decision=SimpleNamespace(sequence=sequence))
        ),
    )

    staging, payload_words, width = gpu_patch.CudaGRDeviceBackend._pack_final_result(
        state, workspace
    )
    payload = staging[workspace.max_batch : workspace.max_batch + payload_words]
    decoded = unpack_gr_final_output(
        {
            "buffer_format": GR_FINAL_OUTPUT_FORMAT,
            "shape": [width, state.step],
            "buffer": payload.numpy().tobytes(),
        }
    )
    assert decoded is not None
    sequences, parents, logprobs, final_scores = decoded
    assert sequences == [[10, 20, 30], [11, 21, 31]]
    assert parents == [[0, 0, 1], [0, 1, 0]]
    torch.testing.assert_close(
        torch.tensor(logprobs),
        torch.tensor([[-0.1, -0.4, -0.5], [-0.2, -0.3, -0.6]]),
    )
    assert final_scores == pytest.approx([-1.0, -2.0])


def test_cuda_backend_invalid_status_is_checked_only_at_finalization() -> None:
    gpu_patch.CudaGRDeviceBackend._check_final_status([0])
    with pytest.raises(RuntimeError, match="no legal candidate"):
        gpu_patch.CudaGRDeviceBackend._check_final_status([1])


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


@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_patched_determine_pure_beam_batch_grants_full(
    enable_prefix_caching: bool,
) -> None:
    """FULL graph dispatch is independent of cross-request prefix caching."""
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
        _vllm_gr_config=SimpleNamespace(beam_graph_enabled=True),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching)
        ),
    )
    wrapped = _make_determine_wrapper(beam_info, native)
    result = wrapped(runner, num_tokens=4)
    assert result[0] == CUDAGraphMode.FULL
    assert result[1].num_tokens == 4
    assert result[1].num_reqs == 1
    assert result[1].uniform is True
    assert runner._beam_graph_full_dispatch_count == 1
    wrapped(runner, num_tokens=4)
    assert runner._beam_graph_full_dispatch_count == 2


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
        _gr_input_prep_ready_skips = 6
        _gr_input_prep_waits = 2

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
    assert result["input_prep"] == {
        "ready_event_skips": 6,
        "not_ready_upstream_waits": 2,
    }


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
    # Keep the modules and Stream/Event types intact: capture performs lazy
    # imports whose annotations and compiler setup inspect those symbols.
    # Intercept allocation, not the Stream class, so this remains CPU-only
    # and independent of which imports earlier tests happened to populate.
    monkeypatch.setattr(
        torch.cuda.Stream, "__new__", lambda cls, *args, **kwargs: side_stream
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _s: contextlib.nullcontext())
    monkeypatch.setattr(
        torch.accelerator, "current_stream",
        lambda: SimpleNamespace(wait_stream=lambda *_: None),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

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

            assert batch_descriptor == BatchDescriptor(
                num_tokens=4,
                num_reqs=1,
                uniform=True,
            )


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
        BatchDescriptor(num_tokens=4, num_reqs=1, uniform=True): mock_entry
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
        BatchDescriptor(num_tokens=8, num_reqs=1, uniform=True): mock_entry
    }

    runner = SimpleNamespace(_beam_graph_keys={4})

    with patch("vllm.compilation.cuda_graph.CUDAGraphWrapper._all_instances", [mock_wrapper]):
        result = gpu_patch._verify_and_record_captured_graph(runner, width=8)

    assert result is True
    assert runner._beam_graph_keys == {4, 8}


def test_beam_graph_descriptor_rejects_preexisting_native_full_graph() -> None:
    from unittest.mock import MagicMock, patch

    from vllm.v1.cudagraph_dispatcher import BatchDescriptor, CUDAGraphMode

    occupied_entry = MagicMock()
    occupied_entry.cudagraph = MagicMock()
    occupied = MagicMock()
    occupied.runtime_mode = CUDAGraphMode.FULL
    occupied.concrete_cudagraph_entries = {
        BatchDescriptor(num_tokens=8, num_reqs=1, uniform=True): occupied_entry
    }
    unrelated = MagicMock()
    unrelated.runtime_mode = CUDAGraphMode.FULL
    unrelated.concrete_cudagraph_entries = {
        BatchDescriptor(num_tokens=4, num_reqs=1, uniform=True): occupied_entry
    }

    with patch(
        "vllm.compilation.cuda_graph.CUDAGraphWrapper._all_instances",
        [unrelated],
    ):
        assert gpu_patch._beam_graph_descriptor_is_free(8) is True
    with patch(
        "vllm.compilation.cuda_graph.CUDAGraphWrapper._all_instances",
        [unrelated, occupied],
    ):
        assert gpu_patch._beam_graph_descriptor_is_free(8) is False


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
