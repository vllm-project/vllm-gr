# SPDX-License-Identifier: Apache-2.0
"""Persistent CUDA lineage checked against an independent host enumeration.

Candidates are synthetic and exactly representable in FP32. The host oracle
never reads intermediate device state, and the backend uses its real final D2H.
This checks the backend protocol/lineage, not model-forward numerical parity.
"""

import ctypes
import random
from types import SimpleNamespace

import pytest
import torch

from tests.test_gr_completion_build import built_completion  # noqa: F401
from vllm_gr.config import ConstraintBackend
from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadataBuilder
from vllm_gr.v1.beam import BeamAttentionContext, BeamSearchSession, beam_decision
from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.beam.decode_geometry import BeamDecodeGeometry
from vllm_gr.v1.worker.gpu_model_runner_patch import (
    CudaGRDeviceBackend,
    make_patched_synchronize_input_prep,
)
from vllm_gr.v1.worker.gr_model_runner_state import (
    GRModelRunnerState,
    unpack_gr_final_output,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


@pytest.mark.parametrize("pending", [False, True])
def test_input_prep_ready_query_preserves_real_pinned_dma_ownership(pending):
    """Use the native V1 guard with an actual delayed read of reused Host data.

    Synchronization/copies outside the guard are test setup/observation only;
    this checks safety of both branches, not the E2E no-sync acceptance gate.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    class CountedEvent:
        def __init__(self):
            self.event = torch.cuda.Event()
            self.queries = []
            self.waits = 0
            self.records = 0

        def query(self):
            ready = self.event.query()
            self.queries.append(ready)
            return ready

        def synchronize(self):
            self.waits += 1
            self.event.synchronize()

        def record(self):
            self.records += 1
            self.event.record()

    host = torch.full((4096,), 27, dtype=torch.int32, pin_memory=True)
    first = torch.empty_like(host, device="cuda")
    second = torch.empty_like(first)
    pointers = host.data_ptr(), first.data_ptr(), second.data_ptr()
    event = CountedEvent()
    runner = SimpleNamespace(
        prepare_inputs_event=event,
        _current_scheduler_output=SimpleNamespace(_gr_cuda_logical_geometry=("gr", 1, 128)),
    )
    guard = make_patched_synchronize_input_prep(GPUModelRunner.synchronize_input_prep)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    try:
        with torch.cuda.stream(stream):
            if pending:
                torch.cuda._sleep(200_000_000)
            first.copy_(host, non_blocking=True)
            event.event.record()
            if not pending:
                stream.synchronize()  # Set up the already-ready branch.
            with guard(runner):
                host.fill_(42)  # Illegal until the first DMA has released it.
                second.copy_(host, non_blocking=True)
        stream.synchronize()
        assert event.queries == [not pending], "requested readiness branch was not exercised"
        assert event.waits == int(pending)
        assert event.records == 1
        assert (host.data_ptr(), first.data_ptr(), second.data_ptr()) == pointers
        assert torch.equal(first.cpu(), torch.full_like(host, 27))
        assert torch.equal(second.cpu(), torch.full_like(host, 42))
    finally:
        stream.synchronize()


@pytest.mark.parametrize("overwrite_key", [("dummy-geometry",), None])
def test_metadata_resident_cache_recovers_graph_inputs_after_overwrite(overwrite_key):
    """Real CUDA buffers, one consumer Graph, and A→B→A metadata preparation."""
    builder = object.__new__(BeamAttentionMetadataBuilder)
    builder.device = torch.device("cuda")
    builder.block_size = 2
    builder.aot_schedule = False
    builder._fixed_buffers = True
    builder._is_cuda = True
    builder._is_npu = False
    for name in (
        "_prefix_seqlens_kv_buf",
        "_prefix_cu_seqlens_q_buf",
        "_suffix_seqlens_kv_buf",
        "_suffix_cu_seqlens_q_buf",
        "_suffix_cu_seqlens_k_buf",
    ):
        setattr(builder, name, torch.zeros(8, dtype=torch.int32, device="cuda"))
    builder._prefix_indices_buf = torch.zeros(8, dtype=torch.long, device="cuda")
    builder._prefix_block_table_buf = torch.zeros((1, 8), dtype=torch.int32, device="cuda")
    builder._h2d_host = torch.zeros(64, dtype=torch.int32, pin_memory=True)
    builder._h2d_dev = torch.zeros(64, dtype=torch.int32, device="cuda")
    prefix_row = torch.arange(8, dtype=torch.int32, device="cuda")
    common = SimpleNamespace(block_table_tensor=prefix_row.unsqueeze(0))
    token = BEAM_DATA_VAR.set(
        {
            "beam_data": {"gr": {"persistent_runner_state": True, "is_beam_decode": True}},
            "beam_session": SimpleNamespace(
                context=SimpleNamespace(
                    _gr_persistent_prefix_binding=(prefix_row,),
                )
            ),
        }
    )

    def prepare(key, prefix_len, width):
        return builder._create_beam_tensors(
            common,
            prefix_indices=list(range(width)),
            prefix_seqlens_q_cpu=[width],
            prefix_cu_seqlens_q_cpu=[width],
            prefix_seqlens_kv_cpu=[prefix_len],
            suffix_seqlens_q_cpu=[1] * width,
            suffix_cu_seqlens_q_cpu=list(range(1, width + 1)),
            suffix_cu_seqlens_kv_cpu=list(range(1, width + 1)),
            suffix_seqlens_kv_cpu=[1] * width,
            suffix_slot_mapping_out_list=[],
            prefix_req_idxs_cpu=[0],
            max_prefix_blocks_count=prefix_len // 2,
            device=builder.device,
            max_num_splits=0,
            geom_key=key,
            decode_step=0 if key is not None else None,
        )

    try:
        initial = prepare(("real-session",), 8, 4)
        sources = (initial[1], initial[3], initial[10])
        outputs = tuple(torch.empty_like(source) for source in sources)
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for output, source in zip(outputs, sources):
                output.copy_(source)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for output, source in zip(outputs, sources):
                output.copy_(source)

        prepare(overwrite_key, 12, 3)
        restored = prepare(("real-session",), 8, 4)
        assert [tensor.data_ptr() for tensor in sources] == [
            restored[index].data_ptr() for index in (1, 3, 10)
        ]
        graph.replay()
        assert [output.cpu().tolist() for output in outputs] == [[8], [0, 4], [0, 1, 2, 3, 4]]
        assert len(builder._geometry_cache()["views"]) == 1
    finally:
        BEAM_DATA_VAR.reset(token)


@pytest.mark.parametrize(
    "initial,updated",
    [
        (BeamDecodeGeometry((4,), (7,)), BeamDecodeGeometry((4,), (11,))),
        (BeamDecodeGeometry((2, 3), (7, 11)), BeamDecodeGeometry((3, 2), (13, 5))),
    ],
)
def test_decode_geometry_refreshes_graph_inputs_without_host_upload(initial, updated):
    """Layout-only Graph test; it does not submit/admit a B>1 GR session."""
    indices = torch.empty(initial.num_rows, dtype=torch.int64, device="cuda")
    prefix_kv = torch.empty(initial.num_requests, dtype=torch.int32, device="cuda")
    prefix_cu = torch.empty(initial.num_requests + 1, dtype=torch.int32, device="cuda")
    suffix_cu = torch.empty(initial.num_rows + 1, dtype=torch.int32, device="cuda")
    sources = (indices, prefix_kv, prefix_cu, suffix_cu)
    destinations = tuple(torch.empty_like(source) for source in sources)

    def prepare(geometry):
        geometry.write_request_indices(indices)
        geometry.write_static_seqlens(prefix_kv, prefix_cu, suffix_cu)

    prepare(initial)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for destination, source in zip(destinations, sources):
            destination.copy_(source)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for destination, source in zip(destinations, sources):
            destination.copy_(source)

    # Only the consumer is captured. New request geometry must update the
    # fixed inputs outside replay, not bake the initial Host constants into it.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        prepare(updated)
    operations = {event.key for event in profile.key_averages()}
    assert not operations.intersection(
        {"aten::to", "aten::_to_copy", "aten::copy_", "aten::_local_scalar_dense"}
    ), operations
    graph.replay()
    actual = [destination.cpu().tolist() for destination in destinations]
    expected_indices = [i for i, width in enumerate(updated.request_widths) for _ in range(width)]
    assert actual == [
        expected_indices,
        list(updated.prefix_lengths),
        list(updated.query_offsets),
        list(range(updated.num_rows + 1)),
    ]


@pytest.mark.parametrize("width,result_width", [(64, 64), (128, 128), (128, 1)])
@pytest.mark.parametrize("deferred", [False, True])
def test_persistent_cuda_final_lineage_matches_enumeration(
    monkeypatch: pytest.MonkeyPatch,
    built_completion,  # noqa: F811 - imported pytest fixture
    width: int,
    result_width: int,
    deferred: bool,
) -> None:
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    from vllm_gr.v1.worker.cuda_async_output import (
        CudaOutputTransport,
        install_async_output_adapter,
        sampling_scope,
    )
    from vllm_gr.v1.worker.cuda_completion import NativeCompletionAPI

    steps = 3
    device = torch.device("cuda", torch.cuda.current_device())
    workspace = beam_decision.BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=width,
        top_k=width,
        max_decode_steps=steps,
        device=device,
        logits_dtype=torch.float16,
        pin_host=True,
    )
    context = BeamAttentionContext(
        max_batch=1,
        max_beams=width,
        num_kv_heads=1,
        max_decode_steps=steps,
        head_dim=16,
        num_layers=1,
        device=device,
        dtype=torch.float16,
    )
    slot = context.registry.allocate("gr", beam_width=width, prefix_len=16)
    session = BeamSearchSession(session_id="gr", slot=slot, context=context)
    session.begin_session(beam_width=width, prefix_len=16)
    backend = CudaGRDeviceBackend()
    state = GRModelRunnerState(
        request_id="gr",
        session=session,
        backend=backend,
        beam_width=width,
        max_tokens=steps,
        continue_token_id=7,
        finish_token_id=9,
    )
    runner = SimpleNamespace(
        device=device,
        _vllm_gr_config=SimpleNamespace(
            persistent_runner_state=True,
            beam_worker_decision=True,
            constraint_backend=ConstraintBackend.CONSTRAINT_TABLE,
        ),
        _current_beam_info={
            "beam_data": {
                "gr": {
                    "is_constraint_prefill": True,
                    "beam_width": width,
                }
            }
        },
        input_batch=SimpleNamespace(
            req_ids=["gr"],
            sampling_metadata=SimpleNamespace(output_token_ids=[]),
        ),
        input_ids=SimpleNamespace(gpu=torch.empty(width, dtype=torch.int32, device=device)),
        use_async_scheduling=True,
        _gr_model_runner_states={"gr": state},
        routed_experts_initialized=False,
    )
    api = NativeCompletionAPI(
        ctypes.CDLL(str(built_completion)),
        ctypes.CDLL(torch._C.__file__),
    )
    transport = CudaOutputTransport(runner, api)
    runner._gr_cuda_output_transport = transport
    install_async_output_adapter(AsyncGPUModelRunnerOutput)
    async_output_copy_stream = torch.cuda.Stream(device=device)

    def construct_control_output() -> AsyncGPUModelRunnerOutput:
        return AsyncGPUModelRunnerOutput(
            ModelRunnerOutput(req_ids=["gr"], req_id_to_index={"gr": 0}),
            state.control_token_ids,
            None,
            [],
            async_output_copy_stream,
            176384,
        )

    monkeypatch.setattr(beam_decision, "get_beam_postprocess_workspace", lambda *_: workspace)
    prefix = (torch.tensor([0], dtype=torch.int32, device=device),)
    monkeypatch.setattr(backend, "_extract_prefix_binding", lambda *_: prefix)

    def sample(tokens, scores, *, decode=False):
        result = SimpleNamespace(
            logprobs_tensors=SimpleNamespace(
                logprob_token_ids=torch.tensor(tokens, dtype=torch.int32, device=device),
                logprobs=torch.tensor(scores, dtype=torch.float32, device=device),
            )
        )
        if deferred and decode:
            result._gr_deferred_decode_status = True
            result._gr_constraint_valid_counts = torch.full(
                (width,), width, dtype=torch.int32, device=device
            )
        return result

    prefill_tokens = list(range(1000, 1000 + width))
    prefill_scores = [-index / 16 for index in range(width)]
    # Oracle records (sequence, cumulative score, lineage, local logprobs).
    expected = [
        ([token], score, [0], [score]) for token, score in zip(prefill_tokens, prefill_scores)
    ]
    rng = random.Random(5405)
    try:
        assert state.control_slots == (None, None)
        state.start_prefill()
        prefill_scheduler_output = SimpleNamespace(
            beam_data={
                "gr": {
                    "persistent_runner_state": True,
                    "is_constraint_prefill": True,
                    "is_last_prefill": True,
                }
            }
        )
        with sampling_scope(runner, prefill_scheduler_output):
            state.bootstrap_from_prefill(
                runner, sample([prefill_tokens], [prefill_scores]), {"generation_index": 0}
            )
            prefill_control = construct_control_output()
        assert prefill_control.get_output().sampled_token_ids == [[7]]
        runner._current_beam_info["beam_data"]["gr"] = {
            "is_beam_decode": True,
            "parent_beam_ids": None,
        }
        for generation in range(1, steps):
            metadata = {"generation_index": generation, "result_width": result_width}
            output = SimpleNamespace(
                gr_outputs={
                    "gr": SimpleNamespace(
                        logical_token_count=1,
                        physical_row_count=width,
                    )
                }
            )
            state.prepare_decode_inputs(runner, output, [width], metadata)
            # Row-sorted candidates exercise both the checked and deferred
            # constrained kernels. Repeated scores exercise stable ties.
            locals_ = [
                sorted([-rng.randrange(256) / 16 for _ in range(width)], reverse=True)
                for _ in range(width)
            ]
            tokens = [
                [generation * 100000 + parent * width + candidate for candidate in range(width)]
                for parent in range(width)
            ]
            ranked = sorted(
                (
                    (expected[parent][1] + locals_[parent][candidate], parent, candidate)
                    for parent in range(width)
                    for candidate in range(width)
                ),
                key=lambda entry: (-entry[0], entry[1], entry[2]),
            )[:width]
            is_final = generation == steps - 1
            if is_final:
                ranked = ranked[:result_width]
            else:
                ranked.sort(key=lambda entry: entry[1])
            next_expected = []
            for score, parent, candidate in ranked:
                sequence, _, lineage, logprobs = expected[parent]
                next_expected.append(
                    (
                        sequence + [tokens[parent][candidate]],
                        score,
                        lineage + [parent],
                        logprobs + [locals_[parent][candidate]],
                    )
                )
            expected = next_expected
            decode_scheduler_output = SimpleNamespace(
                beam_data={
                    "gr": {
                        "persistent_runner_state": True,
                        "is_beam_decode": True,
                        "is_last_step": is_final,
                    }
                }
            )
            with sampling_scope(runner, decode_scheduler_output):
                assert (
                    state.advance_beam(
                        runner,
                        sample(tokens, locals_, decode=True),
                        metadata,
                    )
                    is is_final
                )
                decode_control = construct_control_output()
            expected_control = 9 if is_final else 7
            assert decode_control.get_output().sampled_token_ids == [[expected_control]]

        final = state.finalize_output(runner)
        decoded = unpack_gr_final_output(final)
        assert decoded is not None
        sequences, parents, logprobs, scores = decoded
        assert sequences == [row[0] for row in expected]
        assert scores == [row[1] for row in expected]
        assert parents == [row[2] for row in expected]
        assert logprobs == [row[3] for row in expected]
        assert state.control_slots[0] is workspace.gr_control_output_0
        assert state.control_slots[1] is workspace.gr_control_output_1
        assert state.backend_state.control_phase == 0
    finally:
        try:
            transport.shutdown()
        finally:
            assert state.release()
            assert not state.release()
            assert api.lib.gr_ticket_live() == 0
