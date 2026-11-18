# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Real CUDA capture/replay using the native V2 model factory and PW wrapper.

The toy model covers reload/recovery; a boundary test uses real Prefill attention
and native dummy inputs. OneRec ST covers full-model attention and KV state.
A controlled FULL-only exception exercises recovery inside a real capture.
Reload replaces model weights, FULL/PW graphs and fixed output storage.
A separate process checks that a real CUDA invalidation is never swallowed.
"""

import gc
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CompilationConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode as Mode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.cudagraph_dispatcher import BatchDescriptor
from vllm.v1.worker.gpu import cudagraph_utils as native

from vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_resources import close_v2_runtime
from vllm_gr.v1.worker.cuda.v2.gpu_prefill_graph_runner_v2 import (
    PrefillCapturePlan,
    install_prefill_graph_manager_v2,
)
from vllm_gr.v1.worker.model_runner_patch import make_load_model_with_constraint_backend
from vllm_gr.v1.worker.prefill_graph_common import PrefillGraphEnv


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")  # type: ignore[misc]
@pytest.mark.parametrize("fail_full", [False, True])  # type: ignore[misc]
def test_native_capture_replay_fixed_output_and_piecewise_recovery(
    monkeypatch: pytest.MonkeyPatch, fail_full: bool | str
) -> None:
    from vllm.compilation import monitor

    gc.collect()
    # The production runner opens this window before calling the manager.
    # Other graph tests may have closed it earlier in the same process.
    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", True)
    monkeypatch.setattr(PrefillGraphEnv, "enabled", True)
    monkeypatch.delenv("VLLM_GR_PREFILL_GRAPH_BUCKET_STEP", raising=False)
    monkeypatch.delenv("VLLM_GR_PREFILL_GRAPH_MAX_BUCKET", raising=False)
    config = VllmConfig(
        compilation_config=CompilationConfig(cudagraph_mode=Mode.FULL_AND_PIECEWISE)
    )
    config.scheduler_config.max_num_seqs = 1
    config.scheduler_config.max_num_batched_tokens = 256
    config.compilation_config.cudagraph_capture_sizes = [1, 16, 128, 256]
    device = torch.device("cuda:0")
    monkeypatch.setattr(
        native, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True)
    )
    monkeypatch.setattr(native, "is_global_first_rank", lambda: False)

    @contextmanager
    def graph_capture(device: torch.device) -> Iterator[None]:
        stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            yield
        current.wait_stream(stream)

    monkeypatch.setattr(native, "graph_capture", graph_capture)
    monkeypatch.setattr(
        native,
        "prepare_inputs_to_capture",
        lambda *args, **kwargs: native.AttentionState({}, {}),
    )
    monkeypatch.setattr(native, "has_compiled_submodule", lambda model: True)
    original = native.ModelCudaGraphManager(config, device, Mode.FULL_AND_PIECEWISE, 1)
    runner = SimpleNamespace(
        cudagraph_manager=original,
        vllm_config=config,
        scheduler_config=config.scheduler_config,
        model_config=SimpleNamespace(max_model_len=1024, enforce_eager=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            use_ubatching=False,
        ),
        lora_config=None,
        speculator=None,
        use_aux_hidden_state_outputs=False,
        supports_mm_inputs=False,
        req_states=SimpleNamespace(
            req_id_to_index={"r": 0},
            num_computed_tokens_np=[512],
            prompt_len=SimpleNamespace(np=[1024]),
        ),
    )
    install_prefill_graph_manager_v2(runner)
    manager = runner.cudagraph_manager
    ids = torch.zeros(256, device=device)
    inputs = SimpleNamespace(
        input_ids=ids,
        positions=torch.zeros_like(ids),
        is_padding=torch.zeros(256, device=device, dtype=torch.bool),
    )

    class Model(torch.nn.Module):  # type: ignore[misc]
        def __init__(self, offset: float, failure: bool | str = False) -> None:
            super().__init__()
            self.offset = torch.nn.Parameter(torch.tensor(offset, device=device))
            self.failure = failure
            self.partition = CUDAGraphWrapper(self.compute, config, Mode.PIECEWISE)

        def compute(self, x: torch.Tensor) -> torch.Tensor:
            return (x[:, None] + self.offset).expand(-1, 4).contiguous()

        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            if (
                self.failure
                and input_ids.numel() == 128
                and get_forward_context().cudagraph_runtime_mode == Mode.NONE
                and torch.cuda.is_current_stream_capturing()
            ):
                if self.failure == "invalidated":
                    input_ids[0].item()  # Real CUDA error 900 invalidates capture.
                self.partition(input_ids)
                raise RuntimeError("operation not supported when stream is capturing")
            return cast(torch.Tensor, self.partition(input_ids))

    model = runner.model = Model(3.0, fail_full)
    partition = model.partition

    state = SimpleNamespace(prepare_dummy_inputs=lambda *args: {})
    with torch.inference_mode():
        if fail_full == "invalidated":
            with pytest.raises(RuntimeError, match="previous error during capture"):
                manager.capture(model, state, inputs, None, None, [], None)
            assert not manager._graphs_captured
            return
        manager.capture(model, state, inputs, None, None, [], None)
        address = manager.hidden_states.data_ptr()
        assert manager.hidden_states.shape == (256, 4)
        assert len(partition.concrete_cudagraph_entries) == 4
        assert all(
            entry.cudagraph is not None for entry in partition.concrete_cudagraph_entries.values()
        )
        assert set(manager.full_capture_failures) == (
            {native.BatchExecutionDescriptor(Mode.FULL, 128, 1)} if fail_full else set()
        )
        with manager.prefill_scope(SimpleNamespace(num_scheduled_tokens={"r": 128})):
            for tokens in (128, 1, 256, 16, 128):
                desc = manager.dispatch(1, tokens, None)
                ids.fill_(tokens)
                if tokens == 128 and fail_full:
                    assert desc.cg_mode == Mode.PIECEWISE
                    with set_forward_context(
                        {},
                        config,
                        num_tokens=tokens,
                        cudagraph_runtime_mode=desc.cg_mode,
                        batch_descriptor=BatchDescriptor(num_tokens=tokens),
                        slot_mapping={},
                    ):
                        result = model(ids[:tokens], inputs.positions[:tokens])
                else:
                    assert desc.cg_mode == Mode.FULL
                    result = manager.run_fullgraph(desc)
                torch.testing.assert_close(
                    result, torch.full((tokens, 4), tokens + 3.0, device=device)
                )
                assert manager.hidden_states.data_ptr() == address

        # Keep the old tensors alive: allocator address reuse must not obscure
        # whether reload allocated genuinely new model/output storage.
        old_storage = manager.hidden_states
        old_graphs = dict(manager.graphs)

        def load_new_model(self: Any) -> None:
            assert self.cudagraph_manager is not manager
            assert not self.cudagraph_manager.graphs
            self.model = Model(9.0)

        make_load_model_with_constraint_backend(load_new_model, close_runtime=close_v2_runtime)(
            runner
        )
        assert runner.model is not model
        assert runner.model.offset.data_ptr() != model.offset.data_ptr()
        install_prefill_graph_manager_v2(runner)
        reloaded = runner.cudagraph_manager
        assert reloaded is not manager
        assert not reloaded._capture_finished and not reloaded._graphs_captured
        assert not reloaded._successful and not reloaded.full_capture_failures
        assert not any(reloaded.dispatch_counts.values())
        with reloaded.prefill_scope(SimpleNamespace(num_scheduled_tokens={"r": 128})):
            assert reloaded.dispatch(1, 128, None).cg_mode == Mode.NONE

        reloaded.capture(runner.model, state, inputs, None, None, [], None)
        assert reloaded.hidden_states is not old_storage
        assert reloaded.hidden_states.data_ptr() != old_storage.data_ptr()
        assert not reloaded.full_capture_failures
        assert all(reloaded.graphs[desc] is not graph for desc, graph in old_graphs.items())
        assert runner.model.partition is not partition
        assert len(runner.model.partition.concrete_cudagraph_entries) == 4
        with reloaded.prefill_scope(SimpleNamespace(num_scheduled_tokens={"r": 128})):
            for tokens in (128, 16):
                desc = reloaded.dispatch(1, tokens, None)
                # The old model's optional FULL failure must not survive reload.
                assert desc.cg_mode == Mode.FULL
                ids.fill_(tokens)
                result = reloaded.run_fullgraph(desc)
                torch.testing.assert_close(result, runner.model.compute(ids[:tokens]))
                torch.testing.assert_close(
                    result, torch.full((tokens, 4), tokens + 9.0, device=device)
                )

        # PW belongs to the new model too, including when FULL used to fail.
        desc = reloaded.dispatch(1, 128, None)
        assert desc.cg_mode == Mode.PIECEWISE
        ids.fill_(2)
        with set_forward_context(
            None,
            config,
            num_tokens=128,
            cudagraph_runtime_mode=desc.cg_mode,
            batch_descriptor=BatchDescriptor(num_tokens=128),
            slot_mapping={},
        ):
            result = runner.model(ids[:128], inputs.positions[:128])
        torch.testing.assert_close(result, torch.full((128, 4), 11.0, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")  # type: ignore[misc]
@torch.inference_mode()  # type: ignore[misc]
def test_capacity_boundary_capture_replays_real_prefill_attention() -> None:
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    from tests.test_beam_attention_metadata import _make_builder
    from vllm_gr.v1.attention.backends import beam_attn_gpu

    if not beam_attn_gpu.is_available():
        pytest.skip("requires FlashAttention")
    fa_version = beam_attn_gpu.get_flash_attn_version(head_size=64)
    assert fa_version is not None
    gc.collect()
    device = torch.device("cuda:0")
    plan = PrefillCapturePlan((8189, 8190, 8191, 8192), True, False, (1, 2, 4), 2048)
    full = plan.descriptors()[Mode.FULL]
    assert {(desc.num_tokens, desc.num_reqs) for desc in full} == {
        (size, 4) for size in plan.buckets
    }
    inputs = InputBuffers(4, 8192, device)
    blocks = BlockTables([16], 4, 8192, [128], device, [16])
    # Use the minimal builder harness, with the real capture metadata logic
    # and GPU attention implementation; neither is mocked here.
    builder = _make_builder(block_size=16)
    builder.device = device
    builder._v2_gr_prefill_capture = True
    builder.model_config = SimpleNamespace(max_model_len=2048)
    query = torch.zeros((8192, 8, 64), dtype=torch.float16, device=device)
    key = torch.zeros((1, 16, 2, 64), dtype=torch.float16, device=device)
    value = torch.ones_like(key)

    for desc in full:
        dummy = InputBatch.make_dummy(desc.num_reqs, desc.num_tokens, inputs)
        common = CommonAttentionMetadata(
            query_start_loc=dummy.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(dummy.query_start_loc_np),
            seq_lens=dummy.seq_lens,
            seq_lens_cpu_upper_bound=dummy.seq_lens_cpu_upper_bound,
            num_reqs=desc.num_reqs,
            num_actual_tokens=desc.num_tokens,
            max_query_len=int(dummy.num_scheduled_tokens.max()),
            max_seq_len=int(dummy.num_scheduled_tokens.max()),
            block_table_tensor=blocks.get_dummy_block_tables(desc.num_reqs)[0],
            slot_mapping=blocks.get_dummy_slot_mappings(desc.num_tokens)[0],
        )
        metadata = builder.build_for_cudagraph_capture(common)
        assert int(dummy.num_scheduled_tokens.max()) <= metadata.max_query_len <= 2048
        assert int(dummy.seq_lens_cpu_upper_bound.max()) <= metadata.max_seq_len <= 2048
        output = torch.empty_like(query[: desc.num_tokens])

        forward = partial(
            beam_attn_gpu.prefill_attention,
            output,
            query,
            key,
            value,
            metadata,
            (-1, -1),
            64**-0.5,
            0.0,
            fa_version,
        )

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            forward()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.accelerator.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        for constant in (1.0, 2.0):
            value.fill_(constant)
            graph.replay()
            # Q/K=0 and constant V give an exact reference for every row,
            # including the tail rows omitted by the old oversized dummy.
            torch.testing.assert_close(output, torch.full_like(output, constant))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")  # type: ignore[misc]
def test_real_cuda_invalidation_is_fatal_in_isolated_process() -> None:
    import os
    import subprocess
    import sys

    # A CUDA-invalidated capture can poison the allocator and RNG state on
    # torch 2.11. The worker must exit, so do not contaminate other GPU tests.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pytest; "
                "from tests.kernels.test_prefill_graph_manager_v2 import "
                "test_native_capture_replay_fixed_output_and_piecewise_recovery as run; "
                "run(pytest.MonkeyPatch(), 'invalidated')"
            ),
        ],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
