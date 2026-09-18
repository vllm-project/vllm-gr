# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CUDA capture/replay using the native V2 model factory and PW wrapper.

The toy model omits attention; OneRec ST covers actual attention and KV state.
A controlled FULL-only exception exercises recovery inside a real capture.
A separate process checks that a real CUDA invalidation is never swallowed.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CompilationConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode as Mode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.cudagraph_dispatcher import BatchDescriptor
from vllm.v1.worker.gpu import cudagraph_utils as native

from vllm_gr.v1.worker.gpu_prefill_graph_runner_v2 import (
    GRPrefillCudaGraphManager,
    PrefillCapturePlan,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")  # type: ignore[misc]
@pytest.mark.parametrize("fail_full", [False, True])  # type: ignore[misc]
def test_native_capture_replay_fixed_output_and_piecewise_recovery(
    monkeypatch: pytest.MonkeyPatch, fail_full: bool | str
) -> None:
    from vllm.compilation import monitor

    # The production runner opens this window before calling the manager.
    # Other graph tests may have closed it earlier in the same process.
    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", True)
    config = VllmConfig(
        compilation_config=CompilationConfig(cudagraph_mode=Mode.FULL_AND_PIECEWISE)
    )
    config.scheduler_config.max_num_seqs = 1
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
        lambda *args, skip_attn: native.CapturedAttentionState(None if skip_attn else {}, {}),
    )
    original = native.ModelCudaGraphManager(config, device, Mode.FULL_AND_PIECEWISE, 1)
    runner = SimpleNamespace(
        cudagraph_manager=original,
        req_states=SimpleNamespace(
            req_id_to_index={"r": 0},
            num_computed_tokens_np=[512],
            prompt_len=SimpleNamespace(np=[1024]),
        ),
    )
    manager = GRPrefillCudaGraphManager(runner, PrefillCapturePlan((1, 16, 128, 256), True, True))
    ids = torch.zeros(256, device=device)
    inputs = SimpleNamespace(input_ids=ids, positions=torch.zeros_like(ids))
    partition = CUDAGraphWrapper(
        lambda x: (x[:, None] + 3).expand(-1, 4).contiguous(), config, Mode.PIECEWISE
    )

    def model(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if (
            fail_full
            and input_ids.numel() == 128
            and get_forward_context().cudagraph_runtime_mode == Mode.NONE
            and torch.cuda.is_current_stream_capturing()
        ):
            if fail_full == "invalidated":
                input_ids[0].item()  # Real CUDA error 900 invalidates capture.
            partition(input_ids)
            raise RuntimeError("operation not supported when stream is capturing")
        return cast(torch.Tensor, partition(input_ids))

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
        assert set(manager.full_capture_failures) == ({128} if fail_full else set())
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
