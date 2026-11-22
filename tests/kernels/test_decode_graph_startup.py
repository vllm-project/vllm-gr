# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Real Decode capture recovery; device-invalidating failures run in isolation."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.beam_stage_test_utils import (
    admission_worker,
    install_decode_model,
)
from vllm_gr.config.models import DeploymentOptions
from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1
from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import (
    should_replay_decode_graph_v2,
)
from vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_resources import (
    capture_multibatch_forward,
    get_v2_runtime,
)
from vllm_gr.v1.worker.gr_startup import capture_async_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@torch.inference_mode()  # type: ignore[misc]
@pytest.mark.parametrize("failure", [None, "unsupported"])  # type: ignore[misc]
def test_decode_capture_and_clean_fallback(failure: str | None) -> None:
    worker = admission_worker()
    worker.model_config.enforce_eager = False
    worker.model_config.max_model_len = 64
    worker._vllm_gr_config.beam_graph_enabled = True
    worker._vllm_gr_config.deployment = DeploymentOptions(beam_batch_sizes=(1,))
    worker._vllm_gr_config.beam_worker_decision = True

    worker.vllm_config.scheduler_config = SimpleNamespace(async_scheduling=True, max_num_seqs=1)
    for group in worker.kv_cache_config.kv_cache_groups:
        group.kv_cache_spec = SimpleNamespace(block_size=16)
    invalidations = []

    def forward(ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        result = (ids[:, None] + positions[:, None]).to(torch.float16).expand(-1, 6).clone()
        if torch.cuda.is_current_stream_capturing():
            if failure == "invalidated":
                invalidations.append(True)
                ids[0].item()  # Real CUDA capture invalidation, child process only.
            if failure == "unsupported":
                # Python refusal inside a real capture; CUDA itself stays healthy.
                raise RuntimeError("operation not supported when stream is capturing")
        return result

    install_decode_model(worker, forward)
    original = worker.model
    if failure == "invalidated":
        with pytest.raises(RuntimeError) as raised:
            capture_multibatch_forward(worker, allow_eager_fallback=True)
        assert invalidations == [True]
        errors = []
        error: BaseException | None = raised.value
        while error is not None:
            errors.append(str(error))
            error = error.__cause__ or error.__context__
        assert any("previous error during capture" in message for message in errors), errors
        assert not get_v2_runtime(worker).capture_failed
        assert worker._gr_gpu_stage_runner.graph is None
        return

    captured = capture_multibatch_forward(worker, allow_eager_fallback=True)
    owner = worker._gr_gpu_stage_runner
    assert owner.closed and not worker._beam_sessions
    assert worker.model is original
    assert not worker._beam_decode_graph.installed
    assert captured is (failure is None)
    assert should_replay_decode_graph_v2(worker, owner.graph, batch_size=1) is captured
    if failure:
        assert get_v2_runtime(worker).capture_failed
        assert "not supported" in get_v2_runtime(worker).capture_error
        assert not worker._beam_decode_graph.ready
        assert not worker._beam_graph_keys
    for token, position in [(3, 7), (2, 19)]:
        owner.input_ids.fill_(token)
        owner.positions.fill_(position)
        output = owner.graph.replay() if captured else owner._forward()
        torch.testing.assert_close(
            output, torch.full((4, 6), token + position, device="cuda", dtype=torch.float16)
        )

    # Recovery must leave the CUDA allocator/capture machinery usable too.
    inputs = torch.ones(4, device="cuda")
    probe = torch.cuda.CUDAGraph()
    with torch.cuda.graph(probe):
        outputs = inputs + 2
    inputs.fill_(5)
    probe.replay()
    torch.testing.assert_close(outputs, torch.full_like(inputs, 7))


def test_invalidated_decode_capture_aborts_in_isolated_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tests.kernels.test_decode_graph_startup import "
                "test_decode_capture_and_clean_fallback as run; run('invalidated')"
            ),
        ],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@torch.inference_mode()  # type: ignore[misc]
def test_v1_startup_graphs_replay_with_their_own_batch_buffers() -> None:
    worker = admission_worker()
    worker.model_config.enforce_eager = False
    cfg = worker._vllm_gr_config
    cfg.beam_graph_enabled = cfg.beam_worker_decision = True

    cfg.deployment = DeploymentOptions(beam_batch_sizes=(1, 2))
    worker.vllm_config.scheduler_config = SimpleNamespace(async_scheduling=True, max_num_seqs=2)
    install_decode_model(worker, lambda ids, pos: (ids + pos)[:, None].to(torch.float16))
    assert capture_async_forward(worker, hooks_type=CudaStageHooksV1)
    assert capture_async_forward(worker, hooks_type=CudaStageHooksV1)
    for batch, token in ((1, 5), (2, 9), (1, 17)):
        owner = worker._gr_gpu_stage_runner_cache[batch]
        owner.input_ids.fill_(token)
        owner.positions.fill_(1)
        graph = worker._beam_decode_graph.select(batch, cfg.beam_max_width)
        assert graph.ready
        torch.testing.assert_close(
            graph.replay(),
            torch.full(
                (batch * cfg.beam_max_width, 1), token + 1, device="cuda", dtype=torch.float16
            ),
        )
