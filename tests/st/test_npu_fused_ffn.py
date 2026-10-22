# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return,attr-defined"
"""Real Qwen MLP FFN regression, without downloading a model.

The NPU test gate builds/sources the local extension and sets
VLLM_GR_TEST_REQUIRE_FFN=1. Missing dependencies then fail instead of skipping.
All NPU work runs in a child process to isolate distributed and graph state.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


def _require_runtime() -> None:
    required = os.environ.get("VLLM_GR_TEST_REQUIRE_FFN") == "1"
    try:
        import torch_npu  # noqa: F401

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is unavailable")
        device_name = torch.npu.get_device_name(0)
        if device_name != "Ascend910B4":
            pytest.skip(f"fused FFN is validated only on Ascend910B4, got {device_name}")
        from vllm_gr.ops.npu.ffn import _load_backend

        assert callable(_load_backend())
    except (ImportError, RuntimeError, AssertionError) as exc:
        reason = f"NPU fused FFN dependencies are unavailable: {exc}"
        if required:
            pytest.fail(reason)
        pytest.skip(reason + "; build and source csrc/npu/fused_ffn/set_env.sh")


def _run_device_cases(work_dir: Path) -> None:
    import torch_npu  # noqa: F401
    from transformers import Qwen2Config
    from vllm.config import (
        CUDAGraphMode,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.qwen2 import Qwen2MLP
    from vllm_ascend.utils import WeightPrefetchConfig, set_weight_prefetch_method

    from vllm_gr.ops.npu import ffn

    torch.npu.set_device(0)
    torch.manual_seed(42)
    # The native Ascend prefetch setup reads hf_text_config even when disabled.
    # Supply a real local HF configuration, without weights or tokenizer files.
    model_dir = work_dir / "model_config"
    Qwen2Config(
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=8,
        vocab_size=128,
        max_position_embeddings=512,
        architectures=["Qwen2ForCausalLM"],
    ).save_pretrained(model_dir)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(model_dir),
            tokenizer=str(model_dir),
            dtype="bfloat16",
            enforce_eager=False,
            skip_tokenizer_init=True,
            max_model_len=512,
        )
    )
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=(work_dir / "hccl_init").as_uri(),
            backend="hccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        set_weight_prefetch_method(WeightPrefetchConfig({"enabled": False}))
    previous_dtype = torch.get_default_dtype()
    runner: Any = None
    try:
        torch.set_default_dtype(torch.bfloat16)
        with set_current_vllm_config(config), torch.device("npu:0"):
            layer = Qwen2MLP(
                hidden_size=1024, intermediate_size=3072, hidden_act="silu", prefix="ffn_regression"
            )
        torch.set_default_dtype(previous_dtype)
        with torch.inference_mode():
            # Seeded, bounded synthetic weights keep the test independent of
            # model files while using the real vLLM MLP/projection classes.
            for parameter in layer.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, dtype=torch.float32).to(parameter) * 0.02
                )
        gate_weight = layer.gate_up_proj.weight
        gate_pointer = gate_weight.data_ptr()
        native = layer.forward
        model = torch.nn.Module()
        model.add_module("mlp", layer)
        runner = SimpleNamespace(model=model, vllm_config=config)
        os.environ[ffn.ENV_ENABLE] = "0"
        ffn.install_npu_fused_ffn(runner)
        assert "forward" not in layer.__dict__
        assert runner._gr_npu_fused_ffn_bindings == ()
        os.environ[ffn.ENV_ENABLE] = "1"
        ffn.install_npu_fused_ffn(runner)
        bindings = runner._gr_npu_fused_ffn_bindings
        assert len(bindings) == 1, "enable flag did not bind the real eligible Qwen MLP"
        assert runner._gr_npu_fused_ffn_backend == "ascendc"
        binding = bindings[0]
        assert binding.layer is layer
        assert layer.gate_up_proj.weight is gate_weight
        assert layer.gate_up_proj.weight.data_ptr() == gate_pointer

        # Observe calls while delegating to the real AscendC extension.
        # Counters alone cannot prove graph replay; changed-input replay and
        # comparison against the original full MLP below establish that too.
        kernel = ffn._KERNEL
        assert callable(kernel)
        kernel_calls: list[tuple[int, bool]] = []

        def observed_kernel(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            kernel_calls.append((int(x.shape[0]), bool(torch.npu.is_current_stream_capturing())))
            return kernel(x, weight)

        ffn._KERNEL = observed_kernel
        report = []

        def assert_fused_close(output: torch.Tensor, expected: torch.Tensor) -> float:
            torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
            relative_l2 = torch.linalg.vector_norm((output - expected).float()) / (
                torch.linalg.vector_norm(expected.float()) + 1e-12
            )
            value = float(relative_l2.cpu())
            assert value < 0.01
            assert torch.isfinite(output).all().item()
            return value

        try:
            # M128 deliberately fuses outside capture but stays fully native
            # inside capture. Validate both production routes independently.
            for variant in (0.25, -0.5):
                x = torch.randn((128, 1024), dtype=torch.float32).to("npu", dtype=torch.bfloat16)
                x.add_(variant)
                kernel_calls.clear()
                with torch.inference_mode(), set_forward_context({}, config, num_tokens=128):
                    output = layer(x)
                    expected = native(x)
                torch.npu.synchronize()
                assert kernel_calls == [(128, False)], kernel_calls
                relative_l2 = assert_fused_close(output, expected)
                report.append(
                    {
                        "rows": 128,
                        "route": "uncaptured",
                        "kernel_calls": list(kernel_calls),
                        "relative_l2": relative_l2,
                    }
                )
            for rows in (128, 512, 1024, 4096, 16384):
                x = torch.randn((rows, 1024), dtype=torch.float32).to("npu", dtype=torch.bfloat16)
                with torch.inference_mode():
                    for _ in range(2):
                        with set_forward_context({}, config, num_tokens=rows):
                            layer(x)
                    torch.npu.synchronize()
                    kernel_calls.clear()
                    graph = torch.npu.NPUGraph()
                    with (
                        set_forward_context({}, config, num_tokens=rows),
                        torch.npu.graph(graph, capture_error_mode="global"),
                    ):
                        output = layer(x)
                    torch.npu.synchronize()
                    capture_calls = list(kernel_calls)
                    fused_capture = rows in (512, 4096, 16384)
                    expected_calls = [(rows, True)] if fused_capture else []
                    assert capture_calls == expected_calls, (rows, capture_calls)
                    replay_relative_l2 = []
                    for variant in (0.25, -0.5):
                        x.copy_(torch.randn(x.shape, dtype=torch.float32).to(x) + variant)
                        with set_forward_context({}, config, num_tokens=rows):
                            expected = native(x)
                        graph.replay()
                        torch.npu.synchronize()
                        assert kernel_calls == capture_calls, "replay reran Python dispatch"
                        if fused_capture:
                            replay_relative_l2.append(assert_fused_close(output, expected))
                        else:
                            # M128 capture and the excluded M1024 must retain
                            # the entire native MLP, including BF16 rounding.
                            torch.testing.assert_close(output, expected, atol=0, rtol=0)
                        assert torch.isfinite(output).all().item()
                    report.append(
                        {
                            "rows": rows,
                            "route": "graph_replay",
                            "captured_kernel_calls": capture_calls,
                            "changed_input_replays": 2,
                            "relative_l2": replay_relative_l2,
                        }
                    )
        finally:
            ffn._KERNEL = kernel
        ffn._restore_bindings(runner)
        assert "forward" not in layer.__dict__
        assert ffn._COMPILE_MARKER_KEY not in config.additional_config
        assert gate_weight.data_ptr() == gate_pointer
        print("NPU fused FFN regression: " + json.dumps(report), flush=True)
    finally:
        torch.set_default_dtype(previous_dtype)
        if runner is not None:
            ffn._restore_bindings(runner)
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.npu_integration
@pytest.mark.slow
@pytest.mark.timeout(360)
def test_npu_fused_ffn_opt_in_graph_precision_and_fallback(tmp_path: Path) -> None:
    _require_runtime()
    env = dict(os.environ)
    env.update(
        USE_TF="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_PREFILL_CUDAGRAPH="1",
        VLLM_GR_NPU_FUSED_FFN="1",
    )
    command = [sys.executable, "-m", "tests.st.test_npu_fused_ffn", str(tmp_path.resolve())]
    with subprocess.Popen(
        command, cwd=Path(__file__).parents[2], env=env, start_new_session=True
    ) as process:
        try:
            assert process.wait(timeout=300) == 0, "real NPU fused FFN regression failed"
        finally:
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()


if __name__ == "__main__":
    _run_device_cases(Path(sys.argv[1]))
