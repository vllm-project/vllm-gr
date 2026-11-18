# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
try:
    from vllm_gr.sampling_params import BeamSearchParams
except ImportError:
    from vllm.sampling_params import BeamSearchParams

import multiprocessing
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import psutil
import pytest
import torch
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.launchers.api_server.entry import (
    build_async_engine_client_from_engine_args,
)
from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)


async def beam_search_test(loops: int) -> None:
    model_name = os.getenv("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")
    raw_gpu_memory_utilization = os.getenv("VLLM_GR_TEST_GPU_MEMORY_UTILIZATION", "0.7")
    try:
        gpu_memory_utilization = float(raw_gpu_memory_utilization)
    except ValueError as exc:
        raise ValueError(
            "VLLM_GR_TEST_GPU_MEMORY_UTILIZATION must be a floating-point value in (0, 1]"
        ) from exc
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("VLLM_GR_TEST_GPU_MEMORY_UTILIZATION must be in (0, 1]")
    from tests.e2e.function.test_onerec_v1_scheduler_st import _inputs

    with TemporaryDirectory(prefix="gr-online-v1-") as directory:
        model_name, _, config, _, _, _ = _inputs(Path(directory))
        await _run_serving_v1(loops, model_name, config, gpu_memory_utilization)


async def _run_serving_v1(loops, model_name, config, gpu_memory_utilization):
    from vllm_gr.plugin import initialize_runtime

    initialize_runtime()
    beam_width = config["beam"]["max_width"]
    max_tokens = 5

    engine_args = AsyncEngineArgs(
        model=model_name,
        trust_remote_code=True,
        async_scheduling=True,
        enable_chunked_prefill=False,
        max_model_len=2048,
        max_num_batched_tokens=8192,
        # V1 owns one logical request; beam rows are managed by the worker.
        max_num_seqs=1,
        enforce_eager=not config["beam"]["graph_enabled"],
        enable_log_requests=False,
        shutdown_timeout=10,
        max_logprobs=2 * beam_width,
        gpu_memory_utilization=gpu_memory_utilization,
        additional_config={"vllm_gr": config},
    )

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
        base_model_paths = [BaseModelPath(name=model_name, model_path=model_name)]
        models = OpenAIServingModels(
            engine_client=engine_client, base_model_paths=base_model_paths, lora_modules=None
        )

        serving = GenerateBaseServing(
            engine_client=engine_client,
            models=models,
            request_logger=None,
        )

        # Use the specific prompt
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        prompt_file = os.path.join(os.path.dirname(__file__), "resources/single_one_rec_prompt.txt")
        with open(prompt_file, "r") as f:
            prompt_text = f.read()
        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[-1023:]
        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

        beam_search_params: dict[str, Any] = {
            "beam_width": beam_width,
            "max_tokens": max_tokens,
            "ignore_eos": False,
            "temperature": 0.0,
        }
        if hasattr(BeamSearchParams, "begin_token"):
            beam_search_params["begin_token"] = "<|sid_begin|>"
            beam_search_params["end_token"] = "<|sid_end|>"

        params = BeamSearchParams(**beam_search_params)

        if loops <= 0:
            loops = 1
        print(f"Starting beam search with {model_name}, beam_width={beam_width} iterations={loops}")
        final_output: RequestOutput | None = None
        start: float = time.perf_counter()
        for i in range(loops):
            generator = serving.beam_search(prompt, "req_id", params)
            async for output in generator:
                final_output = output

        end = time.perf_counter()
        print(f"V1 online elapsed mean:{(end - start) / loops}")
        assert final_output is not None
        assert final_output.finished
        assert final_output.request_id == "req_id"
        assert len(final_output.outputs) == beam_width


@pytest.mark.slow  # type: ignore
@pytest.mark.timeout(1200)  # three model/engine builds can exceed 600 s on L20
@pytest.mark.asyncio  # type: ignore
async def test_serving_beam_search(loops: int) -> None:
    await beam_search_test(loops)


def _run_gr_beam_search_target(loops: int) -> None:
    import asyncio

    from vllm_gr.patch import re_apply_patches

    re_apply_patches()
    assert hasattr(BeamSearchParams, "begin_token")

    async def run_test() -> None:
        device = torch.cuda
        if hasattr(torch, "npu") and torch.npu.is_available():
            device = torch.npu
        await beam_search_test(loops)
        device.synchronize()
        before_gpu_mem: int = device.memory_allocated()
        before_host_mem: int = psutil.Process().memory_info().rss
        await beam_search_test(loops)
        await beam_search_test(loops)
        device.synchronize()
        after_gpu_mem: int = device.memory_allocated()
        after_host_mem: int = psutil.Process().memory_info().rss
        print(
            f"before host mem={before_host_mem / 1024**3:.2f}GB, after host mem={after_host_mem / 1024**3:.2f}GB"
        )
        print(
            f"before gpu mem={before_gpu_mem / 1024**3:.2f}GB, after gpu mem={after_gpu_mem / 1024**3:.2f}GB"
        )
        # These thresholds (500MB host RAM, 50MB GPU RAM) match the minimum safe
        # memory headroom required for the 1.7B model to load and run without OOM.
        # They are intentionally conservative to avoid instability on smaller machines.
        if loops > 1:
            # Allow 500MB tolerance
            assert after_host_mem - before_host_mem <= 500 * 1024**2
            # Allow 50MB tolerance
            assert after_gpu_mem - before_gpu_mem <= 50 * 1024**2

    asyncio.run(run_test())


@pytest.mark.slow  # type: ignore
@pytest.mark.timeout(1200)  # three model/engine builds can exceed 600 s on L20
def test_serving_gr_beam_search(loops: int) -> None:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_gr_beam_search_target, args=(loops,))
    p.start()
    p.join()
    assert p.exitcode == 0


pytestmark = pytest.mark.gpu_test
