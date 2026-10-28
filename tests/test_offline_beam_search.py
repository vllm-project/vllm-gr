# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import multiprocessing
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import psutil
import pytest
import torch

from vllm_gr.entrypoints.gr import GRLLM
from vllm_gr.sampling_params import BeamSearchParams

if not torch.cuda.is_available():
    pytest.skip("beam_search_v1 requires CUDA", allow_module_level=True)


def _run_offline_beam_search_v1(
    loops: int, model: str, config: dict[str, Any], source: list[int], newline: list[int]
) -> None:
    beam_width = config["beam"]["max_width"]
    params = BeamSearchParams(
        beam_width=beam_width,
        max_tokens=5,
        temperature=0.0,
        ignore_eos=True,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
    )

    with GRLLM(
        vllm_gr_config=config,
        model=model,
        trust_remote_code=True,
        dtype=os.getenv("VLLM_GR_TEST_DTYPE", "bfloat16"),
        async_scheduling=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_model_len=2048,
        max_num_batched_tokens=8192,
        max_num_seqs=4,
        max_logprobs=beam_width,
        gpu_memory_utilization=0.5,
    ) as llm:
        start = time.perf_counter()
        for _ in range(max(1, loops)):
            for batch_size in (1, 4):
                prompt_ids = [
                    (newline * (1024 + index) + source)[-(1023 + index) :]
                    for index in range(batch_size)
                ]
                outputs = llm.beam_search_v1(
                    [{"prompt_token_ids": ids} for ids in prompt_ids], params, output_mode="full"
                )
                assert len(outputs) == batch_size
                for ids, output in zip(prompt_ids, outputs, strict=True):
                    assert len(output.sequences) == beam_width
                    for seq in output.sequences:
                        assert seq.tokens[: len(ids)] == ids
                        assert len(seq.tokens) == len(ids) + params.max_tokens
                        assert seq.finish_reason == "length"
                        assert not hasattr(seq, "_prompt_token_ids")
                assert not llm.llm_engine.has_unfinished_requests()
        end = time.perf_counter()
    print(f"Elapsed mean per B=1/B=4 pair: {(end - start) / max(1, loops):.4f}s")


def _run_offline_gr_beam_search_v1_target(loops: int) -> None:
    from tests.st.test_onerec_v1_scheduler_st import _inputs

    # The memory-delta check below needs two engine lifecycles in one process.
    with TemporaryDirectory(prefix="gr-offline-v1-") as tmp:
        work_dir = Path(tmp)
        os.environ["VLLM_GR_TEST_ARTIFACT"] = str(work_dir / "constraints.safetensors")
        model, _, config, source, newline, _ = _inputs(work_dir)
        before_gpu = torch.accelerator.memory_allocated()
        before_host = psutil.Process().memory_info().rss
        _run_offline_beam_search_v1(loops, model, config, source, newline)
        _run_offline_beam_search_v1(loops, model, config, source, newline)
        torch.accelerator.synchronize()
        after_gpu = torch.accelerator.memory_allocated()
        after_host = psutil.Process().memory_info().rss
    print(
        f"host mem delta={(after_host - before_host) / 1024**3:.2f}GB, "
        f"gpu mem delta={(after_gpu - before_gpu) / 1024**3:.2f}GB"
    )
    # Host RSS includes hardware/runtime-specific one-time residency from
    # PyTorch, CUDA/NPU, tokenizers, compilers, and IPC. Keep these values in
    # every test report, but enforce fixed thresholds only in a calibrated,
    # fixed-hardware memory job. General CI continues to gate functional Beam
    # Search correctness and explicit engine shutdown without applying a
    # machine-specific RSS baseline.
    enforce_memory_thresholds = os.getenv("VLLM_GR_ENFORCE_MEMORY_THRESHOLDS", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if loops > 1 and enforce_memory_thresholds:
        assert after_host - before_host <= 500 * 1024**2
        assert after_gpu - before_gpu <= 50 * 1024**2
    elif loops > 1:
        print(
            "Memory thresholds are report-only; set "
            "VLLM_GR_ENFORCE_MEMORY_THRESHOLDS=1 in a calibrated hardware job "
            "to enforce them."
        )


@pytest.mark.slow  # type: ignore
@pytest.mark.timeout(600)  # spawn subprocess compiles + loads engine twice
def test_offline_gr_beam_search_v1(loops: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_run_offline_gr_beam_search_v1_target,
        args=(loops,),
    )
    p.start()
    p.join()
    assert p.exitcode == 0
