# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import multiprocessing
import os
import time
from typing import Any, Dict

import psutil
import pytest
import torch

from vllm_gr.entrypoints.gr import GRLLM

try:
    from vllm_gr.sampling_params import BeamSearchParams
except ImportError:
    from vllm.sampling_params import BeamSearchParams

if not torch.cuda.is_available() and not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("GPU/NPU is not available", allow_module_level=True)


def _load_prompt_text() -> str:
    prompt_file = os.path.join(os.path.dirname(__file__), "resources/single_one_rec_prompt.txt")
    with open(prompt_file, "r") as f:
        return f.read()


def _run_offline_beam_search(
    loops: int, use_patch: bool, npu_decode_execution: str = "default"
) -> None:
    model_name = os.getenv("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")
    beam_width = 128
    max_tokens = 5

    if use_patch:
        assert hasattr(BeamSearchParams, "begin_token")

    params_dict: Dict[str, Any] = {
        "beam_width": beam_width,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": False,
    }
    if hasattr(BeamSearchParams, "begin_token"):
        params_dict["begin_token"] = "<|sid_begin|>"
        params_dict["end_token"] = "<|sid_end|>"

    params = BeamSearchParams(**params_dict)

    prompt_text = _load_prompt_text()
    prompts = [{"prompt": prompt_text}]
    print(
        f"Running offline beam search (patch={use_patch}) loops={loops} "
        f"npu_decode_execution={npu_decode_execution}"
    )

    # vllm-gr is designed to use beam_width for max_logprobs (not 2 * beam_width)
    # to improve performance. Reflect that here when use_patch is True.
    max_logprobs = beam_width if use_patch else 2 * beam_width

    engine_kwargs: Dict[str, Any] = {}
    if npu_decode_execution != "default":
        # 'default' deliberately leaves the config unset so the test exercises
        # the platform default resolution (xllm on NPU with xllm-ops
        # installed, cascade otherwise).
        engine_kwargs["additional_config"] = {"vllm_gr_npu_decode_execution": npu_decode_execution}

    with GRLLM(
        model=model_name,
        trust_remote_code=True,
        max_logprobs=max_logprobs,
        beam_max_width=beam_width,
        attention_config={"backend": "CUSTOM"},
        max_num_seqs=64,
        **engine_kwargs,
    ) as llm:
        start = time.perf_counter()
        for _ in range(max(1, loops)):
            outputs = llm.beam_search(prompts, params)
            assert len(outputs) == len(prompts)
            assert len(outputs[0].sequences) == beam_width
            # Regression (PR #345 review): during decode the shared-prefix path
            # keeps only the generated tail in beam.tokens, but it must be
            # restored to the full sequence (prompt + generated tail) before
            # returning. If regressed, tokens length would be <= max_tokens;
            # the full sequence is prompt_len + generated tokens, far longer.
            for seq in outputs[0].sequences:
                assert len(seq.tokens) > max_tokens, (
                    "beam_search returned beams without the full prompt prefix"
                )
                assert not hasattr(seq, "_prompt_token_ids"), (
                    "internal _prompt_token_ids must not leak to callers"
                )
        end = time.perf_counter()
    print(f"Elapsed mean: {(end - start) / max(1, loops):.4f}s")


def _run_offline_gr_beam_search_target(loops: int, npu_decode_execution: str = "default") -> None:
    # The memory-delta check below needs two engine lifecycles in one process.
    # After the first lifecycle the parent has initialized the NPU runtime,
    # and vLLM's default fork start method would fail EngineCore startup with
    # "Cannot re-initialize NPU in forked subprocess" (upstream only force-
    # spawns when CUDA/XPU is initialized). Force spawn for the second run.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    device = torch.cuda
    if hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.npu
    before_gpu = device.memory_allocated()
    before_host = psutil.Process().memory_info().rss
    _run_offline_beam_search(loops, use_patch=True, npu_decode_execution=npu_decode_execution)
    _run_offline_beam_search(loops, use_patch=True, npu_decode_execution=npu_decode_execution)
    device.synchronize()
    after_gpu = device.memory_allocated()
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
@pytest.mark.parametrize(  # type: ignore
    "npu_decode_execution",
    # NPU default aliases one of these backends; its resolution has separate
    # unit coverage. Exercise each implementation's two lifecycles once.
    # Keep explicit xllm so an unavailable operator library still fails.
    ["cascade", "xllm"] if hasattr(torch, "npu") and torch.npu.is_available() else ["default"],
)
def test_offline_gr_beam_search(loops: int, npu_decode_execution: str) -> None:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_run_offline_gr_beam_search_target,
        args=(loops, npu_decode_execution),
    )
    p.start()
    p.join()
    assert p.exitcode == 0
