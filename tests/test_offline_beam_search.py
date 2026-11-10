# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import json
import math
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

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("beam_search_v1 requires CUDA", allow_module_level=True)


def _run_offline_beam_search_v1(
    loops: int,
    model: str,
    config: dict[str, Any],
    source: list[int],
    newline: list[int],
    *,
    evidence: Path | None = None,
    full_prefill: bool = False,
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

    probe: dict[str, Any] = (
        {}
        if evidence is None
        else dict(
            worker_extension_cls="tests.st.test_onerec_v1_scheduler_st.OneRecV1STWorkerExtension"
        )
    )
    if full_prefill:
        # Cover B1=1024 tokens and the non-aligned B4 total=4102, including
        # padding to 4224. V2 consumes this native grid; V1 uses its GR grid.
        probe["compilation_config"] = {"cudagraph_capture_sizes": [1, 4, 128, 512, 1024, 4224]}
    rows = []
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
        enforce_eager=os.getenv("VLLM_GR_TEST_EAGER", "0") == "1",
        shutdown_timeout=10,
        **probe,
    ) as llm:
        if evidence is not None:
            from tests.st.test_onerec_v1_scheduler_st import (
                _check_completion_prefill,
                _check_v2_graph_events,
                _check_v2_idle,
            )

            processes = list(llm.llm_engine.engine_core.resources.engine_manager.processes)
            runner = llm.collective_rpc("onerec_v1_st_runner")[0]
            expected_runner = (
                "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
                if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") == "1"
                else "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
            )
            assert runner == expected_runner, runner
            llm.collective_rpc("onerec_v1_st_observe")
        start = time.perf_counter()
        for _ in range(max(1, loops)):
            for batch_size in (1, 4):
                prompt_ids = [
                    (newline * (1024 + index) + source)[-(1023 + index) :]
                    for index in range(batch_size)
                ]
                if evidence is not None:
                    before = len(llm.collective_rpc("onerec_v1_st_events")[0])
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
                        assert math.isfinite(seq.cum_logprob) and seq.cum_logprob <= 0
                        assert all(
                            0 <= token < llm.llm_engine.model_config.get_vocab_size()
                            for token in seq.tokens[len(ids) :]
                        )
                assert not llm.llm_engine.has_unfinished_requests()
                if evidence is not None:
                    snapshot = llm.collective_rpc("onerec_v1_st_snapshot")[0]
                    assert snapshot["runner"] == expected_runner
                    _check_v2_idle(snapshot)
                    assert snapshot["retired"] and not snapshot["active_sessions"]
                    assert snapshot["registry_size"] == 0 and snapshot["context_released"]
                    assert snapshot["workspace_released"] and not snapshot["pending"]
                    assert not any(snapshot["control_busy"])
                    assert snapshot["constraint_backend"] is None
                    assert snapshot["unconstrained_topk"]
                    request_events = llm.collective_rpc("onerec_v1_st_events")[0][before:]
                    if full_prefill:
                        _check_completion_prefill(request_events, batch_size)
                        prefill = [e for e in request_events if e["method"] == "prefill_inputs"]
                        assert len(prefill) == 1
                        event = prefill[0]
                        assert event["scheduled"] == sum(len(ids) + 1 for ids in prompt_ids)
                        assert event["bucket"] == (1024 if batch_size == 1 else 4224)
                        assert event["input_shape"] == [event["bucket"]]
                    _check_v2_graph_events(request_events)
                    rows.append(
                        dict(
                            batch_size=batch_size,
                            prompts=prompt_ids,
                            snapshot=snapshot,
                            events=request_events,
                            outputs=[
                                [
                                    dict(tokens=s.tokens[len(ids) :], score=s.cum_logprob)
                                    for s in output.sequences
                                ]
                                for ids, output in zip(prompt_ids, outputs, strict=True)
                            ],
                        )
                    )
        end = time.perf_counter()
        if evidence is not None:
            events = llm.collective_rpc("onerec_v1_st_events")[0]
            assert any(e["method"] == "replay" for e in events) is config["beam"]["graph_enabled"]
            decode = "replay" if config["beam"]["graph_enabled"] else "_forward"
            assert sum(e["method"] == decode for e in events) == 4 * max(1, loops)
            assert not any(
                e["method"] == ("_forward" if decode == "replay" else "replay") for e in events
            )
            graphs = llm.collective_rpc("onerec_v1_st_graphs")[0]
    if evidence is not None:
        shutdown = [dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive()) for p in processes]
        assert shutdown and all(not p["alive"] and p["exitcode"] == 0 for p in shutdown), shutdown
        evidence.write_text(
            json.dumps(
                dict(
                    config=config,
                    runner=runner,
                    full_prefill=full_prefill,
                    graphs=graphs,
                    rows=rows,
                    events=events,
                    shutdown=shutdown,
                ),
                indent=2,
            )
        )
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


def _run_unconstrained_target(work_dir: Path, full_prefill: bool = False) -> None:
    from tests.st.test_onerec_v1_scheduler_st import _inputs

    model, _, config, source, newline, _ = _inputs(work_dir)
    config["constraint_table"] = {"enabled": False, "max_top_k": config["beam"]["max_width"]}
    config["constrained_generation"] = False
    _run_offline_beam_search_v1(
        1,
        model,
        config,
        source,
        newline,
        evidence=work_dir / "unconstrained.json",
        full_prefill=full_prefill,
    )


@pytest.mark.slow
@pytest.mark.timeout(660)
@pytest.mark.parametrize("runner", ["v1", "v2"])
@pytest.mark.parametrize(
    ("eager", "full_prefill"),
    [(True, False), (False, False), (False, True)],
    ids=["eager", "graph", "full-graph"],
)
def test_offline_unconstrained_beam_search_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: str, eager: bool, full_prefill: bool
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(runner == "v2")))
    monkeypatch.setenv("VLLM_GR_TEST_EAGER", str(int(eager)))
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    monkeypatch.setenv("VLLM_ENABLE_PREFILL_CUDAGRAPH", "1")
    for name in (
        "VLLM_GR_PREFILL_GRAPH_BUCKETS",
        "VLLM_GR_PREFILL_GRAPH_BUCKET_STEP",
        "VLLM_GR_PREFILL_GRAPH_MAX_BUCKET",
        "VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS",
        "VLLM_GR_PREFILL_GRAPH_KV_BOUND",
    ):
        monkeypatch.delenv(name, raising=False)
    if full_prefill and runner == "v1":
        monkeypatch.setenv("VLLM_GR_PREFILL_GRAPH_BUCKETS", "1024,4224")
    process = multiprocessing.get_context("spawn").Process(
        target=_run_unconstrained_target, args=(tmp_path, full_prefill)
    )
    process.start()
    try:
        process.join(timeout=600)
        assert process.exitcode == 0, "Unconstrained Beam V1 subprocess failed"
    finally:
        if process.is_alive():
            process.kill()
            process.join()
