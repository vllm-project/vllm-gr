# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""Real OneRec weights through the V1 multiprocess Scheduler/Worker pipeline.

Exercise the public offline/online APIs and inspect release at return. Optional
environment settings select representative width/dtype/eager configurations;
test-only observers record actual dispatch/replay without changing execution.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from itertools import product
from pathlib import Path
from typing import Any

import pytest

_MODULE_READY = time.monotonic()
_MEASUREMENTS: dict[str, Any] = {"phases": {}, "rows": [], "events": []}
_EVIDENCE_PATH: Path | None = None


def _persist_measurements() -> None:
    if _EVIDENCE_PATH is not None:
        temporary = _EVIDENCE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(_MEASUREMENTS, indent=2))
        temporary.replace(_EVIDENCE_PATH)


@contextmanager
def _measure(name: str) -> Any:
    started = time.monotonic()
    _MEASUREMENTS["active_phase"] = name
    _persist_measurements()
    try:
        yield
    finally:
        _MEASUREMENTS["phases"][name] = time.monotonic() - started
        _persist_measurements()


@asynccontextmanager
async def _measured_engine(builder: Any, args: Any) -> Any:
    manager = builder(args)
    with _measure("engine_startup_seconds"):
        engine = await manager.__aenter__()
    try:
        yield engine
    finally:
        with _measure("engine_shutdown_seconds"):
            await manager.__aexit__(*sys.exc_info())


_BEAM_WIDTH = int(os.environ.get("VLLM_GR_TEST_WIDTH", "128"))
_EAGER = os.environ.get("VLLM_GR_TEST_EAGER", "0") == "1"


class OneRecV1STWorkerExtension:
    """Expose a read-only test probe through the native named utility RPC."""

    def onerec_v1_st_snapshot(self) -> dict[str, Any]:
        return _worker_snapshot(self)

    def onerec_v1_st_observe(self) -> None:
        from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeGraph
        from vllm_gr.v1.worker.gpu_beam_stage_runner import (
            AsyncGPUBeamOutput,
            GPUBeamStageRunner,
        )
        from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

        events: list[dict[str, Any]] = []
        self._onerec_events = events

        # Observe calls only; each original method executes exactly once with
        # unchanged arguments/results. These probes are excluded from timing runs.
        def wrap(cls: type, name: str, describe: Any) -> None:
            original = getattr(cls, name)

            def observed(instance: Any, *args: Any, **kwargs: Any) -> Any:
                event = {"method": name, "start_ns": time.perf_counter_ns()}
                event.update(describe(instance, args, kwargs))
                events.append(event)
                result = original(instance, *args, **kwargs)
                event["end_ns"] = time.perf_counter_ns()
                if name == "execute_model_prefill":
                    event["outcome"] = instance.last_replay_reason
                return result

            setattr(cls, name, observed)

        def stage(owner: Any, args: Any, kwargs: Any) -> dict[str, Any]:
            m = next(iter(args[0].gr_stage_metadata.values()))
            return {
                "session_id": m.session_id,
                "stage": m.stage_index,
                # How many requests share this stage. One is the shape every
                # other case here runs; a batch route is the one that forms more.
                "size": len(m.request_ids),
                "scheduled": args[0].num_scheduled_tokens[m.session_id],
            }

        wrap(GPUBeamStageRunner, "execute", stage)
        wrap(
            GPUBeamStageRunner,
            "prepare_cpu",
            lambda owner, args, kwargs: {
                "session_id": args[0].session_id,
                "stage": args[0].stage_index,
                "predecessor_gpu_pending": not owner.last_work.query(),
            },
        )
        wrap(
            AsyncGPUBeamOutput,
            "get_output",
            lambda output, args, kwargs: {
                "session_id": output.execution.entries[0].metadata.session_id,
                "stage": output.execution.entries[0].metadata.stage_index,
            },
        )
        wrap(
            BeamDecodeGraph,
            "replay",
            lambda graph, args, kwargs: {
                "batch_size": graph.signature.batch_size,
                "beam_width": graph.signature.beam_width,
                "num_model_rows": graph.signature.num_model_rows,
            },
        )
        wrap(
            GPUPrefillGraphRunner,
            "execute_model_prefill",
            lambda graph, args, kwargs: {
                "bucket": kwargs["bucket"],
                "token_counts": list(kwargs["num_scheduled_tokens"].values())
                if isinstance(kwargs["num_scheduled_tokens"], dict)
                else list(kwargs["num_scheduled_tokens"]),
            },
        )

    def onerec_v1_st_events(self) -> list[dict[str, Any]]:
        return self._onerec_events


def _worker_snapshot(worker: Any) -> dict[str, Any]:
    """Read only after native retirement has drained GPU work and consumers."""
    runner = worker.model_runner
    owner = getattr(runner, "_gr_gpu_stage_runner", None)
    if owner is None or not owner.closed:
        return {"retired": False}
    state = owner.beam.state
    row = owner.binding.state_row
    count = int(state.num_completed[row].item())
    names = ("token_ids", "scores", "parent_ids", "lengths", "finish_reasons")
    return {
        "retired": True,
        "session_id": owner.binding.session_id,
        "active_sessions": list(runner._beam_sessions),
        "registry_size": owner.beam.session.context.registry.num_active_sessions,
        "context_released": owner.beam.session.context._gr_execution_owner is None,
        "workspace_released": owner.beam.workspace.execution_owner is None,
        "control_busy": list(owner.control_busy),
        "pending": owner.pending is not None,
        "generated": int(state.num_generated_tokens[row].item()),
        "decode_steps": int(state.num_decode_steps[row].item()),
        "error_code": int(state.error_code[row].item()),
        "stage_index": owner.previous.stage_index,
        "graph_captured": owner.graph is not None,
        "state_address": state.sequence.data_ptr(),
        "completed": {
            name: getattr(state, "completed_" + name)[row, :count].cpu().tolist() for name in names
        },
        "history_tokens": state.history_token_ids[row].cpu().tolist(),
        "history_parents": state.history_parent_ids[row].cpu().tolist(),
    }


def _completed_sequences(
    snapshot: dict[str, Any], steps: int
) -> list[tuple[tuple[int, ...], float]]:
    from vllm_gr.ops.cuda.beam_state import FINISH_LENGTH

    completed = snapshot["completed"]
    sequences = []
    for token, score, parent, length, reason in zip(
        completed["token_ids"],
        completed["scores"],
        completed["parent_ids"],
        completed["lengths"],
        completed["finish_reasons"],
        strict=True,
    ):
        assert length == steps
        assert reason == FINISH_LENGTH
        assert math.isfinite(score) and score <= 0
        tokens = [token]
        for depth in range(length - 2, -1, -1):
            assert 0 <= parent < _BEAM_WIDTH
            tokens.append(snapshot["history_tokens"][depth][parent])
            parent = snapshot["history_parents"][depth][parent]
        sequences.append((tuple(reversed(tokens)), score))
    assert len(sequences) >= _BEAM_WIDTH
    assert len({tokens for tokens, _ in sequences}) == len(sequences)
    return sorted(sequences, key=lambda item: (-item[1], item[0]))[:_BEAM_WIDTH]


def _inputs(work_dir: Path) -> tuple[Any, ...]:
    from transformers import AutoConfig, AutoTokenizer

    from vllm_gr.constraints.artifact import save_constraint_table_artifact
    from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
    from vllm_gr.constraints.json_compiler import compile_relations_json
    from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

    model = os.environ.get("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    vocab_size = AutoConfig.from_pretrained(model, trust_remote_code=True).vocab_size
    # A deterministic, small SID universe keeps the ST independent of external
    # catalogs while leaving all 128 beams valid at each of the three stages.
    triples = [
        [f"<s_a_{a}>", f"<s_b_{b}>", f"<s_c_{c}>"]
        for a, b, c in product(range(max(128, _BEAM_WIDTH)), range(4), range(4))
    ]
    tokenizer_digest = compute_tokenizer_digest(tokenizer)
    table = ConstraintTableArtifactCompiler(
        tokenizer, vocab_size, tokenizer_digest=tokenizer_digest
    ).compile(compile_relations_json(triples))
    artifact = Path(
        os.environ.get("VLLM_GR_TEST_ARTIFACT", str(work_dir / "onerec_v1_constraints.safetensors"))
    )
    digest = save_constraint_table_artifact(table, artifact)
    legal = {tuple(tokenizer.convert_tokens_to_ids(triple)) for triple in triples}
    config = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "graph_enabled": not _EAGER,
            "worker_decision": True,
            "max_width": _BEAM_WIDTH,
            "max_decode_steps": 3,
        },
        "constraint_table": {
            "enabled": True,
            "backend": "cuda",
            "max_top_k": _BEAM_WIDTH,
            "path": str(artifact),
            "format": "constraint_table_v1",
            "artifact_digest": digest,
            "tokenizer_digest": tokenizer_digest,
        },
    }
    prompt_text = (Path(__file__).parents[1] / "resources/single_one_rec_prompt.txt").read_text()
    source = tokenizer.encode(prompt_text, add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    assert len(newline) == 1
    return model, tokenizer, config, source, newline, legal


def _engine_kwargs(model: str, config: dict[str, Any]) -> dict[str, Any]:
    return dict(
        distributed_executor_backend=os.environ.get("VLLM_GR_TEST_EXECUTOR"),
        model=model,
        trust_remote_code=True,
        dtype=os.environ.get("VLLM_GR_TEST_DTYPE", "bfloat16"),
        seed=0,
        async_scheduling=True,
        max_num_seqs=1,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_logprobs=_BEAM_WIDTH,
        enable_chunked_prefill=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
        enforce_eager=_EAGER,
        # These STs assert Prefill replay for full prompts and cached tails.
        # Native defaults depend on max_num_seqs and need not cover either.
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 16, *range(128, 2049, 128)],
        },
        worker_extension_cls="tests.st.test_onerec_v1_scheduler_st.OneRecV1STWorkerExtension",
    )


def _params(steps: int) -> Any:
    from vllm_gr.sampling_params import BeamSearchParams

    return BeamSearchParams(
        beam_width=_BEAM_WIDTH,
        max_tokens=steps + 2,
        temperature=0.0,
        ignore_eos=True,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
    )


# The first three requests exercise cache miss/hit and N=1/2/3. The
# remaining cold prompts straddle KV-block and graph-bucket boundaries.
_CASES = (
    (900, 1, True),
    (900, 2, False),
    (900, 3, False),
    (1023, 3, True),
    (1024, 3, True),
    (1025, 3, True),
    (1024, 3, True),
)


def _check_snapshot(
    snapshot: dict[str, Any], steps: int, legal: set[tuple[int, ...]]
) -> list[tuple[tuple[int, ...], float]]:
    assert snapshot["retired"], "Public result arrived before Worker release"
    assert not snapshot["active_sessions"] and snapshot["registry_size"] == 0
    assert snapshot["context_released"] and snapshot["workspace_released"]
    assert not any(snapshot["control_busy"]) and not snapshot["pending"]
    assert snapshot["error_code"] == 0
    assert snapshot["generated"] == steps
    assert snapshot["decode_steps"] == steps - 1
    assert snapshot["stage_index"] == steps - 1
    assert snapshot["graph_captured"] is not _EAGER
    sequences = _completed_sequences(snapshot, steps)
    assert all(tokens in {t[:steps] for t in legal} for tokens, _ in sequences)
    return sequences


def _write_evidence(
    work_dir: Path, mode: str, rows: list[dict[str, Any]], events: list[dict[str, Any]]
) -> None:
    dispatches = [e for e in events if e["method"] == "execute"]
    assert any(e["stage"] == 0 and e["scheduled"] < 900 for e in dispatches), "no cache hit"
    assert all(e["scheduled"] == 1 for e in dispatches if e["stage"] > 0)
    if not _EAGER:
        assert any(e["method"] == "replay" for e in events), "no actual Decode replay"
        assert any(e.get("outcome") == "replay" for e in events), "no actual Prefill replay"
    report = dict(
        mode=mode,
        width=_BEAM_WIDTH,
        eager=_EAGER,
        dtype=os.environ.get("VLLM_GR_TEST_DTYPE", "bfloat16"),
        rows=rows,
        events=events,
    )
    (work_dir / f"{mode}.json").write_text(json.dumps(report, indent=2))
    print(f"V1 {mode}: {len(rows)} public results, {len(dispatches)} dispatches", flush=True)


def _run_onerec_v1_scheduler(work_dir: Path) -> None:
    from vllm_gr.entrypoints.gr import GRLLM

    model, tokenizer, config, source, newline, legal = _inputs(work_dir)
    rows: list[dict[str, Any]] = []
    reference: dict[tuple[int, int], list[tuple[tuple[int, ...], float]]] = {}
    with GRLLM(vllm_gr_config=config, **_engine_kwargs(model, config)) as llm:
        expected_client = (
            "InprocClient"
            if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
            else "SyncMPClient"
        )
        assert type(llm.llm_engine.engine_core).__name__ == expected_client
        llm.collective_rpc("onerec_v1_st_observe")
        state_address = None
        for length, steps, cold in _CASES:
            if cold:
                assert llm.reset_prefix_cache()
            ids = (newline * length + source)[-(length - 1) :]
            public = llm.beam_search_v1(
                [{"prompt_token_ids": ids}], _params(steps), output_mode="full"
            )[0]
            snapshot = llm.collective_rpc("onerec_v1_st_snapshot")[0]
            sequences = _check_snapshot(snapshot, steps, legal)
            assert [
                (tuple(s.tokens[len(ids) + 1 : -1]), s.cum_logprob) for s in public.sequences
            ] == sequences
            assert all(s.finish_reason == "length" for s in public.sequences)
            assert all(s.tokens[: len(ids)] == ids for s in public.sequences)
            assert all(
                s.tokens[len(ids)] == tokenizer.convert_tokens_to_ids("<|sid_begin|>")
                and s.tokens[-1] == tokenizer.convert_tokens_to_ids("<|sid_end|>")
                for s in public.sequences
            )
            assert not llm.llm_engine.has_unfinished_requests()
            if state_address is not None:
                assert state_address == snapshot["state_address"]
            state_address = snapshot["state_address"]
            if cold and (length, steps) in reference:
                assert sequences == reference[length, steps]
            reference[length, steps] = sequences
            rows.append(dict(prompt=length, n=steps, cold=cold, sequences=sequences))
        _write_evidence(work_dir, "offline", rows, llm.collective_rpc("onerec_v1_st_events")[0])
        assert llm.reset_prefix_cache()


async def _run_online(work_dir: Path) -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args,
    )
    from vllm.entrypoints.openai.engine.serving import OpenAIServing
    from vllm.entrypoints.openai.models.serving import (
        BaseModelPath,
        OpenAIServingModels,
    )
    from vllm.inputs import tokens_input

    from vllm_gr.plugin import initialize_runtime

    initialize_runtime()
    model, tokenizer, config, source, newline, legal = _inputs(work_dir)
    args = AsyncEngineArgs(additional_config={"vllm_gr": config}, **_engine_kwargs(model, config))
    async with build_async_engine_client_from_engine_args(args) as engine:
        serving = OpenAIServing(
            engine_client=engine,
            request_logger=None,
            models=OpenAIServingModels(
                engine_client=engine,
                base_model_paths=[BaseModelPath(name=model, model_path=model)],
                lora_modules=None,
            ),
        )
        await engine.collective_rpc("onerec_v1_st_observe")
        rows = []

        async def run(length: int, steps: int, cold: bool, request_id: str) -> None:
            if cold:
                assert await engine.reset_prefix_cache()
            ids = (newline * length + source)[-(length - 1) :]
            outputs = [
                out
                async for out in serving.beam_search_v1(
                    tokens_input(ids), request_id, _params(steps)
                )
            ]
            assert len(outputs) == 1 and outputs[0].finished
            output = outputs[0]
            assert output.request_id == request_id and output.prompt_token_ids == ids
            snapshot = (await engine.collective_rpc("onerec_v1_st_snapshot"))[0]
            sequences = _check_snapshot(snapshot, steps, legal)
            assert [
                (tuple(c.token_ids[1:-1]), c.cumulative_logprob) for c in output.outputs
            ] == sequences
            for index, completion in enumerate(output.outputs):
                assert completion.index == index and completion.finish_reason == "length"
                assert completion.stop_reason is None and completion.logprobs is None
                assert len(completion.token_ids) == steps + 2
                assert completion.token_ids[0] == tokenizer.convert_tokens_to_ids("<|sid_begin|>")
                assert completion.token_ids[-1] == tokenizer.convert_tokens_to_ids("<|sid_end|>")
            assert not engine.output_processor.has_unfinished_requests()
            rows.append(dict(prompt=length, n=steps, cold=cold, sequences=sequences))

        for length, steps, cold in _CASES:
            await run(length, steps, cold, "same-public-id")

        async def cancel(*, waiting: bool) -> None:
            if waiting:
                await engine.pause_generation(mode="keep", clear_cache=False)
            before = len((await engine.collective_rpc("onerec_v1_st_events"))[0])
            stream = serving.beam_search_v1(tokens_input(newline * 1023), "cancelled", _params(3))
            task = asyncio.create_task(anext(stream))
            try:
                if waiting:
                    await asyncio.sleep(0.05)
                    assert len((await engine.collective_rpc("onerec_v1_st_events"))[0]) == before
                else:
                    deadline = time.monotonic() + 30
                    while True:
                        events = (await engine.collective_rpc("onerec_v1_st_events"))[0][before:]
                        if any(e["method"] == "execute" for e in events):
                            break
                        assert not task.done() and time.monotonic() < deadline
                        await asyncio.sleep(0.001)
                assert not task.done(), "cancellation missed the active request"
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                await stream.aclose()
                if waiting:
                    await engine.resume_generation()
            await run(1024, 3, False, "after-cancel")

        await cancel(waiting=True)
        await cancel(waiting=False)
        _write_evidence(
            work_dir, "online", rows, (await engine.collective_rpc("onerec_v1_st_events"))[0]
        )


async def _run_online_completion_batch(work_dir: Path) -> None:
    """Drive a prompt list through the completions HTTP route.

    The route is the whole point: the same list posted here used to become one
    session per prompt. The assertions are on what the engine was actually
    asked to run, read off the existing stage probe, not on the response alone
    -- a single response object is what both shapes return.
    """
    with _measure("imports_seconds"):
        import httpx
        from fastapi import FastAPI
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.api_server import (
            build_async_engine_client_from_engine_args,
        )
        from vllm.entrypoints.openai.completion.api_router import attach_router
        from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
        from vllm.entrypoints.openai.models.serving import (
            BaseModelPath,
            OpenAIServingModels,
        )
        from vllm.entrypoints.serve.render.serving import OpenAIServingRender

        from vllm_gr.plugin import initialize_runtime

    with _measure("runtime_init_seconds"):
        initialize_runtime()
    with _measure("inputs_seconds"):
        model, tokenizer, config, source, newline, _legal = _inputs(work_dir)
    sizes = (1, 2, 3, 4, 8)
    _MEASUREMENTS["batch_sizes"] = sizes
    config["beam"]["online_batch_sizes"] = list(sizes)
    kwargs = _engine_kwargs(model, config) | dict(
        max_num_seqs=max(sizes), max_num_batched_tokens=4096
    )
    args = AsyncEngineArgs(additional_config={"vllm_gr": config}, **kwargs)
    async with _measured_engine(build_async_engine_client_from_engine_args, args) as engine:
        models = OpenAIServingModels(
            engine_client=engine,
            base_model_paths=[BaseModelPath(name=model, model_path=model)],
            lora_modules=None,
        )
        serving = OpenAIServingCompletion(
            engine_client=engine,
            models=models,
            openai_serving_render=OpenAIServingRender(
                model_config=engine.model_config,
                renderer=engine.renderer,
                model_registry=models.registry,
                request_logger=None,
                chat_template=None,
                chat_template_content_format="auto",
            ),
            request_logger=None,
        )
        await engine.collective_rpc("onerec_v1_st_observe")
        app = FastAPI()
        app.state.openai_serving_completion = serving
        attach_router(app)
        prompts = [
            tokenizer.encode(f"slot {i}", add_special_tokens=False)
            + newline * (i * 16)
            + source[-256:]
            for i in range(max(sizes))
        ]
        body = {
            "model": model,
            "use_beam_search": True,
            "n": _BEAM_WIDTH,
            "temperature": 0.0,
            "max_tokens": 5,
            "return_token_ids": True,
            "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
        }

        async def stages_since(before: int) -> list[dict[str, Any]]:
            events = (await engine.collective_rpc("onerec_v1_st_events"))[0][before:]
            return [e for e in events if e["method"] == "execute"]

        rows = _MEASUREMENTS["rows"]
        with _measure("request_seconds"):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                # A size this route forms runs as one session holding every prompt.
                for count in sizes:
                    rows.append(dict(count=count, passed=False))
                    _persist_measurements()
                    assert await engine.reset_prefix_cache()
                    before = len((await engine.collective_rpc("onerec_v1_st_events"))[0])
                    response = await client.post(
                        "/v1/completions", json={**body, "prompt": prompts[:count]}
                    )
                    rows[-1]["status"] = response.status_code
                    _persist_measurements()
                    assert response.status_code == 200, response.text
                    data = response.json()
                    assert len(data["choices"]) == count * _BEAM_WIDTH
                    assert [c["index"] for c in data["choices"]] == list(range(count * _BEAM_WIDTH))
                    assert data["usage"]["prompt_tokens"] == sum(map(len, prompts[:count]))
                    assert data["usage"]["completion_tokens"] == count * _BEAM_WIDTH * 5
                    stages = await stages_since(before)
                    rows[-1].update(usage=data["usage"], stages=stages)
                    _MEASUREMENTS["events"] = (await engine.collective_rpc("onerec_v1_st_events"))[
                        0
                    ]
                    _persist_measurements()
                    assert [e["size"] for e in stages if e["stage"] == 0] == [count]
                    assert len({e["session_id"] for e in stages}) == 1
                    snapshot = (await engine.collective_rpc("onerec_v1_st_snapshot"))[0]
                    assert snapshot["retired"] and snapshot["registry_size"] == 0
                    assert not snapshot["active_sessions"]
                    assert not engine.output_processor.has_unfinished_requests()
                    rows[-1]["passed"] = True
                    _persist_measurements()


@pytest.mark.slow
@pytest.mark.timeout(660)
def test_onerec_v1_completion_batch_online(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("OneRec completion batching ST requires CUDA")
    env = dict(os.environ)
    env.update(
        USE_TF="0",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    command = [
        sys.executable,
        "-m",
        "tests.st.test_onerec_v1_scheduler_st",
        str(tmp_path),
        "--completion-batch",
    ]
    process_started = time.monotonic()
    with subprocess.Popen(
        command, cwd=Path(__file__).parents[2], env=env, start_new_session=True
    ) as process:
        try:
            assert process.wait(timeout=600) == 0, "OneRec completion batch subprocess failed"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            (tmp_path / "subprocess-evidence.json").write_text(
                json.dumps(
                    dict(
                        wall_seconds=time.monotonic() - process_started,
                        returncode=process.returncode,
                    ),
                    indent=2,
                )
            )


@pytest.mark.slow
@pytest.mark.timeout(660)
@pytest.mark.parametrize("online", [False, True])
def test_onerec_v1_async_scheduling(tmp_path: Path, online: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("OneRec V1 scheduling ST requires CUDA")
    env = dict(os.environ)
    env.update(
        USE_TF="0",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING=(
            "1" if online else os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
        ),
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    command = [sys.executable, "-m", "tests.st.test_onerec_v1_scheduler_st", str(tmp_path)]
    if online:
        command.append("--online")
    process_started = time.monotonic()
    with subprocess.Popen(
        command, cwd=Path(__file__).parents[2], env=env, start_new_session=True
    ) as process:
        try:
            assert process.wait(timeout=600) == 0, "OneRec V1 scheduling subprocess failed"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            (tmp_path / "subprocess-evidence.json").write_text(
                json.dumps(
                    dict(
                        wall_seconds=time.monotonic() - process_started,
                        returncode=process.returncode,
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    mode = (
        "completion-batch"
        if "--completion-batch" in sys.argv
        else ("online" if "--online" in sys.argv else "offline")
    )
    _EVIDENCE_PATH = Path(sys.argv[1]) / f"{mode}-evidence.json"
    _MEASUREMENTS.update(mode=mode, status="running")
    _persist_measurements()
    try:
        if mode == "completion-batch":
            asyncio.run(_run_online_completion_batch(Path(sys.argv[1])))
        elif mode == "online":
            asyncio.run(_run_online(Path(sys.argv[1])))
        else:
            _run_onerec_v1_scheduler(Path(sys.argv[1]))
    except BaseException as exc:
        _MEASUREMENTS.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        _MEASUREMENTS["status"] = "passed"
    finally:
        _MEASUREMENTS["module_ready_to_exit_seconds"] = time.monotonic() - _MODULE_READY
        _persist_measurements()
