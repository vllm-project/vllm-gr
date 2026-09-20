# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""Real OneRec weights through the public V1 API and native multiprocess pipeline.

Read-only Worker probes validate device records and resource retirement after
public completion. Production scheduling, sampling and output conversion run
without replacements.
"""

from __future__ import annotations

import asyncio
import math
import os
import signal
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pytest

_BEAM_WIDTH = int(os.environ.get("VLLM_GR_ST_BEAM_WIDTH", "128"))


class OneRecV1STWorkerExtension:
    """Expose a read-only test probe through the native named utility RPC."""

    def onerec_v1_st_snapshot(self) -> dict[str, Any]:
        return _worker_snapshot(self)


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
        "final_d2h_count": owner.final_output.final_d2h_count,
        "final_d2h_bytes": owner.final_output.final_d2h_bytes,
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


def _onerec_inputs(
    work_dir: Path,
) -> tuple[str, Any, dict[str, Any], list[int], list[int], set[tuple[int, ...]]]:
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
        for a, b, c in product(range(_BEAM_WIDTH), range(4), range(4))
    ]
    tokenizer_digest = compute_tokenizer_digest(tokenizer)
    table = ConstraintTableArtifactCompiler(
        tokenizer, vocab_size, tokenizer_digest=tokenizer_digest
    ).compile(compile_relations_json(triples))
    artifact = work_dir / "onerec_v1_constraints.safetensors"
    digest = save_constraint_table_artifact(table, artifact)
    legal = {tuple(tokenizer.convert_tokens_to_ids(triple)) for triple in triples}
    config = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "graph_enabled": True,
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
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        vllm_gr_config=config,
        async_scheduling=True,
        worker_extension_cls="tests.st.test_onerec_v1_scheduler_st.OneRecV1STWorkerExtension",
        max_num_seqs=1,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_logprobs=_BEAM_WIDTH,
        enable_chunked_prefill=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
    )


def _run_onerec_v1_scheduler(work_dir: Path) -> None:
    from vllm_gr.config import BeamExecutionMode, GRConfig
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    model, tokenizer, config, source, newline, legal = _onerec_inputs(work_dir)
    with GRLLM(**_engine_kwargs(model, config)) as llm:
        engine = llm.llm_engine
        assert (
            GRConfig.from_vllm_config(engine.vllm_config).beam_execution_mode
            is BeamExecutionMode.V1
        )
        assert engine.vllm_config.scheduler_config.async_scheduling
        assert type(engine.engine_core).__name__ == "SyncMPClient"
        previous_ids = set()
        first_sequences = None
        state_address = None
        # Full Prefill changes length/budget and reuses cached prompt-only KV.
        for index, (length, steps) in enumerate(((1024, 3), (900, 1), (900, 2), (1024, 3))):
            if index == 3:
                # Exact BF16 equality needs matching Prefill geometry: a cache
                # hit can change the bucket and rounding without a scheduler bug.
                assert llm.reset_prefix_cache()
            ids = (newline * length + source)[-(length - 1) :]
            params = BeamSearchParams(
                beam_width=_BEAM_WIDTH,
                max_tokens=steps + 2,
                temperature=0.0,
                ignore_eos=True,
                begin_token="<|sid_begin|>",
                end_token="<|sid_end|>",
            )
            outputs = llm.beam_search_v1([{"prompt_token_ids": ids}], params, use_tqdm=False)
            assert len(outputs) == 1
            snapshot = llm.collective_rpc("onerec_v1_st_snapshot", timeout=30)[0]
            assert snapshot["retired"], "public return preceded Worker retirement"
            assert snapshot["session_id"] not in previous_ids
            previous_ids.add(snapshot["session_id"])
            public = outputs[0].sequences
            assert len(public) == _BEAM_WIDTH
            assert all(
                beam.finish_reason == "length" and beam.stop_reason is None for beam in public
            )
            assert all(beam.logprobs == [] for beam in public)
            assert all(beam.tokens[: len(ids)] == ids for beam in public)
            assert all(
                beam.tokens[-1] == tokenizer.convert_tokens_to_ids("<|sid_end|>") for beam in public
            )
            assert not engine.has_unfinished_requests()
            assert snapshot["active_sessions"] == []
            assert snapshot["registry_size"] == 0
            assert snapshot["context_released"] and snapshot["workspace_released"]
            assert not any(snapshot["control_busy"]) and not snapshot["pending"]
            assert snapshot["error_code"] == 0
            assert snapshot["final_d2h_count"] >= 1
            assert snapshot["generated"] == steps
            assert snapshot["decode_steps"] == steps - 1
            assert snapshot["stage_index"] == steps - 1
            assert snapshot["graph_captured"]
            sequences = _completed_sequences(snapshot, steps)
            assert [
                (tuple(beam.tokens[len(ids) + 1 : -1]), beam.cum_logprob) for beam in public
            ] == sequences
            prefixes = {tokens[:steps] for tokens in legal}
            assert all(tokens in prefixes for tokens, _ in sequences)
            if state_address is None:
                state_address = snapshot["state_address"]
                first_sequences = sequences
            else:
                assert snapshot["state_address"] == state_address
            if index == 3:
                assert sequences == first_sequences, "same-geometry repeat changed device results"
        assert llm.reset_prefix_cache(), "Scheduler retained resources after Worker retirement"


async def _run_onerec_v1_online(work_dir: Path) -> None:
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

    from vllm_gr.sampling_params import BeamSearchParams

    model, tokenizer, config, source, newline, legal = _onerec_inputs(work_dir)
    from vllm_gr.plugin import initialize_runtime

    initialize_runtime()
    kwargs = _engine_kwargs(model, config)
    kwargs.pop("vllm_gr_config")
    kwargs["additional_config"] = {"vllm_gr": config}
    args = AsyncEngineArgs(**kwargs)
    async with build_async_engine_client_from_engine_args(args) as engine:
        serving = OpenAIServing(
            engine_client=engine,
            models=OpenAIServingModels(
                engine_client=engine,
                base_model_paths=[BaseModelPath(name=model, model_path=model)],
                lora_modules=None,
            ),
            request_logger=None,
        )
        prompt = tokens_input((newline * 1023 + source)[-1023:])
        params = BeamSearchParams(
            beam_width=_BEAM_WIDTH,
            max_tokens=5,
            ignore_eos=True,
            begin_token="<|sid_begin|>",
            end_token="<|sid_end|>",
        )

        async def run(request_id: str, budget: int = 3) -> Any:
            from msgspec.structs import replace

            outputs = [
                output
                async for output in serving.beam_search_v1(
                    prompt, request_id, replace(params, max_tokens=budget + 2)
                )
            ]
            assert len(outputs) == 1 and outputs[0].finished
            output = outputs[0]
            assert output.request_id == request_id
            assert output.prompt_token_ids == prompt["prompt_token_ids"]
            snapshot = (await engine.collective_rpc("onerec_v1_st_snapshot"))[0]
            assert snapshot["retired"] and not snapshot["active_sessions"]
            assert snapshot["context_released"] and not any(snapshot["control_busy"])
            assert len(output.outputs) == _BEAM_WIDTH
            for index, completion in enumerate(output.outputs):
                assert completion.index == index and completion.finish_reason == "length"
                assert completion.stop_reason is None
                assert len(completion.token_ids) == budget + 2
                assert completion.logprobs is None
                assert tuple(completion.token_ids[1:-1]) in {t[:budget] for t in legal}
            assert not engine.output_processor.has_unfinished_requests()
            return output

        for budget in (1, 2, 3):
            await run("same-public-id", budget)
        # Cancel at several native handoff points, then verify a subsequent
        # public call can acquire the same Worker workspace safely.
        for delay in (0.0, 0.001, 0.005):
            stream = serving.beam_search_v1(prompt, "disconnected", params)
            task = asyncio.create_task(anext(stream))
            await asyncio.sleep(delay)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await stream.aclose()
            await run("after-cancel")
        # Explicit generator close after terminal delivery is also idempotent.
        stream = serving.beam_search_v1(prompt, "closed", params)
        await anext(stream)
        await stream.aclose()
        assert not engine.output_processor.has_unfinished_requests()


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
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    command = [sys.executable, "-m", "tests.st.test_onerec_v1_scheduler_st", str(tmp_path)]
    if online:
        command.append("online")
    with subprocess.Popen(
        command, cwd=Path(__file__).parents[2], env=env, start_new_session=True
    ) as process:
        try:
            assert process.wait(timeout=600) == 0, "OneRec V1 scheduling subprocess failed"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "online":
        asyncio.run(_run_onerec_v1_online(Path(sys.argv[1])))
    else:
        _run_onerec_v1_scheduler(Path(sys.argv[1]))
