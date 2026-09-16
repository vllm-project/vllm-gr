# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""Real OneRec weights through the V1 multiprocess Scheduler/Worker pipeline.

A5 public output conversion is not implemented yet. Submit through the native
session API and inspect the completed device records after Worker retirement;
no Forward, scheduler, sampling, output conversion or release hook is replaced.
This guards scheduling and device results, not the public beam_search_v1 API.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import pytest

_BEAM_WIDTH = 128


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


def _run_onerec_v1_scheduler(work_dir: Path) -> None:
    from transformers import AutoConfig, AutoTokenizer

    from vllm_gr.config import BeamExecutionMode, GRConfig
    from vllm_gr.constraints.artifact import save_constraint_table_artifact
    from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
    from vllm_gr.constraints.json_compiler import compile_relations_json
    from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest
    from vllm_gr.entrypoints.beam_search_v1.input_processor import (
        BeamSearchInputProcessor,
    )
    from vllm_gr.entrypoints.beam_search_v1.protocol import OfflineBeamSearchRequest
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

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
    with GRLLM(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        vllm_gr_config=config,
        async_scheduling=True,
        worker_extension_cls="tests.st.test_onerec_v1_scheduler_st.OneRecV1STWorkerExtension",
        max_num_seqs=1,
        max_model_len=2048,
        max_num_batched_tokens=512,
        max_logprobs=_BEAM_WIDTH,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
    ) as llm:
        engine = llm.llm_engine
        assert (
            GRConfig.from_vllm_config(engine.vllm_config).beam_execution_mode
            is BeamExecutionMode.V1
        )
        assert engine.vllm_config.scheduler_config.async_scheduling
        assert type(engine.engine_core).__name__ == "SyncMPClient"
        processor = BeamSearchInputProcessor(engine)
        previous_ids = set()
        first_sequences = None
        state_address = None
        # Long Prefill crosses two native chunks; the intervening requests
        # change length/budget and reuse the 900-token prompt's cached KV.
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
            # Direct native submission deliberately stops below the missing
            # A5 frontend converter. The production scheduling path is intact.
            request = processor.process_offline_inputs(
                llm, OfflineBeamSearchRequest([{"prompt_token_ids": ids}], params)
            )
            assert request.batch_id not in previous_ids
            previous_ids.add(request.batch_id)
            handle = engine.submit_gr_session(request)
            result = engine.wait_gr_session(request.batch_id, handle)
            assert result.batch_id == request.batch_id
            if result.failure is not None:
                assert result.failure.code == "worker_result_unavailable", result.failure

            deadline = time.monotonic() + 30
            while True:
                snapshot = llm.collective_rpc("onerec_v1_st_snapshot", timeout=30)[0]
                if snapshot["retired"] and snapshot["session_id"] == request.batch_id:
                    break
                assert time.monotonic() < deadline, "V1 Worker retirement did not finish"
                time.sleep(0.01)
            assert not engine.has_unfinished_requests()
            assert snapshot["active_sessions"] == []
            assert snapshot["registry_size"] == 0
            assert snapshot["context_released"] and snapshot["workspace_released"]
            assert not any(snapshot["control_busy"]) and not snapshot["pending"]
            assert snapshot["error_code"] == 0
            assert snapshot["generated"] == steps
            assert snapshot["decode_steps"] == steps - 1
            assert snapshot["stage_index"] == steps - 1
            assert snapshot["graph_captured"]
            sequences = _completed_sequences(snapshot, steps)
            prefixes = {tokens[:steps] for tokens in legal}
            assert all(tokens in prefixes for tokens, _ in sequences)
            if state_address is None:
                state_address = snapshot["state_address"]
                first_sequences = sequences
            else:
                assert snapshot["state_address"] == state_address
            if index == 3:
                assert sequences == first_sequences, "same-geometry repeat changed device results"
            engine.release_gr_session(request.batch_id)
            print(
                f"V1 OneRec: prompt={length}, steps={steps}, beams={len(sequences)}, retired",
                flush=True,
            )
        assert llm.reset_prefix_cache(), "Scheduler retained resources after Worker retirement"


@pytest.mark.slow
@pytest.mark.timeout(660)
def test_onerec_v1_async_scheduling(tmp_path: Path) -> None:
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
    _run_onerec_v1_scheduler(Path(sys.argv[1]))
