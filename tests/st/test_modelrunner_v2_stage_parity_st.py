# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""OneRec V2 public results, lifecycle and actual Prefill/Decode execution paths.

One shared baseline covers Decode eager, Decode capture fallback and Prefill
FULL-to-PIECEWISE fallback. Every arm checks cache reuse, retirement and repeat
determinism; cross-arm comparisons allow BF16 rounding differences.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_SESSIONS = ((1024, 3), (900, 1), (900, 2), (1024, 3))
_EXTENSIONS = {
    "graph": "tests.st.test_modelrunner_v2_stage_parity_st.OneRecV2ParityWorkerExtension",
    "eager": "tests.st.test_modelrunner_v2_stage_parity_st.OneRecV2ParityWorkerExtension",
    "decode_fallback": "tests.st.v2_decode_failure_worker.FailedDecodeWorkerExtension",
    "prefill_fallback": "tests.st.v2_prefill_failure_worker.FailedFullWorkerExtension",
}


class OneRecV2ParityWorkerExtension:
    """Observe actual graph calls after startup without changing their results."""

    model_runner: Any
    _v2_parity_probe: ExitStack | None = None
    _v2_parity_counts: dict[str, int]
    _v2_prefill_tokens: int

    def onerec_v2_start_decode_probe(self) -> None:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context

        from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeGraph
        from vllm_gr.v1.worker.gpu_beam_stage_runner import GPUBeamStageRunner
        from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import V2StageExecutionHooks
        from vllm_gr.v1.worker.gpu_prefill_graph_runner_v2 import (
            GRPrefillCudaGraphManager,
        )

        assert self._v2_parity_probe is None
        runner = self.model_runner
        manager = runner.cudagraph_manager
        assert isinstance(manager, GRPrefillCudaGraphManager)
        counts = self._v2_parity_counts = {"replay": 0, "eager": 0, "full": 0, "piecewise": 0}
        original_replay = BeamDecodeGraph.replay
        original_eager = GPUBeamStageRunner._forward
        original_full = manager.run_fullgraph
        original_piecewise = CUDAGraphWrapper.__call__
        original_prefill = V2StageExecutionHooks.prefill_logits

        @wraps(original_replay)
        def replay(graph: Any) -> Any:
            result = original_replay(graph)
            if graph.worker is runner:
                counts["replay"] += 1
            return result

        @wraps(original_eager)
        def eager(owner: Any) -> Any:
            result = original_eager(owner)
            if owner.worker is runner:
                counts["eager"] += 1
            return result

        @wraps(original_full)
        def full(desc: Any) -> Any:
            result = original_full(desc)
            counts["full"] += 1
            return result

        @wraps(original_piecewise)
        def piecewise(wrapper: Any, *args: Any, **kwargs: Any) -> Any:
            context = get_forward_context()
            entry = wrapper.concrete_cudagraph_entries.get(context.batch_descriptor)
            is_replay = (
                wrapper.runtime_mode == CUDAGraphMode.PIECEWISE
                and context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                and entry is not None
                and entry.cudagraph is not None
            )
            result = original_piecewise(wrapper, *args, **kwargs)
            if is_replay:
                counts["piecewise"] += 1
            return result

        @wraps(original_prefill)
        def prefill(hooks: Any, original: Any, scheduled: Any, *args: Any, **kwargs: Any) -> Any:
            if hooks.worker is runner:
                self._v2_prefill_tokens = scheduled.total_num_scheduled_tokens
            return original_prefill(hooks, original, scheduled, *args, **kwargs)

        with ExitStack() as probes:
            probes.enter_context(patch.object(BeamDecodeGraph, "replay", replay))
            probes.enter_context(patch.object(GPUBeamStageRunner, "_forward", eager))
            probes.enter_context(patch.object(manager, "run_fullgraph", full))
            probes.enter_context(patch.object(CUDAGraphWrapper, "__call__", piecewise))
            probes.enter_context(patch.object(V2StageExecutionHooks, "prefill_logits", prefill))
            self._v2_parity_probe = probes.pop_all()

    def onerec_v2_stop_decode_probe(self) -> None:
        if self._v2_parity_probe is not None:
            self._v2_parity_probe.close()
            self._v2_parity_probe = None

    def onerec_v2_parity_snapshot(self) -> dict[str, Any]:
        from tests.st.test_onerec_v1_scheduler_st import _worker_snapshot
        from vllm_gr.v1.worker.model_runner_common import get_beam_graph_runtime_stats

        snapshot = _worker_snapshot(self)
        if snapshot["retired"]:
            runner = self.model_runner
            graph = runner._gr_gpu_stage_runner.graph
            snapshot.update(
                decode_graph_ready=graph is not None and graph.ready,
                decode_capture_failed=bool(
                    getattr(runner, "_gr_async_decode_graph_capture_failed", False)
                ),
                decode_capture_error=getattr(runner, "_gr_async_decode_graph_capture_error", None),
                counts=dict(self._v2_parity_counts),
                prefill_tokens=self._v2_prefill_tokens,
                prefill_stats=get_beam_graph_runtime_stats(self),
                native_requests=runner.req_states.num_reqs,
            )
        return snapshot


@contextmanager
def _decode_path_probe(llm: Any) -> Iterator[None]:
    llm.collective_rpc("onerec_v2_start_decode_probe", timeout=30)
    try:
        yield
    finally:
        llm.collective_rpc("onerec_v2_stop_decode_probe", timeout=30)


def _run_arm(work_dir: Path, arm: str) -> None:
    from tests.st.test_onerec_v1_scheduler_st import (
        _completed_sequences,
        _engine_kwargs,
        _inputs,
        _params,
    )
    from vllm_gr.entrypoints.gr import GRLLM

    model, tokenizer, config, source, newline, legal = _inputs(work_dir)
    config["beam"]["graph_enabled"] = arm != "eager"
    kwargs = _engine_kwargs(model, config)
    kwargs["worker_extension_cls"] = _EXTENSIONS[arm]
    decode_graph = arm in ("graph", "prefill_fallback")
    results: list[dict[str, Any]] = []
    with GRLLM(vllm_gr_config=config, **kwargs) as llm, _decode_path_probe(llm):
        previous = {"replay": 0, "eager": 0, "full": 0, "piecewise": 0}
        state_address = None
        session_ids: set[str] = set()
        for index, (length, steps) in enumerate(_SESSIONS):
            if index in (0, 3):
                assert llm.reset_prefix_cache()
            ids = (newline * length + source)[-(length - 1) :]
            public = llm.beam_search_v1([{"prompt_token_ids": ids}], _params(steps))[0]
            snapshot = llm.collective_rpc("onerec_v2_parity_snapshot", timeout=30)[0]
            assert snapshot["retired"], "public output arrived before Worker retirement"
            assert snapshot["session_id"] not in session_ids
            session_ids.add(snapshot["session_id"])
            assert not snapshot["active_sessions"] and snapshot["registry_size"] == 0
            assert snapshot["context_released"] and snapshot["workspace_released"]
            assert not any(snapshot["control_busy"]) and not snapshot["pending"]
            assert snapshot["native_requests"] == 0
            assert not llm.llm_engine.has_unfinished_requests()
            assert snapshot["error_code"] == 0
            assert snapshot["generated"] == steps
            assert snapshot["decode_steps"] == snapshot["stage_index"] == steps - 1
            if state_address is not None:
                assert snapshot["state_address"] == state_address
            state_address = snapshot["state_address"]
            if index in (0, 3):
                assert snapshot["prefill_tokens"] == length
            elif index == 2:
                assert 0 < snapshot["prefill_tokens"] < length, "no prefix-cache hit"

            counts = snapshot["counts"]
            delta = {name: counts[name] - previous[name] for name in counts}
            previous = counts
            assert snapshot["decode_capture_failed"] is (arm == "decode_fallback")
            assert snapshot["decode_graph_ready"] is decode_graph
            assert (delta["replay"], delta["eager"]) == (
                (steps - 1, 0) if decode_graph else (0, steps - 1)
            )
            if arm == "decode_fallback":
                assert "not supported" in snapshot["decode_capture_error"]
            stats = snapshot["prefill_stats"]
            assert stats["prefill_graph_none_dispatches"] == 0
            if arm == "prefill_fallback":
                assert stats["prefill_graph_full_capture_failures"]
                assert not stats["prefill_graph_captured_buckets"]
                assert delta["full"] == 0 and delta["piecewise"] > 0
            else:
                assert not stats["prefill_graph_full_capture_failures"]
                assert delta["full"] == 1 and delta["piecewise"] == 0

            sequences = _completed_sequences(snapshot, steps)
            prefixes = {triple[:steps] for triple in legal}
            assert all(tokens in prefixes for tokens, _ in sequences)
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
            results.append({"length": length, "steps": steps, "sequences": sequences})
            print(f"V2 {arm}: prompt={length}, steps={steps}, paths={delta}, retired", flush=True)
        assert results[0] == results[3], "same-geometry repeat changed results"
        assert llm.reset_prefix_cache(), "scheduler retained resources after retirement"
    (work_dir / "results.json").write_text(json.dumps(results))


@pytest.mark.slow
@pytest.mark.timeout(3900)
def test_modelrunner_v2_stage_parity(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("OneRec V2 requires CUDA")
    env = dict(os.environ)
    env.update(
        USE_TF="0",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ENABLE_PREFILL_CUDAGRAPH="1",
        VLLM_GR_TEST_WIDTH="128",
        VLLM_GR_TEST_EAGER="0",
        VLLM_GR_TEST_DTYPE="bfloat16",
    )
    baseline = None
    for arm in _EXTENSIONS:
        work_dir = tmp_path / arm
        work_dir.mkdir()
        env["VLLM_GR_TEST_ARTIFACT"] = str(work_dir / "constraints.safetensors")
        command = [
            sys.executable,
            "-m",
            "tests.st.test_modelrunner_v2_stage_parity_st",
            str(work_dir),
            arm,
        ]
        with subprocess.Popen(
            command, cwd=Path(__file__).parents[2], env=env, start_new_session=True
        ) as process:
            try:
                assert process.wait(timeout=900) == 0, f"V2 {arm} arm failed"
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        results = json.loads((work_dir / "results.json").read_text())
        if baseline is None:
            baseline = results
        else:
            for reference, result in zip(baseline, results, strict=True):
                assert (reference["length"], reference["steps"]) == (
                    result["length"],
                    result["steps"],
                )
                left = {tuple(seq) for seq, _ in reference["sequences"]}
                right = {tuple(seq) for seq, _ in result["sequences"]}
                assert len(left & right) / len(left) >= 0.95, (arm, reference, result)
                assert reference["sequences"][0][0] == result["sequences"][0][0]


if __name__ == "__main__":
    _run_arm(Path(sys.argv[1]), sys.argv[2])
