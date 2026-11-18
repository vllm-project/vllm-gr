# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""OneRec V1/V2 execution regression; run each runner/mode in a fresh process.

This is an explicit real-model probe, not a pytest-collected test or benchmark.
Copy this file into a baseline checkout to compare V1 with --reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch


class PolicyWorkerExtension:
    """Observe only execution paths needed by the regression and accuracy runs."""

    model_runner: Any

    def policy_observe(self) -> None:
        from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeGraph
        from vllm_gr.v1.worker.gpu_beam_stage_runner import GPUBeamStageRunner
        from vllm_gr.v1.worker.gpu_beam_stage_runner_v1 import CudaStageHooksV1
        from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import CudaStageHooksV2

        self._policy_counts = {"capture": 0, "replay": 0, "eager": 0}
        self._policy_queries: list[list[int]] = []
        self._policy_probes = ExitStack()
        for cls, method, counter in (
            (BeamDecodeGraph, "capture", "capture"),
            (BeamDecodeGraph, "replay", "replay"),
            (GPUBeamStageRunner, "_forward", "eager"),
        ):
            original = getattr(cls, method)

            @wraps(original)
            def observe(
                obj: Any, *args: Any, _fn: Any = original, _key: str = counter, **kwargs: Any
            ) -> Any:
                if obj.worker is self.model_runner:
                    self._policy_counts[_key] += 1
                return _fn(obj, *args, **kwargs)

            self._policy_probes.enter_context(patch.object(cls, method, observe))
        hooks = CudaStageHooksV2 if hasattr(self.model_runner, "req_states") else CudaStageHooksV1
        original_prefill = hooks.prefill_logits

        @wraps(original_prefill)
        def prefill(hook: Any, original: Any, scheduled: Any, *args: Any, **kwargs: Any) -> Any:
            if hook.worker is self.model_runner:
                self._policy_queries.append(list(scheduled.num_scheduled_tokens.values()))
            return original_prefill(hook, original, scheduled, *args, **kwargs)

        self._policy_probes.enter_context(patch.object(hooks, "prefill_logits", prefill))
        if hooks is CudaStageHooksV1:
            from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

            original_execute = GPUPrefillGraphRunner.execute_model_prefill
            self._policy_counts["prefill_replay"] = 0

            @wraps(original_execute)
            def execute(obj: Any, *args: Any, **kwargs: Any) -> Any:
                result = original_execute(obj, *args, **kwargs)
                if obj.model_runner is self.model_runner and obj.last_replay_reason == "replay":
                    self._policy_counts["prefill_replay"] += 1
                return result

            self._policy_probes.enter_context(
                patch.object(GPUPrefillGraphRunner, "execute_model_prefill", execute)
            )

    def policy_stop(self) -> None:
        self._policy_probes.close()

    def policy_snapshot(self) -> dict[str, Any]:
        import torch

        from vllm_gr.v1.worker import model_runner_patch
        from vllm_gr.v1.worker.gr_context import (
            get_beam_graph_runtime_stats,
        )

        runner = self.model_runner
        owner = getattr(runner, "_gr_gpu_stage_runner", None)
        graph = getattr(runner, "_beam_decode_graph", None)
        inputs = getattr(runner, "_gr_input_buffers", None)
        states = getattr(runner, "req_states", None)
        return {
            "source": str(Path(model_runner_patch.__file__).resolve()),
            "runner_class": f"{type(runner).__module__}.{type(runner).__name__}",
            "decode_batches": sorted(s.batch_size for s in graph._bindings) if graph else [],
            "decode_failures": getattr(runner, "_gr_decode_capture_failures", {}),
            "addresses": {
                name: getattr(inputs, name).data_ptr()
                for name in ("device", "input_ids", "positions", "slots", "execution_mask")
            }
            if inputs is not None
            else {},
            "backend_rows": runner.constraint_backend.info.max_rows,
            "owner_closed": owner is None or owner.closed,
            "native_requests": states.num_reqs if states is not None else len(runner.requests),
            "pending_native_output": getattr(runner, "execute_model_state", None) is not None,
            "sessions": len(getattr(runner, "_beam_sessions", {})),
            "resources_released": owner is None
            or (
                owner.beam.session.context.registry.num_active_sessions == 0
                and owner.beam.workspace.execution_owner is None
                and owner.beam.session.context._gr_execution_owner is None
                and not any(owner.control_busy)
            ),
            "memory_allocated": torch.accelerator.memory_allocated(),
            "memory_reserved": torch.accelerator.memory_reserved(),
            "graphs": get_beam_graph_runtime_stats(self),
            "counts": dict(getattr(self, "_policy_counts", {})),
            "queries": list(getattr(self, "_policy_queries", [])),
        }


def check_idle(value: dict[str, Any]) -> None:
    assert value["owner_closed"] and value["native_requests"] == value["sessions"] == 0
    assert value["resources_released"]
    assert not value["pending_native_output"]
    assert not value["decode_failures"]


def startup_expectations(
    *, runner: str, config: dict[str, Any], max_num_seqs: int, enforce_eager: bool
) -> tuple[int, list[int]]:
    """Keep persistent Backend capacity separate from the selected graph set."""
    from vllm_gr.config import gr_config_from_mapping

    cfg = gr_config_from_mapping(config)
    assert cfg.beam_max_width is not None
    if runner == "v2":
        from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES

        max_batch = max(SUPPORTED_BEAM_BATCH_SIZES)
        batches = [b for b in cfg.beam_online_batch_sizes if b <= max_num_seqs]
    else:
        assert runner == "v1", runner
        from vllm_gr.v1.worker.gr_startup import _startup_capture_batch_size

        max_batch = 1
        batches = [_startup_capture_batch_size()]
    if enforce_eager or not cfg.beam_graph_enabled:
        batches = []
    return max_batch * cfg.beam_max_width, batches


def run(args: argparse.Namespace) -> None:
    os.environ.update(
        VLLM_USE_V2_MODEL_RUNNER=str(int(args.runner == "v2")),
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ENABLE_PREFILL_CUDAGRAPH="1",
        VLLM_GR_PREFILL_GRAPH_KV_BOUND="256",
    )
    for name in ("VLLM_GR_PREFILL_GRAPH_BUCKET_STEP", "VLLM_GR_PREFILL_GRAPH_MAX_BUCKET"):
        os.environ.pop(name, None)

    from tests.st.test_onerec_v1_scheduler_st import _engine_kwargs, _inputs, _params
    from vllm_gr import GRLLM

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    model, _tokenizer, config, source, _newline, legal = _inputs(output)
    width = config["beam"]["max_width"]
    config["beam"]["graph_enabled"] = args.mode == "graph"
    kwargs = _engine_kwargs(model, config)
    kwargs.update(
        max_num_seqs=4,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=256,
        max_num_batched_tokens=1024,
        enforce_eager=args.mode == "eager",
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 128, 256, 512, 1024]},
        worker_extension_cls="tests.st.model_runner_policy_isolation.PolicyWorkerExtension",
        shutdown_timeout=10,
    )
    expected_rows, expected_batches = startup_expectations(
        runner=args.runner,
        config=config,
        max_num_seqs=kwargs["max_num_seqs"],
        enforce_eager=kwargs["enforce_eager"],
    )
    # Distinct content and unsorted lengths exercise native row reordering.
    prompts = [source[i * 13 : i * 13 + n] for i, n in enumerate((191, 96, 160, 127))]
    assert [len(p) for p in prompts] == [191, 96, 160, 127]
    result: dict[str, Any] = {
        "runner": args.runner,
        "mode": args.mode,
        "settings": kwargs,
        "prompts": prompts,
        "cases": [],
    }

    def save() -> None:
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    llm = GRLLM(**kwargs, vllm_gr_config=config)
    processes = list(llm.llm_engine.engine_core.resources.engine_manager.processes)
    addresses = None

    def snapshot() -> dict[str, Any]:
        nonlocal addresses
        value: dict[str, Any] = llm.collective_rpc("policy_snapshot")[0]
        check_idle(value)
        if args.expected_source:
            assert Path(value["source"]).parents[3] == args.expected_source.resolve()
        assert (".gpu.model_runner." in value["runner_class"]) == (args.runner == "v2")
        assert value["backend_rows"] == expected_rows
        if args.runner == "v1":
            assert not value["addresses"]
        else:
            if value["addresses"]:
                if addresses is None:
                    addresses = value["addresses"]
                assert value["addresses"] == addresses
            assert value["counts"].get("capture", 0) == 0
        return value

    def generate(ids_list: list[list[int]], steps: int) -> list[list[dict[str, Any]]]:
        outputs = llm.beam_search_v1(
            [{"prompt_token_ids": ids} for ids in ids_list],
            _params(steps),
            use_tqdm=False,
            output_mode="full",
        )
        assert len(outputs) == len(ids_list)
        rows = []
        legal_prefixes = {t[:steps] for t in legal}
        for ids, member in zip(ids_list, outputs, strict=True):
            assert all(seq.tokens[: len(ids)] == ids for seq in member.sequences)
            sequences = [
                {"tokens": seq.tokens[len(ids) + 1 : -1], "score": seq.cum_logprob}
                for seq in member.sequences
            ]
            assert len(sequences) == len({tuple(s["tokens"]) for s in sequences}) == width
            assert all(tuple(s["tokens"]) in legal_prefixes for s in sequences)
            scores = [s["score"] for s in sequences]
            assert all(math.isfinite(score) for score in scores) and scores == sorted(
                scores, reverse=True
            )
            rows.append(sequences)
        return rows

    previous: dict[tuple[int, int, str], list[list[dict[str, Any]]]] = {}

    def measured(batch: int, cache: str, steps: int = 3) -> None:
        before = snapshot()
        rows = generate(prompts[:batch], steps)
        after = snapshot()
        counts = {k: v - before["counts"].get(k, 0) for k, v in after["counts"].items()}
        assert counts["replay"] == (steps - 1 if args.mode == "graph" else 0)
        assert counts["eager"] == (steps - 1 if args.mode == "eager" else 0)
        if args.runner == "v2" and args.mode == "graph":
            assert (
                after["graphs"]["prefill_graph_full_dispatches"]
                - before["graphs"]["prefill_graph_full_dispatches"]
                == 1
            )
        queries = after["queries"][len(before["queries"]) :]
        assert len(queries) == 1 and len(queries[0]) == batch
        if cache == "warm":
            assert all(0 < q <= 16 for q in queries[0]), "prefix cache was not reused"
        elif cache == "partial":
            assert min(queries[0]) <= 16 < max(queries[0]), "expected mixed cache hits and misses"
        key = steps, batch, cache
        if key in previous:
            assert rows == previous[key], "same-shape output changed after batch switch"
        previous[key] = rows
        result["cases"].append(
            {"batch": batch, "cache": cache, "steps": steps, "outputs": rows, "snapshot": after}
        )
        save()
        print(f"{args.runner} {args.mode} B={batch} N={steps} {cache} passed", flush=True)

    try:
        result["initial"] = snapshot()
        source_root = Path(result["initial"]["source"]).parents[3]
        result["runtime_sha256"] = {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source_root / "vllm_gr").rglob("*.py"))
        }
        assert result["initial"]["decode_batches"] == expected_batches
        llm.collective_rpc("policy_observe")
        for batch in (1, 2, 4, 1, 4, 2):
            assert llm.reset_prefix_cache()
            measured(batch, "cold")
            measured(batch, "warm")
        assert llm.reset_prefix_cache()
        measured(4, "cold")
        assert llm.reset_prefix_cache()
        # Warm a partial batch with complete SIDs from the three-layer table.
        for ids in prompts[:3]:
            generate([ids], 3)
            snapshot()
        measured(4, "partial")
        if args.reference:
            reference = json.loads(args.reference.read_text())
            assert all(result[key] == reference[key] for key in ("runner", "mode", "prompts"))
            for actual, expected_case in zip(result["cases"], reference["cases"], strict=True):
                assert all(
                    actual[key] == expected_case[key]
                    for key in ("batch", "cache", "steps", "outputs")
                ), "baseline output changed"
            result["baseline_exact"] = True
        result["inference_passed"] = True
        save()
        llm.collective_rpc("policy_stop")
    finally:
        llm.shutdown()
        shutdown = [dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive()) for p in processes]
        (output / "lifecycle.json").write_text(json.dumps(shutdown, indent=2))
        assert shutdown and all(not p["alive"] and p["exitcode"] == 0 for p in shutdown), shutdown
    result["passed"] = True
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("v1", "v2"), required=True)
    parser.add_argument("--mode", choices=("graph", "eager"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source", type=Path)
    parser.add_argument(
        "--reference", type=Path, help="Matching baseline results.json; require exact outputs"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    run(parser.parse_args())
