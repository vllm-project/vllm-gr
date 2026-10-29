# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""Real OneRec weights through the native multiprocess Scheduler/Worker pipeline.

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
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
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
async def _measured_engine(builder: Any, args: Any) -> AsyncIterator[Any]:
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
_UNCONSTRAINED = os.environ.get("VLLM_GR_TEST_UNCONSTRAINED", "0") == "1"


class OneRecV1STWorkerExtension:
    """Expose a read-only test probe through the native named utility RPC."""

    def onerec_v1_st_snapshot(self) -> dict[str, Any]:
        return _worker_snapshot(self)

    def onerec_v1_st_graphs(self) -> dict[str, Any]:
        from vllm_gr.v1.worker.model_runner_common import get_beam_graph_runtime_stats

        return get_beam_graph_runtime_stats(self)

    def onerec_v1_st_runtime(self) -> dict[str, Any]:
        """Record resolved execution settings before the first request."""
        import torch
        import vllm.envs as envs

        runner = getattr(self, "model_runner")
        config = runner.vllm_config
        model = runner.get_model()
        return {
            "runner": f"{type(runner).__module__}.{type(runner).__name__}",
            "model": type(model).__name__,
            "config_hash": config.compute_hash(),
            "model_config_hash": config.model_config.compute_hash(),
            "dtype": str(config.model_config.dtype),
            "seed": config.model_config.seed,
            "scheduler": str(config.scheduler_config),
            "cache": str(config.cache_config),
            "kernel": str(config.kernel_config),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "aot_compile": envs.VLLM_USE_AOT_COMPILE,
            "mega_aot_artifact": envs.VLLM_USE_MEGA_AOT_ARTIFACT,
            "compile_cache_dir": config.compilation_config.local_cache_dir,
            "inductor_compile_config": dict(config.compilation_config.inductor_compile_config),
            "compiled_modules": {
                name: {
                    "type": type(module).__name__,
                    "loaded_from_disk": module.was_aot_compile_fn_loaded_from_disk,
                }
                for name, module in model.named_modules()
                if hasattr(module, "was_aot_compile_fn_loaded_from_disk")
            },
        }

    def onerec_v1_st_observe(self) -> None:
        from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeGraph
        from vllm_gr.v1.worker.gpu_beam_stage_runner import (
            AsyncGPUBeamOutput,
            GPUBeamStageRunner,
        )
        from vllm_gr.v1.worker.gpu_prefill_graph_runner import GPUPrefillGraphRunner

        runner = getattr(self, "model_runner")
        is_v2 = hasattr(runner, "req_states")
        assert is_v2 == (os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") == "1")
        if is_v2 and os.environ.get("VLLM_GR_TEST_STAGE_TRACE") == "1":
            raise ValueError("The optional V1 numerical trace does not support ModelRunner V2")
        assert not hasattr(self, "_onerec_events"), "observers already installed"
        events: list[dict[str, Any]] = []
        self._onerec_events = events
        trace = _StageTrace() if os.environ.get("VLLM_GR_TEST_STAGE_TRACE") == "1" else None
        self._onerec_trace = trace

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
                if trace is not None and name in ("execute", "sample"):
                    trace.stage(instance, name)
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
        if trace is not None:
            wrap(GPUBeamStageRunner, "sample", lambda owner, args, kwargs: {})
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
        wrap(BeamDecodeGraph, "capture", lambda graph, args, kwargs: {"kind": "decode"})
        wrap(
            GPUBeamStageRunner,
            "_forward",
            lambda owner, args, kwargs: {"kind": "decode_eager"},
        )
        if is_v2:
            _observe_v2_prefill(runner, events, wrap)
            return
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

        runner = getattr(self, "model_runner")
        original_forward = runner._model_forward

        def observe_forward(*args: Any, **kwargs: Any) -> Any:
            scheduled = getattr(runner, "_current_scheduler_output", None)
            stages = getattr(scheduled, "gr_stage_metadata", {})
            if scheduled is None or not stages or any(m.stage_index != 0 for m in stages.values()):
                return original_forward(*args, **kwargs)
            ids = kwargs.get("input_ids", args[0] if args else None)
            positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
            graph = getattr(runner, "_prefill_graph_runner", None)
            before = dict(getattr(runner, "_gr_prefill_graph_dispatches", {}))
            total = scheduled.total_num_scheduled_tokens
            event = dict(
                method="prefill_inputs",
                scheduled=total,
                request_tokens=list(scheduled.num_scheduled_tokens.values()),
                input_shape=None if ids is None else list(ids.shape),
                position_shape=None if positions is None else list(positions.shape),
                inputs_embeds=kwargs.get("inputs_embeds") is not None,
                intermediate_tensors=kwargs.get("intermediate_tensors") is not None,
                bucket=None if graph is None else graph.match_bucket(total),
            )
            events.append(event)
            if trace is not None:
                trace.prefill(runner, ids, positions, total)
            result = original_forward(*args, **kwargs)
            event["dispatches"] = {
                key: count - before.get(key, 0)
                for key, count in getattr(runner, "_gr_prefill_graph_dispatches", {}).items()
                if count != before.get(key, 0)
            }
            return result

        runner._model_forward = observe_forward

    def onerec_v1_st_events(self) -> list[dict[str, Any]]:
        return self._onerec_events

    def onerec_v1_st_save_trace(self, path: str) -> None:
        """Read back only after a complete request, never between stages."""
        if self._onerec_trace is not None:
            self._onerec_trace.save(path)

    def onerec_v1_st_model_identity(self) -> dict[str, Any]:
        """Hash loaded parameters after the paired request, not during startup."""
        import hashlib

        import torch

        model = getattr(self, "model_runner").get_model()
        return dict(
            parameters={
                name: hashlib.sha256(
                    value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                for name, value in model.named_parameters()
            },
            blas_backend=str(torch.backends.cuda.preferred_blas_library()),
            bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            bf16_split_k=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
            matmul_precision=torch.get_float32_matmul_precision(),
        )


def _observe_v2_prefill(runner: Any, events: list[dict[str, Any]], wrap: Any) -> None:
    """Observe native V2 inputs and completed FULL replays without GPU readback."""
    from vllm_gr.v1.worker.gpu_prefill_graph_runner_v2 import GRPrefillCudaGraphManager

    original_prepare = runner.prepare_inputs
    manager = runner.cudagraph_manager

    def prepare(scheduled: Any, request_state: Any, desc: Any) -> Any:
        batch = original_prepare(scheduled, request_state, desc)
        stages = getattr(scheduled, "gr_stage_metadata", {})
        if stages and all(m.stage_index == 0 for m in stages.values()):
            events.append(
                dict(
                    method="prefill_inputs",
                    runner="v2",
                    scheduled=scheduled.total_num_scheduled_tokens,
                    request_tokens=list(scheduled.num_scheduled_tokens.values()),
                    native_request_tokens=batch.num_scheduled_tokens.tolist(),
                    input_shape=list(batch.input_ids.shape),
                    position_shape=list(batch.positions.shape),
                    bucket=desc.num_tokens,
                    dispatches={desc.cg_mode.name: 1},
                )
            )
        return batch

    runner.prepare_inputs = prepare
    original_full = manager.run_fullgraph

    def full(desc: Any) -> Any:
        result = original_full(desc)
        events.append(
            dict(
                method="run_fullgraph", outcome="replay", bucket=desc.num_tokens, size=desc.num_reqs
            )
        )
        return result

    manager.run_fullgraph = full
    wrap(
        GRPrefillCudaGraphManager,
        "capture",
        lambda manager, args, kwargs: {"kind": "prefill"},
    )


class _StageTrace:
    """Opt-in GPU copies outside compiled forwards; still requires observer controls."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.sessions: list[str] = []

    def accepts(self, metadata: Any) -> bool:
        if len(metadata.request_ids) < 2:
            return False
        if metadata.session_id not in self.sessions:
            if len(self.sessions) >= 4:
                return False
            self.sessions.append(metadata.session_id)
        return True

    def append(self, metadata: Any, phase: str, tensors: dict[str, Any]) -> None:
        import torch

        if not self.accepts(metadata):
            return
        self.rows.append(
            dict(
                session_id=metadata.session_id,
                stage=metadata.stage_index,
                batch=len(metadata.request_ids),
                phase=phase,
                tensors={
                    name: value.detach().clone()
                    for name, value in tensors.items()
                    if isinstance(value, torch.Tensor)
                },
            )
        )

    def prefill(self, runner: Any, ids: Any, positions: Any, total: int) -> None:
        metadata = next(iter(runner._current_scheduler_output.gr_stage_metadata.values()))
        count = len(metadata.request_ids)
        self.append(
            metadata,
            "prefill_inputs",
            dict(
                input_ids=ids[:total],
                positions=positions[..., :total],
                query_start_loc=runner.query_start_loc.gpu[: count + 1],
                seq_lens=runner.seq_lens[:count],
                block_table=runner.input_batch.block_table[0].block_table.gpu[:count],
                slot_mapping=runner.input_batch.block_table[0].slot_mapping.gpu[:total],
            ),
        )

    def stage(self, owner: Any, phase: str) -> None:
        metadata = owner.previous
        if not self.accepts(metadata):
            return
        count = len(owner.bindings)
        rows = count if metadata.stage_index == 0 else count * owner.beam.state.beam_width
        if phase == "execute":
            tensors = dict(
                logits=owner.logits,
                prefixes=owner.prefixes[:rows],
                constraint_steps=owner.constraint_steps,
                temperatures=owner.temperatures[:rows],
            )
            if metadata.stage_index == 0:
                native = owner.worker.execute_model_state
                tensors["sample_hidden_states"] = native.sample_hidden_states
                if os.environ.get("VLLM_GR_TEST_PREFILL_KV_TRACE") == "1":
                    tensors.update(self.prefill_result(owner.worker, native.hidden_states))
            if metadata.stage_index > 0:
                tensors.update(
                    input_ids=owner.input_ids,
                    positions=owner.positions,
                    execution_mask=owner.execution_mask,
                )
                tensors.update({f"attention.{k}": v for k, v in vars(owner.metadata).items()})
        else:
            tensors = dict(
                candidate_tokens=owner.tokens[:rows],
                candidate_scores=owner.scores[:rows],
                candidate_valid=owner.valid[:rows],
                candidate_counts=owner.counts[:rows],
            )
            state = owner.beam.state
            for name in (
                "sequence",
                "beam_scores",
                "active_mask",
                "num_generated_tokens",
                "history_token_ids",
                "history_parent_ids",
                "num_completed",
                "completed_token_ids",
                "completed_scores",
                "completed_parent_ids",
            ):
                value = getattr(state, name, None)
                if value is not None:
                    tensors[f"state.{name}"] = value[:count]
        self.append(metadata, phase, tensors)

    def prefill_result(self, runner: Any, result: Any) -> dict[str, Any]:
        import torch

        scheduled = runner._current_scheduler_output
        tensors = {"hidden": result}
        if os.environ.get("VLLM_GR_TEST_PREFILL_KV_TRACE") == "1":
            # Completion pairing resets prefix cache before each request.
            # Gather only real prompt tokens, never padding or unused pages.
            counts = list(scheduled.num_scheduled_tokens.values())
            block_size = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            blocks = runner.input_batch.block_table[0].block_table.gpu
            for layer, cache in enumerate(runner.kv_caches):
                view = cache.transpose(1, 2)
                tensors[f"kv.{layer}"] = torch.cat(
                    [
                        view.index_select(
                            0, blocks[i, : (n + block_size - 1) // block_size].long()
                        ).flatten(0, 1)[:n]
                        for i, n in enumerate(counts)
                    ]
                )
        return tensors

    def save(self, path: str) -> None:
        import torch

        torch.save(
            [
                row | {"tensors": {k: v.cpu() for k, v in row["tensors"].items()}}
                for row in self.rows
            ],
            path,
        )
        self.rows.clear()


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
        "runner": f"{type(runner).__module__}.{type(runner).__name__}",
        "native_requests": runner.req_states.num_reqs if hasattr(runner, "req_states") else None,
        "pending_native_output": getattr(runner, "execute_model_state", None) is not None,
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
        "constraint_backend": type(runner.constraint_backend).__name__
        if runner.constraint_backend is not None
        else None,
        "unconstrained_topk": owner.unconstrained_topk_scratch is not None,
        "state_address": state.sequence.data_ptr(),
        "completed": {
            name: getattr(state, "completed_" + name)[row, :count].cpu().tolist() for name in names
        },
        "history_tokens": state.history_token_ids[row].cpu().tolist(),
        "history_parents": state.history_parent_ids[row].cpu().tolist(),
    }


def _check_v2_idle(snapshot: dict[str, Any]) -> None:
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "1":
        return
    assert snapshot["retired"]
    assert snapshot["native_requests"] == 0 and not snapshot["pending_native_output"], snapshot
    assert not snapshot["active_sessions"] and snapshot["registry_size"] == 0
    assert snapshot["context_released"] and snapshot["workspace_released"]
    assert not any(snapshot["control_busy"]) and not snapshot["pending"]


def _check_unconstrained_mode(snapshot: dict[str, Any]) -> None:
    if not _UNCONSTRAINED:
        return
    assert snapshot["constraint_backend"] is None
    assert snapshot["unconstrained_topk"]


def _check_v2_graph_events(events: list[dict[str, Any]]) -> None:
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "1":
        return
    assert not any(e["method"] == "capture" for e in events), "request-time graph capture"
    if not _EAGER:
        assert not any(e["method"] == "_forward" for e in events), "Decode eager fallback"
        assert any(e["method"] == "replay" for e in events)


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
    if artifact.exists():
        # Reuse a historical artifact without rewriting it, but require the
        # exact same deterministic SID table as this test's configuration.
        with tempfile.TemporaryDirectory(dir=work_dir) as temporary:
            expected = Path(temporary) / "expected.safetensors"
            digest = save_constraint_table_artifact(table, expected)
            assert artifact.read_bytes() == expected.read_bytes(), "ST artifact differs"
    else:
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
    _check_v2_idle(snapshot)
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
    with GRLLM(vllm_gr_config=config, shutdown_timeout=10, **_engine_kwargs(model, config)) as llm:
        expected_client = (
            "InprocClient"
            if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
            else "SyncMPClient"
        )
        assert type(llm.llm_engine.engine_core).__name__ == expected_client
        processes = (
            list(llm.llm_engine.engine_core.resources.engine_manager.processes)
            if expected_client == "SyncMPClient"
            else []
        )
        llm.collective_rpc("onerec_v1_st_observe")
        state_address = None
        for length, steps, cold in _CASES:
            if cold:
                assert llm.reset_prefix_cache()
            ids = (newline * length + source)[-(length - 1) :]
            public = llm.beam_search_v1([{"prompt_token_ids": ids}], _params(steps))[0]
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
    shutdown = [dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive()) for p in processes]
    (work_dir / "offline-lifecycle.json").write_text(json.dumps(shutdown, indent=2))
    assert all(not p["alive"] and p["exitcode"] == 0 for p in shutdown), shutdown


async def _run_online(work_dir: Path) -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.generate.base.serving import GenerateBaseServing
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args,
    )
    from vllm.entrypoints.openai.models.serving import (
        BaseModelPath,
        OpenAIServingModels,
    )
    from vllm.inputs import tokens_input

    from vllm_gr.plugin import initialize_runtime

    initialize_runtime()
    model, tokenizer, config, source, newline, legal = _inputs(work_dir)
    # The native 0.30 default is zero: it kills EngineCore before IPC cleanup.
    # Give the real shutdown path time to run and check its exit below.
    args = AsyncEngineArgs(
        additional_config={"vllm_gr": config},
        shutdown_timeout=10,
        **_engine_kwargs(model, config),
    )
    async with build_async_engine_client_from_engine_args(args) as engine:
        processes = list(engine.engine_core.resources.engine_manager.processes)
        serving = GenerateBaseServing(
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
        cancellations = []

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
                deadline = time.monotonic() + 30
                while not engine.output_processor.request_states:
                    assert not task.done() and time.monotonic() < deadline
                    await asyncio.sleep(0.001)
                (session_id,) = engine.output_processor.request_states
                if waiting:
                    # A sleep alone can pass before the request was submitted.
                    # Fence actual EngineCore admission while scheduling is paused.
                    while await engine.engine_core.call_utility_async(
                        "is_gr_session_retired", session_id
                    ):
                        assert not task.done() and time.monotonic() < deadline
                        await asyncio.sleep(0.001)
                    assert len((await engine.collective_rpc("onerec_v1_st_events"))[0]) == before
                else:
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
            assert await engine.engine_core.call_utility_async("is_gr_session_retired", session_id)
            assert not engine.output_processor.has_unfinished_requests()
            assert not engine.output_processor.external_req_ids
            snapshot = (await engine.collective_rpc("onerec_v1_st_snapshot"))[0]
            assert snapshot["retired"] and not snapshot["active_sessions"]
            assert snapshot["registry_size"] == 0
            assert snapshot["context_released"] and snapshot["workspace_released"]
            assert not any(snapshot["control_busy"]) and not snapshot["pending"]
            events = (await engine.collective_rpc("onerec_v1_st_events"))[0][before:]
            cancellations.append(
                dict(
                    waiting=waiting,
                    session_id=session_id,
                    events=events,
                    core_retired=True,
                    frontend_empty=True,
                    worker_released=True,
                )
            )
            await run(1024, 3, False, "after-cancel")

        await cancel(waiting=True)
        await cancel(waiting=False)
        _write_evidence(
            work_dir, "online", rows, (await engine.collective_rpc("onerec_v1_st_events"))[0]
        )
    shutdown = [dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive()) for p in processes]
    assert shutdown and all(not p["alive"] and p["exitcode"] == 0 for p in shutdown), shutdown
    (work_dir / "online-lifecycle.json").write_text(
        json.dumps(dict(cancellations=cancellations, shutdown=shutdown), indent=2)
    )


@asynccontextmanager
async def _completion_http_client(app: Any) -> AsyncIterator[Any]:
    """Use a real loopback socket so disconnects reach the native HTTP route."""
    import httpx
    import uvicorn

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=30)
        )
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(30):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("HTTP server exited before startup")
                    await asyncio.sleep(0.01)
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{sock.getsockname()[1]}",
                timeout=60,
                trust_env=False,
            ) as client:
                yield client
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=35)


def _completion_inputs(work_dir: Path) -> tuple[Any, ...]:
    model, tokenizer, config, source, newline, legal = _inputs(work_dir)
    if _UNCONSTRAINED:
        config["constraint_table"] = {
            "enabled": False,
            "max_top_k": config["beam"]["max_width"],
        }
        config["constrained_generation"] = False
    configured_sizes = os.environ.get("VLLM_GR_TEST_ONLINE_BATCH_SIZES")
    sizes = (1, 2, 3, 4, 8) if configured_sizes is None else tuple(json.loads(configured_sizes))
    config["beam"]["online_batch_sizes"] = list(sizes)
    kwargs = _engine_kwargs(model, config) | dict(
        max_num_seqs=max(sizes),
        max_num_batched_tokens=4096,
        # Automatic memory profiling differs between cold offline compilation
        # and an online engine loading that cache. Pair the actual KV capacity.
        kv_cache_memory_bytes=8 * 1024**3,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 16, *range(128, 4097, 128)]},
    )
    if os.environ.get("VLLM_GR_TEST_PROFILE") == "1":
        kwargs["profiler_config"] = dict(
            profiler="torch",
            torch_profiler_dir=str(work_dir / "profiles"),
            torch_profiler_with_stack=False,
            ignore_frontend=True,
        )
    if os.environ.get("VLLM_GR_TEST_INDUCTOR_CONFIG"):
        kwargs["compilation_config"]["inductor_compile_config"] = json.loads(
            os.environ["VLLM_GR_TEST_INDUCTOR_CONFIG"]
        )
    prompts = [
        tokenizer.encode(f"slot {i}", add_special_tokens=False) + newline * (i * 16) + source[-256:]
        for i in range(max(sizes))
    ]
    return model, config, sizes, kwargs, prompts, legal


def _check_completion_prefill(events: list[dict[str, Any]], batch_size: int) -> None:
    from vllm_gr.v1.worker.prefill_graph_common import is_prefill_graph_enabled

    if _EAGER or not is_prefill_graph_enabled():
        return
    prefills = [e for e in events if e["method"] == "prefill_inputs"]
    assert len(prefills) == 1
    event = prefills[0]
    assert len(event["request_tokens"]) == batch_size
    assert event["input_shape"] == event["position_shape"]
    assert event["input_shape"][0] >= event["scheduled"]
    if event.get("runner") == "v2":
        assert event["dispatches"] == {"FULL": 1}, event
        assert sorted(event["native_request_tokens"]) == sorted(event["request_tokens"])
        full = [e for e in events if e["method"] == "run_fullgraph"]
        assert len(full) == 1 and full[0]["size"] == batch_size, full
        assert full[0]["bucket"] == event["bucket"]
    else:
        assert event["dispatches"] == {f"replay:{batch_size}": 1}, event
    assert event["bucket"] is not None and event["bucket"] >= event["scheduled"], event


def _run_completion_reference(work_dir: Path) -> None:
    """Run the HTTP inputs in a separate offline engine before starting HTTP."""
    from vllm_gr.entrypoints.gr import GRLLM

    model, config, sizes, kwargs, prompts, _ = _completion_inputs(work_dir)
    rows = []
    with GRLLM(vllm_gr_config=config, shutdown_timeout=10, **kwargs) as llm:
        processes = list(llm.llm_engine.engine_core.resources.engine_manager.processes)
        (work_dir / "completion-runtime-offline.json").write_text(
            json.dumps(llm.collective_rpc("onerec_v1_st_runtime"), indent=2)
        )
        llm.collective_rpc("onerec_v1_st_observe")
        for count in sizes:
            assert llm.reset_prefix_cache()
            before = len(llm.collective_rpc("onerec_v1_st_events")[0])
            if count == 2 and os.environ.get("VLLM_GR_TEST_PROFILE") == "1":
                llm.start_profile("offline-b2")
            outputs = llm.beam_search_v1(
                [{"prompt_token_ids": p} for p in prompts[:count]], _params(3)
            )
            if count == 2 and os.environ.get("VLLM_GR_TEST_PROFILE") == "1":
                llm.stop_profile()
            if count == 2 and os.environ.get("VLLM_GR_TEST_MODEL_IDENTITY") == "1":
                (work_dir / "model-identity-offline.json").write_text(
                    json.dumps(llm.collective_rpc("onerec_v1_st_model_identity"), indent=2)
                )
            if os.environ.get("VLLM_GR_TEST_STAGE_TRACE") == "1":
                llm.collective_rpc(
                    "onerec_v1_st_save_trace", args=(str(work_dir / f"trace-offline-{count}.pt"),)
                )
            _check_completion_prefill(llm.collective_rpc("onerec_v1_st_events")[0][before:], count)
            assert len(outputs) == count
            for prompt, output in zip(prompts[:count], outputs, strict=True):
                assert all(s.tokens[: len(prompt)] == prompt for s in output.sequences)
            rows.append(
                dict(
                    count=count,
                    candidates=[
                        [
                            dict(tokens=s.tokens[len(prompt) :], score=s.cum_logprob)
                            for s in output.sequences
                        ]
                        for prompt, output in zip(prompts[:count], outputs, strict=True)
                    ],
                )
            )
            assert not llm.llm_engine.has_unfinished_requests()
            snapshot = llm.collective_rpc("onerec_v1_st_snapshot")[0]
            _check_v2_idle(snapshot)
            _check_unconstrained_mode(snapshot)
        graphs = llm.collective_rpc("onerec_v1_st_graphs")
        events = llm.collective_rpc("onerec_v1_st_events")[0]
        _check_v2_graph_events(events)
    shutdown = [dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive()) for p in processes]
    assert shutdown and all(not p["alive"] and p["exitcode"] == 0 for p in shutdown), shutdown
    (work_dir / "completion-offline-reference.json").write_text(
        json.dumps(
            dict(
                model=model,
                config=config,
                engine_kwargs=kwargs,
                prompts=prompts,
                rows=rows,
                graphs=graphs,
                events=events,
                shutdown=shutdown,
            ),
            indent=2,
        )
    )


async def _run_online_completion_batch(work_dir: Path) -> None:
    """Drive a prompt list through the completions HTTP route.

    The route is the whole point: the same list posted here used to become one
    session per prompt. The assertions are on what the engine was actually
    asked to run, read off the existing stage probe, not on the response alone
    -- a single response object is what both shapes return.
    """
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
    from vllm.inputs import tokens_input
    from vllm.renderers.online_renderer import OnlineRenderer

    from vllm_gr.plugin import initialize_runtime

    with _measure("runtime_init_seconds"):
        initialize_runtime()
    with _measure("inputs_seconds"):
        model, config, sizes, kwargs, prompts, legal = _completion_inputs(work_dir)
    _MEASUREMENTS["batch_sizes"] = sizes
    _persist_measurements()
    offline = json.loads((work_dir / "completion-offline-reference.json").read_text())
    assert (offline["model"], offline["config"], offline["engine_kwargs"], offline["prompts"]) == (
        model,
        config,
        kwargs,
        prompts,
    )
    args = AsyncEngineArgs(additional_config={"vllm_gr": config}, shutdown_timeout=10, **kwargs)
    async with _measured_engine(build_async_engine_client_from_engine_args, args) as engine:
        processes = list(engine.engine_core.resources.engine_manager.processes)
        (work_dir / "completion-runtime-online.json").write_text(
            json.dumps(await engine.collective_rpc("onerec_v1_st_runtime"), indent=2)
        )
        models = OpenAIServingModels(
            engine_client=engine,
            base_model_paths=[BaseModelPath(name=model, model_path=model)],
            lora_modules=None,
        )
        serving = OpenAIServingCompletion(
            engine_client=engine,
            models=models,
            online_renderer=OnlineRenderer(
                model_config=engine.model_config,
                renderer=engine.renderer,
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

        def check_response(response: Any, inputs: list[list[int]]) -> dict[str, Any]:
            assert response.status_code == 200, response.text
            data = response.json()
            assert len(data["choices"]) == len(inputs) * _BEAM_WIDTH
            assert [c["index"] for c in data["choices"]] == list(range(len(inputs) * _BEAM_WIDTH))
            assert data["usage"]["prompt_tokens"] == sum(map(len, inputs))
            assert data["usage"]["completion_tokens"] == len(inputs) * _BEAM_WIDTH * 5
            for choice in data["choices"]:
                assert choice["finish_reason"] == "length"
                assert len(choice["token_ids"]) == 5
                if _UNCONSTRAINED:
                    vocab_size = engine.model_config.get_vocab_size()
                    assert all(0 <= token < vocab_size for token in choice["token_ids"][1:-1])
                else:
                    assert tuple(choice["token_ids"][1:-1]) in legal
                if choice["index"] % _BEAM_WIDTH == 0:
                    assert choice["prompt_token_ids"] == inputs[choice["index"] // _BEAM_WIDTH]
            return data

        async def check_released() -> None:
            assert not engine.output_processor.has_unfinished_requests()
            assert not engine.output_processor.external_req_ids
            snapshot = (await engine.collective_rpc("onerec_v1_st_snapshot"))[0]
            assert snapshot["retired"] and not snapshot["active_sessions"]
            assert snapshot["registry_size"] == 0
            assert snapshot["context_released"] and snapshot["workspace_released"]
            assert not any(snapshot["control_busy"]) and not snapshot["pending"]
            _check_v2_idle(snapshot)
            _check_unconstrained_mode(snapshot)

        rows = _MEASUREMENTS["rows"]
        cancellations = []
        request_started = time.monotonic()
        _MEASUREMENTS["active_phase"] = "request_seconds"
        _persist_measurements()
        async with _completion_http_client(app) as client:
            # A size this route forms runs as one session holding every prompt.
            for count in sizes:
                assert await engine.reset_prefix_cache()
                before = len((await engine.collective_rpc("onerec_v1_st_events"))[0])
                if count == 2 and os.environ.get("VLLM_GR_TEST_PROFILE") == "1":
                    await engine.start_profile("online-b2")
                response = await client.post(
                    "/v1/completions", json={**body, "prompt": prompts[:count]}
                )
                data = check_response(response, prompts[:count])
                if count == 2 and os.environ.get("VLLM_GR_TEST_PROFILE") == "1":
                    await engine.stop_profile()
                if count == 2 and os.environ.get("VLLM_GR_TEST_MODEL_IDENTITY") == "1":
                    (work_dir / "model-identity-online.json").write_text(
                        json.dumps(
                            await engine.collective_rpc("onerec_v1_st_model_identity"), indent=2
                        )
                    )
                if os.environ.get("VLLM_GR_TEST_STAGE_TRACE") == "1":
                    await engine.collective_rpc(
                        "onerec_v1_st_save_trace", args=(str(work_dir / f"trace-http-{count}.pt"),)
                    )
                stages = await stages_since(before)
                assert [e["size"] for e in stages if e["stage"] == 0] == [count]
                assert len({e["session_id"] for e in stages}) == 1
                _check_completion_prefill(
                    (await engine.collective_rpc("onerec_v1_st_events"))[0][before:], count
                )
                snapshot = (await engine.collective_rpc("onerec_v1_st_snapshot"))[0]
                assert snapshot["retired"] and snapshot["registry_size"] == 0
                assert not snapshot["active_sessions"]
                assert not engine.output_processor.has_unfinished_requests()
                assert await engine.reset_prefix_cache()
                before_direct = len((await engine.collective_rpc("onerec_v1_st_events"))[0])
                direct = [
                    output
                    async for output in serving.beam_search_v1(
                        [tokens_input(p) for p in prompts[:count]], "direct-reference", _params(3)
                    )
                ]
                _check_completion_prefill(
                    (await engine.collective_rpc("onerec_v1_st_events"))[0][before_direct:], count
                )
                assert [out.prompt_token_ids for out in direct] == prompts[:count]
                reference = next(row for row in offline["rows"] if row["count"] == count)
                direct_candidates = [
                    [
                        dict(tokens=list(c.token_ids), score=c.cumulative_logprob)
                        for c in output.outputs
                    ]
                    for output in direct
                ]
                if os.environ.get("VLLM_GR_TEST_STAGE_TRACE") == "1":
                    await engine.collective_rpc(
                        "onerec_v1_st_save_trace",
                        args=(str(work_dir / f"trace-online-{count}.pt"),),
                    )
                (work_dir / f"completion-pair-{count}.json").write_text(
                    json.dumps(
                        dict(
                            count=count,
                            offline=reference["candidates"],
                            online=direct_candidates,
                            response=data,
                            events=(await engine.collective_rpc("onerec_v1_st_events"))[0][before:],
                            graphs=await engine.collective_rpc("onerec_v1_st_graphs"),
                        ),
                        indent=2,
                    )
                )
                assert reference["candidates"] == direct_candidates
                assert [c["token_ids"] for c in data["choices"]] == [
                    list(c.token_ids) for output in direct for c in output.outputs
                ]
                await check_released()
                rows.append(dict(count=count, sessions=1, size=count, response=data))
                _persist_measurements()

            # The native rejection path must leave the engine usable.
            before = len((await engine.collective_rpc("onerec_v1_st_events"))[0])
            invalid = await client.post(
                "/v1/completions", json={**body, "prompt": prompts[:2], "stream": True}
            )
            assert invalid.status_code == 400, invalid.text
            assert not await stages_since(before)
            await check_released()

            references = []
            for prompt in prompts[:2]:
                assert await engine.reset_prefix_cache()
                response = await client.post("/v1/completions", json={**body, "prompt": prompt})
                references.append(check_response(response, [prompt]))
            assert await engine.reset_prefix_cache()
            await engine.pause_generation(mode="keep", clear_cache=False)
            before = len((await engine.collective_rpc("onerec_v1_st_events"))[0])
            pending = [
                asyncio.create_task(
                    client.post(
                        "/v1/completions",
                        json={**body, "prompt": p, "request_id": f"concurrent-{i}"},
                    )
                )
                for i, p in enumerate(prompts[:2])
            ]
            try:
                async with asyncio.timeout(30):
                    while len(engine.output_processor.request_states) != 2:
                        assert not any(task.done() for task in pending)
                        await asyncio.sleep(0.001)
                    sessions = list(engine.output_processor.request_states)
                    for session in sessions:
                        while await engine.engine_core.call_utility_async(
                            "is_gr_session_retired", session
                        ):
                            await asyncio.sleep(0.001)
            finally:
                await engine.resume_generation()
                responses = await asyncio.gather(*pending)
            for i, response in enumerate(responses):
                data = check_response(response, [prompts[i]])
                assert data["id"] == f"cmpl-concurrent-{i}"
                assert [c["token_ids"] for c in data["choices"]] == [
                    c["token_ids"] for c in references[i]["choices"]
                ]
            stages = await stages_since(before)
            assert {e["session_id"] for e in stages} == set(sessions)
            assert [e["size"] for e in stages if e["stage"] == 0] == [1, 1]
            await check_released()

            # Close the TCP connection only after confirmed engine admission
            # (and, for the active case, an actual Worker dispatch).
            for waiting in (True, False):
                assert await engine.reset_prefix_cache()
                if waiting:
                    await engine.pause_generation(mode="keep", clear_cache=False)
                before = len((await engine.collective_rpc("onerec_v1_st_events"))[0])
                payload = json.dumps({**body, "prompt": prompts[: sizes[1]]}).encode()
                reader, writer = await asyncio.open_connection(
                    client.base_url.host, client.base_url.port
                )
                received = asyncio.create_task(reader.read(1))
                try:
                    writer.write(
                        b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                        b"Content-Type: application/json\r\nConnection: close\r\n"
                        + f"Content-Length: {len(payload)}\r\n\r\n".encode()
                        + payload
                    )
                    await writer.drain()
                    async with asyncio.timeout(30):
                        while not engine.output_processor.request_states:
                            assert not received.done(), "request completed before admission probe"
                            await asyncio.sleep(0.001)
                        (session,) = engine.output_processor.request_states
                        while await engine.engine_core.call_utility_async(
                            "is_gr_session_retired", session
                        ):
                            assert not received.done()
                            await asyncio.sleep(0.001)
                        if waiting:
                            assert not await stages_since(before)
                        else:
                            while not await stages_since(before):
                                assert not received.done(), "missed active cancellation"
                                await asyncio.sleep(0.001)
                        assert session in engine.output_processor.request_states
                        assert not received.done(), "disconnect must precede the response"
                        writer.close()
                        await writer.wait_closed()
                        while engine.output_processor.has_unfinished_requests() or not (
                            await engine.engine_core.call_utility_async(
                                "is_gr_session_retired", session
                            )
                        ):
                            await asyncio.sleep(0.001)
                    await check_released()
                    cancellations.append(
                        dict(
                            waiting=waiting,
                            session_id=session,
                            disconnected_before_response=True,
                            stages=await stages_since(before),
                            core_retired=True,
                            frontend_empty=True,
                            worker_released=True,
                        )
                    )
                finally:
                    writer.close()
                    await writer.wait_closed()
                    received.cancel()
                    await asyncio.gather(received, return_exceptions=True)
                    if waiting:
                        await engine.resume_generation()
                response = await client.post(
                    "/v1/completions", json={**body, "prompt": prompts[: sizes[1]]}
                )
                check_response(response, prompts[: sizes[1]])
                await check_released()
            (work_dir / "completion-http-lifecycle.json").write_text(
                json.dumps(
                    dict(
                        transport="loopback TCP",
                        concurrent_sessions=sessions,
                        invalid_status=invalid.status_code,
                        cancellations=cancellations,
                        recoveries=2,
                    ),
                    indent=2,
                )
            )
            _check_v2_graph_events((await engine.collective_rpc("onerec_v1_st_events"))[0])
            (work_dir / "completion-events.json").write_text(
                json.dumps((await engine.collective_rpc("onerec_v1_st_events"))[0], indent=2)
            )
            (work_dir / "completion-online-graphs.json").write_text(
                json.dumps(await engine.collective_rpc("onerec_v1_st_graphs"), indent=2)
            )
        _MEASUREMENTS["phases"]["request_seconds"] = time.monotonic() - request_started
        _MEASUREMENTS["events"] = (await engine.collective_rpc("onerec_v1_st_events"))[0]
        _persist_measurements()
        (work_dir / "completion-batch-evidence.json").write_text(json.dumps(rows, indent=2))
    shutdown = [dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive()) for p in processes]
    (work_dir / "completion-batch-lifecycle.json").write_text(json.dumps(shutdown, indent=2))
    assert shutdown and all(not p["alive"] and p["exitcode"] == 0 for p in shutdown), shutdown


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("runner", ["v1", "v2"])
@pytest.mark.parametrize(
    ("batch_sizes", "unconstrained"),
    [(None, False), ((1, 3, 7), False), (None, True)],
    ids=["default", "custom", "unconstrained"],
)
def test_onerec_v1_completion_batch_online(
    tmp_path: Path,
    batch_sizes: tuple[int, ...] | None,
    unconstrained: bool,
    runner: str,
) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("OneRec completion batching ST requires CUDA")
    env = dict(os.environ)
    env.pop("VLLM_GR_TEST_ONLINE_BATCH_SIZES", None)
    env.pop("VLLM_GR_TEST_UNCONSTRAINED", None)
    if unconstrained:
        env["VLLM_GR_TEST_UNCONSTRAINED"] = "1"
    if batch_sizes is not None:
        env["VLLM_GR_TEST_ONLINE_BATCH_SIZES"] = json.dumps(batch_sizes)
    # The default eight-request and custom seven-request prompt lists exceed
    # max_model_len in total. This ST requires graph replay for every batch, so
    # its explicit grid covers the scheduler budget; the production default
    # remains bounded by max_model_len.
    env["VLLM_GR_PREFILL_GRAPH_BUCKETS"] = ",".join(str(n) for n in range(128, 4097, 128))
    env.update(
        USE_TF="0",
        VLLM_USE_V2_MODEL_RUNNER=str(int(runner == "v2")),
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    for mode in ("--completion-reference", "--completion-batch"):
        process_started = time.monotonic()
        command = [
            sys.executable,
            "-m",
            "tests.st.test_onerec_v1_scheduler_st",
            str(tmp_path),
            mode,
        ]
        with subprocess.Popen(
            command, cwd=Path(__file__).parents[2], env=env, start_new_session=True
        ) as process:
            try:
                assert process.wait(timeout=550) == 0, f"OneRec {mode} subprocess failed"
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                (tmp_path / f"subprocess-{mode.removeprefix('--')}.json").write_text(
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
@pytest.mark.parametrize("runner", ["v1", "v2"])
@pytest.mark.parametrize("online", [False, True])
def test_onerec_v1_async_scheduling(tmp_path: Path, online: bool, runner: str) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("OneRec V1 scheduling ST requires CUDA")
    env = dict(os.environ)
    env.update(
        USE_TF="0",
        VLLM_USE_V2_MODEL_RUNNER=str(int(runner == "v2")),
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
        "completion-reference"
        if "--completion-reference" in sys.argv
        else (
            "completion-batch"
            if "--completion-batch" in sys.argv
            else ("online" if "--online" in sys.argv else "offline")
        )
    )
    _EVIDENCE_PATH = Path(sys.argv[1]) / f"{mode}-evidence.json"
    _MEASUREMENTS.update(mode=mode, status="running")
    _persist_measurements()
    try:
        if mode == "completion-reference":
            _run_completion_reference(Path(sys.argv[1]))
        elif mode == "completion-batch":
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
