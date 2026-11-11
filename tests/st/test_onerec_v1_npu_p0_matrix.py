# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""B=1/4, effective L=128, W=128 constrained NPU eager/FULL regression.

This is a numerical and graph-replay test, not a performance benchmark. The
existing B=2/W=8 switch/fallback test and its default trace bounds are retained.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.st.npu_beam_diagnostics import NPUForwardTrace


def _npu_test_unavailable(reason: str) -> None:
    if os.environ.get("VLLM_GR_TARGET_DEVICE") == "npu":
        pytest.fail(f"required NPU P0 runtime is unavailable: {reason}", pytrace=False)
    pytest.skip(reason, allow_module_level=True)


try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    _npu_test_unavailable(f"torch-npu is not installed: {exc}")

if not hasattr(torch, "npu") or not torch.npu.is_available():
    _npu_test_unavailable("Ascend NPU is unavailable")

from vllm_gr.ops.npu import beam_ops  # noqa: E402

if not beam_ops.is_available():
    _npu_test_unavailable(f"xllm Beam operators are unavailable: {beam_ops.unavailable_reason()}")

from tests.st.test_onerec_v1_npu_e2e import (  # noqa: E402
    _assert_batch_equal,
    _build_constraint_artifact,
    _graph_context_signature,
    _prompt_tokens,
)

pytestmark = [pytest.mark.npu_test, pytest.mark.model_test, pytest.mark.slow, pytest.mark.l4]

_BEAM_WIDTH = 128
_INPUT_LEN = 128
_STEPS = 3
_BEGIN_TOKEN = "<|sid_begin|>"
_END_TOKEN = "<|sid_end|>"


def _matrix_environment(inherited: Mapping[str, str]) -> dict[str, str]:
    env = dict(inherited)
    for name in (
        "VLLM_GR_PREFILL_GRAPH_KV_BOUND",
        "VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS",
        "VLLM_GR_TEST_PADDED_PREFILL",
        "VLLM_GR_NPU_STAGE_LM_HEAD",
    ):
        env.pop(name, None)
    env.update(
        USE_TF="0",
        VLLM_GR_TARGET_DEVICE="npu",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        TORCH_MULTIPROCESSING_METHOD="spawn",
        VLLM_ASCEND_ENABLE_NZ="0",
        VLLM_USE_BREAKABLE_CUDAGRAPH="0",
        VLLM_ENABLE_PREFILL_CUDAGRAPH="1",
        VLLM_GR_PREFILL_GRAPH_BUCKETS="128,512",
        VLLM_GR_PREFILL_GRAPH_BUCKET_STEP="128",
        VLLM_GR_PREFILL_GRAPH_MAX_BUCKET="512",
        VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS="",
        VLLM_GR_NPU_FUSED_FFN="0",
    )
    return env


class OneRecNPUP0WorkerExtension:
    """Read bounded runner metadata outside captured model execution."""

    _matrix_trace: NPUForwardTrace | None = None

    def onerec_p0_trace_start(self, batch_size: int) -> None:
        from vllm.forward_context import get_forward_context

        if self._matrix_trace is not None:
            raise RuntimeError("an NPU P0 trace is already active")
        self._matrix_trace = NPUForwardTrace(
            getattr(self, "model_runner"),
            get_forward_context,
            expected_batch_size=batch_size,
            record_cold=True,
        )

    def onerec_p0_trace_stop(self) -> list[dict[str, Any]]:
        if self._matrix_trace is None:
            raise RuntimeError("no NPU P0 trace is active")
        try:
            return self._matrix_trace.finish()
        finally:
            self._matrix_trace = None

    def onerec_p0_context_signature(self) -> tuple[int, tuple[int, ...]]:
        return _graph_context_signature(self)


def _assert_matrix_trace(
    events: list[dict[str, Any]], prompts: list[list[int]], begin: int, *, graph: bool
) -> None:
    batch_size = len(prompts)
    total = batch_size * _INPUT_LEN
    assert [event["stage"] for event in events] == [0, 1, 2]
    assert [event["kind"] for event in events] == ["prefill", "decode", "decode"]
    prefill = events[0]
    assert prefill["actual_tokens"] == prefill["padded_tokens"] == total
    assert prefill["input_ids_shape"] == prefill["positions_shape"] == [total]
    assert prefill["input_ids"] == [token for prompt in prompts for token in [*prompt, begin]]
    assert prefill["positions"] == list(range(_INPUT_LEN)) * batch_size
    assert len(set(prefill["input_request_ids"])) == batch_size
    assert prefill["attention"]["query_start_loc"] == [
        row * _INPUT_LEN for row in range(batch_size + 1)
    ]
    assert prefill["attention"]["seq_lens"] == [_INPUT_LEN] * batch_size
    assert prefill["cold_rows"] == [
        dict(num_computed_tokens=0, prompt_tokens=_INPUT_LEN, scheduled_tokens=_INPUT_LEN)
        for _ in prompts
    ]
    assert prefill["scheduled_total"] == total
    assert prefill["gr_prefill_replay_delta"] == ({str(total): 1} if graph else {})
    if graph:
        assert prefill["skip_compiled"] is False
    for event in events:
        assert event["enforce_eager"] is (not graph)
        assert event["compilation_mode"] == ("VLLM_COMPILE" if graph else "NONE")
        if not graph:
            assert event["runtime_mode"] == "NONE"
    for event in events[1:]:
        assert event["actual_tokens"] == event["padded_tokens"] == batch_size * _BEAM_WIDTH
        assert event["gr_prefill_replay_delta"] == {}
        if graph:
            assert event["runtime_mode"] == "FULL"


def _assert_outputs(
    outputs: list[Any],
    prompts: list[list[int]],
    legal: set[tuple[int, int, int]],
    begin: int,
    end: int,
) -> None:
    assert len(outputs) == len(prompts)
    for prompt, output in zip(prompts, outputs, strict=True):
        assert len(output.sequences) == _BEAM_WIDTH
        generated = []
        scores = []
        for sequence in output.sequences:
            tokens = [int(token) for token in sequence.tokens]
            assert tokens[: len(prompt)] == prompt
            suffix = tokens[len(prompt) :]
            assert len(suffix) == _STEPS + 2 and suffix[0] == begin and suffix[-1] == end
            semantic = tuple(suffix[1:-1])
            assert semantic in legal
            assert sequence.finish_reason == "length" and sequence.stop_reason is None
            assert math.isfinite(sequence.cum_logprob)
            generated.append(semantic)
            scores.append(float(sequence.cum_logprob))
        assert len(set(generated)) == _BEAM_WIDTH
        assert all(left >= right for left, right in zip(scores, scores[1:]))


def _run_case(
    llm: Any,
    prompts: list[list[int]],
    legal: set[tuple[int, int, int]],
    begin: int,
    end: int,
    *,
    graph: bool,
    label: str,
) -> tuple[list[Any], dict[str, Any]]:
    from vllm.inputs import tokens_input

    from vllm_gr.sampling_params import BeamSearchParams

    before = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
    llm.collective_rpc("onerec_p0_trace_start", args=(len(prompts),))
    try:
        outputs = llm.beam_search_v1(
            [
                tokens_input(prompt, cache_salt=f"{label}-row-{row}")
                for row, prompt in enumerate(prompts)
            ],
            BeamSearchParams(
                beam_width=_BEAM_WIDTH,
                max_tokens=_STEPS + 2,
                temperature=0.0,
                ignore_eos=True,
                begin_token=_BEGIN_TOKEN,
                end_token=_END_TOKEN,
                length_penalty=1.0,
            ),
            output_mode="full",
        )
    finally:
        events = llm.collective_rpc("onerec_p0_trace_stop")[0]
        for event in events:
            print("NPU P0 trace: " + json.dumps(dict(label=label, **event)), flush=True)
    after = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
    _assert_outputs(outputs, prompts, legal, begin, end)
    _assert_matrix_trace(events, prompts, begin, graph=graph)
    if graph:
        assert after["full_dispatches"] - before["full_dispatches"] == 2
        assert after["fallbacks"] == before["fallbacks"]
    else:
        assert after["full_dispatches"] == before["full_dispatches"] == 0
    assert not llm.llm_engine.has_unfinished_requests()
    assert llm.reset_prefix_cache(), "NPU P0 session retained scheduler resources"
    print("NPU P0 dispatch: " + json.dumps(dict(label=label, stats=after)), flush=True)
    return outputs, dict(before=before, after=after, events=events)


def _serialize_outputs(outputs: list[Any]) -> list[list[dict[str, Any]]]:
    return [
        [
            dict(
                tokens=[int(token) for token in sequence.tokens], score=float(sequence.cum_logprob)
            )
            for sequence in output.sequences
        ]
        for output in outputs
    ]


def _run_matrix(work_dir: Path, batch_size: int) -> None:
    import vllm_gr
    from vllm_gr.entrypoints.gr import GRLLM

    assert batch_size in (1, 4)
    source_root = Path(__file__).resolve().parents[2]
    assert Path(vllm_gr.__file__).resolve() == source_root / "vllm_gr/__init__.py"
    work_dir = work_dir.resolve()
    model = os.environ.get("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")
    artifact = work_dir / "p0_constraints.safetensors"
    tokenizer, artifact_digest, tokenizer_digest, legal = _build_constraint_artifact(
        model, artifact, beam_width=_BEAM_WIDTH
    )
    prompts = [_prompt_tokens(tokenizer, 127 + row * 16)[:127] for row in range(batch_size)]
    assert all(len(prompt) + 1 == _INPUT_LEN for prompt in prompts)
    assert len({tuple(prompt) for prompt in prompts}) == batch_size
    begin = int(tokenizer.convert_tokens_to_ids(_BEGIN_TOKEN))
    end = int(tokenizer.convert_tokens_to_ids(_END_TOKEN))
    identity = dict(
        source_root=str(source_root),
        model=model,
        tokenizer_digest=tokenizer_digest,
        artifact_digest=artifact_digest,
        prompts=prompts,
        batch_size=batch_size,
        input_len=_INPUT_LEN,
        beam_width=_BEAM_WIDTH,
        generated_steps=_STEPS,
        dtype="bfloat16",
        seed=0,
    )
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (work_dir / "input-identity.json").write_text(
        json.dumps(dict(sha256=identity_digest, **identity), indent=2) + "\n", encoding="utf-8"
    )
    print(f"NPU P0 input identity: {identity_digest}; B={batch_size}, L=128, W=128", flush=True)
    config: dict[str, Any] = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "graph_enabled": False,
            "graph_batch_size": batch_size,
            "worker_decision": True,
            "max_width": _BEAM_WIDTH,
            "max_decode_steps": _STEPS,
        },
        "constraint_table": {
            "enabled": True,
            "backend": "npu",
            "max_top_k": _BEAM_WIDTH,
            "path": str(artifact),
            "format": "constraint_table_v1",
            "artifact_digest": artifact_digest,
            "tokenizer_digest": tokenizer_digest,
        },
    }
    engine_kwargs: dict[str, Any] = dict(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        vllm_gr_config=config,
        async_scheduling=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=_BEAM_WIDTH,
        max_logprobs=_BEAM_WIDTH,
        block_size=128,
        gpu_memory_utilization=0.5,
        worker_extension_cls="tests.st.test_onerec_v1_npu_p0_matrix.OneRecNPUP0WorkerExtension",
    )
    with GRLLM(**engine_kwargs, enforce_eager=True) as llm:
        eager, eager_evidence = _run_case(
            llm, prompts, legal, begin, end, graph=False, label=f"b{batch_size}-eager"
        )
    config["beam"]["graph_enabled"] = True
    with GRLLM(**engine_kwargs, enforce_eager=False) as llm:
        stats = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
        assert stats["beam_graph_enabled"]
        assert [batch_size, _BEAM_WIDTH] in [list(shape) for shape in stats["captured_shapes"]]
        # NPU TND stores one graph per T, shared across request counts 1..8.
        assert [batch_size * _INPUT_LEN, 1] in stats["gr_prefill_captures"]
        signature = llm.collective_rpc("onerec_p0_context_signature")[0]
        graph, graph_evidence = _run_case(
            llm, prompts, legal, begin, end, graph=True, label=f"b{batch_size}-graph"
        )
        assert llm.collective_rpc("onerec_p0_context_signature")[0] == signature
        repeated, repeat_evidence = _run_case(
            llm, prompts, legal, begin, end, graph=True, label=f"b{batch_size}-graph-repeat"
        )
        assert llm.collective_rpc("onerec_p0_context_signature")[0] == signature
        # Preserve the existing same-B contract: ordered tokens and strict scores.
        _assert_batch_equal(eager, graph, f"P0 eager/graph B={batch_size} W=128")
        _assert_batch_equal(graph, repeated, f"P0 graph repeat B={batch_size} W=128")
    result = dict(
        input_identity_sha256=identity_digest,
        batch_size=batch_size,
        beam_width=_BEAM_WIDTH,
        input_len=_INPUT_LEN,
        generated_steps=_STEPS,
        eager=dict(outputs=_serialize_outputs(eager), **eager_evidence),
        graph=dict(outputs=_serialize_outputs(graph), **graph_evidence),
        graph_repeat=dict(outputs=_serialize_outputs(repeated), **repeat_evidence),
    )
    (work_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"NPU P0 matrix passed: B={batch_size}, L=128, W=128; evidence={work_dir}", flush=True)


@pytest.mark.parametrize("batch_size", [1, 4], ids=["b1", "b4"])
@pytest.mark.timeout(1260)
def test_onerec_v1_npu_p0_matrix(tmp_path: Path, batch_size: int) -> None:
    command = [
        sys.executable,
        "-m",
        "tests.st.test_onerec_v1_npu_p0_matrix",
        str(tmp_path),
        str(batch_size),
    ]
    with subprocess.Popen(
        command,
        cwd=Path(__file__).parents[2],
        env=_matrix_environment(os.environ),
        start_new_session=True,
    ) as process:
        try:
            assert process.wait(timeout=1200) == 0, f"NPU P0 B={batch_size} subprocess failed"
        finally:
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["batch_size"] == batch_size and result["beam_width"] == _BEAM_WIDTH


if __name__ == "__main__":
    _run_matrix(Path(sys.argv[1]), int(sys.argv[2]))
