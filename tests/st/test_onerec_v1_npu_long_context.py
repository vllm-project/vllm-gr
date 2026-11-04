# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""Cold 1024/4096-token NPU whole-model eager/FULL functional regressions."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.st.npu_beam_diagnostics import NPUForwardTrace
from tests.st.npu_long_beam_helpers import (
    assert_long_graph_trace,
    assert_long_outputs_equal,
    long_test_environment,
)


def _unavailable(reason: str) -> None:
    if os.environ.get("VLLM_GR_TARGET_DEVICE") == "npu":
        pytest.fail(f"required long-context NPU ST runtime is unavailable: {reason}", pytrace=False)
    pytest.skip(reason, allow_module_level=True)


try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    _unavailable(f"torch-npu is unavailable: {exc}")
if not hasattr(torch, "npu") or not torch.npu.is_available():
    _unavailable("Ascend NPU is unavailable")

from vllm_gr.ops.npu import beam_ops  # noqa: E402

if not beam_ops.is_available():
    _unavailable(f"xllm Beam operators are unavailable: {beam_ops.unavailable_reason()}")

from tests.st.test_onerec_v1_npu_e2e import (  # noqa: E402
    _BEGIN_TOKEN,
    _build_constraint_artifact,
    _graph_context_signature,
    _run_case,
)


class LongContextNPUWorkerExtension:
    """Observe actual scheduler and forward inputs outside compiled model code."""

    _long_trace: NPUForwardTrace | None = None

    def long_trace_start(self, batch_size: int, total_tokens: int) -> None:
        from vllm.forward_context import get_forward_context

        if self._long_trace is not None:
            raise RuntimeError("a long-context trace is already active")
        self._long_trace = NPUForwardTrace(
            getattr(self, "model_runner"),
            get_forward_context,
            expected_batch_size=batch_size,
            max_padded_tokens=total_tokens,
            record_cold=True,
        )

    def long_trace_stop(self) -> list[dict[str, Any]]:
        if self._long_trace is None:
            raise RuntimeError("no long-context trace is active")
        try:
            return self._long_trace.finish()
        finally:
            self._long_trace = None

    def long_context_signature(self) -> tuple[int, tuple[int, ...]]:
        return _graph_context_signature(self)


def _long_prompts(tokenizer: Any, batch_size: int, input_len: int) -> list[list[int]]:
    source = Path(__file__).parents[1] / "resources/single_one_rec_prompt.txt"
    tokens = tokenizer.encode(source.read_text(), add_special_tokens=False)
    assert len(tokens) >= 128, "long-context fixture must contain varied real tokens"
    # Repeat real fixture tokens, with a different circular offset per request.
    # The public API appends SID_BEGIN, so raw inputs are one token shorter.
    prompts = [
        [int(tokens[(position + 31 * row) % len(tokens)]) for position in range(input_len - 1)]
        for row in range(batch_size)
    ]
    assert len({tuple(prompt) for prompt in prompts}) == batch_size
    return prompts


def _run_long_context(work_dir: Path, *, batch_size: int, input_len: int) -> None:
    from vllm_gr.entrypoints.gr import GRLLM

    total = batch_size * input_len
    model = os.environ.get("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")
    artifact = work_dir / "long-constraints.safetensors"
    tokenizer, digest, tokenizer_digest, legal = _build_constraint_artifact(model, artifact)
    prompts = _long_prompts(tokenizer, batch_size, input_len)
    begin = int(tokenizer.convert_tokens_to_ids(_BEGIN_TOKEN))
    expected_tokens = [token for prompt in prompts for token in [*prompt, begin]]
    expected_positions = list(range(input_len)) * batch_size
    config: dict[str, Any] = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "worker_decision": True,
            "max_width": 8,
            "max_decode_steps": 3,
            "graph_batch_size": batch_size,
        },
        "constraint_table": {
            "enabled": True,
            "backend": "npu",
            "max_top_k": 8,
            "path": str(artifact),
            "format": "constraint_table_v1",
            "artifact_digest": digest,
            "tokenizer_digest": tokenizer_digest,
        },
    }
    # The current NPU runner clamps total-token buckets by max_model_len.
    # Keep the exact B*L bucket and room for Decode without changing real L.
    model_bound = max(1024, total + 128)
    engine_kwargs = dict(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        vllm_gr_config=config,
        async_scheduling=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=model_bound,
        max_num_batched_tokens=model_bound,
        max_num_seqs=8,
        max_logprobs=8,
        block_size=128,
        gpu_memory_utilization=0.5,
        worker_extension_cls=(
            "tests.st.test_onerec_v1_npu_long_context.LongContextNPUWorkerExtension"
        ),
    )
    reference = None
    results = []
    for graph in (False, True):
        config["beam"]["graph_enabled"] = graph
        with GRLLM(**engine_kwargs, enforce_eager=not graph) as llm:
            if graph:
                initial = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
                assert [batch_size, 8] in [list(shape) for shape in initial["captured_shapes"]]
                signature = llm.collective_rpc("long_context_signature")[0]
            repeated = None
            for repeat in range(2):
                label = f"b{batch_size}-l{input_len}-{'full' if graph else 'eager'}-{repeat}"
                before = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
                llm.collective_rpc("long_trace_start", args=(batch_size, total))
                try:
                    outputs = _run_case(llm, tokenizer, prompts, legal, steps=3)
                finally:
                    events = llm.collective_rpc("long_trace_stop")[0]
                assert_long_graph_trace(
                    events, batch_size=batch_size, input_len=input_len, graph=graph
                )
                assert events[0]["input_ids"] == expected_tokens, label
                assert events[0]["positions"] == expected_positions, label
                after = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
                assert after["full_dispatches"] - before["full_dispatches"] == (2 if graph else 0)
                assert after["fallbacks"] - before["fallbacks"] == (0 if graph else 2)
                reasons = {
                    key: value - before["fallback_reasons"].get(key, 0)
                    for key, value in after["fallback_reasons"].items()
                    if value != before["fallback_reasons"].get(key, 0)
                }
                assert reasons == ({} if graph else {"graph_disabled": 2}), label
                packed = [
                    [
                        (
                            list(seq.tokens),
                            float(seq.cum_logprob),
                            seq.finish_reason,
                            seq.stop_reason,
                        )
                        for seq in output.sequences
                    ]
                    for output in outputs
                ]
                if reference is None:
                    reference = packed
                assert_long_outputs_equal(reference, packed, label=label)
                if repeated is not None:
                    assert packed == repeated, f"{label}: identical-mode repeat changed"
                repeated = packed
                if graph:
                    assert llm.collective_rpc("long_context_signature")[0] == signature
                results.append(
                    dict(label=label, trace=events, outputs=packed, before=before, after=after)
                )
                print(f"NPU long-context comparison passed: {label}; stats={after}", flush=True)
    (work_dir / "long-context-result.json").write_text(
        json.dumps(
            dict(
                model=model,
                batch_size=batch_size,
                input_len=input_len,
                beam_width=8,
                results=results,
            )
        )
        + "\n"
    )


@pytest.mark.npu_integration
@pytest.mark.slow
@pytest.mark.timeout(1260)
@pytest.mark.parametrize("input_len", [1024, 4096], ids=["l1024", "l4096"])
@pytest.mark.parametrize("batch_size", [1, 4], ids=["b1", "b4"])
def test_onerec_v1_npu_long_context(tmp_path: Path, batch_size: int, input_len: int) -> None:
    env = long_test_environment(os.environ, batch_size=batch_size, input_len=input_len)
    env.update(
        USE_TF="0",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    command = [
        sys.executable,
        "-m",
        "tests.st.test_onerec_v1_npu_long_context",
        str(tmp_path),
        "--batch-size",
        str(batch_size),
        "--input-len",
        str(input_len),
    ]
    with subprocess.Popen(
        command, cwd=Path(__file__).parents[2], env=env, start_new_session=True
    ) as process:
        try:
            assert process.wait(timeout=1200) == 0, "long-context NPU subprocess failed"
        finally:
            if process.poll() is None:
                if os.name == "posix":
                    getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL"))
                else:
                    process.kill()
                process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--batch-size", type=int, choices=[1, 4], required=True)
    parser.add_argument("--input-len", type=int, choices=[1024, 4096], required=True)
    args = parser.parse_args()
    _run_long_context(args.work_dir, batch_size=args.batch_size, input_len=args.input_len)
