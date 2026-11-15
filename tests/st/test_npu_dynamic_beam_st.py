# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-call"
"""Bounded full-model dynamic Beam regression, including actual FULL replay.

Run with VLLM_GR_TEST_MODEL and pytest --run-slow on an NPU worker. A usable
NPU without the required Beam operators is a failure, not a skipped check.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pytest.skip("torch-npu is not installed", allow_module_level=True)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is unavailable", allow_module_level=True)

from vllm_gr.ops.npu import beam_ops

assert beam_ops.is_available(), f"NPU Beam operators are required: {beam_ops.unavailable_reason()}"

from tests.st.test_onerec_v1_npu_e2e import (  # noqa: E402
    _BEGIN_TOKEN,
    _END_TOKEN,
    _assert_batch_equal,
    _build_constraint_artifact,
    _prompt_tokens,
)

pytestmark = pytest.mark.npu_test

_WIDTH = 32
_SCHEDULES = (None, (32, 32, 32), (32, 16, 8), (8, 16, 32))


class DynamicBeamTestWorkerExtension:
    """Observe real stage geometry and dynamic state outside model operators."""

    model_runner: Any

    def dynamic_beam_trace_start(self, kind: str = "beam") -> None:
        from vllm.forward_context import get_forward_context

        runner = self.model_runner
        original = runner._model_forward
        self._dynamic_beam_original = original
        self._dynamic_beam_had_override = "_model_forward" in vars(runner)
        self._dynamic_beam_events: list[dict[str, Any]] = []

        def observe(*args: Any, **kwargs: Any) -> Any:
            forward = get_forward_context()
            if kind == "ordinary":
                self._dynamic_beam_events.append(
                    {
                        "mode": forward.cudagraph_runtime_mode.name,
                        "tokens": getattr(forward.batch_descriptor, "num_tokens", None),
                    }
                )
                return original(*args, **kwargs)
            plan = runner._gr_npu_stage_plan
            execution = plan.execution
            owner = runner._gr_npu_stage_runner
            self._dynamic_beam_events.append(
                {
                    "stage": execution.stage_index,
                    "input_width": execution.input_width,
                    "output_width": execution.output_width,
                    "candidate_top_k": execution.candidate_top_k,
                    "rows": plan.model_token_count,
                    "dynamic_state": owner.dynamic_state is not None,
                    "mode": get_forward_context().cudagraph_runtime_mode.name,
                }
            )
            return original(*args, **kwargs)

        runner._model_forward = observe

    def dynamic_beam_trace_stop(self) -> list[dict[str, Any]]:
        if self._dynamic_beam_had_override:
            self.model_runner._model_forward = self._dynamic_beam_original
        else:
            del self.model_runner._model_forward
        return self._dynamic_beam_events

    def dynamic_beam_graph_signature(self) -> dict[str, Any]:
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        runner = self.model_runner
        owner = runner._gr_npu_graph_model
        ordinary = owner.original_model
        assert runner.model is owner, "Beam graph owner was replaced"
        assert isinstance(ordinary, ACLGraphWrapper), type(ordinary)

        def entries(wrapper: Any) -> list[tuple[int, bool, int]]:
            return sorted(
                (descriptor.num_tokens, descriptor.uniform, id(entry.aclgraph))
                for descriptor, entry in wrapper.concrete_aclgraph_entries.items()
                if entry.aclgraph is not None
            )

        banks = []
        for batch, width in sorted(runner._gr_npu_beam_graph_shapes):
            context = runner._gr_npu_graph_contexts[(batch, width)]
            pools = (
                context.pools
                if width == context.max_beams
                else context._npu_beam_width_pools[width]
            )
            banks.append(
                (
                    batch,
                    width,
                    [
                        (pool.unshared_key.data_ptr(), pool.unshared_value.data_ptr())
                        for pool in pools
                    ],
                )
            )
        return {"beam": entries(owner), "ordinary": entries(ordinary), "banks": banks}


def _run_case(
    llm: Any,
    tokenizer: Any,
    prompts: list[list[int]],
    legal: set[tuple[int, int, int]],
    schedule: tuple[int, ...] | None,
    result_width: int | None,
    *,
    graph: bool,
    full_stages: tuple[int, ...] | None = None,
    fallback_count: int | None = None,
) -> list[Any]:
    from vllm.inputs import tokens_input

    from vllm_gr.sampling_params import BeamSearchParams

    widths = schedule or (_WIDTH,) * 3
    returned = widths[-1] if result_width is None else result_width
    before = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
    llm.collective_rpc("dynamic_beam_trace_start")
    try:
        outputs: list[Any] = llm.beam_search_v1(
            [
                tokens_input(prompt, cache_salt=f"{schedule}-{result_width}-{row}")
                for row, prompt in enumerate(prompts)
            ],
            BeamSearchParams(
                beam_width=_WIDTH,
                max_tokens=5,
                begin_token=_BEGIN_TOKEN,
                end_token=_END_TOKEN,
                ignore_eos=True,
                temperature=0.0,
                dynamic_beam_widths=None if schedule is None else list(schedule),
                result_width=result_width,
            ),
            output_mode="full",
        )
    finally:
        events = llm.collective_rpc("dynamic_beam_trace_stop")[0]
    assert [event["stage"] for event in events] == [0, 1, 2], events
    assert [event["input_width"] for event in events] == [1, widths[0], widths[1]], events
    assert [event["output_width"] for event in events] == list(widths), events
    assert [event["candidate_top_k"] for event in events] == list(widths), events
    assert [event["rows"] for event in events] == [
        sum(len(prompt) + 1 for prompt in prompts),
        len(prompts) * widths[0],
        len(prompts) * widths[1],
    ], events
    expected_dynamic = widths != (_WIDTH,) * 3 or result_width is not None
    assert all(event["dynamic_state"] == expected_dynamic for event in events[1:]), events
    after = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
    expected_full = ((1, 2) if graph else ()) if full_stages is None else full_stages
    assert after["full_dispatches"] - before["full_dispatches"] == len(expected_full)
    if graph:
        for stage in (1, 2):
            assert (events[stage]["mode"] == "FULL") == (stage in expected_full), events
        expected_fallbacks = 2 - len(expected_full) if fallback_count is None else fallback_count
        assert after["fallbacks"] - before["fallbacks"] == expected_fallbacks, after
    begin = tokenizer.convert_tokens_to_ids(_BEGIN_TOKEN)
    end = tokenizer.convert_tokens_to_ids(_END_TOKEN)
    assert len(outputs) == len(prompts)
    for prompt, output in zip(prompts, outputs, strict=True):
        assert len(output.sequences) == returned
        suffixes = []
        scores = []
        for sequence in output.sequences:
            assert list(sequence.tokens[: len(prompt)]) == prompt
            suffix = list(sequence.tokens[len(prompt) :])
            assert len(suffix) == 5 and suffix[0] == begin and suffix[-1] == end
            assert tuple(suffix[1:-1]) in legal
            assert math.isfinite(sequence.cum_logprob)
            scores.append(float(sequence.cum_logprob))
            suffixes.append(tuple(suffix))
        assert len(set(suffixes)) == returned
        assert scores == sorted(scores, reverse=True)
    assert not llm.llm_engine.has_unfinished_requests()
    assert llm.reset_prefix_cache(), "dynamic Beam did not release request resources"
    print(
        f"Dynamic Beam: graph={graph}, schedule={schedule}, result={returned}, trace={events}",
        flush=True,
    )
    return outputs


def _run_ordinary(llm: Any, prompt_token: int, *, graph: bool) -> list[list[int]]:
    """Use 64 ordinary Decode rows, colliding with B=2/W=32 Beam's descriptor."""
    from vllm.inputs import tokens_input
    from vllm.sampling_params import SamplingParams

    llm.collective_rpc("dynamic_beam_trace_start", args=("ordinary",))
    try:
        outputs = llm.generate(
            [tokens_input([prompt_token], cache_salt=f"ordinary-{row}") for row in range(64)],
            SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
            use_tqdm=False,
        )
    finally:
        events = llm.collective_rpc("dynamic_beam_trace_stop")[0]
    if graph:
        assert any(event["mode"] == "FULL" and event["tokens"] == 64 for event in events), events
    assert len(outputs) == 64
    tokens = [list(output.outputs[0].token_ids) for output in outputs]
    assert all(len(row) == 3 for row in tokens)
    assert not llm.llm_engine.has_unfinished_requests()
    assert llm.reset_prefix_cache()
    print(f"Ordinary graph collision control: graph={graph}, trace={events}", flush=True)
    return tokens


def _assert_graph_isolation(signature: dict[str, Any]) -> None:
    beam = {(tokens, uniform): handle for tokens, uniform, handle in signature["beam"]}
    ordinary = {(tokens, uniform): handle for tokens, uniform, handle in signature["ordinary"]}
    assert (64, True) in beam and (64, True) in ordinary, signature
    assert beam[(64, True)] != ordinary[(64, True)], signature
    assert set(beam.values()).isdisjoint(ordinary.values()), signature


def _run_regression(work_dir: Path) -> None:
    from vllm_gr.entrypoints.gr import GRLLM

    model = os.environ.get("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")
    artifact = work_dir / "dynamic-beam.safetensors"
    tokenizer, digest, tokenizer_digest, legal = _build_constraint_artifact(
        model, artifact, beam_width=_WIDTH
    )
    prompts = [_prompt_tokens(tokenizer, 127), _prompt_tokens(tokenizer, 159)[:127]]
    assert prompts[0] != prompts[1], "multi-request regression needs distinct prompts"
    config: dict[str, Any] = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "worker_decision": True,
            "max_width": _WIDTH,
            "max_decode_steps": 3,
            "graph_enabled": False,
            "graph_batch_size": 2,
            "graph_widths": [8, 16, 32],
        },
        "constraint_table": {
            "enabled": True,
            "backend": "npu",
            "max_top_k": _WIDTH,
            "path": str(artifact),
            "format": "constraint_table_v1",
            "artifact_digest": digest,
            "tokenizer_digest": tokenizer_digest,
        },
    }
    kwargs: dict[str, Any] = dict(
        model=model,
        trust_remote_code=True,
        # Keep this dynamic Beam and FULL Graph regression in FP16. On A5,
        # BF16 VLLM_COMPILE currently diverges from eager in OneRec Prefill
        # hidden states; BF16 compiled/eager precision belongs to separate
        # precision coverage so it does not obscure Graph execution correctness.
        dtype="float16",
        seed=0,
        vllm_gr_config=config,
        async_scheduling=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=64,
        max_logprobs=_WIDTH,
        gpu_memory_utilization=0.5,
        worker_extension_cls="tests.st.test_npu_dynamic_beam_st.DynamicBeamTestWorkerExtension",
    )
    reference: list[list[Any]] = []
    ordinary_reference: list[list[int]] = []
    single_reference: list[Any] = []
    # Vary both pruning and widening, then return to a previous schedule so
    # stale histories/KV banks cannot pass as single-request happy-path coverage.
    cases = [(schedule, None) for schedule in _SCHEDULES] + [((32, 16, 8), 3), ((8, 16, 32), 7)]
    for graph, explicit_widths in ((False, True), (True, True), (True, False)):
        config["beam"]["graph_enabled"] = graph
        if explicit_widths:
            config["beam"]["graph_widths"] = [8, 16, 32]
        else:
            config["beam"].pop("graph_widths")
        with GRLLM(**kwargs, enforce_eager=not graph) as llm:
            if graph:
                stats = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
                expected_shapes = {(2, 8), (2, 16), (2, 32)} if explicit_widths else {(2, 32)}
                assert expected_shapes == {tuple(shape) for shape in stats["captured_shapes"]}, (
                    stats
                )
                signature = llm.collective_rpc("dynamic_beam_graph_signature")[0]
                _assert_graph_isolation(signature)
            ordinary_before = _run_ordinary(llm, prompts[0][0], graph=graph)
            if graph:
                assert llm.collective_rpc("dynamic_beam_graph_signature")[0] == signature
            active_cases = cases if explicit_widths else cases[:3]
            actual = [
                _run_case(
                    llm,
                    tokenizer,
                    prompts,
                    legal,
                    schedule,
                    returned,
                    graph=graph,
                    full_stages=None
                    if explicit_widths or schedule is None or schedule == (32, 32, 32)
                    else (1,),
                )
                for schedule, returned in active_cases
            ]
            _assert_batch_equal(actual[0], actual[1], "fixed/equal-width compatibility")
            truncated_pairs = (
                ((actual[2], actual[4]), (actual[3], actual[5])) if explicit_widths else ()
            )
            for complete, truncated in truncated_pairs:
                for left, right in zip(complete, truncated, strict=True):
                    assert [list(seq.tokens) for seq in right.sequences] == [
                        list(seq.tokens) for seq in left.sequences[: len(right.sequences)]
                    ]
                    assert [seq.cum_logprob for seq in right.sequences] == pytest.approx(
                        [seq.cum_logprob for seq in left.sequences[: len(right.sequences)]],
                        abs=1e-4,
                        rel=1e-5,
                    )
            if graph:
                for index, (left, right) in enumerate(
                    zip(reference[: len(actual)], actual, strict=True)
                ):
                    _assert_batch_equal(left, right, f"eager/FULL dynamic case {cases[index]}")
            else:
                reference = actual
            # B=1/W=32 has the same token count as captured B=2/W=16. It
            # must miss the exact shape gate even though that descriptor exists.
            single = _run_case(
                llm,
                tokenizer,
                prompts[:1],
                legal,
                None,
                None,
                graph=graph,
                full_stages=(),
            )
            if graph:
                _assert_batch_equal(
                    single_reference, single, "same-token different-B shape fallback"
                )
            else:
                single_reference = single
            ordinary_after = _run_ordinary(llm, prompts[0][0], graph=graph)
            assert ordinary_before == ordinary_after
            if graph:
                assert ordinary_reference == ordinary_after
                assert llm.collective_rpc("dynamic_beam_graph_signature")[0] == signature
            else:
                ordinary_reference = ordinary_after


@pytest.mark.npu_test
@pytest.mark.slow
@pytest.mark.timeout(960)
def test_npu_dynamic_beam_full_model_eager_graph(tmp_path: Path) -> None:
    env = dict(os.environ)
    env.update(
        USE_TF="0",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ENABLE_PREFILL_CUDAGRAPH="0",
    )
    command = [sys.executable, "-m", "tests.st.test_npu_dynamic_beam_st", str(tmp_path)]
    with subprocess.Popen(
        command, cwd=Path(__file__).parents[2], env=env, start_new_session=True
    ) as process:
        try:
            assert process.wait(timeout=900) == 0, "NPU dynamic Beam E2E subprocess failed"
        finally:
            if process.poll() is None:
                if os.name == "posix":
                    getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL"))
                else:
                    process.kill()
                process.wait()


if __name__ == "__main__":
    _run_regression(Path(sys.argv[1]))
