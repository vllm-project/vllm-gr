# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""Resource-bounded NPU public-API eager/graph correctness regression.

The graph test uses two different raw 127-token prompts (128 each after SID),
exact Prefill bucket 256, and strict same-B ordered tokens/scores. It retains
native compilation, Beam FULL Decode, repeats and B=2 -> B=1 -> B=2 reuse.
VLLM_GR_TEST_PADDED_PREFILL=1 selects the original 129+161-token case with
caller-configured buckets, preserving the known padding precision reproducer.
An independent B=4 test compares two engines on four distinct cold 128-token
requests, requiring exact outputs and actual Prefill/Decode FULL execution.
Only small forward metadata is observed; no operator hooks or tensor dumps.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from tests.st.npu_beam_diagnostics import NPUForwardTrace


def _npu_test_unavailable(reason: str) -> None:
    if os.environ.get("VLLM_GR_TARGET_DEVICE") == "npu":
        pytest.fail(f"required NPU ST runtime is unavailable: {reason}", pytrace=False)
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

_BEAM_WIDTH = 8
_BEGIN_TOKEN = "<|sid_begin|>"
_END_TOKEN = "<|sid_end|>"


def _prompt_tokens(tokenizer: Any, length: int) -> list[int]:
    source_path = Path(__file__).parents[1] / "resources/single_one_rec_prompt.txt"
    source = tokenizer.encode(source_path.read_text(), add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    assert len(newline) == 1
    if len(source) >= length:
        return [int(token) for token in source[-length:]]
    return [int(newline[0])] * (length - len(source)) + [int(token) for token in source]


def _padded_prefill_requested(env: Mapping[str, str]) -> bool:
    return env.get("VLLM_GR_TEST_PADDED_PREFILL") == "1"


def _graph_test_environment(env: Mapping[str, str]) -> dict[str, str]:
    result = dict(env)
    if not _padded_prefill_requested(env):
        # A bounded exact-bucket regression, not a change to product defaults.
        # Override stale shell settings so passing cannot mean Prefill was off.
        result.update(
            VLLM_ENABLE_PREFILL_CUDAGRAPH="1",
            VLLM_GR_PREFILL_GRAPH_BUCKET_STEP="128",
            VLLM_GR_PREFILL_GRAPH_MAX_BUCKET="256",
        )
    return result


def _comparison_prompts(tokenizer: Any, *, exact_prefill: bool) -> list[list[int]]:
    if not exact_prefill:
        return [_prompt_tokens(tokenizer, length) for length in (128, 160)]
    # Keep different content at equal lengths, rather than testing duplicates
    # only. The public API appends one SID begin token to each raw prompt.
    prompts = [_prompt_tokens(tokenizer, 127), _prompt_tokens(tokenizer, 159)[:127]]
    assert all(len(prompt) == 127 for prompt in prompts)
    assert prompts[0] != prompts[1], "exact-bucket coverage needs distinct prompts"
    return prompts


def _assert_exact_prefill(prefill: dict[str, Any], label: str, *, batch_size: int = 2) -> None:
    total = batch_size * 128
    assert prefill["actual_tokens"] == prefill["padded_tokens"] == total, label
    assert prefill["input_ids_shape"] == prefill["positions_shape"] == [total], label
    assert prefill["attention"]["query_start_loc"] == list(range(0, total + 1, 128)), label
    assert prefill["attention"]["seq_lens"] == [128] * batch_size, label
    replay = prefill["gr_prefill_replay_delta"]
    if prefill["enforce_eager"]:
        assert prefill["compilation_mode"] == "NONE" and not replay, label
    else:
        assert prefill["compilation_mode"] == "VLLM_COMPILE", label
        assert prefill["skip_compiled"] is False, label
        # The outer runner's runtime mode alone does not prove GR Prefill replay.
        assert replay == {str(total): 1}, f"{label}: expected exact Prefill replay, got {replay}"
    print(
        f"NPU exact Prefill: {label}; actual={total}; bucket={total}; replay={replay}", flush=True
    )


def _assert_b4_graph_trace(events: list[dict[str, Any]], *, graph: bool) -> None:
    assert [event["stage"] for event in events] == [0, 1, 2], "incomplete B=4 model trace"
    assert [event["kind"] for event in events] == ["prefill", "decode", "decode"]
    prefill = events[0]
    _assert_exact_prefill(prefill, "cold B=4 Prefill", batch_size=4)
    assert (
        prefill["cold_rows"]
        == [{"num_computed_tokens": 0, "prompt_tokens": 128, "scheduled_tokens": 128}] * 4
    )
    assert prefill["scheduled_total"] == 512
    assert prefill["enforce_eager"] is (not graph)
    for event in events[1:]:
        assert event["actual_tokens"] == 32, "B=4/W=8 must decode all 32 Beam rows"
        assert event["runtime_mode"] == ("FULL" if graph else "NONE")
        assert event["enforce_eager"] is (not graph)
    if not graph:
        assert all(event["runtime_mode"] == "NONE" for event in events)


def _build_constraint_artifact(
    model: str,
    artifact: Path,
    *,
    beam_width: int = _BEAM_WIDTH,
) -> tuple[Any, str, str, set[tuple[int, int, int]]]:
    from transformers import AutoConfig, AutoTokenizer

    from vllm_gr.constraints.artifact import save_constraint_table_artifact
    from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
    from vllm_gr.constraints.json_compiler import compile_relations_json
    from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    vocab_size = AutoConfig.from_pretrained(model, trust_remote_code=True).vocab_size
    triples = [
        [f"<s_a_{a}>", f"<s_b_{b}>", f"<s_c_{c}>"]
        for a, b, c in product(range(beam_width), range(4), range(4))
    ]
    tokenizer_digest = compute_tokenizer_digest(tokenizer)
    table = ConstraintTableArtifactCompiler(
        tokenizer,
        vocab_size,
        tokenizer_digest=tokenizer_digest,
    ).compile(compile_relations_json(triples))
    digest = save_constraint_table_artifact(table, artifact)
    legal = {
        tuple(int(token) for token in tokenizer.convert_tokens_to_ids(triple)) for triple in triples
    }
    return (
        tokenizer,
        cast(str, digest),
        cast(str, tokenizer_digest),
        cast(set[tuple[int, int, int]], legal),
    )


def _assert_public_outputs(
    tokenizer: Any,
    prompts: list[list[int]],
    outputs: list[Any],
    legal: set[tuple[int, int, int]],
    steps: int,
) -> None:
    begin = int(tokenizer.convert_tokens_to_ids(_BEGIN_TOKEN))
    end = int(tokenizer.convert_tokens_to_ids(_END_TOKEN))
    legal_prefixes = {tokens[:steps] for tokens in legal}

    assert len(outputs) == len(prompts)
    for prompt, output in zip(prompts, outputs, strict=True):
        assert len(output.sequences) == _BEAM_WIDTH
        scores = []
        generated = []
        for sequence in output.sequences:
            tokens = [int(token) for token in sequence.tokens]
            assert tokens[: len(prompt)] == prompt
            suffix = tokens[len(prompt) :]
            assert suffix[0] == begin and suffix[-1] == end
            assert len(suffix) == steps + 2
            semantic = tuple(suffix[1:-1])
            assert semantic in legal_prefixes, (
                f"illegal N={steps} prefix ids={semantic}, "
                f"tokens={tokenizer.convert_ids_to_tokens(list(semantic))}"
            )
            assert sequence.finish_reason == "length"
            assert sequence.stop_reason is None
            assert math.isfinite(sequence.cum_logprob)
            scores.append(float(sequence.cum_logprob))
            generated.append(semantic)
        assert all(left >= right for left, right in zip(scores, scores[1:]))
        assert len(set(generated)) == _BEAM_WIDTH


def _run_case(
    llm: Any,
    tokenizer: Any,
    prompts: list[list[int]],
    legal: set[tuple[int, int, int]],
    steps: int,
    *,
    cache_salts: list[str] | None = None,
) -> list[Any]:
    from vllm_gr.sampling_params import BeamSearchParams

    if cache_salts is None:
        request_prompts = [{"prompt_token_ids": prompt} for prompt in prompts]
    else:
        from vllm.inputs import tokens_input

        if len(cache_salts) != len(prompts):
            raise ValueError("cache_salts must contain one value per prompt")
        request_prompts = [
            tokens_input(prompt, cache_salt=cache_salt)
            for prompt, cache_salt in zip(prompts, cache_salts, strict=True)
        ]

    params = BeamSearchParams(
        beam_width=_BEAM_WIDTH,
        max_tokens=steps + 2,
        temperature=0.0,
        ignore_eos=True,
        begin_token=_BEGIN_TOKEN,
        end_token=_END_TOKEN,
        length_penalty=1.0,
    )
    print(
        f"NPU V1 inputs: B={len(prompts)}, N={steps}, W={_BEAM_WIDTH}; "
        f"prompt_lengths={[len(prompt) for prompt in prompts]}; cache_salts={cache_salts}",
        flush=True,
    )
    outputs = llm.beam_search_v1(request_prompts, params, output_mode="full")
    _assert_public_outputs(tokenizer, prompts, outputs, legal, steps)
    assert not llm.llm_engine.has_unfinished_requests()
    assert llm.reset_prefix_cache(), "NPU session retained scheduler resources"
    print(
        f"NPU V1 E2E: B={len(prompts)}, N={steps}, W={_BEAM_WIDTH}, passed",
        flush=True,
    )
    return outputs


def _print_comparison(
    reference: Any,
    candidate: Any,
    *,
    label: str,
    prompt_length: int,
) -> None:
    def rows(output: Any) -> list[tuple[tuple[int, ...], float]]:
        return [
            (
                tuple(int(token) for token in sequence.tokens[prompt_length:]),
                float(sequence.cum_logprob),
            )
            for sequence in output.sequences
        ]

    reference_rows, candidate_rows = rows(reference), rows(candidate)
    reference_tokens = [tokens for tokens, _ in reference_rows]
    candidate_tokens = [tokens for tokens, _ in candidate_rows]
    reference_scores = dict(reference_rows)
    candidate_scores = dict(candidate_rows)
    common = set(reference_scores) & set(candidate_scores)
    max_score_delta = max(
        (abs(candidate_scores[tokens] - reference_scores[tokens]) for tokens in common),
        default=0.0,
    )
    print(
        f"NPU V1 diagnostic: {label}; prompt_length={prompt_length}; "
        f"ordered_equal={reference_tokens == candidate_tokens}; "
        f"set_equal={set(reference_tokens) == set(candidate_tokens)}; "
        f"common={len(common)}/{_BEAM_WIDTH}; "
        f"max_score_delta={f'{max_score_delta:.9f}' if common else 'n/a'}",
        flush=True,
    )


def _graph_context_signature(worker: Any) -> tuple[int, tuple[int, ...]]:
    """Observe captured storage in the same Worker without changing it."""
    context = worker.model_runner._gr_npu_graph_context
    buffers = [
        context.decode_step_buf,
        context.decode_step_1based_buf,
        context.xllm_block_table_buf,
        context.suffix_k_buf,
        context.suffix_v_buf,
        context.decision.sequence,
        context.decision.beam_scores,
        context.decision.group_token_num,
    ]
    for pool in context.pools:
        buffers.extend((pool.unshared_key, pool.unshared_value))
    return id(context), tuple(tensor.data_ptr() for tensor in buffers)


class OneRecNPUGraphTestWorkerExtension:
    """Observe runner metadata and graph buffer addresses, not model operators."""

    _onerec_trace: NPUForwardTrace | None = None

    def onerec_npu_graph_context_signature(self) -> tuple[int, tuple[int, ...]]:
        return _graph_context_signature(self)

    def onerec_npu_trace_start(
        self, expected_batch_size: int = 2, record_cold: bool = False
    ) -> None:
        from vllm.forward_context import get_forward_context

        if getattr(self, "_onerec_trace", None) is not None:
            raise RuntimeError("an NPU forward trace is already active")
        self._onerec_trace = NPUForwardTrace(
            getattr(self, "model_runner"),
            get_forward_context,
            expected_batch_size=expected_batch_size,
            record_cold=record_cold,
        )

    def onerec_npu_trace_stop(self) -> list[dict[str, Any]]:
        if self._onerec_trace is None:
            raise RuntimeError("no NPU forward trace is active")
        try:
            return self._onerec_trace.finish()
        finally:
            self._onerec_trace = None


def _run_traced_case(
    llm: Any,
    tokenizer: Any,
    prompts: list[list[int]],
    legal: set[tuple[int, int, int]],
    label: str,
    *,
    record_cold: bool = False,
    exact_prefill: bool | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    llm.collective_rpc("onerec_npu_trace_start", args=(len(prompts), record_cold))
    try:
        outputs = _run_case(
            llm,
            tokenizer,
            prompts,
            legal,
            steps=3,
            cache_salts=(
                None if record_cold else [f"{label}-row-{row}" for row in range(len(prompts))]
            ),
        )
    finally:
        events = llm.collective_rpc("onerec_npu_trace_stop")[0]
        for event in events:
            print("NPU forward trace: " + json.dumps(dict(label=label, **event)), flush=True)
    assert [event["stage"] for event in events] == [0, 1, 2], f"incomplete trace: {label}"
    prefill = events[0]
    begin = int(tokenizer.convert_tokens_to_ids(_BEGIN_TOKEN))
    expected_tokens = [token for prompt in prompts for token in [*prompt, begin]]
    expected_positions = [position for prompt in prompts for position in range(len(prompt) + 1)]
    actual = prefill["actual_tokens"]
    assert actual == len(expected_tokens), label
    assert prefill["input_ids"][:actual] == expected_tokens, label
    assert prefill["positions"][:actual] == expected_positions, label
    assert len(set(prefill["input_request_ids"])) == len(prompts), label
    if exact_prefill or (exact_prefill is None and not _padded_prefill_requested(os.environ)):
        _assert_exact_prefill(prefill, label, batch_size=len(prompts))
    return outputs, events


def _diagnose_batch(
    tokenizer: Any,
    prompts: list[list[int]],
    reference: list[Any],
    candidate: list[Any],
    label: str,
) -> None:
    for row, (prompt, left, right) in enumerate(zip(prompts, reference, candidate, strict=True)):
        _print_comparison(
            left,
            right,
            label=f"{label} request={row}",
            prompt_length=len(prompt),
        )


def _assert_batch_equal(reference: list[Any], candidate: list[Any], label: str) -> None:
    for row, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        assert [list(seq.tokens) for seq in right.sequences] == [
            list(seq.tokens) for seq in left.sequences
        ], f"{label}: ordered tokens differ, request={row}"
        assert [float(seq.cum_logprob) for seq in right.sequences] == pytest.approx(
            [float(seq.cum_logprob) for seq in left.sequences], abs=1e-4, rel=1e-5
        ), f"{label}: scores differ, request={row}"


def _run_npu_e2e(work_dir: Path, *, graph: bool = False) -> None:
    from vllm_gr.entrypoints.gr import GRLLM

    exact_prefill = graph and not _padded_prefill_requested(os.environ)
    model = os.environ.get("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")
    tokenizer, digest, tokenizer_digest, legal = _build_constraint_artifact(
        model,
        work_dir / "onerec_v1_npu_constraints.safetensors",
    )
    config: dict[str, Any] = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "graph_enabled": False,
            "worker_decision": True,
            "max_width": _BEAM_WIDTH,
            "max_decode_steps": 3,
        },
        "constraint_table": {
            "enabled": True,
            "backend": "npu",
            "max_top_k": _BEAM_WIDTH,
            "path": str(work_dir / "onerec_v1_npu_constraints.safetensors"),
            "format": "constraint_table_v1",
            "artifact_digest": digest,
            "tokenizer_digest": tokenizer_digest,
        },
    }

    engine_kwargs: dict[str, Any] = {
        "model": model,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "seed": 0,
        "vllm_gr_config": config,
        "async_scheduling": True,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "max_num_seqs": _BEAM_WIDTH,
        "max_logprobs": _BEAM_WIDTH,
        "gpu_memory_utilization": 0.5,
    }
    if graph:
        engine_kwargs["worker_extension_cls"] = (
            "tests.st.test_onerec_v1_npu_e2e.OneRecNPUGraphTestWorkerExtension"
        )
    comparison_prompts = _comparison_prompts(tokenizer, exact_prefill=exact_prefill)
    if graph:
        print(
            "NPU Prefill regression: "
            f"case={'exact128x2' if exact_prefill else 'legacy290'}; "
            f"raw_lengths={[len(prompt) for prompt in comparison_prompts]}; "
            f"effective_lengths={[len(prompt) + 1 for prompt in comparison_prompts]}; "
            "padding precision issue remains open (NPU-PREFILL-PADDING-001)",
            flush=True,
        )
    with GRLLM(**engine_kwargs, enforce_eager=True) as llm:
        staged_prompt = comparison_prompts[0]
        isolated = _run_case(
            llm,
            tokenizer,
            [staged_prompt],
            legal,
            steps=3,
            cache_salts=["eager-isolated-n3-row-0"],
        )[0]

        # Separate salts keep prefix caching enabled while forcing both
        # duplicate requests through a complete Multi-Batch Prefill.
        duplicate_outputs = _run_case(
            llm,
            tokenizer,
            [staged_prompt, staged_prompt.copy()],
            legal,
            steps=2,
            cache_salts=[
                "npu-e2e-duplicate-n2-request-row-0",
                "npu-e2e-duplicate-n2-request-row-1",
            ],
        )
        _print_comparison(
            duplicate_outputs[0],
            duplicate_outputs[1],
            label="same-batch duplicate N=2 request rows",
            prompt_length=len(staged_prompt),
        )

        if graph:
            batched_outputs, _ = _run_traced_case(
                llm, tokenizer, comparison_prompts, legal, "eager-reference-n3"
            )
            eager_repeat, _ = _run_traced_case(
                llm,
                tokenizer,
                comparison_prompts,
                legal,
                "eager-repeat-n3",
            )
            _diagnose_batch(
                tokenizer,
                comparison_prompts,
                batched_outputs,
                eager_repeat,
                "eager repeat B=2 N=3",
            )
        else:
            batched_outputs = _run_case(
                llm,
                tokenizer,
                comparison_prompts,
                legal,
                steps=3,
                cache_salts=[
                    f"eager-reference-n3-row-{row}" for row in range(len(comparison_prompts))
                ],
            )
        _print_comparison(
            isolated,
            batched_outputs[0],
            label="cross-B N=3 request=0",
            prompt_length=len(staged_prompt),
        )
        print(
            "NPU V1 E2E: B=1/N=3, duplicate B=2/N=2 and "
            "different-prompt B=2/N=3 functional checks passed",
            flush=True,
        )

    if not graph:
        return
    # Keep native compilation/graph defaults enabled, disabling only Beam
    # FULL. This is NOT an enforce_eager reference: Decode may be PIECEWISE.
    config["beam"].update(graph_enabled=False, graph_batch_size=2)
    with GRLLM(**engine_kwargs, enforce_eager=False) as llm:
        native_before = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
        native_outputs, native_trace = _run_traced_case(
            llm, tokenizer, comparison_prompts, legal, "native-no-beam-full-n3"
        )
        native_after = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
        print(f"NPU native control dispatch: stats={native_after}", flush=True)
        _diagnose_batch(
            tokenizer,
            comparison_prompts,
            batched_outputs,
            native_outputs,
            "eager/native control B=2 N=3",
        )
        assert not native_after["beam_graph_enabled"]
        assert native_after["full_dispatches"] == native_before["full_dispatches"] == 0
        assert native_after["fallbacks"] - native_before["fallbacks"] == 2
        assert (
            native_after["fallback_reasons"].get("graph_disabled", 0)
            - native_before["fallback_reasons"].get("graph_disabled", 0)
            == 2
        )
        for event in native_trace:
            forward = event
            assert forward["compilation_mode"] == "VLLM_COMPILE"
            assert forward["enforce_eager"] is False
            if event["stage"]:
                assert forward["runtime_mode"] in ("NONE", "PIECEWISE")

    config["beam"].update(graph_enabled=True, graph_batch_size=2)
    with GRLLM(**engine_kwargs, enforce_eager=False) as llm:
        stats = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
        assert [2, _BEAM_WIDTH] in [list(shape) for shape in stats["captured_shapes"]]
        signature = llm.collective_rpc("onerec_npu_graph_context_signature")[0]
        for index, batch_size in enumerate((2, 1, 2)):
            prompts = comparison_prompts[:batch_size]
            if batch_size == 2:
                outputs, graph_trace = _run_traced_case(
                    llm, tokenizer, prompts, legal, f"graph-switch-{index}"
                )
                assert all(event["runtime_mode"] == "FULL" for event in graph_trace[1:]), (
                    "B=2 Decode must retain Beam FULL replay"
                )
            else:
                outputs = _run_case(
                    llm,
                    tokenizer,
                    prompts,
                    legal,
                    steps=3,
                    cache_salts=[f"graph-switch-{index}-row-{row}" for row in range(batch_size)],
                )
            current = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
            print(
                f"NPU graph dispatch: iteration={index}; B={batch_size}; stats={current}",
                flush=True,
            )
            if batch_size == 2:
                _diagnose_batch(
                    tokenizer,
                    prompts,
                    batched_outputs,
                    outputs,
                    f"eager/graph B=2 N=3 W={_BEAM_WIDTH} iteration={index}",
                )
            expected_full = 2 if batch_size == 2 else 0
            assert current["full_dispatches"] - stats["full_dispatches"] == expected_full
            assert current["fallbacks"] - stats["fallbacks"] == (0 if batch_size == 2 else 2)
            if batch_size == 1:
                assert (
                    current["fallback_reasons"].get("shape_miss", 0)
                    - stats["fallback_reasons"].get("shape_miss", 0)
                    == 2
                )
            else:
                if index == 0:
                    graph_repeat, _ = _run_traced_case(
                        llm, tokenizer, prompts, legal, "graph-repeat-n3"
                    )
                    _diagnose_batch(
                        tokenizer,
                        prompts,
                        outputs,
                        graph_repeat,
                        "graph repeat B=2 N=3",
                    )
                    _diagnose_batch(
                        tokenizer,
                        prompts,
                        native_outputs,
                        outputs,
                        "native control/graph B=2 N=3",
                    )
                    repeated_stats = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
                    print(f"NPU graph repeat dispatch: stats={repeated_stats}", flush=True)
                    assert repeated_stats["full_dispatches"] - current["full_dispatches"] == 2
                    assert repeated_stats["fallbacks"] == current["fallbacks"]
                    current = repeated_stats
                    _assert_batch_equal(batched_outputs, eager_repeat, "eager repeat")
                    _assert_batch_equal(outputs, graph_repeat, "graph repeat")
                _assert_batch_equal(batched_outputs, outputs, f"eager/graph iteration={index}")
                _assert_batch_equal(native_outputs, outputs, f"native/graph iteration={index}")
            assert llm.collective_rpc("onerec_npu_graph_context_signature")[0] == signature
            print(
                f"NPU graph switch: B={batch_size}, stats={current}, captured addresses stable",
                flush=True,
            )
            stats = current
        _run_case(
            llm,
            tokenizer,
            comparison_prompts,
            legal,
            steps=1,
            cache_salts=["graph-prefill-only-row-0", "graph-prefill-only-row-1"],
        )
        current = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
        assert current["full_dispatches"] == stats["full_dispatches"]
        assert current["fallbacks"] == stats["fallbacks"]
        print("NPU graph correctness: B=2 -> B=1 -> B=2 and N=1 passed", flush=True)


def _b4_test_environment(env: Mapping[str, str]) -> dict[str, str]:
    result = dict(env)
    for key in (
        "VLLM_GR_TEST_PADDED_PREFILL",
        "VLLM_GR_PREFILL_GRAPH_KV_BOUND",
        "VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS",
    ):
        result.pop(key, None)
    result.update(
        VLLM_GR_TARGET_DEVICE="npu",
        VLLM_ENABLE_PREFILL_CUDAGRAPH="1",
        VLLM_GR_PREFILL_GRAPH_BUCKETS="512",
        VLLM_GR_PREFILL_GRAPH_BUCKET_STEP="128",
        VLLM_GR_PREFILL_GRAPH_MAX_BUCKET="512",
        VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS="4",
        VLLM_GR_NPU_FUSED_FFN="0",
        VLLM_ASCEND_ENABLE_NZ="0",
        VLLM_USE_BREAKABLE_CUDAGRAPH="0",
    )
    return result


def _run_b4_full_graph(work_dir: Path) -> None:
    from vllm_gr.entrypoints.gr import GRLLM

    model = os.environ.get("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")
    artifact = work_dir / "onerec_v1_npu_b4_constraints.safetensors"
    tokenizer, digest, tokenizer_digest, legal = _build_constraint_artifact(model, artifact)
    prompts = [_prompt_tokens(tokenizer, 127 + 32 * row)[:127] for row in range(4)]
    assert [len(prompt) for prompt in prompts] == [127] * 4
    assert len({tuple(prompt) for prompt in prompts}) == 4, "B=4 needs distinct prompt content"
    config: dict[str, Any] = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "worker_decision": True,
            "max_width": _BEAM_WIDTH,
            "max_decode_steps": 3,
            "graph_batch_size": 4,
        },
        "constraint_table": {
            "enabled": True,
            "backend": "npu",
            "max_top_k": _BEAM_WIDTH,
            "path": str(artifact),
            "format": "constraint_table_v1",
            "artifact_digest": digest,
            "tokenizer_digest": tokenizer_digest,
        },
    }
    engine_kwargs = dict(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        vllm_gr_config=config,
        async_scheduling=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=_BEAM_WIDTH,
        max_logprobs=_BEAM_WIDTH,
        block_size=128,
        gpu_memory_utilization=0.5,
        worker_extension_cls="tests.st.test_onerec_v1_npu_e2e.OneRecNPUGraphTestWorkerExtension",
    )
    reference = None
    results = []
    for graph in (False, True):
        config["beam"]["graph_enabled"] = graph
        with GRLLM(**engine_kwargs, enforce_eager=not graph) as llm:
            if graph:
                initial = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
                assert [4, _BEAM_WIDTH] in [list(shape) for shape in initial["captured_shapes"]]
                signature = llm.collective_rpc("onerec_npu_graph_context_signature")[0]
            for repeat in range(2):
                label = f"b4-{'full' if graph else 'eager'}-{repeat}"
                before = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
                outputs, events = _run_traced_case(
                    llm,
                    tokenizer,
                    prompts,
                    legal,
                    label,
                    record_cold=True,
                    exact_prefill=True,
                )
                _assert_b4_graph_trace(events, graph=graph)
                after = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
                assert after["full_dispatches"] - before["full_dispatches"] == (2 if graph else 0)
                assert after["fallbacks"] - before["fallbacks"] == (0 if graph else 2)
                reasons = {
                    key: value - before["fallback_reasons"].get(key, 0)
                    for key, value in after["fallback_reasons"].items()
                    if value != before["fallback_reasons"].get(key, 0)
                }
                assert reasons == ({} if graph else {"graph_disabled": 2})
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
                assert packed == reference, f"{label}: B=4 ordered tokens/scores/finish differ"
                if graph:
                    assert llm.collective_rpc("onerec_npu_graph_context_signature")[0] == signature
                results.append(dict(label=label, trace=events, before=before, after=after))
                print(f"NPU B=4 exact comparison passed: {label}; stats={after}", flush=True)
    (work_dir / "b4-full-graph-result.json").write_text(
        json.dumps(dict(model=model, beam_width=_BEAM_WIDTH, results=results), indent=2) + "\n"
    )


def _run_subprocess(tmp_path: Path, *, graph: bool = False, b4: bool = False) -> None:
    if b4:
        env = _b4_test_environment(os.environ)
    else:
        env = _graph_test_environment(os.environ) if graph else dict(os.environ)
    env.update(
        USE_TF="0",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    command = [sys.executable, "-m", "tests.st.test_onerec_v1_npu_e2e", str(tmp_path)]
    if graph:
        command.append("--graph")
    if b4:
        command.append("--b4")
    with subprocess.Popen(
        command,
        cwd=Path(__file__).parents[2],
        env=env,
        start_new_session=True,
    ) as process:
        try:
            assert process.wait(timeout=900 if b4 else 600) == 0, (
                "NPU Beam V1 E2E subprocess failed"
            )
        finally:
            if process.poll() is None:
                if os.name == "posix":
                    kill_process_group = getattr(os, "killpg")
                    sigkill = getattr(signal, "SIGKILL")
                    kill_process_group(process.pid, sigkill)
                else:
                    process.kill()
                process.wait()


@pytest.mark.npu_integration
@pytest.mark.slow
@pytest.mark.timeout(660)
def test_onerec_v1_npu_public_api_end_to_end(tmp_path: Path) -> None:
    _run_subprocess(tmp_path)


@pytest.mark.npu_integration
@pytest.mark.slow
@pytest.mark.timeout(660)
def test_onerec_v1_npu_graph_shape_switch(tmp_path: Path) -> None:
    _run_subprocess(tmp_path, graph=True)


@pytest.mark.npu_integration
@pytest.mark.slow
@pytest.mark.timeout(960)
def test_onerec_v1_npu_b4_full_graph(tmp_path: Path) -> None:
    _run_subprocess(tmp_path, b4=True)


if __name__ == "__main__":
    if "--b4" in sys.argv[2:]:
        _run_b4_full_graph(Path(sys.argv[1]))
    else:
        _run_npu_e2e(Path(sys.argv[1]), graph="--graph" in sys.argv[2:])
