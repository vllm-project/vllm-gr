# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-any-return"
"""Public Beam Search V1 end-to-end coverage on a real Ascend NPU.

The model is loaded once for three focused cases: an isolated B=1 baseline,
duplicate requests at B=2/N=2, and different requests at B=2/N=3. Nothing
below the public ``GRLLM.beam_search_v1`` entry point is replaced or mocked.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any, cast

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"torch-npu is not installed: {exc}", allow_module_level=True)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is unavailable", allow_module_level=True)

from vllm_gr.ops.npu import beam_ops

if not beam_ops.is_available():
    pytest.skip(
        f"xllm Beam operators are unavailable: {beam_ops.unavailable_reason()}",
        allow_module_level=True,
    )

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


def _build_constraint_artifact(
    model: str,
    artifact: Path,
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
        for a, b, c in product(range(_BEAM_WIDTH), range(4), range(4))
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
    outputs = llm.beam_search_v1(
        request_prompts,
        params,
    )
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
        f"NPU V1 diagnostic: {label}; "
        f"ordered_equal={reference_tokens == candidate_tokens}; "
        f"set_equal={set(reference_tokens) == set(candidate_tokens)}; "
        f"common={len(common)}/{_BEAM_WIDTH}; max_score_delta={max_score_delta:.6f}",
        flush=True,
    )


def _run_npu_e2e(work_dir: Path) -> None:
    from vllm_gr.entrypoints.gr import GRLLM

    model = os.environ.get("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")
    tokenizer, digest, tokenizer_digest, legal = _build_constraint_artifact(
        model,
        work_dir / "onerec_v1_npu_constraints.safetensors",
    )
    config = {
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

    with GRLLM(
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
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    ) as llm:
        comparison_prompts = [_prompt_tokens(tokenizer, length) for length in (128, 160)]
        staged_prompt = comparison_prompts[0]
        isolated = _run_case(llm, tokenizer, [staged_prompt], legal, steps=3)[0]

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

        batched_outputs = _run_case(llm, tokenizer, comparison_prompts, legal, steps=3)
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


@pytest.mark.npu_integration
@pytest.mark.slow
@pytest.mark.timeout(660)
def test_onerec_v1_npu_public_api_end_to_end(tmp_path: Path) -> None:
    env = dict(os.environ)
    env.update(
        USE_TF="0",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    command = [sys.executable, "-m", "tests.st.test_onerec_v1_npu_e2e", str(tmp_path)]
    with subprocess.Popen(
        command,
        cwd=Path(__file__).parents[2],
        env=env,
        start_new_session=True,
    ) as process:
        try:
            assert process.wait(timeout=600) == 0, "NPU Beam V1 E2E subprocess failed"
        finally:
            if process.poll() is None:
                if os.name == "posix":
                    kill_process_group = getattr(os, "killpg")
                    sigkill = getattr(signal, "SIGKILL")
                    kill_process_group(process.pid, sigkill)
                else:
                    process.kill()
                process.wait()


if __name__ == "__main__":
    _run_npu_e2e(Path(sys.argv[1]))
