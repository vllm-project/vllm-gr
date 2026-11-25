# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Compare native KDA/MLA Beam forks with independent native requests on CUDA."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from vllm import SamplingParams
from vllm.logprobs import Logprob
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

pytestmark = [pytest.mark.gpu_test, pytest.mark.model_test, pytest.mark.l3, pytest.mark.slow_test]


def _kimi_model_options() -> dict[str, Any]:
    if not torch.cuda.is_available():
        pytest.skip("Kimi Linear hybrid Beam test requires CUDA")
    model = os.environ.get("VLLM_GR_TEST_KIMI_LINEAR_MODEL")
    if not model:
        pytest.skip("Set VLLM_GR_TEST_KIMI_LINEAR_MODEL to a cached Kimi Linear 48B-A3B snapshot")
    options: dict[str, Any] = {
        "model": model,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.84,
        "attention_config": {"backend": "TRITON_MLA"},
    }
    if os.environ.get("VLLM_GR_TEST_KIMI_LINEAR_CPU_OFFLOAD") == "1":
        options.update(
            offload_group_size=4,
            offload_num_in_group=3,
            offload_params={"w13_weight", "w2_weight"},
        )
    return options


class _ForcePrefix(AdapterLogitsProcessor):  # type: ignore[misc]
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> Callable[[list[int], torch.Tensor], torch.Tensor]:
        assert params.extra_args is not None
        prefix = tuple(params.extra_args["prefix"])

        def force(output: list[int], logits: torch.Tensor) -> torch.Tensor:
            if len(output) < len(prefix):
                # The sampler gathers raw logprobs before forcing the token.
                logits.fill_(-float("inf"))
                logits[prefix[len(output)]] = 0
            return logits

        return force


def _score_native_branches(
    llm: Any, prompt: list[int], catalog: Sequence[tuple[int, ...]], width: int
) -> list[tuple[tuple[int, ...], float]]:
    from vllm.inputs import TokensPrompt

    ids = [token for triple in catalog for token in triple]
    window = list(range(min(ids), max(ids) + 1))
    prefixes = sorted({sid[:2] for sid in catalog})
    params = [
        SamplingParams(
            temperature=0,
            max_tokens=3,
            ignore_eos=True,
            logprob_token_ids=window,
            detokenize=False,
            extra_args={"prefix": list(prefix)},
        )
        for prefix in prefixes
    ]
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt) for _ in prefixes], params, use_tqdm=False
    )
    by_prefix: dict[tuple[int, ...], dict[int, Logprob]] = {}
    for prefix, output in zip(prefixes, outputs, strict=True):
        completion = output.outputs[0]
        assert tuple(completion.token_ids[:2]) == prefix
        for depth, logprobs in enumerate(completion.logprobs):
            by_prefix.setdefault(prefix[:depth], logprobs)

    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for depth in range(3):
        candidates: dict[tuple[int, ...], float] = {}
        for prefix, score in beams:
            logprobs = by_prefix[prefix]
            window_mass = torch.logsumexp(
                torch.tensor([logprobs[token].logprob for token in window]), dim=0
            ).item()
            for sid in catalog:
                if sid[:depth] == prefix:
                    candidates[sid[: depth + 1]] = (
                        score + logprobs[sid[depth]].logprob - window_mass
                    )
        beams = sorted(candidates.items(), key=lambda row: (-row[1], row[0]))[:width]
    return beams


@pytest.mark.timeout(1800)  # type: ignore[misc]
def test_kimi_linear_catalog_beam_matches_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_kwargs = _kimi_model_options()
    model_path = model_kwargs["model"]

    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    from vllm_gr.constraints.artifact_builder import build_single_trie_artifact
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # All beams branch from the same first token. Several second-token
    # branches also have multiple terminal children.
    catalog = [(10, b, c) for b in range(20, 25) for c in range(30, 33)]
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "triples": [
                    dict(zip("abc", tokenizer.convert_ids_to_tokens(sid), strict=True))
                    for sid in catalog
                ]
            }
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "table.safetensors"
    artifact_digest, tokenizer_digest = build_single_trie_artifact(
        Path(model_path), catalog_path, artifact
    )
    width = 4
    model_kwargs.update(
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        mamba_cache_mode="none",
        max_num_batched_tokens=8192,
        max_num_seqs=8,
    )
    with GRLLM(
        **model_kwargs,
        async_scheduling=True,
        gr={
            "beam_width": [width] * 3,
            "candidate_width": [width] * 3,
            "max_decode_steps": 3,
            "constraint_mode": "table",
            "constraint_table": {
                "format": "constraint_table_v1",
                "path": str(artifact),
                "backend": "cuda",
                "max_top_k": width,
                "digests": {"artifact": artifact_digest, "tokenizer": tokenizer_digest},
            },
        },
    ) as gr_llm:
        block_size = gr_llm.llm_engine.vllm_config.cache_config.block_size
        assert block_size > 2
        source = tokenizer.encode(
            "A short prompt for item recommendation.", add_special_tokens=False
        )
        lengths = (len(source), block_size - 1, block_size + 5)
        prompts = [[source[i % len(source)] for i in range(length)] for length in lengths]
        params = BeamSearchParams(beam_width=width, max_tokens=3, ignore_eos=True)
        observed = []
        for prompt in prompts:
            (result,) = gr_llm.beam_search(
                [TokensPrompt(prompt_token_ids=prompt)], params, output_mode="compact"
            )
            observed.append([(seq.token_ids, seq.cumulative_logprob) for seq in result.sequences])

    native = LLM(**model_kwargs, logprobs_mode="raw_logprobs", logits_processors=[_ForcePrefix])
    try:
        expected_batches = [
            _score_native_branches(native, prompt, catalog, width) for prompt in prompts
        ]
        for prompt, actual, expected in zip(prompts, observed, expected_batches, strict=True):
            print(
                f"Kimi Linear hybrid Beam prompt_tokens={len(prompt)} results={actual} oracle={expected}"
            )
        for actual, expected in zip(observed, expected_batches, strict=True):
            assert actual[0][0] == expected[0][0]
            assert [score for _, score in actual] == sorted(
                (score for _, score in actual), reverse=True
            )
            # The Beam and native requests use different BF16 batch shapes.
            # Compare scores by SID so close ranks can swap within tolerance.
            assert len(actual) == width
            assert dict(actual) == pytest.approx(dict(expected), abs=0.65)
    finally:
        native.llm_engine.engine_core.shutdown()


@pytest.mark.parametrize("runner_v2", [False, True])  # type: ignore[misc]
@pytest.mark.timeout(1800)  # type: ignore[misc]
def test_kimi_linear_batched_dynamic_trie_graph_and_prefix_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner_v2: bool
) -> None:
    model_kwargs = _kimi_model_options()
    model = model_kwargs["model"]

    from transformers import AutoConfig, AutoTokenizer
    from vllm.inputs import TokensPrompt
    from vllm.v1.metrics.reader import Counter

    from vllm_gr.constraints.artifact import save_constraint_table_artifact
    from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
    from vllm_gr.constraints.json_compiler import compile_relations_json
    from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(runner_v2)))
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    catalogs = [
        [
            (offset + a, offset + b, offset + c)
            for a in (10, 11)
            for b in range(20, 23)
            for c in range(30, 33)
        ]
        for offset in (0, 40)
    ]
    trie_documents = [
        compile_relations_json([tokenizer.convert_ids_to_tokens(sid) for sid in catalog])
        for catalog in catalogs
    ]
    tokenizer_digest = compute_tokenizer_digest(tokenizer)
    artifact = tmp_path / "multitrie.safetensors"
    table = ConstraintTableArtifactCompiler(
        tokenizer, model_config.vocab_size, tokenizer_digest=tokenizer_digest
    ).compile_tries(trie_documents)
    artifact_digest = save_constraint_table_artifact(table, artifact)
    params = BeamSearchParams(
        beam_width=8,
        dynamic_beam_widths=[4, 6, 3],
        per_parent_beam_widths=[4, 3, 2],
        result_width=2,
        max_tokens=3,
        ignore_eos=True,
    )

    with GRLLM(
        **model_kwargs,
        enforce_eager=False,
        async_scheduling=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        mamba_cache_mode="align",
        max_num_batched_tokens=4096,
        max_num_seqs=16,
        disable_log_stats=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16],
        },
        gr={
            "beam_width": [8] * 3,
            "candidate_width": [8] * 3,
            "max_decode_steps": 3,
            "constraint_mode": "table",
            "constraint_table": {
                "format": "constraint_table_v2",
                "path": str(artifact),
                "backend": "cuda",
                "max_top_k": 8,
                "digests": {"artifact": artifact_digest, "tokenizer": tokenizer_digest},
            },
        },
    ) as llm:
        block_size = llm.llm_engine.vllm_config.cache_config.block_size
        prompts = [
            TokensPrompt(prompt_token_ids=[71, 72]),
            TokensPrompt(prompt_token_ids=[46] * (block_size - 1) + [47, 48]),
        ]

        def cache_hits() -> int:
            return sum(
                int(metric.value)
                for metric in llm.get_metrics()
                if isinstance(metric, Counter) and metric.name == "vllm:prefix_cache_hits"
            )

        for repetition in range(2):
            hits_before = cache_hits()
            results = llm.beam_search(prompts, params, trie_ids=[1, 0], output_mode="compact")
            assert len(results) == 2
            for output, permitted in zip(results, catalogs[::-1], strict=True):
                assert len(output.sequences) == 2
                assert all(sequence.token_ids in permitted for sequence in output.sequences)
            if repetition:
                assert cache_hits() >= hits_before + block_size
        (stats,) = llm.collective_rpc("get_gr_beam_graph_runtime_stats")
        assert stats["full_dispatches"] == 4
        assert {8, 12}.issubset(stats["captured_widths"])
        print(f"Kimi Linear dynamic CUDA graphs {stats} and prefix hits {cache_hits()}")
