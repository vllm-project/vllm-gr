# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc,no-untyped-call,no-untyped-def"

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from benchmarks.open_one_rec.generate_native_beam_golden import (
    native_vllm_beam_search,
)
from benchmarks.open_one_rec.open_one_rec_loader import BaseLoader
from benchmarks.open_one_rec.regression import (
    RecommendationMetricAccumulator,
    compare_jsonl,
    comparison_passed,
    load_sample_manifest,
    sha256_file,
    validate_generations,
    write_jsonl,
)
from benchmarks.open_one_rec.run_recommendation_regression import (
    _online_batch,
    _resolve_constraint,
)
from benchmarks.open_one_rec.tasks.recommendation.utils_by_pid import (
    apply_sid_to_pid_strategy,
)
from benchmarks.open_one_rec.tools.prepare_decode_data import prepare_decode_data

SID1 = "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"
SID2 = "<|sid_begin|><s_a_1><s_b_2><s_c_4><|sid_end|>"


def test_online_regression_sends_rendered_prompt_verbatim(monkeypatch) -> None:
    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"text": SID1}, {"text": SID2}]}

    def post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return Response()

    import requests

    monkeypatch.setattr(requests, "post", post)
    result = _online_batch(
        prompts=["<|im_start|>system\nrendered"],
        model="model",
        host="127.0.0.1",
        port=7012,
        endpoint="/v1/completions",
        api_key="token",
        beam_width=2,
        temperature=0.0,
        timeout=10,
    )

    assert calls[0][0] == "http://127.0.0.1:7012/v1/completions"
    assert calls[0][2]["prompt"] == "<|im_start|>system\nrendered"
    assert "messages" not in calls[0][2]
    assert result[0]["generations"] == [SID1, SID2]


def test_native_golden_calls_unmodified_vllm_beam_search() -> None:
    class FakeTokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            return {"<|sid_begin|>": 1, "<|sid_end|>": 2}[token]

        def decode(self, token_ids, **kwargs) -> str:
            return ",".join(map(str, token_ids))

    class FakeLLM:
        def __init__(self):
            self.call = None

        def beam_search(self, prompts, params, use_tqdm, concurrency_limit):
            self.call = (prompts, params, use_tqdm, concurrency_limit)
            return [
                SimpleNamespace(
                    sequences=[
                        SimpleNamespace(tokens=[99, 1, 20, 40, 60], cum_logprob=-0.4),
                        SimpleNamespace(tokens=[99, 1, 10, 30, 50], cum_logprob=-2.3),
                    ]
                )
            ]

    llm = FakeLLM()
    result = native_vllm_beam_search(
        llm=llm,
        tokenizer=FakeTokenizer(),
        prompts=["prompt"],
        beam_width=2,
        temperature=0.0,
        max_tokens=3,
    )[0]

    prompts, params, use_tqdm, concurrency_limit = llm.call
    assert prompts == [{"prompt": "prompt<|sid_begin|>"}]
    assert params.beam_width == 2
    assert params.max_tokens == 3
    assert params.temperature == 0.0
    assert use_tqdm is False
    assert concurrency_limit == 1
    assert result["generation_token_ids"] == [
        [1, 20, 40, 60, 2],
        [1, 10, 30, 50, 2],
    ]
    assert result["cumulative_logprobs"] == [-0.4, -2.3]


def test_native_golden_rejects_non_onerec_generation_depth() -> None:
    with pytest.raises(ValueError, match="exactly 3 semantic tokens"):
        native_vllm_beam_search(
            llm=object(),
            tokenizer=object(),
            prompts=["prompt"],
            beam_width=32,
            temperature=0.0,
            max_tokens=5,
        )


def test_explicit_sample_ids_preserve_order_and_never_cache(monkeypatch) -> None:
    loader = BaseLoader(
        task_config={"name": "video", "size": 3},
        data_dir="unused",
        tokenizer=object(),
    )
    frame = pd.DataFrame({"messages": ["a", "b", "c"], "metadata": ["x", "y", "z"]})
    monkeypatch.setattr(loader, "_load_dataframe", lambda split: frame)
    monkeypatch.setattr(loader, "_process_dataframe", lambda selected: list(selected.index))
    monkeypatch.setattr(
        loader,
        "_load_sample_dataframe",
        lambda *args: pytest.fail("explicit-ID load consulted the random-sample cache"),
    )
    monkeypatch.setattr(
        loader,
        "_save_sample_data",
        lambda *args: pytest.fail("explicit-ID load wrote into the dataset"),
    )

    assert loader.load_data(sample_ids=[2, 0]) == [2, 0]

    with pytest.raises(ValueError, match="duplicates"):
        loader.load_data(sample_ids=[1, 1])
    with pytest.raises(ValueError, match="not present"):
        loader.load_data(sample_ids=[9])
    with pytest.raises(ValueError, match="mutually exclusive"):
        loader.load_data(sample_size=1, sample_ids=[0])


def test_manifest_validation(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 1, "tasks": {"video": [2, 0]}}))
    assert load_sample_manifest(path) == {"video": [2, 0]}

    path.write_text(json.dumps({"version": 1, "tasks": {"video": [1, 1]}}))
    with pytest.raises(ValueError, match="duplicate"):
        load_sample_manifest(path)

    path.write_text(json.dumps({"tasks": {"video": [1]}}))
    with pytest.raises(ValueError, match="version 1"):
        load_sample_manifest(path)


def test_manifest_pins_the_source_parquet(tmp_path) -> None:
    dataset = tmp_path / "data"
    (dataset / "video").mkdir(parents=True)
    parquet = dataset / "video" / "video_test.parquet"
    parquet.write_bytes(b"frozen parquet bytes")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "source_sha256": {"video": sha256_file(parquet)},
                "tasks": {"video": [0]},
            }
        )
    )
    assert load_sample_manifest(manifest, dataset_path=dataset) == {"video": [0]}

    parquet.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source mismatch"):
        load_sample_manifest(manifest, dataset_path=dataset)


def test_pid_popularity_tie_is_stable() -> None:
    values = [
        {"pid": 17, "count": 100, "count_after_downsample": 3},
        {"pid": 99, "count": 10, "count_after_downsample": 3},
    ]
    assert apply_sid_to_pid_strategy(values, "most_popular_after_downsampling") == 17


def test_streaming_recommendation_metrics(tmp_path) -> None:
    mapping = {
        str(1 * 8192 * 8192 + 2 * 8192 + 3): [{"pid": 10, "count": 4, "count_after_downsample": 4}],
        str(1 * 8192 * 8192 + 2 * 8192 + 4): [{"pid": 20, "count": 4, "count_after_downsample": 4}],
    }
    (tmp_path / "sid2pid.json").write_text(json.dumps(mapping))
    accumulator = RecommendationMetricAccumulator("video", tmp_path)
    metrics = accumulator.add(
        [SID1, SID2],
        f"{SID1}{SID2}",
        {"answer_pid": [10, 30]},
    )

    assert metrics["pass@1"] is True
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@32"] == 1.0
    assert metrics["pid_pass@1"] is True
    assert metrics["pid_recall@1"] == 0.5
    assert accumulator.summary()["total_samples"] == 1


def test_generation_contract() -> None:
    result = validate_generations([SID1, SID1, "bad"], beam_width=3)
    assert result == {
        "beam_count": 3,
        "beam_count_ok": True,
        "empty_count": 0,
        "invalid_sid_count": 1,
        "duplicate_sid_count": 1,
    }


def test_jsonl_comparison_reports_semantic_difference(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(
        baseline,
        [
            {
                "task": "video",
                "sample_id": "7",
                "prompt_sha256": "same-prompt",
                "generations": [SID1],
                "metrics": {"pass@1": True},
            }
        ],
        mode="w",
    )
    write_jsonl(
        candidate,
        [
            {
                "task": "video",
                "sample_id": "7",
                "prompt_sha256": "same-prompt",
                "generations": [SID2],
                "metrics": {"pass@1": False},
            }
        ],
        mode="w",
    )

    result = compare_jsonl(baseline, candidate)
    assert result["top1_mismatches"] == 1
    assert result["per_sample_metric_mismatches"] == 1
    assert result["aggregate_metric_regressions"] == ["pass@1"]
    assert not comparison_passed(result)
    assert not comparison_passed(result, policy="no-regression")

    improved = compare_jsonl(candidate, baseline)
    assert improved["aggregate_metrics"]["pass@1"]["delta"] == 1.0
    assert not comparison_passed(improved, policy="exact")
    assert comparison_passed(improved, policy="no-regression")


def test_no_regression_policy_supports_explicit_diagnostic_tolerance() -> None:
    comparison = {
        "aggregate_metrics": {
            "recall@32": {"baseline": 0.04, "candidate": 0.035},
            "pid_recall@32": {"baseline": 0.15, "candidate": 0.135},
        }
    }
    assert comparison_passed(
        comparison,
        policy="no-regression",
        absolute_tolerance=0.005,
        relative_tolerance=0.1,
        metric_names=["recall@32", "pid_recall@32"],
    )
    comparison["aggregate_metrics"]["pid_recall@32"]["candidate"] = 0.134
    assert not comparison_passed(
        comparison,
        policy="no-regression",
        absolute_tolerance=0.005,
        relative_tolerance=0.1,
        metric_names=["recall@32", "pid_recall@32"],
    )
    assert not comparison_passed(comparison, policy="no-regression", metric_names=["missing@32"])


def test_ci_style_strict_no_regression_rejects_any_aggregate_drop() -> None:
    small_drop = {
        "aggregate_metrics": {
            "recall@32": {
                "baseline": 0.06590873015873017,
                "candidate": 0.06465873015873017,
            }
        }
    }
    improvement = {
        "aggregate_metrics": {
            "recall@32": {
                "baseline": 0.06590873015873017,
                "candidate": 0.066,
            }
        }
    }

    assert not comparison_passed(
        small_drop,
        policy="no-regression",
        metric_names=["recall@32"],
    )
    assert comparison_passed(
        improvement,
        policy="no-regression",
        metric_names=["recall@32"],
    )


def test_jsonl_comparison_rejects_prompt_mismatch_and_duplicate_rows(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    record = {
        "task": "video",
        "sample_id": "7",
        "prompt_sha256": "baseline-prompt",
        "generations": [SID1],
        "metrics": {"pass@1": True},
    }
    write_jsonl(baseline, [record, record], mode="w")
    write_jsonl(candidate, [{**record, "prompt_sha256": "candidate-prompt"}], mode="w")

    result = compare_jsonl(baseline, candidate)
    assert result["duplicate_baseline"] == [["video", "7"]]
    assert result["prompt_mismatches"] == 1
    assert not comparison_passed(result, policy="no-regression")


def test_prepared_manifest_resolves_all_constraint_modes() -> None:
    manifest = {
        "tasks": {"video": "video"},
        "domains": {
            "video": {
                "catalog_path": "/catalog.json",
                "catalog_sha256": "catalog-digest",
                "triples_path": "/triples.json",
                "triples_sha256": "triples-digest",
                "artifact": {
                    "path": "/table.safetensors",
                    "artifact_digest": "sha256:artifact",
                    "tokenizer_digest": "sha256:tokenizer",
                },
            }
        },
    }

    trie_args, trie_identity = _resolve_constraint(manifest, "video", "trie", "cuda")
    assert trie_args == {"catalog_path": "/catalog.json", "constraint_backend": "trie"}
    assert trie_identity["digest"] == "catalog-digest"

    triples_args, triples_identity = _resolve_constraint(manifest, "video", "triples", "cuda")
    assert triples_args["constraint_backend"] == "constraint_table"
    assert triples_identity["digest"] == "triples-digest"

    artifact_args, artifact_identity = _resolve_constraint(manifest, "video", "artifact", "cuda")
    assert artifact_args["vllm_gr_config"]["constraint_table"]["path"] == ("/table.safetensors")
    assert artifact_identity["digest"] == "sha256:artifact"


def test_prepare_decode_data_builds_two_shared_universes(tmp_path) -> None:
    dataset = tmp_path / "benchmark_data"
    dataset.mkdir()
    video_key = 1 * 8192 * 8192 + 2 * 8192 + 3
    product_key = 4 * 8192 * 8192 + 5 * 8192 + 6
    (dataset / "sid2pid.json").write_text(json.dumps({str(video_key): []}))
    (dataset / "sid2iid.json").write_text(json.dumps({str(product_key): []}))

    manifest = prepare_decode_data(dataset_path=dataset, output_dir=tmp_path / "prepared")

    assert manifest["tasks"] == {
        "video": "video",
        "ad": "video",
        "label_cond": "video",
        "interactive": "video",
        "product": "product",
    }
    for domain in ("video", "product"):
        info = manifest["domains"][domain]
        assert info["source_count"] == 1
        assert info["catalog_count"] == 1
        assert info["triples_count"] == 1
        assert len(info["logical_universe_sha256"]) == 64
        with open(info["triples_path"], encoding="utf-8") as file_obj:
            assert len(json.load(file_obj)["triples"]) == 1

    video_catalog = manifest["domains"]["video"]["catalog_path"]
    with open(video_catalog, "w", encoding="utf-8") as file_obj:
        json.dump(
            [["<|sid_begin|>", "<s_a_7>", "<s_b_8>", "<s_c_9>", "<|sid_end|>"]],
            file_obj,
        )
    with pytest.raises(ValueError, match="logical SID universe mismatch"):
        prepare_decode_data(dataset_path=dataset, output_dir=tmp_path / "prepared")


def test_triples_only_prepare_preserves_matching_artifact_metadata(tmp_path) -> None:
    dataset = tmp_path / "benchmark_data"
    dataset.mkdir()
    (dataset / "sid2pid.json").write_text(json.dumps({"1": []}))
    (dataset / "sid2iid.json").write_text(json.dumps({"2": []}))
    output = tmp_path / "prepared"
    manifest = prepare_decode_data(dataset_path=dataset, output_dir=output)

    for domain in ("video", "product"):
        artifact = output / domain / "constraint_table.safetensors"
        artifact.write_bytes(f"{domain}-artifact".encode())
        manifest["domains"][domain]["artifact"] = {
            "path": "/stale/cache/workspace/constraint_table.safetensors",
            "sha256": sha256_file(artifact),
            "artifact_digest": f"sha256:{domain}",
            "tokenizer_digest": "sha256:tokenizer",
            "vocab_size": 123,
            "model": "model",
            "revision": "revision",
        }
    (output / "manifest.json").write_text(json.dumps(manifest))

    refreshed = prepare_decode_data(dataset_path=dataset, output_dir=output)
    for domain in ("video", "product"):
        artifact_info = refreshed["domains"][domain]["artifact"]
        assert artifact_info["path"] == str(
            output.resolve() / domain / "constraint_table.safetensors"
        )
        assert artifact_info["artifact_digest"] == f"sha256:{domain}"


def test_golden_identity_blocks_mismatched_baseline(tmp_path) -> None:
    """A reused golden from different model/data/beam settings must not gate."""
    from benchmarks.open_one_rec.regression import validate_golden_identity

    identity = {
        "model": "OpenOneRec/OneRec-1.7B",
        "revision": "abc",
        "implementation_revision": "native-llm-beam-search",
        "source_sha256": "digest",
        "execution": "offline",
        "beam_width": 32,
        "temperature": 0.0,
    }
    base = tmp_path / "base.jsonl"
    cand = tmp_path / "cand.jsonl"
    base.write_text(json.dumps({"run_identity": identity}) + "\n", encoding="utf-8")
    cand.write_text(json.dumps({"run_identity": dict(identity)}) + "\n", encoding="utf-8")

    result = validate_golden_identity(base, cand)
    assert result["mismatches"] == []

    # A weaker baseline (beam width 1) must be rejected outright.
    cand.write_text(
        json.dumps({"run_identity": dict(identity, beam_width=1)}) + "\n",
        encoding="utf-8",
    )
    result = validate_golden_identity(base, cand)
    assert any("beam_width" in mismatch for mismatch in result["mismatches"])

    # Implementation / execution-variant / constraint differences are the
    # intentional subject of the paired differential and must be ignored.
    variant = dict(identity)
    variant.update(
        implementation_revision="vllm-gr",
        execution_variant="gr-grouped-beam",
        constraint={"mode": "constraint-table"},
    )
    cand.write_text(json.dumps({"run_identity": variant}) + "\n", encoding="utf-8")
    assert validate_golden_identity(base, cand)["mismatches"] == []


def test_golden_identity_requires_identity_records(tmp_path) -> None:
    """Golden files without a run_identity cannot be identified at all."""
    from benchmarks.open_one_rec.regression import validate_golden_identity

    base = tmp_path / "base.jsonl"
    cand = tmp_path / "cand.jsonl"
    base.write_text(json.dumps({"sample_id": 0}) + "\n", encoding="utf-8")
    cand.write_text(
        json.dumps({"run_identity": {"model": "m"}, "sample_id": 0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no run_identity"):
        validate_golden_identity(base, cand)


def test_golden_identity_allows_offline_golden_vs_online_candidate(tmp_path) -> None:
    """The paired differential compares offline native goldens against online
    vllm-gr candidates: execution topology is an intentional difference and
    must never block the gate."""
    from benchmarks.open_one_rec.regression import validate_golden_identity

    native = {
        "model": "OpenOneRec/OneRec-1.7B",
        "revision": "abc",
        "implementation_revision": "native-llm-beam-search",
        "execution_variant": "native-llm-beam-search",
        "source_sha256": "digest",
        "execution": "offline",
        "beam_width": 32,
        "temperature": 0.0,
        "evaluation_mode": "both",
    }
    gr = {
        "model": "OpenOneRec/OneRec-1.7B",
        "revision": "abc",
        "implementation_revision": "vllm-gr",
        "execution_variant": "gr-grouped-beam",
        "source_sha256": "digest",
        "execution": "online",
        "beam_width": 32,
        "temperature": 0.0,
        "evaluation_mode": "both",
    }
    base = tmp_path / "base.jsonl"
    cand = tmp_path / "cand.jsonl"
    base.write_text(json.dumps({"run_identity": native}) + "\n", encoding="utf-8")
    cand.write_text(json.dumps({"run_identity": gr}) + "\n", encoding="utf-8")

    result = validate_golden_identity(base, cand)
    assert result["mismatches"] == []


def test_offline_strict_mode_canonicalizes_generated_tail() -> None:
    """Offline strict runs must satisfy the standalone-SID output contract.

    GRLLM returns beam.text as prompt+generated when prompt decoding is on, so
    the offline collector canonicalizes via the final five token ids (SID
    sentinels included) -- never via the raw text.
    """
    from benchmarks.open_one_rec.run_recommendation_regression import (
        _offline_batch,
        validate_generations,
    )

    token_text = {
        1: "<|sid_begin|>",
        101: "<s_a_7>",
        102: "<s_b_8>",
        103: "<s_c_9>",
        2: "<|sid_end|>",
    }

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return "".join(token_text[int(i)] for i in token_ids)

    class FakeSequence:
        # Prompt ids end with the pre-calculated <|sid_begin|>; the worker
        # emits the three semantic ids; finalize appends <|sid_end|.
        def __init__(self, tokens):
            self.tokens = tokens
            self.cum_logprob = -0.5

    class FakeOutput:
        def __init__(self):
            self.sequences = [FakeSequence([40, 41, 1, 101, 102, 103, 2]) for _ in range(4)]

    class FakeLLM:
        def beam_search(self, prompts, params, concurrency_limit):
            assert params.begin_token == "<|sid_begin|>"
            assert params.end_token == "<|sid_end|>"
            return [FakeOutput() for _ in prompts]

    normalized = _offline_batch(FakeLLM(), FakeTokenizer(), ["prompt-a", "prompt-b"], 4, 0.0)
    assert len(normalized) == 2
    for sample in normalized:
        check = validate_generations(sample["generations"], beam_width=4)
        assert check["invalid_sid_count"] == 0
        assert check["beam_count_ok"] is True
        # The canonicalized text is the standalone SID the strict gate wants.
        assert sample["generations"][0] == "<|sid_begin|><s_a_7><s_b_8><s_c_9><|sid_end|>"


def test_offline_prompt_text_would_fail_strict_contract() -> None:
    """Guard the reason canonicalization exists: prompt+generated text cannot
    satisfy the standalone-SID strict contract the offline gate enforces."""
    from benchmarks.open_one_rec.run_recommendation_regression import (
        validate_generations,
    )

    rendered_prompt = "<|im_start|>user\nhistory <s_a_1><s_b_2><s_c_3><|im_end|>"
    gen = "<|sid_begin|><s_a_7><s_b_8><s_c_9><|sid_end|>"
    check = validate_generations([rendered_prompt + gen], beam_width=4)
    assert check["invalid_sid_count"] == 1
    assert check["empty_count"] == 0
