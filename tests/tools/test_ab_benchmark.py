# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.metrics import distribution_ms
from benchmarks.open_one_rec import offline_ab
from tools.AB_Tests import compare_ab, generate_summary

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize("sequences", [[], [[1, 2, 3]], [[1, 2, 3], [1, 2]]])  # type: ignore[misc]
def test_incomplete_beam_response_is_rejected(sequences: list[list[int]]) -> None:
    llm = SimpleNamespace(
        beam_search=lambda *args, **kwargs: [
            SimpleNamespace(sequences=[SimpleNamespace(text="sid", tokens=s) for s in sequences])
        ]
    )
    with pytest.raises(ValueError, match="beam|token"):
        offline_ab.measure_e2e(llm, [1], SimpleNamespace(beam_width=2, max_tokens=2))


def test_historical_percentile_algorithms() -> None:
    assert distribution_ms([3.0, 1.0]) == {
        "mean": 2.0,
        "std": 1.0,
        "p50": 2.0,
        "p90": pytest.approx(2.8),
        "p95": pytest.approx(2.9),
        "p99": pytest.approx(2.98),
        "unit": "ms",
    }
    assert generate_summary.distribution_ms([1.0, 3.0])["p50"] == 1.0


@pytest.mark.parametrize("sample", [0.0, -1.0, float("inf"), float("nan")])  # type: ignore[misc]
def test_distribution_requires_finite_positive_latencies(sample: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        distribution_ms([1.0, sample])


@pytest.mark.parametrize(  # type: ignore[misc]
    ("prompt_length", "queries", "hits", "expected"),
    [(31, 32, 16, 16), (1, 2, 0, 0), (31, 0, 0, None), (31, 64, 32, None), (1, 2, 3, None)],
)
def test_cache_reuse_is_observed_outside_timer(
    monkeypatch: pytest.MonkeyPatch,
    prompt_length: int,
    queries: int,
    hits: int,
    expected: int | None,
) -> None:
    events: list[str] = []
    totals = iter(((100, 32), (100 + queries, 32 + hits)))

    def metrics() -> list[SimpleNamespace]:
        events.append("metrics")
        queried, reused = next(totals)
        return [
            SimpleNamespace(name="vllm:prefix_cache_queries", value=queried),
            SimpleNamespace(name="vllm:prefix_cache_hits", value=reused),
        ]

    def clock() -> int:
        events.append("clock")
        return len(events)

    def beam_search(*args: Any, **kwargs: Any) -> list[SimpleNamespace]:
        events.append("beam")
        sequence = SimpleNamespace(text="sid", tokens=[1] * (prompt_length + 3))
        return [SimpleNamespace(sequences=[sequence])]

    monkeypatch.setattr(offline_ab.time, "perf_counter_ns", clock)
    llm = SimpleNamespace(get_metrics=metrics, beam_search=beam_search)
    params = SimpleNamespace(beam_width=1, max_tokens=3, begin_token="<|sid_begin|>")
    measured = offline_ab.measure_e2e(llm, [1] * prompt_length, params)
    assert measured["cached_tokens"] == expected
    assert measured["e2e_ms"] > 0
    assert events == ["metrics", "clock", "beam", "clock", "metrics"]


def test_unavailable_prefix_cache_counters_are_unknown() -> None:
    driver = offline_ab
    assert driver.prefix_cache_counters(SimpleNamespace()) is None
    hit = SimpleNamespace(name="vllm:prefix_cache_hits", value=16)
    query = SimpleNamespace(name="vllm:prefix_cache_queries", value=32)
    assert driver.prefix_cache_counters(SimpleNamespace(get_metrics=lambda: [hit])) is None
    duplicate = SimpleNamespace(get_metrics=lambda: [hit, query, hit])
    assert driver.prefix_cache_counters(duplicate) is None


def test_workload_digest_uses_actual_ordered_tokens() -> None:
    driver = offline_ab
    ids = ["video_4", "video_2"]
    selection = driver.selection_digests(ids, [[4, 5], [2, 3]])
    changed = driver.selection_digests(ids, [[4, 5], [2, 9]])
    assert selection["sample_ids_sha256"] == changed["sample_ids_sha256"]
    assert selection["workload_sha256"] != changed["workload_sha256"]
    reversed_ids = driver.selection_digests(ids[::-1], [[2, 3], [4, 5]])
    assert selection["sample_ids_sha256"] != reversed_ids["sample_ids_sha256"]


def test_cold_and_legacy_cached_parquet_use_same_source_rows(tmp_path: Path) -> None:
    import pandas as pd

    class Tokenizer:
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            return messages[0]["content"]

        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return list(text.encode())

        def __call__(self, text: str, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(input_ids=self.encode(text))

        def decode(self, tokens: list[int], **kwargs: Any) -> str:
            return bytes(tokens).decode()

    rows = [11, 18, 23, 47, 61, 88, 102]
    original = pd.DataFrame(
        {
            "messages": [json.dumps([{"role": "user", "content": f"{i:03d}"}]) for i in rows],
            "metadata": [json.dumps({"answer": str(i)}) for i in rows],
        },
        index=rows,
    )
    path = tmp_path / "video" / "video_test.parquet"
    path.parent.mkdir()
    original.to_parquet(path)
    cache = path.with_name("video_test_sample_3.parquet")
    tokenizer = Tokenizer()
    cold = offline_ab.sample_prompts(path, "video", tokenizer, 4, 3)
    wrote_cache = cache.exists()
    selected = original.sample(n=3, random_state=42)
    selected.to_parquet(cache, index=False)  # Historical cache discards the source row IDs.
    cached_bytes = cache.read_bytes()
    hot = offline_ab.sample_prompts(path, "video", tokenizer, 4, 3)
    assert cold == hot
    expected_ids = selected.index.tolist()
    random.Random(0).shuffle(expected_ids)
    assert cold[0] == [f"video_{i}" for i in expected_ids]
    assert cold[1] == [list(f"{i:03d}".encode()) for i in expected_ids]
    assert not wrote_cache and cache.read_bytes() == cached_bytes


def test_local_model_identity_tracks_weights(tmp_path: Path) -> None:
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"original")
    model, identity = offline_ab.resolve_model(str(tmp_path), None)
    assert model == tmp_path
    assert offline_ab.resolve_model(str(tmp_path), identity) == (model, identity)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        offline_ab.resolve_model(str(tmp_path), identity)


def _offline_cli_error(
    options: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> str:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "offline",
            "--output",
            "raw.json",
            "--beam-width",
            "128",
            "--input-length",
            "1024",
            *options,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        offline_ab.main()
    assert exc.value.code == 2
    stderr: str = capsys.readouterr().err
    return stderr.splitlines()[-1].partition(": error: ")[2]


def test_catalog_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = _offline_cli_error(["--data-dir", str(tmp_path)], monkeypatch, capsys)
    assert error == "the following arguments are required: --catalog"


@pytest.mark.parametrize("as_directory", [False, True], ids=["nonexistent", "directory"])  # type: ignore[misc]
def test_catalog_must_be_a_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_directory: bool,
) -> None:
    catalog = tmp_path / "triples.json"
    if as_directory:
        catalog.mkdir()
    error = _offline_cli_error(
        ["--data-dir", str(tmp_path), "--catalog", str(catalog)], monkeypatch, capsys
    )
    assert error == f"--catalog is not a regular file: {catalog}"


def test_data_dir_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = tmp_path / "triples.json"
    catalog.write_text("{}", encoding="utf-8")
    error = _offline_cli_error(["--catalog", str(catalog)], monkeypatch, capsys)
    assert error == "the following arguments are required: --data-dir"


def test_old_batched_raw_remains_readable() -> None:
    raw = {
        "num_prompts": 4,
        "completed": 4,
        "failed": 0,
        "batch_size": 2,
        "sample_unit": "logical-batch",
        "beam_width": 2,
        "input_length": 16,
        "warmup_requests": 1,
        "miss": {"e2e_ms": [1, 2]},
        "hit": {"e2e_ms": [1, 2]},
        "input_lens": [15] * 4,
        "output_lens": [20, 20],
    }
    args = SimpleNamespace(beam_width=2, input_length=16, warmup_requests=1)
    generate_summary.validate_offline_raw(raw, args, {})


def test_online_summary_keeps_unknown_provenance_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_query(*args: Any, **kwargs: Any) -> None:
        pytest.fail("report generation queried the local GPU without a UUID")

    monkeypatch.setattr(generate_summary.subprocess, "check_output", unexpected_query)
    raw: dict[str, Any] = {
        "date": "20260101-120000",
        "num_prompts": 1,
        "failed": 0,
        "completed": 1,
        "input_lens": [15],
        "output_lens": [10],
        "duration": 1.0,
        "total_input_tokens": 15,
        "total_output_tokens": 10,
        "request_throughput": 1,
        "output_throughput": 10,
        "total_token_throughput": 25,
        "model_id": "old-model",
        "backend": "http",
        "max_concurrency": 1,
        "request_rate": 1,
        "ttfts": [0.1],
    }
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for percentile in ("mean", "p50", "p90", "p95", "p99"):
            raw[f"{percentile}_{metric}_ms"] = 100.0
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "dataset",
                "revision": "5" * 40,
                "files": [
                    {"task": "video", "path": "/historical/video.parquet", "sha256": "d" * 64}
                ],
            }
        )
    )
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"prefix_cache": {"hit_rate_percent": 50}}')
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summary",
            "--raw-result",
            str(raw_path),
            "--dataset-manifest",
            str(manifest),
            "--output",
            str(output),
            "--run-id",
            "history",
            "--beam-width",
            "2",
            "--input-length",
            "16",
            "--warmup-requests",
            "0",
            "--git-sha",
            "a" * 40,
            "--git-subject",
            "old",
            "--service-metrics",
            str(metrics),
            "--catalog-path",
            "/measured/server/triples.json",
        ],
    )
    assert generate_summary.main() == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["dataset"]["selection"]["sample_ids_sha256"] is None
    assert summary["scenario"]["server_args"]["catalog_path"] == "/measured/server/triples.json"
    assert summary["environment"]["gpu"]["uuid"] is None
    assert all(value is None for value in summary["environment"]["runtime"].values())
    assert summary["run"]["baseline_eligible"] is False


@pytest.fixture  # type: ignore[misc]
def offline_summary_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path, Path]:
    data_sha = "d" * 64
    raw = {
        "schema_version": "vllm-gr.offline.raw.v3",
        "execution_mode": "offline",
        "runtime_source": {"git_sha": "a" * 40},
        "runtime": {"vllm_version": "0.30.0", "torch_version": "2.12.0"},
        "model": "model",
        "model_revision": "1" * 40,
        "task": "product",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "dataset_path": "/original-checkout/product_test.parquet",
        "dataset_sha256": data_sha,
        "selection": {"workload_sha256": "e" * 64, "sample_ids_sha256": "f" * 64},
        "warmup_workload_sha256": "1" * 64,
        "num_prompts": 2,
        "completed": 2,
        "failed": 0,
        "warmup_requests": 1,
        "beam_width": 2,
        "input_length": 16,
        "max_tokens": 5,
        "max_concurrency": 1,
        "input_lens": [15, 15],
        "output_lens": [10, 10],
        "aggregate_output_tokens": 20,
        "duration_seconds": 0.1,
        "miss": {"e2e_ms": [2, 3], "cached_tokens": [0, 0]},
        "hit": {"e2e_ms": [1, 2], "cached_tokens": [8, 8]},
        "distributions": {
            "e2e_ms_miss": distribution_ms([2, 3]),
            "e2e_ms_hit": distribution_ms([1, 2]),
        },
        "engine_config": {"catalog_sha256": "2" * 64},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "dataset",
                "revision": "5" * 40,
                "files": [
                    {"task": "video", "path": "/mirror/video.parquet", "sha256": "0" * 64},
                    {"task": "product", "path": "/mirror/product.parquet", "sha256": data_sha},
                ],
            }
        ),
        encoding="utf-8",
    )
    raw_path = tmp_path / "raw.json"
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summary",
            "--raw-result",
            str(raw_path),
            "--dataset-manifest",
            str(manifest),
            "--output",
            str(output),
            "--run-id",
            "local",
            "--beam-width",
            "2",
            "--input-length",
            "16",
            "--warmup-requests",
            "1",
            "--git-sha",
            "a" * 40,
            "--git-subject",
            "local measurement",
        ],
    )
    return raw, raw_path, output


def test_offline_summary_uses_measured_inputs(
    offline_summary_inputs: tuple[dict[str, Any], Path, Path],
) -> None:
    raw, raw_path, output = offline_summary_inputs
    raw["input_lens"] = [15, 12]
    raw["hit"]["cached_tokens"] = [16, 13]
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    assert generate_summary.main() == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["dataset"]["task"] == "product"
    assert summary["dataset"]["measured_sha256"] == raw["dataset_sha256"]
    assert summary["dataset"]["selection"]["workload_sha256"] == raw["selection"]["workload_sha256"]
    assert summary["environment"]["runtime"]["vllm_version"] == "0.30.0"
    assert summary["source"]["runtime_source"]["git_sha"] == "a" * 40
    assert summary["results"]["samples"]["warm_cached_tokens"] == [16, 13]
    assert summary["run"]["baseline_eligible"] is False


def test_offline_manifest_digest_does_not_supply_missing_measurement_evidence(
    offline_summary_inputs: tuple[dict[str, Any], Path, Path],
) -> None:
    raw, raw_path, output = offline_summary_inputs
    manifest_digest = raw.pop("dataset_sha256")
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    assert generate_summary.main() == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["dataset"]["sha256"] == manifest_digest
    assert summary["dataset"]["measured_sha256"] is None
    assert "measured dataset SHA-256 is unavailable" in summary["run"]["qualification_reasons"]
    assert summary["run"]["baseline_eligible"] is False


def test_legacy_logical_batch_preserves_aggregate_cache_counts(
    offline_summary_inputs: tuple[dict[str, Any], Path, Path],
) -> None:
    raw, raw_path, output = offline_summary_inputs
    raw.update(
        schema_version="vllm-gr.offline.raw.v2",
        sample_unit="logical-batch",
        batch_size=2,
        output_lens=[20],
        miss={"e2e_ms": [2], "cached_tokens": [0]},
        hit={"e2e_ms": [1], "cached_tokens": [30]},
        distributions={
            "e2e_ms_miss": distribution_ms([2]),
            "e2e_ms_hit": distribution_ms([1]),
        },
    )
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    assert generate_summary.main() == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    samples = summary["results"]["samples"]
    assert samples["input_tokens"] == [15, 15]
    assert samples["e2el_hit_ms"] == [1]
    assert samples["warm_cached_tokens"] == [30]
    assert (
        "performance comparisons require single-request measurements"
        in compare_ab.check_summary(summary, 2)
    )


def test_v3_single_request_logical_batch_rejects_excess_cache_reuse(
    offline_summary_inputs: tuple[dict[str, Any], Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw, raw_path, output = offline_summary_inputs
    raw.update(sample_unit="logical-batch", batch_size=1)
    raw["hit"]["cached_tokens"] = [16, 17]
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        generate_summary.main()

    assert exc.value.code == 2
    assert "warm_cached_tokens[1] exceeds the input length" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(  # type: ignore[misc]
    ("field", "value", "expected"),
    [
        ("beam_width", 4, "beam_width"),
        ("completed", True, "completed"),
        ("failed", 1, "failed"),
        ("dataset_sha256", "0" * 64, "dataset_sha256"),
        ("runtime_source", {"git_sha": "b" * 40}, "runtime_source.git_sha"),
        ("hit", {"e2e_ms": []}, "hit.e2e_ms"),
        ("hit", {"e2e_ms": [1, float("inf")]}, "hit.e2e_ms"),
        ("hit", {"e2e_ms": [1, True]}, "hit.e2e_ms"),
        ("hit", {"e2e_ms": [1, 2], "cached_tokens": []}, "warm_cached_tokens"),
        ("hit", {"e2e_ms": [1, 2], "cached_tokens": [16, 17]}, "warm_cached_tokens[1]"),
    ],
)
def test_invalid_offline_raw_prevents_summary(
    offline_summary_inputs: tuple[dict[str, Any], Path, Path],
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: Any,
    expected: str,
) -> None:
    raw, raw_path, output = offline_summary_inputs
    raw[field] = value
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        generate_summary.main()
    stderr = capsys.readouterr().err
    assert exc.value.code == 2
    assert raw_path.name in stderr and expected in stderr
    assert not output.exists()


def test_explicit_full_gpu_uuid_selects_the_measured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uuid = "GPU-12345678-abcd-abcd-abcd-123456789012"

    def gpu_query(command: tuple[str, ...], **kwargs: Any) -> str:
        assert f"--id={uuid}" in command
        return f"{uuid}, NVIDIA H20, 96000, 570\n"

    monkeypatch.setattr(generate_summary.subprocess, "check_output", gpu_query)
    assert generate_summary.gpu_metadata(uuid)["name"] == "NVIDIA H20"
    with pytest.raises(ValueError, match="UUID"):
        generate_summary.gpu_metadata("GPU-1")
