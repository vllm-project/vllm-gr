# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import gzip
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, overload

import pytest

from benchmarks.qwen import benchmark_v1 as benchmark
from benchmarks.qwen import make_workload as workload


class _Tokenizer:
    all_special_ids = [0]

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{index}": index for index in range(1000)}

    @overload
    def convert_tokens_to_ids(self, token: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, token: list[str]) -> list[int]: ...

    def convert_tokens_to_ids(self, token: str | list[str]) -> int | list[int]:
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(item) for item in token]
        return int(token.rsplit("-", 1)[1])


def test_spread_synthetic_tokens_cover_the_vocab() -> None:
    tokens = benchmark._synthetic_constraint_tokens(
        _Tokenizer(),
        count=12,
        vocab_size=1000,
        layout="spread",
        token_id_start=100,
        token_id_end=500,
    )
    token_ids = [_Tokenizer().convert_tokens_to_ids(token) for token in tokens]

    assert len(set(token_ids)) == 12
    assert 100 <= min(token_ids) < 150
    assert 450 < max(token_ids) < 500


def workload_args(tmp_path: Path, **overrides: Any) -> Namespace:
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 1000}))
    values = {
        "model_dir": str(tmp_path),
        "context_len": 128,
        "requests": 2,
        "token_id_start": 100,
        "token_id_end": 150,
        "seed": 42,
    }
    return Namespace(**(values | overrides))


def test_range_workload_roundtrip_and_reproducibility(tmp_path: Path) -> None:
    args = workload_args(tmp_path)
    rows = workload.build_workload(args)
    assert workload.build_workload(args) == rows
    assert workload.build_workload(workload_args(tmp_path, requests=3))[:2] == rows
    assert workload.build_workload(workload_args(tmp_path, seed=43)) != rows
    assert rows[0]["input_ids"] != rows[1]["input_ids"]
    for row in rows:
        assert len(row["input_ids"]) == row["context_len"] == 128
        assert all(100 <= token < 150 for token in row["input_ids"])
        assert row["metadata"]["token_id_end_exclusive"] == 150
    path = tmp_path / "workload.jsonl"
    workload.write_jsonl(rows, path)
    assert benchmark._load_workload(path, requests=2, vocab_size=1000) == rows


def test_single_token_interval_and_vocab_upper_bound(tmp_path: Path) -> None:
    rows = workload.build_workload(
        workload_args(tmp_path, token_id_start=999, token_id_end=1000, context_len=7)
    )
    assert all(row["input_ids"] == [999] * 7 for row in rows)


@pytest.mark.parametrize(  # type: ignore[misc]
    "overrides",
    [
        {"token_id_start": -1},
        {"token_id_start": 150},
        {"token_id_end": 99},
        {"token_id_end": 1001},
        {"context_len": 0},
        {"requests": 0},
    ],
)
def test_invalid_workload_parameters(tmp_path: Path, overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        workload.build_workload(workload_args(tmp_path, **overrides))


def test_synthetic_constraint_document_has_one_path_per_beam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm_gr.constraints.json_compiler",
        SimpleNamespace(compile_relations_json=lambda triples: {"triples": triples}),
    )

    document = benchmark.build_synthetic_constraint_document(
        _Tokenizer(),
        beam_width=4,
        vocab_size=1000,
        layout="spread",
        token_id_start=100,
        token_id_end=500,
    )

    assert len(document["triples"]) == 4
    assert all(len(triple) == 3 for triple in document["triples"])
    flattened = [token for triple in document["triples"] for token in triple]
    assert len(set(flattened)) == 12


def test_run_once_calls_beam_search_v1_and_reports_metrics() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls: list[tuple[list[dict[str, list[int]]], object, dict[str, Any]]] = []

        def beam_search_v1(
            self, prompts: list[dict[str, list[int]]], params: object, **kwargs: Any
        ) -> list[SimpleNamespace]:
            self.calls.append((prompts, params, kwargs))
            prompt = prompts[0]["prompt_token_ids"]
            return [
                SimpleNamespace(
                    sequences=[
                        SimpleNamespace(
                            tokens=[*prompt, 10 + rank, 20 + rank, 30 + rank],
                            cum_logprob=-float(rank),
                            finish_reason="length",
                        )
                        for rank in range(2)
                    ]
                )
            ]

    llm = FakeLLM()
    workload = [
        {"request_id": "r0", "input_ids": [1, 2]},
        {"request_id": "r1", "input_ids": [3, 4]},
    ]

    result = benchmark._run_once(
        llm,
        workload,
        object(),
        beam_width=2,
        beam_steps=3,
        record_outputs=True,
    )

    assert len(llm.calls) == 2
    assert all(call[2]["concurrency_limit"] == 1 for call in llm.calls)
    assert result["generated_tokens_per_s"] == pytest.approx(result["qps"] * 3)
    assert result["beam_candidates_per_s"] == pytest.approx(result["qps"] * 6)
    assert result["outputs"][0]["beams"][1]["token_ids"] == [11, 21, 31]


def test_configure_multiprocessing_forces_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")

    assert benchmark.configure_multiprocessing() == "spawn"
    assert __import__("os").environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_finalize_args_generates_self_contained_paths(tmp_path: Path) -> None:
    args = Namespace(
        output_json=str(tmp_path / "result.json"),
        workload_jsonl=str(tmp_path / "workload.jsonl"),
        constraint_triples=None,
        constraint_artifact=None,
        generate_constraint_triples=False,
        beam_width=128,
        timeline_output=None,
        timeline_repeat=None,
        repeat=5,
    )

    benchmark.finalize_args(args)

    assert args.generate_constraint_triples is True
    assert args.timeline_repeat == 5
    assert args.workload_jsonl.endswith("workload.jsonl")
    assert args.constraint_triples.endswith("synthetic_constraint_beam128.json")
    assert args.constraint_artifact.endswith("constraint_table_beam128.safetensors")


def test_canonical_config_enables_cuda_artifact(tmp_path: Path) -> None:
    args = Namespace(
        disable_beam_graph=False,
        beam_width=128,
        beam_steps=3,
        attention_backend="CUSTOM",
    )
    artifact = benchmark.ConstraintArtifact(
        path=(tmp_path / "table.safetensors").resolve(),
        artifact_digest="sha256:" + "a" * 64,
        tokenizer_digest="sha256:" + "b" * 64,
        vocab_size=151936,
    )

    config = benchmark.canonical_gr_config(args, artifact)

    assert config["beam"] == {
        "execution_mode": "v1",
        "graph_enabled": True,
        "max_width": 128,
        "max_decode_steps": 3,
    }
    assert config["constraint_table"]["backend"] == "cuda"
    assert config["constraint_table"]["path"] == str(artifact.path)


def test_timeline_profiles_repeat_runs_and_moves_gzip_trace(tmp_path: Path) -> None:
    trace_dir = tmp_path / "worker"
    trace_dir.mkdir()
    target = tmp_path / "timeline.json.gz"

    class FakeLLM:
        def start_profile(self, prefix: str) -> None:
            self.prefix = prefix

        def stop_profile(self) -> None:
            path = trace_dir / f"{self.prefix}_dp0_pp0_tp0.1.pt.trace.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as output:
                json.dump({"traceEvents": [{"name": "prepare_decode"}]}, output)

        def beam_search_v1(
            self, prompts: list[dict[str, list[int]]], params: object, **kwargs: Any
        ) -> list[SimpleNamespace]:
            prompt = prompts[0]["prompt_token_ids"]
            return [
                SimpleNamespace(
                    sequences=[
                        SimpleNamespace(
                            tokens=[*prompt, 10],
                            cum_logprob=0.0,
                            finish_reason="length",
                        )
                    ]
                )
            ]

    args = Namespace(
        timeline_requests=1,
        timeline_repeat=5,
        beam_width=1,
        beam_steps=1,
    )

    result = benchmark._run_timeline(
        FakeLLM(),
        [{"request_id": "r0", "input_ids": [1, 2]}],
        object(),
        args,
        trace_dir=trace_dir,
        target=target,
    )

    assert result["repeat"] == 5
    assert len(result["wall_ms_samples"]) == 5
    with gzip.open(target, "rt", encoding="utf-8") as trace:
        assert json.load(trace)["traceEvents"][0]["name"] == "prepare_decode"
    assert list(trace_dir.iterdir()) == []


@pytest.mark.parametrize("beam_steps", [0, 4])  # type: ignore[misc]
def test_validate_args_rejects_unsupported_beam_steps(beam_steps: int) -> None:
    args = Namespace(
        beam_steps=beam_steps,
        beam_width=2,
        context_len=8,
        requests=1,
        repeat=1,
        warmup_runs=0,
        gpu_memory_utilization=0.9,
        timeline_requests=1,
        timeline_repeat=1,
        timeline_output=None,
    )

    with pytest.raises(ValueError, match=r"in \[1, 3\]"):
        benchmark.validate_args(args)


def test_workload_reader_skips_blank_lines_and_limits_requests(tmp_path: Path) -> None:
    path = tmp_path / "inputs.jsonl"
    path.write_text('\n{"input_ids": [1, 2]}\n\n{"input_ids": [3, 4]}\n')
    rows = benchmark._load_workload(path, requests=1, vocab_size=10)
    assert rows == [{"input_ids": [1, 2], "request_id": "beamcmp-0"}]


def test_workload_reader_rejects_non_object_rows(tmp_path: Path) -> None:
    path = tmp_path / "inputs.jsonl"
    path.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="line 1 must be a JSON object"):
        benchmark._load_workload(path, requests=1, vocab_size=10)


@pytest.mark.parametrize("start,end", [(100, 110), (-1, 100), (100, 100), (990, 1010)])  # type: ignore[misc]
def test_synthetic_interval_rejects_insufficient_or_invalid_bounds(start: int, end: int) -> None:
    with pytest.raises(ValueError):
        benchmark._synthetic_constraint_tokens(
            _Tokenizer(),
            count=12,
            vocab_size=1000,
            layout="spread",
            token_id_start=start,
            token_id_end=end,
        )


def test_synthetic_interval_excludes_special_tokens() -> None:
    tokenizer = _Tokenizer()
    tokenizer.all_special_ids = [100, 105]
    selected = benchmark._synthetic_constraint_tokens(
        tokenizer, count=8, vocab_size=1000, layout="low", token_id_start=100, token_id_end=110
    )
    ids = tokenizer.convert_tokens_to_ids(selected)
    assert set(ids) == set(range(100, 110)) - {100, 105}


@pytest.mark.parametrize("field", ["first_token_ids", "prefix1_values", "prefix2_values"])  # type: ignore[misc]
def test_real_constraint_table_range_checks_all_stages(field: str) -> None:
    import numpy as np

    table = SimpleNamespace(
        **{
            name: np.array([100, 199])
            for name in ("first_token_ids", "prefix1_values", "prefix2_values")
        }
    )
    benchmark._validate_constraint_range(table, 100, 200)
    for bad_id in (99, 200):
        setattr(table, field, np.array([bad_id]))
        with pytest.raises(ValueError, match=field):
            benchmark._validate_constraint_range(table, 100, 200)


def test_reused_workload_must_match_shared_interval(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text('{"input_ids": [100, 199]}\n')
    benchmark._load_workload(
        path, requests=1, vocab_size=1000, token_id_start=100, token_id_end=200
    )
    with pytest.raises(ValueError, match="outside"):
        benchmark._load_workload(
            path, requests=1, vocab_size=1000, token_id_start=100, token_id_end=199
        )


pytestmark = pytest.mark.cpu_test
