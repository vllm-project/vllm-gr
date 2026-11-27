# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import venv
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, overload

import pytest

from benchmarks.beam_search.measure_latency import distribution
from benchmarks.metrics import linear_percentile_sorted
from benchmarks.qwen import benchmark_v1 as benchmark
from benchmarks.qwen import make_workload as workload
from tests.helpers.paths import REPO_ROOT


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


@pytest.mark.parametrize(  # type: ignore[misc]
    ("widths", "paths_per_root"), [(None, 1), ([1, 2, 4], 4)]
)
def test_synthetic_constraint_document_supports_requested_widths(
    monkeypatch: pytest.MonkeyPatch,
    widths: list[int] | None,
    paths_per_root: int,
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
        alternate_beam_widths=widths,
    )

    triples = {tuple(triple) for triple in document["triples"]}
    assert len(triples) == len(document["triples"]) == 4 * paths_per_root
    for root in {triple[0] for triple in triples}:
        branches = {(child, leaf) for first, child, leaf in triples if first == root}
        assert len(branches) == paths_per_root
        assert len({child for child, _ in branches}) == (1 if widths is None else 2)
    assert len({token for triple in triples for token in triple}) == 12


@pytest.mark.l1  # type: ignore[misc]
def test_width_switch_keeps_batch_and_counts_same_schedule_outputs() -> None:
    calls: list[str] = []
    captures = 0

    class Engine:
        class llm_engine:
            @staticmethod
            def collective_rpc(method: str, *, kwargs: dict[str, bool]) -> list[dict[str, Any]]:
                assert method == "get_gr_beam_graph_runtime_stats"
                assert kwargs == {"include_cuda_snapshot": True}
                return [
                    {
                        "cuda_snapshot": {
                            "captures": captures,
                            "allocated_bytes": 20,
                            "reserved_bytes": 32,
                        }
                    }
                ]

        def beam_search(
            self, prompts: list[dict[str, list[int]]], params: Any, **kwargs: Any
        ) -> list[SimpleNamespace]:
            nonlocal captures
            assert prompts == [{"prompt_token_ids": [1]}, {"prompt_token_ids": [2]}]
            assert kwargs["concurrency_limit"] == 2
            if not calls or params.label != calls[-1]:
                captures += 1
            calls.append(params.label)
            # The steady schedules produce different ranks. Change a single
            # score on the final return to A to check parity is discriminating.
            output = []
            for row, prompt in enumerate(prompts):
                offset = 0 if params.label == "a" else 10
                output.append(
                    SimpleNamespace(
                        sequences=[
                            SimpleNamespace(
                                tokens=[*prompt["prompt_token_ids"], 20 + offset + rank],
                                cum_logprob=-float(rank + (len(calls) == 11 and row == 0)),
                            )
                            for rank in range(2)
                        ]
                    )
                )
            return output

    workload = [{"request_id": str(i), "input_ids": [i]} for i in (1, 2)]
    params = {
        "a": SimpleNamespace(label="a", dynamic_beam_widths=[2, 2, 2]),
        "b": SimpleNamespace(label="b", dynamic_beam_widths=[1, 2, 2]),
    }
    result = benchmark._run_width_switch(
        Engine(), workload, params, Namespace(beam_width=2, warmup_runs=1, repeat=2)
    )

    assert calls == list("aaabbbababa")
    assert result["batch_size"] == 2
    assert result["phases"]["steady_a"]["cuda_graph_captures"] == 0
    assert result["phases"]["steady_b"]["cuda_graph_captures"] == 0
    switched = result["phases"]["switching"]
    assert switched["cuda_graph_captures"] == 5
    assert switched["outputs_compared"] == 10
    assert switched["exact_outputs"] == 9


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("widths", [[4, 8], [9, 8, 8], [4, 8, 4], [8, 8, 8]])  # type: ignore[misc]
def test_width_switch_rejects_incompatible_schedules(widths: list[int]) -> None:
    args = benchmark.build_parser().parse_args(
        [
            "--model",
            "model",
            "--output-json",
            "result.json",
            "--workload-jsonl",
            "workload.jsonl",
            "--token-id-start",
            "1000",
            "--token-id-end",
            "2000",
            "--beam-width",
            "8",
        ]
    )
    args.alternate_beam_widths = widths
    with pytest.raises(ValueError, match="--alternate-beam-widths"):
        benchmark.validate_args(args)


def test_run_once_calls_beam_search_and_reports_metrics() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls: list[tuple[list[dict[str, list[int]]], object, dict[str, Any]]] = []

        def beam_search(
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

    assert config["beam_graph_enabled"] is True
    assert config["beam_width"] == [128, 128, 128]
    assert config["max_decode_steps"] == 3
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

        def beam_search(
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


@pytest.mark.l1  # type: ignore[misc]
def test_latency_percentiles_match_beam_gate() -> None:
    samples = [10.0, 100.0]
    assert benchmark._percentile(samples, 0.50) == distribution(samples)["p50_ms"] == 55.0
    assert benchmark._percentile(samples, 0.95) == distribution(samples)["p95_ms"] == 95.5
    assert linear_percentile_sorted(samples, 0.95) == 95.5
    assert linear_percentile_sorted([10.0], 0.95) == benchmark._percentile([10.0], 0.95) == 10.0
    assert distribution([10.0])["p95_ms"] == 10.0
    assert benchmark._percentile([], 0.95) is None
    for invalid in ([], [0.0], [-1.0], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="positive finite values"):
            distribution(invalid)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("second_length,common_length", [(128, 128), (256, None)])  # type: ignore[misc]
def test_report_input_lengths_come_from_consumed_jsonl_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, second_length: int, common_length: int | None
) -> None:
    input_path = tmp_path / "workload.jsonl"
    with input_path.open("w") as stream:
        for length in (128, second_length, 999):
            print(json.dumps({"input_ids": [100] * length, "context_len": 5000}), file=stream)
    args = benchmark.build_parser().parse_args(
        [
            "--model",
            "model",
            "--output-json",
            str(tmp_path / "report.json"),
            "--workload-jsonl",
            str(input_path),
            "--requests",
            "2",
            "--token-id-start",
            "100",
            "--token-id-end",
            "200",
            "--warmup-runs",
            "0",
            "--repeat",
            "1",
            "--beam-width",
            "1",
        ]
    )
    benchmark.finalize_args(args)
    inputs: list[int] = []

    class Engine:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def beam_search(self, prompts: list[dict[str, list[int]]], *_: Any, **kw: Any) -> list[Any]:
            ids = prompts[0]["prompt_token_ids"]
            inputs.append(len(ids))
            beam = SimpleNamespace(tokens=[*ids, 101, 102, 103], cum_logprob=-1.0)
            return [SimpleNamespace(sequences=[beam])]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "vllm_gr.entrypoints.gr", SimpleNamespace(GRLLM=lambda **kw: Engine())
    )
    monkeypatch.setitem(
        sys.modules, "vllm_gr.sampling_params", SimpleNamespace(BeamSearchParams=lambda **kw: kw)
    )
    monkeypatch.setattr(benchmark, "_load_model_assets", lambda _: (object(), 1000))
    monkeypatch.setattr(
        benchmark, "ensure_constraint_document", lambda *a, **kw: tmp_path / "triples.json"
    )
    monkeypatch.setattr(
        benchmark,
        "build_constraint_artifact",
        lambda *a, **kw: benchmark.ConstraintArtifact(
            tmp_path / "constraint.safetensors", "a", "b", 1000
        ),
    )
    monkeypatch.setattr(benchmark, "canonical_gr_config", lambda *a: {})
    monkeypatch.setattr(benchmark, "_environment", lambda _: {})

    report = benchmark.run_benchmark(args)
    assert report["context_len"] == common_length
    assert report["declared_context_len"] == 5000
    assert report["input_token_lengths"] == inputs == [128, second_length]
    assert len(report["runs"][0]["request_latencies_ms"]) == len(inputs)


@pytest.mark.l1  # type: ignore[misc]
def test_direct_benchmark_help_imports_shared_metrics(tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, str(Path(benchmark.__file__).resolve()), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--workload-jsonl" in completed.stdout


def _run_qwen_shell(checkout: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    bash = os.environ.get("TEST_BASH") or shutil.which("bash")
    if bash is None:
        raise pytest.skip.Exception("Bash is required to test the Qwen launcher")
    env = {
        **os.environ,
        "MODEL": "Qwen/model-id",
        "TOKEN_ID_START": "10",
        "TOKEN_ID_END": "20",
        "WORKLOAD_JSONL": "",
        "INSTALL_MODE": "skip",
        "BOOTSTRAP_PYTHON": "./unexpected-bootstrap",
        "GET_PIP_URL": "file:///nonexistent-qwen-test-get-pip.py",
        "VENV_DIR": "venv",
        "OUT_DIR": "output",
        "CACHE_ROOT": "cache",
        "CONTEXT_LEN": "3",
        "REQUESTS": "1",
        **overrides,
    }
    return subprocess.run(
        [bash, "benchmarks/qwen/run_beam_search_benchmark.sh"],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


@pytest.mark.l1  # type: ignore[misc]
def test_qwen_shell_autogeneration_requires_local_config_before_install(tmp_path: Path) -> None:
    venv_dir = tmp_path / "missing-venv"
    completed = _run_qwen_shell(REPO_ROOT, INSTALL_MODE="always", VENV_DIR=venv_dir.as_posix())
    assert completed.returncode == 2, completed.stderr
    assert "Qwen/model-id/config.json" in completed.stderr
    assert "WORKLOAD_JSONL" in completed.stderr
    assert "[1/3]" not in completed.stdout
    assert not venv_dir.exists()


@pytest.mark.l1  # type: ignore[misc]
def test_qwen_shell_skip_requires_existing_venv(tmp_path: Path) -> None:
    workload_file = tmp_path / "input.jsonl"
    workload_file.touch()
    missing_venv = tmp_path / "missing-venv"
    completed = _run_qwen_shell(
        REPO_ROOT, WORKLOAD_JSONL=workload_file.as_posix(), VENV_DIR=missing_venv.as_posix()
    )
    assert completed.returncode == 2, completed.stderr
    assert "INSTALL_MODE=skip requires an existing virtual environment" in completed.stderr
    assert missing_venv.as_posix() in completed.stderr
    assert "unexpected-bootstrap" not in completed.stderr
    assert not missing_venv.exists()


@pytest.mark.l1  # type: ignore[misc]
def test_qwen_shell_skip_checks_checkout_and_preserves_workload_options(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout-a"
    other_checkout = tmp_path / "checkout-b"
    for repo in (checkout, other_checkout):
        (repo / "vllm_gr").mkdir(parents=True)
        (repo / "vllm_gr/__init__.py").touch()
    qwen_dir = checkout / "benchmarks/qwen"
    qwen_dir.mkdir(parents=True)
    for name in ("run_beam_search_benchmark.sh", "make_workload.py"):
        source = REPO_ROOT / "benchmarks/qwen" / name
        (qwen_dir / name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )
    (qwen_dir / "benchmark_v1.py").write_text(
        "import json, sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8"
    )
    workload_file = checkout / "input.jsonl"
    workload_file.write_text('{"input_ids": [10, 11, 12]}\n', encoding="utf-8")

    venv_dir = checkout / "venv"
    venv.create(venv_dir, with_pip=False)
    python = venv_dir / "bin/python"
    if os.name == "nt":
        # The launcher uses the POSIX bin/python layout even under Git Bash.
        windows_python = venv_dir / "Scripts/python.exe"
        python.parent.mkdir()
        python.write_text(
            f'#!/usr/bin/env bash\nexec "{windows_python.as_posix()}" "$@"\n',
            encoding="utf-8",
            newline="\n",
        )
        python.chmod(0o755)
        python = windows_python
    site_packages = Path(
        subprocess.check_output(
            [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
    )
    for name in ("vllm", "vllm_gr"):
        metadata = site_packages / f"{name}-0.0.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Name: {name}\nVersion: 0.0.0\n", encoding="utf-8")

    editable = site_packages / "checkout.pth"
    editable.write_text(str(other_checkout), encoding="utf-8")
    completed = _run_qwen_shell(checkout, WORKLOAD_JSONL=workload_file.name)
    assert completed.returncode == 2, completed.stderr
    assert "the venv does not contain this checkout" in completed.stderr
    assert not (checkout / "output").exists()
    assert not (venv_dir / "get-pip.py").exists()

    editable.write_text(str(checkout), encoding="utf-8")
    completed = _run_qwen_shell(checkout, WORKLOAD_JSONL=workload_file.name)
    assert completed.returncode == 0, completed.stderr
    args = next(json.loads(line) for line in completed.stdout.splitlines() if line.startswith('["'))
    assert args[args.index("--model") + 1] == "Qwen/model-id"
    assert args[args.index("--workload-jsonl") + 1] == workload_file.name

    model = checkout / "local-model"
    model.mkdir()
    (model / "config.json").write_text('{"vocab_size": 100}', encoding="utf-8")
    completed = _run_qwen_shell(checkout, MODEL=model.name)
    assert completed.returncode == 0, completed.stderr
    generated = next((checkout / "output").glob("*.jsonl"))
    row = json.loads(generated.read_text(encoding="utf-8"))
    assert len(row["input_ids"]) == 3
    assert all(10 <= token < 20 for token in row["input_ids"])
