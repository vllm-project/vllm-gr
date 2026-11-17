# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import argparse
import gc
import json
import os
import subprocess
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from benchmarks.beam_search import v1_performance_gate as gate
from tests.helpers.paths import TESTS_DIR

SID_IDS = [10, 11, 12, 13, 14]
SID_TEXT = "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"


@pytest.mark.parametrize("compact", [False, True])  # type: ignore[misc]
def test_offline_releases_consumed_outputs_before_next_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, compact: bool
) -> None:
    import torch

    from vllm_gr.entrypoints import gr

    events: list[str] = []
    previous: list[weakref.ReferenceType[Any]] = []
    clock_calls = 0

    class Tokens(list[int]):
        pass

    @dataclass
    class Sequence:
        tokens: Tokens
        cum_logprob: float = -1.0

        @property
        def text(self) -> str:
            events.append("text")
            return SID_TEXT

    @dataclass
    class CompactSequence:
        token_ids: Tokens
        cumulative_logprob: float = -1.0

        @property
        def text(self) -> str:
            raise AssertionError("compact must not materialize full text")

    @dataclass
    class Output:
        sequences: list[Any]
        prompt_token_ids: tuple[int, ...]

    class Tokenizer:
        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return [1] if text == "\n" else [2]

        def decode(self, tokens: list[int], **kwargs: Any) -> str:
            events.append("decode")
            assert tokens == SID_IDS
            assert kwargs == {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            }
            return SID_TEXT

        def convert_ids_to_tokens(self, tokens: list[int]) -> list[str]:
            events.append("validate")
            assert tokens == SID_IDS
            return ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>", "<|sid_end|>"]

    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            self.width = kwargs["max_logprobs"]

        def __enter__(self) -> "LLM":
            return self

        def __exit__(self, *args: object) -> None:
            assert all(ref() is None for ref in previous)

        def get_tokenizer(self) -> Tokenizer:
            return Tokenizer()

        def beam_search_v1(
            self, prompts: list[dict[str, list[int]]], params: Any, **kwargs: Any
        ) -> list[Output]:
            events.append("infer")
            assert "output_mode" not in kwargs
            # Include token buffers: deleting only the top-level outputs list
            # leaves loop variables retaining part of the previous batch.
            assert all(ref() is None for ref in previous)
            outputs = [
                Output(
                    [
                        (
                            CompactSequence(Tokens(SID_IDS))
                            if compact
                            else Sequence(Tokens(prompt["prompt_token_ids"] + SID_IDS))
                        )
                        for _ in range(self.width)
                    ],
                    tuple(prompt["prompt_token_ids"]),
                )
                for prompt in prompts
            ]
            previous[:] = [
                weakref.ref(value)
                for output in outputs
                for value in (
                    output,
                    *output.sequences,
                    *(
                        sequence.token_ids if compact else sequence.tokens
                        for sequence in output.sequences
                    ),
                )
            ]
            return outputs

    def clock() -> int:
        nonlocal clock_calls
        events.append("start" if clock_calls % 2 == 0 else "stop")
        clock_calls += 1
        return clock_calls * 1_000_000

    monkeypatch.setattr(gr, "GRLLM", LLM)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(gate.time, "perf_counter_ns", clock)
    (tmp_path / "setup.json").write_text('{"gr": {}}', encoding="utf-8")
    args = argparse.Namespace(
        output_dir=tmp_path,
        model="unused",
        max_model_len=32,
        beam_width=2,
        gpu_memory_utilization=0.7,
        revision=None,
        generated_steps=3,
        input_len=8,
        batch_sizes=[1, 2],
        warmup_runs=2,
        measured_runs=3,
        ab_legacy_config=False,
    )
    result: dict[str, Any] = {"cases": []}
    gc_policy = gc.isenabled(), gc.get_threshold()
    gate.run_offline(args, "prompt", result)

    assert (gc.isenabled(), gc.get_threshold()) == gc_policy
    assert [case["group_samples_ms"] for case in result["cases"]] == [[1.0] * 3] * 2
    assert events == [
        event
        for count in args.batch_sizes
        for _ in range(args.warmup_runs + args.measured_runs)
        for event in (
            ["start", "infer"]
            + (["decode"] if compact else ["text", "decode"]) * (count * args.beam_width)
            + ["sync", "stop"]
            + ["validate"] * (count * args.beam_width)
        )
    ]


def test_offline_rejects_import_from_another_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("V1_PERF_REPO_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="imported the wrong runtime"):
        gate.run_offline(argparse.Namespace(), "prompt", {"cases": []})


def test_shared_driver_imports_each_selected_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = TESTS_DIR / "system/run_beam_v1_performance.sh"
    # A git hook exports repository-local variables to pytest. They must not
    # redirect these synthetic commits to the caller's repository or index.
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "caller.git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "caller.index"))
    repo_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    roots = [tmp_path / "base", tmp_path / "candidate"]
    for root in roots:
        benchmark_dir = root / "benchmarks/beam_search"
        benchmark_dir.mkdir(parents=True)
        # The old base has a regular package, which prevents Python from
        # discovering candidate-only helpers through namespace-package merging.
        (root / "benchmarks/__init__.py").write_text("", encoding="utf-8")
        validator = Path(gate.__file__).with_name("v1_sid_validation.py")
        (benchmark_dir / validator.name).write_bytes(validator.read_bytes())
        (root / "vllm_gr").mkdir(parents=True)
        (root / "vllm_gr/__init__.py").write_text(f"SOURCE = {root.name!r}\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(root)], env=repo_env, check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--allow-empty",
                "-qm",
                "test",
            ],
            env=repo_env,
            check=True,
        )
    candidate_metrics = roots[1] / "benchmarks/metrics.py"
    candidate_metrics.write_bytes(
        Path(gate.__file__).parents[1].joinpath("metrics.py").read_bytes()
    )
    candidate_driver = roots[1] / "benchmarks/beam_search/v1_performance_gate.py"
    candidate_driver.write_bytes(Path(gate.__file__).read_bytes())
    assert not (roots[0] / "benchmarks/metrics.py").exists()
    driver = candidate_driver.with_name("driver.py")
    driver.write_text(
        "import hashlib, json, runpy, sys\n"
        "from pathlib import Path\n"
        "import vllm_gr\n"
        "driver = Path(__file__).with_name('v1_performance_gate.py')\n"
        "functions = runpy.run_path(str(driver))\n"
        "percentile = functions['linear_percentile_sorted']\n"
        "output = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "name = 'setup' if sys.argv[1] == 'prepare' else 'offline'\n"
        "data = {'source': vllm_gr.SOURCE, "
        "'metrics': str(Path(percentile.__code__.co_filename).resolve()), "
        "'p95': functions['distribution']([10.0, 100.0])['p95_ms'], "
        "'driver': hashlib.sha256(driver.read_bytes()).hexdigest()}\n"
        "(output / (name + '.json')).write_text(json.dumps(data))\n",
        encoding="utf-8",
    )
    driver_digests = set()
    for root in roots:
        output = root / "results"
        env = {
            **repo_env,
            "V1_PERF_REPO_ROOT": str(root),
            "V1_PERF_DRIVER": str(driver),
            "V1_PERF_OUTPUT_DIR": str(output),
            "V1_PERF_MEASURE_ONLY": "1",
            "V1_PERF_OFFLINE_ONLY": "1",
            "TRIPLES_PATH": str(tmp_path / "unused-triples.json"),
            # Simulate an inherited path pointing at the wrong side.
            "PYTHONPATH": str(roots[1 - roots.index(root)]),
        }
        subprocess.run(
            ["bash", str(script.resolve())],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        for name in ("setup", "offline"):
            data = json.loads((output / f"{name}.json").read_text())
            assert data["source"] == root.name
            assert Path(data["metrics"]) == candidate_metrics.resolve()
            assert data["p95"] == 95.5
            driver_digests.add(data["driver"])
    assert len(driver_digests) == 1


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("legacy_config", [False, True])  # type: ignore[misc]
def test_ab_driver_prepares_equivalent_startup_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_config: bool
) -> None:
    import sys
    from types import SimpleNamespace

    from vllm_gr.config.parser import load_engine_gr_config
    from vllm_gr.entrypoints import gr

    class EngineReached(Exception):
        pass

    engine_args: dict[str, Any] = {}

    def record_engine_args(**kwargs: Any) -> None:
        engine_args.update(kwargs)
        raise EngineReached

    monkeypatch.setattr(gr, "GRLLM", record_engine_args)
    monkeypatch.delenv("V1_PERF_REPO_ROOT", raising=False)

    monkeypatch.setitem(
        sys.modules,
        "tools.build_constraint_table",
        SimpleNamespace(
            build_constraint_table=lambda **kwargs: (
                "sha256:" + "a" * 64,
                "sha256:" + "b" * 64,
                None,
            )
        ),
    )
    args = argparse.Namespace(
        output_dir=tmp_path,
        triples=tmp_path / "triples",
        model="model",
        revision=None,
        prompt=tmp_path / "prompt",
        input_len=128,
        beam_width=128,
        generated_steps=3,
        batch_sizes=[1, 2, 4],
        warmup_runs=1,
        measured_runs=2,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        request_timeout=60,
        ab_prefill_grid=True,
        ab_legacy_config=legacy_config,
    )
    gate.prepare(args)
    setup = json.loads((tmp_path / "setup.json").read_text())
    assert setup["beam_width"] == 128
    settings = setup["gr"]
    worker = load_engine_gr_config(None, {}, settings)
    assert worker.beam_max_width == 128
    assert worker.beam_max_decode_steps == 3
    assert worker.deployment.beam_batch_sizes == (1, 2, 4)
    assert worker.deployment.prefill_token_buckets == (1024, 2048, 4096)
    assert worker.constraint.table.artifact_digest == "sha256:" + "a" * 64
    assert worker.constraint.table.tokenizer_digest == "sha256:" + "b" * 64
    assert worker.constraint.table.path == settings["constraint_table"]["path"]
    with pytest.raises(EngineReached):
        gate.run_offline(args, "prompt", {"cases": []})
    if legacy_config:
        assert "gr" not in engine_args
        assert "attention_config" not in engine_args
        config = engine_args["vllm_gr_config"]
        assert config["schema_version"] == 1
        assert config["attention_backend"] == "CUSTOM"
        assert config["beam"] == {
            "execution_mode": "v1",
            "graph_enabled": True,
            "worker_decision": True,
            "max_width": 128,
            "max_decode_steps": 3,
            "online_batch_sizes": [1, 2, 4],
        }
        table = config["constraint_table"]
        assert table["enabled"]
        assert table["candidate_width"] == table["max_top_k"] == 128
        assert table["path"] == worker.constraint.table.path
        assert table["artifact_digest"] == worker.constraint.table.artifact_digest
        assert table["tokenizer_digest"] == worker.constraint.table.tokenizer_digest
    else:
        assert "vllm_gr_config" not in engine_args
        assert engine_args["gr"] == settings
        assert engine_args["attention_config"] == {"backend": "CUSTOM"}
