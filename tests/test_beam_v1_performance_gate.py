# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

SID_IDS = [10, 11, 12, 13, 14]
SID_TEXT = "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"


def test_offline_releases_consumed_outputs_before_next_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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

        @property
        def text(self) -> str:
            events.append("text")
            return SID_TEXT

    @dataclass
    class Output:
        sequences: list[Sequence]

    class Tokenizer:
        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return [1] if text == "\n" else [2]

        def convert_ids_to_tokens(self, tokens: list[int]) -> list[str]:
            events.append("validate")
            assert tokens == SID_IDS
            return ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>", "<|sid_end|>"]

    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            self.width = kwargs["max_logprobs"]

        def __enter__(self) -> "LLM":
            return self

        def __exit__(self, *args: Any) -> None:
            assert all(ref() is None for ref in previous)

        def get_tokenizer(self) -> Tokenizer:
            return Tokenizer()

        def beam_search_v1(
            self, prompts: list[dict[str, list[int]]], params: Any, **kwargs: Any
        ) -> list[Output]:
            events.append("infer")
            # Include token buffers: deleting only the top-level outputs list
            # leaves loop variables retaining part of the previous batch.
            assert all(ref() is None for ref in previous)
            outputs = [
                Output(
                    [
                        Sequence(Tokens(prompt["prompt_token_ids"] + SID_IDS))
                        for _ in range(self.width)
                    ]
                )
                for prompt in prompts
            ]
            previous[:] = [
                weakref.ref(value)
                for output in outputs
                for value in (
                    output,
                    *output.sequences,
                    *(sequence.tokens for sequence in output.sequences),
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
    (tmp_path / "vllm_gr_config.yaml").write_text("{}", encoding="utf-8")
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
        b1_warmup_runs=4,
        measured_runs=3,
    )
    result: dict[str, Any] = {"cases": []}
    gc_policy = gc.isenabled(), gc.get_threshold()
    gate.run_offline(args, "prompt", result)

    assert (gc.isenabled(), gc.get_threshold()) == gc_policy
    assert [case["group_samples_ms"] for case in result["cases"]] == [[1.0] * 3] * 2
    assert events == [
        event
        for count in args.batch_sizes
        for _ in range(
            (args.b1_warmup_runs if count == 1 else args.warmup_runs) + args.measured_runs
        )
        for event in (
            ["start", "infer"]
            + ["text"] * (count * args.beam_width)
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


def test_shared_driver_imports_each_selected_runtime(tmp_path: Path) -> None:
    script = Path(__file__).parent / "system/run_beam_v1_performance.sh"
    roots = [tmp_path / "base", tmp_path / "candidate"]
    for root in roots:
        (root / "vllm_gr").mkdir(parents=True)
        (root / "vllm_gr/__init__.py").write_text(f"SOURCE = {root.name!r}\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--allow-empty",
                "-qm",
                "test",
            ],
            check=True,
        )
    driver = roots[1] / "benchmarks/beam_search/driver.py"
    driver.parent.mkdir(parents=True)
    driver.write_text(
        "import hashlib, json, sys\n"
        "from pathlib import Path\n"
        "import vllm_gr\n"
        "output = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "name = 'setup' if sys.argv[1] == 'prepare' else 'offline'\n"
        "data = {'source': vllm_gr.SOURCE, 'driver': "
        "hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}\n"
        "(output / (name + '.json')).write_text(json.dumps(data))\n",
        encoding="utf-8",
    )
    driver_digests = set()
    for root in roots:
        output = root / "results"
        env = {
            **os.environ,
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
            driver_digests.add(data["driver"])
    assert len(driver_digests) == 1
