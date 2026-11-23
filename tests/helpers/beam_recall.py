# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Build fixed-sample Beam Recall results for benchmark and CI tests."""

import json
from pathlib import Path

import pytest

from benchmarks.beam_search.compare_recall_baseline import TASKS


def make_repeated_results(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "VLLM_GR_PREFILL_GRAPH_BUCKETS": "16,24,32,48",
        "VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS": "4",
        "VLLM_GR_PREFILL_GRAPH_KV_BOUND": "8192",
        "VLLM_GR_V1_STARTUP_CAPTURE_BATCH_SIZE": "4",
    }.items():
        monkeypatch.setenv(name, value)
    (root / "commit.txt").write_text("test-commit\n", encoding="utf-8")
    (root / "total_wall_seconds.txt").write_text("1050\n", encoding="utf-8")
    for number in range(1, 11):
        run_dir = root / "runs" / f"run-{number:02d}"
        run_dir.mkdir(parents=True)
        (run_dir / "total_wall_seconds.txt").write_text(f"{100 + number}\n", encoding="utf-8")
        for group, seconds in (("sid2pid", 60), ("sid2iid", 40)):
            group_dir = run_dir / "groups" / group
            group_dir.mkdir(parents=True)
            (group_dir / "wall_seconds.txt").write_text(f"{seconds}\n", encoding="utf-8")
        for task in TASKS:
            task_dir = run_dir / task
            task_dir.mkdir()
            (task_dir / "wall_seconds.txt").write_text("20\n", encoding="utf-8")
            recall = 0.04 + number * 0.00001
            if task == "video" and number == 7:
                recall -= 0.001
            metrics = {
                "task": task,
                "total_samples": 1000,
                "batch_size": 4,
                "beam_width": 128,
                "temperature": 0.6,
                "sample_ids": list(range(1000)),
                "recall@32": recall,
                "pid_recall@32": 0.03 + number * 0.00001,
                "elapsed_time": 10 + number,
                "source_sha256": f"source-{task}",
                "artifact_digest": "artifact",
                "tokenizer_digest": "tokenizer",
                "interactive_tailcut_count": 0,
                "model_revision": "test-revision",
                "engine_config": {
                    "max_model_len": 8192,
                    "max_num_batched_tokens": 10000,
                    "async_scheduling": True,
                    "enable_prefix_caching": True,
                    "enable_chunked_prefill": False,
                    "max_num_seqs": 128,
                    "max_logprobs": 256,
                    "enforce_eager": False,
                    "dtype": "auto",
                    "seed": 0,
                },
                "graph_profile": {
                    "prefill_bucket_grid": "16,24,32,48",
                    "prefill_capture_num_reqs": "4",
                    "prefill_kv_bound": "8192",
                    "v1_startup_capture_batch_size": "4",
                },
                "graph_runtime": {
                    "captures": [[[16, 4]]],
                    "effective_profiles": [
                        {"runner": "MockRunner", "buckets": [16], "kv_bounds": {"16:4": 8192}}
                    ],
                    "dispatches": {"replay:4": 250},
                    "expected_b4_replays": 250,
                },
                "runtime_provenance": {
                    "accelerator": "test GPU",
                    "cuda": "test",
                    "torch": "test",
                    "vllm": "test",
                    "python": "test",
                },
                "prefill_graph_buckets": "16,24,32,48",
                "prefill_graph_capture_num_reqs": "4",
                "v1_startup_capture_batch_size": "4",
            }
            (task_dir / f"model_{task}_bw128_bs4_1000.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )


def set_sid_recall(root: Path, run: int, task: str, recall: float) -> None:
    file = root / "runs" / f"run-{run:02d}" / task / f"model_{task}_bw128_bs4_1000.json"
    metrics = json.loads(file.read_text(encoding="utf-8"))
    metrics["recall@32"] = recall
    file.write_text(json.dumps(metrics), encoding="utf-8")
