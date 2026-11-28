# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Run the OneRec Video-100 benchmark gate on CUDA or Ascend."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from typing import Any, cast

import pytest

from tests.helpers.paths import REPO_ROOT
from tools.ci.read_benchmark_metrics import (
    ACCURACY_KEYS,
    read_accuracy,
    read_metrics,
    read_serving,
)
from tools.ci.resolve_benchmark_policy import load_policy, resolve_policy

pytestmark = [
    pytest.mark.l4,
    pytest.mark.gpu_test,
    pytest.mark.npu_test,
    pytest.mark.model_test,
    pytest.mark.slow,
    pytest.mark.timeout(5400),
]

BEAM_WIDTH = 128
ENDPOINT = "/v1/chat/completions"
BENCHMARK_MODULE = "benchmarks.open_one_rec"
DATASET_REPO = "OpenOneRec/OpenOneRec-RecIF"
BASELINE_PATH = REPO_ROOT / "tests/baseline/onerec_video_benchmark.json"
PERFORMANCE_GATES = {
    "offline elapsed_time": ("ELAPSED_TIME_OFFLINE_THRESHOLD", "OFFLINE_ELAPSED_FAILURE_MODE"),
    "online elapsed_time": ("ELAPSED_TIME_ONLINE_THRESHOLD", "ONLINE_ELAPSED_FAILURE_MODE"),
    "P99 TTFT": ("P99_TTFT_THRESHOLD", "P99_TTFT_FAILURE_MODE"),
    "P99 TPOT": ("P99_TPOT_THRESHOLD", "P99_TPOT_FAILURE_MODE"),
}
ACCURACY_THRESHOLDS = ("PASS_AT_1_THRESHOLD", "PASS_AT_32_THRESHOLD", "RECALL_AT_32_THRESHOLD")


def detect_accelerator() -> tuple[str, str]:
    import torch

    if torch.cuda.is_available():
        return "gpu", torch.cuda.get_device_name()
    try:
        import torch_npu
    except ModuleNotFoundError as exc:
        if exc.name != "torch_npu":
            raise
    else:
        if torch_npu.npu.is_available():
            return "npu", torch_npu.npu.get_device_name()
    raise pytest.fail.Exception("OneRec Video-100 requires a visible CUDA or Ascend device")


def resolve_device_baseline(device: str, model: str) -> dict[str, str]:
    baseline = load_policy(BASELINE_PATH)
    boards = baseline["device_overrides"].get(device, {}).get("boards", {})
    names = {name.casefold(): name for name in boards}
    board = names.get(model.strip().casefold())
    if board is None:
        warnings.warn(
            f"No Video-100 baseline for {device} model {model!r}; "
            "using the default baseline thresholds.",
            stacklevel=2,
        )
        board = model.strip()
    return cast("dict[str, str]", resolve_policy(baseline, device, board))


def build_gr_overrides(model: Path, triples: Path, artifact: Path, backend: str) -> dict[str, Any]:
    from vllm_gr.constraints.artifact_builder import build_single_trie_artifact

    digest, tokenizer_digest = build_single_trie_artifact(model, triples, artifact)
    return {
        "beam_width": [BEAM_WIDTH] * 3,
        "beam_graph_enabled": backend == "cuda",
        "worker_decision": True,
        "constraint_table": {
            "backend": backend,
            "max_top_k": BEAM_WIDTH,
            "path": str(artifact.resolve()),
            "digests": {"artifact": digest, "tokenizer": tokenizer_digest},
        },
        "candidate_width": [BEAM_WIDTH] * 3,
    }


def server_command(model: str, port: int, gr_overrides: dict[str, Any]) -> list[str]:
    template = REPO_ROOT / "benchmarks/open_one_rec/qwen3_soft_switch.jinja2"
    return [
        sys.executable,
        "-m",
        "vllm_gr.entrypoints.cli.main",
        "serve",
        "--gr",
        model,
        "--attention-config",
        '{"backend":"CUSTOM"}',
        "--gr-config",
        json.dumps(gr_overrides),
        "--async-scheduling",
        "--no-enable-chunked-prefill",
        "--port",
        str(port),
        "--seed",
        "123",
        "--chat-template",
        str(template),
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
        "--max-logprobs",
        "512",
        "--max_num_seqs",
        "2048",
        "--max_num_batched_tokens",
        "16384",
        "--scheduling-policy",
        "priority",
    ]


def wait_for_server(server: subprocess.Popen[bytes], port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        assert server.poll() is None, f"OneRec server exited with code {server.returncode}"
        connection = HTTPConnection("127.0.0.1", port, timeout=min(2, remaining))
        try:
            connection.request("GET", "/v1/models")
            if connection.getresponse().status == 200:
                return
        except (OSError, HTTPException):
            pass
        finally:
            connection.close()
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    pytest.fail(f"OneRec server did not become ready within {timeout}s")


def check_metric(
    label: str, value: float, threshold: float, mode: str, *, minimum: bool = False
) -> None:
    passed = value >= threshold if minimum else value <= threshold
    relation = ">=" if minimum else "<="
    message = f"{label}: {value} (required {relation} {threshold})"
    if passed:
        print(message)
    elif mode == "warning":
        warnings.warn(message, stacklevel=2)
    else:
        pytest.fail(message)


def check_accuracy(
    results: Path,
    phase: str,
    accuracy_gates: list[float],
    accuracy_mode: str,
    elapsed_gate: tuple[float, str],
) -> None:
    files = list(results.glob("*.json"))
    assert len(files) == 1, f"Expected one accuracy result in {results}, found {files}"
    *accuracy, elapsed = read_accuracy(read_metrics(files[0]))
    for key, value, threshold in zip(ACCURACY_KEYS[:3], accuracy, accuracy_gates, strict=True):
        check_metric(f"{phase} {key}", value, threshold, accuracy_mode, minimum=True)
    check_metric(f"{phase} elapsed_time", elapsed, *elapsed_gate)


def run_module(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", module, *args], cwd=REPO_ROOT, check=True)


def test_onerec_video_benchmark() -> None:
    from huggingface_hub import snapshot_download

    model_path = os.environ.get("VLLM_GR_TEST_MODEL")
    model = "OpenOneRec/OneRec-1.7B"
    if model_path:
        local_model = Path(model_path).resolve()
        assert (local_model / "config.json").is_file(), f"Model config missing in {local_model}"
        model = str(local_model)

    device, accelerator = detect_accelerator()
    policy = resolve_device_baseline(device, accelerator)
    snapshot = snapshot_download(
        DATASET_REPO, repo_type="dataset", allow_patterns="benchmark_data/*"
    )
    dataset = Path(snapshot) / "benchmark_data"
    assert dataset.is_dir(), f"{DATASET_REPO} snapshot has no benchmark_data directory: {snapshot}"
    backend = "npu" if device == "npu" else "cuda"
    accuracy_mode = policy["ACCURACY_FAILURE_MODE"]
    accuracy_gates = [float(policy[key]) for key in ACCURACY_THRESHOLDS]
    performance_gates = {
        label: (float(policy[threshold]), policy[mode])
        for label, (threshold, mode) in PERFORMANCE_GATES.items()
    }

    config_dir = Path(
        os.environ.get("VLLM_GR_BENCHMARK_CONFIG_DIR") or REPO_ROOT / "generated_data/benchmark_v1"
    ).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    results = Path(tempfile.mkdtemp(prefix="onerec-results.", dir=config_dir))

    catalog_path = os.environ.get("CATALOG_PATH")
    catalog = Path(catalog_path).resolve() if catalog_path else results / "video_catalog.json"
    if catalog_path:
        assert catalog.is_file(), f"CATALOG_PATH must point to a video catalog: {catalog}"
    else:
        run_module(
            f"{BENCHMARK_MODULE}.tools.create_task_catalog",
            "-t",
            "video",
            "-d",
            str(dataset),
            "-o",
            str(catalog),
        )

    triples_path = os.environ.get("TRIPLES_PATH")
    triples = (
        Path(triples_path).resolve()
        if triples_path
        else catalog.with_name("video_constraint_triples.json")
    )
    if not triples.is_file():
        from tools.constraints.build_constraint_triples import process_catalog

        # Keep newly generated implicit triples in the current run directory.
        if not triples_path:
            triples = results / triples.name
        process_catalog(
            str(catalog), str(triples), begin_token="<|sid_begin|>", end_token="<|sid_end|>"
        )

    print(
        f"Accelerator: {accelerator}, baseline: {policy['BENCHMARK_BOARD_ID']}, "
        f"model: {model}, dataset: {dataset}"
    )
    print(f"Catalog: {catalog}, graph enabled: {backend == 'cuda'}, metrics directory: {results}")
    overrides = build_gr_overrides(
        model=Path(model),
        triples=triples,
        artifact=results / "constraint_table.safetensors",
        backend=backend,
    )
    common = [
        "--model",
        model,
        "--dataset-path",
        str(dataset),
        "--dataset-name",
        "onerec",
        "--task-type",
        "video",
        "--use-beam-search",
        "--n",
        str(BEAM_WIDTH),
    ]
    accuracy_module = f"{BENCHMARK_MODULE}.one_rec_acc_test"
    offline = results / "offline"
    run_module(
        accuracy_module,
        *common,
        "--result-dir",
        str(offline),
        "--num-prompts",
        "100",
        "--batch-size",
        "1",
        "--gr",
        json.dumps(overrides),
    )
    check_accuracy(
        offline, "offline", accuracy_gates, accuracy_mode, performance_gates["offline elapsed_time"]
    )

    port = int(os.environ.get("VLLM_GR_BENCHMARK_PORT") or "7000")
    timeout = int(os.environ.get("VLLM_GR_BENCHMARK_SERVE_TIMEOUT") or "600")
    online_args = [
        "--backend",
        "openai-chat",
        "--endpoint",
        ENDPOINT,
        "--port",
        str(port),
        "--max-concurrency",
        "1",
    ]
    serving_module = f"{BENCHMARK_MODULE}.one_rec_main"
    with subprocess.Popen(server_command(model, port, overrides), cwd=REPO_ROOT) as server:
        try:
            wait_for_server(server, port, timeout)
            run_module(
                serving_module,
                "bench",
                "serve",
                *common,
                *online_args,
                "--num-prompts",
                "5",
                "--request-rate",
                "1",
            )
            run_module(
                serving_module,
                "bench",
                "serve",
                *common,
                *online_args,
                "--num-prompts",
                "10",
                "--request-rate",
                "0.5",
                "--save-result",
                "--result-dir",
                str(results),
                "--result-filename",
                "serving.json",
            )
            failed, ttft, tpot = read_serving(read_metrics(results / "serving.json"))
            assert failed == 0, f"OneRec serving benchmark had {failed} failed requests"
            check_metric("P99 TTFT", float(ttft), *performance_gates["P99 TTFT"])
            check_metric("P99 TPOT", float(tpot), *performance_gates["P99 TPOT"])

            online = results / "online"
            run_module(
                accuracy_module,
                *common,
                *online_args,
                "--num-prompts",
                "100",
                "--result-dir",
                str(online),
            )
            check_accuracy(
                online,
                "online",
                accuracy_gates,
                accuracy_mode,
                performance_gates["online elapsed_time"],
            )
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
