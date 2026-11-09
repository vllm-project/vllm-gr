# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise original weight reload over production gRPC and HTTP control APIs.

Usage: python tests/st/run_checkpoint_reload.py --output /tmp/ckpt-reload
Uses local tiny random Llama/Qwen3 checkpoints, optionally with GR head caches.
No downloads.
"""

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import grpc
import httpx
import numpy as np

from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2 as pb
from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2_grpc as rpc
from vllm_gr.sidecar import default_ckpt_checker

ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def prepare(
    output: Path, head_cache: str = "none", tied_weights: bool = False
) -> tuple[Path, Path, Path]:
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from vllm.utils.torch_utils import set_random_seed

    from examples.online_serving.grpc.prepare_model import prepare_model

    first, config = prepare_model(output / "a", context_length=128)
    if head_cache == "transposed":
        set_random_seed(7)
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                max_position_embeddings=128,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                tie_word_embeddings=tied_weights,
            )
        ).eval().save_pretrained(first)
    if tied_weights and head_cache != "transposed":
        tied = AutoModelForCausalLM.from_pretrained(first)
        tied.config.tie_word_embeddings = True
        tied.tie_weights()
        tied.save_pretrained(first)
    if head_cache == "sliced":
        from transformers import AutoTokenizer

        from vllm_gr.constraints.artifact import save_constraint_table_artifact
        from vllm_gr.constraints.artifact_compiler import (
            ConstraintTableArtifactCompiler,
        )
        from vllm_gr.constraints.json_compiler import compile_relations_json
        from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

        tokenizer = AutoTokenizer.from_pretrained(first)
        digest = compute_tokenizer_digest(tokenizer)
        compiler = ConstraintTableArtifactCompiler(tokenizer, 128, tokenizer_digest=digest)
        relations = [
            [f"tok{x}", f"tok{y}", f"tok{z}"] for x in (0, 1) for y in (10, 11) for z in (14, 15)
        ]
        table = compiler.compile(compile_relations_json(relations))
        path = output / "table.safetensors"
        artifact_digest = save_constraint_table_artifact(table, path)
        data = json.loads(config.read_text())
        data["constrained_generation"] = True
        data["constraint_table"].update(
            enabled=True,
            path=str(path),
            artifact_digest=artifact_digest,
            tokenizer_digest=digest,
        )
        config.write_text(json.dumps(data))
    import shutil

    second = output / "b/model"
    shutil.copytree(first, second, dirs_exist_ok=True)
    original = AutoModelForCausalLM.from_pretrained(first)
    set_random_seed(19)
    replacement = type(original)(original.config).eval()
    replacement.save_pretrained(second)
    assert default_ckpt_checker(first, second)
    return first, second, config


def request() -> Any:
    rows = np.random.default_rng(7).integers(3, 128, size=(4, 64), dtype=np.int64)
    return pb.BatchGenerateRequest(
        requests=[
            pb.GenerateRequest(
                request_id=f"item-{index}",
                tokenized=pb.TokenizedInput(input_ids_i64_le=row.astype("<i8").tobytes()),
                sampling_params=pb.SamplingParams(
                    max_tokens=3, temperature=0, ignore_eos=True, n=4
                ),
                beam_search=pb.BeamSearchOptions(beam_width=4),
            )
            for index, row in enumerate(rows)
        ]
    )


def outputs(reply: Any) -> list[dict[str, list[int]]]:
    return [
        dict(ids=list(item.beams.output_ids), offsets=list(item.beams.output_offsets))
        for item in reply.items
    ]


@contextlib.contextmanager
def server(
    output: Path,
    model: Path,
    config: Path,
    label: str,
    launcher: Path | None = None,
    *,
    graph: bool = False,
    head_cache: str = "none",
) -> Iterator[tuple[Any, httpx.Client]]:
    port, admin = free_port(), free_port()
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT),
        VLLM_GR_ADMIN_PORT=str(admin),
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        HF_HUB_OFFLINE="1",
    )
    env.pop("VLLM_GR_UNCONSTRAINED_HEAD_LAYOUT", None)
    env.pop("VLLM_GR_HEAD_SLICE", None)
    if head_cache == "transposed":
        # The transposed cache is created only when head slicing is enabled.
        env.update(
            VLLM_GR_UNCONSTRAINED_HEAD_LAYOUT="transposed",
            VLLM_GR_HEAD_SLICE="16384",
        )
    if graph:
        data = json.loads(config.read_text())
        data["beam"]["graph_enabled"] = True
        config = output / f"{label}-graph-config.json"
        config.write_text(json.dumps(data, indent=2))
        env.update(
            VLLM_ENABLE_PREFILL_CUDAGRAPH="1",
            VLLM_GR_PREFILL_GRAPH_KV_BOUND="128",
            VLLM_GR_PREFILL_GRAPH_BUCKETS="64,128,256",
            VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS="1,2,4",
        )
    command = [
        sys.executable,
        "-m",
        "vllm_gr.entrypoints.cli.main",
        "serve",
        str(model),
        "--vllm-gr-config",
        str(config),
        "--grpc",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--async-scheduling",
        "--no-enable-chunked-prefill",
        "--enable-prefix-caching",
        "--max-model-len",
        "128",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "16",
        "--gpu-memory-utilization",
        "0.1",
        "--kv-cache-memory-bytes",
        "67108864",
        "--enforce-eager",
    ]
    if graph:
        command.remove("--enforce-eager")
        command.extend(
            [
                "--compilation-config",
                json.dumps({"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8, 16]}),
            ]
        )
    if launcher is not None:
        command = [sys.executable, str(launcher), "--serve", *command[3:]]
    (output / f"{label}-command.json").write_text(json.dumps(command, indent=2))
    with (output / f"{label}-server.log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, cwd=ROOT, stdout=log, stderr=log, start_new_session=True
        )
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        try:
            ready = grpc.channel_ready_future(channel)
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited: {label}-server.log")
                try:
                    ready.result(timeout=0.2)
                    break
                except grpc.FutureTimeoutError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("server startup timed out")
            with httpx.Client(base_url=f"http://127.0.0.1:{admin}", timeout=180) as control:
                yield rpc.VllmEngineStub(channel), control
        finally:
            channel.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", action="store_true", help="Enable compile and CUDA Graph")
    parser.add_argument("--head-cache", choices=("none", "sliced", "transposed"), default="none")
    parser.add_argument("--tied-weights", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    first, second, config = prepare(output, args.head_cache, args.tied_weights)
    launcher = None
    if args.head_cache != "none":
        launcher = Path(__file__).with_name("run_reload_memory.py")
        os.environ["GR_RELOAD_MEMORY_LOG"] = str(output / "cache-memory.jsonl")
        os.environ["GR_RELOAD_HEAD_CACHE"] = args.head_cache
        (output / "cache-memory.jsonl").write_text("")
    results: dict[str, Any] = {
        "model": "tiny Qwen3" if args.head_cache == "transposed" else "tiny Llama",
        "head_cache": args.head_cache,
        "tied_weights": args.tied_weights,
        "batch": 4,
        "prompt_tokens": 64,
        "beam_width": 4,
        "decode_steps": 3,
        "prefix_caching": True,
        "multiprocessing": True,
        "graph": args.graph,
        "cases": [],
    }

    def post(client: httpx.Client, route: str, **body: Any) -> Any:
        response = client.post(route, json=body)
        response.raise_for_status()
        return response.json()

    with server(
        output, first, config, "reload", launcher, graph=args.graph, head_cache=args.head_cache
    ) as (stub, control):
        before = outputs(stub.GenerateBatch(request(), timeout=120))
        post(control, "/pause?mode=wait")
        post(control, "/reload_weights", weights_path=str(second), version="b")
        assert control.get("/service/status").json()["service_state"] == "paused"
        post(control, "/resume")
        after = outputs(stub.GenerateBatch(request(), timeout=120))
        assert before != after
        results["cases"].append("A -> B: outputs change with the checkpoint")
        post(control, "/pause?mode=wait")
        queued = stub.GenerateBatch.future(request(), timeout=120)
        time.sleep(0.2)
        assert not queued.done()
        post(control, "/reload_weights", weights_path=str(first), version="a")
        post(control, "/resume")
        assert outputs(queued.result(timeout=120)) == before
        results["cases"].append("B -> A: queued request waits for resume and matches A")
        post(control, "/pause?mode=wait")
        post(control, "/reload_weights", weights_path=str(second), version="b")
        post(control, "/resume")
        assert outputs(stub.GenerateBatch(request(), timeout=120)) == after
        post(control, "/pause?mode=wait")
        failed = control.post(
            "/reload_weights", json={"weights_path": str(output / "missing"), "version": "bad"}
        )
        assert failed.status_code == 500
        status = control.get("/service/status").json()
        assert status["checkpoint_version"] == "b" and not status["checkpoint_valid"]
        assert status["service_state"] == "paused"
        assert control.post("/resume").status_code == 409
        assert (
            control.post(
                "/reload_weights", json={"weights_path": str(second), "version": "b"}
            ).status_code
            == 409
        )
        results["cases"].append(
            "failed reload stays paused and invalid until manual service restart"
        )
    with server(
        output, second, config, "fresh-b", launcher, graph=args.graph, head_cache=args.head_cache
    ) as (stub, control):
        assert control.get("/service/status").json()["checkpoint_valid"]
        fresh = outputs(stub.GenerateBatch(request(), timeout=120))
        assert fresh == after
        results["cases"].append("reloaded B matches fresh B (prefix-cache invalidation)")
    if launcher is not None:
        records = [
            json.loads(line) for line in (output / "cache-memory.jsonl").read_text().splitlines()
        ]
        reloads = [r for r in records if r["event"] == "weight_after"]
        assert len(reloads) == 3
        assert len({r["cache"]["pointer"] for r in reloads}) == 1
        results["cache_reload_checks"] = [r["cache"] for r in reloads]
    results.update(before=before, after=after, passed=True)
    (output / "result.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
