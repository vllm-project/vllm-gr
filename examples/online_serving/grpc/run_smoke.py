# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run the production --grpc CLI with random Llama weights and binary prompts."""

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import grpc
import numpy as np
from client import generate
from prepare_model import prepare_model

from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2_grpc as rpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    output = args.output.resolve()
    model, config = prepare_model(output)
    root = Path(__file__).resolve().parents[3]
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")])
    env.update(VLLM_USE_V2_MODEL_RUNNER="0", HF_HUB_OFFLINE="1")
    command = [
        sys.executable,
        "-m",
        "vllm_gr.entrypoints.cli.main",
        "serve",
        str(model),
        "--vllm-gr-config",
        str(config),
        "--dtype",
        "float16",
        "--grpc",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--async-scheduling",
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "64",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "16",
        "--gpu-memory-utilization",
        "0.1",
        "--enforce-eager",
    ]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    log_path = output / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log)
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                ready = grpc.channel_ready_future(channel)
                deadline = time.monotonic() + args.timeout
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited ({process.returncode}); see {log_path}")
                    try:
                        ready.result(timeout=0.2)
                        break
                    except grpc.FutureTimeoutError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Server startup timed out; see {log_path}")
                stub = rpc.VllmEngineStub(channel)
                rows = np.random.default_rng(7).integers(3, 128, size=(4, 4), dtype=np.int64)
                replies = [generate(stub, rows) for _ in range(2)]
            for reply in replies:
                assert len(reply["items"]) == 4
                for item in reply["items"]:
                    assert len(item["token_ids"]) == len(item["scores"]) == 4
                    assert all(len(row) == 3 for row in item["token_ids"])
                    assert all(0 <= token < 128 for row in item["token_ids"] for token in row)
                    assert all(math.isfinite(score) for score in item["scores"])
                    assert all(a >= b for a, b in zip(item["scores"], item["scores"][1:]))
            assert replies[0]["items"] == replies[1]["items"]
            result = dict(
                model="LlamaForCausalLM",
                seed=7,
                batch_size=4,
                input_tokens=4,
                beam_width=4,
                output_tokens=3,
                calls=2,
                items=replies[0]["items"],
            )
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
