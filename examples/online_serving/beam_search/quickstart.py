#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Start a local OneRec HTTP server, verify one Beam Search request, then stop it."""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.client import HTTPException
from pathlib import Path
from typing import BinaryIO
from urllib.request import ProxyHandler, Request, build_opener

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CATALOG = HERE.parent / "catalog"
STARTUP_TIMEOUT = 600
ENGINE_SHUTDOWN_TIMEOUT = 10
SHUTDOWN_TIMEOUT = 30


def wait_for_model(process: subprocess.Popen[bytes], env: dict[str, str], log: Path) -> None:
    base_url = env["ONEREC_BASE_URL"]
    model = env["ONEREC_SERVED_MODEL_NAME"]
    headers = {"Authorization": "Bearer " + env["VLLM_API_KEY"]} if env.get("VLLM_API_KEY") else {}
    # A loopback probe must reach the process that is loading our model.
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited during startup ({process.returncode}). See {log}")
        try:
            with opener.open(Request(base_url + "/health", headers=headers), timeout=2):
                pass
            with opener.open(Request(base_url + "/v1/models", headers=headers), timeout=2) as resp:
                body = json.load(resp)
                models = body.get("data", []) if isinstance(body, dict) else []
                if isinstance(models, list) and any(
                    isinstance(item, dict) and item.get("id") == model for item in models
                ):
                    return
        except (OSError, HTTPException, ValueError):
            pass  # The HTTP frontend can become available before the engine.
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for model {model}. See {log}")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT
    # Give the API server time to drain and reap its own engine.
    process.terminate()
    try:
        process.wait(timeout=ENGINE_SHUTDOWN_TIMEOUT + 5)
    except subprocess.TimeoutExpired:
        pass

    # A worker can still hold a device after the CLI has exited or stalled.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        while time.monotonic() < deadline:
            process.poll()  # Reap the leader so an empty group disappears.
            os.killpg(process.pid, 0)
            time.sleep(0.1)
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait()


def stop_compiler(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    # The compiler rolls back its unpublished output on KeyboardInterrupt.
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@contextmanager
def managed_process(
    command: list[str],
    *,
    stop: Callable[[subprocess.Popen[bytes]], None],
    env: dict[str, str] | None = None,
    output: BinaryIO | None = None,
) -> Iterator[subprocess.Popen[bytes]]:
    process: subprocess.Popen[bytes] | None = None
    interrupted_by: int | None = None
    interruptible = False

    def interrupt(signum: int, frame: object) -> None:
        nonlocal interrupted_by
        if interrupted_by is None:
            interrupted_by = signum
            if interruptible:
                if signum == signal.SIGINT:
                    raise KeyboardInterrupt
                raise SystemExit(128 + signum)

    # Defer cancellation during launch until we have a process handle, and
    # during shutdown until the child has been reaped.
    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT if output is not None else None,
            start_new_session=True,
        )
        interruptible = True
        if interrupted_by is None:
            yield process
    finally:
        interruptible = False
        try:
            if process is not None:
                stop(process)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)

    if interrupted_by == signal.SIGINT:
        raise KeyboardInterrupt
    if interrupted_by is not None:
        raise SystemExit(128 + interrupted_by)


def prepare_config(args: argparse.Namespace, workdir: Path, catalog: Path) -> Path:
    source = (
        ["--triples", str(args.catalog.resolve())]
        if args.catalog
        else ["--raw-catalog", str(catalog)]
    )
    command = [
        sys.executable,
        "-m",
        "examples.online_serving.catalog.prepare_serving_config",
        "--model",
        str(args.model.resolve()),
        "--constraint-backend",
        args.constraint_backend,
        "--workdir",
        str(workdir),
        *source,
    ]
    with managed_process(command, stop=stop_compiler) as compiler:
        status = compiler.wait()
    if status:
        raise RuntimeError(
            f"Serving configuration preparation failed (exit {status}). See {workdir}"
        )
    config = workdir / "vllm_gr_config.yaml"
    if not config.is_file():
        raise RuntimeError(f"Serving configuration was not created at {config}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local OneRec model snapshot")
    parser.add_argument(
        "--prompt", type=Path, help='UTF-8 prompt file (default: "Recommend the next video.")'
    )
    parser.add_argument("--constraint-backend", choices=("cuda", "npu"), default="cuda")
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Constraint triples JSON (default: the provided two-SID sample)",
    )
    parser.add_argument(
        "--verify-catalog",
        type=Path,
        help="Source catalog with SID markers (default with --catalog: adjacent catalog.json)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Parent of the retained run directory (default: system temporary directory)",
    )
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("The HTTP quickstart requires a POSIX host for service and worker cleanup.")
    if not args.model.is_dir():
        parser.error("--model must be a local model directory")
    if args.prompt:
        args.prompt = args.prompt.resolve()
        args.prompt.read_text(encoding="utf-8")
    if args.verify_catalog and not args.catalog:
        parser.error("--verify-catalog requires --catalog with constraint triples")
    catalog = CATALOG / "sample_catalog.json"
    if args.catalog:
        if not args.catalog.is_file():
            parser.error("--catalog must point to a constraint triples file")
        catalog = (args.verify_catalog or args.catalog.with_name("catalog.json")).resolve()
        if not catalog.is_file():
            parser.error(
                f"Missing source catalog {catalog}. Pass --verify-catalog FILE with the original SIDs."
            )
    for executable in ("vllm", "bash", "curl"):
        if not shutil.which(executable):
            parser.error(f"Required command not found on PATH: {executable}")
    if args.workdir:
        args.workdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(dir=args.workdir, prefix="onerec-http-")).resolve()
    print(f"Run directory: {workdir}", flush=True)
    config = prepare_config(args, workdir, catalog)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {
        **os.environ,
        "ONEREC_BASE_URL": f"http://127.0.0.1:{port}",
        "ONEREC_SERVED_MODEL_NAME": workdir.name,
    }
    if args.constraint_backend == "npu":
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    command = [
        "vllm",
        "serve",
        str(args.model.resolve()),
        "--gr",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        workdir.name,
        "--config",
        str(config),
        "--trust-remote-code",
        "--max-logprobs",
        "128",
        "--max-num-seqs",
        "128",
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "2048",
        "--async-scheduling",
        "--no-enable-chunked-prefill",
        "--shutdown-timeout",
        str(ENGINE_SHUTDOWN_TIMEOUT),
    ]
    log = workdir / "server.log"
    result = workdir / "result.json"
    print(f"Starting server. Log: {log}", flush=True)
    with (
        log.open("wb") as stream,
        managed_process(command, stop=stop_server, env=env, output=stream) as process,
    ):
        wait_for_model(process, env, log)
        client = ["bash", str(HERE / "client.sh")]
        if args.prompt:
            client += ["-p", str(args.prompt)]
        print(f"Sending request. Response: {result}", flush=True)
        with result.open("wb") as output:
            # Keep local inference on loopback even on hosts with a global HTTP proxy.
            client_env = {**env, "no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}
            request = subprocess.run(client, cwd=ROOT, env=client_env, stdout=output)
        if request.returncode:
            raise RuntimeError(
                f"HTTP request failed (exit {request.returncode}). See {result} and {log}"
            )
        verifier = subprocess.run(
            [
                sys.executable,
                str(CATALOG / "verify_catalog.py"),
                str(result),
                str(catalog),
                "--show-sids",
            ],
        )
        if verifier.returncode:
            raise RuntimeError(f"SID verification failed. See {result} and {catalog}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(str(exc))
