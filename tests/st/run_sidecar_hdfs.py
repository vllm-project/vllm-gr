# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Real HDFS -> Sidecar subprocess -> two GPU gRPC services; isolated test resources."""

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from vllm_gr.config.models import deployment_for


def install_control_trace() -> None:
    from fastapi.responses import JSONResponse

    from vllm_gr.entrypoints.serve import management

    original = management.control_app

    def control_app(engine: Any) -> Any:
        app = original(engine)
        root = Path(os.environ["GR_SIDECAR_TEST_OUTPUT"])
        port = deployment_for(engine.vllm_config).admin_port

        @app.middleware("http")  # type: ignore[misc]
        async def trace(request: Any, call_next: Any) -> Any:
            body = json.loads(await request.body()) if request.method == "POST" else {}

            def log(event: str, status: int = 0) -> None:
                with (root / "control.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            dict(
                                event=event,
                                ns=time.monotonic_ns(),
                                port=port,
                                path=request.url.path,
                                body=body,
                                status=status,
                            )
                        )
                        + "\n"
                    )

            log("begin")
            fault = root / f"{port}.fail"
            if fault.exists() and fault.read_text() == request.url.path:
                fault.unlink()
                log("end", 503)
                return JSONResponse({"detail": "test-injected control failure"}, status_code=503)
            response = await call_next(request)
            log("end", response.status_code)
            return response

        return app

    management.control_app = control_app


def main() -> None:
    from tests.st.run_checkpoint_reload import ROOT, outputs, prepare, request, server
    from tests.st.run_trie_reload import A, B
    from tests.st.run_trie_reload import prepare as prepare_trie
    from vllm_gr.sidecar import HDFSDownloader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdfs-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    fs = HDFSDownloader()
    stamp = int(time.time())
    remote = args.hdfs_root.rstrip("/") + f"/sidecar_e2e_{stamp}_{os.getpid()}"
    (root / "remote.json").write_text(json.dumps({"remote": remote, "binary": fs.binary}))
    os.environ["GR_SIDECAR_TEST_OUTPUT"] = str(root)
    (root / "control.jsonl").write_text("")
    first, second, _ = prepare(root / "weights")
    _, config, trie_a, trie_b = prepare_trie(root / "trie")
    for path in (first, second):
        (path / "_SUCCESS").touch()
    bad = root / "invalid"
    shutil.copytree(second, bad, dirs_exist_ok=True)
    with (bad / "config.json").open("a") as stream:
        stream.write("\n ")
    vb, invalid, va, vb2 = (f"ckpt_{stamp + i}" for i in range(4))
    tb, ta = f"trie_{stamp}", f"trie_{stamp + 1}"

    def hdfs(command: str, *items: str) -> None:
        argv = fs._argv(command, *items)
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        with (root / "uploads.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        argv=argv,
                        returncode=result.returncode,
                        stdout=result.stdout,
                        stderr=result.stderr,
                    )
                )
                + "\n"
            )
        if result.returncode:
            raise RuntimeError(f"HDFS {command}: {result.stderr}")

    def publish(source: Path, resource: str, version: str, ready: bool = True) -> str:
        target = f"{remote}/{resource}/{version}"
        hdfs("mkdir", "-p", target)
        files = (
            [source]
            if source.is_file()
            else sorted(p for p in source.iterdir() if p.name != "_SUCCESS")
        )
        hdfs("put", *(str(p) for p in files), target)
        if ready:
            hdfs("touchz", f"{target}/_SUCCESS")
        return target

    print(f"Uploading test fixtures to {remote}", flush=True)
    publish(second, "checkpoints", vb)
    waiting_a = publish(first, "checkpoints", va, ready=False)
    waiting_trie = publish(trie_b, "tries", tb, ready=False)
    print("HDFS upload complete; starting two graph-enabled GPU services", flush=True)
    report: dict[str, Any] = {
        "remote": remote,
        "instances": 2,
        "graph": True,
        "runs": [],
        "cases": [],
    }

    async def start(contexts: list[Any]) -> list[Any]:
        return await asyncio.gather(
            *(asyncio.to_thread(cm.__enter__) for cm in contexts), return_exceptions=True
        )

    contexts = [
        server(
            root,
            model,
            config,
            f"node-{i}",
            Path(__file__).resolve(),
            graph=True,
            retention_seconds=1,
        )
        for i, model in enumerate((first, second))
    ]
    with contextlib.ExitStack() as stack:
        handles = asyncio.run(start(contexts))
        for cm, handle in zip(contexts, handles):
            if not isinstance(handle, BaseException):
                stack.callback(cm.__exit__, None, None, None)
        for handle in handles:
            if isinstance(handle, BaseException):
                raise handle
        stubs, controls = zip(*handles)
        ports = [client.base_url.port for client in controls]
        config_file = root / "sidecar-config.py"
        config_file.write_text(
            "from pathlib import Path\nfrom vllm_gr.sidecar import (SidecarConfig, Instance, UpdateTask, "
            "HDFSDownloader, RollingUpdate, SyncUpdate, PausedUpdateProcess, OnlineUpdateProcess)\n"
            + "config = SidecarConfig(instances=["
            + ",".join(f"Instance('node-{i}', {str(c.base_url)!r})" for i, c in enumerate(controls))
            + "], tasks=[\n"
            + f"UpdateTask(name='ckpt', resource_type='checkpoint', address={remote + '/checkpoints'!r}, "
            + f"filesystem=HDFSDownloader(binary={fs.binary!r}), target_dir=Path({str(root / 'cache/ckpt')!r}), "
            + "update_mode=RollingUpdate(max_concurrent_updates=1), update_process=PausedUpdateProcess()),\n"
            + f"UpdateTask(name='trie', resource_type='trie', address={remote + '/tries'!r}, "
            + f"filesystem=HDFSDownloader(binary={fs.binary!r}), target_dir=Path({str(root / 'cache/trie')!r}), "
            + "update_mode=SyncUpdate(), update_process=OnlineUpdateProcess()),\n])\n"
        )

        def generate() -> list[Any]:
            with ThreadPoolExecutor(max_workers=2) as pool:
                return list(
                    pool.map(
                        lambda stub: outputs(stub.GenerateBatch(request(), timeout=180)), stubs
                    )
                )

        def states() -> list[Any]:
            return [c.get("/service/status").json() for c in controls]

        def run(label: str, success: bool = True, error_contains: str = "") -> list[Any]:
            print(f"Sidecar: {label}", flush=True)
            begin = time.monotonic_ns()
            attempts = []
            for attempt in range(5):
                name = label if attempt == 0 else f"{label}-retry-{attempt}"
                log_path = root / f"{name}.log"
                with log_path.open("w") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "vllm_gr.sidecar",
                            "--config",
                            str(config_file),
                            "--once",
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=log,
                    )
                    try:
                        code = process.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        raise
                attempts.append(dict(pid=process.pid, returncode=code, log=str(log_path)))
                text = log_path.read_text()
                matched = (code == 0) if success else (code != 0 and error_contains in text)
                if matched:
                    break
                # Preserve real HDFS errors and retry only temporary transport failures.
                # Each retry is a fresh production Sidecar process querying live state.
                transient = "HDFS " in text and any(
                    message in text
                    for message in (
                        "Connection reset by peer",
                        "Connection timed out",
                        "HDFS test timed out",
                        "HDFS ls timed out",
                        "HDFS get timed out",
                    )
                )
                if not transient or attempt == 4:
                    raise AssertionError(f"See {log_path}")
                print(f"Transient HDFS failure retained in {log_path}; retrying", flush=True)
                time.sleep(1)
            entries = [
                json.loads(line) for line in (root / "control.jsonl").read_text().splitlines()
            ]
            posts = [e for e in entries if e["ns"] >= begin and e["body"]]
            report["runs"].append(dict(label=label, attempts=attempts, posts=posts))
            (root / "progress.json").write_text(json.dumps(report, indent=2))
            return posts

        def mutations(events: list[Any], port: int | None = None) -> list[str]:
            return [
                e["path"]
                for e in events
                if e["event"] == "begin" and (port is None or e["port"] == port)
            ]

        def serial(events: list[Any]) -> None:
            finish = max(
                e["ns"]
                for e in events
                if e["port"] == ports[0] and e["path"] == "/resume" and e["event"] == "end"
            )
            begin = min(
                e["ns"]
                for e in events
                if e["port"] == ports[1] and e["path"] == "/pause" and e["event"] == "begin"
            )
            assert finish <= begin

        reference_a, reference_b = generate()
        assert reference_a != reference_b
        print("Both GPU reference requests complete", flush=True)
        events = run("01-ready-version-and-rolling")
        for port in ports:
            assert mutations(events, port) == ["/pause", "/reload_weights", "/resume"]
        serial(events)
        assert all(
            s["service_state"] == "running" and s["checkpoint_version"] == vb for s in states()
        )
        assert generate() == [reference_b, reference_b]
        report["cases"].append(
            "Real download; unfinished newer version skipped; rolling max=1; both services match B"
        )
        assert not run("02-no-change")
        report["cases"].append("Same versions cause no pause/reload/resume")

        publish(bad, "checkpoints", invalid)
        events = run(
            "03-default-checker-rejection",
            success=False,
            error_contains="Resource checker rejected the update",
        )
        assert not events
        assert all(
            s["service_state"] == "running" and s["checkpoint_version"] == vb for s in states()
        )
        assert generate() == [reference_b, reference_b]
        report["cases"].append(
            "Downloaded checkpoint with different file size rejected before pause"
        )

        hdfs("touchz", f"{waiting_a}/_SUCCESS")
        (root / f"{ports[1]}.fail").write_text("/reload_weights")
        events = run("04-partial-rollout-failure", success=False, error_contains="HTTP Error 503")
        current = states()
        assert current[0]["checkpoint_version"] == va and current[0]["service_state"] == "running"
        assert current[1]["checkpoint_version"] == vb and current[1]["service_state"] == "paused"
        assert mutations(events, ports[1]) == ["/pause", "/reload_weights"]
        events = run("05-restart-reconcile")
        assert not mutations(events, ports[0])
        assert mutations(events, ports[1]) == ["/pause", "/reload_weights", "/resume"]
        assert generate() == [reference_a, reference_a]
        report["cases"].append(
            "Injected reload RPC failure; new Sidecar process updates only the remaining paused instance"
        )

        publish(second, "checkpoints", vb2)
        (root / f"{ports[0]}.fail").write_text("/resume")
        run("06-resume-failure", success=False, error_contains="HTTP Error 503")
        current = states()
        assert current[0]["checkpoint_version"] == vb2 and current[0]["service_state"] == "paused"
        assert current[1]["checkpoint_version"] == va and current[1]["service_state"] == "running"
        events = run("07-restart-resume-only")
        assert mutations(events, ports[0]) == ["/resume"]
        assert mutations(events, ports[1]) == ["/pause", "/reload_weights", "/resume"]
        assert generate() == [reference_b, reference_b]
        report["cases"].append(
            "Injected resume failure; restart resumes committed instance without another reload"
        )

        def membership(rows: list[Any], allowed: set[tuple[int, ...]]) -> None:
            for batch in rows:
                assert len(batch) == 4
                for item in batch:
                    ids, offsets = item["ids"], item["offsets"]
                    assert {tuple(ids[a:b]) for a, b in zip(offsets, offsets[1:])} == allowed

        hdfs("touchz", f"{waiting_trie}/_SUCCESS")
        events = run("08-online-trie")
        assert mutations(events) and set(mutations(events)) == {"/reload_trie"}
        assert {e["port"] for e in events} == set(ports)
        membership(generate(), B)
        time.sleep(1.1)
        membership(generate(), B)
        assert all(s["service_state"] == "running" and s["trie_version"] == tb for s in states())
        report["cases"].append(
            "Real Trie download; SyncUpdate reloads both services without pause; B candidates survive expiry"
        )
        publish(trie_a, "tries", ta)
        run("09-trie-back-to-a")
        membership(generate(), A)
        assert generate() == [reference_b, reference_b]
        assert not run("10-final-no-change")
        report["cases"].append(
            "Trie returns to A; unchanged resources are skipped by a new Sidecar process"
        )
        report.update(passed=True, final_status=states())
    (root / "result.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({"passed": True, "cases": report["cases"], "remote": remote}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        sys.argv.pop(1)
        install_control_trace()
        from vllm_gr.entrypoints.cli.main import dispatch_main

        dispatch_main()
    else:
        main()
