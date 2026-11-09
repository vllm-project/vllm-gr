# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real HTTP calls; typed service state and committed versions."""

import asyncio
import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from vllm_gr.sidecar import (
    ControlClient,
    Instance,
    OnlineUpdateProcess,
    PausedUpdateProcess,
    UpdateContext,
    UpdateError,
)


@pytest.fixture  # type: ignore[misc]
def server() -> Iterator[tuple[Instance, list[tuple[str, dict[str, Any]]], dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    state: dict[str, Any] = {
        "service_state": "running",
        "checkpoint_valid": True,
        "checkpoint_version": "old",
        "trie_version": "old",
        "checkpoint_path": None,
        "trie_path": None,
        "paused": False,
        "reloads": 0,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            events.append((self.path, body))
            route = urlsplit(self.path).path
            if route == "/pause":
                state.update(paused=True, service_state="paused")
            elif route in {"/reload_weights", "/reload_trie"}:
                # Only the service publishes the new version after loading.
                state["reloads"] += 1
                resource = "checkpoint" if route == "/reload_weights" else "trie"
                state[f"{resource}_version"] = body["version"]
                state[f"{resource}_path"] = body.get("weights_path", body.get("trie_path"))
            elif route == "/resume":
                state.update(paused=False, service_state="running")
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_GET(self) -> None:
            assert self.path == "/service/status"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(state).encode())

        def log_message(self, format: str, *args: Any) -> None:
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield Instance("node", f"http://127.0.0.1:{http.server_port}"), events, state
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


@pytest.mark.l2  # type: ignore[misc]
def test_paused_process_real_http(
    server: tuple[Instance, list[tuple[str, dict[str, Any]]], dict[str, Any]], tmp_path: Path
) -> None:
    instance, events, state = server
    process = PausedUpdateProcess(ControlClient())
    context = UpdateContext("weights", "checkpoint", "v1", tmp_path)
    assert asyncio.run(process.run(instance, context, checker=lambda old, new: True))
    assert [path for path, _ in events] == ["/pause?mode=wait", "/reload_weights", "/resume"]
    assert all(
        body
        == dict(
            version="v1",
            resource_type="checkpoint",
            **({"weights_path": str(tmp_path)} if path == "/reload_weights" else {}),
        )
        for path, body in events
    )
    assert state["reloads"] == 1
    assert not asyncio.run(process.run(instance, context, checker=lambda old, new: True))
    assert len(events) == 3


@pytest.mark.l2  # type: ignore[misc]
def test_online_process_real_http(
    server: tuple[Instance, list[tuple[str, dict[str, Any]]], dict[str, Any]], tmp_path: Path
) -> None:
    instance, events, state = server
    context = UpdateContext("trie", "trie", "v1", tmp_path / "trie.safetensors")
    process = OnlineUpdateProcess(ControlClient())
    assert asyncio.run(process.run(instance, context, checker=lambda old, new: True))
    assert [path for path, _ in events] == ["/reload_trie"]
    assert events[0][1]["trie_path"] == str(context.local_path)
    assert not asyncio.run(process.run(instance, context, checker=lambda old, new: True))
    state["trie_version"] = "old"
    assert asyncio.run(process.run(instance, context, checker=lambda old, new: True))
    assert state["reloads"] == 2


@pytest.mark.l1  # type: ignore[misc]
def test_sidecar_import_does_not_import_inference_runtime() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vllm_gr.sidecar; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.l2  # type: ignore[misc]
def test_hdfs_to_http_task_end_to_end_without_progress_files(
    server: tuple[Instance, list[tuple[str, dict[str, Any]]], dict[str, Any]],
    hdfs: Any,
    tmp_path: Path,
) -> None:
    from vllm_gr.sidecar import RollingUpdate, Sidecar, SidecarConfig, UpdateTask

    fs, remote, _ = hdfs
    source = remote / "models/ckpt_1700000000000"
    source.mkdir(parents=True)
    (source / "_SUCCESS").touch()
    (source / "config.json").write_text("{}")
    (source / "model.bin").write_bytes(b"weights")
    instance, events, state = server
    import shutil

    old = tmp_path / "old"
    shutil.copytree(source, old)
    state["checkpoint_path"] = str(old)
    cfg = SidecarConfig(
        instances=[instance],
        tasks=[
            UpdateTask(
                name="ckpt",
                filesystem=fs,
                address="hdfs://cluster/models",
                target_dir=tmp_path / "cache",
                resource_type="checkpoint",
                update_mode=RollingUpdate(),
                update_process=PausedUpdateProcess(ControlClient()),
            )
        ],
    )
    assert asyncio.run(Sidecar(cfg).run_once("ckpt"))
    assert [path for path, _ in events] == ["/pause?mode=wait", "/reload_weights", "/resume"]
    assert Path(events[1][1]["weights_path"]) == tmp_path / "cache/ckpt_1700000000000"
    assert not asyncio.run(Sidecar(cfg).run_once("ckpt"))
    assert len(events) == 3 and state["reloads"] == 1
    assert not (tmp_path / "state").exists()


@pytest.mark.l2  # type: ignore[misc]
@pytest.mark.parametrize("invalid", ["idle", "failed", "loading", None, 1])  # type: ignore[misc]
def test_client_rejects_invalid_service_state(
    server: tuple[Instance, list[tuple[str, dict[str, Any]]], dict[str, Any]], invalid: Any
) -> None:
    from vllm_gr.update_status import ServiceState

    instance, events, state = server
    status = asyncio.run(ControlClient().status(instance))
    assert status.service_state is ServiceState.RUNNING
    state["service_state"] = invalid
    with pytest.raises(UpdateError, match="Invalid service status"):
        asyncio.run(ControlClient().status(instance))
    assert not events


@pytest.mark.l2  # type: ignore[misc]
def test_loaded_paused_checkpoint_only_resumes(
    server: tuple[Instance, list[tuple[str, dict[str, Any]]], dict[str, Any]], tmp_path: Path
) -> None:
    instance, events, state = server
    state.update(service_state="paused", checkpoint_version="v1", paused=True)
    context = UpdateContext("weights", "checkpoint", "v1", tmp_path)
    assert asyncio.run(PausedUpdateProcess(ControlClient()).run(instance, context))
    assert [path for path, _ in events] == ["/resume"]


@pytest.mark.l2  # type: ignore[misc]
def test_client_requires_explicit_checkpoint_validity(
    server: tuple[Instance, list[tuple[str, dict[str, Any]]], dict[str, Any]],
) -> None:
    instance, _, state = server
    del state["checkpoint_valid"]
    with pytest.raises(UpdateError, match="Invalid service status"):
        asyncio.run(ControlClient().status(instance))


pytestmark = pytest.mark.cpu_test
