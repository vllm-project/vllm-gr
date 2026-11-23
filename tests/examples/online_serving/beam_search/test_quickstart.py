# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from examples.online_serving.beam_search import quickstart
from tests.helpers.paths import REPO_ROOT

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.l1,
    pytest.mark.skipif(os.name != "posix", reason="The quickstart uses POSIX process groups"),
]

EXAMPLES = REPO_ROOT / "examples" / "online_serving"
QUICKSTART = EXAMPLES / "beam_search" / "quickstart.py"
PROMPT = 'An "exact" prompt\r\n\u63a8\u8350\n'
SID = "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"


def _fake_compiler() -> None:
    workdir = Path(sys.argv[sys.argv.index("--workdir") + 1])
    catalog = Path(sys.argv[sys.argv.index("--raw-catalog") + 1])
    assert catalog == EXAMPLES / "catalog" / "sample_catalog.json"
    if os.environ.get("TEST_BLOCK_COMPILER"):
        (workdir / "compiler-pid").write_text(str(os.getpid()), encoding="ascii")
        time.sleep(60)
    (workdir / "vllm_gr_config.yaml").touch()


def _fake_vllm() -> None:
    scenario = os.environ["TEST_SERVE_SCENARIO"]
    if scenario == "exit":
        sys.exit(17)
    port = int(sys.argv[sys.argv.index("--port") + 1])
    name = sys.argv[sys.argv.index("--served-model-name") + 1]
    run_dir = Path(sys.argv[sys.argv.index("--config") + 1]).parent
    (run_dir / "server-pid").write_text(str(os.getpid()), encoding="ascii")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.headers.get("Authorization") != "Bearer test-key":
                self.send_error(401)
                return
            self.send_response(200)
            self.end_headers()
            if self.path == "/v1/models":
                model = "unrelated-model" if scenario == "wrong-model" else name
                self.wfile.write(json.dumps({"data": [{"id": model}]}).encode())

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["model"] == name and request["n"] == 128
            assert request["prompt"] == PROMPT
            content = SID.replace("<s_c_3>", "<s_c_9>") if scenario == "bad-sid" else SID
            choices = [] if scenario == "empty" else [{"text": content}]
            if scenario == "success":
                choices.append({"text": SID.replace("<s_c_3>", "<s_c_4>")})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"choices": choices}).encode())

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        # The worker keeps the listener open until the service reaps it.
        worker = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            pass_fds=(server.fileno(),),
        )
        retry_shutdown = bool(os.environ.get("TEST_RETRY_SHUTDOWN"))

        def stop(signum: int, frame: object) -> None:
            nonlocal retry_shutdown
            if retry_shutdown:
                retry_shutdown = False
                return
            worker.terminate()
            worker.wait(timeout=5)
            sys.exit(0)

        signal.signal(signal.SIGTERM, stop)
        runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "unset")
        print(f"runner={runner}", flush=True)
        (run_dir / "port").write_text(str(port), encoding="ascii")
        server.serve_forever()


@pytest.fixture  # type: ignore[misc]
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    if not shutil.which("bash") or not shutil.which("curl"):
        pytest.skip("The example requires Bash and curl")

    # Let the real CLI launch stand-ins through PATH and Python module lookup.
    fake_call = f"import runpy\nrunpy.run_path({str(Path(__file__).resolve())!r})"
    vllm = tmp_path / "vllm"
    vllm.write_text(f"#!{sys.executable}\n{fake_call}['_fake_vllm']()\n", encoding="utf-8")
    vllm.chmod(0o700)
    fake_examples = tmp_path / "examples"
    catalog = fake_examples / "online_serving" / "catalog"
    catalog.mkdir(parents=True)
    (fake_examples / "__init__.py").write_text(
        f"__path__.append({str(EXAMPLES.parent)!r})\n", encoding="utf-8"
    )
    (catalog / "prepare_serving_config.py").write_text(
        f"{fake_call}['_fake_compiler']()\n", encoding="utf-8"
    )

    runs = tmp_path / "saved runs"
    prompt = tmp_path / "custom prompt"
    prompt.write_bytes(PROMPT.encode())
    monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(tmp_path), str(REPO_ROOT))))
    monkeypatch.setenv("TEST_SERVE_SCENARIO", "success")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setattr(quickstart, "STARTUP_TIMEOUT", 15)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(QUICKSTART),
            "--model",
            str(tmp_path),
            "--workdir",
            str(runs),
            "--prompt",
            str(prompt),
        ],
    )
    yield runs

    # Keep a failing test from leaving its stand-ins alive.
    if sys.platform != "win32":
        for path in runs.glob("*/server-pid"):
            pid = path.read_text(encoding="ascii")
            if pid:
                with suppress(ProcessLookupError):
                    os.killpg(int(pid), signal.SIGKILL)
        for path in runs.glob("*/compiler-pid"):
            pid = path.read_text(encoding="ascii")
            if pid:
                with suppress(ProcessLookupError):
                    os.kill(int(pid), signal.SIGKILL)


def _assert_port_closed(run_dir: Path) -> None:
    port = int((run_dir / "port").read_text(encoding="ascii"))
    with socket.socket() as probe:
        probe.settimeout(2)
        assert probe.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.parametrize(("scenario", "count"), [("short", 1), ("success", 2)])  # type: ignore[misc]
def test_http_quickstart_verifies_results(
    scenario: str,
    count: int,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TEST_SERVE_SCENARIO", scenario)
    quickstart.main()

    (run_dir,) = workdir.iterdir()
    stdout, _ = capfd.readouterr()
    assert str(run_dir) in stdout
    assert (
        len(json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["choices"]) == count
    )
    assert SID in stdout and f"SUCCESS: All {count}" in stdout
    _assert_port_closed(run_dir)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("backend", "override", "runner"),
    [("cuda", None, "unset"), ("cuda", "0", "0"), ("cuda", "1", "1"), ("npu", "1", "0")],
)
def test_http_quickstart_selects_runner(
    backend: str, override: str | None, runner: str, workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if override is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER")
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", override)
    sys.argv.extend(["--constraint-backend", backend])
    quickstart.main()
    (run_dir,) = workdir.iterdir()
    assert f"runner={runner}" in (run_dir / "server.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("scenario", ["empty", "bad-sid"])  # type: ignore[misc]
def test_http_quickstart_rejects_invalid_results(
    scenario: str,
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TEST_SERVE_SCENARIO", scenario)
    with pytest.raises(RuntimeError, match="SID verification failed"):
        quickstart.main()
    (run_dir,) = workdir.iterdir()
    assert (run_dir / "result.json").is_file()
    assert "FAILURE:" in capfd.readouterr().err
    _assert_port_closed(run_dir)


def test_http_quickstart_reports_early_server_exit(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_SERVE_SCENARIO", "exit")
    with pytest.raises(RuntimeError, match=r"exited during startup \(17\)"):
        quickstart.main()
    assert not list(workdir.glob("*/result.json"))


def test_http_quickstart_times_out_waiting_for_its_model(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_SERVE_SCENARIO", "wrong-model")
    monkeypatch.setattr(quickstart, "STARTUP_TIMEOUT", 3)
    with pytest.raises(TimeoutError, match="waiting for model"):
        quickstart.main()
    (run_dir,) = workdir.iterdir()
    assert not (run_dir / "result.json").exists()
    _assert_port_closed(run_dir)


def test_http_quickstart_stops_process_group_when_server_ignores_first_sigterm(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_RETRY_SHUTDOWN", "1")
    monkeypatch.setattr(quickstart, "ENGINE_SHUTDOWN_TIMEOUT", 0)
    quickstart.main()
    (run_dir,) = workdir.iterdir()
    _assert_port_closed(run_dir)


def _signal_quickstart(workdir: Path, ready_file: str, signum: int, expected_exit: int) -> Path:
    output = workdir.parent / "quickstart.log"
    with (
        output.open("wb") as stream,
        subprocess.Popen(
            [sys.executable, *sys.argv], stdout=stream, stderr=subprocess.STDOUT
        ) as process,
    ):
        try:
            deadline = time.monotonic() + 15
            while process.poll() is None and time.monotonic() < deadline:
                for path in workdir.glob(f"*/{ready_file}"):
                    if path.stat().st_size:
                        process.send_signal(signum)
                        assert process.wait(timeout=20) == expected_exit, output.read_text(
                            encoding="utf-8"
                        )
                        return path
                time.sleep(0.05)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    raise AssertionError(output.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("signum", "exit_code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])  # type: ignore[misc]
def test_http_quickstart_stops_server_on_signal(
    signum: int, exit_code: int, workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_SERVE_SCENARIO", "wrong-model")
    port = _signal_quickstart(workdir, "port", signum, exit_code)
    _assert_port_closed(port.parent)
    assert not list(workdir.glob("*/result.json"))


def test_http_quickstart_stops_compiler_on_sigterm(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_BLOCK_COMPILER", "1")
    compiler = _signal_quickstart(workdir, "compiler-pid", signal.SIGTERM, 143)
    with pytest.raises(ProcessLookupError):
        os.kill(int(compiler.read_text(encoding="ascii")), 0)
    assert not list(workdir.glob("*/server.log"))
