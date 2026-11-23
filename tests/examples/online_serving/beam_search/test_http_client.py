# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
import os
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import pytest

from tests.helpers.paths import REPO_ROOT

pytestmark = pytest.mark.cpu_test

CLIENT = REPO_ROOT / "examples" / "online_serving" / "beam_search" / "client.sh"
SID = "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("status", [200, 404])  # type: ignore[misc]
def test_beam_client_sends_valid_request_from_other_directory(tmp_path: Path, status: int) -> None:
    from examples.offline_inference.beam_search.offline_beam_search import make_prompt

    bash = shutil.which("bash")
    if bash is None or shutil.which("curl") is None:
        pytest.skip("The example requires Bash and curl")
    assert bash is not None

    received: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received["path"] = self.path
            received["auth"] = self.headers.get("Authorization")
            received["payload"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            response: dict[str, object] = (
                {"choices": [{"text": SID}]} if status == 200 else {"error": "model not found"}
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

    prompt = 'A "quoted" path C:\\example\r\n\u63a8\u8350\n'
    prompt_path = tmp_path / "query with spaces.txt"
    prompt_path.write_bytes(prompt.encode("utf-8"))
    tokenizer = Mock()
    tokenizer.encode.return_value = [1]
    assert make_prompt(tokenizer, prompt_path, input_len=None) == [1]
    tokenizer.encode.assert_called_once_with(prompt, add_special_tokens=False)
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            env = {
                **os.environ,
                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
                "ONEREC_BASE_URL": f"http://127.0.0.1:{server.server_port}/",
                "ONEREC_SERVED_MODEL_NAME": "onerec" if status == 200 else "wrong-model",
                "VLLM_API_KEY": "test-key",
            }
            cmd = [bash, CLIENT.as_posix(), "-p", prompt_path.as_posix()]
            completed = subprocess.run(
                cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15, check=False
            )
        finally:
            server.shutdown()
            thread.join()

    assert received["path"] == "/v1/completions", completed.stderr
    assert received["auth"] == "Bearer test-key"
    assert received["payload"] == {
        "model": env["ONEREC_SERVED_MODEL_NAME"],
        "prompt": prompt,
        "use_beam_search": True,
        "n": 128,
        "temperature": 0.0,
        "max_tokens": 5,
        "ignore_eos": True,
        "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
    }
    assert (completed.returncode == 0) == (status == 200)
    assert ("choices" if status == 200 else "error") in json.loads(completed.stdout)
