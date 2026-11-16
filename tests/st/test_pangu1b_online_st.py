# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"

"""System test: Pangu 1B online serving via ``vllm-gr serve``.

Requires:
- vLLM + GPU/NPU runtime
- ``PANGU1B_MODEL_PATH`` or ``PANGU_MODEL_PATH`` with weights

Skip when the model path is unavailable (CI without weights).
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import IO, Any, Iterator
from unittest import mock

import pytest
import requests

from tests.st.pangu1b_st_common import get_model_path, require_pangu_st_runtime

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "tests" / "resources" / "pangu1b" / "gsm8k_mini.json"
PANGU_EXAMPLE_DIR = ROOT / "examples" / "offline_inference" / "pangu1b"
DEFAULT_SERVE_TIMEOUT = int(os.environ.get("PANGU1B_SERVE_TIMEOUT", "600"))
SERVED_MODEL_NAME = os.environ.get("PANGU1B_SERVED_MODEL_NAME", "pangu1b-st")


def _get_min_accuracy() -> float:
    """Read the online accuracy threshold as a numeric value."""
    return float(
        os.environ.get(
            "PANGU1B_ONLINE_ACCURACY_THRESHOLD",
            os.environ.get("PANGU1B_MIN_ACCURACY", "0.5"),
        )
    )


DEFAULT_MIN_ACCURACY = _get_min_accuracy()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_serve_cmd(model_path: str, port: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm_gr.entrypoints.cli.main",
        "serve",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "FLASH_ATTN",
        "--max-model-len",
        os.environ.get("PANGU1B_MAX_MODEL_LEN", "4096"),
        "--max-num-seqs",
        os.environ.get("PANGU1B_MAX_NUM_SEQS", "4"),
        "--tokenizer-mode",
        "slow",
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
    ]
    block_size = os.environ.get("PANGU1B_BLOCK_SIZE")
    if block_size is None:
        try:
            from vllm.platforms import current_platform
        except Exception:
            pass
        else:
            if getattr(current_platform, "device_type", None) == "npu":
                block_size = "128"
    if block_size is not None:
        cmd.extend(["--block-size", block_size])
    return cmd


def _read_log_file(log_file: IO[str] | None) -> str:
    if log_file is None:
        return ""
    pos = log_file.tell()
    log_file.seek(0)
    content = log_file.read()
    log_file.seek(pos)
    return content


def _wait_for_server(
    base_url: str,
    proc: subprocess.Popen[Any],
    log_file: IO[str] | None,
    timeout: int,
) -> None:
    deadline = time.time() + timeout
    last_error = "server process exited before becoming ready"

    while time.time() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            logs = _read_log_file(log_file)
            raise RuntimeError(f"{last_error} (exit={exit_code})\nlogs:\n{logs}")

        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                models = requests.get(f"{base_url}/v1/models", timeout=5)
                if models.status_code == 200:
                    return
                last_error = f"/v1/models returned {models.status_code}"
            else:
                last_error = f"/health returned {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)

        time.sleep(2)

    logs = _read_log_file(log_file)
    raise TimeoutError(
        f"Server at {base_url} not ready within {timeout}s: {last_error}\nlogs:\n{logs}"
    )


def _terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


@contextmanager
def _pangu_serve_server(
    model_path: str,
    *,
    port: int | None = None,
    timeout: int = DEFAULT_SERVE_TIMEOUT,
) -> Iterator[str]:
    chosen_port = port if port is not None else _find_free_port()
    base_url = f"http://127.0.0.1:{chosen_port}"
    cmd = _build_serve_cmd(model_path, chosen_port)

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_server(base_url, proc, log_file, timeout)
            yield base_url
        finally:
            _terminate_process(proc)


def _chat_completion(
    base_url: str,
    *,
    content: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("choices"):
        raise AssertionError(f"chat completion returned no choices: {body}")
    return body


@pytest.mark.gpu_test
@pytest.mark.npu_test
@pytest.mark.model_test
class TestPangu1bOnlineServeST(unittest.TestCase):
    """End-to-end online ST for Pangu LLM passthrough via ``vllm-gr serve``."""

    base_url: str
    _server_ctx: AbstractContextManager[str]

    @classmethod
    def setUpClass(cls) -> None:
        require_pangu_st_runtime()
        cls._server_ctx = _pangu_serve_server(get_model_path())
        cls.base_url = cls._server_ctx.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server_ctx.__exit__(None, None, None)

    def test_health_endpoint(self) -> None:
        response = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(response.status_code, 200)

    def test_v1_models_lists_served_model(self) -> None:
        response = requests.get(f"{self.base_url}/v1/models", timeout=10)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        model_ids = {item["id"] for item in payload.get("data", [])}
        self.assertIn(SERVED_MODEL_NAME, model_ids)

    def test_chat_completions_smoke(self) -> None:
        body = _chat_completion(
            self.base_url,
            content="Question: What is 2 + 2?\nAnswer:",
            max_tokens=64,
        )
        message = body["choices"][0]["message"]
        content = message.get("content") or ""
        self.assertTrue(content.strip(), f"empty completion: {body}")

    def test_recommend_api_disabled_for_llm_mode(self) -> None:
        # After removing the GR adapter / recommend runtime, the /recommend
        # router is no longer registered; requests must fail (404) instead of
        # being routed to a recommendation handler.
        payload = {
            "model": SERVED_MODEL_NAME,
            "prompt": {
                "user_id": "u1",
                "user_history": [],
                "candidates": [{"item_id": 1}],
            },
        }
        response = requests.post(
            f"{self.base_url}/recommend",
            json=payload,
            timeout=30,
        )
        self.assertEqual(response.status_code, 404)

    def test_gsm8k_mini_online_accuracy(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "pangu_generate_online_test",
            PANGU_EXAMPLE_DIR / "pangu_generate.py",
        )
        assert spec is not None and spec.loader is not None
        pangu_generate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pangu_generate)
        _normalize_numeric = pangu_generate._normalize_numeric
        build_prompt = pangu_generate.build_prompt
        extract_final_answer = pangu_generate.extract_final_answer
        load_dataset = pangu_generate.load_dataset

        records = load_dataset(DEFAULT_DATASET, num_prompts=8)
        correct = 0
        for row in records:
            body = _chat_completion(
                self.base_url,
                content=build_prompt(str(row["question"])),
                # Allow the model to finish its reasoning and final answer.
                max_tokens=512,
            )
            generated = body["choices"][0]["message"].get("content") or ""
            predicted = extract_final_answer(generated)
            expected = _normalize_numeric(str(row["answer"]))
            if predicted == expected:
                correct += 1

        accuracy = correct / len(records)
        result_path = os.environ.get("PANGU1B_ONLINE_RESULT_JSON")
        if result_path:
            Path(result_path).write_text(
                json.dumps(
                    {
                        "accuracy": accuracy,
                        "correct": correct,
                        "num_samples": len(records),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if os.environ.get("PANGU1B_MEASURE_ONLY", "") != "1":
            self.assertGreaterEqual(accuracy, DEFAULT_MIN_ACCURACY)


@pytest.mark.cpu_test
class TestPangu1bOnlineUnitHooks(unittest.TestCase):
    """Lightweight checks that do not load the full model or start a server."""

    def test_serve_command_builder(self) -> None:
        cmd = _build_serve_cmd("/tmp/pangu", 18080)
        self.assertIn("serve", cmd)
        self.assertNotIn("--model-family", cmd)
        self.assertIn("--served-model-name", cmd)
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], SERVED_MODEL_NAME)

    def test_min_accuracy_environment_value_is_float(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"PANGU1B_ONLINE_ACCURACY_THRESHOLD": "0.75"},
        ):
            threshold = _get_min_accuracy()

        self.assertIsInstance(threshold, float)
        self.assertEqual(threshold, 0.75)

    def test_serve_command_builder_accepts_block_size_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"PANGU1B_BLOCK_SIZE": "64"},
        ):
            cmd = _build_serve_cmd("/tmp/pangu", 18080)

        self.assertEqual(cmd[cmd.index("--block-size") + 1], "64")

    def test_serve_command_builder_defaults_to_128_on_npu(self) -> None:
        fake_platform = mock.Mock(device_type="npu")
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("vllm.platforms.current_platform", fake_platform),
        ):
            os.environ.pop("PANGU1B_BLOCK_SIZE", None)
            cmd = _build_serve_cmd("/tmp/pangu", 18080)

        self.assertEqual(cmd[cmd.index("--block-size") + 1], "128")

    def test_default_dataset_exists(self) -> None:
        self.assertTrue(DEFAULT_DATASET.is_file())


if __name__ == "__main__":
    unittest.main()
