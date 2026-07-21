# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"

"""System test: Pangu 1B accuracy on GSM8K-style mini fixture.

Requires:
- vLLM + GPU/NPU runtime
- ``PANGU1B_MODEL_PATH`` / ``PANGU_MODEL_PATH`` or ``--model`` with weights

Skip when the model path is unavailable (CI without weights).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.st.pangu1b_st_common import get_model_path, require_pangu_st_runtime

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "tests" / "resources" / "pangu1b" / "gsm8k_mini.json"
DEFAULT_MIN_ACCURACY = os.environ.get("PANGU1B_MIN_ACCURACY", "0.5")
PANGU_EXAMPLE = ROOT / "examples" / "offline_inference" / "pangu1b" / "pangu_generate.py"


def _parse_cli_summary(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        raise json.JSONDecodeError("empty stdout", "", 0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


class TestPangu1bAccuracyST(unittest.TestCase):
    """End-to-end accuracy ST for Pangu via offline inference example."""

    def setUp(self) -> None:
        require_pangu_st_runtime()

    def test_gsm8k_mini_accuracy_cli(self) -> None:
        model_path = get_model_path()
        with tempfile.NamedTemporaryFile(
            suffix=".json",
            delete=False,
        ) as handle:
            output_json = Path(handle.name)

        cmd = [
            sys.executable,
            str(PANGU_EXAMPLE),
            "--model",
            model_path,
            "--dataset-path",
            str(DEFAULT_DATASET),
            "--num-prompts",
            "8",
            "--min-accuracy",
            DEFAULT_MIN_ACCURACY,
            "--output-json",
            str(output_json),
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(
                    f"Pangu accuracy ST failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )

            if output_json.is_file() and output_json.stat().st_size > 0:
                with output_json.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
            else:
                try:
                    summary = _parse_cli_summary(proc.stdout)
                except json.JSONDecodeError as exc:
                    self.fail(
                        "Failed to parse example JSON summary\n"
                        f"stdout:\n{proc.stdout}\n"
                        f"stderr:\n{proc.stderr}\n"
                        f"parse error: {exc}"
                    )
        finally:
            output_json.unlink(missing_ok=True)

        self.assertGreaterEqual(
            float(summary["accuracy"]),
            float(DEFAULT_MIN_ACCURACY),
        )
        self.assertEqual(summary["num_samples"], 8)


class TestPangu1bAccuracyUnitHooks(unittest.TestCase):
    """Lightweight checks that do not load the full model."""

    def test_require_pangu_st_runtime_skips_without_model_path(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch(
                "tests.st.pangu1b_st_common.has_vllm",
                return_value=True,
            ):
                with self.assertRaises(unittest.SkipTest):
                    require_pangu_st_runtime()

    def test_extract_final_answer(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "pangu_generate_test",
            PANGU_EXAMPLE,
        )
        assert spec is not None and spec.loader is not None
        pangu_generate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pangu_generate)
        extract_final_answer = pangu_generate.extract_final_answer

        self.assertEqual(extract_final_answer("Step by step ... answer is 42."), "42")
        self.assertEqual(extract_final_answer("She ate 3/8 of the pizza."), "3/8")

    def test_npu_defaults_to_compatible_block_size(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "pangu_generate_npu_block_size_test",
            PANGU_EXAMPLE,
        )
        assert spec is not None and spec.loader is not None
        pangu_generate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pangu_generate)

        with mock.patch.object(pangu_generate, "_is_npu_platform", return_value=True):
            kwargs = pangu_generate._pangu_engine_kwargs()

        self.assertEqual(kwargs["block_size"], 128)

    def test_explicit_block_size_override_is_preserved(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "pangu_generate_block_size_override_test",
            PANGU_EXAMPLE,
        )
        assert spec is not None and spec.loader is not None
        pangu_generate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pangu_generate)

        with mock.patch.object(pangu_generate, "_is_npu_platform", return_value=True):
            kwargs = pangu_generate._pangu_engine_kwargs(64)

        self.assertEqual(kwargs["block_size"], 64)

    def test_default_dataset_exists(self) -> None:
        self.assertTrue(DEFAULT_DATASET.is_file())
        raw = DEFAULT_DATASET.read_text(encoding="utf-8").strip()
        self.assertTrue(raw, f"Dataset is empty: {DEFAULT_DATASET}")
        records = json.loads(raw)
        self.assertIsInstance(records, list)
        self.assertGreater(len(records), 0)


if __name__ == "__main__":
    unittest.main()
