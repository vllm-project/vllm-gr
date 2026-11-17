# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"

"""System test: Pangu 1B accuracy on GSM8K-style mini fixture.

Requires:
- vLLM + GPU/NPU runtime
- ``PANGU1B_MODEL_PATH`` / ``PANGU_MODEL_PATH`` or ``--model`` with weights

Skip when the model path is unavailable (CI without weights).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

from tests.helpers.paths import REPO_ROOT
from tests.st import pangu_generate
from tests.st.pangu1b_st_common import get_model_path, require_pangu_st_runtime

ROOT = REPO_ROOT
DEFAULT_DATASET = ROOT / "tests" / "resources" / "pangu1b" / "gsm8k_mini.json"
DEFAULT_MIN_ACCURACY = os.environ.get(
    "PANGU1B_OFFLINE_ACCURACY_THRESHOLD",
    os.environ.get("PANGU1B_MIN_ACCURACY", "0.5"),
)


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


@pytest.mark.gpu_test
@pytest.mark.npu_test
@pytest.mark.model_test
class TestPangu1bAccuracyST(unittest.TestCase):
    """End-to-end accuracy ST for the Pangu offline evaluation CLI."""

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
            str(Path(__file__).with_name("pangu_generate.py")),
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
                        "Failed to parse evaluation JSON summary\n"
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


@pytest.mark.cpu_test
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
        self.assertEqual(
            pangu_generate.extract_final_answer("Step by step ... answer is 42."), "42"
        )
        self.assertEqual(pangu_generate.extract_final_answer("She ate 3/8 of the pizza."), "3/8")

    def test_npu_defaults_to_compatible_block_size(self) -> None:
        with mock.patch.object(pangu_generate, "_is_npu_platform", return_value=True):
            kwargs = pangu_generate._pangu_engine_kwargs()

        self.assertEqual(kwargs["block_size"], 128)

    def test_explicit_block_size_override_is_preserved(self) -> None:
        with mock.patch.object(pangu_generate, "_is_npu_platform", return_value=True):
            kwargs = pangu_generate._pangu_engine_kwargs(64)

        self.assertEqual(kwargs["block_size"], 64)

    @pytest.mark.l1
    def test_offline_configuration_keeps_native_generation(self) -> None:
        from vllm_gr.config import load_gr_config

        config = load_gr_config(pangu_generate._pangu_engine_kwargs()["config"])
        self.assertIsNone(config.beam_max_width)
        self.assertFalse(config.beam_graph_enabled)
        self.assertFalse(config.deployment.prefill_graph_enabled)

    def test_default_dataset_exists(self) -> None:
        self.assertEqual(pangu_generate._DEFAULT_DATASET, DEFAULT_DATASET)
        self.assertTrue(DEFAULT_DATASET.is_file())
        raw = DEFAULT_DATASET.read_text(encoding="utf-8").strip()
        self.assertTrue(raw, f"Dataset is empty: {DEFAULT_DATASET}")
        records = json.loads(raw)
        self.assertIsInstance(records, list)
        self.assertGreater(len(records), 0)


if __name__ == "__main__":
    unittest.main()
