# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"

"""System test: Pangu 1B accuracy on GSM8K-style mini fixture.

Requires:
- vLLM + GPU/NPU runtime
- Prepared vLLM runtime; the pangu_model fixture prepares weights on demand.

Explicit local overrides remain supported. Input preparation failures are errors.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

from tests.helpers.gate_process import run_gate_process
from tests.helpers.paths import REPO_ROOT
from tests.st import pangu_generate
from tests.st.pangu1b_st_common import (
    GSM8K_MINI,
    load_dataset,
    require_pangu_st_runtime,
    score_answer,
)

ROOT = REPO_ROOT
DEFAULT_DATASET = ROOT / GSM8K_MINI
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
@pytest.mark.timeout(float(os.environ.get("PANGU1B_OFFLINE_TIMEOUT", "1800")) + 120)
class TestPangu1bAccuracyST(unittest.TestCase):
    """End-to-end accuracy ST for the Pangu offline evaluation CLI."""

    @pytest.fixture(autouse=True)
    def _prepared_inputs(self, pangu_model: Path, gsm8k_mini: Path) -> None:
        self.model_path = pangu_model
        self.dataset_path = gsm8k_mini

    def test_gsm8k_mini_accuracy_cli(self) -> None:
        model_path = str(self.model_path)
        with tempfile.NamedTemporaryFile(
            suffix=".json",
            delete=False,
        ) as handle:
            output_json = Path(handle.name)

        measure_only = os.environ.get("PANGU1B_MEASURE_ONLY") == "1"
        num_prompts = int(os.environ.get("PANGU1B_NUM_PROMPTS", "8"))
        log = output_json.with_suffix(".log")
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("pangu_generate.py")),
            "--model",
            model_path,
            "--dataset-path",
            str(self.dataset_path),
            "--num-prompts",
            str(num_prompts),
            "--min-accuracy",
            "0" if measure_only else DEFAULT_MIN_ACCURACY,
            "--output-json",
            str(output_json),
        ]
        try:
            status = run_gate_process(
                cmd,
                cwd=ROOT,
                log=log,
                timeout=float(os.environ.get("PANGU1B_OFFLINE_TIMEOUT", "1800")),
            )
            stdout = log.read_text(encoding="utf-8")
            self.assertEqual(status, 0, f"Pangu accuracy execution failed:\n{stdout}")

            if output_json.is_file() and output_json.stat().st_size > 0:
                with output_json.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
            else:
                try:
                    summary = _parse_cli_summary(stdout)
                except json.JSONDecodeError as exc:
                    self.fail(
                        "Failed to parse evaluation JSON summary\n"
                        f"stdout:\n{stdout}\n"
                        f"parse error: {exc}"
                    )
        finally:
            output_json.unlink(missing_ok=True)
            log.unlink(missing_ok=True)

        accuracy = float(summary["accuracy"])
        self.assertTrue(math.isfinite(accuracy) and 0 <= accuracy <= 1)
        self.assertGreater(summary["num_samples"], 0)
        self.assertEqual(summary["num_samples"], len(load_dataset(self.dataset_path, num_prompts)))
        if not measure_only:
            self.assertGreaterEqual(accuracy, float(DEFAULT_MIN_ACCURACY))
        result_path = os.environ.get("PANGU1B_OFFLINE_RESULT_JSON")
        if result_path:
            Path(result_path).write_text(json.dumps(summary), encoding="utf-8")


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
        for generated, reference, expected, predicted, correct in (
            ("Step by step ... answer is 42.", "42.0", "42", "42", True),
            (r"She ate \boxed{3/8} and left 5 pieces.", "6/16", "3/8", "3/8", True),
            (r"The answer was \boxed{1,2}.", "12", "12", "1", False),
            ("The total was 1,020.", "1,021", "1021", "1020", False),
            ("I cannot tell.", "0", "0", None, False),
        ):
            with self.subTest(generated=generated):
                self.assertEqual(
                    score_answer(generated, reference),
                    {"expected": expected, "predicted": predicted, "correct": correct},
                )

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
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
