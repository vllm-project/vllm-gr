# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Collected, representative real-NPU shared-KV copy and replay contracts.

Device imports and availability checks happen only while executing a test, so
collecting this module never initializes an accelerator. Select one NPU before
running; the subprocess owns its graph and frees it when the process exits.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.npu_test, pytest.mark.l2]


@pytest.mark.parametrize(  # type: ignore[misc]
    "script",
    ["check_npu_shared_kv_gather_contract.py", "check_npu_shared_kv_length_mask_contract.py"],
    ids=["unmasked", "device-length-mask"],
)
def test_shared_kv_copy_and_graph_replay_contract(script: str, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    if torch.npu.device_count() != 1:
        pytest.fail("Select exactly one NPU before running the shared-KV contract")

    source = Path(__file__).resolve().parents[2]
    report_path = tmp_path / "contract.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(source), environment.get("PYTHONPATH", "")))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name(script)),
            "--quick",
            "--dtypes",
            "bfloat16",
            "--widths",
            "32",
            "--attention",
            "--output",
            str(report_path),
        ],
        cwd=source,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["state"] == "passed"
    assert len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["state"] == "passed" and case["layout"] == "interleaved"
    assert case["batch"] == 1 and case["beam_width"] == 32
    assert case["output_addresses_stable"] is True
    if script == "check_npu_shared_kv_gather_contract.py":
        assert case["graph_replays_with_new_tables_and_kv"] == 3
        assert case["bitwise_copy_equal"] is True
        assert case["native_attention"] == {"executed": True, "bitwise_equal": True}
    else:
        assert len(case["replays"]) == 5
        assert all(
            replay["active_copy_bitwise_equal"]
            and replay["unused_blocks_remain_nan"]
            and replay["attention_bitwise_equal"]
            for replay in case["replays"]
        )
