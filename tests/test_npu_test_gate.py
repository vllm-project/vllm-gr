# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the shell gate without selecting real accelerator devices."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("bash"), "Bash is required for shell-gate controls")
class TestNPUGateRuntime(unittest.TestCase):
    def run_gate(self, *, npu: bool, runtime_status: int = 0) -> tuple[int, str, list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace"
            files = {
                "tools/run_test.sh": (ROOT / "tools/run_test.sh").read_text(),
                "tools/resource.sh": (
                    f"npu_state={'true' if npu else 'false'}\n"
                    'get_gpu() { echo device >> "$GATE_TRACE"; echo 0; }\n'
                    'python3() { echo "runtime:$*" >> "$GATE_TRACE"; '
                    'return "$RUNTIME_STATUS"; }\n'
                    'pre-commit() { echo cuda-tests >> "$GATE_TRACE"; }\n'
                ),
                "tests/development/test_resource.sh": (
                    '#!/bin/bash\necho resources >> "$GATE_TRACE"\n'
                ),
                "tools/install_xllm_ops.sh": '#!/bin/bash\necho operators >> "$GATE_TRACE"\n',
                "tools/pre_commit/run_tests.sh": (
                    '#!/bin/bash\necho "pytest:$VLLM_GR_TARGET_DEVICE" >> "$GATE_TRACE"\n'
                ),
                "toolkit/opp/vendors/custom_xllm_math/bin/set_env.bash": "#!/bin/bash\n",
            }
            for name, contents in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8", newline="\n")
                path.chmod(0o755)
            env = os.environ.copy()
            env.pop("CONDA_PREFIX", None)
            env.pop("VLLM_GR_TARGET_DEVICE", None)
            env.update(
                GATE_TRACE=trace.as_posix(),
                RUNTIME_STATUS=str(runtime_status),
                ASCEND_TOOLKIT_HOME=(root / "toolkit").as_posix(),
                LD_LIBRARY_PATH="",
            )
            result = subprocess.run(
                ["bash", "tools/run_test.sh"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            events = trace.read_text().splitlines() if trace.exists() else []
            return result.returncode, result.stderr, events

    def test_wrong_npu_runtime_stops_before_resources_and_pytest(self) -> None:
        code, error, events = self.run_gate(npu=True, runtime_status=1)
        self.assertNotEqual(code, 0)
        self.assertIn("matching vendor environment", error)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].startswith("runtime:"))
        self.assertIn("resolve --device npu --output", events[0])

    def test_matching_npu_runtime_reaches_tests_with_explicit_target(self) -> None:
        code, error, events = self.run_gate(npu=True)
        self.assertEqual(code, 0, error)
        self.assertTrue(events[0].startswith("runtime:"))
        self.assertEqual(events[1:], ["resources", "device", "operators", "pytest:npu"])

    def test_cuda_gate_does_not_require_an_ascend_runtime(self) -> None:
        code, error, events = self.run_gate(npu=False, runtime_status=1)
        self.assertEqual(code, 0, error)
        self.assertEqual(events, ["resources", "device", "cuda-tests"])


if __name__ == "__main__":
    unittest.main()
