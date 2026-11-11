# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Exercise shell routing with stage executables; no accelerator is accessed."""

import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]

BASH = shutil.which("bash")
RUN_TEST = Path(__file__).resolve().parents[1] / "tools/run_test.sh"


@unittest.skipUnless(BASH, "bash is required for the shell routing contract")
class TestRunTestNPUDevice(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / "routing.log"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.toolkit = self.root / "toolkit"
        vendor = self.toolkit / "opp/vendors/custom_xllm_math"
        (vendor / "op_api/lib").mkdir(parents=True)
        self.write("tools/run_test.sh", RUN_TEST.read_text(encoding="utf-8"), executable=True)
        # Source the real parsing helpers with device discovery hidden, then
        # replace only resource selection with a routing receipt.
        self.write(
            "tools/resource_helpers.sh",
            (RUN_TEST.parent / "resource.sh").read_text(encoding="utf-8"),
        )
        self.write(
            "tools/resource.sh",
            """
            printf 'resource-source\\n' >> "$ROUTING_LOG"
            PATH="$ROUTING_BIN" source ./tools/resource_helpers.sh
            npu_state=$TEST_NPU_STATE
            get_gpu() {
                printf 'device-selection\\n' >> "$ROUTING_LOG"
                printf '2\\n'
            }
            """,
        )
        self.write(
            "tests/development/test_resource.sh",
            """
            #!/bin/bash
            printf 'resource-test\\n' >> "$ROUTING_LOG"
            """,
            executable=True,
        )
        self.write(
            "tools/install_xllm_ops.sh",
            """
            #!/bin/bash
            printf 'install:%s\\n' "$ASCEND_RT_VISIBLE_DEVICES" >> "$ROUTING_LOG"
            if [[ ${TEST_FAIL_STAGE:-} == install ]]; then exit 35; fi
            """,
        )
        self.write(
            "toolkit/opp/vendors/custom_xllm_math/bin/set_env.bash",
            "printf 'operator-env\\n' >> \"$ROUTING_LOG\"\n",
        )
        self.write(
            "tools/pre_commit/run_tests.sh",
            """
            #!/bin/bash
            printf 'legacy-full:%s:%s:%s:%s\\n' \\
                "$ASCEND_RT_VISIBLE_DEVICES" "$PYTORCH_NPU_ALLOW_FORK" \\
                "$VLLM_WORKER_MULTIPROC_METHOD" "$TORCH_MULTIPROCESSING_METHOD" \\
                >> "$ROUTING_LOG"
            """,
        )
        self.write(
            "tools/run_wheel_smoke_test.sh",
            """
            #!/bin/bash
            printf 'wheel:%s:%s:%s\\n' "$VLLM_GR_TARGET_DEVICE" \\
                "${ASCEND_RT_VISIBLE_DEVICES:-}" "${CUDA_VISIBLE_DEVICES:-}" \\
                >> "$ROUTING_LOG"
            """,
            executable=True,
        )
        self.write(
            "bin/python3",
            """
            #!/bin/bash
            if [[ $1 == tools/ci/resolve_runtime.py ]]; then
                printf 'runtime\\n' >> "$ROUTING_LOG"
                if [[ ${TEST_FAIL_STAGE:-} == runtime ]]; then exit 34; fi
            elif [[ $1 == tools/ci/run_npu_tests.py ]]; then
                printf 'runner' >> "$ROUTING_LOG"
                printf '<%s>' "$@" >> "$ROUTING_LOG"
                printf '\\n' >> "$ROUTING_LOG"
                printf 'runner-env:%s:%s:%s:%s\\n' \\
                    "$ASCEND_RT_VISIBLE_DEVICES" "$PYTORCH_NPU_ALLOW_FORK" \\
                    "$VLLM_WORKER_MULTIPROC_METHOD" "$TORCH_MULTIPROCESSING_METHOD" \\
                    >> "$ROUTING_LOG"
                if [[ ${TEST_FAIL_STAGE:-} == runner ]]; then exit 36; fi
            else
                exit 99
            fi
            """,
            executable=True,
        )
        self.write(
            "bin/pre-commit",
            """
            #!/bin/bash
            printf 'cuda-precommit:%s:%s\\n' "$CUDA_VISIBLE_DEVICES" "$*" \\
                >> "$ROUTING_LOG"
            """,
            executable=True,
        )

    def write(self, relative: str, content: str, *, executable: bool = False) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")
        if executable:
            path.chmod(0o755)

    def run_script(
        self, *arguments: str, npu: bool = True, inherited: str | None = None, fail: str = ""
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"ASCEND_RT_VISIBLE_DEVICES", "CONDA_PREFIX", "CUDA_VISIBLE_DEVICES"}
        }
        environment.update(
            PATH=f"{self.bin}{os.pathsep}{environment['PATH']}",
            ROUTING_LOG=self.log.as_posix(),
            ROUTING_BIN=self.bin.as_posix(),
            ASCEND_TOOLKIT_HOME=self.toolkit.as_posix(),
            LD_LIBRARY_PATH="",
            TEST_NPU_STATE="true" if npu else "false",
            TEST_FAIL_STAGE=fail,
        )
        if inherited is not None:
            environment["ASCEND_RT_VISIBLE_DEVICES"] = inherited
        assert BASH is not None
        return subprocess.run(
            [BASH, "tools/run_test.sh", *arguments],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )

    def events(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def test_no_options_keeps_legacy_npu_full_route(self) -> None:
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.events(),
            [
                "resource-source",
                "runtime",
                "resource-test",
                "device-selection",
                "install:2",
                "operator-env",
                "legacy-full:2:true:spawn:spawn",
            ],
        )

    def test_preallocated_device_does_not_select_or_test_other_devices(self) -> None:
        result = self.run_script("--npu-device", "8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.events(),
            [
                "resource-source",
                "runtime",
                "install:8",
                "operator-env",
                "runner<tools/ci/run_npu_tests.py><--device><8><--tier><full>"
                "<--output-dir><results/npu-tests>",
                "runner-env:8:true:spawn:spawn",
            ],
        )
        self.assertIn("reservation remains caller-owned", result.stdout)

    def test_main_device_option_selects_opt_in_npu_runner(self) -> None:
        for arguments in (("--device-id", "8"), ("--device-id=8",)):
            with self.subTest(arguments=arguments):
                result = self.run_script(*arguments, "--npu-tier", "full", inherited="8")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("runner-env:8:true:spawn:spawn", self.events())
                self.assertNotIn("device-selection", self.events())

    def test_matching_device_aliases_are_accepted(self) -> None:
        result = self.run_script("--device-id=8", "--npu-device", "8", inherited="8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("runner-env:8:true:spawn:spawn", self.events())
        self.assertNotIn("device-selection", self.events())

    def test_conflicting_device_aliases_fail_before_resource_queries(self) -> None:
        for arguments in (
            ("--device-id", "7", "--npu-device", "8"),
            ("--npu-device", "8", "--device-id=7"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_script(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("conflicts with --device-id", result.stderr)
                self.assertEqual(self.events(), [])

    def test_opt_in_device_id_retains_visibility_and_single_device_contract(self) -> None:
        for arguments, inherited in (
            (("--device-id", "8"), "7"),
            (("--device-id=8",), ""),
            (("--device-id", "8,9"), None),
            (("--device-id=08",), None),
        ):
            with self.subTest(arguments=arguments, inherited=inherited):
                result = self.run_script(*arguments, "--npu-tier", "full", inherited=inherited)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.events(), [])

    def test_legacy_device_id_keeps_multi_device_route(self) -> None:
        for npu in (True, False):
            with self.subTest(npu=npu):
                self.log.unlink(missing_ok=True)
                result = self.run_script("--device-id=8,9", npu=npu)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = (
                    "legacy-full:8,9:true:spawn:spawn"
                    if npu
                    else "cuda-precommit:8,9:run --all-files"
                )
                self.assertIn(expected, self.events())
                self.assertNotIn("device-selection", self.events())
                self.assertFalse(any(event.startswith("runner") for event in self.events()))

    def test_candidate_and_output_directory_are_passed_as_exact_arguments(self) -> None:
        result = self.run_script(
            "--npu-device",
            "8",
            "--npu-tier",
            "p0-candidate",
            "--npu-results-dir",
            "results/build 42",
            inherited="8",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "runner<tools/ci/run_npu_tests.py><--device><8><--tier><p0-candidate>"
            "<--output-dir><results/build 42>",
            self.events(),
        )
        self.assertNotIn("device-selection", self.events())

    def test_zero_is_a_valid_preallocated_device(self) -> None:
        result = self.run_script("--npu-device", "0", inherited="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("install:0", self.events())
        self.assertIn("runner-env:0:true:spawn:spawn", self.events())
        self.assertNotIn("device-selection", self.events())

    def test_formal_p0_is_passed_without_reallocating_the_device(self) -> None:
        result = self.run_script("--npu-device", "8", "--npu-tier", "p0", inherited="8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "runner<tools/ci/run_npu_tests.py><--device><8><--tier><p0>"
            "<--output-dir><results/npu-tests>",
            self.events(),
        )
        self.assertNotIn("device-selection", self.events())

    def test_explicit_tier_without_device_keeps_legacy_selection(self) -> None:
        result = self.run_script("--npu-tier", "full")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("resource-test", self.events())
        self.assertIn("device-selection", self.events())
        self.assertIn(
            "runner<tools/ci/run_npu_tests.py><--device><2><--tier><full>"
            "<--output-dir><results/npu-tests>",
            self.events(),
        )

    def test_invalid_devices_fail_before_sourcing_resources(self) -> None:
        for device in ("", "-1", "+1", "00", "01", "1,2", " 1", "1 ", "1\n2", "x", "1.0"):
            with self.subTest(device=device):
                result = self.run_script("--npu-device", device)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("canonical nonnegative integer", result.stderr)
                self.assertEqual(self.events(), [])

    def test_inherited_device_mismatch_fails_before_sourcing_resources(self) -> None:
        for inherited in ("", "0", "08", "8,9", " 8"):
            with self.subTest(inherited=inherited):
                result = self.run_script("--npu-device", "8", inherited=inherited)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("conflicts with inherited", result.stderr)
                self.assertEqual(self.events(), [])

    def test_invalid_or_missing_options_fail_before_sourcing_resources(self) -> None:
        for arguments in (
            ("--npu-device",),
            ("--npu-tier",),
            ("--npu-tier", "not-a-tier"),
            ("--npu-results-dir",),
            ("--npu-results-dir", ""),
            ("--unexpected",),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_script(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.events(), [])

    def test_gpu_rejects_all_npu_only_options_before_selection(self) -> None:
        for arguments in (
            ("--npu-device", "8"),
            ("--npu-tier", "full"),
            ("--npu-results-dir", "results/build"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_script(*arguments, npu=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("require an NPU host", result.stderr)
                self.assertTrue(set(self.events()) <= {"resource-source"})

    def test_gpu_legacy_route_and_wheel_remain_unchanged(self) -> None:
        result = self.run_script("--wheel", npu=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.events(),
            [
                "resource-source",
                "resource-test",
                "device-selection",
                "cuda-precommit:2:run --all-files",
                "wheel:cuda::2",
            ],
        )

    def test_preallocated_npu_wheel_uses_the_same_device(self) -> None:
        result = self.run_script("--npu-device", "8", "--wheel", inherited="8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.events()[-1], "wheel:npu:8:")
        self.assertNotIn("device-selection", self.events())

    def test_runtime_failure_stops_before_install_or_runner(self) -> None:
        result = self.run_script("--npu-device", "8", fail="runtime")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.events(), ["resource-source", "runtime"])

    def test_install_failure_does_not_run_tests(self) -> None:
        result = self.run_script("--npu-device", "8", fail="install")
        self.assertEqual(result.returncode, 35)
        self.assertEqual(self.events(), ["resource-source", "runtime", "install:8"])

    def test_runner_failure_does_not_run_wheel(self) -> None:
        result = self.run_script("--npu-device", "8", "--wheel", fail="runner")
        self.assertEqual(result.returncode, 36)
        self.assertFalse(any(event.startswith("wheel:") for event in self.events()))
