# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise the shell gate without selecting real accelerator devices."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

from tests.helpers.paths import REPO_ROOT

ROOT = REPO_ROOT


@unittest.skipUnless(shutil.which("bash"), "Bash is required for shell-gate controls")
class TestNPUGateRuntime(unittest.TestCase):
    def run_gate(
        self,
        *,
        npu: bool,
        runtime_status: int = 0,
        extra_args: list[str] | None = None,
        vendor_layouts: tuple[str, ...] = ("bin",),
    ) -> tuple[int, str, list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace"
            files = {
                "tools/run_test.sh": (ROOT / "tools/run_test.sh").read_text(),
                "tools/devices/resource.sh": (
                    f"npu_state={'true' if npu else 'false'}\n"
                    'get_gpu() { echo device >> "$GATE_TRACE"; echo 0; }\n'
                    # run_test.sh runs the visibility parser check on every path,
                    # so the stand-in must honor the same parsing contract.
                    "_cuda_device_indices() {\n"
                    # One event per gate run: the first call is the parser check.
                    '    [ -e "$GATE_TRACE.indices" ] || {\n'
                    '        echo indices >> "$GATE_TRACE"\n'
                    '        : > "$GATE_TRACE.indices"\n'
                    "    }\n"
                    "    local n=$1\n"
                    '    if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then\n'
                    "        seq 0 $((n - 1))\n"
                    "        return 0\n"
                    "    fi\n"
                    '    [ -n "$CUDA_VISIBLE_DEVICES" ] || return 1\n'
                    "    case $CUDA_VISIBLE_DEVICES in *[!0-9,]*) return 1 ;; esac\n"
                    "    local IFS=',' i\n"
                    "    for i in $CUDA_VISIBLE_DEVICES; do\n"
                    '        [ "$i" -lt "$n" ] && echo "$i"\n'
                    "    done\n"
                    "    return 0\n"
                    "}\n"
                    'python3() { echo "runtime:$*" >> "$GATE_TRACE"; '
                    'return "$RUNTIME_STATUS"; }\n'
                    'pre-commit() { echo cuda-tests >> "$GATE_TRACE"; }\n'
                ),
                "tests/development/test_resource.sh": (
                    '#!/bin/bash\necho resources >> "$GATE_TRACE"\n'
                ),
                "tools/npu/install_xllm_ops.sh": '#!/bin/bash\necho operators >> "$GATE_TRACE"\n',
                "tools/npu/xllm_env.sh": (ROOT / "tools/npu/xllm_env.sh").read_text(),
                "tools/pre_commit/run_tests.sh": (
                    '#!/bin/bash\necho "pytest:$VLLM_GR_TARGET_DEVICE" >> "$GATE_TRACE"\n'
                ),
                "tools/packaging/run_wheel_smoke_test.sh": (
                    "#!/bin/bash\n"
                    'if [[ "$VLLM_GR_TARGET_DEVICE" == npu ]]; then\n'
                    '    device="$ASCEND_RT_VISIBLE_DEVICES"\n'
                    "else\n"
                    '    device="$CUDA_VISIBLE_DEVICES"\n'
                    "fi\n"
                    'echo "wheel:$VLLM_GR_TARGET_DEVICE:$device:$*" >> "$GATE_TRACE"\n'
                ),
            }
            vendor_root = "toolkit/opp/vendors/custom_xllm_math"
            for layout in vendor_layouts:
                files[f"{vendor_root}/{layout}/set_env.bash"] = (
                    '[[ "$LD_LIBRARY_PATH" == "$ASCEND_TOOLKIT_HOME/opp/vendors/'
                    'custom_xllm_math/op_api/lib" ]] || exit 1\n'
                    f'echo "vendor:{layout}" >> "$GATE_TRACE"\n'
                )
            for name, contents in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8", newline="\n")
                path.chmod(0o755)
            env = os.environ.copy()
            env.pop("CONDA_PREFIX", None)
            env.pop("VLLM_GR_TARGET_DEVICE", None)
            env.pop("LD_LIBRARY_PATH", None)
            env.pop("ASCEND_HOME_PATH", None)
            env.update(
                GATE_TRACE=trace.as_posix(),
                RUNTIME_STATUS=str(runtime_status),
                ASCEND_TOOLKIT_HOME=(root / "toolkit").as_posix(),
            )
            result = subprocess.run(
                ["bash", "tools/run_test.sh", *(extra_args or [])],
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
        self.assertEqual(
            events[1:], ["indices", "resources", "device", "operators", "vendor:bin", "pytest:npu"]
        )

    @pytest.mark.l1  # type: ignore[misc]
    def test_vendor_env_layout_selection(self) -> None:
        for layouts, expected in (
            (("new layout",), "vendor:new layout"),
            (("bin", "legacy"), "vendor:bin"),
            (("one", "two"), None),
            ((), None),
        ):
            with self.subTest(layouts=layouts):
                code, error, events = self.run_gate(
                    npu=True, vendor_layouts=layouts, extra_args=["--device-id", "3"]
                )
                if expected:
                    self.assertEqual(code, 0, error)
                    self.assertIn(expected, events)
                else:
                    self.assertNotEqual(code, 0)
                    self.assertNotIn("pytest:npu", events)

    def test_cuda_gate_does_not_require_an_ascend_runtime(self) -> None:
        code, error, events = self.run_gate(npu=False, runtime_status=1)
        self.assertEqual(code, 0, error)
        self.assertEqual(events, ["indices", "resources", "device", "cuda-tests"])

    def test_explicit_device_still_runs_the_parser_check(self) -> None:
        code, error, events = self.run_gate(npu=False, extra_args=["--device-id", "3"])
        self.assertEqual(code, 0, error)
        # The parser check runs, auto-detection does not.
        self.assertEqual(events, ["indices", "cuda-tests"])

    @pytest.mark.l1  # type: ignore[misc]
    def test_wheel_smoke_runs_after_tests_on_selected_device(self) -> None:
        for npu, target, gate in ((True, "npu", "pytest:npu"), (False, "cuda", "cuda-tests")):
            with self.subTest(target=target):
                code, error, events = self.run_gate(
                    npu=npu, extra_args=["--device-id", "3", "--wheel"]
                )
                self.assertEqual(code, 0, error)
                self.assertEqual(events[-2:], [gate, f"wheel:{target}:3:--python python3"])

    def test_empty_device_id_is_rejected(self) -> None:
        for args in (["--device-id="], ["--device-id", ""]):
            code, error, events = self.run_gate(npu=False, extra_args=args)
            self.assertNotEqual(code, 0)
            self.assertIn("requires a non-empty value", error)


if __name__ == "__main__":
    unittest.main()


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash required")  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("initial", "after_build", "built", "ok"),
    [
        ("ready", "fail", False, True),
        ("missing", "ready", True, True),
        ("missing-op", "ready", True, True),
        ("missing-topk", "ready", True, True),
        ("missing", "missing-topk", True, False),
        ("broken", "broken", True, False),
        ("missing", "fail", True, False),
        ("missing", "default", True, True),
    ],
)
def test_install_checks_callable_ops_before_and_after_build(
    tmp_path: Path, initial: str, after_build: str, built: bool, ok: bool
) -> None:
    # Simulate the vendor interpreter, its binding and a custom build offline.
    interpreter = tmp_path / "python3"
    interpreter.write_text('#!/bin/bash\nexec "$TEST_PYTHON" -S "$@"\n', newline="\n")
    interpreter.chmod(0o755)
    (tmp_path / "torch.py").write_text("ops = None\n")
    for name in ("vllm_ascend", "vllm_ascend/ops"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "__init__.py").touch()
    (tmp_path / "vllm_ascend/ops/xllm_ops.py").touch()
    (tmp_path / "libxllm_ops.so").touch()
    attention_kv_ops = (
        "def x_attention(): pass\ndef select_unshared_kv(): pass\ndef cache_unshared_kv(): pass\n"
    )
    ready = attention_kv_ops + "def rec_constrained_topk(): pass\n"
    (tmp_path / "ready.py").write_text(ready)
    (tmp_path / "missing-topk.py").write_text(attention_kv_ops)
    binding = tmp_path / "custom_ops_lib.py"
    if initial == "ready":
        binding.write_text(ready)
    elif initial == "missing-topk":
        binding.write_text(attention_kv_ops)
    elif initial == "missing-op":
        binding.write_text("x_attention = None\n")
    elif initial == "broken":
        binding.write_text("raise ImportError('unresolved library')\n")
    build = tmp_path / "build.sh"
    build.write_text(
        'touch built\ncase "$BUILD_RESULT" in\n'
        "  ready) cp ready.py custom_ops_lib.py ;;\n"
        "  missing-topk) cp missing-topk.py custom_ops_lib.py ;;\n"
        "  fail) exit 7 ;;\nesac\n",
        newline="\n",
    )
    env = os.environ.copy()
    env.pop("ASCEND_TOOLKIT_HOME", None)
    env.pop("ASCEND_HOME_PATH", None)
    env.pop("ASCEND_TOOLKIT_ENV", None)
    env.update(
        PATH=str(tmp_path) + os.pathsep + env["PATH"],
        TEST_PYTHON=Path(sys.executable).as_posix(),
        PYTHONPATH=str(tmp_path),
        LD_LIBRARY_PATH=tmp_path.as_posix(),
        XLLM_OPS_SKIP_NPU_CHECK="1",
        XLLM_OPS_BUILD_SCRIPT=build.as_posix(),
        BUILD_RESULT=after_build,
    )
    if after_build == "default":
        checkout = tmp_path / "upstream"
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        vendor = tmp_path / "toolkit/opp/vendors/custom_xllm_math/bin"
        vendor.mkdir(parents=True)
        (vendor / "set_env.bash").write_text(
            'export VENDOR_READY=1\ncd "$TEST_ROOT"\n', newline="\n"
        )
        toolkit_env = tmp_path / "toolkit/set_env.sh"
        toolkit_env.write_text('export ASCEND_HOME_PATH="$TEST_ROOT/toolkit"\n', newline="\n")
        # Exercise the CANN build instructions without a vendor checkout or network.
        (checkout / "build.sh").write_text(
            '[[ "$1" == -n && "$2" == "beam_search_group;cache_unshared_kv;'
            'select_unshared_kv;x_attention;onerec_final_beam_select;rec_constrained_top_k" '
            '&& "$3" == --no-build-transformer && "$4" == --no-build-flash ]] || exit 1\n'
            'touch "$TEST_ROOT/built"\n',
            newline="\n",
        )
        binding_dir = checkout / "test/python_test"
        binding_dir.mkdir(parents=True)
        (binding_dir / "build_and_run.sh").write_text(
            '[[ "$VENDOR_READY" == 1 ]] || exit 1\n'
            'cp "$TEST_ROOT/ready.py" "$TEST_ROOT/custom_ops_lib.py"\n',
            newline="\n",
        )
        env.pop("XLLM_OPS_BUILD_SCRIPT")
        env.update(
            XLLM_OPS_DIR=checkout.as_posix(),
            ASCEND_TOOLKIT_ENV=toolkit_env.as_posix(),
            TEST_ROOT=tmp_path.as_posix(),
        )
    result = subprocess.run(
        ["bash", (ROOT / "tools/npu/install_xllm_ops.sh").as_posix()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert (result.returncode == 0) == ok, result.stderr
    assert (tmp_path / "built").exists() == built
    if after_build == "missing-topk":
        assert "missing callable operators: rec_constrained_topk" in result.stderr
