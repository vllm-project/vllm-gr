# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Check CI runtime isolation without loading an accelerator runtime."""

import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from tools.ci import resolve_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]


class TestRuntimeResolution(unittest.TestCase):
    def test_shipped_runtime_declarations_match_both_platforms(self) -> None:
        gr_version = runtime.package_version(ROOT, "0.30.0")
        self.assertEqual(
            runtime.declarations(ROOT, "cuda"),
            {"vllm": "0.30.0", "vllm-gr": gr_version},
        )
        self.assertEqual(
            runtime.declarations(ROOT, "npu"),
            {
                "vllm": "0.30.0",
                "vllm-gr": gr_version,
                "vllm-ascend": "0.30.0rc2",
                "triton-ascend": "3.6.0",
            },
        )

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        # A fixed synthetic prior-release fixture stays valid when the live candidate pins change.
        files = {
            "setup.py": 'VLLM_GR_BASE_VERSION = "0.29.0"\n',
            "pyproject.toml": '[tool.setuptools_scm]\nfallback_version = "0.29.0"\n',
            "vllm_gr/_env_check.py": (
                '_SUPPORTED_VLLM_VERSION = "0.29.0"\n'
                '_SUPPORTED_VLLM_ASCEND_VERSION = "0.29.0rc1"\n'
                '_SUPPORTED_TRITON_ASCEND_VERSION = "3.2.1"\n'
            ),
            "requirements/cuda.txt": "-r common.txt\nvllm==0.29.0\n",
            "requirements/common.txt": "numpy==2.2.6\n",
        }
        for name, content in files.items():
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        for name in ("tools/ci/resolve_runtime.py", ".pre-commit-config.yaml"):
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def test_resolve_does_not_execute_setup(self) -> None:
        with (self.repo / "setup.py").open("a") as stream:
            stream.write('\nraise RuntimeError("setup must not execute")\n')
        result = runtime.resolve(self.repo, "gpu")
        self.assertEqual(result["device"], "cuda")
        self.assertEqual(result["expected"], {"vllm": "0.29.0", "vllm-gr": "0.29.0"})
        self.assertEqual(result, runtime.resolve(self.repo, "cuda"))

    def test_resolve_accepts_scm_config_without_duplicate_fallback(self) -> None:
        """Removing TOML fallback_version must not break CI runtime resolution."""
        # setup.py remains the baseline source; 0.29.0 is isolated fixture data.
        (self.repo / "pyproject.toml").write_text("[tool.setuptools_scm]\n")
        self.assertEqual(
            runtime.resolve(self.repo, "cuda")["expected"],
            {"vllm": "0.29.0", "vllm-gr": "0.29.0"},
        )

    def test_tagged_worktree_package_version_is_separate_from_runtime(self) -> None:
        shutil.copyfile(ROOT / "setup.py", self.repo / "setup.py")
        # Keep this fixture's independent runtime baseline.
        setup = self.repo / "setup.py"
        setup.write_text(
            setup.read_text().replace(
                'VLLM_GR_BASE_VERSION = "0.30.0"', 'VLLM_GR_BASE_VERSION = "0.29.0"'
            )
        )
        (self.repo / "pyproject.toml").write_text("[tool.setuptools_scm]\n")
        for args in [
            ["init", "-q"],
            ["config", "user.name", "Runtime Test"],
            ["config", "user.email", "runtime@example.invalid"],
            ["add", "."],
            ["commit", "-qm", "Source"],
            ["tag", "v0.30.1"],
        ]:
            subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)
        worktree = self.repo.parent / (self.repo.name + "-worktree")
        subprocess.run(
            ["git", "-C", str(self.repo), "worktree", "add", "--detach", str(worktree), "v0.30.1"],
            check=True,
            capture_output=True,
        )
        self.addCleanup(shutil.rmtree, worktree)
        spec = runtime.resolve(worktree, "cuda")
        self.assertEqual(spec["expected"], {"vllm": "0.29.0", "vllm-gr": "0.30.1"})
        output = self.repo / "tagged-spec.json"
        subprocess.run(
            [
                sys.executable,
                "-S",
                str(ROOT / "tools/ci/resolve_runtime.py"),
                "resolve",
                "--repo",
                str(worktree),
                "--device",
                "cuda",
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
        self.assertEqual(json.loads(output.read_text())["expected"], spec["expected"])
        prefix = Path(sys.prefix).resolve()
        modules = {
            name: SimpleNamespace(__file__=str(prefix / name / "__init__.py"))
            for name in ("vllm", "torch", "triton")
        }
        modules["torch"].version = SimpleNamespace(cuda="13.0")
        modules["vllm"].__version__ = "0.29.0+cu130"
        modules["vllm_gr"] = SimpleNamespace(
            __file__=str(worktree / "vllm_gr/__init__.py"), __version__="0.30.1"
        )
        modules["vllm.platforms"] = SimpleNamespace(
            current_platform=SimpleNamespace(device_type="cuda")
        )
        with (
            patch.object(runtime, "inventory", return_value=spec["expected"]),
            patch.object(importlib.metadata, "distribution") as distribution,
            patch.object(importlib, "import_module", side_effect=modules.__getitem__),
        ):
            distribution.return_value.locate_file.return_value = prefix
            report = runtime.verify(worktree, spec, prefix)
            self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(
            runtime.check_versions(spec["expected"], {"vllm": "0.29.0+cu130", "vllm-gr": "0.30.1"}),
            [],
        )
        self.assertTrue(
            runtime.check_versions(spec["expected"], {"vllm": "0.30.1", "vllm-gr": "0.30.1"})
        )
        (worktree / "requirements/common.txt").write_text("# Dirty tree\n")
        node = subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "--short=12", "HEAD"], text=True
        ).strip()
        expected_dev = f"0.30.2.dev0+g{node}.dirty"
        self.assertEqual(runtime.resolve(worktree, "cuda")["expected"]["vllm-gr"], expected_dev)
        self.assertTrue(
            runtime.check_versions({"vllm-gr": expected_dev}, {"vllm-gr": "0.30.2.dev0+g000.npu"})
        )

    def test_future_candidate_drives_runtime_without_ci_pin_changes(self) -> None:
        for name in (
            "setup.py",
            "pyproject.toml",
            "vllm_gr/_env_check.py",
            "requirements/cuda.txt",
        ):
            path = self.repo / name
            path.write_text(path.read_text().replace("0.29.0", "0.30.0"))
        self.assertEqual(runtime.resolve(self.repo, "cuda")["expected"]["vllm"], "0.30.0")

    def test_partial_pin_updates_fail(self) -> None:
        for name in (
            "setup.py",
            "pyproject.toml",
            "vllm_gr/_env_check.py",
            "requirements/cuda.txt",
        ):
            with self.subTest(name=name):
                path = self.repo / name
                original = path.read_text()
                path.write_text(original.replace("0.29.0", "0.30.0"))
                with self.assertRaisesRegex(ValueError, "must agree"):
                    runtime.resolve(self.repo, "cuda")
                path.write_text(original)

    def test_annotated_guards_preserve_npu_declarations(self) -> None:
        expected = runtime.declarations(self.repo, "npu")
        path = self.repo / "vllm_gr/_env_check.py"
        original = path.read_text()
        for name in runtime.GUARDS.values():
            with self.subTest(name=name):
                target = f"{name} = "
                self.assertEqual(original.count(target), 1)
                path.write_text(original.replace(target, f"{name}: str | None = "))
                self.assertEqual(runtime.declarations(self.repo, "npu"), expected)
        path.write_text(original)

    def test_source_tree_fallback_must_match_without_importing_version_module(self) -> None:
        path = self.repo / "vllm_gr/version.py"
        original = (
            "try:\n    from ._version import __version__\n"
            'except ImportError:\n    __version__ = "0.29.0"\n'
            'raise RuntimeError("version module must not execute")\n'
        )
        path.write_text(original)
        self.assertEqual(runtime.declarations(self.repo, "cuda")["vllm-gr"], "0.29.0")
        path.write_text(original.replace("0.29.0", "0.30.0"))
        with self.assertRaisesRegex(ValueError, "source-tree fallback version.*does not match"):
            runtime.declarations(self.repo, "cuda")

    def test_source_tree_fallback_invalidates_runtime_cache(self) -> None:
        path = self.repo / "vllm_gr/version.py"
        path.write_text('__version__ = "0.29.0"\n')
        original = runtime.resolve(self.repo, "cuda")["cache_key"]
        path.write_text(path.read_text() + "# Changed version module.\n")
        self.assertNotEqual(original, runtime.resolve(self.repo, "cuda")["cache_key"])

    def test_unpinned_runtime_fails(self) -> None:
        path = self.repo / "requirements/cuda.txt"
        path.write_text(path.read_text().replace("vllm==", "vllm>="))
        with self.assertRaisesRegex(ValueError, "must agree"):
            runtime.resolve(self.repo, "cuda")

    def test_local_build_suffix_is_allowed_but_rc_is_not_stable(self) -> None:
        self.assertEqual(runtime.check_versions({"vllm": "0.30.0"}, {"vllm": "0.30.0+cu130"}), [])
        self.assertTrue(runtime.check_versions({"vllm": "0.30.0"}, {"vllm": "0.30.0rc2"}))
        self.assertTrue(
            runtime.check_versions({"vllm-ascend": "0.30.0rc2"}, {"vllm-ascend": "0.30.0"})
        )

    def test_cache_isolated_by_python_architecture_and_dependencies(self) -> None:
        original = runtime.resolve(self.repo, "cuda")["cache_key"]
        for function, value in (("python_version", "3.13.99"), ("machine", "another-architecture")):
            with patch.object(platform, function, return_value=value):
                self.assertNotEqual(original, runtime.resolve(self.repo, "cuda")["cache_key"])
        with (self.repo / "requirements/common.txt").open("a") as stream:
            stream.write("\nadditional-dependency==1.0\n")
        self.assertNotEqual(original, runtime.resolve(self.repo, "cuda")["cache_key"])

    def test_resolve_supports_base_commit_that_predates_runtime_tool(self) -> None:
        (self.repo / "tools/ci/resolve_runtime.py").unlink()

        spec = runtime.resolve(self.repo, "cuda")

        self.assertEqual(spec["expected"], {"vllm": "0.29.0", "vllm-gr": "0.29.0"})
        self.assertEqual(spec, runtime.resolve(self.repo, "cuda"))

    def test_npu_requires_and_fingerprints_vendor_image(self) -> None:
        packages = {
            "vllm": "0.29.0+npu",
            "vllm-ascend": "0.29.0rc1",
            "triton-ascend": "3.2.1",
            "torch-npu": "2.11.0",
        }
        with patch.object(runtime, "inventory", return_value=packages):
            npu = runtime.resolve(self.repo, "npu")
            self.assertNotEqual(npu["cache_key"], runtime.resolve(self.repo, "cuda")["cache_key"])
            packages["torch-npu"] = "2.11.0+vendor2"
            self.assertNotEqual(npu["cache_key"], runtime.resolve(self.repo, "npu")["cache_key"])
            packages.pop("triton-ascend")
            with self.assertRaisesRegex(ValueError, "NPU runner image is not ready"):
                runtime.resolve(self.repo, "npu")

    def test_verify_rejects_wrong_runtime_before_import(self) -> None:
        spec = runtime.resolve(self.repo, "cuda")
        with (
            patch.object(
                runtime, "inventory", return_value={"vllm-gr": "0.29.0", "vllm": "0.30.0"}
            ),
            patch.object(importlib.metadata, "distribution") as distribution,
            patch.object(importlib, "import_module") as importer,
        ):
            distribution.return_value.locate_file.return_value = Path(sys.prefix)
            report = runtime.verify(self.repo, spec, Path(sys.prefix).resolve())
        self.assertFalse(report["ok"])
        self.assertIn("expected 0.29.0, found 0.30.0", ";".join(report["errors"]))
        importer.assert_not_called()

    def test_verify_checks_real_import_origins_and_platform(self) -> None:
        spec = runtime.resolve(self.repo, "cuda")
        prefix = Path(sys.prefix).resolve()
        modules = {
            name: SimpleNamespace(__file__=str(prefix / name / "__init__.py"))
            for name in ("vllm", "torch", "triton")
        }
        modules["torch"].version = SimpleNamespace(cuda="13.0")
        modules["vllm"].__version__ = "0.29.0+cu130"
        modules["vllm_gr"] = SimpleNamespace(
            __file__=str(self.repo / "vllm_gr/__init__.py"), __version__="0.29.0"
        )
        modules["vllm.platforms"] = SimpleNamespace(
            current_platform=SimpleNamespace(device_type="cuda")
        )
        with (
            patch.object(runtime, "inventory", return_value=spec["expected"]),
            patch.object(importlib.metadata, "distribution") as distribution,
            patch.object(importlib, "import_module", side_effect=modules.__getitem__),
        ):
            distribution.return_value.locate_file.return_value = prefix
            self.assertTrue(runtime.verify(self.repo, spec, prefix)["ok"])
            modules["vllm"].__version__ = "0.30.0"
            self.assertFalse(runtime.verify(self.repo, spec, prefix)["ok"])
            modules["vllm"].__version__ = "0.29.0+cu130"
            modules["vllm_gr"].__file__ = "/another-worktree/vllm_gr/__init__.py"
            report = runtime.verify(self.repo, spec, prefix)
            self.assertFalse(report["ok"])
            self.assertIn("must import from", ";".join(report["errors"]))
            modules["vllm_gr"].__file__ = str(self.repo / "vllm_gr/__init__.py")
            modules["vllm"].__file__ = "/old-environment/vllm/__init__.py"
            self.assertFalse(runtime.verify(self.repo, spec, prefix)["ok"])
            modules["vllm"].__file__ = str(prefix / "vllm/__init__.py")
            modules["vllm.platforms"].current_platform.device_type = "cpu"
            self.assertFalse(runtime.verify(self.repo, spec, prefix)["ok"])

    def test_cli_writes_failure_report(self) -> None:
        path = self.repo / "requirements/cuda.txt"
        path.write_text("vllm==0.30.0\n")
        output = self.repo / "report.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/ci/resolve_runtime.py"),
                "resolve",
                "--repo",
                str(self.repo),
                "--device",
                "cuda",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(output.read_text())["ok"])

    def test_missing_vendor_declaration_does_not_prevent_cuda_resolution(self) -> None:
        path = self.repo / "vllm_gr/_env_check.py"
        path.write_text(
            path.read_text().replace(
                '_SUPPORTED_TRITON_ASCEND_VERSION = "3.2.1"',
                "",
            )
        )
        self.assertEqual(runtime.resolve(self.repo, "cuda")["expected"]["vllm"], "0.29.0")
        with self.assertRaisesRegex(ValueError, "_SUPPORTED_TRITON_ASCEND_VERSION"):
            runtime.resolve(self.repo, "npu")


class TestRuntimeWorkflow(unittest.TestCase):
    def test_every_runtime_job_resolves_before_cache_and_verifies_before_execution(self) -> None:
        workflow = yaml.safe_load((ROOT / ".github/workflows/pre-commit.yml").read_text())
        for name in ("pre-commit", "decode-benchmark", "pangu-offline", "pangu-online"):
            with self.subTest(job=name):
                steps = workflow["jobs"][name]["steps"]
                resolve = next(i for i, step in enumerate(steps) if step.get("id") == "runtime")
                cache = next(i for i, step in enumerate(steps) if step.get("id") == "cache-venv")
                verify = next(
                    i
                    for i, step in enumerate(steps)
                    if step.get("name") == "Verify installed runtime identity"
                )
                cleanup = next(
                    i for i, step in enumerate(steps) if step.get("id") == "cuda-mapping"
                )
                self.assertLess(resolve, cache)
                self.assertLess(cache, verify)
                self.assertLess(verify, cleanup)
                self.assertEqual(
                    steps[cache]["with"]["key"], "${{ steps.runtime.outputs.cache-key }}"
                )
                self.assertIn("BENCHMARK_DEVICE", steps[resolve]["env"]["RUNTIME_DEVICE"])
        base = next(
            step
            for step in workflow["jobs"]["pre-commit"]["steps"]
            if step.get("name") == "Install CUDA base environment"
        )
        self.assertIn("== 'cuda'", base["if"])
        self.assertIn('"$CI_VLLM_REQUIREMENT"', base["run"])


if __name__ == "__main__":
    unittest.main()


pytestmark = pytest.mark.cpu_test
