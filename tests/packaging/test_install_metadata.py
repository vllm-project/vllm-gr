# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Smoke tests for platform-aware install metadata routing."""

from __future__ import annotations

import importlib.util
import os
import runpy
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Protocol, cast

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from tests.helpers.paths import REPO_ROOT
from tools.ci.resolve_runtime import check_versions, package_version

COMMON_FILE = REPO_ROOT / "requirements" / "common.txt"
#: Release tag created by the ``package_source`` fixture and reused in expectations.
FIXTURE_RELEASE = "0.30.1"


def _requirement_line(device: str, name: str) -> str:
    """Return the requirement line a device file declares for one distribution."""
    path: Path = REPO_ROOT / "requirements" / f"{device}.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value and not value.startswith("-") and Requirement(value).name == name:
            return value
    raise AssertionError(f"{path} does not declare {name!r}")


class _SetupModule(Protocol):
    """Interface used by tests loading the build script dynamically."""

    ROOT_DIR: Path
    VLLM_GR_BASE_VERSION: str
    _cached_device: str | None

    def get_install_requires(self) -> list[str]: ...

    def get_vllm_gr_version(self) -> str: ...

    def warn_deprecated_extra_install(self) -> None: ...


def _load_setup_module() -> _SetupModule:
    setup_path = REPO_ROOT / "setup.py"
    spec = importlib.util.spec_from_file_location("vllm_gr_setup", setup_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm_gr_setup"] = module
    spec.loader.exec_module(module)
    return cast(_SetupModule, module)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("device", "expected_subset", "expected_packages", "excluded_packages"),
    [
        (
            "cpu",
            {"msgspec", "regex", "safetensors>=0.6.2", "PyYAML>=6.0"},
            {"msgspec", "regex", "safetensors", "pyyaml"},
            {"vllm"},
        ),
        (
            "cuda",
            {
                "msgspec",
                "regex",
                "safetensors>=0.6.2",
                "PyYAML>=6.0",
            },
            {"msgspec", "regex", "safetensors", "pyyaml", "vllm"},
            set(),
        ),
        (
            "npu",
            {"msgspec", "regex", "safetensors>=0.6.2", "PyYAML>=6.0"},
            {"msgspec", "regex", "safetensors", "pyyaml"},
            {"vllm"},
        ),
        (
            "mlu",
            {"msgspec", "regex", "safetensors>=0.6.2", "PyYAML>=6.0"},
            {"msgspec", "regex", "safetensors", "pyyaml"},
            {"vllm", "torch", "torch-mlu", "vllm-mlu"},
        ),
    ],
)
def test_get_install_requires_by_device(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected_subset: set[str],
    expected_packages: set[str],
    excluded_packages: set[str],
) -> None:
    gr_setup = _load_setup_module()
    # Reset the module-level cache so the monkeypatched function is used.
    monkeypatch.setattr(gr_setup, "_cached_device", None)
    monkeypatch.setattr(gr_setup, "detect_target_device", lambda: device)
    monkeypatch.setattr(gr_setup, "warn_deprecated_extra_install", lambda: None)

    requirements = set(gr_setup.get_install_requires())
    package_names = {
        Requirement(requirement).name.lower().replace("_", "-") for requirement in requirements
    }

    assert expected_subset.issubset(requirements)
    assert expected_packages.issubset(package_names)
    assert package_names.isdisjoint(excluded_packages)
    if "vllm" in expected_packages:
        assert f"vllm=={gr_setup.VLLM_GR_BASE_VERSION}" in requirements


def test_common_requirements_exclude_vllm_and_dev_tools() -> None:
    names = {
        Requirement(line.split("#", 1)[0].strip()).name.lower().replace("_", "-")
        for line in COMMON_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "-"))
    }
    assert names.isdisjoint({"vllm", "pytest", "ruff"})


@pytest.fixture  # type: ignore[misc]
def version_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _SetupModule:
    module = _load_setup_module()
    monkeypatch.setattr(module, "ROOT_DIR", tmp_path)
    shutil.copy2(REPO_ROOT / "setup.py", tmp_path / "setup.py")
    (tmp_path / "vllm_gr").mkdir()
    (tmp_path / ".gitignore").write_text("vllm_gr/_version.py\n")
    for name in (
        "VLLM_GR_VERSION_OVERRIDE",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_GR",
    ):
        monkeypatch.delenv(name, raising=False)
    return module


def _git_env() -> dict[str, str]:
    """Isolate fixture repositories from developer and runner Git configuration."""
    return {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        env=_git_env(),
    )


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, env=_git_env())


def _init_git(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Packaging Test")
    _git(root, "config", "user.email", "packaging@example.invalid")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-qm", "Initial source")


def _assert_version(module: _SetupModule, expected: str) -> None:
    before = dict(os.environ)
    assert module.get_vllm_gr_version() == expected
    generated = runpy.run_path(str(module.ROOT_DIR / "vllm_gr" / "_version.py"))
    assert generated["__version__"] == expected
    release = Version(expected)
    assert generated["__version_tuple__"][:2] == (release.major, release.minor)
    resolved = package_version(module.ROOT_DIR, module.VLLM_GR_BASE_VERSION)
    assert check_versions({"vllm-gr": resolved}, {"vllm-gr": expected}) == []
    assert dict(os.environ) == before


@pytest.mark.parametrize("device", ["cuda", "npu", "mlu", "cpu"])  # type: ignore[misc]
def test_release_version_by_device(version_source: _SetupModule, device: str) -> None:
    version_source._cached_device = device
    suffix = "" if device == "cuda" else f"+{device}"
    _assert_version(version_source, f"0.30.0{suffix}")


@pytest.mark.parametrize("release_tag", [None, "0.30.1", "0.30.1rc1"])  # type: ignore[misc]
def test_git_version_source(version_source: _SetupModule, release_tag: str | None) -> None:
    root = version_source.ROOT_DIR
    _init_git(root)
    version_source._cached_device = "npu"
    if release_tag:
        _git(root, "tag", f"v{release_tag}")
        _assert_version(version_source, f"{release_tag}+npu")

    next_release = {None: "0.30.0", "0.30.1": "0.30.2", "0.30.1rc1": "0.30.1rc2"}[release_tag]
    versions = []
    for step in (1, 2):
        _git(root, "commit", "--allow-empty", "-qm", f"Development commit {step}")
        node = _git_output(root, "rev-parse", "--short=12", "HEAD").strip()
        distance = step if release_tag else step + 1
        expected = f"{next_release}.dev{distance}+g{node}.npu"
        _assert_version(version_source, expected)
        versions.append(Version(expected))
    assert versions[0] < versions[1]
    (root / ".gitignore").write_text("vllm_gr/_version.py\n*.log\n")
    _assert_version(version_source, expected.removesuffix(".npu") + ".dirty.npu")


def test_repeated_version_generation_does_not_leak_override(version_source: _SetupModule) -> None:
    for device, expected in [("cpu", "0.30.0+cpu"), ("npu", "0.30.0+npu"), ("cuda", "0.30.0")]:
        version_source._cached_device = device
        _assert_version(version_source, expected)


@pytest.mark.parametrize(  # type: ignore[misc]
    "name",
    [
        "VLLM_GR_VERSION_OVERRIDE",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_GR",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
    ],
)
def test_release_version_override(
    version_source: _SetupModule, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "v0.30.0.dev1+custom")
    version_source._cached_device = "cpu"
    _assert_version(version_source, "0.30.0.dev1+custom")


def test_version_override_precedence(
    version_source: _SetupModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION", "0.30.2")
    monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_GR", "0.30.1")
    monkeypatch.setenv("VLLM_GR_VERSION_OVERRIDE", "0.30.0rc1")
    _assert_version(version_source, "0.30.0rc1")
    monkeypatch.delenv("VLLM_GR_VERSION_OVERRIDE")
    _assert_version(version_source, "0.30.1")


def test_invalid_override_does_not_leak_environment(
    version_source: _SetupModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_GR_VERSION_OVERRIDE", "not-a-version")
    before = dict(os.environ)
    with pytest.raises(ValueError):
        version_source.get_vllm_gr_version()
    assert dict(os.environ) == before


def test_failed_version_write_restores_override(
    version_source: _SetupModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_GR_VERSION_OVERRIDE", "0.30.1")
    monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_GR", "0.30.0")
    (version_source.ROOT_DIR / "vllm_gr" / "_version.py").mkdir()
    before = dict(os.environ)
    with pytest.raises(OSError):
        version_source.get_vllm_gr_version()
    assert dict(os.environ) == before


@pytest.mark.parametrize("source_version", ["0.30.1", "0.30.1+cpu", "0.30.1+npu"])  # type: ignore[misc]
def test_sdist_version_is_authoritative_across_devices(
    version_source: _SetupModule, source_version: str
) -> None:
    # Metadata from an unpacked sdist is authoritative without a Git checkout,
    # and no target device re-suffixes it.
    (version_source.ROOT_DIR / "PKG-INFO").write_text(
        f"Metadata-Version: 2.1\nName: vllm-gr\nVersion: {source_version}\n"
    )
    for device in ("cuda", "npu", "cpu"):
        version_source._cached_device = device
        _assert_version(version_source, source_version)


@pytest.fixture  # type: ignore[misc]
def package_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Build the real setup/config/requirements with a small package. This
    # exercises PEP 517 and the SCM file finder without importing Torch/vLLM.
    source = tmp_path / "source"
    source.mkdir()
    for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(REPO_ROOT / name, source / name)
    helper = Path("tools/packaging/git_version.py")
    (source / helper.parent).mkdir(parents=True)
    shutil.copy2(REPO_ROOT / helper, source / helper)
    shutil.copytree(REPO_ROOT / "requirements", source / "requirements")
    (source / "vllm_gr").mkdir()
    (source / "vllm_gr" / "__init__.py").write_text("from ._version import __version__\n")
    for name in ("vllm_gr/platforms/compat.py", "vllm_gr/platforms/ascend/environment.py"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, target)
    (source / ".gitignore").write_text("vllm_gr/_version.py\n*.egg-info/\ndist/\nbuild/\n")
    _init_git(source)
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "Package source")
    _git(source, "tag", f"v{FIXTURE_RELEASE}")
    for name in (
        "VLLM_GR_VERSION_OVERRIDE",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_GR",
    ):
        monkeypatch.delenv(name, raising=False)
    return source


@pytest.mark.parametrize(  # type: ignore[misc]
    ("source_device", "device"),
    [
        ("cuda", "cuda"),
        ("npu", "npu"),
        ("mlu", "mlu"),
        ("cpu", "cpu"),
        ("cpu", "cuda"),
        ("cuda", "npu"),
    ],
)
def test_sdist_wheel_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package_source: Path,
    source_device: str,
    device: str,
) -> None:
    source = package_source
    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", source_device)

    def build(root: Path, kind: str) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"from setuptools.build_meta import build_{kind}; build_{kind}('dist')",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    build(source, "sdist")
    sdist = next((source / "dist").glob("*.tar.gz"))
    unpacked = tmp_path / "unpacked"
    with tarfile.open(sdist) as sdist_archive:
        sdist_archive.extractall(unpacked, filter="data")
    extracted = next(unpacked.iterdir())
    assert (extracted / "requirements" / f"{device}.txt").is_file()
    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", device)
    expected = FIXTURE_RELEASE + ("" if source_device == "cuda" else f"+{source_device}")
    # Resolve by name/version from an sdist, so pip checks filename vs prepared
    # metadata before building. Direct backend calls miss this contract.
    wheel_dir = tmp_path / "wheels"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--no-cache-dir",
            "--find-links",
            str(sdist.parent),
            "--wheel-dir",
            str(wheel_dir),
            f"vllm-gr=={expected}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next(wheel_dir.glob("*.whl"))
    assert BytesParser().parsebytes((extracted / "PKG-INFO").read_bytes())["Version"] == expected
    with zipfile.ZipFile(wheel) as wheel_archive:
        metadata_path = next(
            name for name in wheel_archive.namelist() if name.endswith("/METADATA")
        )
        metadata = BytesParser().parsebytes(wheel_archive.read(metadata_path))
        generated: dict[str, object] = {}
        exec(compile(wheel_archive.read("vllm_gr/_version.py"), "_version.py", "exec"), generated)
    assert metadata["Version"] == expected
    assert generated["__version__"] == expected
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    runtime = [str(req) for req in requirements if req.name == "vllm" and req.marker is None]
    assert runtime == ([_requirement_line("cuda", "vllm")] if device == "cuda" else [])
    assert not any(req.name == "black" for req in requirements)
    if device == "mlu":
        expected_deps = {
            Requirement(line.split("#", 1)[0].strip())
            for line in COMMON_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        assert {req for req in requirements if req.marker is None} == expected_deps


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.cpu_test  # type: ignore[misc]
@pytest.mark.parametrize("device", ["", "mllu"])  # type: ignore[misc]
def test_invalid_explicit_target_fails_before_detection(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", device)
    monkeypatch.setenv("READTHEDOCS", "1")
    with pytest.raises(ValueError, match="Invalid VLLM_GR_TARGET_DEVICE.*mlu"):
        _load_setup_module().get_install_requires()


def test_tagged_editable_install_runtime_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package_source: Path
) -> None:
    """Build/install a tagged GR release and verify its real metadata and import."""
    import packaging

    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", "cuda")
    vllm_version = next(iter(Requirement(_requirement_line("cuda", "vllm")).specifier)).version
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / "bin/python"
    site_packages = Path(
        subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            text=True,
        ).strip()
    )
    # Supply the verifier's pure-Python dependency without network access.
    shutil.copytree(Path(packaging.__file__).parent, site_packages / "packaging")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--no-cache-dir",
            "--target",
            str(site_packages),
            "-e",
            str(package_source),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Only the accelerator runtime is synthetic. GR metadata and imports come
    # from the actual editable wheel installed above, as in CI's worktree flow.
    runtime_metadata = site_packages / f"vllm-{vllm_version}.dist-info"
    runtime_metadata.mkdir()
    (runtime_metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: vllm\nVersion: {vllm_version}\n"
    )
    probe = r"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

source = Path(sys.argv[1])
loader = importlib.util.spec_from_file_location("runtime", sys.argv[2])
runtime = importlib.util.module_from_spec(loader)
loader.loader.exec_module(runtime)
spec = runtime.resolve(source, "cuda")
assert spec["expected"] == {"vllm": sys.argv[3], "vllm-gr": sys.argv[4]}
import vllm_gr
assert Path(vllm_gr.__file__).resolve() == source / "vllm_gr/__init__.py"
assert runtime.metadata.version("vllm-gr") == vllm_gr.__version__ == sys.argv[4]
prefix = Path(sys.prefix).resolve()
modules = {
    name: SimpleNamespace(__file__=str(prefix / name / "__init__.py"))
    for name in ("vllm", "torch", "triton")
}
modules["vllm"].__version__ = sys.argv[3]
modules["torch"].version = SimpleNamespace(cuda="13.0")
modules["vllm.platforms"] = SimpleNamespace(current_platform=SimpleNamespace(device_type="cuda"))
original_import = runtime.importlib.import_module

def import_module(name):
    return modules[name] if name in modules else original_import(name)

with patch.object(runtime.importlib, "import_module", side_effect=import_module):
    report = runtime.verify(source, spec, prefix)
    assert report["ok"], report["errors"]
    # The installed artifact and imported module must also agree on local data.
    vllm_gr.__version__ = sys.argv[4] + "+wrong"
    report = runtime.verify(source, spec, prefix)
    assert not report["ok"]
    assert any("module/distribution version mismatch" in error for error in report["errors"])
print(json.dumps({"expected": spec["expected"], "verified": True}))
"""
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            probe,
            str(package_source),
            str(REPO_ROOT / "tools/ci/resolve_runtime.py"),
            vllm_version,
            FIXTURE_RELEASE,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_warn_deprecated_extra_install_emits_warning() -> None:
    import warnings

    gr_setup = _load_setup_module()
    original_argv = sys.argv
    try:
        sys.argv = ["pip", "install", "-e", ".[gpu]"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gr_setup.warn_deprecated_extra_install()
        assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    finally:
        sys.argv = original_argv


pytestmark = pytest.mark.cpu_test
