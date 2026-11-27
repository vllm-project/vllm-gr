# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host-only contracts for the external MLU provider and its canonical factory."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from tests.constraints.test_table import make_table
from tests.helpers.paths import REPO_ROOT
from tests.platforms.test_plugin_loader import FakeProvider
from vllm_gr.config import (
    ConstraintBackendKind,
    ConstraintTableSpec,
    gr_config_from_mapping,
    gr_config_to_mapping,
)
from vllm_gr.config.models import resolve_constraint_backend
from vllm_gr.constraints import loader
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.backend import ConstraintBackendInfo
from vllm_gr.platforms import activate_backend, register_backend
from vllm_gr.platforms.constraint_backend import canonical_constraint_provider
from vllm_gr.platforms.errors import UnsupportedBackendError
from vllm_gr.platforms.registry import _reset_backend_registry_for_tests

pytestmark = pytest.mark.cpu_test


@pytest.fixture  # type: ignore[misc]
def active_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_GR_BACKEND", raising=False)
    _reset_backend_registry_for_tests()
    yield
    _reset_backend_registry_for_tests()


@pytest.mark.l1  # type: ignore[misc]
def test_canonical_factory_is_opt_in_for_existing_providers(active_registry: None) -> None:
    with pytest.raises(UnsupportedBackendError, match="No active GR provider.*mlu"):
        canonical_constraint_provider("mlu")
    provider = FakeProvider(backend_id="existing", device_type="mlu")
    register_backend(provider)
    assert activate_backend("existing") is provider
    with pytest.raises(UnsupportedBackendError, match="requires version 1"):
        canonical_constraint_provider("mlu")
    with pytest.raises(UnsupportedBackendError, match="serves 'mlu', not.*'cuda'"):
        canonical_constraint_provider("cuda")


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("version", [2, True, 1])  # type: ignore[misc]
def test_canonical_factory_rejects_bad_extension(
    monkeypatch: pytest.MonkeyPatch, version: int
) -> None:
    provider = SimpleNamespace(
        backend_id="mlu", device_type="mlu", constraint_backend_api_version=version
    )
    monkeypatch.setattr("vllm_gr.platforms.constraint_backend.get_active_backend", lambda: provider)
    expected = (
        "requires version 1"
        if type(version) is not int or version != 1
        else "missing create_constraint_backend"
    )
    with pytest.raises(UnsupportedBackendError, match=expected):
        canonical_constraint_provider("mlu")


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("backend_name", ["auto", "mlu"])  # type: ignore[misc]
def test_mlu_constraint_config_round_trips(backend_name: str) -> None:
    config = gr_config_from_mapping(
        {
            "gr": {
                "constraint_mode": "table",
                "constraint_table": {
                    "backend": backend_name,
                    "path": "/table.safetensors",
                    "max_top_k": 4,
                    "digests": {
                        "artifact": "sha256:" + "a" * 64,
                        "tokenizer": "sha256:" + "b" * 64,
                    },
                },
            },
        }
    )
    resolved = resolve_constraint_backend(config, "mlu")
    assert resolved.constraint is not None and resolved.constraint.table is not None
    assert resolved.constraint.table.backend is ConstraintBackendKind.MLU
    assert gr_config_from_mapping(gr_config_to_mapping(resolved)) == resolved


@pytest.mark.l1  # type: ignore[misc]
def test_mlu_loader_rejects_other_device_before_loading_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = Mock(side_effect=AssertionError("invalid placement must not load the artifact"))
    monkeypatch.setattr(loader, "load_constraint_table_artifact", artifact)
    with pytest.raises(RuntimeError, match="MLU constraint backend requires a MLU device"):
        loader.load_constraint_backend(
            _spec(Path("/missing.safetensors"), "sha256:" + "a" * 64),
            device=torch.device("cpu"),
            vocab_size=6,
            dtype=torch.float16,
            max_rows=1,
        )
    artifact.assert_not_called()


def _spec(path: Path, digest: str) -> ConstraintTableSpec:
    return ConstraintTableSpec(
        backend=ConstraintBackendKind.MLU,
        enabled=True,
        max_top_k=4,
        path=str(path),
        artifact_digest=digest,
        tokenizer_digest="sha256:" + "b" * 64,
    )


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("ready", [True, False])  # type: ignore[misc]
def test_loader_passes_validated_artifact_and_closes_on_failed_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ready: bool
) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    # Only placement is simulated. Digest and artifact parsing are real.
    device = cast(torch.device, SimpleNamespace(type="mlu"))
    monkeypatch.setattr(loader, "_validate_loader_inputs", lambda **_: None)
    backend = Mock()

    def create(*, artifact: Any, **kwargs: object) -> Any:
        assert artifact.metadata.artifact_digest == digest
        assert kwargs == {"device": device, "dtype": torch.float16, "max_rows": 2, "max_top_k": 4}
        backend.info = ConstraintBackendInfo(
            artifact.metadata, ConstraintBackendKind.MLU, 4, 2, 0, artifact.total_bytes
        )
        return backend

    provider = SimpleNamespace(create_constraint_backend=create)
    monkeypatch.setattr(loader, "canonical_constraint_provider", lambda kind: provider)
    probe = Mock(side_effect=None if ready else RuntimeError("device readiness failed"))
    monkeypatch.setattr(loader, "_run_device_readiness", probe)

    def load(spec: ConstraintTableSpec) -> object:
        return loader.load_constraint_backend(
            spec, device=device, vocab_size=6, dtype=torch.float16, max_rows=2
        )

    with pytest.raises(ValueError, match="digest"):
        load(_spec(path, "sha256:" + "a" * 64))
    probe.assert_not_called()
    if ready:
        assert load(_spec(path, digest)) is backend
        backend.close.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="device readiness failed"):
            load(_spec(path, digest))
        backend.close.assert_called_once()
    probe.assert_called_once()


@pytest.mark.l3  # type: ignore[misc]
def test_mlu_plugin_wheel_discovery_is_lazy_and_startup_fails_closed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(
        REPO_ROOT / "plugins" / "mlu", project, ignore=shutil.ignore_patterns("__pycache__")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('dist')",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel,) = (project / "dist").glob("*.whl")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(
            next(name for name in archive.namelist() if name.endswith("/METADATA"))
        )
        assert b"Requires-Dist:" not in metadata
        archive.extractall(installed)
    smoke = """
import importlib.abc
import os
import sys
from types import ModuleType, SimpleNamespace

sys.path.insert(0, sys.argv[1])
os.environ.pop('VLLM_GR_BACKEND', None)
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
blocked = {'torch', 'torch_mlu', 'torch_npu', 'triton', 'vllm',
           'vllm_gr.ops.cuda', 'vllm_gr.ops.npu'}
class BlockAccelerators(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or any(fullname.startswith(name + '.') for name in blocked):
            raise AssertionError(f'unexpected runtime import: {fullname}')
sys.meta_path.insert(0, BlockAccelerators())

from vllm_gr.platforms import loader
from vllm_gr.platforms.errors import UnsupportedBackendError
loader.load_backend_plugins()
provider = loader.select_backend(SimpleNamespace(device_type='mlu', platform_enum='OOT')).provider
assert provider.backend_id == provider.device_type == 'mlu'
assert type(provider).__module__ == 'vllm_gr_mlu.provider'
assert not blocked.intersection(sys.modules)
assert 'vllm_gr_mlu.constraints.selection' not in sys.modules

native = ModuleType('vllm.platforms')
native.current_platform = SimpleNamespace(device_type='mlu')
sys.modules['vllm.platforms'] = native
from vllm_gr.plugin import initialize_runtime
try:
    initialize_runtime()
except UnsupportedBackendError as exc:
    assert 'RFC #603 P0' in str(exc)
else:
    raise AssertionError('unvalidated native Runner was accepted')
assert 'vllm_gr.patch' not in sys.modules
assert loader.activate_current_backend() is provider

blocked.remove('torch')
from vllm_gr.constraints.loader import load_constraint_backend
provider.create_constrained_topk_execution()
assert not any(name.startswith(('vllm_gr.ops.cuda', 'vllm_gr.ops.npu')) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", smoke, str(installed)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
