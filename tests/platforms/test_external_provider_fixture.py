# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Install a host-only out-of-tree provider and exercise the ordinary loader.

The fixture is a real distribution: a package directory plus ``dist-info``
metadata with an ``entry_points.txt`` in ``vllm_gr.platform_backends``. It is
placed on ``sys.path`` in a subprocess, so discovery goes through
``importlib.metadata`` exactly as an installed third-party package would.

It proves loader behavior only. A host-only provider is not a production CPU
backend and demonstrates no accelerator execution.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_PROVIDER_MODULE = '''\
# SPDX-License-Identifier: Apache-2.0
"""Host-only vllm-gr provider used by tests. Not a production backend."""

PROVIDER_API_VERSION = 2


class HostOnlyProvider:
    backend_id = "host_only"
    api_version = PROVIDER_API_VERSION
    device_type = "cpu"
    gr_version_range = ">=0.30.0,<0.31.0"
    vllm_version_range = "==0.30.0"
    vendor_version_range = None
    vendor_distribution = None

    def matches(self, identity):
        return identity.platform_enum == "CPU" and identity.device_type == "cpu"

    def check_environment(self):
        return {"backend": self.backend_id}

    def apply_runtime_patches(self, *, force=False):
        del force
        return 0

    def get_model_runner_hooks(self):
        raise NotImplementedError("host-only provider")

    def get_attention_execution(self):
        raise NotImplementedError("host-only provider")

    def create_constrained_topk_execution(self, *, device=None):
        del device
        return None


BACKEND = HostOnlyProvider()
'''

_LEGACY_PROVIDER_MODULE = '''\
# SPDX-License-Identifier: Apache-2.0
"""Host-only API-v1 vllm-gr provider used by tests."""


class LegacyProvider:
    device_type = "cuda"

    def check_environment(self):
        return {"backend": "legacy"}

    def apply_runtime_patches(self, *, force=False):
        del force
        return 0

    def get_model_runner_hooks(self):
        raise NotImplementedError("host-only provider")

    def get_attention_execution(self):
        raise NotImplementedError("host-only provider")

    def create_constrained_topk_execution(self, *, device=None):
        del device
        return None


BACKEND = LegacyProvider()
'''

_METADATA = """\
Metadata-Version: 2.1
Name: {name}
Version: 1.0.0
Summary: Host-only vllm-gr platform provider fixture
"""

_ENTRY_POINTS = """\
[vllm_gr.platform_backends]
{backend} = {module}:BACKEND
"""


def _install_provider(
    root: Path,
    *,
    distribution: str,
    module: str,
    backend_id: str,
    source: str,
) -> Path:
    """Write an importable package with installed distribution metadata."""
    package_root = root / "site-packages"
    package_dir = package_root / module
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(source, encoding="utf-8")

    dist_info = package_root / f"{distribution.replace('-', '_')}-1.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(_METADATA.format(name=distribution), encoding="utf-8")
    (dist_info / "entry_points.txt").write_text(
        _ENTRY_POINTS.format(backend=backend_id, module=module), encoding="utf-8"
    )
    return package_root


def _run(
    code: str,
    *,
    site_packages: Path,
    project_root: Path,
    device_type: str = "cpu",
) -> subprocess.CompletedProcess[str]:
    preamble = f"""\
import sys
sys.path.insert(0, {str(site_packages)!r})
from types import ModuleType, SimpleNamespace

# Supply the identity that native vLLM would already have selected.
# Entry-point discovery, provider matching and activation use their real paths.
platforms = ModuleType("vllm.platforms")
vllm = ModuleType("vllm")
vllm.platforms = platforms
sys.modules.update({{"vllm": vllm, "vllm.platforms": platforms}})
platforms.current_platform = SimpleNamespace(
    _enum={{"cuda": "CUDA", "cpu": "CPU"}}.get(sys.argv[1], "CPU"),
    device_type=sys.argv[1],
    device_name=sys.argv[1],
)
"""
    return subprocess.run(
        [sys.executable, "-c", preamble + code, device_type],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.l1  # type: ignore[misc]
def test_installed_external_provider_registers_through_the_entry_point_group(
    tmp_path: Path,
) -> None:
    """An installed distribution registers without editing registry or loader."""
    site_packages = _install_provider(
        tmp_path,
        distribution="vllm-gr-host-only",
        module="vllm_gr_host_only",
        backend_id="host_only",
        source=_PROVIDER_MODULE,
    )

    result = _run(
        """
from vllm_gr.platforms import loader
from vllm_gr.platforms.registry import _reset_backend_registry_for_tests

_reset_backend_registry_for_tests()
provider = loader.activate_current_backend()
assert provider.backend_id == "host_only", provider
assert provider.check_environment() == {"backend": "host_only"}
print("ok")
""",
        site_packages=site_packages,
        project_root=Path(__file__).resolve().parents[2],
        device_type="cpu",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


@pytest.mark.l1  # type: ignore[misc]
def test_installed_legacy_provider_is_adapted_only_for_fixed_identities(
    tmp_path: Path,
) -> None:
    site_packages = _install_provider(
        tmp_path,
        distribution="vllm-gr-host-legacy",
        module="vllm_gr_host_legacy",
        backend_id="cuda",
        source=_LEGACY_PROVIDER_MODULE,
    )
    root = Path(__file__).resolve().parents[2]

    accepted = _run(
        """
import os
os.environ["VLLM_GR_DISABLE_BUILTIN_BACKENDS"] = "cuda"
from vllm_gr.platforms import loader
from vllm_gr.platforms.registry import _reset_backend_registry_for_tests

_reset_backend_registry_for_tests()
provider = loader.activate_current_backend()
assert provider.backend_id == "cuda", provider
assert provider.check_environment() == {"backend": "legacy"}
print("ok")
""",
        site_packages=site_packages,
        project_root=root,
        device_type="cuda",
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert "API version 1" in accepted.stderr

    # Without the disable switch the legacy "cuda" entry point collides with
    # the built-in, and a duplicate ID is never resolved by an explicit choice.
    conflicted = _run(
        """
from vllm_gr.platforms import loader
from vllm_gr.platforms.errors import DuplicateBackendRegistrationError
from vllm_gr.platforms.registry import _reset_backend_registry_for_tests

_reset_backend_registry_for_tests()
try:
    loader.activate_current_backend()
except DuplicateBackendRegistrationError as error:
    assert "builtin:cuda" in str(error), error
    print("conflicted")
""",
        site_packages=site_packages,
        project_root=root,
        device_type="cpu",
    )
    assert conflicted.returncode == 0, conflicted.stdout + conflicted.stderr
    assert "conflicted" in conflicted.stdout


@pytest.mark.l1  # type: ignore[misc]
def test_installed_provider_duplicate_id_is_reported_with_both_origins(
    tmp_path: Path,
) -> None:
    site_packages = _install_provider(
        tmp_path,
        distribution="vllm-gr-host-duplicate",
        module="vllm_gr_host_duplicate",
        backend_id="cuda",
        source=_PROVIDER_MODULE.replace('backend_id = "host_only"', 'backend_id = "cuda"'),
    )

    result = _run(
        """
from vllm_gr.platforms import loader
from vllm_gr.platforms.errors import DuplicateBackendRegistrationError
from vllm_gr.platforms.registry import _reset_backend_registry_for_tests

_reset_backend_registry_for_tests()
try:
    loader.activate_current_backend()
except DuplicateBackendRegistrationError as error:
    message = str(error)
    assert "builtin:cuda" in message, message
    assert "entrypoint:cuda" in message, message
    assert "vllm-gr-host-duplicate==1.0.0" in message, message
    print("reported")
""",
        site_packages=site_packages,
        project_root=Path(__file__).resolve().parents[2],
        device_type="cuda",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "reported" in result.stdout


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
def test_installed_provider_plan_round_trip_and_factory_binding(tmp_path: Path) -> None:
    """An external planning extension adds options without a core serializer edit."""
    source = _PROVIDER_MODULE.replace(
        "BACKEND = HostOnlyProvider()",
        """
from types import SimpleNamespace
from vllm_gr.platforms.plan import BackendPlan, PlanData

class PlannedProvider(HostOnlyProvider):
    plan_api_version = 1
    adapter_revision = "host-only-v1"

    def resolve(self, request):
        tile = request.options.to_dict().get("tile", 16)
        if type(tile) is not int or tile not in (16, 32):
            raise ValueError("tile must be 16 or 32")
        return BackendPlan(self.backend_id, self.api_version, request,
                           PlanData({"tile": tile, "requires_attention": True}))

    def validate_worker(self, *, plan, native_runner):
        assert native_runner.device.startswith("cpu")

    def create_model_runner_adapter(self, *, plan, native_runner):
        return SimpleNamespace(plan=plan, owner=native_runner,
                               implementation_id="host.runner.v1")

    def create_attention_executor(self, *, plan, device):
        return SimpleNamespace(plan=plan, device=device,
                               implementation_id="host.attention.v1")

    def create_constrained_topk_execution(self, *, plan=None, device=None):
        return None

BACKEND = PlannedProvider()
""",
    )
    site_packages = _install_provider(
        tmp_path,
        distribution="vllm-gr-host-planned",
        module="host_planned",
        backend_id="host_only",
        source=source,
    )
    code = """
import json
import sys
from types import SimpleNamespace
from vllm_gr.config import GRConfig
from vllm_gr.platforms import loader
from vllm_gr.platforms.identity import PlatformIdentity
from vllm_gr.platforms.plan import BackendPlan
from vllm_gr.platforms.planning import admit_backend_plan
from vllm_gr.platforms.binding import bind_backend_services

loader.current_platform_identity = lambda: PlatformIdentity("CPU", "cpu", "cpu", "test.CPU", ())
provider = loader.activate_current_backend()
engines = []
for tile in (16, 32):
    config = SimpleNamespace(additional_config={"vllm_gr_backend_options": {"tile": tile}})
    GRConfig(beam_max_width=tile).write_to_vllm_config(config)
    plan = admit_backend_plan(config)
    assert plan == BackendPlan.from_wire(json.loads(json.dumps(plan.to_wire())))
    assert ["vllm-gr-host-planned", "1.0.0"] in plan.request.compatibility.to_dict()["provider_distributions"]
    runner = SimpleNamespace(vllm_config=config, device="cpu:0")
    services = bind_backend_services(runner)
    assert services.runner_adapter.owner is runner
    assert services.runner_adapter.plan == services.attention_executor.plan == plan
    assert services.attention_executor.device == "cpu:0"
    assert plan.choices.to_dict()["tile"] == tile
    engines.append(services)
assert engines[0].plan != engines[1].plan
assert not hasattr(provider, "current_plan")
assert not any(name in sys.modules for name in ("torch", "torch_npu", "vllm", "vllm_ascend"))
print("planned")
"""
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(site_packages)!r})\n" + code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "planned" in result.stdout
