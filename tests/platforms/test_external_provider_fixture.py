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

import os
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
        # The built-in CPU backend also matches a CPU platform; select the fixture.
        env={**os.environ, **({"VLLM_GR_BACKEND": "host_only"} if device_type == "cpu" else {})},
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
@pytest.mark.parametrize(
    "plan_version,with_layers", [(1, False), (2, False), (3, False), (3, True)]
)  # type: ignore[misc]
def test_installed_provider_plan_round_trip_and_factory_binding(
    tmp_path: Path, plan_version: int, with_layers: bool
) -> None:
    """An external planning extension adds options without a core serializer edit."""
    source = _PROVIDER_MODULE.replace(
        "BACKEND = HostOnlyProvider()",
        """
from types import SimpleNamespace
from vllm_gr.platforms.plan import BackendPlan, PlanData

class Attention:
    def __init__(self, plan=None, device=None):
        self.plan = plan
        self.device = device
        self.width = plan.choices.to_dict()["tile"] if plan else 0
        self.implementation_id = f"host.attention.{self.width}"
        self.closed = False

    def get_flash_attn_version(self, **kwargs):
        return 2

    def cascade_attention(self, *, output, **kwargs):
        assert self.plan is not None and not self.closed
        output.fill_(self.width)

    prefill_attention = cascade_attention

    def reshape_and_cache(self, *args):
        pass

    def cache_unshared_kv(self, *args, **kwargs):
        pass

    def close(self):
        self.closed = True

class PlannedProvider(HostOnlyProvider):
    plan_api_version = PLAN_VERSION
    adapter_revision = "host-only-v1"

    def resolve(self, request):
        fields = request.config.to_dict()
        assert "admin_port" not in fields["deployment"]
        if self.plan_api_version == 1:
            table = fields["constraint_table"]
        else:
            table = fields["constraint"]["table"]
        assert table["backend"] == "cpu"
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
        return Attention(plan, device)

    def get_attention_execution(self):
        return Attention()

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
        source=source.replace("PLAN_VERSION", str(plan_version)),
    )
    if plan_version < 3:
        result = _run(
            """
from vllm_gr.platforms import loader
from vllm_gr.platforms.errors import InvalidBackendProviderError
try:
    loader.activate_current_backend()
except InvalidBackendProviderError as exc:
    assert "plan_api_version=" in str(exc), exc
    assert "supported=3" in str(exc), exc
else:
    raise AssertionError("plan API v1 provider was registered")
""",
            site_packages=site_packages,
            project_root=Path(__file__).resolve().parents[2],
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    code = """
import json
import sys
from types import SimpleNamespace
from dataclasses import replace
from tests.helpers.gr_config import table_constraint
from vllm_gr.config import ConstraintBackendKind, GRConfig
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
    constraint = table_constraint()
    constraint = replace(constraint, table=replace(constraint.table, backend=ConstraintBackendKind.CPU))
    GRConfig(beam_max_width=tile, constraint=constraint).write_to_vllm_config(config)
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
    if with_layers:
        pytest.importorskip("torch")
        pytest.importorskip("vllm")
        code += """
import torch
from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl
from vllm_gr.v1.engine.worker_readiness import worker_backend_readiness
from vllm_gr.v1.worker.runner_binding import close_backend_binding

def run(layer, prefill):
    metadata = SimpleNamespace(slot_mapping=torch.tensor([0]), is_prefill_graph_exec=prefill)
    x = torch.zeros(1, 1, 8)
    return layer.impl.forward(layer, x, x, x, torch.zeros(2, 1, 16, 16), metadata,
                              output=torch.empty_like(x))

live = []
for services in engines:
    runner = services.runner_adapter.owner
    close_backend_binding(runner)
    layer = torch.nn.Module()
    layer.impl = BeamAttentionImpl(1, 8, 1., 1, None, None, "auto",
                                   kv_sharing_target_layer_name="shared")
    runner.model = layer
    binding = bind_backend_services(runner)
    report = worker_backend_readiness(SimpleNamespace(model_runner=runner))
    assert report["services"]["attention"] == layer.impl.execution.implementation_id
    assert binding.attention_executor is layer.impl.execution
    live.append((runner, layer, binding.attention_executor))
for runner, layer, executor in live:
    for prefill in (False, True):
        assert torch.all(run(layer, prefill) == executor.width)
    close_backend_binding(runner)
    assert executor.closed
    assert "execution" not in vars(layer.impl)

# A provider which cannot execute the already-loaded layer must fail atomically.
from unittest.mock import patch
bad = provider.create_attention_executor(plan=plan, device="cpu:0")
bad.get_flash_attn_version = lambda **kwargs: 3
with patch.object(provider, "create_attention_executor", return_value=bad):
    try:
        bind_backend_services(runner)
    except RuntimeError as error:
        assert "FlashAttention version" in str(error)
    else:
        raise AssertionError("incompatible attention executor was published")
assert bad.closed
assert runner._gr_backend_binding is None
assert layer.impl.execution is not bad
"""
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(site_packages)!r})\n" + code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "VLLM_GR_BACKEND": "host_only"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "planned" in result.stdout


@pytest.mark.l1  # type: ignore[misc]
def test_installed_provider_installs_and_verifies_native_runner(tmp_path: Path) -> None:
    source = _PROVIDER_MODULE.replace(
        "BACKEND = HostOnlyProvider()",
        """
from types import SimpleNamespace
from vllm_gr.platforms.plan import BackendPlan, PlanData
from vllm_gr.v1.worker.runner_lifecycle import has_runner_lifecycle, install_runner_lifecycle

class HostRunner:
    def prepare_inputs(self, value):
        return value

    def execute_model(self, value):
        return value

class Hooks:
    def discover_worker_runners(self):
        return (("HostRunner", HostRunner),)

    def is_patched(self, cls):
        return has_runner_lifecycle(cls, __name__)

    def install(self):
        install_runner_lifecycle(HostRunner, __name__, prepare="prepare_inputs")

class RunnerProvider(HostOnlyProvider):
    plan_api_version = 3
    adapter_revision = "host-v1"

    def get_model_runner_hooks(self):
        return Hooks()

    def resolve(self, request):
        return BackendPlan(self.backend_id, self.api_version, request, PlanData({"runner": __name__}))

    def validate_worker(self, *, plan, native_runner):
        assert self.get_model_runner_hooks().is_patched(type(native_runner))

    def create_model_runner_adapter(self, *, plan, native_runner):
        width = plan.request.config.to_dict()["beam_max_width"]
        return SimpleNamespace(implementation_id="host.runner", prepare_inputs=lambda v: v,
                               execute_model=lambda v: (width, v))

    def create_attention_executor(self, *, plan, device):
        raise AssertionError("not requested")

BACKEND = RunnerProvider()
""",
    )
    site_packages = _install_provider(
        tmp_path,
        distribution="vllm-gr-host-runner",
        module="host_runner",
        backend_id="host_only",
        source=source,
    )
    result = _run(
        """
import logging
from unittest.mock import patch
from types import ModuleType

logger = ModuleType("vllm.logger")
logger.init_logger = logging.getLogger
sys.modules["vllm.logger"] = logger
from vllm_gr.config import GRConfig
from vllm_gr.platforms import interface, protocol
from vllm_gr.platforms.binding import worker_backend_readiness
from vllm_gr.platforms.planning import admit_backend_plan
from vllm_gr.platforms.registry import get_active_backend
from vllm_gr.v1.worker import model_runner_patch as worker

assert protocol.GRBackendProvider is interface.GRBackendProvider
assert protocol.ModelRunnerPatchHooks is interface.ModelRunnerPatchHooks
assert protocol.BeamAttentionExecution is interface.BeamAttentionExecution
runners = worker.discover_active_worker_runners()
from host_runner import BACKEND, HostRunner
assert runners == (("HostRunner", HostRunner),)
assert get_active_backend() is BACKEND
assert not worker.is_active_worker_runner_patched(HostRunner)
# The native multiprocessing guard is independent of the runner protocol.
with patch("vllm_gr.v1.worker.gr_async_output.install_async_output_guard"):
    worker.apply_worker_patches()
    installed = HostRunner.execute_model
    worker.apply_worker_patches()
assert HostRunner.execute_model is installed
assert HostRunner().execute_model(42) == 42
assert worker.is_active_worker_runner_patched(HostRunner)
for width in (4, 8):
    runner = HostRunner()
    runner.vllm_config = SimpleNamespace(additional_config={})
    GRConfig(beam_max_width=width).write_to_vllm_config(runner.vllm_config)
    plan = admit_backend_plan(runner.vllm_config)
    report = worker_backend_readiness(SimpleNamespace(model_runner=runner))
    assert report["fingerprint"] == plan.fingerprint
    assert runner.execute_model(42) == (width, 42)
assert not any(f"vllm_gr.v1.worker.{name}" in sys.modules for name in
               ("cuda.v1.gpu_model_runner_patch", "cuda.v2.gpu_model_runner_v2_patch", "npu.npu_model_runner_patch"))
""",
        site_packages=site_packages,
        project_root=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stdout + result.stderr
