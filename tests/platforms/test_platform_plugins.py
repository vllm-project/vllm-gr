# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Discover installed GR platform plugins in fresh processes, including workers."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.helpers.paths import REPO_ROOT

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture(scope="module")  # type: ignore[misc]
def platform_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root: Path = tmp_path_factory.mktemp("vendor-wheel")
    src = root / "project"
    package = src / "gr_plugin_fixture"
    package.mkdir(parents=True)
    (src / "pyproject.toml").write_text(
        """[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gr-test-vendor-platform"
version = "1.2.3"

[project.entry-points."vllm_gr.platform_plugins"]
vendor = "gr_plugin_fixture:probe"
alternate = "gr_plugin_fixture:alternate"
"""
    )
    (package / "__init__.py").write_text(
        """import os
from vllm_gr.platforms.interface import GRPlatform

calls = []
instances = []


class VendorPlatform(GRPlatform):
    from vllm.platforms.interface import Platform as native_platform_type
    backend_id = "vendor"

    def __init__(self):
        instances.append(os.getpid())

    def validate_gr_request(self, config, request):
        from vllm_gr.platforms.errors import GRDeviceRequestError
        if request.beam_params.temperature != 1:
            raise GRDeviceRequestError("vendor requires unit temperature")

    def create_constraint_backend(self, table, *, device, dtype, max_rows, max_top_k):
        from vllm_gr.config import ConstraintBackendKind
        from vllm_gr.constraints.backend import ConstraintBackendInfo
        from vllm_gr.constraints.cpu_backend import CPUConstraintBackend

        assert device.type == "cpu"  # This test wheel runs its operator on CPU.
        info = ConstraintBackendInfo(table.metadata, ConstraintBackendKind.PLATFORM,
                                     max_top_k, max_rows, table.total_bytes, 0)
        return CPUConstraintBackend(table, info)

    device_type = "vendor"
    supports_unconstrained = True

    def get_beam_limits(self, config):
        from dataclasses import replace
        return replace(super().get_beam_limits(config), max_decode_rows=16)


def probe():
    from vllm.platforms import current_platform as native
    calls.append(("vendor", os.getpid()))
    if native.device_type == "cuda" and os.environ.get("GR_TEST_CUDA_REUSE") == "1":
        return "gr_plugin_fixture.cuda.VendorCudaPlatform"
    if native.device_type in ("cuda", "npu", "vendor"):
        return "gr_plugin_fixture.VendorPlatform"
    return None


def alternate():
    calls.append(("alternate", os.getpid()))
    if os.environ.get("GR_TEST_DUPLICATE") == "1":
        return "gr_plugin_fixture.VendorPlatform"
    return None
"""
    )
    (package / "cuda.py").write_text(
        """from vllm_gr.platforms.cuda.platform import CudaGRPlatform

class VendorCudaPlatform(CudaGRPlatform):
    backend_id = "vendor_cuda"
"""
    )
    site = root / "site-packages"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "--no-index",
            "--target",
            str(site),
            str(src),
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return site


def _run_platform_check(platform_wheel: Path, script: str, **env: str) -> None:
    # Use the checkout from this test, including in subprocesses launched by it.
    path = os.pathsep.join((str(REPO_ROOT), str(platform_wheel), os.getenv("PYTHONPATH", "")))
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        env={**os.environ, **env, "PYTHONPATH": path},
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(  # type: ignore[misc]
    "enum,device", [("CUDA", "cuda"), ("OOT", "npu"), ("OOT", "vendor")]
)
def test_installed_plugin_wins_and_is_cached(platform_wheel: Path, enum: str, device: str) -> None:
    _run_platform_check(
        platform_wheel,
        """
        import os
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        import vllm_gr.platforms as table
        import vllm.platforms
        from tests.helpers.native_platform import native_platform
        import gr_plugin_fixture as vendor

        vllm.platforms.current_platform = native_platform(os.environ["ENUM"], os.environ["DEVICE"])
        barrier = Barrier(4)
        def select(_):
            barrier.wait()
            return table.get_current_gr_platform()
        with ThreadPoolExecutor(max_workers=4) as pool:
            platforms = list(pool.map(select, range(4)))
        first = table.get_current_gr_platform()
        assert all(item is first for item in platforms) and type(first) is vendor.VendorPlatform
        assert vendor.instances == [os.getpid()]
        assert sorted(vendor.calls) == [("alternate", os.getpid()), ("vendor", os.getpid())]
        """,
        ENUM=enum,
        DEVICE=device,
    )


def test_multiple_external_matches_fail_before_instantiation(platform_wheel: Path) -> None:
    _run_platform_check(
        platform_wheel,
        """
        from types import SimpleNamespace
        import gr_plugin_fixture as vendor
        import vllm_gr.platforms as table
        from vllm_gr.platforms.errors import AmbiguousBackendError
        import vllm.platforms
        from tests.helpers.native_platform import native_platform

        vllm.platforms.current_platform = native_platform("CUDA", "cuda")
        try:
            table.get_current_gr_platform()
        except AmbiguousBackendError as exc:
            assert "gr-test-vendor-platform:alternate" in str(exc)
            assert "gr-test-vendor-platform:vendor" in str(exc)
        else:
            raise AssertionError("expected ambiguity including both plugins from one wheel")
        assert not vendor.instances
        """,
        GR_TEST_DUPLICATE="1",
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")  # type: ignore[misc]
def test_forked_worker_resolves_its_own_platform(platform_wheel: Path) -> None:
    _run_platform_check(
        platform_wheel,
        """
        import os
        from types import SimpleNamespace
        import gr_plugin_fixture as vendor
        import vllm_gr.platforms as table
        import vllm.platforms
        from tests.helpers.native_platform import native_platform

        vllm.platforms.current_platform = native_platform("OOT", "vendor")
        parent_pid = os.getpid()
        parent_platform = table.get_current_gr_platform()
        child = os.fork()
        if child == 0:
            local = table.get_current_gr_platform()
            assert type(local) is type(parent_platform) and local is not parent_platform
            assert vendor.instances == [parent_pid, os.getpid()]
            assert ("vendor", os.getpid()) in vendor.calls
            os._exit(0)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        """,
    )


def test_incompatible_or_broken_winner_never_falls_back(platform_wheel: Path) -> None:
    _run_platform_check(
        platform_wheel,
        """
        from types import SimpleNamespace
        import gr_plugin_fixture as vendor
        import vllm_gr.platforms as table
        import vllm.platforms
        from tests.helpers.native_platform import native_platform

        vllm.platforms.current_platform = native_platform("CUDA", "cuda")
        original_probe = vendor.probe
        for malformed, diagnostic in ((lambda identity: None, "zero arguments"),
                                      (lambda: vendor.VendorPlatform, "class path or None")):
            vendor.probe = malformed
            try:
                table.get_current_gr_platform()
            except TypeError as exc:
                assert diagnostic in str(exc)
            else:
                raise AssertionError("malformed probe silently fell back to CUDA")
        assert not vendor.instances and "gr_plugin_fixture.cuda" not in __import__("sys").modules
        vendor.probe = original_probe
        vendor.VendorPlatform.api_version = 300
        try:
            table.get_current_gr_platform()
        except RuntimeError as exc:
            assert "platform API 300" in str(exc) and "gr-test-vendor-platform" in str(exc)
        else:
            raise AssertionError("incompatible external platform was accepted")
        assert vendor.instances == []
        vendor.VendorPlatform.api_version = 3
        from tests.helpers.native_platform import VendorNative
        vendor.VendorPlatform.native_platform_type = VendorNative
        try:
            table.get_current_gr_platform()
        except RuntimeError as exc:
            assert "incompatible native platform" in str(exc)
        else:
            raise AssertionError("incompatible native worker family was accepted")
        vendor.VendorPlatform.native_platform_type = type(vllm.platforms.current_platform)
        def broken(self):
            raise ValueError("driver unavailable")
        vendor.VendorPlatform.__init__ = broken
        try:
            table.get_current_gr_platform()
        except RuntimeError as exc:
            assert "gr-test-vendor-platform" in str(exc) and isinstance(exc.__cause__, ValueError)
        else:
            raise AssertionError("broken external platform fell back to a built-in")
        """,
    )


def test_builtin_uniqueness_only_applies_without_external_match(platform_wheel: Path) -> None:
    _run_platform_check(
        platform_wheel,
        """
        from types import SimpleNamespace
        import gr_plugin_fixture as vendor
        import vllm_gr.platforms as table
        from vllm_gr.platforms.errors import AmbiguousBackendError, UnsupportedBackendError
        import vllm.platforms
        from tests.helpers.native_platform import native_platform

        table.BUILTIN_PROBES["duplicate"] = table.cuda_platform_plugin
        vllm.platforms.current_platform = native_platform("CUDA", "cuda")
        assert type(table.get_current_gr_platform()) is vendor.VendorPlatform

        vllm.platforms.current_platform = native_platform("CPU", "cpu")
        try:
            table.get_current_gr_platform()
        except UnsupportedBackendError:
            pass
        else:
            raise AssertionError("matched a platform on a CPU-only host")

        def inactive():
            return None
        vendor.probe = inactive
        vllm.platforms.current_platform = native_platform("CUDA", "cuda")
        try:
            table.get_current_gr_platform()
        except AmbiguousBackendError:
            pass
        else:
            raise AssertionError("ambiguous built-ins must raise when no external plugin matches")
        """,
    )


def test_external_eager_engine_serves_a_frontend_without_the_wheel(
    platform_wheel: Path, tmp_path: Path
) -> None:
    wire = str(tmp_path / "engine.json")
    _run_platform_check(
        platform_wheel,
        """
        import json, os
        from pathlib import Path
        from types import SimpleNamespace
        from vllm_gr.engine.startup import configure_model_runner_default
        import vllm_gr.platforms as table
        from vllm_gr.platforms.beam_requests import engine_request_validator
        import vllm.platforms
        from tests.helpers.native_platform import native_platform
        from tests.entrypoints.beam_search.test_beam_search_input_processor import _make_engine
        from tests.helpers.beam_requests import make_batch_request
        from vllm_gr.config import ConstraintConfig, ConstraintMode, GRConfig

        vllm.platforms.current_platform = native_platform("OOT", "vendor")
        native = _make_engine().vllm_config
        native.device_config = SimpleNamespace(device="vendor")
        GRConfig(
            beam_max_width=2, beam_max_decode_steps=3, beam_graph_enabled=False,
            constraint=ConstraintConfig(ConstraintMode.NONE),
        ).write_to_vllm_config(native)
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        configure_model_runner_default()
        assert "VLLM_USE_V2_MODEL_RUNNER" not in os.environ
        from dataclasses import asdict
        native.additional_config["vllm_gr_limits"] = asdict(table.get_current_gr_platform().get_beam_limits(native))
        check = engine_request_validator(native)
        assert check is not None
        from msgspec.structs import replace
        request = make_batch_request()
        def with_temperature(t):
            return replace(request, beam_params=replace(request.beam_params, temperature=t))
        check(with_temperature(1.0))
        try:
            check(with_temperature(0.5))
        except ValueError as exc:
            assert "vendor requires unit temperature" in str(exc)
        else:
            raise AssertionError("missing vendor request rejection")
        Path(os.environ["WIRE"]).write_text(json.dumps(native.additional_config))
        """,
        WIRE=wire,
    )
    _run_platform_check(
        tmp_path,  # No installed platform wheel on the frontend's import path.
        """
        import asyncio, importlib.util, json, os
        from pathlib import Path
        from types import SimpleNamespace
        from tests.entrypoints.beam_search.test_beam_search_input_processor import _make_engine
        from vllm_gr.entrypoints.beam_search.input_processor import BeamSearchInputProcessor, OnlineBeamSearchRequest
        from vllm_gr.sampling_params import BeamSearchParams

        assert importlib.util.find_spec("gr_plugin_fixture") is None
        frontend = _make_engine(additional_config=json.loads(Path(os.environ["WIRE"]).read_text()))
        frontend.vllm_config.device_config = SimpleNamespace(device="cpu")
        params = BeamSearchParams(beam_width=2, max_tokens=3, ignore_eos=True)
        serving = SimpleNamespace(input_processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2)))
        request = asyncio.run(BeamSearchInputProcessor(frontend).process_online_inputs(
            serving, OnlineBeamSearchRequest(([1, 2],), "external", params)))
        assert request.beam_params.beam_width == 2 and len(request.items) == 1
        """,
        WIRE=wire,
    )


@pytest.mark.parametrize("enum,device", [("OOT", "vendor"), ("CUDA", "cuda")])  # type: ignore[misc]
def test_external_canonical_table_load_and_reload(
    platform_wheel: Path, tmp_path: Path, enum: str, device: str
) -> None:
    _run_platform_check(
        platform_wheel,
        """
        import os
        from pathlib import Path
        from types import SimpleNamespace
        import torch
        import gr_plugin_fixture as vendor
        from tests.constraints.test_table import make_table
        from vllm_gr.config import ConstraintBackendKind, gr_config_from_mapping, gr_config_to_mapping
        from vllm_gr.constraints.artifact import save_constraint_table_artifact
        from vllm_gr.constraints.backend import ConstraintMetadata
        from vllm_gr.engine.arg_utils import _resolve_constraint_backend
        import vllm_gr.platforms as table
        import vllm.platforms
        from tests.helpers.native_platform import native_platform
        from vllm_gr.platforms.interface import GRPlatform
        from vllm_gr.v1.worker.model_runner_patch import make_load_model_with_constraint_backend

        vllm.platforms.current_platform = native_platform(os.environ["ENUM"], os.environ["DEVICE"])
        path = Path(os.environ["ARTIFACT"])
        artifact = make_table()
        digest = save_constraint_table_artifact(artifact, path)
        config = gr_config_from_mapping({"gr": {
            "beam_width": [2] * 3, "candidate_width": [2] * 3, "constraint_mode": "table",
            "constraint_table": {"path": str(path), "backend": "auto", "max_top_k": 2,
                                 "digests": {"artifact": digest, "tokenizer": artifact.metadata.tokenizer_digest}},
        }})
        from vllm_gr.constraints.loader import load_constraint_backend
        direct = load_constraint_backend(config.constraint.table, device=torch.device("cpu"), vocab_size=6,
                                         dtype=torch.float32, max_rows=2)
        assert direct.info.backend is ConstraintBackendKind.PLATFORM
        direct.close()
        config = gr_config_from_mapping(gr_config_to_mapping(_resolve_constraint_backend(config)))
        assert config.constraint.table.backend is ConstraintBackendKind.PLATFORM
        vllm_config = SimpleNamespace(additional_config={})
        config.write_to_vllm_config(vllm_config)


        runner = SimpleNamespace(vllm_config=vllm_config, device=torch.device("cpu"), model=torch.nn.Module(),
                                 model_config=SimpleNamespace(dtype=torch.float32, get_vocab_size=lambda: 6))
        make_load_model_with_constraint_backend(lambda self: None)(runner)
        versions = runner.constraint_backend
        first = versions.get()
        metadata = ConstraintMetadata(torch.zeros((1, 3), dtype=torch.int32), torch.ones(1), 0, 2)
        assert set(first.select(torch.arange(6).view(1, 6).float(), metadata).token_ids[0].tolist()) == {1, 3}
        versions.reload(str(path), "v2")
        assert versions.get() is not first and versions.info.table_metadata.artifact_digest == digest

        original = versions.get()
        from unittest.mock import Mock
        factory = vendor.VendorPlatform.create_constraint_backend
        rejected = []
        def wrong_capacity(self, table, *, device, dtype, max_rows, max_top_k):
            candidate = factory(self, table, device=device, dtype=dtype, max_rows=max_rows - 1, max_top_k=max_top_k)
            wrapped = Mock(wraps=candidate, info=candidate.info)
            rejected.append(wrapped)
            return wrapped
        vendor.VendorPlatform.create_constraint_backend = wrong_capacity
        try:
            versions.reload(str(path), "incompatible")
        except ValueError as exc:
            assert "incompatible constraint metadata" in str(exc)
        else:
            raise AssertionError("published an undersized table")
        rejected[0].close.assert_called_once_with()
        assert versions.get() is original

        vendor.VendorPlatform.create_constraint_backend = GRPlatform.create_constraint_backend
        try:
            versions.reload(str(path), "v3")
        except NotImplementedError as exc:
            assert "vendor" in str(exc) and "canonical constraint backend" in str(exc)
        else:
            raise AssertionError("missing factory accepted")
        assert versions.get() is original
        versions.close()
        """,
        ENUM=enum,
        DEVICE=device,
        ARTIFACT=str(tmp_path / "constraints.safetensors"),
    )


def test_external_cuda_subclass_reuses_builtin_worker_and_tracks_wheel_version(
    platform_wheel: Path,
) -> None:
    _run_platform_check(
        platform_wheel,
        """
        import vllm.platforms
        from tests.helpers.native_platform import native_platform
        vllm.platforms.current_platform = native_platform("CUDA", "cuda")
        from vllm_gr.platforms import current_gr_platform
        from vllm_gr.platforms.cuda.platform import CudaGRPlatform
        from vllm_gr.v1.engine.worker_readiness import _runtime_versions
        assert type(current_gr_platform).__name__ == "VendorCudaPlatform"
        assert isinstance(current_gr_platform, CudaGRPlatform)
        assert current_gr_platform.get_gr_worker_cls() == CudaGRPlatform().get_gr_worker_cls()
        assert _runtime_versions()["gr-test-vendor-platform"] == "1.2.3"
    """,
        GR_TEST_CUDA_REUSE="1",
    )
