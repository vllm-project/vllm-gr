# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Provider API-version adaptation and declared compatibility ranges."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_gr.platforms import compat, legacy
from vllm_gr.platforms.errors import InvalidBackendProviderError
from vllm_gr.platforms.identity import PlatformIdentity
from vllm_gr.platforms.protocol import PROVIDER_API_VERSION


@dataclass(frozen=True)
class LegacyProvider:
    """A version-1 provider: identity by device type, no declared metadata."""

    device_type: str = "cuda"

    def check_environment(self) -> dict[str, str]:
        return {"legacy": self.device_type}

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return 7

    def get_model_runner_hooks(self) -> Any:
        return "hooks"

    def get_attention_execution(self) -> Any:
        return "attention"

    def create_constrained_topk_execution(self, *, device: Any = None) -> Any:
        return ("adapter", device)


def identity(
    device_type: str, platform_enum: str, *, lineage: tuple[str, ...] = ()
) -> PlatformIdentity:
    return PlatformIdentity(
        platform_enum=platform_enum,
        device_type=device_type,
        device_name=device_type,
        platform_class=f"tests.{platform_enum}",
        platform_lineage=lineage or (f"tests.{platform_enum}", "builtins.object"),
    )


def test_absent_api_version_is_read_as_v1() -> None:
    assert legacy.declared_api_version(LegacyProvider()) == 1


def test_non_integer_api_version_is_rejected_explicitly() -> None:
    provider = SimpleNamespace(api_version="2", device_type="cuda")

    with pytest.raises(InvalidBackendProviderError, match="api_version must be an integer"):
        legacy.declared_api_version(provider)


def test_v2_provider_is_not_adapted() -> None:
    provider = SimpleNamespace(api_version=PROVIDER_API_VERSION)

    assert legacy.adapt_legacy_provider(provider, source="entrypoint:x") is None


def test_legacy_cuda_and_npu_map_to_fixed_identities() -> None:
    cuda = legacy.adapt_legacy_provider(LegacyProvider("cuda"), source="entrypoint:cuda")
    npu = legacy.adapt_legacy_provider(LegacyProvider("npu"), source="entrypoint:npu")

    assert cuda is not None and cuda.backend_id == "cuda"
    assert npu is not None and npu.backend_id == "npu"


def test_legacy_adapter_delegates_every_operation() -> None:
    adapted = legacy.adapt_legacy_provider(LegacyProvider("npu"), source="entrypoint:npu")
    assert adapted is not None

    assert adapted.check_environment() == {"legacy": "npu"}
    assert adapted.apply_runtime_patches(force=True) == 7
    assert adapted.get_model_runner_hooks() == "hooks"
    assert adapted.get_attention_execution() == "attention"
    assert adapted.create_constrained_topk_execution(device="npu:0") == ("adapter", "npu:0")


def test_legacy_adapter_rejects_classes_and_missing_methods() -> None:
    with pytest.raises(InvalidBackendProviderError, match="not a class"):
        legacy.adapt_legacy_provider(LegacyProvider, source="test")
    with pytest.raises(InvalidBackendProviderError, match="create_constrained_topk_execution"):
        legacy.adapt_legacy_provider(SimpleNamespace(device_type="cuda"), source="test")


def test_legacy_adapter_matches_exact_device_placement() -> None:
    adapted = legacy.adapt_legacy_provider(LegacyProvider("npu"), source="entrypoint:npu")
    assert adapted is not None

    assert adapted.matches(identity("npu", "OOT")) is True
    assert adapted.matches(identity("cuda", "CUDA")) is False


def test_legacy_cuda_adapter_requires_the_native_cuda_identity() -> None:
    adapted = legacy.adapt_legacy_provider(LegacyProvider("cuda"), source="entrypoint:cuda")
    assert adapted is not None

    assert adapted.matches(identity("cuda", "CUDA"))
    assert not adapted.matches(identity("cuda", "ROCM"))
    assert not adapted.matches(identity("cuda", "OOT"))


def test_legacy_provider_with_an_arbitrary_device_type_is_rejected() -> None:
    with pytest.raises(InvalidBackendProviderError) as exc_info:
        legacy.adapt_legacy_provider(LegacyProvider("xpu"), source="entrypoint:xpu")

    message = str(exc_info.value)
    assert "API version 1" in message
    assert "device_type='xpu'" in message
    assert "api_version=2" in message


def test_legacy_adapter_never_claims_declared_ranges() -> None:
    adapted = legacy.adapt_legacy_provider(LegacyProvider("cuda"), source="entrypoint:cuda")
    assert adapted is not None

    # Capabilities of a version-1 object are never inferred from optional
    # attributes it happens to expose.
    assert adapted.gr_version_range is None
    assert adapted.vllm_version_range is None
    assert adapted.vendor_version_range is None


@dataclass(frozen=True)
class RangedProvider:
    backend_id: str = "ranged"
    api_version: int = PROVIDER_API_VERSION
    gr_version_range: str | None = None
    vllm_version_range: str | None = None
    vendor_version_range: str | None = None
    vendor_distribution: str | None = None


def test_satisfied_declared_ranges_report_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "installed_version", lambda name: "0.30.0")

    assert compat.check_declared_ranges(RangedProvider(vllm_version_range="==0.30.0")) == ()


def test_unsatisfied_range_is_reported_with_provider_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compat, "installed_version", lambda name: "0.22.1")

    failures = compat.check_declared_ranges(RangedProvider(vllm_version_range="==0.30.0"))

    assert len(failures) == 1
    assert "0.22.1" in failures[0]
    assert "'ranged'" in failures[0]


def test_missing_vendor_distribution_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        compat,
        "installed_version",
        lambda name: None if name == "vllm-ascend" else "0.30.0",
    )
    provider = RangedProvider(
        vendor_version_range="==0.30.0rc2",
        vendor_distribution="vllm-ascend",
    )

    failures = compat.check_declared_ranges(provider)

    assert any("vllm-ascend" in failure for failure in failures)


@pytest.mark.parametrize(  # type: ignore[misc]
    "ambient_log_level", [logging.WARNING, logging.ERROR], ids=["warning-root", "error-root"]
)
def test_strict_mode_fails_and_default_mode_only_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    ambient_log_level: int,
) -> None:
    monkeypatch.setattr(compat, "installed_version", lambda name: "0.22.1")
    provider = RangedProvider(vllm_version_range="==0.30.0")

    monkeypatch.delenv(compat._STRICT_ENV, raising=False)
    compat._warn_range_once.cache_clear()
    try:
        with caplog.at_level(ambient_log_level):
            # Collection can raise the root threshold. Capture this logger at
            # WARNING explicitly and restore both levels after the assertion.
            with caplog.at_level(logging.WARNING, logger=compat.logger.name):
                compat.enforce_declared_ranges(provider)
                compat.enforce_declared_ranges(provider)
                records = [record for record in caplog.records if record.name == compat.logger.name]
                assert len(records) == 1
                assert records[0].levelno == logging.WARNING
                assert "compatibility range that does not hold" in records[0].getMessage()

                monkeypatch.setenv(compat._STRICT_ENV, "1")
                with pytest.raises(InvalidBackendProviderError, match=compat._STRICT_ENV):
                    compat.enforce_declared_ranges(provider)
                assert [
                    record for record in caplog.records if record.name == compat.logger.name
                ] == records
    finally:
        compat._warn_range_once.cache_clear()


def test_unparsable_declared_range_is_a_provider_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compat, "installed_version", lambda name: "0.30.0")

    with pytest.raises(InvalidBackendProviderError, match="unparsable version range"):
        compat.check_declared_ranges(RangedProvider(vllm_version_range="not a specifier"))


def test_builtin_providers_declare_the_supported_runtime_pair() -> None:
    from vllm_gr.platforms.cuda import CUDA_BACKEND
    from vllm_gr.platforms.npu import NPU_BACKEND

    for provider in (CUDA_BACKEND, NPU_BACKEND):
        assert provider.vllm_version_range == f"=={compat.SUPPORTED_VLLM_VERSION}"
        assert provider.gr_version_range is not None

    assert NPU_BACKEND.vendor_distribution == "vllm-ascend"
    assert CUDA_BACKEND.vendor_distribution is None


def test_legacy_adapter_is_a_version_two_provider() -> None:
    adapted = legacy.adapt_legacy_provider(LegacyProvider("cuda"), source="entrypoint:cuda")
    assert adapted is not None
    assert adapted.api_version == PROVIDER_API_VERSION

    from vllm_gr.platforms.registry import (
        _reset_backend_registry_for_tests,
        register_backend,
        resolve_backend,
    )

    _reset_backend_registry_for_tests()
    try:
        register_backend(adapted, source="entrypoint:cuda")
        assert resolve_backend("cuda") is adapted
    finally:
        _reset_backend_registry_for_tests()


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
