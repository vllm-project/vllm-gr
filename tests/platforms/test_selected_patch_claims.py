# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Selection and verification using the same preflight as plugin initialization."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.platforms.test_registry import FakeProvider
from vllm_gr.platforms.patches import CORE_PATCH_IDS, SelectedPatchClaims
from vllm_gr.platforms.registry import (
    _reset_backend_registry_for_tests,
    register_backend,
)

pytestmark = [pytest.mark.l1, pytest.mark.cpu_test]


@pytest.fixture  # type: ignore[misc]
def provider() -> Iterator[Any]:
    original = FakeProvider()
    # API v2 selection and failure reporting need no legacy device_type.
    provider = SimpleNamespace(
        **{
            name: getattr(original, name)
            for name in dir(original)
            if not name.startswith("_") and name != "device_type"
        }
    )
    provider.patch_ids = ("sample.install",)
    provider.apply_runtime_patches = Mock(return_value=0)
    provider.runtime_patch_checks = {"sample.install": Mock(return_value=True)}
    _reset_backend_registry_for_tests()
    register_backend(provider, source="fixture-distribution==1.0")
    yield provider
    _reset_backend_registry_for_tests()


def test_rechecks_live_patch_even_when_installer_reports_zero(provider: Any) -> None:
    claim = SelectedPatchClaims.from_provider(provider, CORE_PATCH_IDS)
    assert claim.install(force=True) == 0
    probe = provider.runtime_patch_checks["sample.install"]
    probe.return_value = False
    with pytest.raises(RuntimeError, match="sample.install.*fixture-distribution.*backend_id=fake"):
        claim.install(force=True)
    assert probe.call_count == 2
    assert provider.apply_runtime_patches.call_count == 2


@pytest.mark.parametrize(  # type: ignore[misc]
    "claimed", [(CORE_PATCH_IDS[0],), ("sample.install", "sample.install"), ("extra",)]
)
def test_invalid_ownership_fails_before_installation(
    provider: Any, claimed: tuple[str, ...]
) -> None:
    provider.patch_ids = claimed
    with pytest.raises(ValueError, match="(Duplicate|Missing).*fixture-distribution"):
        SelectedPatchClaims.from_provider(provider, CORE_PATCH_IDS)
    provider.apply_runtime_patches.assert_not_called()


def test_bad_native_installer_preserves_original_exception_and_origin(provider: Any) -> None:
    cause = ValueError("missing binding")
    provider.apply_runtime_patches.side_effect = cause
    claim = SelectedPatchClaims.from_provider(provider, CORE_PATCH_IDS)
    with pytest.raises(RuntimeError, match="backend 'fake'.*fixture-distribution") as raised:
        claim.install()
    assert raised.value.__cause__ is cause
