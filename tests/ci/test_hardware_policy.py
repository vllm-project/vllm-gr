# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Host policy tests: selection, legacy compatibility, and preflight failures."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.helpers.hardware import (
    SKUS,
    DeviceInventory,
    Hardware,
    model_from_name,
    parse_target,
    probe_devices,
    validate_inventory,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
pytest_plugins = ("pytester",)


@pytest.mark.parametrize(
    "backend,name,expected",
    [
        ("cuda", "NVIDIA L20", "L20"),
        ("cuda", "NVIDIA A100-SXM4-80GB", "A100"),
        ("cuda", "NVIDIA H800 PCIe", "H800"),
        ("cuda", "NVIDIA H100 80GB HBM3", "H100"),
        ("cuda", "NVIDIA L4", "L4"),
        ("cuda", "NVIDIA B200", "B200"),
        ("cuda", "NVIDIA L40S", None),
        ("cuda", "NVIDIA A1000", None),
        ("npu", "Ascend910B2", "A2"),
        ("npu", "Ascend910_9391", "A3"),
        ("npu", "Ascend910C", "A3"),
        ("npu", "Ascend950PR_9599", "A5"),
        ("npu", "Ascend910", None),
    ],
)
def test_device_name_mapping(backend: str, name: str, expected: str | None) -> None:
    assert model_from_name(backend, name) == expected


@pytest.mark.parametrize("backend", list(SKUS))
def test_all_registered_targets_parse(backend: str) -> None:
    for sku in SKUS[backend]:
        assert parse_target(f"{backend}:{sku}") == Hardware(backend, (sku,))
    assert parse_target("host") == Hardware("host", ())


@pytest.mark.parametrize("target", ["cuda", "gpu:H100", "npu:H100", "cuda:A3", "cpu", ""])
def test_invalid_target_rejected(target: str) -> None:
    with pytest.raises(ValueError, match="Invalid test hardware"):
        parse_target(target)


@pytest.mark.parametrize(
    "names",
    [
        (),
        ("NVIDIA H100",),
        ("NVIDIA H800", "NVIDIA H800"),
        ("NVIDIA H100", "NVIDIA H800"),
        ("unknown", "unknown"),
    ],
)
def test_preflight_rejects_missing_insufficient_wrong_or_mixed_devices(
    names: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="Required cuda"):
        validate_inventory(Hardware("cuda", ("H100",), 2), DeviceInventory("cuda", names))


def test_preflight_accepts_visible_matching_devices() -> None:
    validate_inventory(
        Hardware("cuda", ("H100",), 2),
        DeviceInventory("cuda", ("NVIDIA H100", "NVIDIA H100")),
    )
    with pytest.raises(ValueError, match="Required npu"):
        validate_inventory(Hardware("npu", ("A3",)), DeviceInventory("cuda", ("NVIDIA H100",)))


def test_probe_uses_runtime_visible_indices_and_rejects_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    names = ("NVIDIA H100", "NVIDIA H100")
    device = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: len(names),
        get_device_name=names.__getitem__,
    )
    version = SimpleNamespace(hip=None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=device, version=version))
    assert probe_devices("cuda") == DeviceInventory("cuda", names)
    version.hip = "6.0"
    with pytest.raises(ValueError, match="ROCm"):
        probe_devices("cuda")
    version.hip = None
    device.is_available = lambda: False
    with pytest.raises(ValueError, match="No usable cuda"):
        probe_devices("cuda")


def test_npu_probe_loads_vendor_runtime_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    device = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda index: "Ascend910_9391",
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=device))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    assert probe_devices("npu") == DeviceInventory("npu", ("Ascend910_9391",))


@pytest.fixture
def hardware_suite(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from tests import conftest

    probes: list[str] = []

    def probe(backend: str) -> DeviceInventory:
        probes.append(backend)
        return DeviceInventory(backend, ("NVIDIA H100" if backend == "cuda" else "Ascend910C",))

    monkeypatch.setattr(conftest, "probe_devices", probe)
    pytester.makeconftest(
        "from tests.conftest import pytest_addoption, pytest_configure, "
        "pytest_collection_modifyitems, pytest_runtest_setup"
    )
    pytester.makepyfile("""
        import pytest

        @pytest.mark.cpu_test
        def test_host(): pass

        @pytest.mark.l2
        @pytest.mark.gpu_test
        def test_generic(): pass

        @pytest.mark.gpu_test
        @pytest.mark.skipif("False", reason="enabled")
        @pytest.mark.parametrize("model", [
            pytest.param("large", marks=[pytest.mark.H100, pytest.mark.cards_2]),
            pytest.param("small", marks=[pytest.mark.H100, pytest.mark.cards_1]),
            pytest.param("other", marks=pytest.mark.H800),
        ])
        def test_cuda(model): assert model in ("small", "other")

        @pytest.mark.npu_test
        @pytest.mark.A3
        def test_npu(): pass

        @pytest.mark.gpu_test
        @pytest.mark.H100
        @pytest.mark.cards_2
        @pytest.mark.parametrize("case", [
            pytest.param(0, marks=pytest.mark.skip(reason="disabled")),
            pytest.param(1, marks=pytest.mark.skipif("True", reason="conditional")),
            pytest.param(2, marks=pytest.mark.slow),
        ])
        def test_disabled(case): pytest.fail("requires two cards")
    """)
    return probes


def test_native_and_target_selection(pytester: pytest.Pytester, hardware_suite: list[str]) -> None:
    pytester.runpytest("-m", "gpu_test and H100 and cards_1").assert_outcomes(
        passed=1, deselected=8
    )
    pytester.runpytest("--test-hardware=npu:A3").assert_outcomes(passed=2, deselected=7)
    pytester.runpytest("--test-hardware=cuda:H100", "-m", "l2").assert_outcomes(
        passed=1, deselected=8
    )
    pytester.runpytest("--test-hardware=cuda:H100").assert_outcomes(
        passed=3, skipped=4, deselected=2
    )
    assert hardware_suite == ["cuda", "npu", "cuda", "cuda"]


def test_skip_and_wrong_runner(pytester: pytest.Pytester, hardware_suite: list[str]) -> None:
    target = "--test-hardware=cuda:H100"
    pytester.runpytest(target, "--collect-only").assert_outcomes(deselected=2)
    result = pytester.runpytest(target, "-k", "host or disabled", "-rs")
    result.assert_outcomes(passed=1, skipped=3, deselected=5)
    for reason in ("disabled", "conditional", "need --run-slow"):
        assert reason in result.stdout.str()
    assert hardware_suite == []
    result = pytester.runpytest(target, "-k", "disabled", "--run-slow", "-rs")
    result.assert_outcomes(skipped=3, deselected=6)
    assert "2 cuda cards, 1 visible" in result.stdout.str()
    mismatch = pytester.runpytest("--test-hardware=cuda:H800", "-k", "generic")
    assert mismatch.ret == pytest.ExitCode.USAGE_ERROR
