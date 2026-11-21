# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Host policy tests: selection, legacy compatibility, and preflight failures."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from types import SimpleNamespace

import pytest

from tests.helpers.hardware import (
    SKUS,
    DeviceInventory,
    Hardware,
    hardware_marks,
    hardware_test,
    model_from_name,
    parse_target,
    probe_devices,
    validate_inventory,
)

pytestmark = [pytest.mark.host, pytest.mark.l1]
pytest_plugins = ("pytester",)


@pytest.fixture(autouse=True)
def isolate_legacy_target(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nested pytest sessions simulate several platforms on any CI runner.
    # Keep the runner's target out of those sessions unless a test sets it
    # explicitly; monkeypatch restores the outer environment after each test.
    monkeypatch.delenv("VLLM_GR_TARGET_DEVICE", raising=False)


@pytest.fixture(autouse=True)
def reset_vocab_slice_config() -> Generator[None, None, None]:
    # Hardware policy has no model/runtime dependency, like test_level_policy.
    yield


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
    "res,cards",
    [
        ({}, 1),
        ({"gpu": "H100"}, 1),
        ({"cuda": "A3"}, 1),
        ({"npu": []}, 1),
        ({"cuda": "H100"}, True),
        ({"cuda": "H100"}, 0),
        ({"cuda": "H100"}, 9),
        ({"cuda": "H100"}, {"npu": 1}),
    ],
)
def test_invalid_declaration_rejected(
    res: Mapping[str, str | Sequence[str]], cards: int | Mapping[str, int]
) -> None:
    with pytest.raises(ValueError):
        hardware_test(res=res, num_cards=cards)


def test_param_marks_reject_mixed_platform_counts() -> None:
    with pytest.raises(ValueError, match="Different platform card counts"):
        hardware_marks(res={"cuda": "H100", "npu": "A3"}, num_cards={"cuda": 4, "npu": 2})


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


def _sample(pytester: pytest.Pytester, *, names: tuple[str, ...] = ("NVIDIA H100",) * 4) -> None:
    pytester.makeconftest(
        'pytest_plugins = ("tests.hardware_plugin",)\n'
        "import pytest\n"
        "from tests import hardware_plugin\n"
        "from tests.helpers.hardware import DeviceInventory\n"
        "patcher = pytest.MonkeyPatch()\n"
        "def pytest_configure(config):\n"
        f"    patcher.setattr(hardware_plugin, 'probe_devices', "
        f"lambda backend: DeviceInventory(backend, {names!r}))\n"
        "def pytest_unconfigure(config):\n"
        "    patcher.undo()\n"
    )
    pytester.makepyfile(
        test_sample="""
import pytest
from tests.helpers.hardware import hardware_test, hardware_marks

@pytest.mark.cpu_test
def test_old_host(): pass

@pytest.mark.host
def test_new_host(): pass

@pytest.mark.gpu_test
def test_old_cuda(): pass

@pytest.mark.npu_test
def test_old_npu(): pass

@pytest.mark.cpu_test
@hardware_test(res={'cuda': ['H100', 'H800'], 'npu': 'A3'},
               num_cards={'cuda': 4, 'npu': 2})
def test_multi_platform(): pass

@pytest.mark.parametrize('value', [pytest.param(1, marks=hardware_marks(res={'cuda': 'L4'}))])
def test_param(value): pass
"""
    )


def test_legacy_filters_see_new_tests_and_platform_counts_stay_separate(
    pytester: pytest.Pytester,
) -> None:
    _sample(pytester)
    result = pytester.runpytest("--collect-only", "-q", "-m", "gpu_test")
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*test_multi_platform*cuda*", "*test_param*"])
    assert "test_multi_platform[npu]" not in result.stdout.str()
    result = pytester.runpytest("--collect-only", "-q", "-m", "npu and A3 and cards_4")
    assert result.ret == 5
    result = pytester.runpytest("--collect-only", "-q", "-m", "npu and A3 and cards_2")
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*test_multi_platform*npu*"])


@pytest.mark.parametrize(
    "target,passed,deselected",
    [
        ("host", 2, 5),
        ("cuda:H100", 4, 3),
        ("cuda:H800", 4, 3),
        ("cuda:L4", 4, 3),
        ("cuda:L20", 3, 4),
        ("cuda:A100", 3, 4),
        ("cuda:B200", 3, 4),
        ("npu:A2", 3, 4),
        ("npu:A3", 4, 3),
        ("npu:A5", 3, 4),
    ],
)
def test_target_selects_host_legacy_and_matching_model(
    pytester: pytest.Pytester, target: str, passed: int, deselected: int
) -> None:
    sku = target.partition(":")[2] or "H100"
    name = {"A2": "Ascend910B2", "A3": "Ascend910_9391", "A5": "Ascend950"}.get(
        sku, f"NVIDIA {sku}"
    )
    _sample(pytester, names=(name,) * 4)
    result = pytester.runpytest("--test-hardware", target)
    result.assert_outcomes(passed=passed, deselected=deselected)


@pytest.mark.parametrize("selection", ["", "old_host"])
def test_preflight_fails_before_test_body_on_wrong_machine(
    pytester: pytest.Pytester, selection: str
) -> None:
    _sample(pytester, names=("NVIDIA H800",) * 4)
    result = pytester.runpytest("--test-hardware=cuda:H100", "-k", selection)
    assert result.ret == 4
    assert "Hardware preflight failed" in result.stderr.str()
    assert "H800" in result.stderr.str()


def test_preflight_rejects_card_shortage(pytester: pytest.Pytester) -> None:
    _sample(pytester, names=("NVIDIA H100",))
    result = pytester.runpytest("--test-hardware=cuda:H100")
    assert result.ret == 4
    assert "at least 4" in result.stderr.str()


def test_collection_only_never_probes_hardware(pytester: pytest.Pytester) -> None:
    _sample(pytester, names=())
    result = pytester.runpytest("--collect-only", "--test-hardware=npu:A3")
    assert result.ret == 0
    assert "Hardware preflight failed" not in result.stderr.str()


def test_new_declaration_checks_hardware_without_explicit_target(pytester: pytest.Pytester) -> None:
    _sample(pytester, names=("NVIDIA H800",))
    result = pytester.runpytest("-k", "multi_platform and cuda")
    result.assert_outcomes(errors=1, deselected=6)
    assert "Hardware preflight failed" in result.stdout.str()


@pytest.mark.parametrize("value", ["", "npu:H100"])
def test_invalid_target_fails_collection(pytester: pytest.Pytester, value: str) -> None:
    _sample(pytester)
    result = pytester.runpytest("--collect-only", f"--test-hardware={value}")
    assert result.ret == 4


@pytest.mark.parametrize(
    "legacy_target,target",
    [("cpu", "cuda:H100"), ("cuda", "npu:A3"), ("npu", "host"), ("npu", "cuda:H100")],
)
def test_conflicting_legacy_target_rejected(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    legacy_target: str,
    target: str,
) -> None:
    _sample(pytester)
    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", legacy_target)
    result = pytester.runpytest("--collect-only", "--test-hardware", target)
    assert result.ret == 4
    assert "conflicts with VLLM_GR_TARGET_DEVICE" in result.stderr.str()


@pytest.mark.parametrize(
    "legacy_target,target,name,passed,deselected",
    [
        ("cpu", "host", "unused", 2, 5),
        ("cuda", "cuda:H100", "NVIDIA H100", 4, 3),
        ("npu", "npu:A3", "Ascend910_9391", 4, 3),
    ],
)
def test_matching_legacy_target_accepted(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    legacy_target: str,
    target: str,
    name: str,
    passed: int,
    deselected: int,
) -> None:
    _sample(pytester, names=(name,) * 4)
    monkeypatch.setenv("VLLM_GR_TARGET_DEVICE", legacy_target)
    result = pytester.runpytest("--test-hardware", target)
    result.assert_outcomes(passed=passed, deselected=deselected)


def test_parameter_ids_are_not_hardware_markers(pytester: pytest.Pytester) -> None:
    pytester.makeconftest('pytest_plugins = ("tests.hardware_plugin",)')
    pytester.makepyfile(
        "import pytest\n@pytest.mark.host\n"
        "@pytest.mark.parametrize('name', ['cuda', 'npu', 'H100', 'cards_2'])\n"
        "def test_host(name): pass\n"
    )
    result = pytester.runpytest("--test-hardware=host")
    result.assert_outcomes(passed=4)


@pytest.mark.parametrize("directory", ["kernels", "st"])
def test_directory_defaults_preserve_modern_platforms(
    pytester: pytest.Pytester, directory: str
) -> None:
    from tests.helpers.paths import TESTS_DIR

    pytester.makeconftest((TESTS_DIR / "conftest.py").read_text(encoding="utf-8"))
    folder = pytester.path / directory
    folder.mkdir()
    (folder / "conftest.py").write_text(
        (TESTS_DIR / directory / "conftest.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (folder / "test_sample.py").write_text(
        "import pytest\n@pytest.mark.host\ndef test_host(): pass\n"
        "@pytest.mark.npu\ndef test_npu(): pass\n"
        "def test_default(): pass\n",
        encoding="utf-8",
    )
    result = pytester.runpytest(
        "--collect-only", "-q", "-m", "cpu_test and not gpu_test and not npu_test"
    )
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*test_sample.py::test_host*"])
    assert "::test_npu" not in result.stdout.str()
    assert "::test_default" not in result.stdout.str()
    result = pytester.runpytest("--collect-only", "-q", "-m", "cuda")
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*test_sample.py::test_default*"])
    assert "::test_host" not in result.stdout.str()
    assert "::test_npu" not in result.stdout.str()


@pytest.mark.parametrize(
    "marks,message",
    [
        ("@pytest.mark.H100", "SKU marker requires platform cuda"),
        (
            "@pytest.mark.cuda\n@pytest.mark.cards_1\n@pytest.mark.cards_2",
            "conflicting card counts",
        ),
        ("@pytest.mark.host\n@pytest.mark.cards_2", "card count requires an accelerator"),
    ],
)
def test_invalid_marker_combinations_rejected(
    pytester: pytest.Pytester, marks: str, message: str
) -> None:
    pytester.makeconftest('pytest_plugins = ("tests.hardware_plugin",)')
    pytester.makepyfile(f"import pytest\n{marks}\ndef test_bad(): pass\n")
    result = pytester.runpytest("--collect-only")
    assert result.ret == 4
    assert message in result.stderr.str()
