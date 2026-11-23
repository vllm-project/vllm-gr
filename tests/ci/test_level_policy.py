# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Hardware-free tests of the CI level policy and pytest selection hook."""

from __future__ import annotations

import json
import sys
from collections.abc import Generator
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import ModuleType

import pytest

from tests.helpers.paths import TESTS_DIR
from tests.level_policy import (
    ClassificationError,
    canonical_nodeid,
    normalize,
    resolve_level,
)

pytestmark = pytest.mark.cpu_test

pytest_plugins = ("pytester",)
INVENTORY = json.loads((TESTS_DIR / "level_inventory.json").read_text(encoding="utf-8"))["tests"]


@pytest.fixture(autouse=True)  # type: ignore[misc]
def reset_vocab_slice_config() -> Generator[None, None, None]:
    # These policy tests do not need to import the model/runtime stack.
    yield


def test_canonical_nodeid_keeps_class_and_removes_parameter_suffix() -> None:
    assert canonical_nodeid(r"tests\a.py::TestA::test_x[one-two]") == "tests/a.py::TestA::test_x"
    fake_dtype = type("dtype", (), {"__module__": "torch", "__str__": lambda _: "torch.float16"})()
    fake_device = type("device", (), {"__module__": "torch", "type": "cuda"})()
    assert normalize(fake_dtype) == "float16"
    assert normalize(fake_device) == "cuda"


def test_unknown_nodes_and_explicit_errors() -> None:
    assert resolve_level({}, "tests/new.py::test_x", {}, {}, strict=False) is None
    with pytest.raises(ClassificationError, match="Unclassified"):
        resolve_level({}, "tests/new.py::test_x", {}, {})
    policy = {
        "t.py::test_x": {
            "rules": [
                {"level": "l1", "when": {"ref": "params.n", "le": 2}},
                {"level": "l2", "when": {"ref": "params.n", "ge": 1}},
            ]
        }
    }
    with pytest.raises(ClassificationError, match="Overlapping"):
        resolve_level(policy, "t.py::test_x", {"n": 1}, {})
    with pytest.raises(ClassificationError, match="Missing parameter"):
        resolve_level(policy, "t.py::test_x", {}, {})


def test_cascade_gpu_specification_partitions_full_parameter_matrix() -> None:
    key = "tests/kernels/attention/test_cascade_beam_attn_gpu.py::test_beam_cascade"
    values = product(
        [1, 4],
        [16, 32],
        [(1, 8, 8), (1, 32, 8), (1, 64, 4), (1, 16, 4), (1, 16, 7), (1, 128, 1), (1, 32, 7)],
        [(4, 4), (8, 2), (16, 2)],
        [128, 256],
        ["float16", "bfloat16"],
        [16, 32],
        [None, 50],
        [2, 3],
    )
    names = (
        "batch_size",
        "beam_width",
        "seq_lens_and_common_prefix",
        "num_heads",
        "head_size",
        "dtype",
        "block_size",
        "soft_cap",
        "fa_version",
    )
    levels = [
        resolve_level(INVENTORY, key, dict(zip(names, values_row, strict=True)), {})
        for values_row in values
    ]
    assert len(levels) == 2688
    assert levels.count("l2") == 24
    assert levels.count("l4") == 2664
    baseline = dict(
        zip(names, (1, 16, (1, 32, 8), (4, 4), 128, "float16", 16, None, 2), strict=True)
    )
    assert resolve_level(INVENTORY, key, {**baseline, "dtype": "float64"}, {}) == "l4"


def test_npu_case_specification_partitions_full_parameter_matrix() -> None:
    @dataclass
    class Case:
        batch_size: int
        beam_width: int
        scenario: tuple[int, int, int]
        num_query_heads: int
        num_kv_heads: int
        head_size: int
        block_size: int
        dtype: str

    key = "tests/kernels/attention/test_cascade_beam_attn_npu.py::test_beam_cascade_npu"
    vals = product(
        [1, 4],
        [16, 32],
        [(1, 32, 8), (1, 64, 4), (1, 16, 4), (1, 16, 7), (1, 128, 1), (1, 32, 7)],
        [(4, 4), (8, 2), (16, 2)],
        [128],
        [32, 128],
        ["float16", "bfloat16"],
    )
    levels = []
    for batch, beam, scenario, (qh, kvh), head_size, block, dtype in vals:
        levels.append(
            resolve_level(
                INVENTORY,
                key,
                {"case": Case(batch, beam, scenario, qh, kvh, head_size, block, dtype)},
                {},
            )
        )
    assert len(levels) == 288
    assert levels.count("l2") == 12
    assert levels.count("l4") == 276
    assert (
        resolve_level(
            INVENTORY, key, {"case": Case(1, 16, (1, 32, 8), 4, 4, 128, 128, "float32")}, {}
        )
        == "l4"
    )


def test_cpu_npu_and_gpu_parameters_have_exclusive_levels() -> None:
    key = "tests/v1/worker/test_npu_model_runner_patch.py::test_npu_single_step_filters_candidates_before_return_limit"
    for device, batch, width, case in product(["cpu", "npu"], [1, 4], [1, 2], ["nan", "finite"]):
        expected = "l1" if device == "cpu" else "l2" if batch == width == 1 else "l4"
        assert (
            resolve_level(
                INVENTORY,
                key,
                {"device": device, "batch_size": batch, "result_width": width, "case": case},
                {},
            )
            == expected
        )
    assert (
        resolve_level(
            INVENTORY,
            key,
            {"device": "npu", "batch_size": 1, "result_width": 1, "case": "future"},
            {},
        )
        == "l4"
    )
    key = "tests/ops/test_sliced_logits.py::test_native_logits_forward"
    common = {
        "device": "cuda",
        "dtype": "float16",
        "head_dtype": None,
        "skip_gather": False,
        "slice_mode": "global",
    }
    assert resolve_level(INVENTORY, key, common, {}) == "l2"
    assert resolve_level(INVENTORY, key, {**common, "skip_gather": True}, {}) == "l4"
    assert resolve_level(INVENTORY, key, {**common, "device": "cpu"}, {}) == "l1"
    assert resolve_level(INVENTORY, key, {**common, "dtype": "float64"}, {}) == "l4"
    assert resolve_level(INVENTORY, key, {**common, "slice_mode": "future"}, {}) == "l4"
    key = "tests/kernels/test_beam_attn_npu_xllm_parity.py::test_cascade_execution_batch_matches_individual_requests"
    assert (
        resolve_level(
            INVENTORY, key, {"batch_size": 1, "decode_step": 0, "num_query_heads": 32}, {}
        )
        == "l2"
    )
    key = (
        "tests/kernels/test_rec_constrained_topk_cuda.py::test_cuda_supported_specialization_matrix"
    )
    assert resolve_level(INVENTORY, key, {"dtype": "float32", "step": 0, "top_k": 8}, {}) == "l4"


def test_runtime_option_and_default_device_are_classified() -> None:
    key = "tests/entrypoints/openai/test_serving_beam_search.py::test_serving_beam_search"
    assert resolve_level(INVENTORY, key, {}, {"loops": 1}) == "l3"
    assert resolve_level(INVENTORY, key, {}, {"loops": 2}) == "l4"
    key = "tests/constraints/test_constraint_table_execution.py::test_npu_default_device_falls_back_without_crash"
    assert resolve_level(INVENTORY, key, {"device_class": "cpu"}, {}) == "l1"
    assert resolve_level(INVENTORY, key, {"device_class": "accelerator"}, {}) == "l2"


def _make_sample(pytester: pytest.Pytester) -> Path:
    pytester.makeconftest('pytest_plugins = ("tests.level_plugin",)')
    pytester.makepyfile(
        test_sample="""
import pytest
@pytest.mark.parametrize('n', [1, 2, 3])
def test_grid(n):
    assert n > 0
"""
    )
    manifest = pytester.path / "levels.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "tests": {
                    "test_sample.py::test_grid": {
                        "rules": [{"level": "l2", "when": {"ref": "params.n", "eq": 1}}],
                        "default": "l4",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return Path(manifest)


def test_hook_assigns_markers_before_pytest_m_filter(pytester: pytest.Pytester) -> None:
    manifest = _make_sample(pytester)
    result = pytester.runpytest(
        "--strict-test-levels", "--test-level-manifest", str(manifest), "-m", "l2"
    )
    result.assert_outcomes(passed=1, deselected=2)
    result = pytester.runpytest(
        "--strict-test-levels", "--test-level-manifest", str(manifest), "-m", "l4"
    )
    result.assert_outcomes(passed=2, deselected=1)


def test_strict_hook_rejects_unclassified_and_normal_pytest_allows_it(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest('pytest_plugins = ("tests.level_plugin",)')
    pytester.makepyfile(test_sample="def test_unclassified(): pass")
    manifest = pytester.path / "levels.json"
    manifest.write_text('{"version": 1, "tests": {}}', encoding="utf-8")
    result = pytester.runpytest(
        "--strict-test-levels", "--test-level-manifest", str(manifest), "-m", "l1"
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*Unclassified test test_sample.py::test_unclassified*"])
    for hint in (
        "assign exactly one @pytest.mark.l1 through",
        "tests/level_inventory.json",
        "docs/contributing/ci/test_levels.md#test-levels-and-ci-selection",
    ):
        assert hint in result.stderr.str()
    result = pytester.runpytest("--test-level-manifest", str(manifest))
    result.assert_outcomes(passed=1)


def test_strict_hook_accepts_explicit_level_and_rejects_conflict(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest('pytest_plugins = ("tests.level_plugin",)')
    pytester.makepyfile(
        test_sample="""
import pytest
@pytest.mark.l1
def test_explicit():
    pass
"""
    )
    manifest = pytester.path / "levels.json"
    manifest.write_text('{"version": 1, "tests": {}}', encoding="utf-8")
    result = pytester.runpytest(
        "--strict-markers",
        "--strict-test-levels",
        "--test-level-manifest",
        str(manifest),
        "-m",
        "l1",
    )
    result.assert_outcomes(passed=1)
    manifest.write_text(
        json.dumps({"version": 1, "tests": {"test_sample.py::test_explicit": "l2"}}),
        encoding="utf-8",
    )
    result = pytester.runpytest(
        "--strict-markers",
        "--strict-test-levels",
        "--test-level-manifest",
        str(manifest),
        "-m",
        "l1",
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*Explicit levels*conflict with l2*"])


@pytest.mark.l1  # type: ignore[misc]
def test_conflicting_explicit_levels_fail_even_without_inventory(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest('pytest_plugins = ("tests.level_plugin",)')
    pytester.makepyfile(
        test_sample="""
import pytest
@pytest.mark.l1
@pytest.mark.l2
def test_conflict():
    pass
"""
    )
    manifest = pytester.path / "levels.json"
    manifest.write_text('{"version": 1, "tests": {}}', encoding="utf-8")
    for strict in ([], ["--strict-test-levels"]):
        result = pytester.runpytest("--test-level-manifest", str(manifest), *strict)
        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(["*Conflicting explicit levels*test_sample.py::test_conflict*"])


def test_default_device_collection_does_not_probe_runtime(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_execution = ModuleType("vllm_gr.constraints.execution")

    def forbidden_probe(*args: object, **kwargs: object) -> None:
        raise AssertionError("collection tried to initialize a device")

    fake_execution._default_device = forbidden_probe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm_gr.constraints.execution", fake_execution)
    pytester.makeconftest('pytest_plugins = ("tests.level_plugin",)')
    pytester.makepyfile(
        test_default_device="""
import pytest
@pytest.mark.parametrize('device_class', ['cpu', 'accelerator'])
def test_default(device_class):
    raise AssertionError('collect-only must not execute the test')
"""
    )
    original = "tests/constraints/test_constraint_table_execution.py::test_npu_default_device_falls_back_without_crash"
    manifest = pytester.path / "levels.json"
    manifest.write_text(
        json.dumps(
            {"version": 1, "tests": {"test_default_device.py::test_default": INVENTORY[original]}}
        ),
        encoding="utf-8",
    )
    result = pytester.runpytest(
        "--collect-only", "--strict-test-levels", "--test-level-manifest", str(manifest), "-m", "l1"
    )
    assert result.ret == pytest.ExitCode.OK
    result.stdout.fnmatch_lines(["*test_default*cpu*", "*1/2 tests collected*1 deselected*"])
