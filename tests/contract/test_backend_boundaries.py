# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
from pathlib import Path

import pytest

from tools.pre_commit import check_backend_boundaries as guard
from tools.pre_commit.check_backend_boundaries import findings, main

pytestmark = [pytest.mark.l1, pytest.mark.cpu_test]


@pytest.mark.parametrize(  # type: ignore[misc]
    "code",
    [
        "from vllm.platforms import current_platform as platform\nif platform.device_type == 'privateuseone': run()",
        "def dispatch(device):\n kind = str(device).split(':')[0]\n return lookup() if kind in ('newchip', 'cuda') else other()",
        "from importlib import import_module as load\nload('vllm_ascend.worker')",
        "from torch import cuda as accelerator",
        "import vllm_gr.v1.worker.cudagraph_capture",
        "from vllm_gr.v1.worker import cudagraph_capture",
        "from importlib import import_module as load\nload('private.cudagraphs')",
        "kind: str = tensor.device.type\nif kind == 'future': run()",
        "if current_platform.is_cuda_alike(): install()",
        "match runner.device.type:\n case 'future': run()",
        "adapters = {'future': FastPath}\ndef select(device):\n return adapters.get(device.type)",
        "class ConstraintTableBackendEnum(Enum):\n FUTURE = 'sample'",
        "class ConstraintBackendKind(str, Enum):\n FUTURE: str = 'future'",
    ],
)
def test_rejects_shared_hardware_selection_or_vendor_imports(code: str) -> None:
    assert findings(code)


def test_diagnostics_identity_bridging_and_typed_dependencies_remain_legal() -> None:
    assert not findings(
        "def describe(device):\n logger.info('selected %s', device.type)\n return device.type\n"
        "if provider.backend_id != plan.backend_id: raise ValueError('mismatch')\n"
        "if TYPE_CHECKING:\n from torch_npu import Tensor\n"
        "from vllm_gr.v1.worker.graph_capture import is_recoverable_capture_error"
    )


def test_scans_new_shared_modules_and_skips_only_hardware_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(guard, "SHARED", ("constraints/execution.py",))
    shared = [
        "config/parser.py",
        "config/policy/new.py",
        "engine/startup.py",
        "entrypoints/beam_search_v1/request_params.py",
        "entrypoints/grpc/serving.py",
        "entrypoints/openai/completion_batch_patch.py",
        "v1/engine/engine_core_patch.py",
        "ops/shared/new.py",
        "ops/cuda_new.py",
        "v1/beam/new/shared.py",
        "platforms/legacy.py",
        "platforms/future.py",
        "constraints/execution.py",
    ]
    owned = ["ops/cuda/new.py", "ops/npu/new.py", "platforms/ascend/provider.py"]
    for path in [*shared, *owned]:
        source = tmp_path / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("if tensor.device.type == 'future': run()", encoding="utf-8")
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text("[]", encoding="utf-8")
    assert main(tmp_path, allowlist) == 1
    output = capsys.readouterr().out
    assert all(f"unreviewed: {path}:" in output for path in shared)
    assert all(path not in output for path in owned)


def test_exemption_matches_only_reviewed_occurrences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(guard, "SHARED", ())
    source = tmp_path / "ops" / "oracle.py"
    source.parent.mkdir()
    clause = "if tensor.device.type == 'npu': run()\n"
    expression = "tensor.device.type == 'npu'"
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps([{"file": "ops/oracle.py", "scope": "<module>", "expression": expression}]),
        encoding="utf-8",
    )
    source.write_text(clause, encoding="utf-8")
    assert main(tmp_path, allowlist) == 0
    source.write_text(clause * 2, encoding="utf-8")
    assert main(tmp_path, allowlist) == 1
    assert f"unreviewed: ops/oracle.py:<module>: {expression} (1)" in capsys.readouterr().out
    source.write_text(clause.replace("npu", "future"), encoding="utf-8")
    assert main(tmp_path, allowlist) == 1
    output = capsys.readouterr().out
    assert "unreviewed: ops/oracle.py:<module>: tensor.device.type == 'future'" in output
    assert f"stale exemption: ops/oracle.py:<module>: {expression}" in output


def test_backend_helpers_live_with_their_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(guard, "SHARED", ())
    provider = tmp_path / "platforms" / "example" / "provider.py"
    provider.parent.mkdir(parents=True)
    provider.touch()
    helper = provider.parents[1] / "example_config.py"
    helper.touch()
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text("[]", encoding="utf-8")
    assert main(tmp_path, allowlist) == 1
    assert "platforms/example_config.py:<module>: flat backend module" in capsys.readouterr().out
    helper.rename(provider.with_name("config.py"))
    assert main(tmp_path, allowlist) == 0


def test_production_boundaries_match_reviewed_exemptions() -> None:
    assert main() == 0
