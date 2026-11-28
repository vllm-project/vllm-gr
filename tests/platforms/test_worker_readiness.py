# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Check actual loaded workers before EngineCore accepts GR requests."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.config import GRAttentionBackend, GRConfig
from vllm_gr.v1.engine import worker_readiness as readiness

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def config() -> Any:
    value = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(max_model_len=4096),
        parallel_config=SimpleNamespace(
            world_size=1, worker_cls="types.SimpleNamespace", distributed_executor_backend="mp"
        ),
        attention_config=SimpleNamespace(backend_per_kind={}),
    )
    GRConfig(
        beam_max_width=16, beam_graph_enabled=False, attention_backend=GRAttentionBackend.CUSTOM
    ).write_to_vllm_config(value)
    return value


def worker(value: Any) -> Any:
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            vllm_config=value,
            _gr_ready=True,
            _gr_loaded_config=GRConfig.from_vllm_config(value),
            _gr_attention_execution=SimpleNamespace(
                __name__="vendor.attention", DECODE_MODE="cascade"
            ),
            gr_graph_status=lambda: {"captured": [], "fallback": {}},
        )
    )


def test_reports_native_autofit_and_loaded_configuration() -> None:
    host = config()
    local = deepcopy(host)
    local.model_config.max_model_len = 2048
    report = readiness.worker_backend_readiness(worker(local))
    readiness.validate_worker_readiness(host, [report], expected_workers=1)
    assert report["max_model_len"] == 2048
    assert report["runner"] == "types.SimpleNamespace"
    assert report["attention_execution"] == "vendor.attention"
    assert report["versions"] == readiness._runtime_versions()


@pytest.mark.parametrize(  # type: ignore[misc]
    "change,message",
    [
        ({"ready": False}, "not ready"),
        ({"platform": "different.Platform"}, "different platform"),
        ({"versions": {}}, "different runtime or plugin versions"),
        ({"worker_classes": ["different.Worker"]}, "configured Worker"),
        ({"gr_config": {}}, "different GR configuration"),
        ({"max_model_len": 4097}, "max_model_len"),
        ({"max_model_len": True}, "max_model_len"),
        ({"attention_execution": None}, "attention implementation"),
        ({"graph": None}, "graph execution status"),
        ({"graph": {"captured": [[1, 16, 1024]], "fallback": {}}}, "eager configuration"),
    ],
)
def test_incompatible_worker_is_rejected(change: Any, message: Any) -> None:
    value = config()
    report = readiness.worker_backend_readiness(worker(value))
    with pytest.raises(RuntimeError, match=message):
        readiness.validate_worker_readiness(value, [report | change], expected_workers=1)


def test_every_rank_must_report_same_initialized_implementation() -> None:
    value = config()
    report = readiness.worker_backend_readiness(worker(value))
    with pytest.raises(RuntimeError, match="requires 2 worker reports"):
        readiness.validate_worker_readiness(value, [report], expected_workers=2)
    with pytest.raises(RuntimeError, match="workers disagree"):
        readiness.validate_worker_readiness(
            value, [report, report | {"runner": "other.Runner"}], expected_workers=2
        )


def test_explicit_decode_and_per_kind_attention_are_honored() -> None:
    from dataclasses import replace

    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    value = config()
    gr = GRConfig.from_vllm_config(value)
    replace(gr, npu_decode_execution="xllm").write_to_vllm_config(value)
    value.attention_config.backend_per_kind = {"full_attention": AttentionBackendEnum.CUSTOM}
    report = readiness.worker_backend_readiness(worker(value))
    with pytest.raises(RuntimeError, match="requested decode mode"):
        readiness.validate_worker_readiness(value, [report], expected_workers=1)
    report["decode_mode"] = "xllm"
    report["attention_by_kind"] = {"full_attention": ["FLASH_ATTN"]}
    with pytest.raises(RuntimeError, match="explicit full_attention"):
        readiness.validate_worker_readiness(value, [report], expected_workers=1)
    report["attention_by_kind"] = {"full_attention": ["CUSTOM"]}
    readiness.validate_worker_readiness(value, [report], expected_workers=1)


def test_uninitialized_or_modified_worker_cannot_report_ready() -> None:
    value = config()
    instance = worker(value)
    instance.model_runner._gr_ready = False
    with pytest.raises(RuntimeError, match="initialize its GR ModelRunner"):
        readiness.worker_backend_readiness(instance)
    instance.model_runner._gr_ready = True
    GRConfig(beam_max_width=32).write_to_vllm_config(value)
    with pytest.raises(RuntimeError, match="changed after model initialization"):
        readiness.worker_backend_readiness(instance)


def test_failed_engine_gate_shuts_down_executor() -> None:
    value = config()
    executor = SimpleNamespace(collective_rpc=Mock(return_value=[]), shutdown=Mock())
    engine = SimpleNamespace(vllm_config=value, model_executor=executor)
    with pytest.raises(RuntimeError, match="worker reports"):
        readiness.verify_engine_backend_readiness(engine)
    executor.shutdown.assert_called_once_with()
    assert not hasattr(engine, "_gr_validate_platform_request")


@pytest.mark.parametrize("fails", [False, True])  # type: ignore[misc]
def test_external_launcher_collects_all_ranks_even_after_local_failure(
    monkeypatch: pytest.MonkeyPatch, fails: Any
) -> None:
    import torch.distributed as dist
    from vllm.distributed import parallel_state

    group = object()
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: SimpleNamespace(cpu_group=group))
    monkeypatch.setattr(dist, "get_world_size", lambda actual: 2)
    remote = {"ready": True, "rank": 1}

    def gather(output: Any, local: Any, group: Any) -> Any:
        output[:] = [local, remote]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    rpc = (
        Mock(side_effect=RuntimeError("failed load"))
        if fails
        else Mock(return_value=[{"ready": True}])
    )
    reports = readiness._external_launcher_reports(SimpleNamespace(collective_rpc=rpc), 2)
    assert reports[0]["ready"] is (not fails)
    assert reports[1] is remote
    if fails:
        assert "failed load" in reports[0]["error"]


def test_constraint_artifact_identity_must_match() -> None:
    from tests.helpers.gr_config import table_constraint

    value = config()
    GRConfig(beam_max_width=16, constraint=table_constraint()).write_to_vllm_config(value)
    instance = worker(value)
    report = readiness.worker_backend_readiness(instance)
    with pytest.raises(RuntimeError, match="different constraint artifact"):
        readiness.validate_worker_readiness(value, [report], expected_workers=1)
    report["constraint_digest"] = table_constraint().table.artifact_digest
    readiness.validate_worker_readiness(value, [report], expected_workers=1)


def test_remote_limits_do_not_invoke_another_platform_on_same_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import asdict

    import vllm_gr.platforms
    from tests.helpers.ascend_platform import AscendGRPlatform
    from vllm_gr.platforms import beam_requests

    native = config()
    native.scheduler_config = SimpleNamespace(max_num_batched_tokens=128)
    limits = asdict(AscendGRPlatform().get_beam_limits(native))
    limits["platform_class"] = "vendor.platform.AnotherNPUPlatform"
    native.additional_config["vllm_gr_limits"] = limits
    builtin = AscendGRPlatform()
    builtin.validate_gr_request = Mock(side_effect=AssertionError("foreign policy was applied"))
    monkeypatch.setattr(vllm_gr.platforms, "get_current_gr_platform", lambda **kw: builtin)
    beam_requests.validate_gr_device_config(native)
    builtin.validate_gr_request.assert_not_called()


def test_native_executor_rpc_reports_the_loaded_worker_class() -> None:
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    value = config()
    wrapper = WorkerWrapperBase()
    wrapper.worker = worker(value)
    executor = object.__new__(UniProcExecutor)
    executor.driver_worker = wrapper
    reports = executor.collective_rpc(readiness.worker_backend_readiness)
    assert reports[0]["worker_classes"] == ["types.SimpleNamespace", "builtins.object"]
    readiness.validate_worker_readiness(value, reports, expected_workers=1)
