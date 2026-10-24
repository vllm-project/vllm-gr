# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"
"""Regression coverage for the native Ascend worker memory sampler."""

from __future__ import annotations

import subprocess
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from vllm_gr.v1.worker import npu_memory_counters


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> Any:
    state = SimpleNamespace(
        stats={
            "reserved_bytes": {"all": {"current": 3 << 30}},
            "allocated_bytes": {"all": {"current": 1 << 30}},
        },
        original_calls=0,
    )
    native_logger = mock.Mock()
    namespace: dict[str, Any] = {
        "__name__": "vllm_ascend.worker.worker",
        "logger": native_logger,
        "state": state,
    }
    exec(
        "def profile_memory(self):\n"
        "    state.original_calls += 1\n"
        "    self.torch_reserved = 17\n"
        "    self.torch_allocated = 11\n",
        namespace,
    )
    original = namespace["profile_memory"]
    worker_class = type("NPUWorker", (), {"profile_memory": original})
    memory_module = ModuleType("torch_npu.npu.memory")
    snapshot = mock.Mock(side_effect=lambda: state.stats)
    setattr(memory_module, "memory_stats_as_nested_dict", snapshot)
    monkeypatch.setitem(sys.modules, "torch_npu.npu.memory", memory_module)
    return SimpleNamespace(
        state=state,
        logger=native_logger,
        namespace=namespace,
        worker_class=worker_class,
        original=original,
        memory_module=memory_module,
        snapshot=snapshot,
    )


@pytest.mark.parametrize("module", ["vllm_ascend.worker.worker", "vllm_ascend.worker.worker_v1"])
def test_one_snapshot_preserves_counters_and_original_logger(context: Any, module: str) -> None:
    ctx = context
    ctx.original.__module__ = module
    assert npu_memory_counters.install_npu_memory_counter_patch(ctx.worker_class)
    worker = ctx.worker_class()
    worker.profile_memory()

    assert worker.torch_reserved == 3 << 30
    assert worker.torch_allocated == 1 << 30
    ctx.snapshot.assert_called_once_with()
    assert ctx.state.original_calls == 0
    ctx.logger.debug.assert_called_once_with(
        "torch reserved memory: %.2f GiB, torch allocated memory: %.2f GiB",
        3.0,
        1.0,
    )


@pytest.mark.parametrize(
    "stats,reserved,allocated",
    [
        ({}, 0, 0),
        ({"reserved_bytes": {"all": {"current": 31}}}, 31, 0),
        ({"allocated_bytes": {"all": {"current": 19}}}, 0, 19),
        ({"reserved_bytes": {"large_pool": {"current": 31}}}, 0, 0),
    ],
)
def test_uninitialized_and_missing_counters_match_native_zero_defaults(
    context: Any, stats: Any, reserved: int, allocated: int
) -> None:
    ctx = context
    ctx.state.stats = stats
    assert npu_memory_counters.install_npu_memory_counter_patch(ctx.worker_class)
    worker = ctx.worker_class()
    worker.profile_memory()
    assert (worker.torch_reserved, worker.torch_allocated) == (reserved, allocated)
    assert ctx.state.original_calls == 0
    ctx.snapshot.assert_called_once_with()


def test_each_call_refreshes_fields_and_installation_is_idempotent(context: Any) -> None:
    ctx = context
    assert npu_memory_counters.install_npu_memory_counter_patch(ctx.worker_class)
    wrapper = ctx.worker_class.profile_memory
    assert wrapper.__wrapped__ is ctx.original
    assert getattr(wrapper, npu_memory_counters._MEMORY_COUNTER_PATCH_MARKER)
    assert not npu_memory_counters.install_npu_memory_counter_patch(ctx.worker_class)
    assert ctx.worker_class.profile_memory is wrapper
    worker = ctx.worker_class()
    worker.profile_memory()
    ctx.state.stats = {
        "reserved_bytes": {"all": {"current": 29}},
        "allocated_bytes": {"all": {"current": 23}},
    }
    worker.profile_memory()
    assert (worker.torch_reserved, worker.torch_allocated) == (29, 23)
    assert ctx.snapshot.call_count == 2


@pytest.mark.parametrize("api", [None, "missing"])
def test_missing_nested_api_keeps_original_method(context: Any, api: Any) -> None:
    ctx = context
    if api == "missing":
        delattr(ctx.memory_module, "memory_stats_as_nested_dict")
    else:
        setattr(ctx.memory_module, "memory_stats_as_nested_dict", api)
    assert not npu_memory_counters.install_npu_memory_counter_patch(ctx.worker_class)
    assert ctx.worker_class.profile_memory is ctx.original
    ctx.worker_class().profile_memory()
    assert ctx.state.original_calls == 1


@pytest.mark.parametrize("variant", ["custom", "wrapper", "signature", "logger", "method"])
def test_unknown_worker_profiles_keep_original_semantics(context: Any, variant: str) -> None:
    ctx = context
    original = ctx.original
    if variant == "custom":
        original.__module__ = "custom_worker"
    elif variant == "wrapper":

        @wraps(original)
        def wrapped(self: Any) -> None:
            original(self)

        ctx.worker_class.profile_memory = wrapped
    elif variant == "signature":
        exec("def profile_memory(self, device=None): pass", ctx.namespace)
        ctx.worker_class.profile_memory = ctx.namespace["profile_memory"]
    elif variant == "logger":
        ctx.namespace.pop("logger")
    else:
        del ctx.worker_class.profile_memory
    before = getattr(ctx.worker_class, "profile_memory", None)
    assert not npu_memory_counters.install_npu_memory_counter_patch(ctx.worker_class)
    assert getattr(ctx.worker_class, "profile_memory", None) is before
    ctx.snapshot.assert_not_called()


@pytest.mark.parametrize(
    "stats",
    [
        [],
        {"reserved_bytes": None},
        {"allocated_bytes": 12},
        {"allocated_bytes": {"all": {"current": "changed"}}},
    ],
)
def test_changed_nested_schema_falls_back_to_native(context: Any, stats: Any) -> None:
    ctx = context
    ctx.state.stats = stats
    assert npu_memory_counters.install_npu_memory_counter_patch(ctx.worker_class)
    worker = ctx.worker_class()
    worker.profile_memory()
    assert (worker.torch_reserved, worker.torch_allocated) == (17, 11)
    assert ctx.state.original_calls == 1
    ctx.logger.debug.assert_not_called()


def test_allocator_errors_are_propagated(context: Any) -> None:
    ctx = context
    ctx.snapshot.side_effect = RuntimeError("allocator failed")
    assert npu_memory_counters.install_npu_memory_counter_patch(ctx.worker_class)
    with pytest.raises(RuntimeError, match="allocator failed"):
        ctx.worker_class().profile_memory()
    assert ctx.state.original_calls == 0


def test_module_import_does_not_load_npu_runtime() -> None:
    path = Path(npu_memory_counters.__file__).resolve()
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('counter_test', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert 'torch_npu' not in sys.modules\n"
        "assert 'vllm_ascend.worker.worker' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True, capture_output=True)
