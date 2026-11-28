# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""GR subclasses reuse native execution without installing native Runner methods."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from vllm.platforms import current_platform

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA native classes required")  # type: ignore[misc]
@pytest.mark.parametrize("v2", [False, True])  # type: ignore[misc]
def test_cuda_worker_selects_runner_from_resolved_config(
    v2: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as NativeV2
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as NativeV1
    from vllm.v1.worker.gpu_worker import Worker

    original = (Worker.init_device, NativeV1.execute_model, NativeV2.execute_model)
    from vllm_gr.v1.worker.cuda.worker import CudaGRWorker

    worker = object.__new__(CudaGRWorker)
    worker.use_v2_model_runner = v2
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0" if v2 else "1")
    cls = worker.get_model_runner_cls()
    assert issubclass(cls, NativeV2 if v2 else NativeV1)
    assert cls.__module__.startswith("vllm_gr.")
    assert original == (Worker.init_device, NativeV1.execute_model, NativeV2.execute_model)
    assert issubclass(CudaGRWorker, Worker)
    from vllm_gr.v1.worker.hot_update import reload_trie

    assert CudaGRWorker.reload_trie is reload_trie


def test_execute_failure_is_rethrown_at_native_sample_boundary() -> None:
    from vllm_gr.v1.worker.gr_execution import execute_gr_stage
    from vllm_gr.v1.worker.runner import sample_gr_tokens

    error = RuntimeError("failed before stage owner allocation")
    runner = SimpleNamespace(
        _gr_ready=True,
        _gr_pending_stage_sample=False,
        execute_gr_stage=Mock(side_effect=error),
        sample_gr_stage=Mock(),
    )
    scheduled = SimpleNamespace(
        num_scheduled_tokens={},
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )
    with pytest.raises(RuntimeError) as caught:
        execute_gr_stage(runner, scheduled, Mock())
    assert caught.value is error
    native = Mock()
    sample = sample_gr_tokens(native)
    with pytest.raises(RuntimeError) as caught:
        sample(runner, None)
    assert caught.value is error
    native.assert_not_called()
    runner.sample_gr_stage.assert_not_called()
    assert runner._gr_pending_stage_sample is False
    sample(runner, None)
    native.assert_called_once_with(runner, None)


def test_successful_async_stage_samples_once_and_rejects_overlap() -> None:
    from vllm_gr.v1.worker.gr_execution import execute_gr_stage
    from vllm_gr.v1.worker.runner import sample_gr_tokens

    runner = SimpleNamespace(
        _gr_ready=True,
        _gr_pending_stage_sample=False,
        execute_gr_stage=Mock(return_value=None),
        sample_gr_stage=Mock(return_value="beam"),
    )
    scheduled = SimpleNamespace(
        num_scheduled_tokens={},
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )
    assert execute_gr_stage(runner, scheduled, Mock()) is None
    with pytest.raises(RuntimeError, match="consume the previous"):
        execute_gr_stage(runner, scheduled, Mock())
    native = Mock(return_value="native")
    sample = sample_gr_tokens(native)
    assert sample(runner, None) == "beam"
    assert sample(runner, None) == "native"
    runner.sample_gr_stage.assert_called_once_with(None)
