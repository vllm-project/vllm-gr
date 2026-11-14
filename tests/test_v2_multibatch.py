# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""GR/native row boundary and independent multi-B capture policy."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_gr.v1.worker.gpu_beam_stage_runner_v1 import CudaStageHooksV1
from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import CudaStageHooksV2


def test_prefill_restores_member_order_without_reordering_native_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.utils import torch_utils

    logical = ("a", "b", "c", "d")
    native = ("d", "b", "c", "a")
    batch = SimpleNamespace(
        req_ids=list(native),
        cu_num_logits_np=np.arange(5),
        logits_indices=torch.tensor([0, 2, 5, 9]),
        idx_mapping=torch.tensor([6, 1, 7, 3]),
        query_start_loc=torch.tensor([0, 1, 3, 6, 10]),
    )
    worker = SimpleNamespace(
        device="cpu",
        model=SimpleNamespace(compute_logits=lambda h: h),
        execute_model_state=SimpleNamespace(
            hidden_states=torch.arange(10)[:, None], input_batch=batch
        ),
        req_states=SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=object())),
        postprocess_num_computed_tokens=Mock(),
    )
    monkeypatch.setattr(
        torch_utils, "async_tensor_h2d", lambda x, dtype, device: torch.tensor(x, dtype=dtype)
    )
    scheduled = SimpleNamespace(gr_stage_metadata={"a": SimpleNamespace(request_ids=logical)})
    state = worker.execute_model_state
    result = CudaStageHooksV2(worker).prefill_logits(lambda *args: None, scheduled)
    expected = {"a": 9, "b": 2, "c": 5, "d": 0}
    assert result[:, 0].tolist() == [expected[r] for r in logical]
    assert batch.req_ids == list(native)
    worker.postprocess_num_computed_tokens.assert_called_once_with(batch)
    assert worker.execute_model_state is None
    worker.execute_model_state = state
    scheduled.gr_stage_metadata["a"].request_ids = ("a", "a", "c", "d")
    with pytest.raises(ValueError, match="one logit row"):
        CudaStageHooksV2(worker).prefill_logits(lambda *args: None, scheduled)


def test_capture_attention_bounds_are_local_to_v2_pure_prefill() -> None:
    from dataclasses import dataclass

    from vllm_gr.v1.attention.backends.beam_attn_metadata import (
        BeamAttentionMetadataBuilder,
    )

    @dataclass
    class Common:
        num_actual_tokens: int = 512
        max_query_len: int = 128
        max_seq_len: int = 128

    builder = BeamAttentionMetadataBuilder.__new__(BeamAttentionMetadataBuilder)
    builder._v2_gr_prefill_capture = True
    builder.model_config = SimpleNamespace(max_model_len=2048)
    builder.build = lambda **kw: kw["common_attn_metadata"]
    common = Common()
    bounded = builder.build_for_cudagraph_capture(common)
    assert (bounded.max_query_len, bounded.max_seq_len) == (512, 2048)
    assert common.max_query_len == 128
    builder._v2_gr_prefill_capture = False
    assert builder.build_for_cudagraph_capture(common) is common


@pytest.mark.parametrize("v2", [False, True])  # type: ignore[misc]
def test_retirement_preserves_runner_policy_and_scheduler_output(v2: bool) -> None:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_gr.v1.engine.beam_search_v1 import GRRetireMetadata
    from vllm_gr.v1.worker.gpu_beam_stage_runner import execute_gr_retirement

    scheduled = SchedulerOutput.make_empty()
    scheduled.gr_retire_metadata = GRRetireMetadata("a", 0, 1)
    scheduled.finished_req_ids = {"a", "unrelated"}
    hooks = (CudaStageHooksV2 if v2 else CudaStageHooksV1)(SimpleNamespace())
    hooks.prefill_stage_fence = Mock()
    hooks.consume_native_state = Mock()
    owner = SimpleNamespace(
        closed=False,
        request_release=Mock(),
        hooks=hooks,
        bindings=tuple(SimpleNamespace(native_request_id=r) for r in ("a", "b", "c", "d")),
    )
    worker = SimpleNamespace(_gr_gpu_stage_runner=owner)
    execute_gr_retirement(worker, scheduled)
    cleanup = hooks.consume_native_state.call_args.args[0]
    assert cleanup.finished_req_ids == (
        {"a", "b", "c", "d", "unrelated"} if v2 else {"a", "unrelated"}
    )
    assert (cleanup is scheduled) is (not v2)
    owner.request_release.assert_called_once()
    assert scheduled.finished_req_ids == {"a", "unrelated"}


@pytest.mark.parametrize("v2", [False, True])  # type: ignore[misc]
def test_stage_replacement_and_capacity_are_runner_specific(v2: bool) -> None:
    graph = Mock()
    worker = SimpleNamespace(_beam_decode_graph=graph)
    hooks = (CudaStageHooksV2 if v2 else CudaStageHooksV1)(worker)
    hooks.prepare_new_adapter(SimpleNamespace(batch_size=2, beam_width=4))
    if v2:
        # Every V2 adapter shares the Worker storage, so captures survive.
        graph.discard_geometry.assert_not_called()
    else:
        # A new V1 stage replaces the input storage: drop every length bucket.
        graph.discard_geometry.assert_called_once_with(2, 4)
    assert hooks.required_constraint_rows(2, 4) == (8 if v2 else 4)


pytestmark = pytest.mark.cpu_test
