# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for NPU ModelRunner wrappers."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker import npu_model_runner_patch as npu_patch
from vllm_gr.v1.worker.model_runner_common import InputBatchProxy


def test_npu_prepare_unproxies_and_preserves_four_item_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_batch = SimpleNamespace(req_ids=["request"])
    proxy = InputBatchProxy(input_batch, ["request"], [0], [0])
    spec_metadata = object()
    compressed = object()
    runner = SimpleNamespace(
        input_batch=proxy,
        _current_beam_info={"req_ids": []},
        device=torch.device("cpu"),
    )

    def original(
        self: Any,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
    ) -> tuple[object, object, int, object]:
        assert self.input_batch is input_batch
        assert self._current_beam_info["req_ids"] is input_batch.req_ids
        return object(), spec_metadata, 7, compressed

    monkeypatch.setattr(npu_patch, "_beam_remap_inputs", lambda *args: [3])
    wrapped = npu_patch.make_patched_prepare_inputs(original)
    assert npu_patch.make_patched_prepare_inputs(wrapped) is wrapped

    ordinary = wrapped(runner, SimpleNamespace(beam_data={}), [1])
    assert ordinary[1:] == (spec_metadata, 7, compressed)

    runner.input_batch = proxy
    runner._current_beam_info["req_ids"] = []
    logits, returned_spec, total, returned_compressed = wrapped(
        runner,
        SimpleNamespace(beam_data={"request": {}}),
        [1],
    )
    assert logits.dtype is torch.int32
    assert logits.tolist() == [3]
    assert returned_spec is spec_metadata
    assert total == 7
    assert returned_compressed is compressed


def test_npu_bookkeeping_preserves_six_items_and_restores_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_batch = SimpleNamespace()
    proxy = SimpleNamespace(_obj=original_batch)
    runner = SimpleNamespace(
        input_batch=original_batch,
        _beam_logprobs_cache={"finished": {}, "active": {}},
    )
    ordinary_result = (["lp"], ["valid"], object(), ["request"], {"request": 0}, [3])
    sampler_output = object()

    def original(
        self: Any,
        scheduler_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        if getattr(scheduler_output, "fail", False):
            raise RuntimeError("boom")
        return ordinary_result

    def collapse(
        self: Any,
        orig_batch: Any,
        configs: Any,
        beam_data: Any,
        sampler: Any,
    ) -> tuple[list[str], dict[str, int], int]:
        assert orig_batch is original_batch
        assert sampler is sampler_output
        return ["beam"], {"beam": 0}, 1

    monkeypatch.setattr(npu_patch, "_beam_collapse_bookkeeping", collapse)
    wrapped = npu_patch.make_patched_bookkeeping_sync(original)

    assert wrapped(runner, SimpleNamespace(beam_data={}), sampler_output) is ordinary_result

    beam_output = SimpleNamespace(
        beam_data={"beam": {}},
        _gr_request_row_configs=[("beam", 0, 1, [0])],
        finished_req_ids={"finished"},
    )
    runner.input_batch = proxy
    result = wrapped(runner, beam_output, sampler_output)
    assert len(result) == 6
    assert result[:2] == ([[]], [[]])
    assert result[3:5] == (["beam"], {"beam": 0})
    assert runner.input_batch is original_batch
    assert "finished" not in runner._beam_logprobs_cache

    runner.input_batch = proxy
    with pytest.raises(RuntimeError, match="boom"):
        wrapped(
            runner,
            SimpleNamespace(
                beam_data={"beam": {}},
                _gr_request_row_configs=[],
                fail=True,
            ),
            sampler_output,
        )
    assert runner.input_batch is original_batch


def test_npu_prepare_publishes_scheduler_output() -> None:
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r1"]),
        _current_beam_info={"req_ids": []},
        device=torch.device("cpu"),
    )
    scheduler_output = SimpleNamespace(beam_data={})

    def original(
        self: Any, scheduler_output: Any, num_scheduled_tokens: Any
    ) -> tuple[Any, Any, int, Any]:
        return (torch.tensor([0]), None, 1, [object()])

    wrapped = npu_patch.make_patched_prepare_inputs(original)
    wrapped(runner, scheduler_output, [1])
    assert runner._current_scheduler_output is scheduler_output


def test_npu_capture_cudagraphs_triggers_prefill_capture_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.config import CUDAGraphMode

    from vllm_gr.v1.worker import npu_model_runner_patch as npu_patch_mod

    monkeypatch.setattr(
        "vllm_gr.v1.worker.prefill_graph_runner.is_prefill_graph_enabled", lambda: True
    )
    calls = {"capture": 0}

    class FakePrefillRunner:
        def capture_graphs(self: Any, runner: Any) -> None:
            calls["capture"] += 1

    monkeypatch.setattr(
        npu_patch_mod, "get_prefill_graph_runner", lambda runner: FakePrefillRunner()
    )

    def original(
        self: Any,
        batch_descriptors: Any,
        cudagraph_runtime_mode: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return "res"

    wrapped = npu_patch.make_patched_capture_cudagraphs_npu(original)
    runner = SimpleNamespace()
    assert wrapped(runner, [], CUDAGraphMode.FULL) == "res"
    assert calls["capture"] == 1

    # Second FULL pass does not re-trigger.
    assert wrapped(runner, [], CUDAGraphMode.FULL) == "res"
    assert calls["capture"] == 1

    # Non-FULL passes never trigger.
    assert wrapped(runner, [], CUDAGraphMode.NONE) == "res"
    assert calls["capture"] == 1


def test_npu_model_forward_intercepts_pure_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker import npu_model_runner_patch as npu_patch_mod

    output_buf = torch.zeros(4, 16)
    replay = {"bucket": None, "token_counts": None}

    class FakePrefillRunner:
        def match_bucket(self: Any, total_tokens: Any) -> Any:
            return 128

        def has_captured_graph(self: Any, bucket: Any, num_reqs: Any) -> Any:
            return True

        def log_replay(self: Any, bucket: Any, num_reqs: Any, total_tokens: Any) -> None:
            pass

        def execute_model_prefill(
            self: Any,
            bucket: Any,
            input_ids: Any,
            positions: Any,
            slot_mapping: Any,
            num_scheduled_tokens: Any,
        ) -> Any:
            if slot_mapping is None:
                return None
            replay["bucket"] = bucket
            replay["token_counts"] = num_scheduled_tokens
            return output_buf

    monkeypatch.setattr(
        "vllm_gr.v1.worker.prefill_graph_runner.is_prefill_graph_enabled", lambda: True
    )
    monkeypatch.setattr(
        "vllm_gr.v1.worker.prefill_graph_runner.is_prefill_phase_batch",
        lambda self, scheduler_output: scheduler_output.pure,
    )
    monkeypatch.setattr(
        npu_patch_mod, "get_prefill_graph_runner", lambda runner: FakePrefillRunner()
    )

    original_calls = []

    def original(
        self: Any,
        num_tokens_padded: Any,
        input_ids: Any = None,
        positions: Any = None,
        intermediate_tensors: Any = None,
        inputs_embeds: Any = None,
        **model_kwargs: Any,
    ) -> Any:
        original_calls.append((num_tokens_padded, input_ids, positions))
        return "orig_out"

    wrapped = npu_patch.make_patched_model_forward_npu(original)

    meta = SimpleNamespace(slot_mapping=torch.zeros(64, dtype=torch.int64))
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"l": meta}),
    )
    scheduler_output = SimpleNamespace(pure=True, num_scheduled_tokens={"r1": 30, "r2": 34})
    runner = SimpleNamespace(_current_scheduler_output=scheduler_output)
    result = wrapped(runner, 64, input_ids="ids", positions="pos")
    assert result is output_buf
    assert replay["bucket"] == 128
    assert replay["token_counts"] == {"r1": 30, "r2": 34}
    assert original_calls == []

    # Missing forward-context metadata falls back to eager.
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )
    scheduler_output.pure = True
    result = wrapped(runner, 64, input_ids="ids", positions="pos")
    assert result == "orig_out"
    assert original_calls == [(64, "ids", "pos")]

    # Non-pure batches fall through to the original forward (positional args).
    scheduler_output.pure = False
    result = wrapped(runner, 64, input_ids="ids", positions="pos")
    assert result == "orig_out"
    assert original_calls == [(64, "ids", "pos"), (64, "ids", "pos")]


def test_npu_prefill_phase_batch_classification() -> None:
    from types import SimpleNamespace

    import numpy as np

    from vllm_gr.v1.worker.prefill_graph_runner import is_prefill_phase_batch

    def classify(scheduler_output: Any, input_batch: Any) -> bool:
        return bool(
            is_prefill_phase_batch(SimpleNamespace(input_batch=input_batch), scheduler_output)
        )

    base_output = {
        "beam_data": {},
        "scheduled_spec_decode_tokens": {},
        "scheduled_encoder_inputs": {},
        "num_scheduled_tokens": {"r1": 30, "r2": 34},
    }
    input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([0, 0]),
        num_prompt_tokens=np.array([128, 64]),
        req_id_to_index={"r1": 0, "r2": 1},
    )

    # Fresh prefill (first chunks).
    assert classify(SimpleNamespace(**base_output), input_batch) is True

    # Cached / chunked prefill continuations (num_computed > 0, still prefill).
    input_batch.num_computed_tokens_cpu = np.array([30, 32])
    assert classify(SimpleNamespace(**base_output), input_batch) is True

    # Mixed decode (num_computed == num_prompt) is rejected.
    input_batch.num_computed_tokens_cpu = np.array([128, 32])
    assert classify(SimpleNamespace(**base_output), input_batch) is False

    # Beam decode is rejected.
    output = {**base_output, "beam_data": {"r1": {"is_beam_decode": True}}}
    input_batch.num_computed_tokens_cpu = np.array([0, 0])
    assert classify(SimpleNamespace(**output), input_batch) is False

    # Spec decode and encoder batches are rejected.
    output = {**base_output, "scheduled_spec_decode_tokens": {"r1": [7]}}
    assert classify(SimpleNamespace(**output), input_batch) is False
    output = {**base_output, "scheduled_encoder_inputs": {"r1": [0]}}
    assert classify(SimpleNamespace(**output), input_batch) is False

    # Empty batch is rejected.
    output = {**base_output, "num_scheduled_tokens": {}}
    assert classify(SimpleNamespace(**output), input_batch) is False
