# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for GPU ModelRunner wrappers."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker import gpu_model_runner_patch as gpu_patch


def test_gpu_prepare_preserves_order_and_two_item_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    spec_metadata = object()
    original_logits = object()
    input_batch = SimpleNamespace(req_ids=["request"])
    runner = SimpleNamespace(
        input_batch=input_batch,
        _current_beam_info={"req_ids": []},
        device=torch.device("cpu"),
    )

    def original(
        self: Any,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
    ) -> tuple[object, object]:
        assert self._current_beam_info["req_ids"] == []
        events.append("original")
        return original_logits, spec_metadata

    def remap(
        self: Any,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
    ) -> list[int]:
        assert self._current_beam_info["req_ids"] is input_batch.req_ids
        events.append("remap")
        return [2, 4]

    monkeypatch.setattr(gpu_patch, "_beam_remap_inputs", remap)
    wrapped = gpu_patch.make_patched_prepare_inputs(original)
    assert gpu_patch.make_patched_prepare_inputs(wrapped) is wrapped

    ordinary = wrapped(runner, SimpleNamespace(beam_data={}), [1])
    assert ordinary == (original_logits, spec_metadata)
    assert events == ["original"]

    runner._current_beam_info["req_ids"] = []
    logits, returned_spec = wrapped(
        runner,
        SimpleNamespace(beam_data={"request": {}}),
        [1],
    )
    assert events == ["original", "original", "remap"]
    assert logits.dtype is torch.int32
    assert logits.tolist() == [2, 4]
    assert returned_spec is spec_metadata


def test_gpu_bookkeeping_preserves_seven_items_and_restores_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_batch = SimpleNamespace()
    proxy = SimpleNamespace(_obj=original_batch)
    runner = SimpleNamespace(
        input_batch=original_batch,
        _beam_logprobs_cache={"finished": {}, "active": {}},
    )
    ordinary_result = (1, ["lp"], ["valid"], object(), ["request"], {"request": 0}, [3])
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

    monkeypatch.setattr(gpu_patch, "_beam_collapse_bookkeeping", collapse)
    wrapped = gpu_patch.make_patched_bookkeeping_sync(original)

    assert wrapped(runner, SimpleNamespace(beam_data={}), sampler_output) is ordinary_result

    beam_output = SimpleNamespace(
        beam_data={"beam": {}},
        _gr_request_row_configs=[("beam", 0, 1, [0])],
        finished_req_ids={"finished"},
    )
    runner.input_batch = proxy
    result = wrapped(runner, beam_output, sampler_output)
    assert len(result) == 7
    assert result[:3] == (1, [[]], [[]])
    assert result[4:6] == (["beam"], {"beam": 0})
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


def test_gpu_prepare_publishes_scheduler_output() -> None:
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r1"]),
        _current_beam_info={"req_ids": []},
        device=torch.device("cpu"),
    )
    scheduler_output = SimpleNamespace(beam_data={})

    def original(self: Any, scheduler_output: Any, num_scheduled_tokens: Any) -> tuple[Any, Any]:
        return (torch.tensor([0]), object())

    wrapped = gpu_patch.make_patched_prepare_inputs(original)
    wrapped(runner, scheduler_output, [1])
    assert runner._current_scheduler_output is scheduler_output


def test_gpu_model_forward_intercepts_pure_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.worker import gpu_model_runner_patch as gpu_patch_mod

    output_buf = torch.zeros(4, 16)
    replay = {"bucket": None, "token_counts": None}

    class FakePrefillRunner:
        def match_bucket(self: Any, total_tokens: Any) -> Any:
            return 128

        def has_captured_graph(self: Any, bucket: Any, num_reqs: Any) -> Any:
            return True

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

    monkeypatch.setattr(gpu_patch_mod, "is_prefill_graph_enabled", lambda: True)
    monkeypatch.setattr(
        gpu_patch_mod,
        "is_prefill_phase_batch",
        lambda self, scheduler_output: scheduler_output.pure,
    )
    monkeypatch.setattr(
        gpu_patch_mod, "get_prefill_graph_runner", lambda runner: FakePrefillRunner()
    )

    original_calls = []

    def original(
        self: Any,
        input_ids: Any = None,
        positions: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_calls.append((input_ids, positions))
        return "orig_out"

    wrapped = gpu_patch.make_patched_model_forward_gpu(original)

    meta = SimpleNamespace(slot_mapping=torch.zeros(64, dtype=torch.int64))
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"l": meta}),
    )
    scheduler_output = SimpleNamespace(pure=True, num_scheduled_tokens={"r1": 30, "r2": 34})
    slot_mapping_gpu = torch.zeros(64, dtype=torch.int64)
    runner = SimpleNamespace(
        _current_scheduler_output=scheduler_output,
        input_batch=SimpleNamespace(
            block_table=[SimpleNamespace(slot_mapping=SimpleNamespace(gpu=slot_mapping_gpu))]
        ),
    )
    result = wrapped(runner, input_ids="ids", positions="pos")
    assert result is output_buf
    assert replay["bucket"] == 128
    assert replay["token_counts"] == {"r1": 30, "r2": 34}
    assert original_calls == []

    # No published scheduler output falls through to the original forward.
    runner = SimpleNamespace(_current_scheduler_output=None)
    result = wrapped(runner, input_ids="ids", positions="pos")
    assert result == "orig_out"
    assert original_calls == [("ids", "pos")]

    # Non-pure batches fall through as well.
    scheduler_output.pure = False
    runner = SimpleNamespace(_current_scheduler_output=scheduler_output)
    result = wrapped(runner, input_ids="ids", positions="pos")
    assert result == "orig_out"
    assert original_calls == [("ids", "pos"), ("ids", "pos")]

    # A pure batch without a slot mapping falls back to eager (the graph
    # must never replay with stale slots).
    scheduler_output.pure = True
    runner = SimpleNamespace(
        _current_scheduler_output=scheduler_output,
        input_batch=SimpleNamespace(block_table=None),
    )
    result = wrapped(runner, input_ids="ids", positions="pos")
    assert result == "orig_out"
    assert original_calls == [("ids", "pos"), ("ids", "pos"), ("ids", "pos")]
