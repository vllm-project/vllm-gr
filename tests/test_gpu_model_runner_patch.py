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
        _to_list=lambda tensor: tensor.tolist(),
    )
    ordinary_result = (1, ["lp"], ["valid"], object(), ["request"], {"request": 0}, [3])
    remuxed_logprobs = object()
    sampler_output = SimpleNamespace(
        logprobs_tensors=SimpleNamespace(tolists=lambda: remuxed_logprobs),
        sampled_token_ids=torch.tensor([[9]], dtype=torch.int32),
    )

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
    assert result[:3] == (1, remuxed_logprobs, [[9]])
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
