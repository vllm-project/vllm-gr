# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for NPU ModelRunner wrappers."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker import npu_model_runner_patch as npu_patch
from vllm_gr.v1.worker.model_runner_common import (
    InputBatchProxy,
    _build_p2d_bookkeeping_result,
)


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


def test_npu_model_forward_routes_p2d_first(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class MockGraphRunner:
        def match_bucket(self, total_tokens: int) -> int | None:
            return None

        def has_captured_graph(self, bucket: int, num_reqs: int) -> bool:
            return False

        def has_2d_graph(self, num_reqs: int, bw: int) -> bool:
            return True

        def execute_decode_p2d(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append("execute_decode_p2d")
            return (
                torch.zeros((1, 32, 3), dtype=torch.int64),
                torch.zeros((1, 32), dtype=torch.float32),
            )

        def get_output_buf(self, total_tokens: int, model_config: Any) -> torch.Tensor:
            return torch.zeros((total_tokens, 64), dtype=torch.bfloat16)

        def get_prefill_logit_indices(self, num_scheduled_tokens: Any) -> list[int]:
            counts = (
                list(num_scheduled_tokens.values())
                if isinstance(num_scheduled_tokens, dict)
                else list(num_scheduled_tokens)
            )
            indices = []
            curr = 0
            for cnt in counts:
                curr += cnt
                indices.append(curr - 1)
            return indices

    p_runner = MockGraphRunner()
    from vllm_gr.v1.worker import prefill_graph_runner as pgr

    monkeypatch.setattr(pgr, "is_prefill_graph_enabled", lambda: True)
    monkeypatch.setattr(pgr, "is_prefill_phase_batch", lambda self, sched: sched.pure)
    monkeypatch.setattr(npu_patch, "get_prefill_graph_runner", lambda runner: p_runner)
    from vllm_gr.v1.worker.prefill_graph_common import StepBeamWidths

    monkeypatch.setattr(
        npu_patch, "match_p2d_step_beam_widths", lambda r, s: StepBeamWidths(32, 32, 32)
    )
    monkeypatch.setattr(npu_patch, "is_p2d_eligible", lambda *args, **kwargs: True)

    import vllm.forward_context as fc

    fake_ctx = SimpleNamespace(
        attn_metadata={
            "layer0": SimpleNamespace(
                block_table=torch.zeros((1, 4), dtype=torch.int32),
                slot_mapping=torch.zeros(64, dtype=torch.int64),
            )
        }
    )
    monkeypatch.setattr(fc, "get_forward_context", lambda: fake_ctx)

    def dummy_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append("orig_forward")
        return torch.zeros((64, 64))

    wrapped = npu_patch.make_patched_model_forward_npu(dummy_forward)
    runner = SimpleNamespace(
        _current_scheduler_output=SimpleNamespace(
            pure=True,
            total_num_scheduled_tokens=64,
            num_scheduled_tokens={"r1": 64},
        ),
        input_batch=SimpleNamespace(req_ids=["r1"]),
        model=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16, get_hidden_size=lambda: 64),
        device=torch.device("cpu"),
    )
    out = wrapped(runner, 64, input_ids=torch.zeros(64), positions=torch.zeros(64))
    assert "execute_decode_p2d" in calls
    # Eager prefill fallback ran (no prefill graph), then the decode graph.
    assert "orig_forward" in calls
    assert out.shape[0] >= 64
    assert runner._current_p2d_result is not None


def test_npu_model_forward_extracts_numpy_computed_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    received_kwargs: dict[str, Any] = {}

    class MockGraphRunner:
        def match_bucket(self, total_tokens: int) -> int | None:
            return None

        def has_captured_graph(self, bucket: int, num_reqs: int) -> bool:
            return False

        def has_2d_graph(self, num_reqs: int, bw: int) -> bool:
            return True

        def execute_decode_p2d(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            received_kwargs.update(kwargs)
            return (
                torch.zeros((1, 32, 3), dtype=torch.int64),
                torch.zeros((1, 32), dtype=torch.float32),
            )

        def get_output_buf(self, total_tokens: int, model_config: Any) -> torch.Tensor:
            return torch.zeros((total_tokens, 64), dtype=torch.bfloat16)

        def get_prefill_logit_indices(self, num_scheduled_tokens: Any) -> list[int]:
            counts = (
                list(num_scheduled_tokens.values())
                if isinstance(num_scheduled_tokens, dict)
                else list(num_scheduled_tokens)
            )
            indices = []
            curr = 0
            for cnt in counts:
                curr += cnt
                indices.append(curr - 1)
            return indices

    p_runner = MockGraphRunner()
    from vllm_gr.v1.worker import prefill_graph_runner as pgr

    monkeypatch.setattr(pgr, "is_prefill_graph_enabled", lambda: True)
    monkeypatch.setattr(pgr, "is_prefill_phase_batch", lambda self, sched: sched.pure)
    monkeypatch.setattr(npu_patch, "get_prefill_graph_runner", lambda runner: p_runner)
    from vllm_gr.v1.worker.prefill_graph_common import StepBeamWidths

    monkeypatch.setattr(
        npu_patch, "match_p2d_step_beam_widths", lambda r, s: StepBeamWidths(32, 32, 32)
    )
    monkeypatch.setattr(npu_patch, "is_p2d_eligible", lambda *args, **kwargs: True)

    import vllm.forward_context as fc

    fake_ctx = SimpleNamespace(
        attn_metadata={
            "layer0": SimpleNamespace(
                block_table=torch.zeros((1, 4), dtype=torch.int32),
                slot_mapping=torch.zeros(64, dtype=torch.int64),
            )
        }
    )
    monkeypatch.setattr(fc, "get_forward_context", lambda: fake_ctx)

    def dummy_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        return torch.zeros((64, 64))

    wrapped = npu_patch.make_patched_model_forward_npu(dummy_forward)
    runner = SimpleNamespace(
        _current_scheduler_output=SimpleNamespace(
            pure=True,
            total_num_scheduled_tokens=64,
            num_scheduled_tokens={"r1": 64},
        ),
        input_batch=SimpleNamespace(
            req_ids=["r1"],
            req_id_to_index={"r1": 0},
            num_computed_tokens_cpu=np.array([128], dtype=np.int32),
        ),
        model=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16, get_hidden_size=lambda: 64),
        device=torch.device("cpu"),
    )
    out = wrapped(runner, 64, input_ids=torch.zeros(64), positions=torch.zeros(64))
    assert out is not None
    assert received_kwargs.get("computed_tokens") == [128]


def test_npu_bookkeeping_sync_p2d_six_item(monkeypatch: pytest.MonkeyPatch) -> None:
    req_ids = ["r1"]
    input_batch = SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={"r1": 0},
        num_tokens_no_spec=[4],
        token_ids_cpu=torch.zeros((1, 8), dtype=torch.int64),
        is_token_ids=torch.zeros((1, 8), dtype=torch.bool),
    )
    requests = {"r1": SimpleNamespace(output_token_ids=[])}
    runner = SimpleNamespace(
        input_batch=input_batch,
        requests=requests,
        use_async_scheduling=False,
        _current_p2d_result=(
            torch.tensor([[[7, 8, 9], [1, 2, 3]]], dtype=torch.int64),
            torch.tensor([[1.5, 0.5]], dtype=torch.float32),
        ),
        _beam_logprobs_cache={},
    )

    def orig_bookkeeping(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("orig bookkeeping must not run for the p2d branch")

    wrapped = npu_patch.make_patched_bookkeeping_sync_npu(orig_bookkeeping)
    res = wrapped(
        runner,
        SimpleNamespace(finished_req_ids=[], beam_data={}),
        SimpleNamespace(sampled_token_ids=None),
    )
    assert len(res) == 6
    logprobs_lists, valid_sampled_token_ids, prompt_logprobs, req_ids_copy, idx_copy, invalid = res
    assert prompt_logprobs == {}
    assert invalid == []
    assert req_ids_copy == req_ids
    assert idx_copy == {"r1": 0}
    assert valid_sampled_token_ids == [[7]]
    assert logprobs_lists.logprob_token_ids[0].tolist() == [7, 8, 9, 1, 2, 3]
    assert requests["r1"].output_token_ids == [7]
    assert input_batch.num_tokens_no_spec[0] == 5
    assert input_batch.token_ids_cpu[0, 4].item() == 7
    assert runner._current_p2d_result is None


def test_npu_sample_bypasses_sampler_on_p2d() -> None:
    def orig_sample(self: Any, logits: Any, spec_decode_metadata: Any) -> Any:
        return "real_sample"

    wrapped = npu_patch.make_patched_sample_npu(orig_sample)
    assert npu_patch.make_patched_sample_npu(wrapped) is wrapped  # idempotent

    runner = SimpleNamespace(
        _current_p2d_result=(
            torch.tensor([[[7, 8, 9]]], dtype=torch.int64),
            torch.tensor([[1.5]], dtype=torch.float32),
        )
    )
    out = wrapped(runner, None, None)
    assert out.sampled_token_ids.tolist() == [[7]]
    assert out.logprobs_tensors is None

    runner._current_p2d_result = None
    assert wrapped(runner, None, None) == "real_sample"


def test_build_p2d_bookkeeping_result_shared() -> None:
    req_ids = ["r1", "r2"]
    input_batch = SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={"r1": 0, "r2": 1},
        num_tokens_no_spec=[4, 6],
        token_ids_cpu=torch.zeros((2, 8), dtype=torch.int64),
        is_token_ids=torch.zeros((2, 8), dtype=torch.bool),
    )
    requests = {
        "r1": SimpleNamespace(output_token_ids=[]),
        "r2": SimpleNamespace(output_token_ids=[]),
    }
    runner = SimpleNamespace(
        input_batch=input_batch,
        requests=requests,
        use_async_scheduling=False,
    )
    final_tokens = torch.tensor(
        [[[7, 8, 9], [1, 2, 3]], [[4, 5, 6], [10, 11, 12]]], dtype=torch.int64
    )
    final_cum_lps = torch.tensor([[1.5, 0.5], [2.0, 1.0]], dtype=torch.float32)

    (
        logprobs_lists,
        valid_sampled_token_ids,
        req_ids_copy,
        idx_copy,
        invalid,
    ) = _build_p2d_bookkeeping_result(runner, final_tokens, final_cum_lps, None, None)

    assert valid_sampled_token_ids == [[7], [4]]
    assert logprobs_lists.logprob_token_ids[0].tolist() == [7, 8, 9, 1, 2, 3]
    assert logprobs_lists.logprobs[0].tolist() == [1.5, 1.5, 1.5, 0.5, 0.5, 0.5]
    assert (logprobs_lists.sampled_token_ranks == -999).all()
    assert req_ids_copy == req_ids
    assert idx_copy == {"r1": 0, "r2": 1}
    assert invalid == []
    assert requests["r1"].output_token_ids == [7]
    assert requests["r2"].output_token_ids == [4]


def test_npu_p2d_attention_signature_accepts_kv_scales(monkeypatch: Any) -> None:
    """Verifies that beam_attn_npu.p2d_attention accepts kv_cache_dtype, k_scale, v_scale without TypeError."""
    import sys
    from unittest.mock import MagicMock

    sys.modules.setdefault("torch_npu", MagicMock())
    from vllm_gr.v1.attention.backends.beam_attn_npu import p2d_attention

    meta = SimpleNamespace(p2d_pass=0)
    # Pass 0 without paged-cache tensors raises RuntimeError — proving the
    # function signature (incl. kv_cache_dtype/k_scale/v_scale) was accepted.
    with pytest.raises(RuntimeError, match="pass 0 requires the paged KV cache"):
        p2d_attention(
            output=torch.zeros(1),
            query=torch.zeros(1),
            key=torch.zeros(1),
            value=torch.zeros(1),
            attn_metadata=meta,
            layer_idx=0,
            scale=1.0,
            num_heads=1,
            num_kv_heads=1,
            head_size=64,
            kv_cache_dtype="fp8",
            k_scale=torch.tensor(1.0),
            v_scale=torch.tensor(1.0),
        )


def test_npu_execute_decode_p2d_handles_computed_tokens(monkeypatch: Any) -> None:
    """Verifies that NPUPrefillGraphRunner.execute_decode_p2d sets positions and prefix_kvlen_list to computed + scheduled."""
    import sys
    from unittest.mock import MagicMock

    sys.modules.setdefault("torch_npu", MagicMock())
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner

    class MockModelRunner:
        def __init__(self) -> None:
            self.model_config = SimpleNamespace(
                max_model_len=128,
                get_hidden_size=lambda: 64,
                dtype=torch.bfloat16,
            )
            self.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
            self.device = torch.device("cpu")

    runner = MockModelRunner()
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[128])
    npu_runner.p2d_pool = SimpleNamespace(
        max_beams=128,
        positions_1=torch.zeros(128, dtype=torch.int64),
        positions_2=torch.zeros(128, dtype=torch.int64),
        logits_0=torch.zeros((1, 100), dtype=torch.bfloat16),
        tokens_0=torch.zeros((1, 128), dtype=torch.int64),
        cum_lps_0=torch.zeros((1, 128), dtype=torch.float32),
        input_ids_1=torch.zeros(128, dtype=torch.int32),
        final_tokens=torch.ones((1, 32, 3), dtype=torch.int64),
        final_cum_lps=torch.zeros((1, 32), dtype=torch.float32),
        valid_beam_counts=torch.zeros(1, dtype=torch.int32),
        step_scores=torch.zeros((1, 32, 3), dtype=torch.float32),
    )

    replayed = []
    entry = SimpleNamespace()
    npu_runner.captured_2d_graphs[(1, 32)] = (
        SimpleNamespace(replay=lambda: replayed.append(True)),
        entry,
    )

    captured_params = {}

    def fake_update(stream: Any, ent: Any, qlens: Any, kvlens: Any, bt: Any) -> bool:
        captured_params["kvlens"] = kvlens
        return True

    monkeypatch.setattr(
        "vllm_gr.v1.attention.backends.beam_attn_npu.update_decode_graph_params",
        fake_update,
    )
    monkeypatch.setattr(
        npu_runner,
        "_get_prefix_block_table_buf",
        lambda cols: torch.zeros((32, cols), dtype=torch.int32),
    )
    mock_npu = SimpleNamespace(
        current_stream=lambda: SimpleNamespace(wait_stream=lambda s: None),
    )
    monkeypatch.setattr(torch, "npu", mock_npu, raising=False)
    monkeypatch.setattr(npu_runner, "_get_update_stream", lambda: None)

    mock_runner = SimpleNamespace(
        model=SimpleNamespace(compute_logits=lambda hidden: torch.zeros((1, 100)))
    )

    res = npu_runner.execute_decode_p2d(
        runner=mock_runner,
        hidden=torch.zeros((50, 64)),
        block_table=torch.zeros((1, 4), dtype=torch.int32),
        num_scheduled_tokens=[50],
        beam_width=32,
        computed_tokens=[100],
    )

    assert res is not None
    assert npu_runner.p2d_pool.positions_1[0].item() == 150  # 100 computed + 50 scheduled
    assert npu_runner.p2d_pool.positions_2[0].item() == 151
    assert captured_params["kvlens"] == [150] * 32


def _make_p2d_routing_runner(
    monkeypatch: pytest.MonkeyPatch,
    total_tokens: int,
    computed: int = 0,
) -> tuple[Any, Any, list[str], dict[str, Any]]:
    calls: list[str] = []
    received: dict[str, Any] = {}

    class MockGraphRunner:
        def match_p2d_bucket(self, total_tokens: int) -> int:
            return 512

        def match_bucket(self, total_tokens: int) -> int | None:
            return None

        def has_captured_graph(self, bucket: int, num_reqs: int) -> bool:
            return False

        def has_monolithic_graph(self, bucket: int, num_reqs: int, bw: int) -> bool:
            return True

        def has_2d_graph(self, num_reqs: int, bw: int) -> bool:
            return True

        def execute_p2d(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append("execute_p2d")
            received.update(kwargs)
            return (
                torch.zeros((1, 32, 3), dtype=torch.int64),
                torch.zeros((1, 32), dtype=torch.float32),
            )

        def execute_decode_p2d(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append("execute_decode_p2d")
            received.update(kwargs)
            return (
                torch.zeros((1, 32, 3), dtype=torch.int64),
                torch.zeros((1, 32), dtype=torch.float32),
            )

        def get_output_buf(self, total_tokens: int, model_config: Any) -> torch.Tensor:
            return torch.zeros((total_tokens, 64), dtype=torch.bfloat16)

        def get_prefill_logit_indices(self, num_scheduled_tokens: Any) -> list[int]:
            counts = (
                list(num_scheduled_tokens.values())
                if isinstance(num_scheduled_tokens, dict)
                else list(num_scheduled_tokens)
            )
            indices = []
            curr = 0
            for cnt in counts:
                curr += cnt
                indices.append(curr - 1)
            return indices

    p_runner = MockGraphRunner()
    from vllm_gr.v1.worker import prefill_graph_runner as pgr

    monkeypatch.setattr(pgr, "is_prefill_graph_enabled", lambda: True)
    monkeypatch.setattr(pgr, "is_prefill_phase_batch", lambda self, sched: sched.pure)
    monkeypatch.setattr(npu_patch, "get_prefill_graph_runner", lambda runner: p_runner)
    from vllm_gr.v1.worker.prefill_graph_common import StepBeamWidths

    monkeypatch.setattr(
        npu_patch, "match_p2d_step_beam_widths", lambda r, s: StepBeamWidths(32, 32, 32)
    )
    monkeypatch.setattr(npu_patch, "is_p2d_eligible", lambda *args, **kwargs: True)

    import vllm.forward_context as fc

    fake_ctx = SimpleNamespace(
        attn_metadata={
            "layer0": SimpleNamespace(
                block_table=torch.zeros((1, 4), dtype=torch.int32),
                slot_mapping=torch.zeros(512, dtype=torch.int64),
            )
        }
    )
    monkeypatch.setattr(fc, "get_forward_context", lambda: fake_ctx)

    def dummy_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append("orig_forward")
        return torch.zeros((total_tokens, 64))

    wrapped = npu_patch.make_patched_model_forward_npu(dummy_forward)
    runner = SimpleNamespace(
        _current_scheduler_output=SimpleNamespace(
            pure=True,
            total_num_scheduled_tokens=total_tokens,
            num_scheduled_tokens={"r1": total_tokens},
        ),
        input_batch=SimpleNamespace(
            req_ids=["r1"],
            req_id_to_index={"r1": 0},
            num_computed_tokens_cpu=[computed],
        ),
        model=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16, get_hidden_size=lambda: 64),
        device=torch.device("cpu"),
    )
    return wrapped, runner, calls, received


def test_npu_model_forward_routes_monolithic_when_padding_within_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 500 tokens into bucket 512 -> pad 12/500 = 2.4% <= 10% -> monolithic.
    wrapped, runner, calls, received = _make_p2d_routing_runner(monkeypatch, total_tokens=500)
    out = wrapped(runner, 500, input_ids=torch.zeros(500), positions=torch.zeros(500))
    assert "execute_p2d" in calls
    assert "execute_decode_p2d" not in calls
    assert "orig_forward" not in calls
    assert received.get("bucket") == 512
    assert received.get("computed_tokens") == [0]
    assert out.shape[0] >= 500
    assert runner._current_p2d_result is not None


def test_npu_model_forward_falls_back_to_2d_when_padding_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 400 tokens into bucket 512 -> pad 112/400 = 28% > 10% -> Eager-P + 2D.
    wrapped, runner, calls, received = _make_p2d_routing_runner(monkeypatch, total_tokens=400)
    out = wrapped(runner, 400, input_ids=torch.zeros(400), positions=torch.zeros(400))
    assert "execute_p2d" not in calls
    assert "execute_decode_p2d" in calls
    assert "orig_forward" in calls  # eager prefill stage (no prefill graph)
    assert received.get("computed_tokens") == [0]
    assert out.shape[0] >= 400
    assert runner._current_p2d_result is not None


def test_npu_monolithic_routing_threads_computed_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cached prefix (computed=128) + 500-token chunk -> 2.4% pad -> monolithic.
    wrapped, runner, calls, received = _make_p2d_routing_runner(
        monkeypatch, total_tokens=500, computed=128
    )
    wrapped(runner, 500, input_ids=torch.zeros(500), positions=torch.zeros(500))
    assert "execute_p2d" in calls
    assert received.get("computed_tokens") == [128]


def test_npu_has_monolithic_graph(monkeypatch: Any) -> None:
    import sys
    from unittest.mock import MagicMock

    sys.modules.setdefault("torch_npu", MagicMock())
    from vllm_gr.v1.worker import prefill_graph_common

    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: False)
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner

    runner = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128, get_hidden_size=lambda: 64),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        device=torch.device("cpu"),
    )
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[512])
    assert npu_runner.has_monolithic_graph(512, 1, 128) is False
    npu_runner.monolithic_graphs[(512, 1, 128)] = ("graph", "entry")
    assert npu_runner.has_monolithic_graph(512, 1, 128) is True
    assert npu_runner.has_monolithic_graph(512, 1, 32) is False
    monkeypatch.setattr(prefill_graph_common, "is_npu_platform", lambda: True)
    assert npu_runner.has_monolithic_graph(512, 1, 128) is False


def test_npu_monolithic_dummy_metadata_missing_bucket_uses_pad_zero(monkeypatch: Any) -> None:
    import importlib.machinery
    import sys
    from unittest.mock import MagicMock

    _mock_torch_npu = MagicMock()
    _mock_torch_npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    sys.modules.setdefault("torch_npu", _mock_torch_npu)
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner
    from vllm_gr.v1.worker.prefill_graph_common import P2DBufferPool

    class MockBuilder:
        num_heads_q = 4
        num_kv_heads = 2
        block_size = 16

    runner = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128, get_hidden_size=lambda: 64),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        device=torch.device("cpu"),
    )
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[512])
    npu_runner.p2d_pool = P2DBufferPool(
        max_batch=2,
        max_bucket=1024,
        max_beams=128,
        num_layers=1,
        device=torch.device("cpu"),
    )

    monkeypatch.setattr(
        npu_runner, "_get_attn_builder_and_layers", lambda r: (MockBuilder(), ["l0"])
    )
    monkeypatch.setattr(npu_runner, "_max_blocks_for", lambda r: 64)

    # Bucket 768 is not in buckets=[512] -> buffers lacks it; the pad fallback
    # must not raise KeyError.
    meta0, meta1, meta2 = npu_runner._build_monolithic_dummy_metadata(
        runner, MockBuilder(), npu_runner.p2d_pool, bucket=768, num_reqs=1, beam_width=32
    )
    assert meta0.p2d_pass == 0
    assert meta0.pass0_qlen_list == [768]
    assert meta1.p2d_pass == 1
    assert meta2.p2d_pass == 2
    assert int(npu_runner.p2d_pool.input_ids_0[0]) == 0  # pad fallback


def test_npu_execute_p2d_slot_mapping_shorter_than_batch(monkeypatch: Any) -> None:
    import importlib.machinery
    import sys
    from unittest.mock import MagicMock

    _mock_torch_npu = MagicMock()
    _mock_torch_npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    sys.modules.setdefault("torch_npu", _mock_torch_npu)
    from vllm_gr.v1.attention.backends import beam_attn_npu as ban
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner
    from vllm_gr.v1.worker.prefill_graph_common import P2DBufferPool

    runner = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128, get_hidden_size=lambda: 64),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        device=torch.device("cpu"),
    )
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[512])
    npu_runner.p2d_pool = P2DBufferPool(
        max_batch=2,
        max_bucket=512,
        max_beams=128,
        num_layers=1,
        device=torch.device("cpu"),
        slot_mapping_dtype=torch.int32,
    )
    replayed: list[bool] = []
    entry = SimpleNamespace()
    npu_runner.monolithic_graphs[(512, 1, 32)] = (
        SimpleNamespace(replay=lambda: replayed.append(True)),
        entry,
    )
    monkeypatch.setattr(npu_runner, "_max_blocks_for", lambda r: 64)
    monkeypatch.setattr(ban, "update_monolithic_graph_params", lambda *args, **kwargs: True)

    class _FakeStream:
        def wait_stream(self, other: Any) -> None:
            pass

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: _FakeStream()), raising=False
    )
    monkeypatch.setattr(npu_runner, "_get_update_stream", lambda: _FakeStream())

    res = npu_runner.execute_p2d(
        bucket=512,
        input_ids=torch.zeros(400, dtype=torch.int32),
        positions=torch.zeros(400, dtype=torch.int64),
        num_scheduled_tokens={"r1": 400},
        beam_width=32,
        block_table=torch.zeros((1, 4), dtype=torch.int32),
        slot_mapping=torch.zeros(300, dtype=torch.int32),  # shorter than the batch
    )
    assert res is not None
    assert replayed == [True]
    # Tail beyond the staged slots is padded with -1 (no stale slot writes).
    assert int(npu_runner.p2d_pool.slot_mapping_0[300]) == -1
    assert int(npu_runner.p2d_pool.slot_mapping_0[511]) == -1
    assert int(npu_runner.p2d_pool.slot_mapping_0[299]) == 0


def test_npu_monolithic_dummy_metadata_batch_4(monkeypatch: Any) -> None:
    import importlib.machinery
    import sys
    from unittest.mock import MagicMock

    _mock_torch_npu = MagicMock()
    _mock_torch_npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    sys.modules.setdefault("torch_npu", _mock_torch_npu)
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner
    from vllm_gr.v1.worker.prefill_graph_common import P2DBufferPool

    class MockBuilder:
        num_heads_q = 4
        num_kv_heads = 2
        block_size = 16

    runner = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128, get_hidden_size=lambda: 64),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        device=torch.device("cpu"),
    )
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[1024])
    npu_runner.p2d_pool = P2DBufferPool(
        max_batch=4,
        max_bucket=1024,
        max_beams=128,
        num_layers=1,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        npu_runner, "_get_attn_builder_and_layers", lambda r: (MockBuilder(), ["l0"])
    )
    monkeypatch.setattr(npu_runner, "_max_blocks_for", lambda r: 64)

    meta0, meta1, meta2 = npu_runner._build_monolithic_dummy_metadata(
        runner, MockBuilder(), npu_runner.p2d_pool, bucket=1024, num_reqs=4, beam_width=32
    )
    # Evenly distributed: 1024 // 4 = 256 per request.
    assert meta0.pass0_qlen_list == [256, 512, 768, 1024]
    assert npu_runner.p2d_pool.cu_seqlens_q_0[:5].tolist() == [0, 256, 512, 768, 1024]
    assert npu_runner.p2d_pool.logits_indices_0[:4].tolist() == [255, 511, 767, 1023]
    assert meta1.p2d_pass == 1
    assert meta2.p2d_pass == 2


def test_npu_execute_p2d_batch_4(monkeypatch: Any) -> None:
    import importlib.machinery
    import sys
    from unittest.mock import MagicMock

    _mock_torch_npu = MagicMock()
    _mock_torch_npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    sys.modules.setdefault("torch_npu", _mock_torch_npu)
    from vllm_gr.v1.attention.backends import beam_attn_npu as ban
    from vllm_gr.v1.worker import npu_prefill_graph_runner as npr
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner
    from vllm_gr.v1.worker.prefill_graph_common import P2DBufferPool

    monkeypatch.setattr(npr, "get_p2d_supported_num_reqs", lambda: [1, 2, 4])
    runner = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128, get_hidden_size=lambda: 64),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        device=torch.device("cpu"),
    )
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[512])
    npu_runner.p2d_pool = P2DBufferPool(
        max_batch=4,
        max_bucket=512,
        max_beams=128,
        num_layers=1,
        device=torch.device("cpu"),
        slot_mapping_dtype=torch.int32,
    )
    replayed: list[bool] = []
    npu_runner.monolithic_graphs[(512, 4, 32)] = (
        SimpleNamespace(replay=lambda: replayed.append(True)),
        SimpleNamespace(),
    )
    monkeypatch.setattr(npu_runner, "_max_blocks_for", lambda r: 64)

    captured: dict[str, Any] = {}

    def fake_update(
        stream: Any,
        ent: Any,
        pass0_q: Any,
        pass0_k: Any,
        prefix_q: Any,
        prefix_k: Any,
        bt0: Any,
        bt1: Any,
    ) -> bool:
        captured["pass0_qlen"] = pass0_q
        captured["pass0_kvlen"] = pass0_k
        captured["prefix_kvlen"] = prefix_k
        return True

    monkeypatch.setattr(ban, "update_monolithic_graph_params", fake_update)

    class _FakeStream:
        def wait_stream(self, other: Any) -> None:
            pass

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: _FakeStream()), raising=False
    )
    monkeypatch.setattr(npu_runner, "_get_update_stream", lambda: _FakeStream())

    token_counts = [100, 110, 120, 130]  # total 460 <= 512
    res = npu_runner.execute_p2d(
        bucket=512,
        input_ids=torch.zeros(460, dtype=torch.int32),
        positions=torch.zeros(460, dtype=torch.int64),
        num_scheduled_tokens=token_counts,
        beam_width=32,
        block_table=torch.zeros((4, 4), dtype=torch.int32),
        slot_mapping=torch.zeros(460, dtype=torch.int32),
        computed_tokens=[50, 60, 70, 80],
    )
    assert res is not None
    assert replayed == [True]
    assert captured["pass0_qlen"] == [100, 210, 330, 460]
    assert captured["pass0_kvlen"] == [150, 170, 190, 210]
    assert captured["prefix_kvlen"] == [150] * 32 + [170] * 32 + [190] * 32 + [210] * 32
    assert npu_runner.p2d_pool.positions_1[0].item() == 150
    assert npu_runner.p2d_pool.positions_1[32].item() == 170
    assert npu_runner.p2d_pool.positions_1[96].item() == 210
    assert npu_runner.p2d_pool.positions_2[96].item() == 211


def test_npu_execute_decode_p2d_batch_4(monkeypatch: Any) -> None:
    import importlib.machinery
    import sys
    from unittest.mock import MagicMock

    _mock_torch_npu = MagicMock()
    _mock_torch_npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    sys.modules.setdefault("torch_npu", _mock_torch_npu)
    from vllm_gr.v1.attention.backends import beam_attn_npu as ban
    from vllm_gr.v1.worker import npu_prefill_graph_runner as npr
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner
    from vllm_gr.v1.worker.prefill_graph_common import P2DBufferPool

    monkeypatch.setattr(npr, "get_p2d_supported_num_reqs", lambda: [1, 2, 4])
    runner = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128, get_hidden_size=lambda: 64),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        device=torch.device("cpu"),
    )
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[512])
    npu_runner.p2d_pool = P2DBufferPool(
        max_batch=4,
        max_bucket=512,
        max_beams=128,
        num_layers=1,
        device=torch.device("cpu"),
    )
    replayed: list[bool] = []
    npu_runner.captured_2d_graphs[(4, 32)] = (
        SimpleNamespace(replay=lambda: replayed.append(True)),
        SimpleNamespace(),
    )
    monkeypatch.setattr(npu_runner, "_max_blocks_for", lambda r: 64)

    captured: dict[str, Any] = {}

    def fake_update(stream: Any, ent: Any, qlens: Any, kvlens: Any, bt: Any) -> bool:
        captured["kvlens"] = kvlens
        return True

    monkeypatch.setattr(ban, "update_decode_graph_params", fake_update)
    monkeypatch.setattr(
        npu_runner,
        "_get_prefix_block_table_buf",
        lambda cols: torch.zeros((128, cols), dtype=torch.int32),
    )

    class _FakeStream:
        def wait_stream(self, other: Any) -> None:
            pass

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: _FakeStream()), raising=False
    )
    monkeypatch.setattr(npu_runner, "_get_update_stream", lambda: _FakeStream())

    mock_runner = SimpleNamespace(
        model=SimpleNamespace(compute_logits=lambda hidden: torch.zeros((hidden.shape[0], 176256)))
    )
    token_counts = [100, 110, 120, 130]
    res = npu_runner.execute_decode_p2d(
        runner=mock_runner,
        hidden=torch.zeros((460, 64)),
        block_table=torch.zeros((4, 4), dtype=torch.int32),
        num_scheduled_tokens=token_counts,
        beam_width=32,
        computed_tokens=[50, 60, 70, 80],
    )
    assert res is not None
    assert replayed == [True]
    assert captured["kvlens"] == [150] * 32 + [170] * 32 + [190] * 32 + [210] * 32
    assert npu_runner.p2d_pool.positions_1[0].item() == 150
    assert npu_runner.p2d_pool.positions_1[96].item() == 210
    assert npu_runner.p2d_pool.positions_2[96].item() == 211


def test_npu_prefix_block_table_buf_reallocates_on_expanded_batch(monkeypatch: Any) -> None:
    import importlib.machinery
    import sys
    from unittest.mock import MagicMock

    _mock_torch_npu = MagicMock()
    _mock_torch_npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    sys.modules.setdefault("torch_npu", _mock_torch_npu)
    from vllm_gr.v1.worker import npu_prefill_graph_runner as npr
    from vllm_gr.v1.worker.npu_prefill_graph_runner import NPUPrefillGraphRunner

    monkeypatch.setattr(npr, "get_p2d_supported_num_reqs", lambda: [1, 2])
    runner = SimpleNamespace(
        # The pool capacity derives from --max-logprobs (hard cap 1024).
        model_config=SimpleNamespace(
            max_model_len=128, get_hidden_size=lambda: 64, max_logprobs=128
        ),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        device=torch.device("cpu"),
    )
    npu_runner = NPUPrefillGraphRunner(runner, buckets=[512])

    # Initial call with default batch sizes (max 2 -> 2 * 128 = 256 rows)
    buf1 = npu_runner._get_prefix_block_table_buf(max_blocks=64)
    assert buf1.shape == (256, 64)

    # Expanded batch size (max 4 -> 4 * 128 = 512 rows) must trigger reallocation
    monkeypatch.setattr(npr, "get_p2d_supported_num_reqs", lambda: [1, 2, 4])
    buf2 = npu_runner._get_prefix_block_table_buf(max_blocks=64)
    assert buf2.shape == (512, 64)
    assert buf2 is not buf1

    # Smaller max_blocks with same row count must reuse the buffer (no churn)
    buf3 = npu_runner._get_prefix_block_table_buf(max_blocks=32)
    assert buf3 is buf2
