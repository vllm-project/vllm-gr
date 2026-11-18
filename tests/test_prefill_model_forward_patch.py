# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Unit tests for the pure-prefill graph interception (_model_forward patch)."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker import gpu_model_runner_patch as gpu_patch_mod


def test_gpu_model_forward_intercepts_pure_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    wrapped = gpu_patch_mod.make_patched_model_forward_gpu(original)

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


@pytest.mark.l1  # type: ignore[misc]
def test_gpu_model_forward_dense_prefill_replays_only_fresh_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.v1.attention.backends.beam_dense_prefill import ENV_DENSE_PREFILL

    output_buf = torch.zeros(4, 16)

    class FakePrefillRunner:
        def match_bucket(self: Any, total_tokens: Any) -> Any:
            return 128

        def has_captured_graph(self: Any, bucket: Any, num_reqs: Any) -> Any:
            return True

        def execute_model_prefill(self: Any, *args: Any, **kwargs: Any) -> Any:
            return output_buf

    monkeypatch.setattr(gpu_patch_mod, "is_prefill_graph_enabled", lambda: True)
    monkeypatch.setattr(gpu_patch_mod, "is_prefill_phase_batch", lambda self, so: True)
    monkeypatch.setattr(gpu_patch_mod, "scheduled_requests_are_fresh", lambda self, so: so.fresh)
    monkeypatch.setattr(
        gpu_patch_mod, "get_prefill_graph_runner", lambda runner: FakePrefillRunner()
    )
    meta = SimpleNamespace(slot_mapping=torch.zeros(64, dtype=torch.int64))
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"l": meta}),
    )
    wrapped = gpu_patch_mod.make_patched_model_forward_gpu(lambda self, *args, **kwargs: "orig_out")

    def run(fresh: bool) -> Any:
        scheduler_output = SimpleNamespace(fresh=fresh, num_scheduled_tokens={"r1": 30})
        runner = SimpleNamespace(
            _current_scheduler_output=scheduler_output,
            input_batch=SimpleNamespace(
                block_table=[
                    SimpleNamespace(
                        slot_mapping=SimpleNamespace(gpu=torch.zeros(64, dtype=torch.int64))
                    )
                ]
            ),
        )
        return wrapped(runner, input_ids="ids", positions="pos")

    # Without dense Prefill, a cached-Prefill suffix may replay the graph.
    monkeypatch.delenv(ENV_DENSE_PREFILL, raising=False)
    assert run(fresh=False) is output_buf
    # With dense Prefill, only fresh prompts replay; others run the original forward.
    monkeypatch.setenv(ENV_DENSE_PREFILL, "cudnn")
    assert run(fresh=False) == "orig_out"
    assert run(fresh=True) is output_buf


pytestmark = pytest.mark.cpu_test
