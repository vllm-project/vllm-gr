# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the offline Rust BeamProc wiring (no GPU/model required)."""

import time

import pytest

from vllm_gr.entrypoints.gr import GRLLM
from vllm_gr.v1.engine.core_client import SyncBeamProcReceiver


class _FakeTokenizer:
    eos_token_id = 1

    def batch_decode(self, sequences):
        return [str(seq) for seq in sequences]


def _make_final_plan():
    return {
        "version": 3,
        "root_request_id": "root-1",
        "beam_width": 2,
        "num_processed_tokens": 12,
        "final_token_ids": [[5, 6], [7, 8]],
        "final_scores": [-0.1, -0.2],
        "final_finish_reasons": ["stop", "length"],
        "final_stop_token_ids": [1, None],
        "final_token_ids_include_prompt": False,
    }


def test_sync_receiver_dispatch_and_await():
    receiver = SyncBeamProcReceiver("ipc:///tmp/unused-offline-beamproc.sock")
    receiver._dispatch(
        {
            "root_request_id": "root-1",
            "beam_width": 4,
            "final_token_ids": [[1, 2, 3]],
        }
    )
    plan = receiver.await_plan("root-1", 4, timeout_s=0.0)
    assert plan is not None
    assert plan["final_token_ids"] == [[1, 2, 3]]
    assert receiver.await_plan("root-1", 4, timeout_s=0.0) is None
    receiver.shutdown()


def test_sync_receiver_timeout():
    receiver = SyncBeamProcReceiver("ipc:///tmp/unused-offline-beamproc.sock")
    start = time.monotonic()
    assert receiver.await_plan("missing", 4, timeout_s=0.05) is None
    assert time.monotonic() - start < 1.0
    receiver.shutdown()


def test_build_beam_output_from_plan_no_logprobs():
    grllm = GRLLM.__new__(GRLLM)
    final_plan = _make_final_plan()
    prompt_cache = {"text": "<prompt>", "len": 3, "tokens": [10, 11, 12]}
    tokenizer = _FakeTokenizer()

    output = grllm._build_beam_output_from_plan(
        final_plan=final_plan,
        orig_prompt={"type": "token", "prompt_token_ids": [10, 11, 12], "prompt": "<prompt>"},
        prompt_cache=prompt_cache,
        prompt_len=3,
        tokenizer=tokenizer,
        beam_width=2,
        sid_begin_token_id=None,
        sid_end_token_id=None,
        ignore_eos=False,
        include_stop_str_in_output=False,
        return_logprobs=False,
        lora_request=None,
    )

    assert len(output.sequences) == 2
    # final_token_ids_include_prompt=False, so response tokens == final tokens.
    assert output.sequences[0].tokens == [10, 11, 12, 5, 6]
    assert output.sequences[1].tokens == [10, 11, 12, 7, 8]
    assert output.sequences[0].cum_logprob == -0.1
    assert output.sequences[0].finish_reason == "stop"
    assert output.sequences[1].finish_reason == "length"
    assert output.processed_tokens == 12
