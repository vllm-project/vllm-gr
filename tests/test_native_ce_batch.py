# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire parity between Python preparation and direct CE-to-Rust submissions."""

import copy
import threading
import time
from types import SimpleNamespace
from typing import Any

import msgspec
import numpy as np
import pytest

from tests.test_native_ce_transport import Peer
from tests.test_native_ce_transport import peer as _peer
from tests.test_native_ce_transport import start_test_transport
from vllm_gr.ops.sliced_logits import set_vocab_slice_config
from vllm_gr.v1.engine import core
from vllm_gr.v1.engine.native_ce_batch import send_engine_outputs
from vllm_gr.v1.engine.native_ce_transport import NativeBridge

peer = _peer


def output(request_id: str, tokens: Any = None, *, new_tokens: Any = ()) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        new_token_ids=list(new_tokens),
        finish_reason="length",
        stop_reason=7,
        prefill_stats=SimpleNamespace(num_cached_tokens=3),
        new_logprobs=(
            SimpleNamespace(
                logprob_token_ids=tokens,
                logprobs=np.full_like(tokens, -1.25, dtype=np.float64),
                sampled_token_ranks=np.arange(max(1, tokens.shape[0]), dtype=np.int32),
            )
            if tokens is not None
            else None
        ),
    )


def engine(groups: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        engine_index=2,
        beamproc_grouped_output_states=copy.deepcopy(groups or {}),
        beam_cache={},
        beam_cache_lock=threading.Lock(),
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "kind",
    [
        "ordinary",
        "grouped_flat",
        "grouped_matrix",
        "grouped_1d",
        "grouped_invalid",
        "mixed",
        "many",
        "noncontiguous",
        "terminal",
    ],
)
def test_direct_batch_matches_python_wire(peer: Peer, kind: str) -> None:
    receiver, _, tx, rx = peer
    set_vocab_slice_config(offset=151669, slice_size=24576)
    groups = {}
    context = {
        "root_request_id": "root",
        "step_index": 1,
        "is_final_step": False,
        "parent_token_ids": [10, 20],
        "prompt_token_count": 2,
    }
    ids = np.asarray([[0, -1, 24575, 24576, 151700, 7]], dtype=np.int32)
    outputs = [output("root", ids, new_tokens=[99])]
    if kind.startswith("grouped") or kind == "mixed":
        ids = np.arange(128 * 64, dtype=np.int32)
        if kind != "grouped_1d":
            ids = ids.reshape(128, 64) if kind == "grouped_matrix" else ids.reshape(1, -1)
        children = [f"root-beam-{i}" for i in range(128 if kind != "grouped_invalid" else 3)]
        groups["group"] = {
            "child_ids": children,
            "trace_context": context,
            "is_final_step": False,
            "root_request_id": "root",
        }
        outputs = [output("group", ids)]
    if kind == "mixed":
        outputs = [
            output("first", np.asarray([[1, 2]], dtype=np.int32), new_tokens=[3]),
            *outputs,
            output("last", None, new_tokens=[4]),
        ]
    elif kind == "many":
        outputs = [output(f"r{i}", ids, new_tokens=[i]) for i in range(40)]
    elif kind == "noncontiguous":
        outputs = [output("root", np.arange(48, dtype=np.int64).reshape(4, 12)[:, ::2])]
    elif kind == "terminal":
        outputs = [output("root", None, new_tokens=[4, 5])]
    contexts = {item.request_id: context for item in outputs}
    batches = {3: SimpleNamespace(outputs=outputs)}
    expected_state, actual_state = engine(groups), engine(groups)
    try:
        expected = core._prepare_beamproc_engine_output_batches(
            expected_state, batches, 2, 99, contexts
        )
        with NativeBridge(tx, rx) as bridge:
            send_engine_outputs(
                actual_state,
                bridge,
                {
                    "outputs": batches,
                    "engine_index": 2,
                    "correlation_base": 99,
                    "output_contexts": contexts,
                },
            )
            actual = receiver.recv_multipart()
            assert actual[0] == expected[0]["frames"][0]
            assert msgspec.msgpack.decode(actual[1]) == msgspec.msgpack.decode(
                expected[0]["frames"][1]
            )
            assert actual[2:] == [bytes(frame) for frame in expected[0]["frames"][2:]]
            bridge.check()
        assert (
            actual_state.beamproc_grouped_output_states
            == expected_state.beamproc_grouped_output_states
        )
    finally:
        set_vocab_slice_config(offset=None, slice_size=None)


def test_direct_submission_order_and_no_python_preparer(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receiver, _, tx, rx = peer
    state = engine()
    state.scheduler = SimpleNamespace()
    state.BEAM_PROC_ENGINECORE_TX_ZMQ_ADDR = tx
    state.BEAM_PROC_ENGINECORE_RX_ZMQ_ADDR = rx
    monkeypatch.setattr(
        core,
        "_prepare_beamproc_engine_output_batches",
        lambda *args: pytest.fail("Python batch preparation was invoked"),
    )
    start_test_transport(state)
    try:
        assert state.beamproc_outbound_queue is None
        assert state.beamproc_io_bp_tx_thread is None
        core._enqueue_beamproc_message(state, b"STATE_UPDATE", {"request_id": "root"})
        tokens = np.asarray([[1, 2, 3]], dtype=np.int32)
        context = {"root_request_id": "root", "is_final_step": False}
        state.beam_cache["root"] = (
            SimpleNamespace(
                sampling_params=SimpleNamespace(),
                prompt_token_ids=[],
                _beamproc_trace_context=context,
            ),
            0,
        )
        core._enqueue_beamproc_engine_outputs(
            state, {0: SimpleNamespace(outputs=[output("root", tokens)])}
        )
        tokens.fill(-999)
        core._enqueue_beamproc_abort(state, ["root"])
        frames = [receiver.recv_multipart() for _ in range(3)]
        assert [message[0] for message in frames] == [
            b"STATE_UPDATE",
            b"ENGINE_OUTPUT_BATCH",
            b"ABORT",
        ]
        assert np.frombuffer(frames[1][2], dtype="<i4").tolist() == [1, 2, 3]
        assert state.beamproc_pending_plan_count == 1
    finally:
        core._stop_beamproc_native_io(state)


def test_native_batch_validation_failure_surfaces(peer: Peer) -> None:
    _, _, tx, rx = peer
    with NativeBridge(tx, rx) as bridge:
        bridge.send_batch(
            {
                "engine_index": 0,
                "client_index": 0,
                "correlation_base": 0,
                "vocab_offset": None,
                "vocab_size": None,
                "outputs": [
                    {
                        "request_id": "root",
                        "child_ids": None,
                        "context": None,
                        "finish_reason": None,
                        "stop_reason": None,
                        "num_cached_tokens": None,
                        "rows": 1,
                        "cols": 2,
                    }
                ],
            },
            [b"", b"", b"", b""],
        )
        deadline = time.monotonic() + 2
        with pytest.raises(RuntimeError, match="numeric dimensions mismatch"):
            while time.monotonic() < deadline:
                bridge.check()
                time.sleep(0.01)
            pytest.fail("Native validation error was not propagated")
