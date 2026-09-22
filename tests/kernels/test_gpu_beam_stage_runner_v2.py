# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 stage-execution hooks: forward/prefill/state-mapping onto the V2 runner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import (
    V2StageExecutionHooks,
    make_v2_stage_hooks,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
# mypy: disable-error-code="no-untyped-def,no-untyped-call"


class _Calls:
    def __init__(self) -> None:
        self.state_machine: list[str] = []


def _v2_worker(forward_result: Any = None) -> SimpleNamespace:
    """A minimal V2-runner surface: compiled model + ExecuteModelState attrs."""

    class Model:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []

        def __call__(self, *, input_ids, positions):
            self.calls.append(("forward", tuple(input_ids.tolist()), tuple(positions.tolist())))
            if forward_result is not None:
                return forward_result
            return torch.zeros_like(input_ids, dtype=torch.float32)

        def compute_logits(self, hidden):
            self.calls.append(("logits", hidden.shape))
            return hidden * 2

    calls = _Calls()
    model = Model()

    def finish(so):
        calls.state_machine.append("finish")

    def free(so):
        calls.state_machine.append("free")

    def add(so):
        calls.state_machine.append("add")

    def update(so):
        calls.state_machine.append("update")

    def apply():
        calls.state_machine.append("apply")

    worker = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(),
        finish_requests=finish,
        free_states=free,
        add_requests=add,
        update_requests=update,
        block_tables=SimpleNamespace(apply_staged_writes=apply),
        req_states=SimpleNamespace(
            num_computed_tokens=SimpleNamespace(
                gpu=torch.zeros(1, dtype=torch.int32, device="cuda")
            )
        ),
        _calls=calls,
    )
    return worker


def test_forward_hidden_routes_through_compiled_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """forward_hidden calls worker.model(input_ids, positions) inside a forward
    context — the V2 route (no _model_forward)."""
    captured: dict[str, Any] = {}

    class _Ctx:
        def __init__(self, *a, **k):
            captured.update(k)
            captured["args"] = a

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import vllm.forward_context as fc

    monkeypatch.setattr(fc, "set_forward_context", _Ctx)

    worker = _v2_worker()
    hooks = make_v2_stage_hooks(worker)
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    positions = torch.tensor([7, 7, 7, 7], dtype=torch.int64, device="cuda")
    out = hooks.forward_hidden(input_ids, positions, {"attn": object()}, {"l": None}, False)

    assert worker.model.calls[0] == ("forward", (1, 2, 3, 4), (7, 7, 7, 7))
    assert captured["num_tokens"] == 4
    assert captured["slot_mapping"] == {"l": None}
    assert torch.equal(out, torch.zeros(4, dtype=torch.float32, device="cuda"))


def test_prefill_logits_extracts_hidden_and_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    """prefill_logits runs native execute, then computes logits from
    hidden_states[logits_indices] (the patched_sample_v2 contract)."""
    worker = _v2_worker()
    hidden = torch.arange(8, dtype=torch.float32, device="cuda").reshape(4, 2)
    input_batch = SimpleNamespace(
        logits_indices=torch.tensor([0, 3, 2, 1], device="cuda"),
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
    )
    worker.execute_model_state = SimpleNamespace(hidden_states=hidden, input_batch=input_batch)

    calls: list[Any] = []

    def original(w, so, *a, **k):
        calls.append(("original", so))
        # V2 contract: last PP rank returns None (falls through).

    hooks = make_v2_stage_hooks(worker)
    logits = hooks.prefill_logits(original, "SO-token")

    assert calls == [("original", "SO-token")]
    # hidden[logits_indices] -> rows (0,3,2,1) of arange(8).reshape(4,2)
    expected = hidden[torch.tensor([0, 3, 2, 1], device="cuda")] * 2
    assert torch.equal(logits, expected)


def test_prefill_logits_rejects_non_none_output() -> None:
    worker = _v2_worker()
    hooks = make_v2_stage_hooks(worker)

    def original(w, so, *a, **k):
        return object()  # must be None for the state contract

    with pytest.raises(RuntimeError, match="hidden states"):
        hooks.prefill_logits(original, None)


def test_prefill_logits_takes_no_device_to_host_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook must not read ``logits_indices`` back to the Host.

    Native ``GPUModelRunner.sample`` indexes ``hidden_states`` unconditionally;
    a Host-side bounds check (``int(indices.max())``) would instead block the GR
    pipeline on the in-flight Prefill forward (measured ~4.7ms per session
    under beam_search_v1).  Any device->host read therefore fails this test.
    """
    worker = _v2_worker()
    hidden = torch.arange(8, dtype=torch.float32, device="cuda").reshape(4, 2)
    input_batch = SimpleNamespace(
        logits_indices=torch.tensor([0, 3, 2, 1], device="cuda"),
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
    )
    worker.execute_model_state = SimpleNamespace(hidden_states=hidden, input_batch=input_batch)
    hooks = make_v2_stage_hooks(worker)

    def forbidden(self, *args, **kwargs):
        raise AssertionError("prefill_logits must not sync a device tensor to the Host")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)

    logits = hooks.prefill_logits(lambda w, so, *a, **k: None, None)
    assert torch.equal(logits, hidden[torch.tensor([0, 3, 2, 1], device="cuda")] * 2)


def test_prefill_stage_fence_is_noop() -> None:
    worker = _v2_worker()
    hooks = make_v2_stage_hooks(worker)
    assert hooks.prefill_stage_fence() is None


def test_consume_native_state_runs_v2_state_machine_in_order() -> None:
    worker = _v2_worker()
    hooks = make_v2_stage_hooks(worker)
    hooks.consume_native_state("SO")
    assert worker._calls.state_machine == ["finish", "free", "add", "update", "apply"]


def test_decode_block_ids_requires_metadata() -> None:
    """V2 keeps no host request state: without scheduler-published kv_blocks
    the hook must fail loudly rather than consult missing state."""
    worker = _v2_worker()
    hooks = make_v2_stage_hooks(worker)
    with pytest.raises(RuntimeError, match="kv_blocks"):
        hooks.decode_block_ids("session-0")


def test_v2_hooks_subclass_contract() -> None:
    worker = _v2_worker()
    hooks = make_v2_stage_hooks(worker)
    assert isinstance(hooks, V2StageExecutionHooks)
    # The base-class worker binding is shared.
    assert hooks.worker is worker


def _native_execute_state_machine_order() -> list[str]:
    """Read the native V2 ``execute_model`` pre-forward state-machine method
    sequence dynamically, past the vllm-gr slim wrapper (R5 drift guard)."""
    import inspect

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    execute = GPUModelRunner.execute_model
    seen: set[Any] = set()
    while hasattr(execute, "__wrapped__") and execute not in seen:
        seen.add(execute)
        execute = execute.__wrapped__
    src = inspect.getsource(execute)

    probes = (
        ("finish_requests", "self.finish_requests(scheduler_output)"),
        ("free_states", "self.free_states(scheduler_output)"),
        ("add_requests", "self.add_requests(scheduler_output)"),
        ("update_requests", "self.update_requests(scheduler_output)"),
        ("apply_staged_writes", "self.block_tables.apply_staged_writes()"),
    )
    positioned = [(name, src.find(probe)) for name, probe in probes]
    return [name for name, pos in sorted(positioned, key=lambda kv: kv[1]) if pos != -1]


def test_consume_native_state_mirrors_native_execute_state_machine() -> None:
    """R5: the V2 hooks' consume_native_state must call the same five methods,
    in the same order, as the native V2 execute_model pre-forward block.  The
    native side is read from source at test time, so a vLLM upgrade that
    reorders/renames that block fails here and flags the drift."""
    native_order = _native_execute_state_machine_order()
    expected = [
        "finish_requests",
        "free_states",
        "add_requests",
        "update_requests",
        "apply_staged_writes",
    ]
    assert native_order == expected, f"native execute_model state machine drifted: {native_order}"

    worker = _v2_worker()
    hooks = make_v2_stage_hooks(worker)
    hooks.consume_native_state("SO")
    label_to_method = {
        "finish": "finish_requests",
        "free": "free_states",
        "add": "add_requests",
        "update": "update_requests",
        "apply": "apply_staged_writes",
    }
    observed = [label_to_method[label] for label in worker._calls.state_machine]
    assert observed == expected, f"consume_native_state drifted: {observed}"
