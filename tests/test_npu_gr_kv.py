# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for persistent NPU KV reorder.

Injected CPU operators model tensor semantics and call ABI only.  These tests
are not NPU kernel, stream, data-residency, or performance acceptance.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker.npu_gr_decision import (
    NPUDecisionStep,
    advance_fused_decision,
    bootstrap_from_constrained_prefill,
)
from vllm_gr.v1.worker.npu_gr_kv import NPUKVReorder


class _Session:
    def __init__(self, *, width: int = 8, layers: int = 2, steps: int = 3) -> None:
        self.session_id = "gr"
        self.slot = 0
        self._state = SimpleNamespace(
            is_active=True,
            beam_width=width,
            decode_step=0,
        )
        pools = []
        for layer in range(layers):
            key = torch.empty((1, width, 1, steps, 2), dtype=torch.bfloat16)
            value = torch.empty_like(key)
            for beam in range(width):
                key[0, beam].fill_(layer * 1000 + beam * 10 + 1)
                value[0, beam].fill_(layer * 1000 + beam * 10 + 2)
            pools.append(SimpleNamespace(unshared_key=key, unshared_value=value))
        decision = SimpleNamespace(
            sequence=torch.zeros((1, width, steps), dtype=torch.int32),
            beam_scores=torch.zeros((1, width), dtype=torch.float32),
            group_token_num=torch.zeros((1, width), dtype=torch.int32),
            worker_decode_steps=0,
        )
        self.context = SimpleNamespace(
            pools=pools,
            aux_key_buf=torch.zeros((layers, width, 1, steps, 2), dtype=torch.bfloat16),
            aux_value_buf=torch.zeros((layers, width, 1, steps, 2), dtype=torch.bfloat16),
            xllm_block_table_buf=torch.tensor([0], dtype=torch.int32),
            decision=decision,
        )

    @property
    def is_active(self) -> bool:
        return self._state.is_active

    @property
    def beam_width(self) -> int:
        return self._state.beam_width

    @property
    def decode_step(self) -> int:
        return self._state.decode_step


class _DecisionOps:
    parents = torch.tensor([[0, 0, 1, 3, 3, 3, 6, 7]], dtype=torch.int32)
    child_columns = torch.tensor([0, 1, 0, 0, 1, 2, 0, 0], dtype=torch.int64)
    prefix = torch.tensor([[2, 3, 3, 6, 6, 6, 7, 8]], dtype=torch.int32)

    @staticmethod
    def is_available() -> bool:
        return True

    def beam_search_group(
        self,
        cumulative: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_scores: torch.Tensor,
        sequence: torch.Tensor,
        step: int,
        width: int,
    ) -> tuple[torch.Tensor, ...]:
        parents = self.parents.to(sequence.device)
        columns = self.child_columns.to(sequence.device)
        rows = parents.view(-1).to(torch.int64)
        selected_tokens = candidate_tokens[rows, columns].view(1, width)
        selected_scores = (cumulative.view(-1)[rows] + candidate_scores[rows, columns]).view(
            1, width
        )
        selected_sequence = sequence.clone()
        selected_sequence[:, :, :step] = sequence[:, rows, :step]
        selected_sequence[:, :, step] = selected_tokens
        return (
            selected_tokens.to(torch.int32),
            parents,
            selected_scores.to(torch.float32),
            self.prefix.to(sequence.device),
            selected_sequence,
        )

    def beam_search_rec_final_select(self, *_args: Any) -> Any:
        raise AssertionError("non-final CPU golden must not call FinalSelect")


def _actual_nonfinal_decision(session: _Session) -> NPUDecisionStep:
    width = session.beam_width
    ops = _DecisionOps()
    bootstrap_from_constrained_prefill(
        session,
        torch.arange(10, 10 + width, dtype=torch.int32).view(1, width),
        -torch.arange(1, width + 1, dtype=torch.float32).view(1, width),
        expected_width=width,
        ops=ops,
    )
    candidates = torch.arange(width * width, dtype=torch.int32).view(width, width) + 100
    local_scores = -torch.arange(1, width * width + 1, dtype=torch.float32).view(width, width) / 100
    return advance_fused_decision(
        session,
        candidates,
        local_scores,
        expected_device_step=1,
        is_last_step=False,
        result_width=width,
        ops=ops,
    )


def _fanout(
    parents: torch.Tensor,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
) -> None:
    indices = parents.to(torch.int64)
    for tensor in (*keys, *values):
        original = tensor.clone()
        tensor[0].copy_(original[0].index_select(0, indices))


def test_actual_group_decision_drives_exact_xllm_abi_and_multilayer_fanout() -> None:
    session = _Session()
    decision = _actual_nonfinal_decision(session)
    before_keys = [pool.unshared_key.clone() for pool in session.context.pools]
    before_values = [pool.unshared_value.clone() for pool in session.context.pools]
    calls: list[tuple[Any, ...]] = []

    def select(*args: Any) -> None:
        calls.append(args)
        _fanout(args[0], args[1], args[2])

    reorder = NPUKVReorder(
        session,
        mode="xllm",
        xllm_ops=SimpleNamespace(select_unshared_kv=select),
    )
    reorder.reorder(session, decision, expected_device_step=1, is_final=False)

    assert len(calls) == 1
    parents, keys, values, block_table, prefix, step, width, layers = calls[0]
    assert parents.data_ptr() == decision.reorder_parents.data_ptr()
    assert prefix.data_ptr() == decision.reorder_prefix.data_ptr()
    assert all(
        actual is pool.unshared_key
        for actual, pool in zip(keys, session.context.pools, strict=True)
    )
    assert all(
        actual is pool.unshared_value
        for actual, pool in zip(values, session.context.pools, strict=True)
    )
    assert block_table.data_ptr() == session.context.xllm_block_table_buf.data_ptr()
    assert (step, width, layers) == (1, 8, 2)
    assert session.decode_step == 1
    expected_parents = decision.reorder_parents.view(-1).to(torch.int64)
    for layer, pool in enumerate(session.context.pools):
        assert torch.equal(
            pool.unshared_key[0],
            before_keys[layer][0].index_select(0, expected_parents),
        )
        assert torch.equal(
            pool.unshared_value[0],
            before_values[layer][0].index_select(0, expected_parents),
        )
        assert torch.equal(pool.unshared_key[0, 0], pool.unshared_key[0, 1])


def test_cascade_uses_fixed_per_layer_aux_and_never_materializes_host_state() -> None:
    session = _Session()
    decision = _actual_nonfinal_decision(session)
    before_keys = [pool.unshared_key.clone() for pool in session.context.pools]
    before_values = [pool.unshared_value.clone() for pool in session.context.pools]
    key_aux_ptrs = tuple(session.context.aux_key_buf[layer].data_ptr() for layer in range(2))
    value_aux_ptrs = tuple(session.context.aux_value_buf[layer].data_ptr() for layer in range(2))
    calls: list[tuple[Any, ...]] = []

    def select(
        parents: torch.Tensor,
        slot: int,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        decode_step: int,
        *,
        aux_key_bufs: list[torch.Tensor],
        aux_value_bufs: list[torch.Tensor],
    ) -> None:
        calls.append((parents, slot, keys, values, decode_step, aux_key_bufs, aux_value_bufs))
        indices = parents.to(torch.int64)
        for layer, tensor in enumerate(keys):
            aux_key_bufs[layer].copy_(tensor[0])
            tensor[0].copy_(aux_key_bufs[layer].index_select(0, indices))
        for layer, tensor in enumerate(values):
            aux_value_bufs[layer].copy_(tensor[0])
            tensor[0].copy_(aux_value_bufs[layer].index_select(0, indices))

    reorder = NPUKVReorder(session, mode="cascade", cascade_op=select)
    reorder.reorder(session, decision, expected_device_step=1, is_final=False)

    assert len(calls) == 1
    call = calls[0]
    assert call[0].data_ptr() == decision.reorder_parents.data_ptr()
    assert call[1] == 0
    assert call[4] == 0
    assert tuple(tensor.data_ptr() for tensor in call[5]) == key_aux_ptrs
    assert tuple(tensor.data_ptr() for tensor in call[6]) == value_aux_ptrs
    assert session.decode_step == 1
    expected_parents = decision.reorder_parents.view(-1).to(torch.int64)
    for layer, pool in enumerate(session.context.pools):
        assert torch.equal(
            pool.unshared_key[0],
            before_keys[layer][0].index_select(0, expected_parents),
        )
        assert torch.equal(
            pool.unshared_value[0],
            before_values[layer][0].index_select(0, expected_parents),
        )
        assert torch.equal(pool.unshared_key[0, 0], pool.unshared_key[0, 1])
    source = textwrap.dedent(inspect.getsource(NPUKVReorder.reorder))
    attributes = {
        node.attr for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Attribute)
    }
    assert attributes.isdisjoint({"cpu", "tolist", "item", "numpy"})


def _manual_final(session: _Session, *, expected: int = 1) -> NPUDecisionStep:
    session.context.decision.worker_decode_steps = expected + 1
    return NPUDecisionStep(
        device_step=expected + 1,
        token_ids=session.context.decision.sequence[:, :, expected],
        parent_ids=torch.zeros((1, session.beam_width), dtype=torch.int32),
        scores=session.context.decision.beam_scores,
        sequence=session.context.decision.sequence,
    )


@pytest.mark.parametrize("mode", ["xllm", "cascade"])
def test_final_decision_skips_reorder_and_does_not_advance_session(mode: str) -> None:
    session = _Session()
    calls = 0

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        nonlocal calls
        calls += 1

    kwargs = (
        {"xllm_ops": SimpleNamespace(select_unshared_kv=forbidden)}
        if mode == "xllm"
        else {"cascade_op": forbidden}
    )
    reorder = NPUKVReorder(session, mode=mode, **kwargs)
    decision = _manual_final(session)
    reorder.reorder(session, decision, expected_device_step=1, is_final=True)

    assert calls == 0
    assert session.decode_step == 0
    with pytest.raises(RuntimeError, match="final decision"):
        reorder.reorder(session, decision, expected_device_step=1, is_final=True)


def test_half_failed_kernel_poison_is_not_retried_and_preserves_error_identity() -> None:
    session = _Session()
    decision = _actual_nonfinal_decision(session)
    error = OSError("partial xllm failure")
    calls = 0
    untouched_second = session.context.pools[1].unshared_key.clone()

    def half_fail(
        parents: torch.Tensor,
        keys: list[torch.Tensor],
        _values: list[torch.Tensor],
        *_args: Any,
    ) -> None:
        nonlocal calls
        calls += 1
        original = keys[0].clone()
        keys[0][0].copy_(original[0].index_select(0, parents.to(torch.int64)))
        raise error

    reorder = NPUKVReorder(
        session,
        mode="xllm",
        xllm_ops=SimpleNamespace(select_unshared_kv=half_fail),
    )
    with pytest.raises(OSError) as caught:
        reorder.reorder(session, decision, expected_device_step=1, is_final=False)
    assert caught.value is error
    assert reorder.poisoned
    assert calls == 1
    assert session.decode_step == 0
    assert torch.equal(session.context.pools[1].unshared_key, untouched_second)

    with pytest.raises(RuntimeError, match="poisoned") as retry:
        reorder.reorder(session, decision, expected_device_step=1, is_final=False)
    assert retry.value.__cause__ is error
    assert calls == 1


def test_release_is_idempotent_but_foreign_or_post_release_reorder_is_rejected() -> None:
    session = _Session()
    decision = _actual_nonfinal_decision(session)
    reorder = NPUKVReorder(
        session,
        mode="xllm",
        xllm_ops=SimpleNamespace(select_unshared_kv=lambda *_args: None),
    )
    foreign = _Session()
    with pytest.raises(RuntimeError, match="foreign"):
        reorder.release(foreign)
    assert reorder.release(session)
    assert not reorder.release(session)
    with pytest.raises(RuntimeError, match="released"):
        reorder.reorder(session, decision, expected_device_step=1, is_final=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "canonical-sequence",
        "canonical-scores",
        "canonical-prefix",
        "pool-resize",
        "pool-set-storage",
        "session-slot",
        "session-width",
    ],
)
def test_captured_tensor_and_session_layout_drift_fails_before_kernel(
    mutation: str,
) -> None:
    session = _Session()
    decision = _actual_nonfinal_decision(session)
    calls = 0

    def forbidden(*_args: Any) -> None:
        nonlocal calls
        calls += 1

    reorder = NPUKVReorder(
        session,
        mode="xllm",
        xllm_ops=SimpleNamespace(select_unshared_kv=forbidden),
    )
    if mutation == "canonical-sequence":
        session.context.decision.sequence = session.context.decision.sequence.clone()
    elif mutation == "canonical-scores":
        session.context.decision.beam_scores = session.context.decision.beam_scores.clone()
    elif mutation == "canonical-prefix":
        session.context.decision.group_token_num = session.context.decision.group_token_num.clone()
    elif mutation == "pool-resize":
        session.context.pools[0].unshared_key.resize_(1, 8, 1, 2, 2)
    elif mutation == "pool-set-storage":
        tensor = session.context.pools[0].unshared_key
        tensor.set_(torch.empty_like(tensor))
    elif mutation == "session-slot":
        session.slot = 1
    elif mutation == "session-width":
        session._state.beam_width = 7

    with pytest.raises(RuntimeError):
        reorder.reorder(session, decision, expected_device_step=1, is_final=False)
    assert calls == 0
    assert session.decode_step == 0


def test_device_step_beyond_canonical_capacity_fails_before_kernel() -> None:
    session = _Session(steps=3)
    session._state.decode_step = 2
    calls = 0

    def forbidden(*_args: Any) -> None:
        nonlocal calls
        calls += 1

    reorder = NPUKVReorder(
        session,
        mode="xllm",
        xllm_ops=SimpleNamespace(select_unshared_kv=forbidden),
    )
    decision = NPUDecisionStep(
        device_step=4,
        token_ids=torch.zeros((1, 8), dtype=torch.int32),
        parent_ids=torch.zeros((1, 8), dtype=torch.int32),
        scores=session.context.decision.beam_scores,
        sequence=session.context.decision.sequence,
        reorder_parents=torch.zeros((1, 8), dtype=torch.int32),
        reorder_prefix=torch.arange(1, 9, dtype=torch.int32).view(1, 8),
    )

    with pytest.raises(RuntimeError, match="exceeds canonical capacity"):
        reorder.reorder(session, decision, expected_device_step=3, is_final=False)
    assert calls == 0
    assert session.decode_step == 2


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("same-decision", "already consumed"),
        ("stale-step", "stale or out of order"),
        ("foreign", "foreign"),
        ("epoch", "epoch is stale"),
        ("context", "context identity"),
        ("parents-shape", "parents geometry"),
        ("prefix-shape", "prefix geometry"),
        ("parent-identity", "parent payload identity"),
    ],
)
def test_reorder_rejects_repeated_stale_foreign_or_changed_layout(
    failure: str,
    match: str,
) -> None:
    session = _Session()
    decision = _actual_nonfinal_decision(session)
    calls = 0

    def select(*_args: Any) -> None:
        nonlocal calls
        calls += 1

    reorder = NPUKVReorder(
        session,
        mode="xllm",
        xllm_ops=SimpleNamespace(select_unshared_kv=select),
    )
    target = session
    expected = 1
    if failure == "same-decision":
        reorder.reorder(session, decision, expected_device_step=1, is_final=False)
        session.context.decision.worker_decode_steps = 2
    elif failure == "stale-step":
        expected = 2
    elif failure == "foreign":
        target = _Session()
    elif failure == "epoch":
        session._state = SimpleNamespace(is_active=True, beam_width=8, decode_step=0)
    elif failure == "context":
        session.context = SimpleNamespace(pools=[])
    elif failure == "parents-shape":
        wrong = torch.zeros((8,), dtype=torch.int32)
        decision = NPUDecisionStep(
            decision.device_step,
            decision.token_ids,
            wrong,
            decision.scores,
            decision.sequence,
            wrong,
            decision.reorder_prefix,
        )
    elif failure == "prefix-shape":
        decision = NPUDecisionStep(
            decision.device_step,
            decision.token_ids,
            decision.parent_ids,
            decision.scores,
            decision.sequence,
            decision.reorder_parents,
            torch.zeros((8,), dtype=torch.int32),
        )
    elif failure == "parent-identity":
        decision = NPUDecisionStep(
            decision.device_step,
            decision.token_ids,
            decision.parent_ids.clone(),
            decision.scores,
            decision.sequence,
            decision.reorder_parents,
            decision.reorder_prefix,
        )

    with pytest.raises(RuntimeError, match=match):
        reorder.reorder(target, decision, expected_device_step=expected, is_final=False)
    assert calls == (1 if failure == "same-decision" else 0)


@pytest.mark.parametrize(
    ("mutation", "mode", "match"),
    [
        (lambda s: setattr(s, "slot", 1), "xllm", "slot zero"),
        (lambda s: setattr(s._state, "is_active", False), "xllm", "active"),
        (lambda s: setattr(s.context, "pools", []), "xllm", "pools"),
        (
            lambda s: setattr(
                s.context.pools[1],
                "unshared_key",
                torch.zeros((1, 7, 1, 3, 2)),
            ),
            "xllm",
            "geometry|capacity|layouts",
        ),
        (
            lambda s: setattr(
                s.context,
                "aux_key_buf",
                torch.zeros((1, 8, 1, 3, 2), dtype=torch.bfloat16),
            ),
            "cascade",
            "aux geometry",
        ),
        (
            lambda s: setattr(
                s.context,
                "xllm_block_table_buf",
                torch.zeros((1,), dtype=torch.int64),
            ),
            "xllm",
            "block table geometry",
        ),
    ],
)
def test_constructor_rejects_session_or_fixed_layout_changes(
    mutation: Any,
    mode: str,
    match: str,
) -> None:
    session = _Session()
    mutation(session)
    kwargs = (
        {"xllm_ops": SimpleNamespace(select_unshared_kv=lambda *_args: None)}
        if mode == "xllm"
        else {"cascade_op": lambda *_args, **_kwargs: None}
    )
    with pytest.raises(RuntimeError, match=match):
        NPUKVReorder(session, mode=mode, **kwargs)


@pytest.mark.parametrize("zero_axis", [2, 3, 4])
def test_constructor_rejects_zero_head_time_or_dimension_pool(zero_axis: int) -> None:
    session = _Session()
    shape = [1, 8, 1, 3, 2]
    shape[zero_axis] = 0
    for pool in session.context.pools:
        pool.unshared_key = torch.empty(tuple(shape), dtype=torch.bfloat16)
        pool.unshared_value = torch.empty(tuple(shape), dtype=torch.bfloat16)
    session.context.aux_key_buf = torch.empty((2, *shape[1:]), dtype=torch.bfloat16)
    session.context.aux_value_buf = torch.empty((2, *shape[1:]), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="dimensions|capacity"):
        NPUKVReorder(
            session,
            mode="xllm",
            xllm_ops=SimpleNamespace(select_unshared_kv=lambda *_args: None),
        )


@pytest.mark.parametrize(
    ("mode", "kwargs", "match"),
    [
        ("xllm", {"xllm_ops": object()}, "unavailable"),
        ("cascade", {"cascade_op": object()}, "unavailable"),
    ],
)
def test_missing_injected_operator_fails_fast(
    mode: str,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        NPUKVReorder(_Session(), mode=mode, **kwargs)
