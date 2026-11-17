# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Host contracts for request-local NPU Trie routing outside model graphs."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm_gr.constraints.backend import ConstraintMetadata, ConstraintOutput
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.worker.npu.npu_beam_stage_adapter import NPUEagerStagePlan
from vllm_gr.v1.worker.npu.npu_beam_stage_runner import NPUEagerBeamStageRunner

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def _plan(stage: int = 0, widths: tuple[int, ...] | None = None) -> NPUEagerStagePlan:
    execution = BeamBatchExecutionView(
        session_id="batch",
        dispatch_id=stage,
        stage_index=stage,
        kind="prefill" if stage == 0 else "decode",
        decode_step=None if stage == 0 else stage - 1,
        owner_generation=0,
        request_ids=("first", "second"),
        num_prompt_tokens_by_request=(3, 5),
        beam_width=4,
        produces_output=True,
        dynamic_beam_widths=widths,
    )
    return NPUEagerStagePlan(execution, (3, 5), 8, (2, 7), (0, 1), None)


def _request(roots: tuple[Any, ...] = (0, 1)) -> Any:
    return SimpleNamespace(
        items=tuple(
            SimpleNamespace(
                item_index=index,
                trie_id=root,
                request=SimpleNamespace(request_id=name),
            )
            for index, (name, root) in enumerate(zip(("first", "second"), roots, strict=True))
        )
    )


def _runner(trie_count: int = 2) -> NPUEagerBeamStageRunner:
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker = SimpleNamespace(
        constraint_backend=SimpleNamespace(
            info=SimpleNamespace(table_metadata=SimpleNamespace(trie_count=trie_count))
        )
    )
    stage.session_request = None
    stage._session_trie_ids = None
    stage._trie_ids_by_width = {}
    stage.previous = None
    stage.pending_plan = None
    stage.execution_error = None
    stage.closed = False
    stage.attention_batch = None
    return stage


@pytest.mark.parametrize("bad_root", [-1, True, 2, 2**32 - 1])
def test_invalid_worker_selector_rejects_before_mutation_and_can_retry(
    monkeypatch: pytest.MonkeyPatch, bad_root: Any
) -> None:
    stage = _runner()
    plan = _plan()
    monkeypatch.setattr(NPUEagerStagePlan, "from_scheduler_output", lambda output: plan)
    metadata = SimpleNamespace(session_init=_request((0, bad_root)))
    scheduled = SimpleNamespace(gr_stage_metadata={"batch": metadata})
    original = Mock(return_value=SimpleNamespace(logits=torch.zeros(2, 16)))
    before = dict(vars(stage))
    worker_before = dict(vars(stage.worker))

    with pytest.raises(ValueError, match="trie_id is outside"):
        stage.execute(scheduled, original)

    assert vars(stage) == before
    assert vars(stage.worker) == worker_before
    original.assert_not_called()
    metadata.session_init = _request((1, 0))
    stage.execute(scheduled, original)
    original.assert_called_once()
    assert stage._session_trie_ids == (1, 0)
    assert stage.pending_plan is plan


@pytest.mark.parametrize("fault", ["missing", "reordered", "missing_member"])
def test_multi_trie_requires_the_complete_original_request_order(fault: str) -> None:
    stage, request = _runner(), _request()
    if fault == "missing":
        request = None
    elif fault == "reordered":
        request.items = tuple(reversed(request.items))
    else:
        request.items = request.items[:1]
    with pytest.raises(ValueError, match="initial batch request|request IDs"):
        stage._validate_session_trie_ids(_plan(), request)
    assert stage.session_request is None and stage._session_trie_ids is None


@pytest.mark.parametrize("widths", [None, (4, 2, 1), (1, 2, 4), (4, 4, 4)])
@pytest.mark.parametrize("roots", [(0, 1), (1, 0), (1, 1)])
def test_request_selectors_follow_input_rows_across_widths_and_parent_changes(
    widths: tuple[int, ...] | None, roots: tuple[int, int]
) -> None:
    stage = _runner()
    request = _request(roots)
    stage._session_trie_ids = stage._validate_session_trie_ids(_plan(0, widths), request)
    # Selector ownership is detached from the request object's later lifetime.
    request.items[0].trie_id = 0 if roots[0] else 1
    for index in range(3):
        plan = _plan(index, widths)
        execution = plan.execution
        actual = stage._constraint_trie_ids(plan)
        expected = tuple(roots[row] for row in execution.row_request_indices)
        assert actual == expected
        assert stage._constraint_trie_ids(plan) is actual
        if index:
            # Duplicate and reverse parents within each logical request. The
            # next request's Trie is never inherited from a global parent row.
            parents = tuple(
                row * execution.input_width
                + (execution.input_width - 1 - beam % execution.input_width)
                for row in range(2)
                for beam in range(execution.output_width)
            )
            execution.localize_global_parent_rows(parents)
            assert actual is not None
            expected_children = tuple(
                roots[row] for row in range(2) for _ in range(execution.output_width)
            )
            assert tuple(actual[parent] for parent in parents) == expected_children
            with pytest.raises(ValueError, match="crosses a logical request"):
                execution.localize_global_parent_rows((execution.input_width,) + parents[1:])


def test_request_order_cannot_change_after_selector_admission() -> None:
    stage = _runner()
    stage.previous = _plan().execution
    stage.latest_token_ids = torch.zeros(2, 4, dtype=torch.int32)
    stage._session_trie_ids = (1, 0)
    plan = _plan(stage=1)
    plan = replace(plan, execution=replace(plan.execution, request_ids=("second", "first")))
    with pytest.raises(ValueError, match="logical Beam batch identity changed"):
        stage._validate_sequence(plan)
    assert stage._session_trie_ids == (1, 0) and not stage._trie_ids_by_width


def test_one_trie_and_catalog_free_sessions_keep_selector_free_routing() -> None:
    stage = _runner(trie_count=1)
    assert stage._validate_session_trie_ids(_plan(), _request((0, 0))) is None
    assert stage._validate_session_trie_ids(_plan(), None) is None
    stage.worker.constraint_backend = None
    assert stage._validate_session_trie_ids(_plan(), _request((0, 0))) is None
    with pytest.raises(ValueError, match="trie_id is outside"):
        stage._validate_session_trie_ids(_plan(), _request((0, 1)))
    assert stage._constraint_trie_ids(_plan()) is None
    assert not stage._trie_ids_by_width


@pytest.mark.parametrize("max_rows", [3, 5, 32])
@pytest.mark.parametrize("scalar_temperature", [False, True])
def test_chunked_host_routing_preserves_selectors_prefixes_and_row_order_without_sync(
    monkeypatch: pytest.MonkeyPatch, max_rows: int, scalar_temperature: bool
) -> None:
    stage = _runner()
    stage._session_trie_ids = (1, 0)
    plan = _plan(stage=1)
    roots = stage._constraint_trie_ids(plan)
    assert roots == (1, 1, 1, 1, 0, 0, 0, 0)
    rows = plan.execution.num_execution_rows
    logits = torch.arange(rows, dtype=torch.float32)[:, None].expand(-1, 4).contiguous()
    prefixes = torch.arange(rows * 3, dtype=torch.int32).reshape(rows, 3)
    temperatures = torch.arange(1 if scalar_temperature else rows, dtype=torch.float32)
    metadata = ConstraintMetadata(prefixes, temperatures, 1, 2)
    calls: list[tuple[torch.Tensor, ConstraintMetadata, tuple[int, ...]]] = []

    def select(logits: torch.Tensor, metadata: ConstraintMetadata, roots: tuple[int, ...]) -> Any:
        calls.append((logits, metadata, roots))
        assert metadata.trie_ids is None
        token_ids = torch.tensor(roots, dtype=torch.int32)[:, None].expand(-1, 2).contiguous()
        return ConstraintOutput(
            token_ids, logits[:, :2].contiguous(), torch.ones_like(token_ids, dtype=torch.bool)
        )

    backend = SimpleNamespace(info=SimpleNamespace(max_rows=max_rows), select_for_rows=select)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("NPU host routing must not transfer or read device selectors")

    with monkeypatch.context() as guard:
        for name in ("cpu", "item", "tolist", "nonzero"):
            guard.setattr(torch.Tensor, name, forbidden)
        actual = stage._select_constraint_candidates(backend, logits, metadata, trie_ids=roots)

    assert tuple(root for _, _, part in calls for root in part) == roots
    torch.testing.assert_close(torch.cat([m.prefix_token_ids for _, m, _ in calls]), prefixes)
    if scalar_temperature:
        assert all(m.temperatures is temperatures for _, m, _ in calls)
    else:
        torch.testing.assert_close(torch.cat([m.temperatures for _, m, _ in calls]), temperatures)
    assert actual.token_ids[:, 0].tolist() == list(roots)
    torch.testing.assert_close(actual.logprobs, logits[:, :2])
    assert all(m.current_step == 1 and m.top_k == 2 for _, m, _ in calls)


def test_host_routing_fails_closed_if_backend_cannot_consume_selectors() -> None:
    stage = _runner()
    logits = torch.zeros(2, 4)
    metadata = ConstraintMetadata(torch.zeros(2, 3, dtype=torch.int32), torch.ones(1), 0, 2)
    fallback = Mock()
    backend = SimpleNamespace(info=SimpleNamespace(max_rows=4), select=fallback)
    with pytest.raises(RuntimeError, match="host-row routing"):
        stage._select_constraint_candidates(backend, logits, metadata, trie_ids=(0, 1))
    with pytest.raises(RuntimeError, match="physical candidate rows"):
        stage._select_constraint_candidates(backend, logits, metadata, trie_ids=(0,))
    fallback.assert_not_called()


def test_retirement_releases_selectors_and_next_owner_uses_its_new_trie_order() -> None:
    stage = _runner()
    stage._session_trie_ids = (0, 1)
    stage._constraint_trie_ids(_plan(stage=1))
    assert stage._trie_ids_by_width
    stage._release()
    assert stage.closed and stage._session_trie_ids is None and not stage._trie_ids_by_width
    new = _runner()
    new._session_trie_ids = new._validate_session_trie_ids(_plan(), _request((1, 0)))
    assert new._constraint_trie_ids(_plan()) == (1, 0)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_npu_startup_publishes_the_loaded_backend_count_not_a_config_guess(count: int) -> None:
    from vllm_gr.v1.engine.core import loaded_constraint_trie_count
    from vllm_gr.v1.engine.engine_core_patch import make_patched_engine_init

    worker = SimpleNamespace(model_runner=_runner(count).worker)
    engine = SimpleNamespace()

    def initialize(owner: Any) -> None:
        owner.vllm_config = SimpleNamespace(
            device_config=SimpleNamespace(device="npu"),
            additional_config={
                "vllm_gr": {"gr": {"beam_width": [128] * 3, "candidate_width": [128] * 3}}
            },
        )
        owner.scheduler = SimpleNamespace()
        owner.model_executor = SimpleNamespace(
            collective_rpc=Mock(side_effect=lambda function: [function(worker)])
        )

    make_patched_engine_init(initialize)(engine)
    assert engine._gr_trie_count == engine.scheduler._gr_trie_count == count
    engine.model_executor.collective_rpc.assert_called_once_with(loaded_constraint_trie_count)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("bad_root", ["outside", -1, True])
def test_npu_core_rejects_invalid_later_member_before_queueing_and_accepts_retry(
    count: int, bad_root: Any
) -> None:
    from queue import Queue

    import msgspec

    from tests.helpers.beam_requests import make_multi_batch_request
    from vllm_gr.v1.engine.core import _handle_submit_gr_session
    from vllm_gr.v1.engine.gr_async_scheduler import GRAdmissionError
    from vllm_gr.v1.engine.wire import register_engine_core_request_types

    register_engine_core_request_types()
    core = SimpleNamespace(
        _gr_trie_count=count,
        input_queue=Queue(),
        vllm_config=SimpleNamespace(
            device_config=SimpleNamespace(device="npu"), additional_config={}
        ),
    )
    request = make_multi_batch_request()
    bad = msgspec.structs.replace(
        request,
        items=(
            request.items[0],
            msgspec.structs.replace(
                request.items[1], trie_id=count if bad_root == "outside" else bad_root
            ),
        ),
    )
    with pytest.raises(GRAdmissionError, match="trie_id"):
        _handle_submit_gr_session(core, bad)
    assert core.input_queue.empty()
    good = msgspec.structs.replace(
        request,
        items=(
            msgspec.structs.replace(request.items[0], trie_id=count - 1),
            request.items[1],
        ),
    )
    _handle_submit_gr_session(core, good)
    assert core.input_queue.get_nowait()[1] is good
    assert core.input_queue.empty()
