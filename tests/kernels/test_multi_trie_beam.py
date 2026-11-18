# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="no-untyped-def,no-untyped-call"

import gc
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
import torch

from tests.constraints.test_multi_trie import SIDS, make_tries
from tests.helpers.beam_requests import make_multi_batch_request
from tests.kernels.beam_stage_test_utils import (
    admission_worker,
    dispatch,
    install_decode_model,
    metadata,
)
from vllm_gr.constraints.loader import _upload_cuda_backend
from vllm_gr.constraints.versioned_table import VersionedConstraintTable
from vllm_gr.v1.beam.dispatch import GR_WORKER_RESULT_ATTR
from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import get_stage_runner
from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def root_dispatch(stage, roots, widths=None, factors=None):
    m = metadata(stage)
    request = make_multi_batch_request()
    params = msgspec.structs.replace(
        m.beam_params, dynamic_beam_widths=widths, per_parent_beam_widths=factors
    )
    options = msgspec.structs.replace(m.output_options, num_return_sequences=2)
    request = msgspec.structs.replace(
        request,
        beam_params=params,
        output_options=options,
        items=tuple(
            msgspec.structs.replace(item, trie_id=root) for item, root in zip(request.items, roots)
        ),
    )
    lengths = tuple(len(item.request.prompt_token_ids) for item in request.items)
    m = replace(
        m,
        beam_params=params,
        output_options=options,
        kv_blocks=((1,), (2,)),
        request_ids=tuple(item.request.request_id for item in request.items),
        num_prompt_tokens=sum(lengths),
        num_prompt_tokens_by_request=lengths,
        session_init=request if stage == 0 else None,
    )
    scheduled = dispatch(m)
    scheduled.num_scheduled_tokens = {
        req: lengths[i] if stage == 0 else 1 for i, req in enumerate(m.request_ids)
    }
    return scheduled


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("widths", [None, (2, 3, 4)])
@pytest.mark.parametrize("independent_k", [False, True])
def test_mixed_roots_beam_selection_and_reuse(graph, widths, independent_k):
    # Retire cyclic graph owners before a new capture can allocate and trigger GC.
    # Destroying a previous CUDAGraph during capture invalidates the active stream.
    gc.collect()
    worker = admission_worker(max_rows=8)
    backend, _ = _upload_cuda_backend(make_tries(), device=worker.device, max_top_k=4, max_rows=8)
    backend._mark_ready()
    worker.constraint_backend = VersionedConstraintTable(backend, loader=lambda _: backend)
    worker.model_config.enforce_eager = not graph
    worker._vllm_gr_config.beam_graph_enabled = graph
    base = torch.tensor([0.1, 0.7, -0.3, 0.2, 0.5, -0.9], device=worker.device, dtype=torch.float16)
    install_decode_model(worker, lambda ids, positions: base.expand(ids.numel(), -1).contiguous())
    owner_before = None
    pointers = None
    factors = ((4, 2, 1) if widths is None else (2, 2, 2)) if independent_k else None
    for roots in ((0, 1), (1, 0), (0, 1)):
        expected: list[list[tuple[tuple[int, ...], float]]] = [[((), 0.0)] for _ in roots]
        for stage in range(3):
            scheduled = root_dispatch(stage, roots, widths, factors)
            owner = get_stage_runner(worker, scheduled, hooks_type=CudaStageHooksV1)
            if owner_before is None:
                owner_before = owner
                pointers = {w: tensor.data_ptr() for w, tensor in owner.trie_ids_by_width.items()}
            assert owner is owner_before
            assert {
                w: tensor.data_ptr() for w, tensor in owner.trie_ids_by_width.items()
            } == pointers

            def prefill(w, scheduled):
                w.execute_model_state = SimpleNamespace(logits=base.repeat(2, 1))

            owner.execute(scheduled, prefill)
            scores = base.float().log_softmax(-1).cpu().tolist()
            width = 4 if widths is None else widths[stage]
            for index, root in enumerate(roots):
                choices = []
                for prefix, score in expected[index]:
                    tokens = sorted(
                        {sid[stage] for sid in SIDS[root] if sid[:stage] == prefix},
                        key=lambda token: scores[token],
                        reverse=True,
                    )
                    if factors is not None:
                        tokens = tokens[: factors[stage]]
                    for token in tokens:
                        total = float(torch.tensor(score, dtype=torch.float32) + scores[token])
                        choices.append((prefix + (token,), total))
                expected[index] = sorted(choices, key=lambda entry: -entry[1])[:width]
            receipt = owner.sample()
            result = getattr(receipt.get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
        assert result.resources_released and owner.closed
        assert result.terminal_result.failure is None
        for index, item in enumerate(result.terminal_result.output.items):
            assert [s.token_ids for s in item.sequences] == [s[0] for s in expected[index][:2]]
            assert [s.cumulative_logprob for s in item.sequences] == pytest.approx(
                [s[1] for s in expected[index][:2]], abs=2e-6
            )


def test_unknown_root_is_rejected_before_registry_allocation():
    worker = admission_worker(max_rows=8)
    with pytest.raises(ValueError, match="trie_id"):
        get_stage_runner(worker, root_dispatch(0, (0, 1)), hooks_type=CudaStageHooksV1)
    assert not getattr(worker, "_beam_sessions", {})
    assert getattr(worker, "_beam_context", None) is None


def test_multi_trie_expansion_capacity_is_checked_before_allocation():
    worker = admission_worker(max_rows=8)
    backend, _ = _upload_cuda_backend(make_tries(), device=worker.device, max_top_k=4, max_rows=8)
    backend._mark_ready()
    worker.constraint_backend = VersionedConstraintTable(backend, loader=lambda _: backend)
    with pytest.raises(ValueError, match="constraint backend capacity"):
        get_stage_runner(
            worker, root_dispatch(0, (0, 1), factors=(4, 8, 1)), hooks_type=CudaStageHooksV1
        )
    assert not getattr(worker, "_beam_sessions", {})
    assert getattr(worker, "_beam_context", None) is None


def test_multi_trie_startup_capture_and_request_reuse():
    from vllm_gr.v1.worker.gr_startup import _capture_batch

    gc.collect()
    worker = admission_worker(max_rows=8)
    backend, _ = _upload_cuda_backend(make_tries(), device=worker.device, max_top_k=4, max_rows=8)
    backend._mark_ready()
    worker.constraint_backend = VersionedConstraintTable(backend, loader=lambda _: backend)
    worker.model_config.enforce_eager = False
    worker._vllm_gr_config.beam_graph_enabled = True
    base = torch.arange(6, device=worker.device, dtype=torch.float16)
    install_decode_model(worker, lambda ids, positions: base.expand(ids.numel(), -1).contiguous())
    assert (
        _capture_batch(worker, 2, hooks_type=CudaStageHooksV1, allow_eager_fallback=False) is None
    )
    owner = worker._gr_gpu_stage_runner
    assert owner.closed and owner.graph.ready
    assert worker._beam_context.registry.num_active_sessions == 0
    pointers = {w: ids.data_ptr() for w, ids in owner.trie_ids_by_width.items()}
    for ids in owner.trie_ids_by_width.values():
        assert not ids.any()
    # Real mixed-Trie requests replace capture's default selectors in place.
    scheduled = root_dispatch(0, (1, 0))
    assert get_stage_runner(worker, scheduled, hooks_type=CudaStageHooksV1) is owner
    for w, ids in owner.trie_ids_by_width.items():
        assert ids.data_ptr() == pointers[w]
        assert ids.tolist() == [1] * w + [0] * w
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)
