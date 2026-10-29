# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real CUDA selection, control/D2H, width changes and graph replay."""

from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
import torch

from tests.kernels.test_gpu_beam_stage_runner import (
    admission_worker,
    dispatch,
    install_decode_model,
    metadata,
)
from vllm_gr.v1.engine.gr_async_scheduler import GR_WORKER_RESULT_ATTR
from vllm_gr.v1.worker.gpu_beam_stage_runner import get_stage_runner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("per_parent_beam_widths", [(4, 8, 1), (4, 4, 8)])
def test_admission_rejects_later_candidate_width_beyond_backend_capacity(per_parent_beam_widths):
    worker = admission_worker()
    m = metadata()
    m = replace(
        m,
        beam_params=msgspec.structs.replace(
            m.beam_params, per_parent_beam_widths=per_parent_beam_widths
        ),
    )
    with pytest.raises(ValueError, match="constraint backend capacity"):
        get_stage_runner(worker, dispatch(m))
    assert not hasattr(worker, "_beam_context")
    assert not hasattr(worker, "_gr_gpu_stage_runner")
    worker._beam_postprocess_workspaces.require.assert_not_called()


@pytest.mark.parametrize("constrained", [True, False])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize(
    "widths,per_parent_beam_widths",
    [
        ((4, 2, 1), None),
        ((1, 3, 4), None),
        ((3, 3, 2), None),
        ((2,), None),
        ((3, 2), None),
        ((4, 4, 4), None),
        ((2, 4, 4), (2, 2, 1)),
    ],
)
def test_dynamic_stage_execution_and_reuse(
    widths, per_parent_beam_widths, graph, batch, constrained
):
    worker = admission_worker()
    from tests.constraints.test_table import make_table
    from vllm_gr.constraints.loader import _upload_cuda_backend

    worker.constraint_backend, _ = _upload_cuda_backend(
        make_table(),
        device=worker.device,
        max_top_k=2 if per_parent_beam_widths is not None else 4,
        max_rows=batch * 4,
    )
    worker.constraint_backend._mark_ready()
    if not constrained:
        worker.constraint_backend = None
        worker._vllm_gr_config.constrained_generation = False
    worker.model_config.enforce_eager = not graph
    worker.model_config.max_model_len = 64
    worker._vllm_gr_config.beam_graph_enabled = graph
    worker.kv_cache_config.kv_cache_groups[0].kv_cache_spec = SimpleNamespace(block_size=16)
    base = torch.tensor([0.1, 0.7, -0.3, 0.2, 0.5, -0.9], device="cuda", dtype=torch.float16)
    weights = torch.arange(6, device="cuda", dtype=torch.float16) * 0.031
    install_decode_model(worker, lambda ids, positions: base + ids[:, None] * weights)
    legal = {(): [1, 3], (1,): [2, 4], (3,): [5], (1, 2): [0, 1], (1, 4): [3], (3, 5): [1, 4]}
    first_owner = None
    for repeat in range(2):
        expected: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        for stage, width in enumerate(widths):
            m = metadata(stage, n=len(widths))
            m = replace(
                m,
                kv_blocks=tuple((i + 1,) for i in range(batch)),
                request_ids=tuple(
                    "session-0" if i == 0 else f"session-0:{i}" for i in range(batch)
                ),
                num_prompt_tokens=sum(7 + i for i in range(batch)),
                num_prompt_tokens_by_request=tuple(7 + i for i in range(batch)),
                beam_params=msgspec.structs.replace(
                    m.beam_params,
                    dynamic_beam_widths=widths,
                    per_parent_beam_widths=per_parent_beam_widths,
                ),
                output_options=msgspec.structs.replace(
                    m.output_options, num_return_sequences=min(2, widths[-1])
                ),
            )
            scheduled = dispatch(m)
            scheduled.num_scheduled_tokens = {
                req: (7 + i if stage == 0 else 1) for i, req in enumerate(m.request_ids)
            }
            owner = get_stage_runner(worker, scheduled)
            assert (owner.dynamic is not None) == (
                per_parent_beam_widths is not None or any(width != 4 for width in widths)
            )
            if first_owner is None:
                first_owner = owner
            assert owner is first_owner

            def prefill(w, scheduled):
                w.execute_model_state = SimpleNamespace(
                    logits=base[None, :].expand(batch, -1).contiguous()
                )

            owner.execute(scheduled, prefill)
            assert owner.logits.shape[0] == batch * (1 if stage == 0 else widths[stage - 1])
            if stage and graph:
                assert owner.graph.ready
                assert owner.graph.signature.beam_width == widths[stage - 1]
            # Scalar enumeration uses full-vocabulary CUDA score semantics,
            # independently of the production Beam state and selection helper.
            choices = []
            for prefix, score in expected:
                ids = torch.tensor([prefix[-1] if prefix else 0], device="cuda")
                logits = base if not prefix else base + ids[:, None] * weights
                logprobs = logits.float().reshape(-1).log_softmax(0).cpu().tolist()
                parent_choices = []
                for token in legal.get(prefix, []) if constrained else range(6):
                    total = float(torch.tensor(score, dtype=torch.float32) + logprobs[token])
                    parent_choices.append((prefix + (token,), total))
                top_k = width if per_parent_beam_widths is None else per_parent_beam_widths[stage]
                choices.extend(sorted(parent_choices, key=lambda item: -item[1])[:top_k])
            expected = sorted(choices, key=lambda item: -item[1])[:width]
            receipt = owner.sample()
            result = getattr(receipt.get_output(), GR_WORKER_RESULT_ATTR)[m.session_id]
        assert result.resources_released and owner.closed
        assert result.terminal_result.failure is None
        assert len(result.terminal_result.output.items) == batch
        for item in result.terminal_result.output.items:
            assert [s.token_ids for s in item.sequences] == [s[0] for s in expected[:2]]
        actual = result.terminal_result.output.items[0].sequences
        assert [s.token_ids for s in actual] == [s[0] for s in expected[:2]]
        assert [s.cumulative_logprob for s in actual] == pytest.approx(
            [s[1] for s in expected[:2]], abs=2e-6
        )
        assert not worker._beam_sessions
        assert owner.beam.session.context.registry.num_active_sessions == 0
