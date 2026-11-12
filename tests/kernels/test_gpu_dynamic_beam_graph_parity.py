# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="no-untyped-def,no-untyped-call"
"""Real attention, identical schedules: eager vs cold and reused CUDA graphs."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("VLLM_GR_RUN_MODEL_SMOKE") != "1" or not torch.cuda.is_available(),
    reason="opt-in CUDA model smoke",
)


def test_irregular_graph_parity(tmp_path):
    from vllm.inputs import tokens_input

    from tests.kernels.test_gpu_dynamic_beam_model import build_tiny_engine
    from vllm_gr.sampling_params import BeamSearchParams

    llm, _ = build_tiny_engine(tmp_path, capacity=16, dense_catalog=True)
    with llm:
        worker = (
            llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        )
        # Equal row counts with different (B,W), singleton output, irregular
        # widening/narrowing, then fixed -> dynamic -> fixed buffer ownership.
        cases = [
            (4, [7, 3, 11], 5),
            (2, [8, 4, 11], 5),
            (4, [4, 2, 11], 5),
            (1, [16, 16, 16], 1),
            (4, [16, 16, 16], 16),
            (4, [7, 3, 11], 5),
            (4, [16, 16, 16], 16),
        ]
        for batch, widths, result_width in cases:
            prompts = [tokens_input([1, 4, 3] + [5] * i) for i in range(batch)]
            params = BeamSearchParams(
                beam_width=16,
                dynamic_beam_widths=widths,
                result_width=result_width,
                max_tokens=3,
                ignore_eos=True,
                temperature=0.0,
            )
            results = []
            captures = None
            for eager in (True, False, False):
                worker.model_config.enforce_eager = eager
                outputs = llm.beam_search_v1(prompts, params, output_mode="full")
                results.append(
                    [[(tuple(s.tokens), s.cum_logprob) for s in out.sequences] for out in outputs]
                )
                if not eager:
                    graph = worker._beam_decode_graph
                    assert graph.ready
                    assert graph.signature.beam_width == widths[-2]
                    current = {
                        k: id(next(iter(v.values())).cudagraph)
                        for k, v in graph._registries.items()
                        if v
                    }
                    if captures is not None:
                        assert current == captures
                    captures = current
            for run in results[1:]:
                for item, reference in zip(run, results[0], strict=True):
                    assert len(item) == result_width
                    assert [t for t, _ in item] == [t for t, _ in reference], (
                        batch,
                        widths,
                        results,
                    )
                    for (tokens, score), (_, expected) in zip(item, reference, strict=True):
                        assert score == pytest.approx(expected, abs=1e-4, rel=1e-5), (
                            batch,
                            widths,
                            tokens,
                            score,
                            expected,
                        )
        worker.model_config.enforce_eager = False


@pytest.mark.parametrize("explicit_schedule", [False, True])
def test_uniform_result_limit_keeps_fixed_adapter(tmp_path, explicit_schedule):
    from vllm.inputs import tokens_input

    from tests.kernels.test_gpu_dynamic_beam_model import build_tiny_engine
    from vllm_gr.sampling_params import BeamSearchParams

    llm, _ = build_tiny_engine(tmp_path, capacity=16, dense_catalog=True)
    with llm:
        worker = (
            llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        )
        prompts = [tokens_input([1, 4, 3] + [5] * i) for i in range(2)]
        eager_reference = None
        for eager in (True, False):
            worker.model_config.enforce_eager = eager
            first_owner = None
            captures = None
            reference = None
            for returned in (16, 5, 1, 16):
                outputs = llm.beam_search_v1(
                    prompts,
                    BeamSearchParams(
                        beam_width=16,
                        dynamic_beam_widths=[16, 16, 16] if explicit_schedule else None,
                        result_width=returned,
                        max_tokens=3,
                        ignore_eos=True,
                        temperature=0.0,
                    ),
                    output_mode="full",
                )
                owner = worker._gr_gpu_stage_runner
                assert owner.dynamic is None
                if first_owner is None:
                    first_owner = owner
                assert owner is first_owner
                rows = [
                    [(tuple(s.tokens), s.cum_logprob) for s in out.sequences] for out in outputs
                ]
                if reference is None:
                    reference = rows
                for row, full in zip(rows, reference, strict=True):
                    assert len(row) == returned
                    assert row == full[:returned]
                if not eager:
                    graph = worker._beam_decode_graph
                    assert graph.ready
                    current = {
                        k: id(next(iter(v.values())).cudagraph)
                        for k, v in graph._registries.items()
                        if v
                    }
                    if captures is not None:
                        assert current == captures
                    captures = current
            if eager:
                eager_reference = reference
            else:
                assert reference is not None and eager_reference is not None
                for row, full in zip(reference, eager_reference, strict=True):
                    assert [t for t, _ in row] == [t for t, _ in full]
                    assert [s for _, s in row] == pytest.approx(
                        [s for _, s in full], abs=1e-4, rel=1e-5
                    )
