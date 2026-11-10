# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.ops.cuda.beam_selection import (
    beam_search_group_cuda,
    beam_search_group_cuda_out,
    beam_search_rec_final_select_cuda,
    beam_search_rec_final_select_grouped_cuda_out,
)
from vllm_gr.ops.cuda.rec_constrained_topk import (
    rec_constrained_topk_cuda,
    rec_constrained_topk_cuda_out,
)
from vllm_gr.v1.beam import BeamAttentionContext, BeamSearchSession, beam_decision
from vllm_gr.v1.beam.beam_decision import (
    BeamPostprocessWorkspace,
    BeamPostprocessWorkspacePool,
    init_beam_decision_from_parents,
)
from vllm_gr.v1.worker import model_runner_common

pytestmark = pytest.mark.gpu

# pytest parametrization decorators are untyped under strict mypy.
# mypy: disable-error-code="misc"


def _runtime(
    beam_width: int = 4,
    max_steps: int = 4,
) -> tuple[BeamAttentionContext, BeamSearchSession, BeamPostprocessWorkspace, SimpleNamespace]:
    context = BeamAttentionContext(
        max_batch=1,
        max_beams=beam_width,
        num_kv_heads=1,
        max_decode_steps=max_steps,
        head_dim=8,
        num_layers=2,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    slot = context.registry.allocate("s", beam_width=0, prefix_len=0)
    session = BeamSearchSession("s", slot, context)
    session.begin_session(beam_width=beam_width, prefix_len=8)
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=beam_width,
        top_k=beam_width,
        max_decode_steps=max_steps,
        device="cuda",
        logits_dtype=torch.float16,
    )
    pool = BeamPostprocessWorkspacePool()
    pool.add(workspace)
    runner = SimpleNamespace(
        device=torch.device("cuda"),
        model_config=SimpleNamespace(dtype=torch.float16),
        _beam_postprocess_workspaces=pool,
        _current_beam_info={
            "beam_data": {
                "s": {
                    "is_beam_decode": True,
                    "parent_beam_ids": list(range(beam_width)),
                }
            }
        },
    )
    return context, session, workspace, runner


def _candidates(beam_width: int) -> tuple[torch.Tensor, torch.Tensor, SimpleNamespace]:
    generator = torch.Generator(device="cuda").manual_seed(20260901)
    tokens = torch.arange(
        beam_width * beam_width,
        dtype=torch.int32,
        device="cuda",
    ).view(beam_width, beam_width)
    scores = torch.randn(
        (beam_width, beam_width),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    sampler = SimpleNamespace(
        logprobs_tensors=SimpleNamespace(
            logprob_token_ids=tokens,
            logprobs=scores,
        )
    )
    return tokens, scores, sampler


def _seed_reference(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    max_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    beam_width = tokens.shape[0]
    cumulative = scores[:, :1].contiguous()
    sequence = torch.zeros(
        (1, beam_width, max_steps),
        dtype=torch.int32,
        device="cuda",
    )
    sequence[0, :, 0].copy_(tokens[:, 0])
    return cumulative, sequence


def _uniform_constraint_inputs(
    dtype: torch.dtype,
    rows: int,
    beam_width: int,
    temperature: float,
) -> tuple[torch.Tensor, ...]:
    """Build one all-hit canonical table with at least W legal candidates."""

    device = torch.device("cuda")
    vocab_size = 512
    degree = 300
    generator = torch.Generator(device=device).manual_seed(20260901 + rows + beam_width)
    logits = torch.randn(
        (rows, vocab_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    prefixes = torch.zeros((rows, 3), dtype=torch.int32, device=device)
    candidates = torch.arange(degree, dtype=torch.int32, device=device)
    prefix1_offsets = torch.full(
        (vocab_size + 1,),
        degree,
        dtype=torch.int32,
        device=device,
    )
    prefix1_offsets[0] = 0
    return (
        logits,
        prefixes,
        candidates,
        prefix1_offsets,
        candidates,
        torch.zeros(1, dtype=torch.int64, device=device),
        torch.tensor([0, degree], dtype=torch.int32, device=device),
        candidates,
        torch.full((rows,), temperature, dtype=torch.float32, device=device),
    )


def test_workspace_decode_pipeline_matches_allocating_group_and_reuses_addresses() -> None:
    beam_width, max_steps = 4, 4
    context, session, workspace, runner = _runtime(beam_width, max_steps)
    tokens, scores, sampler = _candidates(beam_width)
    cumulative, sequence = _seed_reference(tokens, scores, max_steps)
    # Mirror the engine protocol: decode-step decisions require seeded parent
    # state — run_beam_decision bails out when worker_decode_steps == 0
    # (production seeds via request metadata at session creation).
    init_beam_decision_from_parents(
        session,
        [tokens[i, :1].tolist() for i in range(beam_width)],
        cumulative[:, 0].tolist(),
    )
    expected = beam_search_group_cuda(
        cumulative,
        tokens,
        scores,
        sequence,
        1,
        beam_width,
    )
    expected_parent = expected[1][0].to(torch.long)
    expected_pools: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer, pool in enumerate(context.pools):
        values = (
            torch.arange(beam_width * 8, dtype=torch.float16, device="cuda")
            .view(beam_width, 1, 8)
            .add_(layer * 100)
        )
        pool.unshared_key[0, :, :, 0, :].copy_(values)
        pool.unshared_value[0, :, :, 0, :].copy_(values + 1000)
        expected_pools.append(
            (
                pool.unshared_key[0, expected_parent, :, :1, :].clone(),
                pool.unshared_value[0, expected_parent, :, :1, :].clone(),
            )
        )
    pointer_addresses = (
        context.pool_key_ptrs.data_ptr(),
        context.pool_value_ptrs.data_ptr(),
    )
    addresses = tuple(tensor.data_ptr() for tensor in workspace.device_tensors())

    actual = beam_decision.run_beam_decision(
        runner,
        session,
        sampler,
        is_last_step=False,
        result_width=beam_width,
    )

    assert actual is not None
    for key, expected_tensor in zip(
        ("out_token_ids", "out_token_index", "out_log_probs", "out_sequence"),
        (expected[0], expected[1], expected[2], expected[4]),
    ):
        torch.testing.assert_close(actual[key], expected_tensor, rtol=0, atol=0)
    assert tuple(tensor.data_ptr() for tensor in workspace.device_tensors()) == addresses
    assert (
        context.pool_key_ptrs.data_ptr(),
        context.pool_value_ptrs.data_ptr(),
    ) == pointer_addresses
    for pool, (expected_key, expected_value) in zip(context.pools, expected_pools):
        assert torch.equal(pool.unshared_key[0, :, :, :1, :], expected_key)
        assert torch.equal(pool.unshared_value[0, :, :, :1, :], expected_value)
    assert context.decision.worker_decode_steps == 2
    assert session.decode_step == 1


def _seed_worker_decision(
    session: BeamSearchSession,
    tokens: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    """Mirror the engine protocol: seed the parent step-0 state.

    ``run_beam_decision`` bails out when ``worker_decode_steps == 0``
    (production seeds from request metadata at session creation), so every
    decode/terminal decision test must seed before calling it.
    """
    init_beam_decision_from_parents(
        session,
        [tokens[i, :1].tolist() for i in range(tokens.shape[0])],
        scores[:, 0].tolist(),
    )


def test_workspace_terminal_pipeline_matches_direct_final_and_skips_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beam_width, max_steps = 8, 4
    context, session, workspace, runner = _runtime(beam_width, max_steps)
    tokens, scores, sampler = _candidates(beam_width)
    cumulative, sequence = _seed_reference(tokens, scores, max_steps)
    _seed_worker_decision(session, tokens, scores)
    expected = beam_search_rec_final_select_cuda(
        cumulative,
        tokens,
        scores,
        sequence,
        1,
        beam_width,
    )
    session._state.is_last_step = True

    def unexpected_reorder(*args: object, **kwargs: object) -> None:
        raise AssertionError("terminal postprocess must not call SelectUnsharedKV")

    monkeypatch.setattr(beam_decision, "_reorder_kv_same_step", unexpected_reorder)
    addresses = tuple(tensor.data_ptr() for tensor in workspace.device_tensors())
    actual = beam_decision.run_beam_decision(
        runner,
        session,
        sampler,
        is_last_step=True,
        result_width=beam_width,
    )

    assert actual is not None
    for key, expected_tensor in zip(
        ("out_token_ids", "out_token_index", "out_log_probs", "out_sequence"),
        (
            expected[0].view(1, beam_width),
            expected[1].view(1, beam_width),
            expected[2].view(1, beam_width),
            expected[3],
        ),
    ):
        torch.testing.assert_close(actual[key], expected_tensor, rtol=0, atol=0)
    assert tuple(tensor.data_ptr() for tensor in workspace.device_tensors()) == addresses
    assert context.decision.worker_decode_steps == 2
    assert session.decode_step == 0


def test_workspace_candidate_copy_casts_without_replacing_fixed_buffers() -> None:
    _, _, workspace, _ = _runtime()
    tokens = torch.arange(16, dtype=torch.int64, device="cuda").view(4, 4)
    scores = torch.randn((4, 4), dtype=torch.float16, device="cuda")
    sampler = SimpleNamespace(
        logprobs_tensors=SimpleNamespace(
            logprob_token_ids=tokens,
            logprobs=scores,
        )
    )
    token_ptr = workspace.topk_token_ids.data_ptr()
    score_ptr = workspace.topk_scores.data_ptr()

    result = beam_decision._select_candidates_into(sampler, 4, workspace)

    assert result is not None
    out_tokens, out_scores, valid_mask = result
    assert valid_mask is None
    assert out_tokens.data_ptr() == token_ptr
    assert out_scores.data_ptr() == score_ptr
    assert torch.equal(out_tokens, tokens.to(torch.int32))
    torch.testing.assert_close(out_scores, scores.to(torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("temperature", [0.0, 0.6, 1.0, 2.0])
@pytest.mark.parametrize("phase", ["step0", "step1", "final"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("beam_width", [64, 128, 256])
def test_fixed_postprocess_precision_matrix(
    dtype: torch.dtype,
    temperature: float,
    phase: str,
    batch_size: int,
    beam_width: int,
) -> None:
    current_step = {"step0": 0, "step1": 1, "final": 2}[phase]
    candidate_rows = batch_size if current_step == 0 else batch_size * beam_width
    topk_inputs = _uniform_constraint_inputs(
        dtype,
        candidate_rows,
        beam_width,
        temperature,
    )
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=2,
        beam_width=beam_width,
        top_k=beam_width,
        max_decode_steps=3,
        device="cuda",
        logits_dtype=dtype,
    )
    out_tokens = workspace.candidate_rows(
        workspace.topk_token_ids,
        batch_size,
        compact=current_step == 0,
    )
    out_scores = workspace.candidate_rows(
        workspace.topk_scores,
        batch_size,
        compact=current_step == 0,
    )
    out_mask = workspace.candidate_rows(
        workspace.topk_valid_mask,
        batch_size,
        compact=current_step == 0,
    )
    out_counts = workspace.topk_valid_counts.view(-1)[:candidate_rows]
    expected_topk = rec_constrained_topk_cuda(
        *topk_inputs,
        current_step=current_step,
        top_k=beam_width,
        max_prefix1_degree=300,
        max_prefix2_degree=300,
    )
    rec_constrained_topk_cuda_out(
        *topk_inputs,
        current_step,
        beam_width,
        300,
        300,
        out_tokens,
        out_scores,
        out_mask,
        out_counts,
    )
    for actual, expected in zip(
        (out_tokens, out_scores, out_mask, out_counts),
        expected_topk,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    generator = torch.Generator(device="cuda").manual_seed(20260902 + batch_size + beam_width)
    cumulative_rows = batch_size if current_step == 0 else batch_size * beam_width
    cumulative = torch.randn(
        (cumulative_rows, 1),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    sequence = torch.arange(
        batch_size * beam_width * 3,
        dtype=torch.int32,
        device="cuda",
    ).view(batch_size, beam_width, 3)
    expected_group = beam_search_group_cuda(
        cumulative,
        expected_topk[0],
        expected_topk[1],
        sequence,
        current_step,
        beam_width,
        expected_topk[2],
    )
    selected = (
        workspace.selected_token_ids[:batch_size],
        workspace.selected_parent_ids[:batch_size],
        workspace.selected_scores[:batch_size],
        workspace.selected_parent_count_prefix[:batch_size],
        workspace.next_sequence[:batch_size],
    )
    scratch = workspace.selection_scratch
    beam_search_group_cuda_out(
        cumulative,
        out_tokens,
        out_scores,
        sequence,
        current_step,
        beam_width,
        *selected,
        scratch.local_scores,
        scratch.local_flat_indices,
        scratch.merge_scores,
        scratch.merge_flat_indices,
        out_mask,
        workspace.selected_flat_indices[:batch_size],
    )
    for actual, expected in zip(selected, expected_group):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    if phase == "final":
        expected_final = beam_search_rec_final_select_cuda(
            cumulative,
            expected_topk[0],
            expected_topk[1],
            sequence,
            current_step,
            beam_width,
            expected_topk[2],
        )
        final = (
            workspace.final_token_ids[:batch_size],
            workspace.final_parent_ids[:batch_size],
            workspace.final_scores[:batch_size],
            workspace.final_sequence[:batch_size],
        )
        beam_search_rec_final_select_grouped_cuda_out(
            selected[0],
            selected[1],
            selected[2],
            selected[4],
            beam_width,
            *final,
            workspace.selected_flat_indices[:batch_size],
        )
        expected_decision = {
            "out_token_ids": expected_final[0].view(batch_size, beam_width),
            "out_token_index": expected_final[1].view(batch_size, beam_width),
            "out_log_probs": expected_final[2].view(batch_size, beam_width),
        }
        actual_decision = {
            "out_token_ids": final[0],
            "out_token_index": final[1],
            "out_log_probs": final[2],
        }
        for actual, expected in zip(final, (*expected_decision.values(), expected_final[3])):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        expected_decision = {
            "out_token_ids": expected_group[0],
            "out_token_index": expected_group[1],
            "out_log_probs": expected_group[2],
        }
        actual_decision = {
            "out_token_ids": selected[0],
            "out_token_index": selected[1],
            "out_log_probs": selected[2],
        }

    expected_payload = beam_decision._encode_cuda_beam_result(
        expected_decision,
        is_last_step=phase == "final",
    )
    actual_payload = beam_decision._encode_cuda_beam_result(
        actual_decision,
        is_last_step=phase == "final",
        workspace=workspace,
    )
    assert actual_payload == expected_payload


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("upstream_mode", ["eager", "full_graph"])
@pytest.mark.parametrize("beam_width", [64, 128, 256])
def test_eager_and_full_graph_upstream_outputs_are_postprocess_identical(
    dtype: torch.dtype,
    upstream_mode: str,
    beam_width: int,
) -> None:
    inputs = list(
        _uniform_constraint_inputs(
            dtype,
            beam_width,
            beam_width,
            temperature=0.6,
        )
    )
    eager_logits = inputs[0]
    if upstream_mode == "full_graph":
        graph_logits = torch.empty_like(eager_logits)
        graph_logits.copy_(eager_logits)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_logits.copy_(eager_logits)
        graph.replay()
        inputs[0] = graph_logits

    expected_topk = rec_constrained_topk_cuda(
        eager_logits,
        *inputs[1:],
        current_step=1,
        top_k=beam_width,
        max_prefix1_degree=300,
        max_prefix2_degree=300,
    )
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=beam_width,
        top_k=beam_width,
        max_decode_steps=3,
        device="cuda",
        logits_dtype=dtype,
    )
    outputs = (
        workspace.topk_token_ids.view(beam_width, beam_width),
        workspace.topk_scores.view(beam_width, beam_width),
        workspace.topk_valid_mask.view(beam_width, beam_width),
        workspace.topk_valid_counts.view(-1),
    )
    rec_constrained_topk_cuda_out(
        *inputs,
        1,
        beam_width,
        300,
        300,
        *outputs,
    )
    for actual, expected in zip(outputs, expected_topk):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    cumulative = torch.linspace(
        -1.0,
        1.0,
        beam_width,
        dtype=torch.float32,
        device="cuda",
    ).view(beam_width, 1)
    sequence = torch.arange(
        beam_width * 3,
        dtype=torch.int32,
        device="cuda",
    ).view(1, beam_width, 3)
    expected_group = beam_search_group_cuda(
        cumulative,
        expected_topk[0],
        expected_topk[1],
        sequence,
        1,
        beam_width,
        expected_topk[2],
    )
    selected = (
        workspace.selected_token_ids,
        workspace.selected_parent_ids,
        workspace.selected_scores,
        workspace.selected_parent_count_prefix,
        workspace.next_sequence,
    )
    scratch = workspace.selection_scratch
    beam_search_group_cuda_out(
        cumulative,
        outputs[0],
        outputs[1],
        sequence,
        1,
        beam_width,
        *selected,
        scratch.local_scores,
        scratch.local_flat_indices,
        scratch.merge_scores,
        scratch.merge_flat_indices,
        outputs[2],
        workspace.selected_flat_indices,
    )
    for actual, expected in zip(selected, expected_group):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prefix_cache_hit_or_miss_does_not_change_postprocess_result() -> None:
    payloads = []
    for prefix_cache_hit in (False, True):
        _, session, workspace, runner = _runtime(64)
        runner._current_beam_info["beam_data"]["s"]["prefix_cache_hit"] = prefix_cache_hit
        tokens, scores, sampler = _candidates(64)
        _seed_worker_decision(session, tokens, scores)
        decision = beam_decision.run_beam_decision(
            runner,
            session,
            sampler,
            is_last_step=True,
            result_width=64,
        )
        assert decision is not None
        encoded = beam_decision._encode_cuda_beam_result(
            decision,
            is_last_step=True,
            workspace=workspace,
        )
        assert encoded is not None
        payloads.append(encoded)

    assert payloads[0] == payloads[1]


def test_workspace_reduces_sampler_payload_into_fixed_buffers() -> None:
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    beam_width = 4
    _, _, workspace, runner = _runtime(beam_width)
    runner._beam_decision_tensors = {
        "out_token_ids": workspace.selected_token_ids[:1],
        "out_token_index": workspace.selected_parent_ids[:1],
        "out_log_probs": workspace.selected_scores[:1],
    }
    runner._beam_decision_tensors["out_token_ids"].copy_(
        torch.tensor([[41, 42, 43, 44]], dtype=torch.int32, device="cuda")
    )
    runner._beam_decision_tensors["out_log_probs"].copy_(
        torch.tensor([[-0.1, -0.2, -0.3, -0.4]], dtype=torch.float32, device="cuda")
    )
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.zeros((beam_width, 1), dtype=torch.int32, device="cuda"),
        logprobs_tensors=LogprobsTensors(
            logprob_token_ids=torch.zeros(
                (beam_width, beam_width), dtype=torch.int32, device="cuda"
            ),
            logprobs=torch.zeros((beam_width, beam_width), dtype=torch.float32, device="cuda"),
            selected_token_ranks=torch.arange(beam_width, dtype=torch.int64, device="cuda"),
        ),
    )
    sampled_ptr = workspace.sampled_token_ids.data_ptr()
    ranks_ptr = workspace.selected_token_ranks.data_ptr()

    assert model_runner_common._reduce_sampler_output_for_worker_decision(runner, sampler_output)

    assert sampler_output.sampled_token_ids.data_ptr() == sampled_ptr
    assert sampler_output.logprobs_tensors.selected_token_ranks.data_ptr() == ranks_ptr
    assert sampler_output.logprobs_tensors.logprob_token_ids.data_ptr() == (
        workspace.selected_token_ids.data_ptr()
    )
    assert sampler_output.logprobs_tensors.logprobs.data_ptr() == (
        workspace.selected_scores.data_ptr()
    )
    assert sampler_output.sampled_token_ids.tolist() == [[41]]
    assert sampler_output.logprobs_tensors.selected_token_ranks.tolist() == [0]


def test_async_sampler_payload_owns_device_snapshot() -> None:
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    beam_width = 4
    _, _, workspace, runner = _runtime(beam_width)
    runner.use_async_scheduling = True
    runner._beam_result_host_ready = False
    runner._beam_decision_tensors = {
        "out_token_ids": workspace.selected_token_ids[:1],
        "out_token_index": workspace.selected_parent_ids[:1],
        "out_log_probs": workspace.selected_scores[:1],
    }
    runner._beam_decision_tensors["out_token_ids"].copy_(
        torch.tensor([[41, 42, 43, 44]], dtype=torch.int32, device="cuda")
    )
    runner._beam_decision_tensors["out_log_probs"].copy_(
        torch.tensor([[-0.1, -0.2, -0.3, -0.4]], dtype=torch.float32, device="cuda")
    )
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.zeros((beam_width, 1), dtype=torch.int32, device="cuda"),
        logprobs_tensors=LogprobsTensors(
            logprob_token_ids=torch.zeros(
                (beam_width, beam_width), dtype=torch.int32, device="cuda"
            ),
            logprobs=torch.zeros((beam_width, beam_width), dtype=torch.float32, device="cuda"),
            selected_token_ranks=torch.arange(beam_width, dtype=torch.int64, device="cuda"),
        ),
    )

    assert model_runner_common._reduce_sampler_output_for_worker_decision(runner, sampler_output)

    output_ids = sampler_output.logprobs_tensors.logprob_token_ids
    output_scores = sampler_output.logprobs_tensors.logprobs
    output_ranks = sampler_output.logprobs_tensors.selected_token_ranks
    assert output_ids.data_ptr() != workspace.selected_token_ids.data_ptr()
    assert output_scores.data_ptr() != workspace.selected_scores.data_ptr()
    assert output_ranks.data_ptr() != workspace.selected_token_ranks.data_ptr()
    assert sampler_output.sampled_token_ids.data_ptr() != workspace.sampled_token_ids.data_ptr()

    expected_ids = output_ids.clone()
    expected_scores = output_scores.clone()
    workspace.selected_token_ids.fill_(-1)
    workspace.selected_scores.fill_(123.0)
    workspace.sampled_token_ids.fill_(-1)
    workspace.selected_token_ranks.fill_(99)

    torch.testing.assert_close(output_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(output_scores, expected_scores, rtol=0, atol=0)
    assert sampler_output.sampled_token_ids.tolist() == [[41]]
    assert output_ranks.tolist() == [0]


def test_fixed_result_pack_matches_existing_bytes_and_reuses_storage() -> None:
    beam_width = 4
    _, _, workspace, _ = _runtime(beam_width)
    decision = {
        "out_token_ids": workspace.selected_token_ids[:1],
        "out_token_index": workspace.selected_parent_ids[:1],
        "out_log_probs": workspace.selected_scores[:1],
    }
    decision["out_token_ids"].copy_(
        torch.tensor([[11, 12, 13, 14]], dtype=torch.int32, device="cuda")
    )
    decision["out_token_index"].copy_(
        torch.tensor([[3, 1, 1, 0]], dtype=torch.int32, device="cuda")
    )
    # Include signed zero and a non-round decimal so exact float bits, rather
    # than merely numeric equality, are exercised by the golden comparison.
    decision["out_log_probs"].copy_(
        torch.tensor([[-0.0, -0.1, -2.5, -17.0]], dtype=torch.float32, device="cuda")
    )
    expected = beam_decision._encode_cuda_beam_result(decision, is_last_step=True)
    assert expected is not None
    packed_ptr = workspace.packed_result.data_ptr()
    host_ptr = workspace.host_result.data_ptr()

    first = beam_decision._encode_cuda_beam_result(
        decision,
        is_last_step=True,
        workspace=workspace,
    )
    second = beam_decision._encode_cuda_beam_result(
        decision,
        is_last_step=True,
        workspace=workspace,
    )

    assert first == expected
    assert second == expected
    assert workspace.packed_result.data_ptr() == packed_ptr
    assert workspace.host_result.data_ptr() == host_ptr
    assert workspace.host_result.is_pinned()


def test_fixed_result_pack_waits_only_for_its_copy_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, workspace, _ = _runtime()
    decision = {
        "out_token_ids": workspace.selected_token_ids[:1].zero_(),
        "out_token_index": workspace.selected_parent_ids[:1].zero_(),
        "out_log_probs": workspace.selected_scores[:1].zero_(),
    }
    real_event = workspace.copy_event
    assert real_event is not None

    class RecordingEvent:
        def __init__(self) -> None:
            self.records = 0
            self.waits = 0

        def record(self) -> None:
            self.records += 1
            real_event.record()

        def synchronize(self) -> None:
            self.waits += 1
            real_event.synchronize()

    event = RecordingEvent()
    workspace.copy_event = event

    def unexpected_global_sync(*args: object, **kwargs: object) -> None:
        raise AssertionError("fixed result pack must not globally synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_global_sync)
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        monkeypatch.setattr(accelerator, "synchronize", unexpected_global_sync)

    result = beam_decision._encode_cuda_beam_result(
        decision,
        is_last_step=False,
        workspace=workspace,
    )

    assert result is not None
    assert event.records == 1
    assert event.waits == 1


def test_worker_publish_path_uses_fixed_pack_without_torch_cat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, session, workspace, runner = _runtime(64)
    tokens, scores, sampler = _candidates(64)
    _seed_worker_decision(session, tokens, scores)
    runner._current_beam_info["beam_session"] = session
    runner._current_beam_info["beam_data"]["s"]["result_width"] = 64
    monkeypatch.setattr(beam_decision, "_run_cuda_prefill_decision", lambda *args: None)
    monkeypatch.setattr(beam_decision, "worker_decision_enabled", lambda _runner: True)

    def unexpected_cat(*args: object, **kwargs: object) -> None:
        raise AssertionError("CUDA Worker result publication must not call torch.cat")

    monkeypatch.setattr(torch, "cat", unexpected_cat)

    beam_decision.maybe_run_worker_beam_decision(runner, sampler)

    assert runner._beam_decision_result is not None
    assert runner._beam_decision_result["buffer_format"] == "beam_result_i32_f32_v1"
    unpacked = beam_decision.unpack_beam_result_buffer(runner._beam_decision_result)
    assert unpacked is not None
    assert len(unpacked[0]) == 64
    assert context.decision.worker_decode_steps == 2
    assert workspace.host_result.is_pinned()


def test_constraint_worker_publish_uses_deferred_staging_and_host_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beam_width = 64
    context, session, workspace, runner = _runtime(beam_width)
    tokens, scores, sampler = _candidates(beam_width)
    scores, order = torch.sort(scores, dim=1, descending=True, stable=True)
    tokens = torch.gather(tokens, 1, order)
    valid_mask = torch.ones_like(tokens, dtype=torch.bool)
    counts = torch.full((beam_width,), beam_width, dtype=torch.int32, device="cuda")
    _seed_worker_decision(session, tokens, scores)
    sampler.logprobs_tensors.logprob_token_ids = tokens
    sampler.logprobs_tensors.logprobs = scores
    sampler._gr_constraint_token_ids = tokens
    sampler._gr_constraint_logprobs = scores
    sampler._gr_constraint_valid_mask = valid_mask
    sampler._gr_constraint_valid_counts = counts
    sampler._gr_deferred_decode_status = True
    sampler._gr_beam_postprocess_workspace = workspace
    runner._current_beam_info["beam_session"] = session
    runner._current_beam_info["beam_data"]["s"]["result_width"] = beam_width
    monkeypatch.setattr(beam_decision, "_run_cuda_prefill_decision", lambda *args: None)
    monkeypatch.setattr(beam_decision, "worker_decision_enabled", lambda _runner: True)

    beam_decision.maybe_run_worker_beam_decision(runner, sampler)

    assert runner._beam_result_host_ready is True
    assert runner._beam_decision_result is not None
    unpacked = beam_decision.unpack_beam_result_buffer(runner._beam_decision_result)
    assert unpacked is not None
    device_decision = runner._beam_decision_tensors
    assert unpacked[0] == device_decision["out_token_ids"].reshape(-1).tolist()
    assert unpacked[1] == device_decision["out_log_probs"].reshape(-1).tolist()
    assert unpacked[2] == device_decision["out_token_index"].reshape(-1).tolist()
    assert torch.equal(
        device_decision["_host_token_ids"],
        device_decision["out_token_ids"].cpu(),
    )
    assert torch.equal(
        device_decision["_host_log_probs"],
        device_decision["out_log_probs"].cpu(),
    )
    assert context.decision.worker_decode_steps == 2
    assert workspace.host_prefill_staging.is_pinned()

    # A later step may reuse the pinned staging before this async output is
    # consumed.  Published compatibility views must own an independent snapshot.
    expected_ids = device_decision["_host_token_ids"].clone()
    expected_scores = device_decision["_host_log_probs"].clone()
    expected_ranks = device_decision["_host_selected_token_ranks"].clone()
    assert device_decision["_host_token_ids"].untyped_storage().data_ptr() != (
        workspace.host_prefill_staging.untyped_storage().data_ptr()
    )
    workspace.host_prefill_staging.fill_(54321)
    assert torch.equal(device_decision["_host_token_ids"], expected_ids)
    assert torch.equal(device_decision["_host_log_probs"], expected_scores)
    assert torch.equal(device_decision["_host_selected_token_ranks"], expected_ranks)


def test_constraint_worker_deferred_zero_candidate_raises_before_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beam_width = 64
    _, session, workspace, runner = _runtime(beam_width)
    tokens, scores, sampler = _candidates(beam_width)
    scores, order = torch.sort(scores, dim=1, descending=True, stable=True)
    tokens = torch.gather(tokens, 1, order)
    _seed_worker_decision(session, tokens, scores)
    sampler.logprobs_tensors.logprob_token_ids = tokens
    sampler.logprobs_tensors.logprobs = scores
    sampler._gr_constraint_token_ids = tokens
    sampler._gr_constraint_logprobs = scores
    sampler._gr_constraint_valid_mask = torch.ones_like(tokens, dtype=torch.bool)
    sampler._gr_constraint_valid_counts = torch.full(
        (beam_width,), beam_width, dtype=torch.int32, device="cuda"
    )
    sampler._gr_constraint_valid_counts[9] = 0
    sampler._gr_deferred_decode_status = True
    sampler._gr_beam_postprocess_workspace = workspace
    runner._current_beam_info["beam_session"] = session
    monkeypatch.setattr(beam_decision, "_run_cuda_prefill_decision", lambda *args: None)
    monkeypatch.setattr(beam_decision, "worker_decision_enabled", lambda _runner: True)

    with pytest.raises(RuntimeError, match=r"no legal candidate for batch rows \[0\]"):
        beam_decision.maybe_run_worker_beam_decision(runner, sampler)
    assert getattr(runner, "_beam_decision_result", None) is None


def test_fixed_decode_and_pack_profile_has_no_allocator_or_conversion_ops() -> None:
    from torch.profiler import ProfilerActivity, profile

    beam_width = 64
    context, session, workspace, runner = _runtime(beam_width)
    generator = torch.Generator(device="cuda").manual_seed(20260903)
    candidates = workspace.topk_token_ids.view(beam_width, beam_width)
    candidate_scores = workspace.topk_scores.view(beam_width, beam_width)
    candidates.copy_(
        torch.arange(
            beam_width * beam_width,
            dtype=torch.int32,
            device="cuda",
        ).view(beam_width, beam_width)
    )
    candidate_scores.copy_(
        torch.randn(
            (beam_width, beam_width),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
    )
    sampler = SimpleNamespace(
        logprobs_tensors=SimpleNamespace(
            logprob_token_ids=candidates,
            logprobs=candidate_scores,
        ),
        _gr_constraint_token_ids=candidates,
        _gr_constraint_logprobs=candidate_scores,
    )
    initial_scores = torch.randn(
        (1, beam_width),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    initial_sequence = torch.arange(
        beam_width * workspace.max_decode_steps,
        dtype=torch.int32,
        device="cuda",
    ).view(1, beam_width, workspace.max_decode_steps)

    def reset_state() -> None:
        context.decision.beam_scores[:1].copy_(initial_scores)
        context.decision.sequence[:1].copy_(initial_sequence)
        context.decision.worker_decode_steps = 1
        session._state.decode_step = 0
        session._state.is_last_step = False

    def run_once() -> None:
        decision = beam_decision.run_beam_decision(
            runner,
            session,
            sampler,
            is_last_step=False,
            result_width=beam_width,
        )
        assert decision is not None
        assert (
            beam_decision._encode_cuda_beam_result(
                decision,
                is_last_step=False,
                workspace=workspace,
            )
            is not None
        )

    reset_state()
    run_once()
    reset_state()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        run_once()

    names = {event.key for event in prof.key_averages()}
    banned = {
        "aten::empty",
        "aten::cat",
        "aten::to",
        "aten::_to_copy",
        "aten::item",
        "aten::_local_scalar_dense",
    }
    violations = [
        (event.key, event.input_shapes, event.stack)
        for event in prof.events()
        if event.key in banned
    ]
    assert names.isdisjoint(banned), violations


def test_fixed_postprocess_1000_calls_have_stable_device_memory() -> None:
    beam_width = 64
    context, session, workspace, runner = _runtime(beam_width)
    generator = torch.Generator(device="cuda").manual_seed(20260904)
    candidates = workspace.topk_token_ids.view(beam_width, beam_width)
    candidate_scores = workspace.topk_scores.view(beam_width, beam_width)
    candidates.copy_(
        torch.arange(
            beam_width * beam_width,
            dtype=torch.int32,
            device="cuda",
        ).view(beam_width, beam_width)
    )
    candidate_scores.copy_(
        torch.randn(
            (beam_width, beam_width),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
    )
    sampler = SimpleNamespace(
        logprobs_tensors=SimpleNamespace(
            logprob_token_ids=candidates,
            logprobs=candidate_scores,
        ),
        _gr_constraint_token_ids=candidates,
        _gr_constraint_logprobs=candidate_scores,
    )
    initial_scores = torch.randn(
        (1, beam_width),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    initial_sequence = torch.arange(
        beam_width * workspace.max_decode_steps,
        dtype=torch.int32,
        device="cuda",
    ).view(1, beam_width, workspace.max_decode_steps)

    def run_once(is_last_step: bool) -> None:
        context.decision.beam_scores[:1].copy_(initial_scores)
        context.decision.sequence[:1].copy_(initial_sequence)
        context.decision.worker_decode_steps = 1
        session._state.decode_step = 0
        session._state.is_last_step = is_last_step
        decision = beam_decision.run_beam_decision(
            runner,
            session,
            sampler,
            is_last_step=is_last_step,
            result_width=beam_width,
        )
        assert decision is not None
        assert (
            beam_decision._encode_cuda_beam_result(
                decision,
                is_last_step=is_last_step,
                workspace=workspace,
            )
            is not None
        )

    for is_last_step in (False, True, False, True):
        run_once(is_last_step)
    torch.accelerator.synchronize()
    addresses = tuple(tensor.data_ptr() for tensor in workspace.device_tensors())
    allocated_before = torch.accelerator.memory_allocated()
    reserved_before = torch.accelerator.memory_reserved()

    for iteration in range(1000):
        run_once(bool(iteration & 1))
    torch.accelerator.synchronize()

    assert torch.accelerator.memory_allocated() == allocated_before
    assert torch.accelerator.memory_reserved() == reserved_before
    assert tuple(tensor.data_ptr() for tensor in workspace.device_tensors()) == addresses


@pytest.mark.parametrize(("beam_width", "candidate_width"), [(8, 4), (1024, 256)])
def test_prefill_decision_zero_pads_inactive_beam_parents(
    beam_width: int,
    candidate_width: int,
) -> None:
    """Prefill only seeds min(W, K) live beams; the tail must be parent 0.

    EngineCore's engine-driven self-advance rejects any negative parent index,
    so the legacy ``-1`` tail made every prefill -> step0 transition fall back
    to the frontend path.  K < W is exactly the case that exposes the tail.
    """
    from vllm_gr.config import ConstraintBackend, ConstraintTableConfig, GRConfig

    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=beam_width,
        top_k=candidate_width,
        max_decode_steps=3,
        device="cuda",
        logits_dtype=torch.float16,
    )
    pool = BeamPostprocessWorkspacePool()
    pool.add(workspace)
    table = ConstraintTableConfig(
        max_top_k=candidate_width,
        enabled=True,
        path="/tmp/catalog.json",
        artifact_digest="sha256:" + "0" * 64,
        tokenizer_digest="sha256:" + "0" * 64,
        candidate_width=candidate_width,
    )
    runner = SimpleNamespace(
        device=torch.device("cuda"),
        model_config=SimpleNamespace(dtype=torch.float16),
        _beam_postprocess_workspaces=pool,
        _vllm_gr_config=GRConfig(
            catalog_path="/tmp/catalog.json",
            constraint_backend=ConstraintBackend.CONSTRAINT_TABLE,
            beam_max_width=beam_width,
            constraint_table=table,
            beam_worker_decision=True,
        ),
        _current_beam_info={
            "beam_data": {
                "s": {
                    "is_constraint_prefill": True,
                    "beam_width": beam_width,
                    "candidate_width": candidate_width,
                }
            }
        },
    )
    token_ids = torch.arange(candidate_width, dtype=torch.int32, device="cuda").view(1, -1) + 1000
    scores = torch.linspace(-1.0, -2.0, candidate_width, device="cuda").view(1, -1)
    sampler = SimpleNamespace(
        logprobs_tensors=SimpleNamespace(logprob_token_ids=token_ids, logprobs=scores)
    )

    decision = beam_decision._run_cuda_prefill_decision(runner, sampler)

    assert decision is not None
    live = min(beam_width, candidate_width)
    assert decision["stage"] == "prefill"
    assert decision["out_token_ids"].shape == (1, beam_width)
    assert torch.equal(decision["out_token_ids"][0, :live], token_ids[0, :live])
    assert torch.equal(decision["out_log_probs"][0, :live], scores[0, :live])
    # Inactive tail: no token, -inf score and -- critically -- no negative parent.
    assert torch.all(decision["out_token_ids"][0, live:] == 0)
    assert torch.all(decision["out_log_probs"][0, live:] == -float("inf"))
    assert torch.all(decision["out_token_index"] == 0)
