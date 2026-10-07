# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for dynamic per-step beam widths (W0, W1, W2) in P2D."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker import prefill_graph_common as p2d_common
from vllm_gr.v1.worker.prefill_graph_common import (
    P2DBufferPool,
    P2DMetadata,
    StepBeamWidths,
    get_p2d_max_beams,
    get_p2d_step_beam_widths_config,
    is_p2d_eligible,
    match_p2d_step_beam_widths,
    run_decode_pass_eager,
    run_decode_passes,
    run_p2d_stepwise_tail,
    step1_prune_and_kv_gather,
    step2_prune_and_backtrack,
)

pytestmark = pytest.mark.skipif(
    p2d_common.is_npu_platform(),
    reason="P2D is gated off on NPU (is_p2d_enabled); P2D unit tests are GPU-only",
)


def _make_pool(
    max_beams: int,
    vocab: int = 16,
    num_layers: int = 1,
    num_kv_heads: int = 2,
    head_dim: int = 4,
    max_batch: int = 1,
) -> P2DBufferPool:
    return P2DBufferPool(
        max_batch=max_batch,
        max_bucket=128,
        max_beams=max_beams,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


class _ConstLogitsModel:
    """Deterministic model: every row emits the same logits vector."""

    def __init__(self, logits_vec: torch.Tensor) -> None:
        self.logits_vec = logits_vec

    def __call__(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        assert input_ids is not None
        hidden = torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)
        hidden[torch.arange(input_ids.shape[0]), input_ids.long() % 4] = 1.0
        return hidden

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.logits_vec.unsqueeze(0).expand(hidden.shape[0], -1).clone()


def test_step_beam_widths_env_and_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_GR_P_BW", "32, 64")
    monkeypatch.setenv("VLLM_GR_D0_BW", "64,128")
    monkeypatch.setenv("VLLM_GR_D1_BW", "128,256")
    p_bw, d0_bw, d1_bw = get_p2d_step_beam_widths_config()
    assert p_bw == [32, 64]
    assert d0_bw == [64, 128]
    assert d1_bw == [128, 256]
    # No max-logprobs bound -> hard cap (1024); a 256 logprobs cap keeps the
    # capture-set maximum (256) since it exceeds the server bound.
    assert get_p2d_max_beams() == 256
    assert get_p2d_max_beams(256) == 256
    assert get_p2d_max_beams(64) == 256

    # Triplet from trace_headers (the primary channel in vLLM 0.22.1).
    sched_trace = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                trace_headers={"vllm_gr_step_beam_widths": "128,256,32"},
                sampling_params=None,
            )
        ],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_step_beam_widths(
        SimpleNamespace(input_batch=None), sched_trace
    ) == StepBeamWidths(128, 256, 32)

    # Triplet from extra_body.
    sched = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                sampling_params=SimpleNamespace(
                    extra_body={"step_beam_widths": [128, 256, 32]}, beam_width=None, logprobs=None
                )
            )
        ],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_step_beam_widths(SimpleNamespace(input_batch=None), sched) == StepBeamWidths(
        128, 256, 32
    )

    # Scalar beam_width fallback -> uniform triplet.
    sched_scalar = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(sampling_params=SimpleNamespace(extra_body={}, beam_width=64))
        ],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_step_beam_widths(
        SimpleNamespace(input_batch=None), sched_scalar
    ) == StepBeamWidths(64, 64, 64)

    # Beam-style logprobs (max_tokens=1 + flat_logprobs) -> uniform triplet.
    sched_beam_lp = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                sampling_params=SimpleNamespace(logprobs=128, max_tokens=1, flat_logprobs=True)
            )
        ],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_step_beam_widths(
        SimpleNamespace(input_batch=None), sched_beam_lp
    ) == StepBeamWidths(128, 128, 128)

    # Standard autoregressive requests requesting logprobs are NEVER promoted
    # into P2D: no beam markers means no widths.
    for standard_sp in (
        SimpleNamespace(logprobs=5, max_tokens=100, flat_logprobs=False),
        SimpleNamespace(logprobs=5, max_tokens=1, flat_logprobs=False),
        SimpleNamespace(logprobs=5),
    ):
        sched_std = SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(sampling_params=standard_sp)],
            num_scheduled_tokens={"r1": 100},
        )
        assert match_p2d_step_beam_widths(SimpleNamespace(input_batch=None), sched_std) is None


def test_heterogeneous_triplets_fall_back_to_engine() -> None:
    """Conflicting per-request triplets must reject P2D, never max() them."""
    sched = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                trace_headers={"vllm_gr_step_beam_widths": "1,2,4"},
                sampling_params=None,
            ),
            SimpleNamespace(
                trace_headers={"vllm_gr_step_beam_widths": "4,2,1"},
                sampling_params=None,
            ),
        ],
        num_scheduled_tokens={"r1": 100, "r2": 100},
    )
    assert match_p2d_step_beam_widths(SimpleNamespace(input_batch=None), sched) is None

    # Identical triplets stay eligible.
    sched_same = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                trace_headers={"vllm_gr_step_beam_widths": "1,2,4"},
                sampling_params=None,
            ),
            SimpleNamespace(
                trace_headers={"vllm_gr_step_beam_widths": "1,2,4"},
                sampling_params=None,
            ),
        ],
        num_scheduled_tokens={"r1": 100, "r2": 100},
    )
    assert match_p2d_step_beam_widths(
        SimpleNamespace(input_batch=None), sched_same
    ) == StepBeamWidths(1, 2, 4)


def test_mixed_triplet_and_scalar_fall_back_to_engine() -> None:
    """A batch mixing triplet and scalar-width requests must reject P2D."""
    sched = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                trace_headers={"vllm_gr_step_beam_widths": "1,2,4"},
                sampling_params=None,
            ),
            SimpleNamespace(
                sampling_params=SimpleNamespace(extra_body={}, beam_width=64),
            ),
        ],
        num_scheduled_tokens={"r1": 100, "r2": 100},
    )
    assert match_p2d_step_beam_widths(SimpleNamespace(input_batch=None), sched) is None


def test_heterogeneous_requested_step_counts_reject_p2d() -> None:
    """Batches with conflicting P2D step budgets report ineligibility (-1)."""
    from vllm_gr.v1.worker.prefill_graph_common import (
        P2D_TOTAL_STEPS_HEADER,
        _requested_total_steps,
    )

    sched_mixed = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(trace_headers={P2D_TOTAL_STEPS_HEADER: "3"}),
            SimpleNamespace(trace_headers={P2D_TOTAL_STEPS_HEADER: "5"}),
        ]
    )
    assert _requested_total_steps(sched_mixed) == -1

    sched_same = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(trace_headers={P2D_TOTAL_STEPS_HEADER: "3"}),
            SimpleNamespace(trace_headers={P2D_TOTAL_STEPS_HEADER: "3"}),
        ]
    )
    assert _requested_total_steps(sched_same) == 3

    sched_unstamped = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(trace_headers={})])
    assert _requested_total_steps(sched_unstamped) is None

    # Mixed stamped/unstamped still yields the single stamped count, matching
    # the reviewer's requested contract (only conflicting stamps reject).
    sched_partial = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(trace_headers={P2D_TOTAL_STEPS_HEADER: "3"}),
            SimpleNamespace(trace_headers={}),
        ]
    )
    assert _requested_total_steps(sched_partial) == 3

    # num_logprobs fallback requires a batch-level beam marker.
    runner_with_batch = SimpleNamespace(
        input_batch=SimpleNamespace(num_logprobs={"r1": 32}, req_ids=["r1"])
    )
    sched_no_marker = SimpleNamespace(
        scheduled_new_reqs=[],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_step_beam_widths(runner_with_batch, sched_no_marker) is None
    sched_marked = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                trace_headers={"vllm_gr_step_beam_widths": "32,32,32"},
                sampling_params=None,
            )
        ],
        num_scheduled_tokens={"r1": 100},
    )
    assert match_p2d_step_beam_widths(runner_with_batch, sched_marked) == StepBeamWidths(32, 32, 32)


def test_pool_capacity_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p2d_common, "is_prefill_graph_enabled", lambda: True)

    sched_out = SimpleNamespace(
        num_scheduled_tokens={"r1": 100},
        scheduled_spec_decode_tokens=[],
        scheduled_encoder_inputs=[],
    )

    def _runner(max_logprobs: int) -> SimpleNamespace:
        return SimpleNamespace(
            input_batch=SimpleNamespace(
                req_id_to_index={"r1": 0},
                num_computed_tokens_cpu=[0],
                num_prompt_tokens=[100],
                req_ids=["r1"],
            ),
            vllm_config=SimpleNamespace(additional_config={}),
            _prefill_graph_runner=SimpleNamespace(max_bucket=512),
            model_config=SimpleNamespace(max_logprobs=max_logprobs),
        )

    # Default capture capacity is 128; a server cap of 128 rejects 256.
    assert get_p2d_max_beams(128) == 128
    assert (
        is_p2d_eligible(
            _runner(128),
            sched_out,
            num_reqs=1,
            total_tokens=100,
            bucket=None,
            step_bws=StepBeamWidths(128, 256, 32),
        )
        is False
    )
    # A larger --max-logprobs grows the pool capacity and the triplet passes.
    assert get_p2d_max_beams(256) == 256
    assert (
        is_p2d_eligible(
            _runner(256),
            sched_out,
            num_reqs=1,
            total_tokens=100,
            bucket=None,
            step_bws=StepBeamWidths(128, 256, 32),
        )
        is True
    )
    # Beyond the hard cap the gate rejects even with a huge server cap.
    assert get_p2d_max_beams(4096) == 256
    assert (
        is_p2d_eligible(
            _runner(4096),
            sched_out,
            num_reqs=1,
            total_tokens=100,
            bucket=None,
            step_bws=StepBeamWidths(128, 2048, 32),
        )
        is False
    )


def test_k_cand_equals_w_out_exactness() -> None:
    """K_cand = W_out must reproduce the full-vocabulary top-W_out exactly."""
    vocab = 16
    w0, w1 = 3, 5
    pool = _make_pool(max_beams=8, vocab=vocab)
    torch.manual_seed(0)
    logits = torch.randn(w0, vocab)
    cum_parent = torch.tensor([0.0, -1.5, -0.2])

    pool.logits_1[:w0].copy_(logits)
    pool.cum_lps_0[0, :w0].copy_(cum_parent)
    step1_prune_and_kv_gather(pool, 1, w0, w1, num_layers=1, num_kv_heads=2, head_dim=4)

    # Reference: full-vocab scores over all w0 x vocab candidates.
    lps_full = torch.log_softmax(logits, dim=-1)
    ref_scores = (cum_parent.unsqueeze(-1) + lps_full).reshape(-1)
    ref_val, ref_idx = torch.topk(ref_scores, k=w1)
    ref_parents = ref_idx // vocab
    ref_tokens = ref_idx % vocab

    assert torch.equal(pool.parents_1[0, :w1], ref_parents)
    assert torch.equal(pool.tokens_1[0, :w1], ref_tokens)
    assert torch.allclose(pool.cum_lps_1[0, :w1], ref_val)


def test_step1_kv_gather_mismatched_widths() -> None:
    num_layers, num_kv_heads, head_dim = 2, 2, 4
    w0 = 4

    def _run(w1: int) -> tuple[torch.Tensor, torch.Tensor]:
        pool = _make_pool(
            max_beams=8, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        # Distinct step-0 KV per beam: every element = the beam index.
        for b in range(w0):
            pool.suffix_kv[:, :, b, 0].fill_(float(b))
        torch.manual_seed(7)
        pool.logits_1[:w0].copy_(torch.randn(w0, 16))
        pool.cum_lps_0[0, :w0].fill_(0.0)
        step1_prune_and_kv_gather(
            pool, 1, w0, w1, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        return pool.suffix_kv[:, :, :w1, 0], pool.parents_1[0, :w1]

    # Invariant: reordered slot-0 KV of beam j equals the source beam's KV
    # recorded in parents_1[j], for both w1 == w0 and w1 < w0.
    for w1 in (w0, 2):
        gathered, parents = _run(w1)
        assert gathered.shape[2] == w1
        for j in range(w1):
            src = int(parents[j])
            assert gathered[0, 0, j, 0, 0].item() == float(src)


def test_step2_backtracking_to_roots() -> None:
    w0, w1, w2 = 8, 5, 3
    pool = _make_pool(max_beams=8)
    # Build a known tree: step-0 roots, step-1 parents, logits constant.
    pool.tokens_0[0, :w0] = torch.arange(100, 100 + w0)
    pool.parents_1[0, :w1] = torch.tensor([0, 2, 4, 6, 7])
    pool.tokens_1[0, :w1] = torch.arange(200, 200 + w1)
    # Strictly descending parent scores so best_p_2 == [0, 1, 2] deterministically.
    for p in range(w1):
        pool.cum_lps_1[0, p] = -10.0 * p
    # Step-2 logits: force best_p_2 = [0, 1, 2, 3, 4] (scores descending per
    # column since each parent's top token dominates with value 100 - p).
    vocab = 16
    logits_2 = torch.full((w1, vocab), 1.0)
    for p in range(w1):
        logits_2[p, 0] = 100.0 - p
    pool.logits_2[:w1].copy_(logits_2)
    step2_prune_and_backtrack(pool, 1, w0, w1, w2)

    # Each parent's top token wins -> best_p_2 == parent index (best_idx // w2
    # with K_cand = w2 and scores ordered by parent cum_lps + token value).
    for j in range(w2):
        # best_p_2[j] == j (strictly ordered parent scores) -> root = parents_1[j].
        root_parent = int(pool.parents_1[0, j])
        assert pool.final_tokens[0, j, 0].item() == pool.tokens_0[0, root_parent].item()
        assert pool.final_tokens[0, j, 1].item() == pool.tokens_1[0, j].item()
    assert pool.final_tokens[0, 0, 0].item() == pool.tokens_0[0, 0].item()
    assert pool.final_tokens[0, 1, 0].item() == pool.tokens_0[0, 2].item()
    assert pool.final_tokens[0, 2, 0].item() == pool.tokens_0[0, 4].item()


def test_monolithic_and_stepwise_bitwise_equivalence() -> None:
    vocab = 4
    beam_width = 2
    logits_vec = torch.arange(vocab, 0, -1, dtype=torch.float32)

    def _run_monolithic() -> P2DBufferPool:
        pool = _make_pool(max_beams=2, vocab=vocab)
        model = _ConstLogitsModel(logits_vec)
        runner = SimpleNamespace(model=model, vllm_config=None)
        logits_0 = model.compute_logits(torch.zeros(1, 4))
        pool.logits_0[:1].copy_(logits_0)
        from vllm_gr.v1.worker.prefill_graph_common import step0_select_topk

        step0_select_topk(logits_0=pool.logits_0, num_reqs=1, beam_width=beam_width, pool=pool)
        run_decode_passes(
            runner=runner,
            p2d_pool=pool,
            attn_meta_1=SimpleNamespace(),
            attn_meta_2=SimpleNamespace(),
            total_beam_tokens=beam_width,
            num_reqs=1,
            beam_width=beam_width,
            num_layers=1,
            num_kv_heads=2,
            head_dim=4,
            forward_ctx_fn=lambda r, m, nt: nullcontext(),
        )
        return pool

    def _run_stepwise() -> P2DBufferPool:
        pool = _make_pool(max_beams=2, vocab=vocab)
        model = _ConstLogitsModel(logits_vec)
        runner = SimpleNamespace(model=model, vllm_config=None)
        attn_1 = P2DMetadata(p2d_pool=pool, p2d_pass=1)
        attn_2 = P2DMetadata(p2d_pool=pool, p2d_pass=2)
        run_p2d_stepwise_tail(
            runner=runner,
            p2d_pool=pool,
            hidden=torch.zeros(1, 4),
            sample_indices=[0],
            attn_meta_1=attn_1,
            attn_meta_2=attn_2,
            num_reqs=1,
            w0=beam_width,
            w1=beam_width,
            w2=beam_width,
            num_layers=1,
            num_kv_heads=2,
            head_dim=4,
            forward_ctx_fn=lambda r, m, nt: nullcontext(),
            decode1_fn=lambda tokens: run_decode_pass_eager(
                runner, pool, attn_1, 1, tokens, lambda r, m, nt: nullcontext()
            ),
            decode2_fn=lambda tokens: run_decode_pass_eager(
                runner, pool, attn_2, 2, tokens, lambda r, m, nt: nullcontext()
            ),
        )
        return pool

    mono = _run_monolithic()
    step = _run_stepwise()
    assert torch.equal(mono.final_tokens[:1, :beam_width], step.final_tokens[:1, :beam_width])
    assert torch.equal(mono.final_cum_lps[:1, :beam_width], step.final_cum_lps[:1, :beam_width])


def test_buffer_reuse_across_width_triplets(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _make_pool(max_beams=get_p2d_max_beams(256), vocab=320)
    assert pool.max_beams == 256

    model = _ConstLogitsModel(torch.arange(320, 0, -1, dtype=torch.float32))
    runner = SimpleNamespace(model=model, vllm_config=None)

    def _run(w0: int, w1: int, w2: int) -> None:
        attn_1 = P2DMetadata(p2d_pool=pool, p2d_pass=1)
        attn_2 = P2DMetadata(p2d_pool=pool, p2d_pass=2)
        run_p2d_stepwise_tail(
            runner=runner,
            p2d_pool=pool,
            hidden=torch.zeros(1, 4),
            sample_indices=[0],
            attn_meta_1=attn_1,
            attn_meta_2=attn_2,
            num_reqs=1,
            w0=w0,
            w1=w1,
            w2=w2,
            num_layers=1,
            num_kv_heads=2,
            head_dim=4,
            forward_ctx_fn=lambda r, m, nt: nullcontext(),
            decode1_fn=lambda tokens: run_decode_pass_eager(
                runner, pool, attn_1, 1, tokens, lambda r, m, nt: nullcontext()
            ),
            decode2_fn=lambda tokens: run_decode_pass_eager(
                runner, pool, attn_2, 2, tokens, lambda r, m, nt: nullcontext()
            ),
        )

    # (32, 128, 32) then (128, 256, 64): same pool object, no reallocation.
    _run(32, 128, 32)
    assert torch.all(pool.final_tokens[0, :32, 2] >= 0)
    _run(128, 256, 64)
    assert torch.all(pool.final_tokens[0, :64, 2] >= 0)
    assert pool.max_beams == 256


def test_gpu_three_tier_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-uniform triplet routes to the stepwise tier; uncaptured steps to
    the eager-on-device tier; decode-step keys are (pass, num_reqs, W)."""
    from vllm_gr.v1.worker import gpu_model_runner_patch as gpu_patch

    calls: list[str] = []
    received: dict[str, Any] = {}

    class MockGraphRunner:
        def match_p2d_bucket(self, total_tokens: int) -> int | None:
            return None

        def match_bucket(self, total_tokens: int) -> int | None:
            return None

        def has_captured_graph(self, bucket: int, num_reqs: int) -> bool:
            return False

        def has_2d_graph(self, num_reqs: int, bw: int) -> bool:
            return False

        def has_decode_step_graph(self, pass_idx: int, num_reqs: int, bw: int) -> bool:
            return pass_idx == 1

        def execute_stepwise_p2d(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append("execute_stepwise_p2d")
            received.update(kwargs)
            return (
                torch.zeros((1, 32, 3), dtype=torch.int64),
                torch.zeros((1, 32), dtype=torch.float32),
            )

        def execute_eager_p2d_stepped(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append("execute_eager_p2d_stepped")
            received.update(kwargs)
            return (
                torch.zeros((1, 32, 3), dtype=torch.int64),
                torch.zeros((1, 32), dtype=torch.float32),
            )

        def get_output_buf(self, total_tokens: int, model_config: Any) -> torch.Tensor:
            return torch.zeros((total_tokens, 64), dtype=torch.bfloat16)

        def get_prefill_logit_indices(self, num_scheduled_tokens: Any) -> list[int]:
            counts = list(num_scheduled_tokens.values())
            return [sum(counts) - 1]

    p_runner = MockGraphRunner()
    monkeypatch.setattr(gpu_patch, "is_prefill_graph_enabled", lambda: True)
    monkeypatch.setattr(gpu_patch, "is_prefill_phase_batch", lambda self, sched: sched.pure)
    monkeypatch.setattr(gpu_patch, "get_prefill_graph_runner", lambda runner: p_runner)
    monkeypatch.setattr(
        gpu_patch,
        "match_p2d_step_beam_widths",
        lambda r, s: StepBeamWidths(32, 128, 32),
    )
    monkeypatch.setattr(gpu_patch, "is_p2d_eligible", lambda *args, **kwargs: True)

    def dummy_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append("orig_forward")
        return torch.zeros((100, 64))

    wrapped = gpu_patch.make_patched_model_forward_gpu(dummy_forward)
    runner = SimpleNamespace(
        _current_scheduler_output=SimpleNamespace(
            pure=True,
            total_num_scheduled_tokens=100,
            num_scheduled_tokens={"r1": 100},
        ),
        input_batch=SimpleNamespace(
            req_ids=["r1"],
            req_id_to_index={"r1": 0},
            num_computed_tokens_cpu=[0],
            block_table=None,
        ),
        model=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16, get_hidden_size=lambda: 64),
        device=torch.device("cpu"),
    )
    out = wrapped(runner, input_ids=torch.zeros(100), positions=torch.zeros(100))
    # Step 2 graph missing (has_decode_step_graph(2, ...) -> False) -> tier 3.
    assert "execute_eager_p2d_stepped" in calls
    assert "execute_stepwise_p2d" not in calls
    assert received.get("step_bws") == StepBeamWidths(32, 128, 32)
    assert out.shape[0] >= 100
    assert runner._current_p2d_result is not None


def test_decode_step_graph_key_collision() -> None:
    """(pass, num_reqs, W) keys never collide across (1, 64) vs (2, 32)."""
    graphs: dict[tuple[int, int, int], str] = {}
    graphs[(1, 1, 64)] = "pass1-b1-w64"
    graphs[(2, 2, 32)] = "pass2-b2-w32"
    assert graphs[(1, 1, 64)] != graphs[(2, 2, 32)]
    assert (1, 1, 64) != (2, 2, 32)
    # Equal token counts but distinct configs resolve to distinct keys.
    graphs[(2, 1, 64)] = "pass2-b1-w64"
    assert len(graphs) == 3


def test_get_p2d_pool_max_beams_cap_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that P2D pool max beams cap resolves from runner config or sys.argv, defaulting to 256."""
    import sys

    from vllm_gr.v1.worker.prefill_graph_common import get_p2d_pool_max_beams_cap

    # 1. Default fallback when no config or args present
    monkeypatch.setattr(sys, "argv", ["vllm-gr", "serve"])
    assert get_p2d_pool_max_beams_cap(None) == 256

    # 2. From sys.argv --beam-max-width
    monkeypatch.setattr(sys, "argv", ["vllm-gr", "serve", "--beam-max-width", "128"])
    assert get_p2d_pool_max_beams_cap(None) == 128
    assert get_p2d_max_beams(512, runner=None) == 128

    monkeypatch.setattr(sys, "argv", ["vllm-gr", "serve", "--beam-max-width=512"])
    assert get_p2d_pool_max_beams_cap(None) == 512
    assert get_p2d_max_beams(512, runner=None) == 512

    # 3. From runner vllm_config.additional_config["vllm_gr_beam_max_width"]
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"vllm_gr_beam_max_width": 64},
        )
    )
    assert get_p2d_pool_max_beams_cap(runner) == 64
