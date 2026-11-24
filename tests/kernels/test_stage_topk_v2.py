# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Stage-specialized kernels preserve the authoritative device step and exact scores."""

# mypy: disable-error-code="no-untyped-def,no-untyped-call"

import pytest
import torch

from tests.kernels.test_rec_constrained_topk_cuda import _wide_window_inputs
from vllm_gr.ops.gpu.rec_constrained_topk import (
    _rec_constrained_topk_cuda_out_unchecked as select,
)

pytestmark = [
    pytest.mark.l2,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [1, 127, 128, 129])
@pytest.mark.parametrize("case", ["normal", "large", "ties", "invalid", "nan"])
def test_stage_capacities_and_graph_replay(dtype, k, case):
    args = list(_wide_window_inputs(dtype, 8, 24576, 5939))
    # Complete degree bounds differ substantially, as in the real OneRec table.
    args[3].fill_(478)
    args[3][0] = 0
    args[4] = args[4][:478]
    args[6][1] = 34
    args[7] = args[7][:34]
    if case == "ties":
        args[0].zero_()
    elif case == "large":
        args[0].mul_(10)
    elif case == "invalid":
        args[1][-1].fill_(-1)
    elif case == "nan":
        args[0][0, 1] = float("nan")
    step = torch.zeros(1, dtype=torch.int32, device="cuda")

    def output():
        return (
            torch.empty((8, k), dtype=torch.int32, device="cuda"),
            torch.empty((8, k), dtype=torch.float32, device="cuda"),
            torch.empty((8, k), dtype=torch.bool, device="cuda"),
            torch.empty(8, dtype=torch.int32, device="cuda"),
        )

    baseline, specialized = output(), output()

    def run(out, specialize):
        select(
            *args,
            0,
            k,
            478,
            34,
            *out,
            vocab_offset=0,
            table_vocab_size=24576,
            out_wire_logprobs=None,
            device_step=step,
            specialize_steps=specialize,
        )

    run(specialized, True)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(specialized, True)
    for value in (0, 1, 2, 1, -1, 5):
        step.fill_(value)
        run(baseline, False)
        graph.replay()
        for old, new in zip(baseline, specialized):
            torch.testing.assert_close(new, old, rtol=0, atol=0, equal_nan=True)


def test_stage_specialization_preserves_multi_trie_roots():
    from vllm_gr.constraints.backend import ConstraintMetadata
    from vllm_gr.constraints.loader import _upload_cuda_backend

    table = _stage_tries()
    # The larger root requires capacities (16, 8, 4) at K=4. Ascending
    # logits make candidates beyond a wrongly truncated capacity win Top-K.
    assert table.root_offsets.tolist() == [0, 3, 12]
    assert (table.metadata.max_prefix1_degree, table.metadata.max_prefix2_degree) == (5, 2)
    backend, _ = _upload_cuda_backend(table, device=torch.device("cuda"), max_top_k=4, max_rows=2)
    backend._mark_ready()
    logits = torch.arange(32, dtype=torch.bfloat16, device="cuda").reshape(2, 16)
    metadata = ConstraintMetadata(
        torch.tensor([[1, 2, 0]] * 2, dtype=torch.int32, device="cuda"),
        torch.ones(2, device="cuda"),
        0,
        4,
        torch.arange(2, dtype=torch.int32, device="cuda"),
    )

    def output():
        return {
            "out_token_ids": torch.empty((2, 4), dtype=torch.int32, device="cuda"),
            "out_logprobs": torch.empty((2, 4), device="cuda"),
            "out_valid_mask": torch.empty((2, 4), dtype=torch.bool, device="cuda"),
            "out_valid_counts": torch.empty(2, dtype=torch.int32, device="cuda"),
        }

    expected, actual = output(), output()
    step = torch.zeros(1, dtype=torch.int32, device="cuda")
    for value in (0, 1, 2, 0):
        step.fill_(value)
        backend.select_out_unchecked(logits, metadata, **expected, device_step=step)
        backend.select_out_unchecked(
            logits, metadata, **actual, device_step=step, specialize_steps=True
        )
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    backend.close()


def _stage_tries(prefix1_degree=5, trie_count=2):
    from tests.constraints.test_multi_trie import DIGEST, ToyTokenizer
    from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
    from vllm_gr.constraints.json_compiler import compile_relations_json

    small = [(a, b, 7) for a in (1, 2, 3) for b in (2, 3)]
    large = [(a, b, c) for a in range(1, 10) for b in range(2, 2 + prefix1_degree) for c in (0, 1)]
    compiler = ConstraintTableArtifactCompiler(ToyTokenizer(), 16, tokenizer_digest=DIGEST)
    return compiler.compile_tries(
        [
            compile_relations_json([[str(token) for token in sid] for sid in source])
            for source in [small, large][-trie_count:]
        ]
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("trie_count,prefix1_degree,max_top_k", [(1, 5, 129), (2, 9, 256)])
def test_stage_warmup_covers_request_variants(
    monkeypatch, dtype, trie_count, prefix1_degree, max_top_k
):
    from vllm_gr.constraints.backend import ConstraintMetadata
    from vllm_gr.constraints.loader import _upload_cuda_backend
    from vllm_gr.ops.gpu.rec_constrained_topk import _stage_topk_kernel

    backend, _ = _upload_cuda_backend(
        _stage_tries(prefix1_degree, trie_count),
        device=torch.device("cuda"),
        max_top_k=max_top_k,
        max_rows=4,
    )
    backend._mark_ready()
    # Start with no in-process specialization cache, even after another test
    # has used the same table capacities. Disk-cache hits still enter JIT.
    _stage_topk_kernel.device_caches[torch.accelerator.current_device_index()][0].clear()
    try:
        backend.warmup_stage_topk(dtype)

        def unexpected_compile(*args, **kwargs):
            pytest.fail("stage Top-K entered JIT compilation after backend warmup")

        monkeypatch.setattr(_stage_topk_kernel, "_do_compile", unexpected_compile)
        step = torch.zeros(1, dtype=torch.int32, device="cuda")
        # Non-power-of-two K, batch sizes and full/sliced vocabulary layouts
        # must reuse the variants warmed with one row and power-of-two K.
        for k in range(1, max_top_k + 1):
            rows = 1 if k % 2 else 4
            vocab = 16 if k % 2 else backend.vocab_slice[1]
            logits = torch.zeros((rows, vocab), dtype=dtype, device="cuda")
            metadata = ConstraintMetadata(
                torch.tensor([[1, 2, 0]] * rows, dtype=torch.int32, device="cuda"),
                torch.ones(rows, device="cuda"),
                0,
                k,
                torch.arange(rows, dtype=torch.int32, device="cuda") % trie_count,
            )
            outputs = {
                "out_token_ids": torch.empty((rows, k), dtype=torch.int32, device="cuda"),
                "out_logprobs": torch.empty((rows, k), device="cuda"),
                "out_valid_mask": torch.empty((rows, k), dtype=torch.bool, device="cuda"),
                "out_valid_counts": torch.empty(rows, dtype=torch.int32, device="cuda"),
            }
            for stage in (0, 1, 2):
                step.fill_(stage)
                backend.select_out_unchecked(
                    logits, metadata, **outputs, device_step=step, specialize_steps=True
                )
        torch.accelerator.synchronize()
    finally:
        backend.close()
