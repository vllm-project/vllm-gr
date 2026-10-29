# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real-model smoke: random tiny Llama, no downloads or business data.

Run in a fresh process with VLLM_GR_RUN_MODEL_SMOKE=1 and
VLLM_ENABLE_V1_MULTIPROCESSING=0. This is execution correctness, not accuracy.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("VLLM_GR_RUN_MODEL_SMOKE") != "1" or not torch.cuda.is_available(),
    reason="opt-in CUDA model smoke",
)


def build_tiny_engine(tmp_path, *, capacity=4, dense_catalog=False, graph=True):
    """Build a random tiny Llama and a matching CUDA Constraint Table engine."""
    from dataclasses import replace

    import numpy as np
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from tests.constraints.test_table import make_table
    from vllm_gr.constraints.artifact import save_constraint_table_artifact
    from vllm_gr.entrypoints.gr import GRLLM

    torch.manual_seed(7)
    model_path = tmp_path / "model"
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
    ).save_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"tok{i}": i for i in range(128)}, unk_token="tok0")),
        unk_token="tok0",
        bos_token="tok1",
        eos_token="tok2",
    )
    tokenizer.save_pretrained(model_path)
    original_table = make_table()
    table = make_table(
        metadata=replace(original_table.metadata, vocab_size=128, total_bytes=596),
        prefix1_offsets=np.pad(original_table.prefix1_offsets, (0, 122), mode="edge"),
        prefix1_pair_keys=np.asarray([130, 132, 389], dtype=np.int64),
    )
    if dense_catalog:
        # A full 16 x 16 x 16 catalog keeps irregular widths genuinely active.
        from vllm_gr.constraints.table import ConstraintTable, ConstraintTableMetadata

        degree = 16
        arrays = dict(
            first_token_ids=np.arange(degree, dtype=np.int32),
            prefix1_offsets=np.minimum(np.arange(129), degree).astype(np.int32) * degree,
            prefix1_values=np.tile(np.arange(degree, dtype=np.int32), degree),
            prefix1_pair_keys=np.asarray(
                [a * 128 + b for a in range(degree) for b in range(degree)], dtype=np.int64
            ),
            prefix2_value_offsets=np.arange(degree * degree + 1, dtype=np.int32) * degree,
            prefix2_values=np.tile(np.arange(degree, dtype=np.int32), degree * degree),
        )
        table = ConstraintTable(
            **arrays,
            metadata=ConstraintTableMetadata(
                format=table.metadata.format,
                artifact_digest=None,
                tokenizer_digest=table.metadata.tokenizer_digest,
                vocab_size=128,
                max_prefix1_degree=degree,
                max_prefix2_degree=degree,
                total_bytes=sum(a.nbytes for a in arrays.values()),
            ),
        )
    artifact = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(table, artifact)
    config = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "graph_enabled": graph,
            "worker_decision": True,
            "max_width": capacity,
            "max_decode_steps": 3,
        },
        "constraint_table": {
            "enabled": True,
            "backend": "cuda",
            "max_top_k": capacity,
            "path": str(artifact),
            "format": "constraint_table_v1",
            "artifact_digest": digest,
            "tokenizer_digest": table.metadata.tokenizer_digest,
        },
    }
    llm = GRLLM(
        model=str(model_path),
        dtype="float16",
        vllm_gr_config=config,
        async_scheduling=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_model_len=64,
        max_num_batched_tokens=128,
        max_num_seqs=max(16, 4 * capacity),
        max_logprobs=capacity,
        gpu_memory_utilization=0.1,
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
    )
    return llm, model_path


def test_tiny_model_dynamic_decode(tmp_path):
    from vllm.inputs import tokens_input

    from vllm_gr.sampling_params import BeamSearchParams

    llm, _ = build_tiny_engine(tmp_path)
    with llm:
        legal = {(1, 2, 0), (1, 2, 1), (1, 4, 3), (3, 5, 1), (3, 5, 4)}
        for batch in (1, 2, 4, 1):
            prompts = [tokens_input([1, 4, 2] + [3] * i) for i in range(batch)]
            for widths in ([4, 2, 1], [1, 3, 4], [3, 3, 2], [4, 4, 4]):
                params = BeamSearchParams(
                    beam_width=4,
                    dynamic_beam_widths=widths,
                    result_width=min(2, widths[-1]),
                    max_tokens=3,
                    ignore_eos=True,
                    temperature=0.0,
                )
                outputs = llm.beam_search_v1(prompts, params, output_mode="full")
                runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
                graph = runner._beam_decode_graph
                expected_shapes = {(batch, width) for width in widths[:-1]}
                assert {
                    (shape.batch_size, shape.beam_width) for shape in graph._bindings
                } == expected_shapes
                captures = {
                    shape: id(next(iter(entries.values())).cudagraph)
                    for shape, entries in graph._registries.items()
                    if entries
                }
                repeated = llm.beam_search_v1(prompts, params, output_mode="full")
                assert captures == {
                    shape: id(next(iter(entries.values())).cudagraph)
                    for shape, entries in graph._registries.items()
                    if entries
                }
                assert [[tuple(s.tokens) for s in out.sequences] for out in repeated] == [
                    [tuple(s.tokens) for s in out.sequences] for out in outputs
                ]
                assert len(outputs) == batch
                for output in outputs:
                    sequences = [tuple(sequence.tokens[-3:]) for sequence in output.sequences]
                    assert 1 <= len(sequences) <= params.result_width
                    assert len(set(sequences)) == len(sequences)
                    assert set(sequences) <= legal
