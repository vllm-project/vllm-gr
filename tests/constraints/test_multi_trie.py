# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from vllm_gr.config.models import (
    MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
    ConstraintTableBackendEnum,
    ConstraintTableConfig,
)
from vllm_gr.constraints.artifact import (
    load_constraint_table_artifact,
    save_constraint_table_artifact,
)
from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
from vllm_gr.constraints.backend import (
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
)
from vllm_gr.constraints.cpu_backend import CPUConstraintBackend
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.constraints.loader import _upload_cuda_backend, load_constraint_backend
from vllm_gr.constraints.table import ConstraintTable, validate_constraint_table
from vllm_gr.ops.cuda.rec_constrained_topk import rec_constrained_topk_cuda

SIDS = (
    ((1, 2, 0), (1, 2, 1), (1, 4, 3), (3, 5, 1), (3, 5, 4)),
    ((1, 2, 5), (1, 4, 0), (4, 0, 2), (4, 0, 3), (4, 5, 1)),
)
DIGEST = "sha256:" + "b" * 64


class ToyTokenizer:
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [int(token) for token in tokens]


def make_tries(count: int = 2) -> ConstraintTable:
    compiler = ConstraintTableArtifactCompiler(ToyTokenizer(), 6, tokenizer_digest=DIGEST)
    documents = [
        compile_relations_json([[str(x) for x in sid] for sid in source]) for source in SIDS[:count]
    ]
    return compiler.compile_tries(documents)


def test_compiler_preserves_root_identity_for_every_prefix() -> None:
    table = make_tries()
    assert len(table.tensor_names) == 7
    assert table.root_offsets.tolist() == [0, 2, 4]
    assert table.prefix1_offsets.size == 2 * 6 + 1
    for root, sids in enumerate(SIDS):
        for a in range(6):
            row = root * 6 + a
            actual = table.prefix1_values[
                table.prefix1_offsets[row] : table.prefix1_offsets[row + 1]
            ]
            assert actual.tolist() == sorted({b for aa, b, c in sids if aa == a})
            for b in range(6):
                key = (root * 6 + a) * 6 + b
                row = np.searchsorted(table.prefix1_pair_keys, key)
                cs = []
                if row < table.prefix1_pair_keys.size and table.prefix1_pair_keys[row] == key:
                    cs = table.prefix2_values[
                        table.prefix2_value_offsets[row] : table.prefix2_value_offsets[row + 1]
                    ].tolist()
                assert cs == sorted({c for aa, bb, c in sids if (aa, bb) == (a, b)})


@pytest.mark.parametrize("count", [1, 2])  # type: ignore[misc]
@pytest.mark.parametrize("step", [0, 1, 2])  # type: ignore[misc]
def test_v2_artifact_roundtrip_and_cpu_query(tmp_path: Path, count: int, step: int) -> None:
    table = make_tries(count)
    path = tmp_path / "roots.safetensors"
    digest = save_constraint_table_artifact(table, path)
    data = path.read_bytes()
    header_size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + header_size])
    assert header["__metadata__"]["trie_count"] == str(count)
    loaded = load_constraint_table_artifact(
        path,
        expected_digest=digest,
        expected_tokenizer_digest=DIGEST,
        expected_vocab_size=6,
        expected_format=MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
    )
    assert loaded.metadata.trie_count == count
    for name in table.tensor_names:
        assert np.array_equal(getattr(loaded, name), getattr(table, name))
        with pytest.raises(ValueError):
            getattr(loaded, name).setflags(write=True)
    backend = load_constraint_backend(
        ConstraintTableConfig(
            max_top_k=3,
            enabled=True,
            backend="cpu",
            path=str(path),
            format=MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
            artifact_digest=digest,
            tokenizer_digest=DIGEST,
        ),
        device=torch.device("cpu"),
        vocab_size=6,
        dtype=torch.float32,
        max_rows=count,
    )
    output = backend.select(
        torch.arange(6, dtype=torch.float32).repeat(count, 1),
        ConstraintMetadata(
            torch.tensor([[1, 2, 0]] * count, dtype=torch.int32),
            torch.ones(count),
            step,
            3,
            torch.arange(count, dtype=torch.int32),
        ),
    )
    prefix = (1, 2, 0)
    for trie_id, (row, mask) in enumerate(zip(output.token_ids, output.valid_mask)):
        expected = sorted(
            {sid[step] for sid in SIDS[trie_id] if sid[:step] == prefix[:step]},
            reverse=True,
        )[:3]
        assert row[mask].tolist() == expected
    backend.close()


@pytest.mark.parametrize("count", [1, 2])  # type: ignore[misc]
@pytest.mark.parametrize("mode", ["cpu", "cuda", "cuda_out", "cuda_raw"])  # type: ignore[misc]
def test_checked_backends_share_root_defaults_and_range_errors(count: int, mode: str) -> None:
    if mode != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    table = make_tries(count)
    device = torch.device("cpu" if mode == "cpu" else "cuda")
    backend: Any
    if mode == "cpu":
        backend = CPUConstraintBackend(
            table,
            ConstraintBackendInfo(
                table.metadata, ConstraintTableBackendEnum.CPU, 2, 1, table.total_bytes, 0
            ),
        )
    else:
        backend, _ = _upload_cuda_backend(table, device=device, max_top_k=2, max_rows=1)
        backend._mark_ready()
    logits = torch.arange(6, dtype=torch.float16, device=device).reshape(1, 6)
    try:
        for root in (None, 0, count - 1, -1, count):
            metadata = ConstraintMetadata(
                torch.tensor([[1, 2, 0]], dtype=torch.int32, device=device),
                torch.ones(1, device=device),
                2,
                2,
                None if root is None else torch.tensor([root], dtype=torch.int32, device=device),
            )

            def select() -> Any:
                if mode == "cuda_raw":
                    tokens, scores, valid, _ = rec_constrained_topk_cuda(
                        logits,
                        metadata.prefix_token_ids,
                        *backend._tables(),
                        metadata.temperatures,
                        metadata.current_step,
                        metadata.top_k,
                        table.metadata.max_prefix1_degree,
                        table.metadata.max_prefix2_degree,
                        trie_ids=metadata.trie_ids,
                        first_token_offsets=backend._first_token_offsets,
                        max_first_degree=table.max_first_degree,
                    )
                    return ConstraintOutput(tokens, scores, valid)
                if mode != "cuda_out":
                    return backend.select(logits, metadata)
                return backend.select_out(
                    logits,
                    metadata,
                    out_token_ids=torch.empty((1, 2), dtype=torch.int32, device=device),
                    out_logprobs=torch.empty((1, 2), dtype=torch.float32, device=device),
                    out_valid_mask=torch.empty((1, 2), dtype=torch.bool, device=device),
                    out_valid_counts=torch.empty(1, dtype=torch.int32, device=device),
                )

            if root is None and count > 1:
                with pytest.raises(ValueError, match="requires trie_ids"):
                    select()
            elif root is not None and not 0 <= root < count:
                with pytest.raises(ValueError, match="trie_id is outside"):
                    select()
            else:
                output = select()
                assert output.token_ids[output.valid_mask].tolist() == (
                    [5] if root == 1 else [1, 0]
                )
    finally:
        backend.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")  # type: ignore[misc]
def test_worker_unchecked_selection_skips_public_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.constraints import cuda_backend

    def reject_checked_call(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Worker sampling must not call checked validation")

    monkeypatch.setattr(cuda_backend, "validate_select_inputs", reject_checked_call)
    backend, _ = _upload_cuda_backend(
        make_tries(), device=torch.device("cuda"), max_top_k=2, max_rows=1
    )
    backend._mark_ready()
    tokens = torch.empty((1, 2), dtype=torch.int32, device="cuda")
    scores = torch.empty((1, 2), dtype=torch.float32, device="cuda")
    valid = torch.empty((1, 2), dtype=torch.bool, device="cuda")
    counts = torch.empty(1, dtype=torch.int32, device="cuda")
    try:
        backend.select_out_unchecked(
            torch.arange(6, dtype=torch.float16, device="cuda").reshape(1, 6),
            ConstraintMetadata(
                torch.tensor([[1, 2, 0]], dtype=torch.int32, device="cuda"),
                torch.ones(1, device="cuda"),
                2,
                2,
                torch.zeros(1, dtype=torch.int32, device="cuda"),
            ),
            out_token_ids=tokens,
            out_logprobs=scores,
            out_valid_mask=valid,
            out_valid_counts=counts,
        )
        assert tokens[valid].tolist() == [1, 0]
    finally:
        backend.close()


def test_rejects_cross_root_key_and_boundary_corruption() -> None:
    table = make_tries()
    keys = table.prefix1_pair_keys.copy()
    keys[3] -= 36
    with pytest.raises(ValueError, match="pair_keys"):
        validate_constraint_table(replace(table, prefix1_pair_keys=keys))
    with pytest.raises(ValueError, match="first_token_offsets"):
        validate_constraint_table(
            replace(table, first_token_offsets=np.array([0, 2, 5], dtype=np.int32))
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")  # type: ignore[misc]
@pytest.mark.parametrize("count", [1, 2])  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_cuda_loader_readiness_and_all_stages_match_scalar_reference(
    tmp_path: Path, count: int, dtype: torch.dtype
) -> None:
    table = make_tries(count)
    path = tmp_path / "roots.safetensors"
    digest = save_constraint_table_artifact(table, path)
    backend = load_constraint_backend(
        ConstraintTableConfig(
            max_top_k=4,
            enabled=True,
            path=str(path),
            format=MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
            artifact_digest=digest,
            tokenizer_digest=DIGEST,
        ),
        device=torch.device("cuda"),
        vocab_size=6,
        dtype=dtype,
        max_rows=count,
    )
    logits = torch.tensor([[0.1, 0.7, -0.3, 0.2, 0.5, -0.9]] * count, device="cuda", dtype=dtype)
    for step, prefix in enumerate(((), (1,), (1, 2))):
        output = backend.select(
            logits,
            ConstraintMetadata(
                torch.tensor(
                    [list(prefix) + [0] * (3 - step)] * count, device="cuda", dtype=torch.int32
                ),
                torch.ones(count, device="cuda"),
                step,
                4,
                torch.arange(count, device="cuda", dtype=torch.int32),
            ),
        )
        expected_scores = logits.float().log_softmax(-1)
        for root in range(count):
            candidates = {sid[step] for sid in SIDS[root] if sid[:step] == prefix}
            expected = sorted(candidates, key=lambda token: -float(logits[root, token]))[:4]
            valid = output.valid_mask[root]
            assert output.token_ids[root, valid].tolist() == expected
            assert torch.allclose(
                output.logprobs[root, valid], expected_scores[root, expected], atol=1e-6
            )
    assert (
        backend._first_token_offsets is None
        if count == 1
        else backend._first_token_offsets is not None
    )
    backend.close()
