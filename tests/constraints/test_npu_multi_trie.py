# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"

"""NPU-specific routing, table ownership and legal-set scoring regressions."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
import torch

from tests.constraints.test_npu_stage_vocab import _candidate_set_op
from vllm_gr.constraints import loader
from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
from vllm_gr.constraints.backend import ConstraintMetadata, ConstraintOutput
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.constraints.npu_backend import NPUConstraintBackend
from vllm_gr.constraints.npu_tables import npu_trie_arrays
from vllm_gr.constraints.table import ConstraintTable

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]

TRIPLES = (
    ((11, 7, 0), (11, 7, 451), (11, 137, 53), (237, 399, 211)),
    ((11, 7, 101), (11, 7, 211), (53, 7, 101), (53, 137, 451)),
    ((237, 399, 0), (237, 7, 53), (399, 137, 101)),
)


class _Tokenizer:
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [int(token) for token in tokens]


def make_npu_tries() -> ConstraintTable:
    return ConstraintTableArtifactCompiler(
        _Tokenizer(), 512, tokenizer_digest="sha256:" + "b" * 64
    ).compile_tries(
        [
            compile_relations_json([[str(token) for token in triple] for triple in root])
            for root in TRIPLES
        ]
    )


def upload_npu_tries(
    table: ConstraintTable,
    monkeypatch: pytest.MonkeyPatch,
    *,
    compact: bool,
    op: Callable[..., tuple[torch.Tensor, torch.Tensor]] = _candidate_set_op,
) -> NPUConstraintBackend:
    monkeypatch.setattr(loader, "_synchronize_current_npu_stream", lambda device: None)
    backend, _ = loader._upload_npu_backend(  # noqa: SLF001
        table,
        device=torch.device("cpu"),
        op=op,
        max_top_k=3,
        max_rows=16,
        enable_npu_stage_vocab=compact,
    )
    backend._mark_ready()  # noqa: SLF001
    return backend


def test_each_root_rebases_every_csr_level_without_mutating_artifact() -> None:
    table = make_npu_tries()
    original = {name: getattr(table, name).tobytes() for name in table.tensor_names}
    for root, triples in enumerate(TRIPLES):
        first, offsets1, values1, keys, offsets2, values2 = npu_trie_arrays(table, root)
        assert first.tolist() == sorted({a for a, _b, _c in triples})
        assert offsets1[0] == offsets2[0] == 0
        assert offsets1[-1] == values1.size == keys.size
        assert offsets2[-1] == values2.size
        for a, b, _c in triples:
            assert values1[offsets1[a] : offsets1[a + 1]].tolist() == sorted(
                {bb for aa, bb, _cc in triples if aa == a}
            )
            pair = int(np.searchsorted(keys, a * 512 + b))
            assert keys[pair] == a * 512 + b
            assert values2[offsets2[pair] : offsets2[pair + 1]].tolist() == sorted(
                {cc for aa, bb, cc in triples if (aa, bb) == (a, b)}
            )
    assert original == {name: getattr(table, name).tobytes() for name in table.tensor_names}
    assert all(not getattr(table, name).flags.writeable for name in table.tensor_names)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("broadcast_temperature", [False, True])
def test_mixed_roots_preserve_row_order_prefix_isolation_and_single_root_scores(
    compact: bool, step: int, broadcast_temperature: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = make_npu_tries()
    calls: list[int] = []

    def observe(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(args[0].shape[0])
        tokens, scores = _candidate_set_op(*args)
        return tokens, scores

    backend = upload_npu_tries(table, monkeypatch, compact=compact, op=observe)
    roots = (1, 1, 0, 2, 2, 1)
    prefixes = torch.tensor(
        [[11, 7, 0], [53, 137, 0], [11, 7, 0], [237, 399, 0], [399, 137, 0], [11, 7, 0]],
        dtype=torch.int32,
    )
    temperatures = (
        torch.tensor([2.0])
        if broadcast_temperature
        else torch.tensor([0.0, 2.0, 0.5, 1.0, 3.0, 1.0])
    )
    full = torch.randn(len(roots), 512, generator=torch.Generator().manual_seed(913))
    vocab = backend.stage_vocab
    logits = (
        full
        if vocab is None
        else torch.nn.functional.pad(
            full[:, vocab.stage_token_ids[step]],
            (0, vocab.vocab_size - vocab.stage_token_ids[step].numel()),
            value=-float("inf"),
        )
    )
    metadata = ConstraintMetadata(
        prefixes, temperatures, step, 3, torch.tensor(roots, dtype=torch.int32)
    )
    try:
        output = backend.select_for_rows(logits, metadata, roots)
        assert calls == [2, 1, 2, 1]
        checked = backend.select(logits, metadata)
        for name in ("token_ids", "logprobs", "valid_mask"):
            torch.testing.assert_close(
                getattr(output, name), getattr(checked, name), rtol=0, atol=0
            )
        for row, root in enumerate(roots):
            first, offsets1, values1, pair_keys, offsets2, values2 = (
                torch.from_numpy(np.array(a, copy=True)) for a in npu_trie_arrays(table, root)
            )
            temp = temperatures if broadcast_temperature else temperatures[row : row + 1]
            tokens, scores = _candidate_set_op(
                full[row : row + 1],
                prefixes[row : row + 1].unsqueeze(1),
                first,
                offsets1,
                values1,
                pair_keys,
                offsets2,
                values2,
                temp,
                step,
                3,
                table.metadata.max_prefix1_degree,
                table.metadata.max_prefix2_degree,
            )
            valid = output.valid_mask[row]
            assert output.token_ids[row, valid].tolist() == tokens[0, valid].tolist()
            torch.testing.assert_close(output.logprobs[row], scores[0], rtol=0, atol=0)
        if step == 2:
            assert set(output.token_ids[0, output.valid_mask[0]].tolist()) == {101, 211}
            assert set(output.token_ids[2, output.valid_mask[2]].tolist()) == {0, 451}
    finally:
        backend.close()


def test_host_row_routing_never_reads_device_root_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = upload_npu_tries(make_npu_tries(), monkeypatch, compact=False)
    # Device metadata is intentionally opaque. The hot path must use the
    # validated immutable Host IDs, not tolist/item/bool on this field.
    metadata = ConstraintMetadata(
        torch.tensor([[11, 7, 0]] * 2, dtype=torch.int32),
        torch.ones(2),
        2,
        3,
        cast(torch.Tensor, object()),
    )
    try:
        output = backend.select_for_rows(torch.zeros(2, 512), metadata, (0, 1))
        assert output.token_ids[0, output.valid_mask[0]].tolist() == [0, 451]
        assert output.token_ids[1, output.valid_mask[1]].tolist() == [101, 211]
    finally:
        backend.close()


@pytest.mark.parametrize("roots", [(0,), (0, 3), (-1, 0), (True, 0)])
def test_host_root_validation_rejects_before_native_call(
    roots: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("invalid root routing must fail before a native call")

    backend = upload_npu_tries(make_npu_tries(), monkeypatch, compact=False, op=forbidden)
    try:
        with pytest.raises(ValueError, match="trie_id"):
            backend.select_for_rows(
                torch.zeros(2, 512),
                ConstraintMetadata(torch.zeros(2, 3, dtype=torch.int32), torch.ones(2), 0, 3),
                roots,
            )
    finally:
        backend.close()


def test_union_stage_vocabulary_contains_every_root_and_keeps_separate_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = make_npu_tries()
    backend = upload_npu_tries(table, monkeypatch, compact=True)
    try:
        vocab = backend.stage_vocab
        assert vocab is not None and vocab.vocab_size == 128
        assert [ids.tolist() for ids in vocab.stage_token_ids] == [
            [11, 53, 237, 399],
            [7, 137, 399],
            [0, 53, 101, 211, 451],
        ]
        assert len(vocab.trie_tables) == 3
        for root in range(3):
            first, offsets, values, keys, offsets2, third = vocab.tables_for_trie(root)
            assert int(offsets[-2]) == int(offsets[-1])
            assert (first < vocab.vocab_size - 1).all()
            assert int(offsets[0]) == int(offsets2[0]) == 0
            assert int(offsets[-1]) == values.numel() == keys.numel()
            assert int(offsets2[-1]) == third.numel()
        # The loader tracks every upload and also exposes stage ownership
        # names. Memory diagnostics must deduplicate those aliases.
        owned = vocab.owned_tensors()
        expected = sum(t.untyped_storage().nbytes() for t in owned.values())
        assert loader._unique_storage_bytes(owned) == expected  # noqa: SLF001
        canonical = [tensor for root in range(3) for tensor in backend._tables(root)]  # noqa: SLF001
        assert backend.info.device_memory_bytes == expected + sum(
            t.untyped_storage().nbytes() for t in canonical
        )
    finally:
        backend.close()
    assert backend.stage_vocab is None and backend._trie_tables is None  # noqa: SLF001


def test_readiness_covers_every_root_and_both_projection_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = make_npu_tries()
    observed: list[tuple[int, int, bool]] = []
    monkeypatch.setattr(loader, "_synchronize_current_npu_stream", lambda device: None)
    backend, cases = loader._upload_npu_backend(  # noqa: SLF001
        table,
        device=torch.device("cpu"),
        op=_candidate_set_op,
        max_top_k=3,
        max_rows=2,
        enable_npu_stage_vocab=True,
    )
    original = backend._select_for_readiness  # noqa: SLF001

    def observe(logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        assert metadata.trie_ids is not None
        observed.append((int(metadata.trie_ids[0]), metadata.current_step, logits.shape[1] == 128))
        return original(logits, metadata)

    monkeypatch.setattr(
        NPUConstraintBackend,
        "_select_for_readiness",
        lambda self, logits, metadata: observe(logits, metadata),
    )
    try:
        loader._run_npu_readiness(  # noqa: SLF001
            backend,
            cases=cases,
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_rows=2,
        )
        assert set(observed) == {
            (root, step, compact)
            for root in range(3)
            for step in range(3)
            for compact in (False, True)
        }
        assert len(observed) == 18
    finally:
        backend.close()


@pytest.mark.parametrize("root", [None, -1, 3])
def test_checked_selection_validates_root_ranges_before_native_call(
    root: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("checked invalid IDs must not call the native operator")

    backend = upload_npu_tries(make_npu_tries(), monkeypatch, compact=False, op=forbidden)
    metadata = ConstraintMetadata(
        torch.zeros(1, 3, dtype=torch.int32),
        torch.ones(1),
        0,
        3,
        None if root is None else torch.tensor([root], dtype=torch.int32),
    )
    try:
        with pytest.raises(ValueError, match="trie_id"):
            backend.select(torch.zeros(1, 512), metadata)
    finally:
        backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.select(torch.zeros(1, 512), metadata)
    with pytest.raises(RuntimeError, match="closed"):
        backend.select_for_rows(torch.zeros(1, 512), metadata, (0,))


def test_reserved_last_prefix_guard_covers_nonzero_roots() -> None:
    table = ConstraintTableArtifactCompiler(
        _Tokenizer(), 512, tokenizer_digest="sha256:" + "b" * 64
    ).compile_tries(
        [
            compile_relations_json([["11", "7", "0"]]),
            compile_relations_json([["511", "7", "101"]]),
        ]
    )
    with pytest.raises(RuntimeError, match="vocab_size - 1"):
        loader._reject_unsupported_last_vocab_prefix(table)  # noqa: SLF001


@pytest.mark.parametrize("failure_index", [7, 18, 19, 25, 36, 37, 40, 41])
def test_multi_root_partial_upload_releases_owned_tensors_with_original_error(
    failure_index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = make_npu_tries()
    error = RuntimeError("multi-root upload failed")
    references: list[weakref.ReferenceType[torch.Tensor]] = []
    calls = 0

    def upload(array: Any, device: torch.device) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == failure_index:
            raise error
        tensor = torch.from_numpy(np.array(array, copy=True))
        references.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(loader, "_upload_table_tensor", upload)
    monkeypatch.setattr(loader, "_synchronize_current_npu_stream", lambda device: None)
    with pytest.raises(RuntimeError) as raised:
        loader._upload_npu_backend(  # noqa: SLF001
            table,
            device=torch.device("cpu"),
            op=_candidate_set_op,
            max_top_k=3,
            max_rows=2,
            enable_npu_stage_vocab=True,
        )
    assert raised.value is error
    gc.collect()
    assert all(reference() is None for reference in references)
