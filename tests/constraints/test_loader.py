# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"

from __future__ import annotations

import gc
import importlib
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintTableBackendEnum, ConstraintTableConfig
from vllm_gr.constraints import loader
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.backend import INVALID_LOGIT, ConstraintBackendInfo
from vllm_gr.constraints.npu_backend import NPUConstraintBackend
from vllm_gr.constraints.table import CONSTRAINT_TABLE_TENSOR_NAMES, ConstraintTable
from vllm_gr.constraints.validity import (
    constraint_candidate_counts,
    valid_mask_from_counts,
)
from vllm_gr.ops.rec_constrained_topk import rec_constrained_topk_torch

npu_op_module = importlib.import_module("vllm_gr.ops.npu.rec_constrained_topk")


def _config(path: Path, digest: str, **overrides: object) -> ConstraintTableConfig:
    values: dict[str, object] = {
        "enabled": True,
        "backend": ConstraintTableBackendEnum.CPU,
        "path": str(path),
        "artifact_digest": digest,
        "tokenizer_digest": "sha256:" + "b" * 64,
        "max_top_k": 4,
    }
    values.update(overrides)
    return ConstraintTableConfig(**values)


def test_disabled_loader_returns_before_validation_or_io(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ConstraintTableConfig(max_top_k=1, enabled=False)
    monkeypatch.setattr(
        loader,
        "load_constraint_table_artifact",
        lambda *args, **kwargs: pytest.fail("Artifact I/O must not run"),
    )
    assert (
        loader.load_constraint_backend(
            config,
            device="not a device",
            vocab_size=-1,
            dtype=torch.int64,
            max_rows=0,
        )
        is None
    )


def test_cpu_loader_constructs_info_from_canonical_artifact(tmp_path: Path) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    backend = loader.load_constraint_backend(
        _config(path, digest),
        device=torch.device("cpu"),
        vocab_size=6,
        dtype=torch.float32,
        max_rows=7,
    )
    assert backend is not None
    assert backend.info.backend is ConstraintTableBackendEnum.CPU
    assert backend.info.host_memory_bytes == make_table().total_bytes
    assert backend.info.device_memory_bytes == 0
    assert backend.info.max_rows == 7


def test_npu_top_k_limit_fails_before_artifact_io(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ConstraintTableConfig(
        enabled=True,
        backend=ConstraintTableBackendEnum.NPU,
        path="/does/not/exist",
        artifact_digest="sha256:" + "a" * 64,
        tokenizer_digest="sha256:" + "b" * 64,
        max_top_k=257,
    )
    monkeypatch.setattr(
        loader,
        "load_constraint_table_artifact",
        lambda *args, **kwargs: pytest.fail("Artifact I/O must not run"),
    )
    with pytest.raises(ValueError, match="256"):
        loader.load_constraint_backend(
            config,
            device=torch.device("privateuseone"),
            vocab_size=6,
            dtype=torch.float16,
            max_rows=2,
        )


def test_strict_npu_resolver_never_returns_torch_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(npu_op_module, "_custom_ops", lambda: None)
    with pytest.raises(RuntimeError, match="custom_ops_lib"):
        npu_op_module.resolve_rec_constrained_topk_npu()


def test_npu_known_last_vocab_prefix_limitation_fails_closed() -> None:
    table = make_table(
        first_token_ids=np.asarray([1, 3, 5], dtype=np.int32),
        prefix1_offsets=np.asarray([0, 0, 2, 2, 3, 3, 4], dtype=np.int32),
        prefix1_values=np.asarray([2, 4, 5, 0], dtype=np.int32),
        prefix1_pair_keys=np.asarray([8, 10, 23, 30], dtype=np.int64),
        prefix2_value_offsets=np.asarray([0, 2, 3, 5, 6], dtype=np.int32),
        prefix2_values=np.asarray([0, 1, 3, 1, 4, 2], dtype=np.int32),
    )
    with pytest.raises(RuntimeError, match="vocab_size - 1"):
        loader._reject_unsupported_last_vocab_prefix(table)  # noqa: SLF001


@pytest.mark.parametrize("failure_index", range(1, 7))
def test_upload_failure_rethrows_original_exception_and_releases_tensors(
    failure_index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = make_table()
    error = RuntimeError("third upload failed")
    references: list[weakref.ReferenceType[torch.Tensor]] = []
    calls = 0

    def upload(array: np.ndarray, device: torch.device) -> torch.Tensor:
        nonlocal calls
        del array, device
        calls += 1
        if calls == failure_index:
            raise error
        tensor = torch.zeros(1)
        references.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(loader, "_upload_table_tensor", upload)
    with pytest.raises(RuntimeError) as raised:
        loader._upload_npu_backend(  # noqa: SLF001
            table,
            device=torch.device("cpu"),
            op=lambda *args: (torch.empty(0), torch.empty(0)),
            max_top_k=2,
            max_rows=2,
        )
    assert raised.value is error
    gc.collect()
    assert all(reference() is None for reference in references)


def test_upload_sync_failure_releases_all_created_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = make_table()
    error = RuntimeError("upload stream sync failed")
    references: list[weakref.ReferenceType[torch.Tensor]] = []

    def upload(array: np.ndarray, device: torch.device) -> torch.Tensor:
        del array, device
        tensor = torch.zeros(1)
        references.append(weakref.ref(tensor))
        return tensor

    def fail_sync(device: torch.device) -> None:
        del device
        raise error

    monkeypatch.setattr(loader, "_upload_table_tensor", upload)
    monkeypatch.setattr(loader, "_synchronize_current_npu_stream", fail_sync)
    with pytest.raises(RuntimeError) as raised:
        loader._upload_npu_backend(  # noqa: SLF001
            table,
            device=torch.device("cpu"),
            op=lambda *args: (torch.empty(0), torch.empty(0)),
            max_top_k=2,
            max_rows=2,
        )
    assert raised.value is error
    gc.collect()
    assert all(reference() is None for reference in references)


def test_unique_storage_memory_is_deduplicated() -> None:
    base = torch.zeros(16, dtype=torch.int32)
    other = torch.zeros(3, dtype=torch.int64)
    tensors = {"base": base, "view": base[2:8], "other": other}
    assert loader._unique_storage_bytes(tensors) == (  # noqa: SLF001
        base.untyped_storage().nbytes() + other.untyped_storage().nbytes()
    )


def _readiness_backend(
    op: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    *,
    table: ConstraintTable,
    top_k: int,
) -> NPUConstraintBackend:
    tensors = {
        name: torch.from_numpy(np.array(getattr(table, name), copy=True))
        for name in CONSTRAINT_TABLE_TENSOR_NAMES
    }
    return NPUConstraintBackend(
        **tensors,
        info=ConstraintBackendInfo(
            table_metadata=table.metadata,
            backend=ConstraintTableBackendEnum.NPU,
            max_top_k=top_k,
            max_rows=2,
            host_memory_bytes=0,
            device_memory_bytes=table.total_bytes,
        ),
        op=op,
    )


def _fused_selection_op(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate the NPU wrapper's output contract for readiness.

    The fused operator's own candidate-set logprobs are discarded by the
    wrapper, which re-scores the selection over the shared full-window domain
    with ``score_selected_tokens`` — exactly what the unified Torch reference
    now emits — so readiness validates that output.
    """
    tokens, logprobs = rec_constrained_topk_torch(*args)
    return tokens.to(torch.int32), logprobs.contiguous()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_readiness_accepts_normalized_reference_at_top_k_256(
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = make_table(first_token_ids=np.asarray([1, 3, 4], dtype=np.int32))

    backend = _readiness_backend(_fused_selection_op, table=table, top_k=256)
    monkeypatch.setattr(loader, "_synchronize_current_npu_stream", lambda device: None)
    try:
        loader._run_npu_readiness(  # noqa: SLF001
            backend,
            cases=loader._readiness_cases(table, 256),  # noqa: SLF001
            device=torch.device("cpu"),
            dtype=dtype,
            max_rows=2,
        )
    finally:
        backend.close()


@pytest.mark.parametrize("failure_mode", ["raw", "ignore_temperature", "top_k_normalized"])
def test_readiness_rejects_unnormalized_logprobs(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = make_table(first_token_ids=np.asarray([1, 3, 4], dtype=np.int32))

    def broken_op(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
        if failure_mode == "ignore_temperature":
            modified = list(args)
            temperatures = modified[8]
            assert isinstance(temperatures, torch.Tensor)
            modified[8] = torch.ones_like(temperatures)
            tokens, logprobs = rec_constrained_topk_torch(*modified)
            return tokens.to(torch.int32), logprobs

        tokens, _ = rec_constrained_topk_torch(*args)
        logits = args[0]
        assert isinstance(logits, torch.Tensor)
        selected_logits = logits.gather(1, tokens.to(torch.long)).float()
        sequence_group = args[1]
        current_step = args[9]
        top_k = args[10]
        assert isinstance(sequence_group, torch.Tensor)
        assert isinstance(current_step, int)
        assert isinstance(top_k, int)
        counts = constraint_candidate_counts(
            sequence_group,
            cast(torch.Tensor, args[2]),
            cast(torch.Tensor, args[3]),
            cast(torch.Tensor, args[5]),
            cast(torch.Tensor, args[6]),
            current_step=current_step,
            vocab_size=table.metadata.vocab_size,
        )
        valid = valid_mask_from_counts(counts, top_k)
        if failure_mode == "raw":
            logprobs = torch.where(
                valid,
                selected_logits,
                torch.full_like(selected_logits, INVALID_LOGIT),
            )
        else:
            temperatures = args[8]
            assert isinstance(temperatures, torch.Tensor)
            temperatures = temperatures.reshape(-1, 1)
            temperatures = torch.where(
                temperatures == 0,
                torch.ones_like(temperatures),
                temperatures,
            )
            selected_scores = (selected_logits / temperatures).masked_fill(~valid, float("-inf"))
            logprobs = selected_scores - torch.logsumexp(selected_scores, dim=1, keepdim=True)
            logprobs = torch.where(
                valid,
                logprobs,
                torch.full_like(logprobs, INVALID_LOGIT),
            )
        return tokens.to(torch.int32), logprobs.contiguous()

    backend = _readiness_backend(broken_op, table=table, top_k=2)
    monkeypatch.setattr(loader, "_synchronize_current_npu_stream", lambda device: None)
    try:
        with pytest.raises(RuntimeError, match="normalized reference"):
            loader._run_npu_readiness(  # noqa: SLF001
                backend,
                cases=loader._readiness_cases(table, 2),  # noqa: SLF001
                device=torch.device("cpu"),
                dtype=torch.float32,
                max_rows=2,
            )
    finally:
        backend.close()


def test_npu_loader_injects_wrapper_not_raw_op(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NPU loads must wire the full wrapper, not the raw fused callable.

    The wrapper widens bf16 logits, preserves the operator's candidate-set
    normalized logprobs, and restores the deterministic oracle tie order.
    Injecting the raw op would skip all three.
    """
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    config = _config(
        path,
        digest,
        backend=ConstraintTableBackendEnum.NPU,
        max_top_k=2,
    )

    class _DummyFused:
        def rec_constrained_topk(self, *args: object) -> tuple[torch.Tensor, torch.Tensor]:
            del args
            raise AssertionError("resolver result is discarded; the wrapper is injected")

    monkeypatch.setattr(
        npu_op_module,
        "_custom_ops",
        lambda: cast("object", _DummyFused()),
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def _mark_ready(self) -> None: ...

    def fake_upload(
        table: object,
        *,
        device: torch.device,
        op: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        max_top_k: int,
        max_rows: int,
    ) -> tuple[object, tuple[object, ...]]:
        del table, device, max_top_k, max_rows
        captured["op"] = op
        return _StubBackend(), ()

    monkeypatch.setattr(loader, "_upload_npu_backend", fake_upload)
    monkeypatch.setattr(loader, "_run_npu_readiness", lambda *a, **k: None)
    backend = loader.load_constraint_backend(
        config,
        device=torch.device("npu"),
        vocab_size=6,
        dtype=torch.float32,
        max_rows=2,
    )
    assert isinstance(backend, _StubBackend)
    assert captured["op"] is loader.rec_constrained_topk_npu


@pytest.mark.parametrize("failed_step", [0, 1, 2])
def test_readiness_failure_closes_backend_and_preserves_exception(
    failed_step: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = make_table()
    tensors = {
        name: torch.from_numpy(np.array(getattr(table, name), copy=True))
        for name in CONSTRAINT_TABLE_TENSOR_NAMES
    }
    error = RuntimeError(f"step {failed_step} launch failed")

    def op(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
        if args[9] == failed_step:
            raise error
        return _fused_selection_op(*args)

    backend = NPUConstraintBackend(
        **tensors,
        info=ConstraintBackendInfo(
            table_metadata=table.metadata,
            backend=ConstraintTableBackendEnum.NPU,
            max_top_k=2,
            max_rows=2,
            host_memory_bytes=0,
            device_memory_bytes=table.total_bytes,
        ),
        op=op,
    )
    monkeypatch.setattr(loader, "_synchronize_current_npu_stream", lambda device: None)
    cases = loader._readiness_cases(table, 2)  # noqa: SLF001
    with pytest.raises(RuntimeError) as raised:
        try:
            loader._run_npu_readiness(  # noqa: SLF001
                backend,
                cases=cases,
                device=torch.device("cpu"),
                dtype=torch.float32,
                max_rows=2,
            )
        except BaseException:
            backend.close()
            raise
    assert raised.value is error
    assert backend._first_token_ids is None  # noqa: SLF001
