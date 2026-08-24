# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-only Backend construction from normalized ``VllmConfig`` data."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_gr.config import ConstraintTableBackend, ConstraintTableConfig
from vllm_gr.constraints.artifact import load_constraint_table_artifact
from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintBackend,
    ConstraintBackendInfo,
    ConstraintMetadata,
)
from vllm_gr.constraints.cpu_backend import CPUConstraintBackend
from vllm_gr.constraints.npu_backend import (
    NPU_CONSTRAINT_MAX_TOP_K,
    NPUConstraintBackend,
    resolve_npu_constraint_op,
)
from vllm_gr.constraints.table import TABLE_NAMES, ConstraintTable


def _device(value: str | torch.device) -> torch.device:
    try:
        return value if isinstance(value, torch.device) else torch.device(value)
    except RuntimeError as exc:
        raise RuntimeError(f"invalid constraint backend device {value!r}: {exc}") from exc


def _synchronize_npu(device: torch.device) -> None:
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError("torch.npu is unavailable")
    current_stream = getattr(npu, "current_stream", None)
    if callable(current_stream):
        current_stream(device).synchronize()
        return
    synchronize = getattr(npu, "synchronize", None)
    if not callable(synchronize):
        raise RuntimeError("torch.npu does not expose stream synchronization")
    synchronize(device)


def _npu_memory_allocated(device: torch.device) -> int:
    npu = getattr(torch, "npu", None)
    memory_allocated = getattr(npu, "memory_allocated", None)
    if not callable(memory_allocated):
        raise RuntimeError("torch.npu.memory_allocated is unavailable")
    return int(memory_allocated(device))


def _ready_cases(table: ConstraintTable) -> list[tuple[int, list[int], int, int]]:
    """Pick real step-0/1/2 paths and their expected highest-logit token."""

    if table.first_token_ids.size == 0:
        raise ValueError("constraint Artifact has no step-0 candidates")
    a = int(table.first_token_ids[0])
    begin1 = int(table.prefix1_offsets[a])
    end1 = int(table.prefix1_offsets[a + 1])
    if end1 <= begin1:
        raise ValueError("constraint Artifact has no step-1 candidate for its first token")
    b = int(table.prefix1_values[begin1])
    pair_key = a * table.info.vocab_size + b
    pair_index = int(np.searchsorted(table.prefix1_pair_keys, pair_key))
    if (
        pair_index >= len(table.prefix1_pair_keys)
        or int(table.prefix1_pair_keys[pair_index]) != pair_key
    ):
        raise ValueError("constraint Artifact is missing the selected step-2 pair key")
    begin2 = int(table.prefix2_value_offsets[pair_index])
    end2 = int(table.prefix2_value_offsets[pair_index + 1])
    if end2 <= begin2:
        raise ValueError("constraint Artifact has no step-2 candidate for its first pair")
    c = int(table.prefix2_values[begin2])
    return [
        (0, [0, 0, 0], a, len(table.first_token_ids)),
        (1, [a, 0, 0], b, end1 - begin1),
        (2, [a, b, 0], c, end2 - begin2),
    ]


def _validate_ready_output(
    output: Any,
    *,
    expected_token: int,
    valid_count: int,
    top_k: int,
    device: torch.device,
) -> None:
    expected_shape = (1, top_k)
    if tuple(output.token_ids.shape) != expected_shape:
        raise RuntimeError(f"token_ids shape {tuple(output.token_ids.shape)} != {expected_shape}")
    if tuple(output.logprobs.shape) != expected_shape:
        raise RuntimeError(f"logprobs shape {tuple(output.logprobs.shape)} != {expected_shape}")
    if tuple(output.valid_mask.shape) != expected_shape:
        raise RuntimeError(f"valid_mask shape {tuple(output.valid_mask.shape)} != {expected_shape}")
    if output.token_ids.dtype != torch.int32:
        raise RuntimeError(f"token_ids dtype must be int32, got {output.token_ids.dtype}")
    if output.logprobs.dtype != torch.float32:
        raise RuntimeError(f"logprobs dtype must be float32, got {output.logprobs.dtype}")
    if output.valid_mask.dtype != torch.bool:
        raise RuntimeError(f"valid_mask dtype must be bool, got {output.valid_mask.dtype}")
    for name, tensor in (
        ("token_ids", output.token_ids),
        ("logprobs", output.logprobs),
        ("valid_mask", output.valid_mask),
    ):
        if tensor.device != device:
            raise RuntimeError(f"{name} device {tensor.device} != {device}")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    derived_mask = output.logprobs > INVALID_LOGIT / 2
    if not torch.equal(output.valid_mask, derived_mask):
        raise RuntimeError("valid_mask does not match the INVALID_LOGIT threshold")
    if int(output.token_ids[0, 0].item()) != expected_token:
        raise RuntimeError(
            f"expected highest-logit token {expected_token}, got "
            f"{int(output.token_ids[0, 0].item())}"
        )
    expected_valid = min(valid_count, top_k)
    if int(output.valid_mask.sum().item()) != expected_valid:
        raise RuntimeError(
            f"expected {expected_valid} valid outputs, got {int(output.valid_mask.sum().item())}"
        )
    if expected_valid < top_k:
        invalid = output.logprobs[~output.valid_mask]
        if invalid.numel() == 0 or not bool((invalid == INVALID_LOGIT).all()):
            raise RuntimeError("padding slots must carry the INVALID_LOGIT sentinel")


def run_npu_ready_check(
    backend: NPUConstraintBackend,
    *,
    cases: list[tuple[int, list[int], int, int]],
    device: torch.device,
    dtype: torch.dtype,
    top_k: int,
) -> None:
    """Execute the real NPU op for steps 0, 1, and 2 in strict order."""

    for step, prefix, expected_token, valid_count in cases:
        try:
            logits = torch.full(
                (1, backend.info.table.vocab_size),
                -10.0,
                dtype=dtype,
                device=device,
            )
            logits[0, expected_token] = 10.0
            metadata = ConstraintMetadata(
                prefix_token_ids=torch.tensor(
                    [prefix],
                    dtype=torch.int32,
                    device=device,
                ),
                temperatures=torch.ones(1, dtype=torch.float32, device=device),
                current_step=step,
                top_k=top_k,
            )
            output = backend.select(logits, metadata)
            _synchronize_npu(device)
            _validate_ready_output(
                output,
                expected_token=expected_token,
                valid_count=valid_count,
                top_k=top_k,
                device=device,
            )
        except Exception as exc:
            backend.close()
            raise RuntimeError(
                "NPU constraint readiness failed: "
                f"step={step}, dtype={dtype}, top_k={top_k}, device={device}, "
                f"artifact_digest={backend.info.table.artifact_digest}: {exc}"
            ) from exc


def _create_npu_constraint_backend(
    table: ConstraintTable,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_top_k: int,
) -> NPUConstraintBackend:
    op = resolve_npu_constraint_op()
    cases = _ready_cases(table)
    before = _npu_memory_allocated(device)
    device_tables: dict[str, torch.Tensor] = {}
    backend: NPUConstraintBackend | None = None
    try:
        for name in TABLE_NAMES:
            host_copy = np.array(getattr(table, name), copy=True)
            device_tables[name] = torch.from_numpy(host_copy).to(device=device)
        _synchronize_npu(device)
        allocator_delta = max(0, _npu_memory_allocated(device) - before)
        info = ConstraintBackendInfo(
            table=table.info,
            max_top_k=max_top_k,
            device_memory_bytes=allocator_delta,
        )
        backend = NPUConstraintBackend(
            **device_tables,
            info=info,
            op=op,
        )
        run_npu_ready_check(
            backend,
            cases=cases,
            device=device,
            dtype=dtype,
            top_k=max_top_k,
        )
        return backend
    except Exception:
        if backend is not None:
            backend.close()
        device_tables.clear()
        raise


def load_constraint_backend(
    *,
    config: ConstraintTableConfig,
    device: str | torch.device,
    vocab_size: int,
    dtype: torch.dtype,
) -> ConstraintBackend | None:
    """Construct one independent Backend for one Worker ModelRunner."""

    if not config.enabled:
        return None
    if config.path is None or config.artifact_digest is None or config.tokenizer_digest is None:
        raise ValueError("enabled constraint table configuration is incomplete")

    target_device = _device(device)
    if config.backend is ConstraintTableBackend.NPU:
        if target_device.type != "npu":
            raise RuntimeError(
                f"NPU constraint backend requires an NPU model device, got {target_device}"
            )
        if config.max_top_k > NPU_CONSTRAINT_MAX_TOP_K:
            raise ValueError(
                "NPU constraint backend supports max_top_k <= "
                f"{NPU_CONSTRAINT_MAX_TOP_K}, got {config.max_top_k}"
            )
    elif config.backend is not ConstraintTableBackend.CPU:
        raise RuntimeError(f"unsupported constraint table backend {config.backend!r}")

    table = load_constraint_table_artifact(
        config.path,
        expected_digest=config.artifact_digest,
        expected_tokenizer_digest=config.tokenizer_digest,
        expected_vocab_size=vocab_size,
        expected_format=config.format,
    )
    if config.backend is ConstraintTableBackend.CPU:
        info = ConstraintBackendInfo(
            table=table.info,
            max_top_k=config.max_top_k,
            device_memory_bytes=0,
        )
        return CPUConstraintBackend(table, info=info)
    backend = _create_npu_constraint_backend(
        table,
        device=target_device,
        dtype=dtype,
        max_top_k=config.max_top_k,
    )
    # NPU creation completed upload and readiness without retaining ``table``;
    # release the Worker's last Host-table owner now.
    del table
    return backend
