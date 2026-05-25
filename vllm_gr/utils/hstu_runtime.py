# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared runtime helpers for HSTU execution."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from vllm_gr.utils.npu_adapter import is_cuda_available, is_npu_available


def _is_embedding_tensor(name: str) -> bool:
    return "embedding_collection" in name


@dataclass(frozen=True)
class HSTURuntimeDevices:
    """Resolved runtime device policy for the two-phase HSTU execution path."""

    dense_device: torch.device
    embedding_device: torch.device


@dataclass(frozen=True)
class HSTUSparseTransferPlan:
    """Resolved transfer contract from phase-1 sparse outputs to dense phase."""

    source_device: torch.device
    target_device: torch.device
    requires_transfer: bool


def resolve_hstu_runtime_devices(
    module: Optional[torch.nn.Module] = None,
    *,
    explicit_dense_device: Optional[torch.device] = None,
    explicit_embedding_device: Optional[torch.device] = None,
) -> HSTURuntimeDevices:
    """Resolve the single runtime policy for dense and embedding phases.

    Current policy:
    - dense phase follows an explicit override, otherwise the active
      accelerator when available
    - embedding phase defaults to CPU unless explicitly overridden

    The embedding decision intentionally does not infer from model weights.
    It must come from an explicit runtime choice or the conservative default.
    """
    if explicit_dense_device is not None:
        dense_device = torch.device(explicit_dense_device)
    else:
        dense_device = _infer_hstu_dense_device(module)

    if explicit_embedding_device is not None:
        embedding_device = torch.device(explicit_embedding_device)
    else:
        embedding_device = torch.device("cpu")

    return HSTURuntimeDevices(
        dense_device=dense_device,
        embedding_device=embedding_device,
    )


def resolve_hstu_sparse_transfer_plan(
    *,
    source_device: torch.device,
    target_device: torch.device,
) -> HSTUSparseTransferPlan:
    """Resolve whether sparse outputs need an explicit device transfer."""
    normalized_source = torch.device(source_device)
    normalized_target = torch.device(target_device)
    return HSTUSparseTransferPlan(
        source_device=normalized_source,
        target_device=normalized_target,
        requires_transfer=normalized_source != normalized_target,
    )


def _infer_hstu_dense_device(
    module: Optional[torch.nn.Module],
) -> torch.device:
    if module is None:
        return _get_active_hstu_dense_device()

    for name, param in module.named_parameters():
        if not _is_embedding_tensor(name):
            return param.device

    for name, buffer in module.named_buffers():
        if not _is_embedding_tensor(name):
            return buffer.device

    try:
        return next(module.parameters()).device
    except StopIteration:
        return _get_active_hstu_dense_device()


def _get_active_hstu_dense_device() -> torch.device:
    """Resolve the active accelerator for dense HSTU execution."""
    if is_npu_available():
        try:
            return torch.device("npu", torch.npu.current_device())
        except (AttributeError, RuntimeError):
            return torch.device("npu")

    if is_cuda_available():
        try:
            return torch.device("cuda", torch.cuda.current_device())
        except (AttributeError, RuntimeError):
            return torch.device("cuda")

    return torch.device("cpu")


def resolve_hstu_dense_device(
    module: torch.nn.Module,
    explicit_device: Optional[torch.device] = None,
) -> torch.device:
    """Resolve the accelerator device used by the dense HSTU path."""
    return resolve_hstu_runtime_devices(
        module,
        explicit_dense_device=explicit_device,
    ).dense_device


def resolve_hstu_embedding_device(
    explicit_device: Optional[torch.device] = None,
) -> torch.device:
    """Resolve the embedding device for the conservative phase-1 runtime."""
    return resolve_hstu_runtime_devices(
        explicit_embedding_device=explicit_device,
    ).embedding_device


def move_hstu_sparse_outputs_to_device(
    seq_embeddings: Dict[str, Any],
    payload_features: Dict[str, torch.Tensor],
    uih_seq_lengths: torch.Tensor,
    num_candidates: torch.Tensor,
    device: torch.device,
    *,
    source_device: Optional[torch.device] = None,
    target_dtype: Optional[torch.dtype] = None,
) -> Tuple[
    Dict[str, Any],
    Dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Move sparse preprocess outputs onto the dense execution device.

    The transfer contract is explicit:
    - sparse outputs are re-homed onto the dense execution device
    - when ``target_dtype`` is provided, sequence embeddings are cast at the
      phase boundary to match dense forward expectations
    - payload features preserve their original dtype semantics; only device is
      normalized for them
    """
    from vllm_gr.models.dlrm_v3.modules.dlrm_hstu import SequenceEmbedding

    resolved_source = torch.device(source_device or device)
    resolved_target = torch.device(device)
    transfer_plan = resolve_hstu_sparse_transfer_plan(
        source_device=resolved_source,
        target_device=resolved_target,
    )

    if not transfer_plan.requires_transfer and target_dtype is None:
        return (
            seq_embeddings,
            payload_features,
            uih_seq_lengths,
            num_candidates,
        )

    moved_seq_embeddings = {
        key: SequenceEmbedding(
            lengths=value.lengths.to(transfer_plan.target_device),
            embedding=value.embedding.to(
                device=transfer_plan.target_device,
                dtype=target_dtype or value.embedding.dtype,
            ),
        )
        for key, value in seq_embeddings.items()
    }
    moved_payload_features = {
        key: value.to(transfer_plan.target_device)
        for key, value in payload_features.items()
    }

    return (
        moved_seq_embeddings,
        moved_payload_features,
        uih_seq_lengths.to(transfer_plan.target_device),
        num_candidates.to(transfer_plan.target_device),
    )


def move_hstu_sparse_inputs_to_device(
    *,
    uih_features: Any,
    candidates_features: Any,
    device: torch.device,
) -> tuple[Any, Any]:
    """Move KJT sparse inputs onto the embedding runtime device.

    vLLM may materialize model kwargs on the active accelerator. HSTU's sparse
    phase keeps embedding tables on ``embedding_device`` by contract, so we
    explicitly rebuild the KJTs on that device before calling preprocess().
    """
    return (
        _move_keyed_jagged_tensor_to_device(uih_features, device),
        _move_keyed_jagged_tensor_to_device(candidates_features, device),
    )


def _move_keyed_jagged_tensor_to_device(features: Any, device: torch.device) -> Any:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    target_device = torch.device(device)
    values = features.values()
    lengths = features.lengths()
    weights = (
        features.weights_or_none()
        if hasattr(features, "weights_or_none")
        else None
    )

    current_device = values.device
    if current_device == target_device and lengths.device == target_device:
        if weights is None or weights.device == target_device:
            return features

    kwargs: Dict[str, Any] = {
        "keys": list(features.keys()),
        "values": values.to(target_device),
        "lengths": lengths.to(target_device),
    }
    if weights is not None:
        kwargs["weights"] = weights.to(target_device)

    return KeyedJaggedTensor.from_lengths_sync(**kwargs)


def get_hstu_autocast_context(
    tensor: torch.Tensor,
    *,
    enabled: bool,
    dtype: torch.dtype = torch.bfloat16,
):
    """Return a conservative autocast context for the tensor's device."""
    if not enabled or tensor.device.type == "cpu":
        return nullcontext()

    try:
        return torch.autocast(
            device_type=tensor.device.type,
            dtype=dtype,
            enabled=True,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return nullcontext()
