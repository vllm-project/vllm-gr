# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Update ordinary CUSTOM attention inside the native Ascend FULL graph.

Beam decode and the private prefill runner own different graphs and never use
this registry. The native runner enqueues replay before its update hook: each
captured FIA task therefore waits on an ExternalEvent recorded by that hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NoReturn

import torch
import torch_npu
from vllm.forward_context import get_forward_context
from vllm.v1.cudagraph_dispatcher import CUDAGraphMode
from vllm_ascend.compilation.acl_graph import (
    ACLGraphEntry,
    ACLGraphWrapper,
    get_graph_params,
)
from vllm_ascend.utils import weak_ref_tensors

_REGISTRY_ATTRIBUTE = "_vllm_gr_standard_fia_entries"
_WORKSPACE_POOL_ATTRIBUTE = "_vllm_gr_standard_fia_workspace_pool"


@dataclass
class _StandardFIATask:
    layer_name: str
    handle: Any
    event: Any
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    mask: torch.Tensor | None
    output: torch.Tensor
    lse: torch.Tensor
    slot_mapping: torch.Tensor
    workspace: torch.Tensor | None
    block_size: int
    num_heads: int
    num_kv_heads: int
    scale: float
    sparse_mode: int


@dataclass
class _StandardGraphEntry:
    expected_layers: tuple[str, ...]
    max_model_len: int
    tasks: dict[str, _StandardFIATask] = field(default_factory=dict)
    # Cache each exact geometry's selected binding through the complete capture.
    # The owner pool separately retains allocations when entry bindings weaken.
    workspaces: dict[tuple[Any, ...], torch.Tensor] = field(default_factory=dict)
    workspaces_weakened: bool = False


def _fail(message: str) -> NoReturn:
    raise RuntimeError(f"Ordinary NPU FULL attention: {message}")


def _binding(tensor: torch.Tensor | None) -> tuple[Any, ...] | None:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        _fail("a captured tensor is missing")
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.stride(),
        tensor.dtype,
        tensor.device,
        tensor.storage_offset(),
    )


def _metadata_lists(metadata: Any, num_tokens: int) -> tuple[list[int], list[int]]:
    count = int(metadata.num_reqs)
    starts = [int(value) for value in metadata.query_start_loc_cpu[: count + 1]]
    lengths = [int(value) for value in metadata.seq_lens_cpu[:count]]
    if (
        count <= 0
        or len(starts) != count + 1
        or len(lengths) != count
        or starts[0] != 0
        or starts[-1] != num_tokens
        or any(right <= left for left, right in zip(starts, starts[1:]))
        or any(value < 0 for value in lengths)
    ):
        _fail("invalid current query/KV length lists")
    table = metadata.block_table
    if not isinstance(table, torch.Tensor) or table.ndim != 2 or table.shape[0] < count:
        _fail("block table does not cover the current requests")
    return starts[1:], lengths


def _context_entry_context(
    forward_context: Any, num_tokens: int
) -> tuple[Any, Any, dict[str, Any]]:
    descriptor = getattr(forward_context, "batch_descriptor", None)
    if (
        getattr(forward_context, "cudagraph_runtime_mode", None) != CUDAGraphMode.FULL
        or descriptor is None
        or descriptor.num_tokens != num_tokens
    ):
        _fail("FULL mode and an exact matching batch descriptor are required")
    metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(metadata, dict) or not metadata:
        _fail("per-layer attention metadata is missing")
    owner = get_graph_params()
    if owner is None:
        _fail("native graph parameter owner is missing")
    return owner, descriptor, metadata


def _kwargs(
    task: _StandardFIATask,
    block_table: torch.Tensor,
    qlen: list[int],
    kvlen: list[int],
) -> dict[str, Any]:
    return {
        "query": task.query,
        "key": task.key,
        "value": task.value,
        "atten_mask": task.mask,
        "block_table": block_table,
        "input_layout": "TND",
        "block_size": task.block_size,
        "actual_seq_qlen": qlen,
        "actual_seq_kvlen": kvlen,
        "num_key_value_heads": task.num_kv_heads,
        "num_query_heads": task.num_heads,
        "softmax_scale": task.scale,
        "sparse_mode": task.sparse_mode,
    }


def _enpu_first_capture_pending(forward_context: Any, descriptor: Any) -> bool:
    """Recognize only the native ENPU hook that precedes a first capture.

    ENPU calls the update hook before the model wrapper, including before its
    first capture. Ordinary replay still requires a complete task registry:
    an existing native graph must never hide a missing ordinary task entry.
    """
    if getattr(forward_context, "capturing", None) is not False:
        return False
    model = getattr(forward_context, "model_instance", None)
    wrapper = getattr(model, "cudagraph_wrapper", None)
    if (
        not isinstance(wrapper, ACLGraphWrapper)
        or wrapper.runtime_mode != CUDAGraphMode.FULL
        or wrapper.enable_enpu is not True
    ):
        return False
    native_entries = wrapper.concrete_aclgraph_entries
    if not isinstance(native_entries, dict):
        return False
    if descriptor not in native_entries:
        return True
    native_entry = native_entries[descriptor]
    return (
        isinstance(native_entry, ACLGraphEntry)
        and native_entry.batch_descriptor == descriptor
        and native_entry.aclgraph is None
    )


def _tensor_layout(tensor: torch.Tensor | None, *, varying_rows: bool = False) -> Any:
    if tensor is None:
        return None
    return (
        tuple(tensor.shape[1:] if varying_rows else tensor.shape),
        tensor.stride(),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tensor.storage_offset(),
    )


def _workspace_bytes(workspace: torch.Tensor) -> int:
    if (
        not isinstance(workspace, torch.Tensor)
        or workspace.layout != torch.strided
        or not workspace.is_contiguous()
        or workspace.storage_offset() != 0
    ):
        _fail("workspace must be a contiguous tensor with zero storage offset")
    return int(workspace.numel()) * int(workspace.element_size())


def _select_workspace(
    owner: Any,
    context: Any,
    task: _StandardFIATask,
    workspace_kwargs: dict[str, Any],
    max_model_len: int,
) -> torch.Tensor:
    """Size every new geometry, then reuse a sufficient compatible allocation.

    The pool owns every selected allocation until the native graph owner is
    reset. A later larger requirement adds a slot without changing pointers
    already captured by older graphs. Capacity is never inferred from T.
    Sharing stays within the same native wrapper and its serialized forwards.
    """
    model = getattr(context, "model_instance", None)
    wrapper = getattr(model, "cudagraph_wrapper", None)
    if not isinstance(wrapper, ACLGraphWrapper) or wrapper.runtime_mode != CUDAGraphMode.FULL:
        _fail("workspace sharing requires the actual native FULL graph wrapper")
    candidate = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(**workspace_kwargs)
    required_bytes = _workspace_bytes(candidate)
    if candidate.device != task.query.device:
        _fail("workspace is on a different device from its FIA query")
    compatibility = (
        wrapper,
        "FIA-v2/TND",
        max_model_len,
        _tensor_layout(task.query, varying_rows=True),
        _tensor_layout(task.output, varying_rows=True),
        _tensor_layout(task.key),
        _tensor_layout(task.value),
        _tensor_layout(task.mask),
        _tensor_layout(workspace_kwargs["block_table"], varying_rows=True),
        task.block_size,
        task.num_heads,
        task.num_kv_heads,
        task.scale,
        task.sparse_mode,
        candidate.dtype,
        candidate.device,
        candidate.layout,
        candidate.ndim,
    )
    pool = getattr(owner, _WORKSPACE_POOL_ATTRIBUTE, None)
    if pool is None:
        pool = {}
        setattr(owner, _WORKSPACE_POOL_ATTRIBUTE, pool)
    if not isinstance(pool, dict):
        _fail("invalid ordinary workspace pool")
    slots = pool.setdefault(compatibility, [])
    if not isinstance(slots, list):
        _fail("invalid ordinary workspace slots")
    sufficient = []
    for workspace in slots:
        capacity = _workspace_bytes(workspace)
        if (
            workspace.dtype != candidate.dtype
            or workspace.device != candidate.device
            or workspace.ndim != candidate.ndim
        ):
            _fail("incompatible tensor in ordinary workspace pool")
        if capacity >= required_bytes:
            sufficient.append((capacity, workspace))
    if sufficient:
        # Returning releases the unused candidate before capturing the FIA op.
        return min(sufficient, key=lambda item: item[0])[1]
    slots.append(candidate)
    return candidate


def capture_standard_fia(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: Any,
    scale: float,
    layer_name: str,
    max_model_len: int,
) -> None:
    """Capture one ordinary FIA-v2 task using the native event protocol."""
    context = get_forward_context()
    num_tokens = int(query.shape[0])
    owner, descriptor, metadatas = _context_entry_context(context, num_tokens)
    if not getattr(context, "capturing", False):
        _fail("task capture requires an active native graph capture")
    if layer_name not in metadatas or metadatas[layer_name] is not metadata:
        _fail("capture metadata does not belong to its attention layer")
    qlen, kvlen = _metadata_lists(metadata, num_tokens)
    if max_model_len <= 0 or max(kvlen) > max_model_len:
        _fail("invalid maximum model context")
    registry = getattr(owner, _REGISTRY_ATTRIBUTE, None)
    if registry is None:
        registry = {}
        setattr(owner, _REGISTRY_ATTRIBUTE, registry)
    if not isinstance(registry, dict):
        _fail("invalid ordinary graph registry")
    entry = registry.get(descriptor)
    if entry is None:
        entry = _StandardGraphEntry(tuple(metadatas), max_model_len)
        registry[descriptor] = entry
    if (
        not isinstance(entry, _StandardGraphEntry)
        or entry.expected_layers != tuple(metadatas)
        or entry.max_model_len != max_model_len
        or entry.workspaces_weakened
        or layer_name in entry.tasks
    ):
        _fail("duplicate or incompatible graph capture")

    key = key_cache.view(*key_cache.shape[:-2], -1)
    value = value_cache.view(*value_cache.shape[:-2], -1)
    mask = metadata.npu_causal_atten_mask if metadata.causal else None
    slot_mapping = metadata.slot_mapping
    if _binding(slot_mapping) is None:
        _fail("capture slot mapping is missing")
    task = _StandardFIATask(
        layer_name=layer_name,
        handle=None,
        event=None,
        query=query,
        key=key,
        value=value,
        mask=mask,
        output=output,
        lse=torch.empty(1, dtype=output.dtype, device=output.device),
        slot_mapping=slot_mapping,
        workspace=None,
        block_size=int(metadata.block_size),
        num_heads=int(metadata.num_heads),
        num_kv_heads=int(metadata.num_kv_heads),
        scale=float(scale),
        sparse_mode=3 if metadata.causal else 0,
    )
    kwargs = _kwargs(task, metadata.block_table, qlen, kvlen)
    workspace_key = (
        _tensor_layout(query),
        _tensor_layout(output),
        _tensor_layout(key),
        _tensor_layout(value),
        _binding(mask),
        _tensor_layout(metadata.block_table),
        task.block_size,
        task.num_heads,
        task.num_kv_heads,
        task.scale,
        task.sparse_mode,
        tuple(qlen),
    )
    workspace = entry.workspaces.get(workspace_key)
    if workspace is None:
        # Use every captured request, including the ordinary 64-row bucket.
        # The private prefill runner's eight-request policy does not apply.
        workspace_kwargs = dict(kwargs, actual_seq_kvlen=[max_model_len] * len(qlen))
        workspace = _select_workspace(owner, context, task, workspace_kwargs, max_model_len)
        entry.workspaces[workspace_key] = workspace
    task.workspace = weak_ref_tensors(workspace)

    stream = torch.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
    torch.npu.graph_task_group_begin(stream)
    torch_npu.npu_fused_infer_attention_score_v2.out(
        **kwargs, workspace=workspace, out=[output, task.lse]
    )
    task.handle = torch.npu.graph_task_group_end(stream)
    task.event = event
    for name in ("query", "key", "value", "mask", "output", "lse", "slot_mapping"):
        tensor = getattr(task, name)
        if tensor is not None:
            setattr(task, name, weak_ref_tensors(tensor))
    entry.tasks[layer_name] = task


def update_standard_graph_params(
    update_stream: Any,
    forward_context: Any,
    num_tokens: int,
) -> None:
    """Refresh all ordinary tasks, then release their captured event waits.

    Every task is validated before the first task update. Errors propagate;
    replay must never proceed with absent or partially accepted metadata.
    """
    owner, descriptor, metadatas = _context_entry_context(forward_context, num_tokens)
    registry = getattr(owner, _REGISTRY_ATTRIBUTE, None)
    if (
        registry is None or isinstance(registry, dict) and descriptor not in registry
    ) and _enpu_first_capture_pending(forward_context, descriptor):
        return
    entry = registry.get(descriptor) if isinstance(registry, dict) else None
    if (
        not isinstance(entry, _StandardGraphEntry)
        or set(entry.expected_layers) != set(metadatas)
        or set(entry.tasks) != set(entry.expected_layers)
    ):
        _fail("ordinary graph capture is absent or incomplete")
    prepared = []
    for name in entry.expected_layers:
        task = entry.tasks[name]
        if not isinstance(task, _StandardFIATask) or task.layer_name != name:
            _fail("invalid captured attention task")
        metadata = metadatas[name]
        qlen, kvlen = _metadata_lists(metadata, num_tokens)
        if (
            task.handle is None
            or task.event is None
            or max(kvlen) > entry.max_model_len
            or int(metadata.block_size) != task.block_size
            or int(metadata.num_heads) != task.num_heads
            or int(metadata.num_kv_heads) != task.num_kv_heads
            or (3 if metadata.causal else 0) != task.sparse_mode
            or _binding(metadata.npu_causal_atten_mask if metadata.causal else None)
            != _binding(task.mask)
            or _binding(metadata.slot_mapping) != _binding(task.slot_mapping)
        ):
            _fail(f"incompatible current metadata for {name}")
        for tensor in (
            task.query,
            task.key,
            task.value,
            task.output,
            task.lse,
            task.workspace,
        ):
            if _binding(tensor) is None:
                _fail(f"missing captured FIA tensor for {name}")
        prepared.append((task, _kwargs(task, metadata.block_table, qlen, kvlen)))
    if not entry.workspaces_weakened:
        entry.workspaces = {key: weak_ref_tensors(value) for key, value in entry.workspaces.items()}
        entry.workspaces_weakened = True
    with torch.npu.stream(update_stream):
        for task, kwargs in prepared:
            torch.npu.graph_task_update_begin(update_stream, task.handle)
            torch_npu.npu_fused_infer_attention_score_v2.out(
                **kwargs, workspace=task.workspace, out=[task.output, task.lse]
            )
            torch.npu.graph_task_update_end(update_stream)
            task.event.record(update_stream)
