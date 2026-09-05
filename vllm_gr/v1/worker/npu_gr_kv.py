# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-step, fail-closed NPU KV reorder for persistent GR ModelRunner V1.

The object freezes one session epoch, pool layout, scratch layout, xllm slot
table and execution mode at construction.  It does not own Beam history and
does not modify canonical decision or GRModelRunnerState cursors.

An operator call only enqueues device work; returning from :meth:`reorder`
does not claim device completion.  Because either fused or cascade reorder
may have partially modified KV before raising, a failed attempt permanently
poisons this object.  References are retained for the future backend's drain
and release path, and retry is forbidden.

The cascade reference uses fixed aux storage and avoids its explicit
``clone`` fallback, but its advanced-indexing RHS can still allocate a
temporary.  Zero-allocation/capture suitability requires NPU hardware work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm_gr.v1.worker.npu_gr_decision import NPUDecisionStep
from vllm_gr.v1.worker.npu_gr_masked_decision import NPUMaskedDecisionStep

_DecisionStep = NPUDecisionStep | NPUMaskedDecisionStep


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"persistent NPU KV {name} is invalid")
    return value


def _tensor(value: Any, name: str, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise RuntimeError(f"persistent NPU KV {name} must be a Tensor")
    if value.device != device or not value.is_contiguous():
        raise RuntimeError(f"persistent NPU KV {name} device/layout mismatch")
    return value


def _storage_id(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


@dataclass(frozen=True, slots=True)
class _TensorBinding:
    name: str
    tensor: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    data_ptr: int
    storage_ptr: int

    @classmethod
    def capture(cls, name: str, tensor: torch.Tensor) -> "_TensorBinding":
        return cls(
            name,
            tensor,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
            int(tensor.data_ptr()),
            _storage_id(tensor),
        )

    def validate(self, current: Any) -> None:
        if (
            current is not self.tensor
            or not torch.is_tensor(current)
            or tuple(current.shape) != self.shape
            or tuple(current.stride()) != self.stride
            or current.dtype is not self.dtype
            or current.device != self.device
            or int(current.data_ptr()) != self.data_ptr
            or _storage_id(current) != self.storage_ptr
        ):
            raise RuntimeError(f"persistent NPU KV fixed {self.name} binding changed")


class NPUKVReorder:
    """One fixed-layout reorder owner for a single active session epoch."""

    def __init__(
        self,
        session: Any,
        *,
        mode: Literal["xllm", "cascade"],
        xllm_ops: Any | None = None,
        cascade_op: Any | None = None,
    ) -> None:
        if mode not in ("xllm", "cascade"):
            raise ValueError("persistent NPU KV mode must be 'xllm' or 'cascade'")
        if getattr(session, "is_active", None) is not True:
            raise RuntimeError("persistent NPU KV requires an active session")
        if _int(getattr(session, "slot", None), "session slot") != 0:
            raise RuntimeError("persistent NPU KV currently requires isolated slot zero")
        session_id = getattr(session, "session_id", None)
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("persistent NPU KV session identity is invalid")
        state = getattr(session, "_state", None)
        context = getattr(session, "context", None)
        if state is None or context is None:
            raise RuntimeError("persistent NPU KV session epoch/context is missing")
        width = _int(getattr(session, "beam_width", None), "Beam width", minimum=1)
        pools = getattr(context, "pools", None)
        if not isinstance(pools, (list, tuple)) or not pools:
            raise RuntimeError("persistent NPU KV pools are missing")
        first_key = getattr(pools[0], "unshared_key", None)
        if not torch.is_tensor(first_key):
            raise RuntimeError("persistent NPU KV pool key is missing")
        device = first_key.device
        injected = xllm_ops is not None if mode == "xllm" else cascade_op is not None
        if not injected and device.type != "npu":
            raise RuntimeError("persistent NPU KV production operators require NPU tensors")

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        pool_storages: set[int] = set()
        reference_shape: tuple[int, ...] | None = None
        reference_dtype: torch.dtype | None = None
        for layer, pool in enumerate(pools):
            key = _tensor(getattr(pool, "unshared_key", None), f"pool key {layer}", device)
            value = _tensor(getattr(pool, "unshared_value", None), f"pool value {layer}", device)
            if key.ndim != 5 or value.shape != key.shape or key.dtype != value.dtype:
                raise RuntimeError("persistent NPU KV pool key/value geometry mismatch")
            if key.dtype not in (torch.bfloat16, torch.float16):
                raise RuntimeError("persistent NPU KV pool dtype is unsupported")
            if any(dimension < 1 for dimension in key.shape):
                raise RuntimeError("persistent NPU KV pool dimensions must be positive")
            if key.shape[1] != width:
                raise RuntimeError("persistent NPU KV pool capacity/Beam width mismatch")
            if reference_shape is None:
                reference_shape = tuple(key.shape)
                reference_dtype = key.dtype
            elif tuple(key.shape) != reference_shape or key.dtype != reference_dtype:
                raise RuntimeError("persistent NPU KV layer layouts must be identical")
            key_storage = _storage_id(key)
            value_storage = _storage_id(value)
            if (
                key_storage == value_storage
                or key_storage in pool_storages
                or value_storage in pool_storages
            ):
                raise RuntimeError("persistent NPU KV pool tensors alias")
            pool_storages.update((key_storage, value_storage))
            keys.append(key)
            values.append(value)

        aux_key = _tensor(getattr(context, "aux_key_buf", None), "aux key", device)
        aux_value = _tensor(getattr(context, "aux_value_buf", None), "aux value", device)
        expected_aux_shape = (len(keys), *reference_shape[1:])  # type: ignore[index]
        if (
            tuple(aux_key.shape) != expected_aux_shape
            or tuple(aux_value.shape) != expected_aux_shape
            or aux_key.dtype != reference_dtype
            or aux_value.dtype != reference_dtype
        ):
            raise RuntimeError("persistent NPU KV fixed aux geometry mismatch")
        aux_storages = {_storage_id(aux_key), _storage_id(aux_value)}
        if len(aux_storages) != 2 or aux_storages & pool_storages:
            raise RuntimeError("persistent NPU KV aux storage aliases KV storage")
        block_table = _tensor(
            getattr(context, "xllm_block_table_buf", None), "xllm block table", device
        )
        if block_table.ndim != 1 or block_table.numel() < 1 or block_table.dtype is not torch.int32:
            raise RuntimeError("persistent NPU KV xllm block table geometry mismatch")

        if mode == "xllm":
            if cascade_op is not None:
                raise ValueError("cascade_op cannot be supplied in xllm mode")
            if xllm_ops is None:
                from vllm_gr.ops.npu import beam_ops
                from vllm_gr.ops.npu._xllm_loader import has_op

                if not has_op("select_unshared_kv"):
                    raise RuntimeError("persistent NPU KV select_unshared_kv is unavailable")
                xllm_ops = beam_ops
            select = getattr(xllm_ops, "select_unshared_kv", None)
            if not callable(select):
                raise RuntimeError("persistent NPU KV select_unshared_kv is unavailable")
            self._xllm_select = select
            self._cascade_select = None
        else:
            if xllm_ops is not None:
                raise ValueError("xllm_ops cannot be supplied in cascade mode")
            if cascade_op is None:
                from vllm_gr.v1.attention.backends.beam_kernels import (
                    select_unshared_kv,
                )

                cascade_op = select_unshared_kv
            if not callable(cascade_op):
                raise RuntimeError("persistent NPU KV cascade select operator is unavailable")
            self._xllm_select = None
            self._cascade_select = cascade_op

        canonical = getattr(context, "decision", None)
        if canonical is None:
            raise RuntimeError("persistent NPU KV canonical decision buffers are missing")
        canonical_sequence = _tensor(
            getattr(canonical, "sequence", None), "canonical sequence", device
        )
        canonical_scores = _tensor(
            getattr(canonical, "beam_scores", None), "canonical scores", device
        )
        canonical_prefix = _tensor(
            getattr(canonical, "group_token_num", None), "canonical prefix", device
        )
        if (
            canonical_sequence.ndim != 3
            or canonical_sequence.shape[0] != 1
            or canonical_sequence.shape[1] != width
            or canonical_sequence.dtype is not torch.int32
            or canonical_scores.shape != (1, width)
            or canonical_scores.dtype is not torch.float32
            or canonical_prefix.shape != (1, width)
            or canonical_prefix.dtype is not torch.int32
            or reference_shape is None
            or reference_shape[3] < canonical_sequence.shape[2]
        ):
            raise RuntimeError("persistent NPU KV canonical/pool capacity mismatch")
        self._mode = mode
        self._session = session
        self._session_id = session_id
        self._session_epoch = state
        self._context = context
        self._canonical = canonical
        self._device = device
        self._slot = 0
        self._width = width
        self._canonical_capacity = canonical_sequence.shape[2]
        self._pools = tuple(pools)
        self._keys = keys
        self._values = values
        self._aux_key = aux_key
        self._aux_value = aux_value
        self._aux_keys = [aux_key[layer] for layer in range(len(keys))]
        self._aux_values = [aux_value[layer] for layer in range(len(values))]
        self._block_table = block_table[:1]
        self._block_table_source = block_table
        bound_tensors = (
            ("canonical sequence", canonical_sequence),
            ("canonical scores", canonical_scores),
            ("canonical prefix", canonical_prefix),
            *((f"pool key {layer}", tensor) for layer, tensor in enumerate(keys)),
            *((f"pool value {layer}", tensor) for layer, tensor in enumerate(values)),
            ("aux key", aux_key),
            ("aux value", aux_value),
            ("xllm block table", block_table),
        )
        self._tensor_bindings = tuple(
            _TensorBinding.capture(name, tensor) for name, tensor in bound_tensors
        )
        self._next_expected_step = (
            _int(getattr(session, "decode_step", None), "initial decode step") + 1
        )
        self._last_decision: _DecisionStep | None = None
        self._terminal = False
        self._poisoned = False
        self._poison_error: BaseException | None = None
        self._released = False

    def _validate_owner(self, session: Any) -> Any:
        if session is not self._session or getattr(session, "session_id", None) != self._session_id:
            raise RuntimeError("stale or foreign persistent NPU KV session")
        if self._released:
            raise RuntimeError("persistent NPU KV reorder is released")
        if self._poisoned:
            raise RuntimeError("persistent NPU KV reorder is poisoned") from self._poison_error
        if self._terminal:
            raise RuntimeError("persistent NPU KV reorder already consumed its final decision")
        if (
            getattr(session, "is_active", None) is not True
            or getattr(session, "_state", None) is not self._session_epoch
        ):
            raise RuntimeError("persistent NPU KV session epoch is stale")
        if _int(getattr(session, "slot", None), "session slot") != self._slot:
            raise RuntimeError("persistent NPU KV session slot changed")
        if _int(getattr(session, "beam_width", None), "Beam width", minimum=1) != self._width:
            raise RuntimeError("persistent NPU KV session Beam width changed")
        if getattr(session, "context", None) is not self._context:
            raise RuntimeError("persistent NPU KV context identity changed")
        current_pools = getattr(self._context, "pools", ())
        if len(current_pools) != len(self._pools) or any(
            pool is not bound for pool, bound in zip(current_pools, self._pools)
        ):
            raise RuntimeError("persistent NPU KV pool binding changed")
        if any(
            getattr(pool, "unshared_key", None) is not key
            or getattr(pool, "unshared_value", None) is not value
            for pool, key, value in zip(self._pools, self._keys, self._values)
        ):
            raise RuntimeError("persistent NPU KV pool tensor identity changed")
        if (
            getattr(self._context, "decision", None) is not self._canonical
            or getattr(self._context, "aux_key_buf", None) is not self._aux_key
            or getattr(self._context, "aux_value_buf", None) is not self._aux_value
            or getattr(self._context, "xllm_block_table_buf", None) is not self._block_table_source
        ):
            raise RuntimeError("persistent NPU KV fixed buffer identity changed")
        current_tensors = (
            getattr(self._canonical, "sequence", None),
            getattr(self._canonical, "beam_scores", None),
            getattr(self._canonical, "group_token_num", None),
            *(getattr(pool, "unshared_key", None) for pool in self._pools),
            *(getattr(pool, "unshared_value", None) for pool in self._pools),
            getattr(self._context, "aux_key_buf", None),
            getattr(self._context, "aux_value_buf", None),
            getattr(self._context, "xllm_block_table_buf", None),
        )
        if len(current_tensors) != len(self._tensor_bindings):
            raise RuntimeError("persistent NPU KV fixed buffer binding count changed")
        for binding, current in zip(self._tensor_bindings, current_tensors):
            binding.validate(current)
        return self._session_epoch

    def reorder(
        self,
        session: Any,
        decision: _DecisionStep,
        *,
        expected_device_step: int,
        is_final: bool,
    ) -> None:
        state = self._validate_owner(session)
        expected = _int(expected_device_step, "expected device step", minimum=1)
        if not isinstance(is_final, bool):
            raise TypeError("is_final must be a boolean")
        if not isinstance(decision, (NPUDecisionStep, NPUMaskedDecisionStep)):
            raise TypeError("decision must be a persistent NPU decision step")
        if decision is self._last_decision:
            raise RuntimeError("persistent NPU KV decision was already consumed")
        if expected != self._next_expected_step:
            raise RuntimeError("persistent NPU KV reorder/device step is stale or out of order")
        if expected + 1 > self._canonical_capacity:
            raise RuntimeError("persistent NPU KV decision exceeds canonical capacity")
        if decision.device_step != expected + 1:
            raise RuntimeError("persistent NPU KV decision/device step mismatch")
        if _int(getattr(state, "decode_step", None), "session decode step") != expected - 1:
            raise RuntimeError("persistent NPU KV session/device step mismatch")
        canonical = self._canonical
        if (
            _int(
                getattr(canonical, "worker_decode_steps", None),
                "canonical decision step",
                minimum=1,
            )
            != expected + 1
        ):
            raise RuntimeError("persistent NPU KV canonical/device step mismatch")
        canonical_sequence = getattr(canonical, "sequence", None)
        canonical_scores = getattr(canonical, "beam_scores", None)
        if (
            not torch.is_tensor(canonical_sequence)
            or not torch.is_tensor(canonical_scores)
            or not torch.is_tensor(decision.sequence)
            or not torch.is_tensor(decision.scores)
        ):
            raise RuntimeError("persistent NPU KV canonical buffers are missing")
        if (
            _storage_id(decision.sequence) != _storage_id(canonical_sequence)
            or decision.sequence.data_ptr() != canonical_sequence.data_ptr()
            or _storage_id(decision.scores) != _storage_id(canonical_scores)
            or decision.scores.data_ptr() != canonical_scores.data_ptr()
        ):
            raise RuntimeError("persistent NPU KV decision is not canonical")

        parents = decision.reorder_parents
        prefix = decision.reorder_prefix
        if is_final:
            if parents is not None or prefix is not None:
                raise RuntimeError("final persistent NPU KV decision requested reorder")
            self._last_decision = decision
            self._terminal = True
            return
        if parents is None or prefix is None:
            raise RuntimeError("non-final persistent NPU KV decision is missing reorder data")
        for name, tensor in (("parents", parents), ("prefix", prefix)):
            tensor = _tensor(tensor, name, self._device)
            if tensor.shape != (1, self._width) or tensor.dtype is not torch.int32:
                raise RuntimeError(f"persistent NPU KV {name} geometry mismatch")
        if decision.parent_ids is not parents:
            raise RuntimeError("persistent NPU KV parent payload identity changed")

        # Consume before enqueue: an exception may follow partial in-place KV
        # writes, so this exact decision and step must never be retried.
        self._last_decision = decision
        self._poisoned = True
        try:
            if self._mode == "xllm":
                assert self._xllm_select is not None
                result = self._xllm_select(
                    parents.view(-1),
                    self._keys,
                    self._values,
                    self._block_table,
                    prefix.view(-1),
                    expected,
                    self._width,
                    len(self._keys),
                )
            else:
                assert self._cascade_select is not None
                result = self._cascade_select(
                    parents.view(-1),
                    self._slot,
                    self._keys,
                    self._values,
                    expected - 1,
                    aux_key_bufs=self._aux_keys,
                    aux_value_bufs=self._aux_values,
                )
            if result is not None:
                raise RuntimeError("persistent NPU KV reorder operator returned a value")
            state.decode_step += 1
        except BaseException as error:
            self._poison_error = error
            raise
        self._poisoned = False
        self._next_expected_step = expected + 1

    def release(self, session: Any) -> bool:
        if session is not self._session:
            raise RuntimeError("stale or foreign persistent NPU KV session")
        if self._released:
            return False
        self._released = True
        return True

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def mode(self) -> Literal["xllm", "cascade"]:
        return self._mode


__all__ = ["NPUKVReorder"]
