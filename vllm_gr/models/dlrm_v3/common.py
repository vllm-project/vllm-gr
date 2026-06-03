# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

"""
DLRM_v3 Model Common Utilities

Contains HammerModule base class, kernel selection, and helper functions.

This fork preserves the upstream kernel-selection API, but only a subset of
helpers currently performs backend-aware dispatch. In practice, the jagged
tensor wrappers below may route into torch.ops.fbgemm via ReliableOps, while
most higher-level operators still execute through PyTorch implementations.
"""

import abc
import copy
from enum import Enum, unique
from typing import Any, List, Optional

import torch

from vllm_gr.utils import ReliableOps


@unique
class HammerKernel(Enum):
    """Requested kernel backends.

    Values in this enum describe the caller's preferred backend, not a blanket
    guarantee that every operator in this fork has a specialized implementation
    for that backend.
    """

    TRITON = "TRITON"
    TLX = "TLX"
    PYTORCH = "PYTORCH"
    CUDA = "CUDA"
    NPU = "NPU"
    TRITON_CC = "TRITON_CC"


class HammerModule(torch.nn.Module, abc.ABC):
    """
    HSTU Module Base Class

    Provides kernel selection and inference mode control.
    Tracks the desired kernel selection and inference mode control.
    The current fork still executes most operators through the PyTorch path,
    but we preserve the kernel-selection semantics so runtime policy and tests
    stay aligned with the upstream contract.
    """

    _use_triton_cc: bool = False
    _training_dtype: torch.dtype = torch.float32
    _hammer_kernel: Optional[HammerKernel] = None

    def __init__(
        self,
        training_dtype: torch.dtype = torch.float32,
        use_triton_cc: bool = False,
        hammer_kernel: Optional[HammerKernel] = None,
    ) -> None:
        super().__init__()
        self._training_dtype = training_dtype
        self._hammer_kernel = hammer_kernel
        self._use_triton_cc = use_triton_cc

    def hammer_kernel(self) -> HammerKernel:
        """Return the desired kernel type for this module.

        Downstream operators may still fall back to PyTorch if this fork does
        not currently implement the requested backend-specific path.
        """
        kernel = self._hammer_kernel
        if kernel is not None:
            return kernel
        if self._use_triton_cc:
            return HammerKernel.TRITON_CC
        return HammerKernel.TRITON

    def recursive_setattr(self, name: str, value: Any) -> None:
        for _, module in self.named_modules():
            if hasattr(module, name):
                setattr(module, name, value)

    def set_use_triton_cc(self, use_triton_cc: bool) -> None:
        self._use_triton_cc = use_triton_cc
        self.recursive_setattr("_use_triton_cc", use_triton_cc)

    def set_training_dtype(self, training_dtype: torch.dtype) -> None:
        self._training_dtype = training_dtype
        self.recursive_setattr("_training_dtype", training_dtype)

    def set_hammer_kernel(self, hammer_kernel: HammerKernel) -> None:
        self._hammer_kernel = hammer_kernel
        self.recursive_setattr("_hammer_kernel", hammer_kernel)

    @property
    def is_eval(self) -> bool:
        return not self.training

    @property
    def is_train(self) -> bool:
        return self.training


# Global settings
STATIC_MAX_SEQ_LENS: List[int] = []
USE_RUNTIME_MAX_SEQ_LEN: bool = True


def set_static_max_seq_lens(max_seq_lens: List[int]) -> None:
    global STATIC_MAX_SEQ_LENS
    STATIC_MAX_SEQ_LENS = copy.deepcopy(max_seq_lens)
    STATIC_MAX_SEQ_LENS.sort()


def set_use_runtime_max_seq_len(use_runtime_max_seq_len: bool) -> None:
    global USE_RUNTIME_MAX_SEQ_LEN
    USE_RUNTIME_MAX_SEQ_LEN = use_runtime_max_seq_len


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def prev_power_of_2(x: int) -> int:
    out = next_power_of_2(x)
    return out // 2 if out > x else out


def autotune_max_seq_len(runtime_max_seq_len: int) -> int:
    global USE_RUNTIME_MAX_SEQ_LEN

    if USE_RUNTIME_MAX_SEQ_LEN:
        return prev_power_of_2(runtime_max_seq_len)
    else:
        if STATIC_MAX_SEQ_LENS == []:
            return 1
        for max_len in STATIC_MAX_SEQ_LENS:
            if max_len >= runtime_max_seq_len:
                return max_len
        return STATIC_MAX_SEQ_LENS[-1]


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


@torch.fx.wrap
def fx_unwrap_optional_tensor(optional: Optional[torch.Tensor]) -> torch.Tensor:
    assert optional is not None, "Expected optional to be non-None Tensor"
    return optional


@torch.fx.wrap
def fx_infer_max_len(lengths: torch.Tensor) -> int:
    max_len = int(lengths.max().item())
    return max_len


@torch.fx.wrap
def fx_mark_length_features(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


@torch.fx.wrap
def fx_arange(len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(len, device=device)


@torch.fx.wrap
def fx_torch_ones(
    shape: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ones(shape, device=device, dtype=dtype)


@torch.fx.wrap
def fx_torch_zeros(shape: List[int], device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, device=device)


def init_mlp_weights_optional_bias(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


@torch.fx.wrap
def jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[int],
    padding_value: float,
) -> torch.Tensor:
    """
    Convert Jagged Tensor to padded dense tensor

    Prefers fbgemm, falls back to PyTorch implementation if unavailable.
    """
    return ReliableOps.jagged_to_padded_dense(
        values=values,
        offsets=offsets,
        max_lengths=max_lengths,
        padding_value=padding_value,
    )


@torch.fx.wrap
def dense_to_jagged(
    dense: torch.Tensor,
    x_offsets: List[torch.Tensor],
) -> torch.Tensor:
    """
    Convert padded dense tensor to Jagged Tensor

    Prefers fbgemm, falls back to PyTorch implementation if unavailable.
    """
    return ReliableOps.dense_to_jagged(
        dense=dense,
        x_offsets=x_offsets,
    )
