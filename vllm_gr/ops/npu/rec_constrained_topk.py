# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU dispatch wrapper for the fused constrained Top-K operator."""

from __future__ import annotations

from types import ModuleType

import torch


def _load_xllm_ops() -> None:
    """Load the shared xLLM extension before probing its Python bindings.

    ACS uses this lazy load because the ``custom_xllm_math`` CANN vendor is
    linked by ``libxllm_ops.so``.  Importing ``custom_ops_lib`` alone does not
    necessarily register the rec_constrained_topk binding.
    """
    if hasattr(torch.ops, "xllm_ops"):
        return
    try:
        torch.utils.cpp_extension.load(
            name="xllm_ops",
            sources=[],
            extra_ldflags=["-lxllm_ops"],
        )
    except Exception:
        # The extension may already be loaded by the runtime or by an
        # LD_PRELOAD/setup script.  The availability probe below is the
        # authoritative check.
        pass


def _custom_ops() -> ModuleType | None:
    """Load the xLLM Python binding used by the ACS integration lazily.

    ``custom_xllm_math`` is the CANN OPP vendor name, not the Python module
    name.  Its environment script makes ``libcust_opapi.so`` discoverable;
    the callable Python API is exported by ``custom_ops_lib``.
    """
    _load_xllm_ops()
    try:
        import custom_ops_lib as custom_ops  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        return None
    return custom_ops


def is_available() -> bool:
    """Return whether the NPU extension exposes the fused operator."""
    custom_ops = _custom_ops()
    return custom_ops is not None and callable(getattr(custom_ops, "rec_constrained_topk", None))


def unavailable_reason() -> str:
    """Describe why the fused NPU operator cannot be called."""
    custom_ops = _custom_ops()
    if custom_ops is None:
        return "custom_ops_lib cannot be imported"
    if not callable(getattr(custom_ops, "rec_constrained_topk", None)):
        return "custom_ops_lib does not export rec_constrained_topk"
    return ""


def rec_constrained_topk_npu(
    logits: torch.Tensor,
    sequence_group: torch.Tensor,
    first_token_ids: torch.Tensor,
    prefix1_offsets: torch.Tensor,
    prefix1_values: torch.Tensor,
    prefix1_pair_keys: torch.Tensor,
    prefix2_value_offsets: torch.Tensor,
    prefix2_values: torch.Tensor,
    temperatures: torch.Tensor,
    current_step: int,
    top_k: int,
    max_prefix1_degree: int = 0,
    max_prefix2_degree: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the fused NPU operator using the shared operator signature."""
    custom_ops = _custom_ops()
    if custom_ops is None or not callable(getattr(custom_ops, "rec_constrained_topk", None)):
        raise RuntimeError(
            f"NPU rec_constrained_topk is unavailable ({unavailable_reason()}); "
            "use the Torch implementation instead"
        )
    return custom_ops.rec_constrained_topk(
        logits,
        sequence_group,
        first_token_ids,
        prefix1_offsets,
        prefix1_values,
        prefix1_pair_keys,
        prefix2_value_offsets,
        prefix2_values,
        temperatures,
        current_step,
        top_k,
        max_prefix1_degree,
        max_prefix2_degree,
    )


__all__ = ["is_available", "rec_constrained_topk_npu", "unavailable_reason"]
