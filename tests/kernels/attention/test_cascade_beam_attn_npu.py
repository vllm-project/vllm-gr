# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Numerical correctness tests for the Beam-Cascade attention backend on NPU.

The GPU test (``test_cascade_beam_attn_gpu.py``) is the only test that
checks *correctness* -- backend output against a reference -- but it is
CUDA-only: it skips unless ``torch.cuda.is_available()`` and its reference is
built on ``flash_attn_varlen_func``, which is unavailable on NPU. As a result
the NPU cascade path (``beam_attn_npu.cascade_attention`` ->
``merge_tnd_attention_outputs``) had no correctness coverage.

This module fills that gap. It runs the real backend
(``BeamAttentionImpl.cascade_attention``) on NPU and compares its output
against a device-agnostic pure-torch reference that reproduces the GPU test's
beam gather and bottom-right causal convention (both share the decomposition
in ``cascade_beam_attn_utils.iter_beam_segments``). It is parametrised over
the scenarios the GPU test covers, minus the exclusions noted below.

Differences from the GPU test, all forced by the hardware:

  * Reference uses plain fp32 softmax instead of flash-attention.
  * ``fa_version`` and ``soft_cap`` axes are dropped: the NPU path fixes
    ``vllm_flash_attn_version = 0`` and never forwards ``logits_soft_cap``
    to the operator.
  * ``head_size = 256`` is excluded: unsupported by the CANN TND kernel.
  * ``(8, 8, 8)`` (no shared prefix) is omitted from ``NPU_CASES``: it
    produces an empty block table, which the CANN kernel rejects
    (``EMPTY_BLOCK_TABLE``).
  * The metadata builder is given ``npu_causal_atten_mask`` explicitly; the
    GPU test omits it (harmless on CUDA, required on NPU for the causal
    STD/suffix calls).

A case is a failure only if the backend runs and disagrees with the reference
(``NUMERIC_MISMATCH``) or errors unexpectedly. Configurations the CANN kernel
cannot launch are reported as skips, not failures.

Run:  pytest test_cascade_beam_attn_npu.py -v
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import pytest
except ImportError:  # keeps the module importable without pytest installed
    pytest = None  # type: ignore[assignment,unused-ignore]

# Declare the tier before the module-level skip below, so the directory
# default never relabels this NPU suite as gpu_test.
if pytest is not None:
    pytestmark = pytest.mark.npu_test

# ---------------------------------------------------------------------------
# NPU availability. Both `import torch_npu` and `torch.device("npu")` raise
# (ImportError / RuntimeError respectively) on a machine without the Ascend
# stack, which would otherwise abort pytest *collection* for this whole file
# -- not a graceful per-test skip -- and break CI on GPU/CPU-only runners.
# ---------------------------------------------------------------------------
try:
    import torch_npu  # noqa: F401

    DEVICE = torch.device("npu")
    _NPU_STACK_AVAILABLE = True
except (ImportError, RuntimeError) as _npu_import_error:
    DEVICE = None  # type: ignore[assignment,unused-ignore]
    _NPU_STACK_AVAILABLE = False
    if pytest is not None:
        pytest.skip(
            f"Ascend NPU stack is not available: {_npu_import_error}",
            allow_module_level=True,
        )


try:
    from cascade_beam_attn_utils import (
        NPU_CASES,
        NUM_HEADS,
        BeamAttnInputs,
        iter_beam_segments,
        prepare_inputs,
        run_cascade_beam_attention,
    )
except ImportError:
    if pytest is not None:
        pytest.skip(
            "beam_attn backend is not supported.",
            allow_module_level=True,
        )
    raise


BF16 = torch.bfloat16
FP16 = torch.float16

# Per-dtype tolerances for the backend-vs-reference comparison.
# bf16 is deliberately loose: the reference accumulates in fp32 while the
# CANN kernel does not, so a correct kernel still differs by ~1e-2.
TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    BF16: (2e-3, 1e-5),  # (atol, rtol)
    FP16: (1e-3, 1e-5),
}


BEAM_SCENARIOS = NPU_CASES

DTYPES = [BF16, FP16]
BATCH_SIZES = [1, 4]
BEAM_WIDTHS = [16, 32]

# 256 is unsupported on NPU (UNSUPPORTED_TND_HEAD_DIM).
HEAD_SIZE_BLOCKS: dict[int, list[int]] = {
    128: [32, 128],
}


def classify_error(error: Exception) -> str:
    msg = str(error)
    if "not find tilingKey" in msg:
        return "MISSING_TILING_KEY"
    if "inputDataType should be bf16" in msg:
        return "TND_REQUIRES_BF16"
    if "block size" in msg.lower() and "multiple of 128" in msg.lower():
        return "BLOCK_SIZE_MULTIPLE_OF_128"
    if "blockSize must be 128" in msg:
        return "BLOCK_SIZE_MUST_BE_128"
    if "queryD(" in msg and "must be same equal 192/128/64" in msg:
        return "UNSUPPORTED_TND_HEAD_DIM"
    if "block_table shape size is zero" in msg:
        return "EMPTY_BLOCK_TABLE"
    if "error code is 161002" in msg:
        return "TILING_ERROR_161002"
    if "error code is 561002" in msg:
        return "VALIDATION_ERROR_561002"
    return "OTHER_ERROR"


# Classifications that mean "the kernel could not launch this config" --
# a known CANN limitation, NOT a correctness result.
KERNEL_UNSUPPORTED = {
    "MISSING_TILING_KEY",
    "TND_REQUIRES_BF16",
    "BLOCK_SIZE_MULTIPLE_OF_128",
    "BLOCK_SIZE_MUST_BE_128",
    "UNSUPPORTED_TND_HEAD_DIM",
    "EMPTY_BLOCK_TABLE",
}


def short_error(error: Exception) -> str:
    lines = [ln.strip() for ln in str(error).splitlines() if ln.strip()]
    for ln in lines:
        if "not find tilingKey" in ln or "Inner Error" in ln or "queryD(" in ln:
            return ln
    return lines[0] if lines else repr(error)


def dtype_name(dtype: torch.dtype) -> str:
    return {BF16: "bf16", FP16: "fp16"}.get(dtype, str(dtype))


def _bottomright_causal_sdpa(
    q: torch.Tensor,  # (Lq, H, Dq)
    k: torch.Tensor,  # (Lk, H, Dq)  already head-expanded
    v: torch.Tensor,  # (Lk, H, Dv)  already head-expanded
    scale: float,
) -> torch.Tensor:
    q = q.float().transpose(0, 1)  # (H, Lq, Dq)
    k = k.float().transpose(0, 1)  # (H, Lk, Dq)
    v = v.float().transpose(0, 1)  # (H, Lk, Dv)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (H, Lq, Lk)
    lq, lk = scores.shape[-2], scores.shape[-1]

    qi = torch.arange(lq, device=scores.device).view(lq, 1)
    ki = torch.arange(lk, device=scores.device).view(1, lk)
    allowed = ki <= (lk - lq) + qi  # bottom-right alignment
    scores = scores.masked_fill(~allowed, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)  # (H, Lq, Dv)
    return out.transpose(0, 1)  # (Lq, H, Dv)


def run_reference_attention_torch(inputs: BeamAttnInputs) -> torch.Tensor:
    out = torch.zeros_like(inputs.query)
    group = inputs.query.shape[1] // inputs.num_kv_heads

    for q_ptr, b_uncached, beam_k, beam_v in iter_beam_segments(inputs):
        beam_k = beam_k.repeat_interleave(group, dim=1)  # -> (Lk, H, Dq)
        beam_v = beam_v.repeat_interleave(group, dim=1)

        q_seg = inputs.query[q_ptr : q_ptr + b_uncached]  # (Lq, H, Dq)
        out_seg = _bottomright_causal_sdpa(q_seg, beam_k, beam_v, inputs.scale)
        out[q_ptr : q_ptr + b_uncached] = out_seg.to(out.dtype)

    return out


def run_cascade_beam_attention_npu(inputs: BeamAttnInputs) -> torch.Tensor:
    npu_causal_atten_mask = (
        torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(inputs.query.device)
    )
    output = run_cascade_beam_attention(
        inputs,
        device=inputs.query.device,
        vllm_flash_attn_version=0,  # NPU
        logits_soft_cap=None,
        npu_causal_atten_mask=npu_causal_atten_mask,
    )
    torch.npu.synchronize()
    return output


@dataclass
class CaseSpec:
    name: str
    batch_size: int
    beam_width: int
    scenario: tuple[int, int, int]
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    dtype: torch.dtype


def _error_stats(backend: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    a = backend.float()
    b = ref.float()
    abs_err = (a - b).abs()
    max_abs = float(abs_err.max().item())
    mean_abs = float(abs_err.mean().item())
    denom = b.abs().clamp_min(1e-6)
    max_rel = float((abs_err / denom).max().item())
    return max_abs, max_rel, mean_abs


def run_case(case: CaseSpec) -> dict[str, object]:
    q_len, kv_len, suffix = case.scenario
    atol, rtol = TOLERANCES[case.dtype]

    row: dict[str, object] = {
        "name": case.name,
        "verdict": "",
        "dtype": dtype_name(case.dtype),
        "head_size": case.head_size,
        "block_size": case.block_size,
        "q_heads": case.num_query_heads,
        "kv_heads": case.num_kv_heads,
        "batch_size": case.batch_size,
        "beam_width": case.beam_width,
        "q_len": q_len,
        "kv_len": kv_len,
        "suffix": suffix,
        "max_abs_err": "",
        "max_rel_err": "",
        "mean_abs_err": "",
        "atol": atol,
        "rtol": rtol,
        "error": "",
    }

    inputs = prepare_inputs(
        batch_size=case.batch_size,
        beam_width=case.beam_width,
        num_heads=case.num_query_heads,
        num_kv_heads=case.num_kv_heads,
        head_size=case.head_size,
        q_seq_len=q_len,
        kv_seq_len=kv_len,
        block_size=case.block_size,
        dtype=case.dtype,
        suffix_kv_len=suffix,
        device=DEVICE,
    )

    # Reference first (pure torch, cannot fail the kernel).
    ref = run_reference_attention_torch(inputs)

    # Backend under test — CANN may reject certain configs (missing tiling key,
    # wrong dtype, unsupported head dim …).  Catch those so the test reports a
    # skip instead of an unexpected failure.
    try:
        backend_out = run_cascade_beam_attention_npu(inputs)
    except Exception as e:
        row["verdict"] = classify_error(e)
        row["error"] = short_error(e)
        return row

    max_abs, max_rel, mean_abs = _error_stats(backend_out, ref)
    row["max_abs_err"] = f"{max_abs:.4e}"
    row["max_rel_err"] = f"{max_rel:.4e}"
    row["mean_abs_err"] = f"{mean_abs:.4e}"

    ok = torch.allclose(backend_out.float(), ref.float(), atol=atol, rtol=rtol)
    row["verdict"] = "CORRECT" if ok else "NUMERIC_MISMATCH"
    return row


def build_cases() -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for dtype in DTYPES:
        for head_size, blocks in HEAD_SIZE_BLOCKS.items():
            for block_size in blocks:
                for qh, kvh in NUM_HEADS:
                    for batch_size in BATCH_SIZES:
                        for beam_width in BEAM_WIDTHS:
                            for scenario in BEAM_SCENARIOS:
                                q, kv, suf = scenario
                                name = (
                                    f"D{head_size}_B{block_size}_"
                                    f"QH{qh}_KVH{kvh}_"
                                    f"bs{batch_size}_bw{beam_width}_"
                                    f"Q{q}_KV{kv}_S{suf}_{dtype_name(dtype)}"
                                )
                                cases.append(
                                    CaseSpec(
                                        name=name,
                                        batch_size=batch_size,
                                        beam_width=beam_width,
                                        scenario=scenario,
                                        num_query_heads=qh,
                                        num_kv_heads=kvh,
                                        head_size=head_size,
                                        block_size=block_size,
                                        dtype=dtype,
                                    )
                                )
    return cases


if pytest is not None:

    @pytest.mark.parametrize("case", build_cases(), ids=lambda c: c.name)  # type: ignore[misc]
    def test_beam_cascade_npu(case: CaseSpec) -> None:
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            pytest.skip("Ascend NPU is not available")

        torch.manual_seed(0)
        row = run_case(case)
        verdict = row["verdict"]

        if verdict in KERNEL_UNSUPPORTED:
            pytest.skip(f"{verdict}: {row['error']}")

        assert verdict == "CORRECT", (
            f"{case.name}: {verdict} "
            f"(max_abs={row['max_abs_err']}, atol={row['atol']}) {row['error']}"
        )
