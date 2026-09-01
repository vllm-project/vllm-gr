# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone CUDA microbenchmark for the frozen Beam postprocess boundary.

The benchmark starts with logits plus current Beam metadata and ends when the
existing compact ``beam_result_buffer`` bytes are ready.  It intentionally does
not include EngineCore, Scheduler, request reconstruction, or the next decode.

Examples:
    python tests/kernels/benchmark_beam_postprocess_cuda.py
    python tests/kernels/benchmark_beam_postprocess_cuda.py --iterations 100
    python tests/kernels/benchmark_beam_postprocess_cuda.py --beam-widths 256

Output is CSV. ``*_kernel_ms`` measures an operator in isolation;
``postprocess_gpu_ms`` spans TopK through KV selection or FinalSelect; and
``wall_ms`` spans the Python postprocess entry through reusable pinned D2H and
bytes materialization.  The legacy columns use the allocating convenience
wrappers and pageable result encoder as a before-reference.

The current Worker owns one active Beam session.  B=2 is nevertheless measured
at the operator/workspace boundary.  Its KV timing applies the two already
selected request-local parent rows sequentially to two pool slots, which is the
future batched equivalent without introducing cross-decode state here.
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, cast

import torch

from vllm_gr.ops.cuda.beam_selection import (
    beam_search_group_cuda,
    beam_search_group_cuda_out,
    beam_search_group_cuda_worker_out,
    beam_search_rec_final_select_cuda,
    beam_search_rec_final_select_grouped_cuda_out,
)
from vllm_gr.ops.cuda.rec_constrained_topk import (
    rec_constrained_topk_cuda,
    rec_constrained_topk_cuda_out,
)
from vllm_gr.v1.attention.backends.beam_kernels import select_unshared_kv_fused
from vllm_gr.v1.beam.beam_decision import (
    BeamPostprocessWorkspace,
    _encode_cuda_beam_result,
)


@dataclass(slots=True)
class BenchmarkCase:
    batch_size: int
    beam_width: int
    phase: str
    dtype: torch.dtype
    temperature: float
    upstream_mode: str
    current_step: int
    workspace: BeamPostprocessWorkspace
    topk_inputs: tuple[torch.Tensor, ...]
    topk_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    cumulative: torch.Tensor
    sequence: torch.Tensor
    key_caches: list[torch.Tensor]
    value_caches: list[torch.Tensor]
    aux_key: torch.Tensor
    aux_value: torch.Tensor
    key_ptrs: torch.Tensor
    value_ptrs: torch.Tensor
    local_parents: list[torch.Tensor]


def _constraint_inputs(
    *,
    rows: int,
    beam_width: int,
    dtype: torch.dtype,
    temperature: float,
    upstream_mode: str,
) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    vocab_size = 512
    degree = 300
    generator = torch.Generator(device=device).manual_seed(20260905 + rows + beam_width)
    eager_logits = torch.randn(
        (rows, vocab_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    logits = eager_logits
    if upstream_mode == "full_graph":
        logits = torch.empty_like(eager_logits)
        logits.copy_(eager_logits)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits.copy_(eager_logits)
        graph.replay()

    candidates = torch.arange(degree, dtype=torch.int32, device=device)
    prefix1_offsets = torch.full(
        (vocab_size + 1,),
        degree,
        dtype=torch.int32,
        device=device,
    )
    prefix1_offsets[0] = 0
    return (
        logits,
        torch.zeros((rows, 3), dtype=torch.int32, device=device),
        candidates,
        prefix1_offsets,
        candidates,
        torch.zeros(1, dtype=torch.int64, device=device),
        torch.tensor([0, degree], dtype=torch.int32, device=device),
        candidates,
        torch.full((rows,), temperature, dtype=torch.float32, device=device),
    )


def _build_case(
    batch_size: int,
    beam_width: int,
    phase: str,
    dtype: torch.dtype,
    temperature: float,
    upstream_mode: str,
) -> BenchmarkCase:
    current_step = {"step0": 0, "step1": 1, "final": 2}[phase]
    candidate_rows = batch_size if current_step == 0 else batch_size * beam_width
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=batch_size,
        beam_width=beam_width,
        top_k=beam_width,
        max_decode_steps=3,
        device="cuda",
        logits_dtype=dtype,
    )
    topk_outputs = (
        workspace.candidate_rows(
            workspace.topk_token_ids,
            batch_size,
            compact=current_step == 0,
        ),
        workspace.candidate_rows(
            workspace.topk_scores,
            batch_size,
            compact=current_step == 0,
        ),
        workspace.candidate_rows(
            workspace.topk_valid_mask,
            batch_size,
            compact=current_step == 0,
        ),
        workspace.topk_valid_counts.view(-1)[:candidate_rows],
    )
    topk_inputs = _constraint_inputs(
        rows=candidate_rows,
        beam_width=beam_width,
        dtype=dtype,
        temperature=temperature,
        upstream_mode=upstream_mode,
    )
    cumulative_rows = batch_size if current_step == 0 else batch_size * beam_width
    generator = torch.Generator(device="cuda").manual_seed(20260906 + batch_size + beam_width)
    cumulative = torch.randn(
        (cumulative_rows, 1),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    sequence = torch.arange(
        batch_size * beam_width * 3,
        dtype=torch.int32,
        device="cuda",
    ).view(batch_size, beam_width, 3)

    pool_shape = (batch_size, beam_width, 1, 3, 8)
    key = torch.randn(pool_shape, dtype=dtype, device="cuda", generator=generator)
    value = torch.randn(pool_shape, dtype=dtype, device="cuda", generator=generator)
    aux_shape = (1, beam_width, 1, 3, 8)
    aux_key = torch.empty(aux_shape, dtype=dtype, device="cuda")
    aux_value = torch.empty_like(aux_key)
    key_ptrs = torch.tensor([key.data_ptr()], dtype=torch.int64, device="cuda")
    value_ptrs = torch.tensor([value.data_ptr()], dtype=torch.int64, device="cuda")
    case = BenchmarkCase(
        batch_size=batch_size,
        beam_width=beam_width,
        phase=phase,
        dtype=dtype,
        temperature=temperature,
        upstream_mode=upstream_mode,
        current_step=current_step,
        workspace=workspace,
        topk_inputs=topk_inputs,
        topk_outputs=topk_outputs,
        cumulative=cumulative,
        sequence=sequence,
        key_caches=[key],
        value_caches=[value],
        aux_key=aux_key,
        aux_value=aux_value,
        key_ptrs=key_ptrs,
        value_ptrs=value_ptrs,
        local_parents=[],
    )
    _fixed_topk(case)
    grouped = _fixed_group(case)
    case.local_parents = [
        (grouped[1][request] - request * beam_width).contiguous() for request in range(batch_size)
    ]
    return case


def _fixed_topk(case: BenchmarkCase) -> tuple[torch.Tensor, ...]:
    rec_constrained_topk_cuda_out(
        *case.topk_inputs,
        case.current_step,
        case.beam_width,
        300,
        300,
        *case.topk_outputs,
    )
    return case.topk_outputs


def _legacy_topk(case: BenchmarkCase) -> tuple[torch.Tensor, ...]:
    return cast(
        tuple[torch.Tensor, ...],
        rec_constrained_topk_cuda(
            *case.topk_inputs,
            current_step=case.current_step,
            top_k=case.beam_width,
            max_prefix1_degree=300,
            max_prefix2_degree=300,
        ),
    )


def _fixed_group(case: BenchmarkCase) -> tuple[torch.Tensor, ...]:
    workspace = case.workspace
    batch = case.batch_size
    outputs = (
        workspace.selected_token_ids[:batch],
        workspace.selected_parent_ids[:batch],
        workspace.selected_scores[:batch],
        workspace.selected_parent_count_prefix[:batch],
        workspace.next_sequence[:batch],
    )
    scratch = workspace.selection_scratch
    if case.current_step > 0:
        beam_search_group_cuda_worker_out(
            case.cumulative,
            case.topk_outputs[0],
            case.topk_outputs[1],
            case.sequence,
            case.current_step,
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[4],
            workspace.selected_flat_indices[:batch],
            scratch.local_scores,
            scratch.local_flat_indices,
            case.topk_outputs[2],
            case.topk_outputs[3],
            workspace.prefill_staging[:batch],
            workspace.prefill_staging[workspace.max_batch :],
        )
    else:
        beam_search_group_cuda_out(
            case.cumulative,
            case.topk_outputs[0],
            case.topk_outputs[1],
            case.sequence,
            case.current_step,
            case.beam_width,
            *outputs,
            scratch.local_scores,
            scratch.local_flat_indices,
            scratch.merge_scores,
            scratch.merge_flat_indices,
            case.topk_outputs[2],
            workspace.selected_flat_indices[:batch],
        )
    return outputs


def _legacy_group(
    case: BenchmarkCase,
    topk: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return cast(
        tuple[torch.Tensor, ...],
        beam_search_group_cuda(
            case.cumulative,
            topk[0],
            topk[1],
            case.sequence,
            case.current_step,
            case.beam_width,
            topk[2],
        ),
    )


def _fixed_final(
    case: BenchmarkCase,
    grouped: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    workspace = case.workspace
    batch = case.batch_size
    outputs = (
        workspace.final_token_ids[:batch],
        workspace.final_parent_ids[:batch],
        workspace.final_scores[:batch],
        workspace.final_sequence[:batch],
    )
    beam_search_rec_final_select_grouped_cuda_out(
        grouped[0],
        grouped[1],
        grouped[2],
        grouped[4],
        case.beam_width,
        *outputs,
        workspace.selected_flat_indices[:batch],
    )
    return outputs


def _legacy_final(
    case: BenchmarkCase,
    topk: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    outputs = beam_search_rec_final_select_cuda(
        case.cumulative,
        topk[0],
        topk[1],
        case.sequence,
        case.current_step,
        case.beam_width,
        topk[2],
    )
    batch, width = case.batch_size, case.beam_width
    return (
        outputs[0].view(batch, width),
        outputs[1].view(batch, width),
        outputs[2].view(batch, width),
        outputs[3],
    )


def _fixed_kv(case: BenchmarkCase) -> None:
    for slot, parents in enumerate(case.local_parents):
        ok = select_unshared_kv_fused(
            parents,
            slot,
            case.key_caches,
            case.value_caches,
            0,
            case.aux_key,
            case.aux_value,
            case.key_ptrs,
            case.value_ptrs,
        )
        if not ok:
            raise RuntimeError("fixed SelectUnsharedKV benchmark path was unavailable")


def _legacy_kv(case: BenchmarkCase) -> None:
    for slot, parents in enumerate(case.local_parents):
        ok = select_unshared_kv_fused(
            parents,
            slot,
            case.key_caches,
            case.value_caches,
            0,
            case.aux_key,
            case.aux_value,
        )
        if not ok:
            raise RuntimeError("legacy SelectUnsharedKV benchmark path was unavailable")


def _decision(outputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    return {
        "out_token_ids": outputs[0],
        "out_token_index": outputs[1],
        "out_log_probs": outputs[2],
    }


def _run_fixed(case: BenchmarkCase, *, pack: bool) -> dict[str, torch.Tensor]:
    _fixed_topk(case)
    grouped = _fixed_group(case)
    if case.phase == "final":
        result = _fixed_final(case, grouped)
    else:
        result = grouped
        if case.phase == "step1":
            _fixed_kv(case)
    decision = _decision(result)
    if pack:
        encoded = _encode_cuda_beam_result(
            decision,
            is_last_step=case.phase == "final",
            workspace=case.workspace,
        )
        if encoded is None:
            raise RuntimeError("fixed Beam result encoder rejected benchmark outputs")
    return decision


def _run_legacy(case: BenchmarkCase, *, pack: bool) -> dict[str, torch.Tensor]:
    topk = _legacy_topk(case)
    if case.phase == "final":
        result = _legacy_final(case, topk)
    else:
        result = _legacy_group(case, topk)
        if case.phase == "step1":
            _legacy_kv(case)
    decision = _decision(result)
    if pack:
        encoded = _encode_cuda_beam_result(
            decision,
            is_last_step=case.phase == "final",
        )
        if encoded is None:
            raise RuntimeError("legacy Beam result encoder rejected benchmark outputs")
    return decision


def _cuda_time_ms(fn: Callable[[], Any], iterations: int, warmups: int) -> float:
    for _ in range(warmups):
        fn()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _wall_time_ms(fn: Callable[[], Any], iterations: int, warmups: int) -> float:
    for _ in range(warmups):
        fn()
    torch.accelerator.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / iterations


def _check_precision(case: BenchmarkCase) -> None:
    legacy = _run_legacy(case, pack=False)
    fixed = _run_fixed(case, pack=False)
    torch.accelerator.synchronize()
    for key in ("out_token_ids", "out_token_index", "out_log_probs"):
        torch.testing.assert_close(fixed[key], legacy[key], rtol=0, atol=0)
    legacy_encoded = _encode_cuda_beam_result(
        legacy,
        is_last_step=case.phase == "final",
    )
    fixed_encoded = _encode_cuda_beam_result(
        fixed,
        is_last_step=case.phase == "final",
        workspace=case.workspace,
    )
    if legacy_encoded != fixed_encoded:
        raise AssertionError("fixed and legacy beam_result_buffer bytes differ")


def benchmark_case(
    case: BenchmarkCase,
    *,
    iterations: int,
    warmups: int,
) -> dict[str, float]:
    _check_precision(case)
    _fixed_topk(case)
    grouped = _fixed_group(case)
    legacy_topk = _legacy_topk(case)

    metrics = {
        "topk_kernel_ms": _cuda_time_ms(lambda: _fixed_topk(case), iterations, warmups),
        "group_kernel_ms": _cuda_time_ms(lambda: _fixed_group(case), iterations, warmups),
        "kv_kernel_ms": 0.0,
        "final_kernel_ms": 0.0,
    }
    if case.phase == "step1":
        metrics["kv_kernel_ms"] = _cuda_time_ms(lambda: _fixed_kv(case), iterations, warmups)
    if case.phase == "final":
        metrics["final_kernel_ms"] = _cuda_time_ms(
            lambda: _fixed_final(case, grouped), iterations, warmups
        )
    # Keep the allocating reference's group/final specializations warm before
    # measuring the complete paths.
    if case.phase == "final":
        _legacy_final(case, legacy_topk)
    else:
        _legacy_group(case, legacy_topk)

    metrics["postprocess_gpu_ms"] = _cuda_time_ms(
        lambda: _run_fixed(case, pack=False), iterations, warmups
    )
    metrics["legacy_gpu_ms"] = _cuda_time_ms(
        lambda: _run_legacy(case, pack=False), iterations, warmups
    )
    metrics["wall_ms"] = _wall_time_ms(lambda: _run_fixed(case, pack=True), iterations, warmups)
    metrics["legacy_wall_ms"] = _wall_time_ms(
        lambda: _run_legacy(case, pack=True), iterations, warmups
    )
    metrics["wall_speedup"] = metrics["legacy_wall_ms"] / metrics["wall_ms"]
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["step0", "step1", "final"],
        default=["step0", "step1", "final"],
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float16", "bfloat16"],
        default=["float16", "bfloat16"],
    )
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.6])
    parser.add_argument(
        "--upstream-modes",
        nargs="+",
        choices=["eager", "full_graph"],
        default=["eager", "full_graph"],
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    args = parse_args()
    if args.iterations < 1 or args.warmups < 0:
        raise SystemExit("iterations must be positive and warmups non-negative")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    header = (
        "beam_width,batch_size,phase,dtype,temperature,upstream_mode,"
        "topk_kernel_ms,group_kernel_ms,kv_kernel_ms,final_kernel_ms,"
        "postprocess_gpu_ms,wall_ms,legacy_gpu_ms,legacy_wall_ms,wall_speedup"
    )
    print(header, flush=True)
    combinations = itertools.product(
        args.beam_widths,
        args.batch_sizes,
        args.phases,
        args.dtypes,
        args.temperatures,
        args.upstream_modes,
    )
    for width, batch, phase, dtype_name, temperature, upstream_mode in combinations:
        if not 1 <= width <= 256:
            raise SystemExit("this fixed CUDA benchmark supports beam widths in [1, 256]")
        case = _build_case(
            batch,
            width,
            phase,
            dtype_map[dtype_name],
            temperature,
            upstream_mode,
        )
        metrics = benchmark_case(
            case,
            iterations=args.iterations,
            warmups=args.warmups,
        )
        values = (
            width,
            batch,
            phase,
            dtype_name,
            temperature,
            upstream_mode,
            metrics["topk_kernel_ms"],
            metrics["group_kernel_ms"],
            metrics["kv_kernel_ms"],
            metrics["final_kernel_ms"],
            metrics["postprocess_gpu_ms"],
            metrics["wall_ms"],
            metrics["legacy_gpu_ms"],
            metrics["legacy_wall_ms"],
            metrics["wall_speedup"],
        )
        print(
            ",".join(
                f"{value:.6f}" if isinstance(value, float) else str(value) for value in values
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
