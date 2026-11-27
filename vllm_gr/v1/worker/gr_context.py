# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared GR configuration, context construction and graph diagnostics."""

from typing import Any

from vllm_gr.config import GRConfig


def _get_gr_config(runner: Any) -> GRConfig:
    return GRConfig.from_vllm_config(getattr(runner, "vllm_config", None))


def _beam_full_graph_ready(runner: Any, num_tokens: int) -> bool:
    """Query the GPU Beam component; retain the device-neutral NPU key contract."""
    graph = getattr(runner, "_beam_decode_graph", None)
    if graph is not None:
        return bool(graph.ready and num_tokens == graph.signature.num_model_rows)
    beam_keys = getattr(runner, "_beam_graph_keys", None)
    return beam_keys is not None and num_tokens in beam_keys


def _beam_graph_enabled_for_run(runner: Any) -> bool:
    """Beam graphs are only used when the config enables them (GRConfig)."""
    return bool(_get_gr_config(runner).beam_graph_enabled)


def get_beam_graph_runtime_stats(
    worker: Any, *, cuda_allocator: str | None = None
) -> dict[str, Any]:
    """Return capture/dispatch state and optional Worker-local allocator peaks.

    A benchmark can request ``reset`` after warmup and ``read`` after the
    measured calls. Using this named RPC avoids serializing Python callables
    when the inference worker lives in a separate EngineCore.

    vLLM utility RPC callables receive the Worker, while Beam graph state is
    owned by its ModelRunner.  Keeping this probe in an importable module makes
    it safe under both ``fork`` and ``spawn`` and lets profiling fail loudly
    when startup capture succeeded but the measured request ran eager.
    """

    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("worker has no model_runner for Beam graph diagnostics")
    config = _get_gr_config(runner)
    stats: dict[str, Any] = {
        "runner": type(runner).__name__,
        "beam_graph_enabled": bool(config.beam_graph_enabled),
        "beam_max_width": config.beam_max_width,
        "captured_widths": sorted(int(width) for width in getattr(runner, "_beam_graph_keys", ())),
        "full_dispatches": int(getattr(runner, "_beam_graph_full_dispatch_count", 0)),
    }
    prefill = getattr(runner, "_prefill_graph_runner", None)
    if prefill is not None:
        stats["gr_prefill_buckets"] = list(prefill.buckets)
        stats["gr_prefill_captures"] = sorted(
            [int(bucket), int(num_reqs)]
            for bucket, buffers in prefill.buffers.items()
            for num_reqs in buffers.captured_graphs
        )
        stats["gr_prefill_kv_bounds"] = {
            f"{bucket}:{num_reqs}": int(bound)
            for bucket, buffers in prefill.buffers.items()
            for num_reqs, bound in buffers.max_seq_len_bounds.items()
            if num_reqs in buffers.captured_graphs
        }
    stats["gr_prefill_dispatches"] = dict(getattr(runner, "_gr_prefill_graph_dispatches", {}))
    manager = getattr(runner, "cudagraph_manager", None)
    counts = getattr(manager, "dispatch_counts", None)
    if counts is not None:
        assert manager is not None
        stats["prefill_graph_captured_buckets"] = sorted(
            {desc.num_tokens for desc in manager._successful if desc.cg_mode.name == "FULL"}
        )
        stats["prefill_graph_captured_shapes"] = sorted(
            (desc.num_reqs, desc.num_tokens)
            for desc in manager._successful
            if desc.cg_mode.name == "FULL"
        )
        stats["prefill_graph_piecewise_buckets"] = sorted(
            desc.num_tokens for desc in manager._successful if desc.cg_mode.name == "PIECEWISE"
        )
        stats["prefill_graph_full_capture_failures"] = {
            f"{desc.cg_mode.name}:T={desc.num_tokens}:B={desc.num_reqs}": reason
            for desc, reason in manager.full_capture_failures.items()
        }
        stats["prefill_graph_full_dispatches"] = counts["FULL"]
        stats["prefill_graph_piecewise_dispatches"] = counts["PIECEWISE"]
        stats["prefill_graph_none_dispatches"] = counts["NONE"]
    device_type = str(getattr(runner, "device", "")).split(":", 1)[0].lower()
    if device_type == "npu" or hasattr(runner, "_gr_npu_beam_graph_shapes"):
        stats.update(
            {
                "beam_graph_batch_size": int(getattr(config, "beam_graph_batch_size", 1)),
                "captured_shapes": sorted(
                    tuple(int(value) for value in shape)
                    for shape in getattr(runner, "_gr_npu_beam_graph_shapes", ())
                ),
                "fallbacks": int(getattr(runner, "_beam_graph_fallback_count", 0)),
                "fallback_reasons": dict(getattr(runner, "_beam_graph_fallback_reasons", {})),
            }
        )
    if cuda_allocator is not None:
        stats["cuda_allocator"] = _cuda_allocator_stats(worker, runner, cuda_allocator)
    return stats


def _cuda_allocator_stats(worker: Any, runner: Any, operation: str) -> dict[str, int]:
    import torch

    device = torch.device(getattr(worker, "device", None) or runner.device)
    if device.type != "cuda" or operation not in ("reset", "read"):
        raise ValueError("CUDA allocator diagnostics require a CUDA worker and reset or read")
    torch.accelerator.synchronize(device)
    if operation == "reset":
        torch.accelerator.reset_peak_memory_stats(device)
    stage = getattr(runner, "_gr_gpu_stage_runner", None)
    head = getattr(stage, "_stream_head", None)
    return {
        "peak_allocated_bytes": torch.accelerator.max_memory_allocated(device),
        "peak_reserved_bytes": torch.accelerator.max_memory_reserved(device),
        "streaming_workspace_bytes": head.workspace_bytes if head is not None else 0,
    }


def _create_beam_context(runner: Any, gr_cfg: Any, max_batch: int = 1) -> Any:
    """Create a BeamAttentionContext with sizing derived from model + GRConfig.

    Shared by the dummy-capture paths (GPU/NPU ``capture_beam_decode_graph``)
    and the persistent NPU stage runner so
    that the pool dimensions, dtype, and device are always derived from the
    same sources.  A single source of truth prevents silent size mismatches
    that would invalidate captured graphs if the runtime path rebuilt the
    context with different parameters.

    Args:
        runner: The ModelRunner (must have ``model_config``,
            ``parallel_config``, ``device``).
        gr_cfg: The GRConfig (provides ``beam_max_width``,
            ``beam_max_decode_steps``).  ``beam_max_width`` must not be
            None — callers must guard before reaching this function.
        max_batch: Maximum concurrent beam search sessions.

    Returns:
        A new ``BeamAttentionContext``.

    Raises:
        TypeError: If ``gr_cfg.beam_max_width`` is None.
    """
    if gr_cfg.beam_max_width is None:
        raise TypeError(
            "beam_max_width must be set before creating a BeamAttentionContext "
            "(beam KV pool is disabled when beam_max_width is None)"
        )
    from vllm_gr.v1.beam.attention_context import BeamAttentionContext

    model_config = runner.model_config
    parallel_config = runner.parallel_config

    return BeamAttentionContext(
        max_batch=max_batch,
        max_beams=int(gr_cfg.beam_max_width),
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        max_decode_steps=int(gr_cfg.beam_max_decode_steps),
        head_dim=model_config.get_head_size(),
        num_layers=model_config.get_num_layers(parallel_config),
        device=runner.device,
        dtype=model_config.dtype,
        enable_short_suffix_attention=(
            runner.device.type == "cuda" and gr_cfg.enable_short_suffix_attention
        ),
    )
