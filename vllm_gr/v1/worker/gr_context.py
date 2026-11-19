# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared GR configuration, context construction and graph diagnostics."""

from typing import Any, cast


def _get_gr_config(runner: Any) -> Any:
    """Return the worker's GRConfig, cached on the runner (frozen dataclass).

    Configuration flows exclusively through the engine config chain
    (CLI flags / additional_config -> GRConfig -> VllmConfig.additional_config);
    no environment variables are consulted at runtime.
    """
    cached = getattr(runner, "_vllm_gr_config", None)
    if cached is not None:
        return cached
    from vllm_gr.config import GRConfig

    cfg = GRConfig.from_vllm_config(getattr(runner, "vllm_config", None))
    runner._vllm_gr_config = cfg
    return cfg


def _beam_full_graph_ready(runner: Any, num_tokens: int) -> bool:
    """Query the GPU Beam component; retain the device-neutral NPU key contract."""
    graph = getattr(runner, "_beam_decode_graph", None)
    if graph is not None:
        return bool(graph.ready and num_tokens == graph.signature.num_model_rows)
    beam_keys = getattr(runner, "_beam_graph_keys", None)
    return beam_keys is not None and num_tokens in beam_keys


def _beam_graph_enabled_for_run(runner: Any) -> bool:
    """Beam graphs are only used when the config enables them (GRConfig)."""
    return cast(bool, _get_gr_config(runner).beam_graph_enabled)


def get_beam_graph_runtime_stats(worker: Any) -> dict[str, Any]:
    """Return capture/dispatch state through ``collective_rpc``.

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
    return stats


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
    from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext

    model_config = runner.model_config
    parallel_config = runner.parallel_config

    from vllm_gr.platforms.attention import for_runner

    executor = for_runner(runner)
    return BeamAttentionContext(
        attention=executor,
        max_batch=max_batch,
        max_beams=int(gr_cfg.beam_max_width),
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        max_decode_steps=int(gr_cfg.beam_max_decode_steps),
        head_dim=model_config.get_head_size(),
        num_layers=model_config.get_num_layers(parallel_config),
        device=runner.device,
        dtype=model_config.dtype,
        enable_short_suffix_attention=(
            executor.supports_short_suffix_attention() and gr_cfg.enable_short_suffix_attention
        ),
    )
