# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared GR configuration, context construction and graph diagnostics."""

from typing import Any

from vllm_gr.platforms.graph import existing_graph, native_graph_counters


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


def _beam_graph_enabled_for_run(runner: Any) -> bool:
    """Beam graphs are only used when the config enables them (GRConfig)."""
    graph = existing_graph(runner)
    if graph is not None:
        return bool(graph.decode_enabled)
    return not getattr(getattr(runner, "model_config", None), "enforce_eager", False) and bool(
        _get_gr_config(runner).beam_graph_enabled
    )


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
    graph = existing_graph(runner)
    service_stats = graph.stats() if graph is not None else None
    fields = service_stats.get("runtime_fields") if service_stats is not None else None
    if isinstance(fields, dict):
        stats = {"captured_widths": [], "full_dispatches": 0, "gr_prefill_dispatches": {}, **fields}
    else:
        stats = native_graph_counters(runner)
    if service_stats is not None:
        stats["graph_service"] = service_stats
    stats["runner"] = type(runner).__name__
    stats["beam_graph_enabled"] = bool(config.beam_graph_enabled)
    stats["beam_max_width"] = config.beam_max_width
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
            executor.supports_short_suffix_device(runner.device)
            and gr_cfg.enable_short_suffix_attention
        ),
    )
