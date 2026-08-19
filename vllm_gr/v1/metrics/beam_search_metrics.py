# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Beam search metrics collector for exposing metrics via Prometheus.

This module provides metrics collectors that extract beam search statistics
from RequestOutput.metrics and expose them via Prometheus.
"""

from __future__ import annotations

from typing import Any

try:
    from prometheus_client import REGISTRY, Counter, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Prometheus client not available
    Counter = None
    Histogram = None
    generate_latest = None
    REGISTRY = None
    PROMETHEUS_AVAILABLE = False


# Global metric objects - created once at module import
_metrics_initialized = False
_beam_prefill_time_total = None
_beam_decode_time_total = None
_beam_sort_time_total = None
_beam_requests_total = None
_total_beams_generated = None
_beam_prefill_time_histogram = None
_beam_decode_time_histogram = None
_beam_sort_time_histogram = None
_beams_per_request_histogram = None
_beam_output_requests_total = None
_beam_result_width_histogram = None
_beam_result_shortfall_total = None
_beam_result_reduction_requests_total = None


def _initialize_metrics() -> None:
    """Initialize all Prometheus metrics. Called once per process via lazy initialization."""
    global _metrics_initialized
    global _beam_prefill_time_total, _beam_decode_time_total, _beam_sort_time_total
    global _beam_requests_total, _total_beams_generated
    global _beam_prefill_time_histogram, _beam_decode_time_histogram
    global _beam_sort_time_histogram, _beams_per_request_histogram
    global _beam_output_requests_total, _beam_result_width_histogram
    global _beam_result_shortfall_total, _beam_result_reduction_requests_total

    if _metrics_initialized or not PROMETHEUS_AVAILABLE:
        return

    try:
        # Counters for cumulative totals
        _beam_prefill_time_total = Counter(
            "vllm_gr_beam_search_prefill_time_seconds_total",
            "Cumulative prefill time for beam search requests",
            labelnames=["engine", "model_name"],
        )
        _beam_decode_time_total = Counter(
            "vllm_gr_beam_search_decode_time_seconds_total",
            "Cumulative decode time for beam search requests",
            labelnames=["engine", "model_name"],
        )
        _beam_sort_time_total = Counter(
            "vllm_gr_beam_search_sort_time_seconds_total",
            "Cumulative beam sort operation time",
            labelnames=["engine", "model_name"],
        )
        _beam_requests_total = Counter(
            "vllm_gr_beam_search_requests_total",
            "Total number of beam search requests processed",
            labelnames=["engine", "model_name"],
        )
        _total_beams_generated = Counter(
            "vllm_gr_beam_search_total_beams_generated_total",
            "Total number of beams generated across all requests",
            labelnames=["engine", "model_name"],
        )

        # Histograms for distributions
        _beam_prefill_time_histogram = Histogram(
            "vllm_gr_beam_search_prefill_duration_seconds",
            "Prefill time distribution for beam search requests",
            labelnames=["engine", "model_name"],
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        _beam_decode_time_histogram = Histogram(
            "vllm_gr_beam_search_decode_duration_seconds",
            "Decode time distribution for beam search requests",
            labelnames=["engine", "model_name"],
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        _beam_sort_time_histogram = Histogram(
            "vllm_gr_beam_search_sort_duration_seconds",
            "Beam sort operation time distribution",
            labelnames=["engine", "model_name"],
            buckets=(0.0001, 0.001, 0.01, 0.1, 0.5, 1.0),
        )
        _beams_per_request_histogram = Histogram(
            "vllm_gr_beam_search_beams_per_request",
            "Distribution of number of beams per request",
            labelnames=["engine", "model_name"],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128),
        )
        _beam_output_requests_total = Counter(
            "vllm_gr_beam_output_requests_total",
            "Beam requests by low-cardinality final logprobs mode",
            labelnames=["engine", "model_name", "logprobs_mode"],
        )
        _beam_result_width_histogram = Histogram(
            "vllm_gr_beam_result_width",
            "Distribution of actual final Beam result counts",
            labelnames=["engine", "model_name"],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128),
        )
        _beam_result_shortfall_total = Counter(
            "vllm_gr_beam_result_shortfall_total",
            "Beam requests returning fewer results than requested",
            labelnames=["engine", "model_name"],
        )
        _beam_result_reduction_requests_total = Counter(
            "vllm_gr_beam_result_reduction_requests_total",
            "Beam requests whose requested result width is below beam width",
            labelnames=["engine", "model_name"],
        )
    except ValueError:
        # Metrics already registered in this registry (multiprocessing scenario).
        # Metrics are available in the registry, initialization is complete.
        pass
    finally:
        _metrics_initialized = True


def record_beam_search_metrics(
    metrics: Any,
    engine: str = "0",
    model_name: str = "unknown",
) -> None:
    """Record beam search metrics from a RequestOutput.metrics object.

    Args:
        metrics: RequestStateStats object with beam search metrics
        engine: Engine ID for Prometheus label
        model_name: Model name for Prometheus label
    """
    # Lazy initialization: ensure metrics are created in THIS process
    global _metrics_initialized
    if not _metrics_initialized:
        _initialize_metrics()

    if not PROMETHEUS_AVAILABLE or metrics is None:
        return

    # Early exit if initialization failed (metrics are None)
    if _beam_prefill_time_total is None:
        return

    labels = {"engine": engine, "model_name": model_name}

    try:
        prefill_time = getattr(metrics, "beam_search_prefill_time", 0.0)
        if prefill_time > 0:
            _beam_prefill_time_total.labels(**labels).inc(prefill_time)
            if _beam_prefill_time_histogram is not None:
                _beam_prefill_time_histogram.labels(**labels).observe(prefill_time)

        decode_time = getattr(metrics, "beam_search_decode_time", 0.0)
        if decode_time > 0 and _beam_decode_time_total is not None:
            _beam_decode_time_total.labels(**labels).inc(decode_time)
            if _beam_decode_time_histogram is not None:
                _beam_decode_time_histogram.labels(**labels).observe(decode_time)

        sort_time = getattr(metrics, "beam_search_sort_time", 0.0)
        if sort_time > 0 and _beam_sort_time_total is not None:
            _beam_sort_time_total.labels(**labels).inc(sort_time)
            if _beam_sort_time_histogram is not None:
                _beam_sort_time_histogram.labels(**labels).observe(sort_time)

        # Record beam count
        num_beams = getattr(metrics, "num_beams", 1)
        if num_beams > 0:
            if _beam_requests_total is not None:
                _beam_requests_total.labels(**labels).inc()
            if _total_beams_generated is not None:
                _total_beams_generated.labels(**labels).inc(num_beams)
            if _beams_per_request_histogram is not None:
                _beams_per_request_histogram.labels(**labels).observe(num_beams)
    except Exception:
        # Silently ignore errors - don't break main flow
        pass


def record_beam_output_metrics(
    output_spec: Any,
    actual_width: int,
    *,
    beam_width: int,
    engine: str = "0",
    model_name: str = "unknown",
) -> None:
    """Record the frozen output contract without request/token labels."""
    if not _metrics_initialized:
        _initialize_metrics()
    if not PROMETHEUS_AVAILABLE or _beam_output_requests_total is None:
        return
    labels = {"engine": engine, "model_name": model_name}
    try:
        _beam_output_requests_total.labels(**labels, logprobs_mode=output_spec.logprobs_mode).inc()
        if _beam_result_width_histogram is not None:
            _beam_result_width_histogram.labels(**labels).observe(actual_width)
        if actual_width < output_spec.result_width and _beam_result_shortfall_total is not None:
            _beam_result_shortfall_total.labels(**labels).inc()
        if (
            output_spec.result_width < beam_width
            and _beam_result_reduction_requests_total is not None
        ):
            _beam_result_reduction_requests_total.labels(**labels).inc()
    except Exception:
        pass


# Metrics will be initialized lazily on first use via record_beam_search_metrics()
# instead of at module import time to handle multiprocessing correctly


def get_beam_search_metrics_text() -> str:
    """Get all beam search metrics in Prometheus text format.

    Returns:
        String containing all beam search metrics in Prometheus format,
        or empty string if Prometheus is unavailable.
    """
    if not PROMETHEUS_AVAILABLE or generate_latest is None:
        return ""

    try:
        metrics_text = generate_latest(REGISTRY).decode("utf-8")
        return metrics_text
    except Exception:
        return ""
