# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Prometheus beam search metrics (no GPU required)."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

try:
    import prometheus_client

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

import pytest

from vllm_gr.v1.metrics.beam_search_metrics import (
    _initialize_metrics,
    get_beam_search_metrics_text,
    record_beam_search_metrics,
)
from vllm_gr.v1.metrics.stats import RequestStateStats


@pytest.fixture(autouse=True)
def _reset_prometheus_state(monkeypatch):
    """Reset global Prometheus state before each test to avoid collisions."""
    if not PROMETHEUS_AVAILABLE:
        return

    # Unregister any previously-registered metrics so tests don't collide.
    collectors_to_remove = []
    for collector in list(prometheus_client.REGISTRY._collector_to_names.keys()):
        collectors_to_remove.append(collector)
    for collector in collectors_to_remove:
        try:
            prometheus_client.REGISTRY.unregister(collector)
        except KeyError:
            pass

    # Reset module-level state directly so _initialize_metrics runs fresh.
    import vllm_gr.v1.metrics.beam_search_metrics as _mod

    _mod._metrics_initialized = False
    _mod._beam_prefill_time_total = None
    _mod._beam_decode_time_total = None
    _mod._beam_sort_time_total = None
    _mod._beam_requests_total = None
    _mod._total_beams_generated = None
    _mod._beam_processed_tokens_total = None
    _mod._beam_prefill_time_histogram = None
    _mod._beam_decode_time_histogram = None
    _mod._beam_sort_time_histogram = None
    _mod._beams_per_request_histogram = None


class TestRecordBeamSearchMetrics:
    """Tests for record_beam_search_metrics()."""

    def test_record_metrics_with_request_state_stats(self):
        """Metrics from RequestStateStats are recorded correctly."""
        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        stats = RequestStateStats()
        stats.beam_search_prefill_time = 0.1
        stats.beam_search_decode_time = 0.5
        stats.beam_search_sort_time = 0.01
        stats.num_beams = 4
        stats.num_processed_tokens = 10

        record_beam_search_metrics(stats, engine="0", model_name="test_model")

        text = get_beam_search_metrics_text()
        assert "vllm_gr_beam_search_prefill_time_seconds_total" in text
        assert "vllm_gr_beam_search_decode_time_seconds_total" in text
        assert "vllm_gr_beam_search_sort_time_seconds_total" in text
        assert "vllm_gr_beam_search_requests_total" in text
        assert "vllm_gr_beam_search_total_beams_generated_total" in text
        assert "vllm_gr_beam_search_processed_tokens_total" in text
        # Counters should contain the recorded values.
        assert 'engine="0"' in text
        assert 'model_name="test_model"' in text

    def test_record_metrics_with_plain_namespace(self):
        """record_beam_search_metrics works with arbitrary objects bearing the right attrs."""
        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        metrics = SimpleNamespace(
            beam_search_prefill_time=0.2,
            beam_search_decode_time=1.0,
            beam_search_sort_time=0.05,
            num_beams=8,
        )

        record_beam_search_metrics(metrics, engine="1", model_name="ns_model")

        text = get_beam_search_metrics_text()
        assert 'engine="1"' in text
        assert 'model_name="ns_model"' in text
        assert "vllm_gr_beam_search_prefill_time_seconds_total" in text

    def test_record_metrics_none_metrics_does_not_crash(self):
        """Passing None for metrics should be a no-op."""
        record_beam_search_metrics(None, engine="0", model_name="test_model")

    def test_record_metrics_missing_attrs_does_not_crash(self):
        """Objects without beam_search attrs should not raise."""
        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        metrics = SimpleNamespace()
        record_beam_search_metrics(metrics, engine="0", model_name="no_attrs")

        text = get_beam_search_metrics_text()
        # Counters won't be incremented but request counter should still record.
        assert "vllm_gr_beam_search_requests_total" in text

    def test_record_metrics_zero_values_not_recorded(self):
        """Zero or negative metric values should be skipped for time counters."""
        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        stats = RequestStateStats()
        stats.beam_search_prefill_time = 0.0
        stats.beam_search_decode_time = -0.5
        stats.beam_search_sort_time = 0.0
        stats.num_beams = 0

        record_beam_search_metrics(stats, engine="0", model_name="zero_model")

        from vllm_gr.v1.metrics.beam_search_metrics import (
            _beam_decode_time_total,
            _beam_prefill_time_total,
            _beam_requests_total,
            _beam_sort_time_total,
            _total_beams_generated,
        )

        labels = {"engine": "0", "model_name": "zero_model"}
        # Zero/negative times should not increment counters.
        assert _beam_prefill_time_total.labels(**labels)._value.get() == 0.0
        assert _beam_decode_time_total.labels(**labels)._value.get() == 0.0
        assert _beam_sort_time_total.labels(**labels)._value.get() == 0.0
        # Zero beams should not increment request or beam counters.
        assert _beam_requests_total.labels(**labels)._value.get() == 0.0
        assert _total_beams_generated.labels(**labels)._value.get() == 0.0

    def test_record_metrics_cumulative_increment(self):
        """Calling record twice should accumulate counter values."""
        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        stats1 = RequestStateStats()
        stats1.beam_search_prefill_time = 0.1
        stats1.beam_search_decode_time = 0.2
        stats1.beam_search_sort_time = 0.01
        stats1.num_beams = 2
        stats1.num_processed_tokens = 10

        stats2 = RequestStateStats()
        stats2.beam_search_prefill_time = 0.3
        stats2.beam_search_decode_time = 0.4
        stats2.beam_search_sort_time = 0.02
        stats2.num_beams = 4
        stats2.num_processed_tokens = 20

        record_beam_search_metrics(stats1, engine="0", model_name="cum_model")
        record_beam_search_metrics(stats2, engine="0", model_name="cum_model")

        # Look up the counter values with the prometheus_client API directly.
        from vllm_gr.v1.metrics.beam_search_metrics import (
            _beam_prefill_time_total,
            _beam_processed_tokens_total,
            _beam_requests_total,
            _total_beams_generated,
        )

        labels = {"engine": "0", "model_name": "cum_model"}
        assert _beam_prefill_time_total.labels(**labels)._value.get() == pytest.approx(0.4)
        assert _beam_requests_total.labels(**labels)._value.get() == 2.0
        assert _total_beams_generated.labels(**labels)._value.get() == 6.0
        assert _beam_processed_tokens_total.labels(**labels)._value.get() == 30.0

    def test_record_metrics_different_labels(self):
        """Metrics should be kept separate per label set."""
        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        stats = RequestStateStats()
        stats.beam_search_prefill_time = 0.1
        stats.beam_search_decode_time = 0.2
        stats.beam_search_sort_time = 0.01
        stats.num_beams = 2

        record_beam_search_metrics(stats, engine="0", model_name="model_a")
        record_beam_search_metrics(stats, engine="1", model_name="model_b")

        from vllm_gr.v1.metrics.beam_search_metrics import (
            _beam_prefill_time_total,
        )

        val_a = _beam_prefill_time_total.labels(engine="0", model_name="model_a")._value.get()
        val_b = _beam_prefill_time_total.labels(engine="1", model_name="model_b")._value.get()
        assert val_a == pytest.approx(0.1)
        assert val_b == pytest.approx(0.1)


class TestGetBeamSearchMetricsText:
    """Tests for get_beam_search_metrics_text()."""

    def test_returns_text_after_recording(self):
        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        stats = RequestStateStats()
        stats.beam_search_prefill_time = 0.1
        stats.beam_search_decode_time = 0.2
        stats.beam_search_sort_time = 0.01
        stats.num_beams = 2

        record_beam_search_metrics(stats, engine="0", model_name="text_model")
        text = get_beam_search_metrics_text()

        assert isinstance(text, str)
        assert len(text) > 0
        assert "vllm_gr_beam_search" in text
        assert "HELP" in text or "TYPE" in text

    def test_returns_empty_when_prometheus_unavailable(self):
        """When prometheus_client is missing, get_beam_search_metrics_text returns ''."""
        with patch(
            "vllm_gr.v1.metrics.beam_search_metrics.PROMETHEUS_AVAILABLE",
            False,
        ):
            text = get_beam_search_metrics_text()
        assert text == ""


class TestInitializeMetrics:
    """Tests for _initialize_metrics()."""

    def test_initialize_metrics_idempotent(self):
        """Calling _initialize_metrics multiple times should be safe."""
        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        _initialize_metrics()
        _initialize_metrics()
        _initialize_metrics()

        text = get_beam_search_metrics_text()
        # No crash and metrics still accessible.
        assert isinstance(text, str)

    def test_initialize_metrics_without_prometheus_is_noop(self):
        """If prometheus_client is absent, _initialize_metrics should be a no-op."""
        with patch(
            "vllm_gr.v1.metrics.beam_search_metrics.PROMETHEUS_AVAILABLE",
            False,
        ):
            _initialize_metrics()

    def test_all_histogram_buckets_present(self):
        """Histograms should use the expected custom buckets."""
        if not PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        _initialize_metrics()

        from vllm_gr.v1.metrics.beam_search_metrics import (
            _beam_prefill_time_histogram,
            _beams_per_request_histogram,
        )

        assert _beam_prefill_time_histogram._upper_bounds == [
            0.01,
            0.05,
            0.1,
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            float("inf"),
        ]
        assert _beams_per_request_histogram._upper_bounds == [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            float("inf"),
        ]
