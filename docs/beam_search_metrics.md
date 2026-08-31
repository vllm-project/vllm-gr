# Beam Search Metrics

This document describes the new Prometheus metrics added to track beam search performance.

## Available Metrics

All metrics are exposed at the `/metrics` endpoint (Prometheus format) with labels:

- `engine`: Engine ID (usually "0")
- `model_name`: Model name (falls back to "unknown" if it cannot be determined)

### Counter Metrics (Cumulative Totals)

1. **`vllm_gr_beam_search_prefill_time_seconds_total`**
   - Cumulative prefill (first token) time across all beam search requests
   - Unit: seconds

2. **`vllm_gr_beam_search_decode_time_seconds_total`**
   - Cumulative decode time (generation phase) across all beam search requests
   - Unit: seconds

3. **`vllm_gr_beam_search_sort_time_seconds_total`**
   - Cumulative time spent sorting beams by log probability
   - Unit: seconds

4. **`vllm_gr_beam_search_requests_total`**
   - Total number of beam search requests processed

5. **`vllm_gr_beam_search_total_beams_generated_total`**
   - Total number of beams generated across all requests

6. **`vllm_gr_beam_search_processed_tokens_total`**
   - Total number of processed tokens across all beam search requests.
   - Each decode step contributes `active_beams * beam_width` candidate tokens
     (the count after catalog/beam-star pruning), so for vanilla beam search
     with beam width `BW` and `n` decode steps this equals
     `BW + (n-1) * BW^2`. Pruning constrainers reduce this value.

### Histogram Metrics (Distributions)

1. **`vllm_gr_beam_search_prefill_duration_seconds`**
   - Distribution of prefill times per request
   - Buckets: 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0 seconds

2. **`vllm_gr_beam_search_decode_duration_seconds`**
   - Distribution of decode times per request
   - Buckets: 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0 seconds

3. **`vllm_gr_beam_search_sort_duration_seconds`**
   - Distribution of beam sort times per request
   - Buckets: 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0 seconds

4. **`vllm_gr_beam_search_beams_per_request`**
   - Distribution of number of beams per request
   - Buckets: 1, 2, 4, 8, 16, 32, 64, 128 beams

## Querying Metrics

### Using curl to fetch all metrics

```bash
curl http://localhost:8000/metrics | grep beam_search
```

### Using a Prometheus-compatible monitoring stack

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Example queries with Prometheus

```promql
# Total prefill time across all requests
vllm_gr_beam_search_prefill_time_seconds_total

# Average prefill time per request
rate(vllm_gr_beam_search_prefill_time_seconds_total[5m]) / rate(vllm_gr_beam_search_requests_total[5m])

# 95th percentile of beam sort time
histogram_quantile(0.95, vllm_gr_beam_search_sort_duration_seconds_bucket)

# Number of beams per request
vllm_gr_beam_search_beams_per_request
```

## Implementation Details

The metrics are automatically collected when:

1. A beam search request completes on the server
2. The `RequestStateStats` object contains timing information
3. The `record_beam_search_metrics()` function is called from the OpenAI serving engine

The metrics are created using the Prometheus Python client library and are automatically exposed as part of the vLLM Prometheus metrics collection.
