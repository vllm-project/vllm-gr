# OneRec offline V1 cache-miss benchmark

For historical projection and numerical diagnostics, see the
[vocabulary slicing validation](VOCAB_SLICE_REVIEW.md).

The existing entry point supports V1 with `--beam-search-v1`:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 USE_TF=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --model /absolute/path/to/OneRec-1.7B \
  --dataset-name onerec --dataset-path /absolute/path/to/RecIF \
  --task-types video --custom-input-len 1023 --num-prompts 100 \
  --n 128 --output-len 5 --beam-search-v1 --warmup-requests 10 \
  --constraint-triples /absolute/path/to/video_constraint_triples.json \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --output-json /absolute/path/to/result.json
```

This reuses `OneRecDataset`, chat-template processing and model tokenizer.
The data directory must contain `video/video_test.parquet` in the existing
OneRec messages/metadata format. The constraint JSON is compiled with the
same model tokenizer into a canonical CUDA artifact.

- `--custom-input-len 1023` sets the text prompt length before V1 appends
  SID begin; the effective model input is **1024** tokens.
- `--output-len 5` includes SID begin/end, leaving **3 generated SID tokens**.
- The runner loads one model, runs **10 warmups**, then **100 measured
  single-request calls**. Samples are not silently duplicated if data is short.
- Prefix caching stays enabled. Before every call, reset it and verify success
  **outside** the timer. Decode still reuses the current request's KV cache.
- Timing starts before submitting the **text** prompt to `beam_search_v1()`
  and ends after accessing **every beam's text**, forcing lazy output
  materialization. Input encoding and output decoding are included.
- Dataset loading, chat-template construction/length normalization, artifact
  compilation, model startup, warmup, cache reset and result serialization
  are excluded. This is offline text-to-text E2E, without HTTP/network overhead.
- Engine settings: CUSTOM attention, V1 async execution, beam graphs enabled,
  chunked prefill disabled, prefix caching enabled, beam-width sequence capacity,
  memory utilization 0.9, seed 0. Model length and batched-token capacity follow
  the supplied options above.
- Report p50/p90/p99 using linear interpolation at `(n-1)*q`.
  Requests/s is measured requests divided by the sum of timed E2E durations;
  it excludes between-request cache-reset and inspection time.

The JSON records implementation commit, actual input lengths, workload and
constraint/tokenizer digests, raw latencies, candidate IDs and scores.
Compare these digests and outputs in addition to latency.

For A/B, use the **same benchmark code** against separate implementation
worktrees, sharing model files, dataset and constraints. Run A-B-B-A sequentially,
with each pass independently initialized and warmed up. Do not install/switch
packages between passes or run concurrent GPU workloads.

Run from outside both implementation checkouts. Put a common benchmark-only
harness first on `PYTHONPATH`, followed by the implementation checkout under
test. The harness must not contain `vllm_gr/`; a full repository checkout here
would shadow the implementation being compared. Verify `vllm_gr.__file__`
and the implementation revision before each pass.

## Measured A/B result (2026-09-18)

> Measured on NVIDIA L20 (48 GB) / 2026-09-18 / runtime versions not recorded.
> Not a performance guarantee.

Both groups used the same [OneRec-1.7B model snapshot](https://huggingface.co/OpenOneRec/OneRec-1.7B/tree/cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b)
in BF16 with tensor parallelism set to 1 and the same benchmark adapter.
Group A used [full-vocabulary projection](https://github.com/JiusiServe/vllm-gr/commit/59299263902e27bf184bf75b2c3847c49b053778).
Group B used [vocabulary slicing for CUDA Constraint Tables](https://github.com/JiusiServe/vllm-gr/commit/7b0f04c21aa06b3f0e0a55a1c59dc8cfd4f17962),
which projects only the token range needed by the table.

**Synthetic data**, not the gated RecIF video dataset (access unavailable).
100 messages/metadata Parquet rows are loaded via the existing OneRec loader,
chat template and tokenizer. Python Random(0) samples 250 video constraint
triples without replacement per row, wrapping each triple with SID begin/end.
The user message is `Video history: {history}. Recommend the next video.`.
The existing loader normalizes each formatted prompt to 1023 tokens; V1
appends SID begin. The metadata answer is synthetic and not used for accuracy scoring.

Execution order A1-B1-B2-A2; each pass independently warms up 10 requests and
measures 100 cache misses, SID=3, beam width=128, effective input=1024.
All four passes have matching input lengths and workload/tokenizer/artifact digests.

| Pass | p50 ms | p90 ms | p99 ms | Timed requests/s |
| --- | ---: | ---: | ---: | ---: |
| A1 | 64.921 | 70.449 | 75.497 | 15.205 |
| B1 | 60.476 | 67.141 | 72.461 | 16.220 |
| B2 | 61.215 | 63.616 | 65.414 | 16.304 |
| A2 | 66.003 | 71.695 | 75.257 | 14.987 |

A1/B1 and A2/B2 respectively:
p50 reduction **6.85% / 7.25%**, p90 **4.70% / 11.27%**, p99 **4.02% / 13.08%**.
Timed requests/s increases **6.68% / 8.79%**.
Tail latency varies between runs; 100 samples do not establish a stable p99 gain.

### Output equivalence is not satisfied

A1/A2 outputs and scores are exactly equal on all 100 requests, as are B1/B2.
However, in both A/B pairs:

- Top-1 differs for **5/100 requests**.
- Mean candidate overlap is **121.44/128 (94.875%)**, range 116–127.
- No request has the exact same full candidate set.
- Maximum shared-candidate score difference is **0.2529010772705078**.

This was reproducible across restarts. The
[vocabulary slicing diagnostics](VOCAB_SLICE_REVIEW.md) localized the
discrepancy to BF16 projection reduction. The [original PR evidence](https://github.com/JiusiServe/vllm-gr/pull/405)
records the historical comparison.
