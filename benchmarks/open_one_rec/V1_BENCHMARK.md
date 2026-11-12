# OneRec offline V1 cache-miss benchmark

The existing entry point supports V1 with `--beam-search`:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 USE_TF=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --model /absolute/path/to/OneRec-1.7B \
  --dataset-name onerec --dataset-path /absolute/path/to/RecIF \
  --task-types video --custom-input-len 1023 --num-prompts 100 \
  --n 128 --output-len 5 --beam-search --warmup-requests 10 \
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
- Timing starts before submitting the **text** prompt to `beam_search()`
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

Run outside both implementation checkouts. Put a common benchmark-only harness
first on `PYTHONPATH`, followed by the implementation checkout. The harness must
not contain `vllm_gr/`, which would shadow the implementation under comparison.
Verify the import source and implementation revision for each pass.

## Numerical comparisons

Vocabulary slicing changes projection shape and can change BF16 scores and Beam
ordering. TP greater than one retains full projection; quantized heads project
fully before cropping and therefore do not gain projection FLOPs from slicing.
Do not infer bitwise parity or business accuracy from a latency improvement.

For targeted projection comparisons, use
`tests/st/run_vocab_slice_v1_review.py` with `--mode eager|graph` and
`--projection full|slice`. Its `--strict-bf16` option is an isolated numerical
control, not a production default. Diagnostic GPU stage timing is separate from
the public end-to-end timing defined above; mocked TP routing is not multi-GPU
validation. Use the [accuracy gates](../../docs/usage/benchmarks/README.md) for business
acceptance.
