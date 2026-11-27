# Examples

Choose an offline example for a Python workload, or an online example for
a client that sends requests to a running server. Start with the
[quickstart](../getting_started/quickstart.md) to run OneRec for the first
time. The repository's `examples/` directory contains the runnable scripts.

## Offline inference

- [Run OneRec-1.7B on CUDA or Ascend](../../examples/README.md#run-onerec-offline) and inspect the returned candidates.
- [Run with a constraint table](../../examples/offline_inference/beam_search/README.md#run-with-a-constraint-table) to generate candidates from a catalog.
- [OneRec asynchronous Beam search on CUDA](../user_guide/onerec_quickstart.md) walks through a catalog-constrained run with ModelRunner V1.
- [Sample OneRec SIDs with replacement on CUDA](../../examples/offline_inference/iid_sampling.py)
  using a compiled constraint table and ModelRunner V1.

  ```bash
  VLLM_USE_V2_MODEL_RUNNER=0 python examples/offline_inference/iid_sampling.py \
      --model "$ONEREC_MODEL" --table "$ONEREC_TABLE" --num-samples 128
  ```

  The example uses the sampling settings in the model checkpoint. For OneRec-1.7B
  these are temperature 0.6, top-k 20 and top-p 0.95. Pass a single tokenized
  user history to `GRLLM.sample_sids` without adding the next SID begin marker.
  One GR session runs prefill and both decode stages. Each stage groups draws
  with the same prefix, scores one row per group and draws independently from
  that prefix's filtered distribution. The CUDA worker shares prompt KV and
  copies each selected parent's suffix KV to its distinct children. The result
  contains SID multiplicities and the number of distinct prefixes per stage.
  For up to eight users, call `engine.sample_sids_batch(tokenized_histories,
  num_samples=128, temperature=0.6, top_k=20, top_p=0.95, seed=42)`.
  Results and optional `trie_ids` follow input order. A seed applies to the
  whole ordered batch. Each user samples from its own prefix distribution.
  Batched attention pads to the largest live prefix count among the users.
- [Full-vocabulary Beam search on Ascend](../user_guide/beam_search_npu.md) covers the Python API and supported execution modes.

## Online serving

- [Serve OneRec over HTTP](../../examples/online_serving/beam_search/README.md) and inspect a completions response.
- [Serve OneRec over gRPC](../../examples/online_serving/grpc/README.md) and send token inputs.
- [Prepare a catalog and select a Trie](../../examples/online_serving/catalog/README.md) for a serving request.
- [Run sidecar updates](../../examples/README.md#run-sidecar-hot-updates) to publish new weights or switch catalogs on running instances.

## Profiling IID sampling

The [RecIF Video benchmark](../../benchmarks/open_one_rec/profile_iid_sampling.py)
compares shared and unmerged draws in one eager CUDA V1 engine. It times the
interval from completed prefill to delivery of all SID counts, including both
decode forwards, sampling and scheduler overhead. It also records whole-call
wall times with the diagnostic synchronization enabled. Use the model revision,
test data and full Video catalog in the [OneRec reference](../../benchmarks/beam_search/recall_baseline_batch4.json).

```bash
VLLM_USE_V2_MODEL_RUNNER=0 python -m benchmarks.open_one_rec.profile_iid_sampling \
    --model "$ONEREC_MODEL" --dataset-path "$RECIF_BENCHMARK_DATA" \
    --table "$ONEREC_TABLE" --batch-size 8 --cache hot --output iid-profile.json
```
