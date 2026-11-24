# Examples

Choose an offline example for a Python workload, or an online example for
a client that sends requests to a running server. Start with the
[quickstart](../getting_started/quickstart.md) to run OneRec for the first
time. The repository's `examples/` directory contains the runnable scripts.

## Offline inference

- [Run OneRec-1.7B on CUDA or Ascend](../../examples/README.md#run-onerec-offline) and inspect the returned candidates.
- [Run with a constraint table](../../examples/offline_inference/beam_search/README.md#run-with-a-constraint-table) to generate candidates from a catalog.
- [Beam search on CUDA](../user_guide/beam_search_cuda.md) covers full-vocabulary and catalog-constrained search.
- [Full-vocabulary Beam search on Ascend](../user_guide/beam_search_npu.md) covers the Python API and supported execution modes.
- [Prepare public openPangu weights for a SID smoke test](pangu_sid.md).

## Online serving

- [Serve OneRec over HTTP](../../examples/online_serving/beam_search/README.md) and inspect a completions response.
- [Serve OneRec over gRPC](../../examples/online_serving/grpc/README.md) and send token inputs.
- [Prepare a catalog and select a Trie](../../examples/online_serving/catalog/README.md) for a serving request.
- [Run sidecar updates](../../examples/README.md#run-sidecar-hot-updates) to publish new weights or switch catalogs on running instances.
