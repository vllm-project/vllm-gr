# Troubleshoot OneRec benchmarks

[Benchmark overview](README.md)

## Dataset not found or unreadable

Run from the repository root and make sure `--dataset-path` points to the
directory above the task directories. For the `video` task, check that the file
is available and that Git LFS downloaded the contents:

```bash
ls -l ./data/video/video_test.parquet
git -C ./data lfs pull
python -c 'import pyarrow.parquet as pq; print(pq.read_schema("data/video/video_test.parquet"))'
```

If the loader skips rows, check the [Parquet schema](dataset.md#parquet-schema)
and the task's expected answer format.

## Model or tokenizer cannot load

Verify that the process can read the model snapshot. Use the same tokenizer
snapshot for the benchmark client, artifact compilation, and inference. If
the server uses an alias, use its served model name for the HTTP request and
follow the client-specific instructions in the [throughput](throughput.md)
or [accuracy](accuracy.md) guide.

## Device out of memory

Try a lower request concurrency for HTTP, or a smaller `--batch-size` for
offline accuracy. If you reduce the beam width, keep the client width and
the engine configuration compatible. The constrained V1 output budget
includes the SID markers; keep it consistent with the task and configuration.
Record changed settings before comparing results.

## Beam requests fail or produce unexpected results

- Start with one request from the [online Beam example](../../examples/online_serving/beam_search/README.md#send-a-request)
  before running the HTTP load generator.
- Check that the canonical V1 config matches the device, artifact, and
  tokenizer. See the [V1 quickstart troubleshooting table](../onerec_v1_async_quickstart.md#5-validate-and-troubleshoot).
- Confirm the host, port, served model name, endpoint, and API credentials.
  Both HTTP benchmark clients in these guides use `/v1/chat/completions`.
- Read the server logs and failed request count before interpreting metrics.
  The accuracy runner also accepts `--debug` for evaluator diagnostics.
