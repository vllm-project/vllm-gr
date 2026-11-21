# On-demand model and data preparation

Use `ensure_model(snapshot_spec("onerec1b7"))` or `ensure_dataset` from
`tests.helpers.artifacts` inside fixtures. The registry pins reviewed Hub commits.
Set `HF_TOKEN` for gated downloads and `VLLM_GR_TEST_CACHE` to change the cache.
Pass `local_override` for a validated local input. The managed cache checks
hashes, tokenizer and weight files, and Parquet metadata under a per-key lock.
For `ensure_derived` include source files, parameters and builder version.
Use `artifact_identity` to record local input hashes.
