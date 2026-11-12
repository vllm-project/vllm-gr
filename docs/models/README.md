# Models and workflows

Choose the documented workflow for your model. Runtime integration, model
weights, tokenizer identity and constraint artifacts must agree; a model name
alone does not establish backend or feature support.

| Model family | Documented workload | Start here | Important limit |
| --- | --- | --- | --- |
| OneRec | Generative recommendation with ranked SID outputs | [First inference](../getting_started/first-inference.md), [catalog quickstart](../getting_started/quickstart.md) | Catalog tables use three SID components and matching tokenizer digests |
| Pangu Embedded 1B | Native sampling, evaluation, serving and an explicit SID expansion workflow | [Pangu guide](../contributing/tools/pangu1b.md) | Newly added SID weight rows are untrained; conversion does not prove recommendation quality |
| Qwen | Offline constrained Beam benchmarking | [Qwen benchmark](https://github.com/JiusiServe/vllm-gr/blob/main/benchmarks/qwen/README.md) | Synthetic constraints establish execution measurements, not a production recommendation catalog |

## Prepare a reproducible snapshot

Keep model weights, configuration and tokenizer from one revision. Compile
constraint artifacts with that same tokenizer. Rebuild artifacts after a
tokenizer change; renaming or copying a table does not change its identity.

Use local model paths in deployment configuration and keep machine-specific
locations out of shared examples. See the [version policy](../configuration/dependency_version_policy.md)
for runtime requirements and [Configuration](../configuration/README.md) for
startup snapshots and relative paths.

## Validate the workload you will deploy

A successful model load establishes neither catalog accuracy nor graph replay.
Follow the selected example, inspect real outputs, then run the
[quality or performance measurement](../usage/README.md) needed for your use case.
Consult the [feature guides](../features/README.md) before enabling backend-specific
optimizations or changing batch geometry.
