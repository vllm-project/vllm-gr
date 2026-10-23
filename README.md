# vllm-gr

<div align="center">
  <img src="docs/assets/vLLM-GR.png" alt="vllm-gr logo" width="60%">
</div>

vLLM-based Generative Recommender Inference Engine

This repository focuses on vLLM-based generative recommendation inference, including the OneRec beam-search / catalog-constrained generation pipeline, the Pangu LLM passthrough path, example scripts, test suites, and GPU / NPU runtime adaptation code.

## Current Scope

The current implementation targets inference scenarios and does not aim for full training-phase feature coverage.

## Repository Contents

- `vllm_gr/` — GRLLM (offline beam search), beam-search/catalog patches, NPU runtime utils, ops
- `examples/` — OneRec beam-search and Pangu LLM passthrough offline examples
- `tools/` — unified test runner and local development utilities
- `scripts/` — wheel-build and device lifecycle helpers used by CI
- `tests/` — unit, contract, and Pangu system tests
- `requirements/` — platform dependency lists (auto-selected by `setup.py` as `cuda` / `npu` / `cpu`)

## Installation

**Install vLLM 0.22.1 first (and vLLM-Ascend 0.22.1rc1 on NPU), then
`pip install vllm-gr`.** This release targets that API pair only. Platform
dependencies are auto-detected by `setup.py` — no `[gpu]` / `[npu]` extras
needed.

For detailed instructions, see [quickstart](docs/quickstart.md) or [quick start ascend](docs/quickstart-ascend.md).

## Unified self-validation

Recommended local validation command:

```bash
./tools/run_test.sh
```

The runner acquires an available accelerator and executes the repository's
pre-commit and test gates. See [Development Utils Manual](docs/tools/README.md)
for environment and model-cache options.

Refer to [Run benchmarks](docs/benchmarks/README.md) for benchmark-related test cases.
The standalone offline `beam_search_v1` performance and timeline benchmark is
documented in [benchmarks/qwen/README.md](benchmarks/qwen/README.md).

## Executing

- For the EngineCore-owned asynchronous Beam Search V1 path, start with the
  [OneRec V1 asynchronous scheduling quickstart](docs/onerec_v1_async_quickstart.md).
  It gives a beginner-friendly, end-to-end reproduction path; consult the
  [offline Beam Search examples](examples/offline_inference/beam_search/README.md#beam-search-v1)
  for the full configuration reference.
- Please refer to [offline inference](examples/offline_inference/beam_search/README.md) for offline beam search example.
- Please refer to [online inference](examples/online_serving/beam_search/README.md) for online inference example.

## NPU Serving with Constraint Tables

Use a canonical `--vllm-gr-config` file with `beam.execution_mode: v1` and
`constraint_table.backend: npu`. Build the `.safetensors` artifact with the
same model/tokenizer snapshot used by the server and include the printed
artifact and tokenizer digests in the configuration.

Follow the [artifact build instructions](examples/offline_inference/beam_search/README.md#constraint-table-artifact),
then the [online V1 example](examples/online_serving/beam_search/README.md).
The old frontend `--catalog-path` injection flow is retired; a triples JSON
is an artifact build input, not a replacement for the canonical configuration.

The [NPU quickstart](docs/quickstart-ascend.md) describes the Ascend environment
and execution choices. Benchmark gates additionally require local dataset and
catalog paths plus a resolved benchmark policy; see
[benchmark instructions](docs/benchmarks/README.md).

## Uninstall

```bash
pip uninstall vllm-gr
```
