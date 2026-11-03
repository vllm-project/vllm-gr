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

The current CUDA release candidate targets **vLLM 0.30.0**. A CUDA install of
`vllm-gr` resolves that exact vLLM version through `requirements/cuda.txt`;
platform dependencies are selected by `setup.py`, so no `[gpu]` extra is
needed.

The Ascend path targets vLLM 0.30.0 with vLLM-Ascend 0.30.0rc2 and completed
NPU runtime validation on that vendor stack. The environment guard pins the
validated vLLM-Ascend/Triton-Ascend pair; the vendor runtime comes from the
base image. See the [Ascend installation guide](docs/quickstart-ascend.md) for
the installation procedure, runtime requirements, and execution contract.

For the validated CUDA installation path, see [quickstart](docs/quickstart.md).

## Unified self-validation

For code changes, start with [Contributing](CONTRIBUTING.md) and the
[development navigation](docs/development.md). Use the
[validation guide](docs/tools/README.md#validation-levels) to select checks.
For full local runtime validation in a prepared accelerator environment:

```bash
./tools/run_test.sh
```

The runner selects an available accelerator and runs the platform-specific
validation path. See [Development utilities](docs/tools/README.md) for CUDA/NPU
behavior, environment prerequisites, and model-cache options.

Refer to [Run benchmarks](docs/benchmarks/README.md) for benchmark-related test cases.
The standalone offline `beam_search_v1` performance and timeline benchmark is
documented in [benchmarks/qwen/README.md](benchmarks/qwen/README.md).

Shared AI coding instructions are in [AGENTS.md](AGENTS.md), imported by
[CLAUDE.md](CLAUDE.md). See [AI tool integration](docs/agent_instructions.md)
for loading checks and guidance maintenance.

## Executing

- For the EngineCore-owned asynchronous Beam Search V1 path, start with the
  [OneRec V1 asynchronous scheduling quickstart](docs/onerec_v1_async_quickstart.md).
  It gives a beginner-friendly, end-to-end reproduction path; consult the
  [offline Beam Search examples](examples/offline_inference/beam_search/README.md#beam-search-v1)
  for the full configuration reference.
- Please refer to [offline inference](examples/offline_inference/beam_search/README.md) for offline beam search example.
- Please refer to [online inference](examples/online_serving/beam_search/README.md) for online inference example.
- For gRPC beam generation with binary token inputs, see the
  [gRPC and NumPy upgrade guide](docs/grpc_numpy_upgrade.md).

## NPU Serving

The following is the serving configuration contract for the validated NPU
stack (vLLM 0.30.0 with vLLM-Ascend 0.30.0rc2).

Use a canonical `--vllm-gr-config` file with `beam.execution_mode: v1` and
`constraint_table.backend: npu`. Build the `.safetensors` artifact with the
same model/tokenizer snapshot used by the server and include the printed
artifact and tokenizer digests in the configuration.

Follow the [artifact build instructions](examples/offline_inference/beam_search/README.md#constraint-table-artifact),
then the [online V1 example](examples/online_serving/beam_search/README.md).
The old frontend `--catalog-path` injection flow is retired; a triples JSON
is an artifact build input, not a replacement for the canonical configuration.

The [NPU installation guide](docs/quickstart-ascend.md) records the runtime
requirements and execution choices. Benchmark gates additionally require local dataset and
catalog paths plus a resolved benchmark policy; see
[benchmark instructions](docs/benchmarks/README.md).

## Uninstall

```bash
pip uninstall vllm-gr
```
