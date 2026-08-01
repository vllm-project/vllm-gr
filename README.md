# vllm-gr

<div align="center">
  <img src="docs/assets/vLLM-GR.png" alt="vllm-gr logo" width="60%">
</div>

vLLM-based Generative Recommender Inference Engine

This repository focuses on vLLM-based generative recommendation inference, including the OneRec beam-search / catalog-constrained generation pipeline, the Pangu LLM passthrough path, example scripts, test suites, and CUDA runtime integration.

## Current Scope

The current implementation targets inference scenarios and does not aim for full training-phase feature coverage.

## Repository Contents

- `vllm_gr/` — GRLLM (offline beam search), beam-search/catalog patches, and CUDA runtime integration
- `examples/` — OneRec beam-search and Pangu LLM passthrough offline examples
- `tools/` — unified test runner and local development utilities
- `scripts/` — wheel-build and device lifecycle helpers used by CI
- `tests/` — unit, contract, and Pangu system tests
- `requirements/` — dependency lists selected by `setup.py` for CUDA or documentation builds

## Installation

**Install vLLM 0.22.1 first, then `pip install vllm-gr`.** This release targets
that API version. CUDA dependencies are auto-detected by `setup.py`.

For detailed instructions, see the [quickstart](docs/quickstart.md).

## Unified self-validation

Recommended local validation command:

```bash
./tools/run_test.sh
```

The runner acquires an available accelerator and executes the repository's
pre-commit and test gates. See [Development Utils Manual](docs/tools/README.md)
for environment and model-cache options.

refer to [Run benchmarks](docs/benchmarks/README.md) for benchmark related test cases.

## Executing

- Please refer to [offline inference](examples/offline_inference/beam_search/README.md) for offline beam search example.
- Please refer to [online inference](examples/online_serving/beam_search/README.md) for online inference example.

## Uninstall

```bash
pip uninstall vllm-gr
```
