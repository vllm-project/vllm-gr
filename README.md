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
- `scripts/` — unified test runner
- `tests/` — unit, contract, and Pangu system tests
- `requirements/` — platform dependency lists (auto-selected by `setup.py` as `cuda` / `npu` / `cpu`)

## Installation

**Install vLLM 0.22.1rc1 first (and vLLM-Ascend 0.22.1rc1 on NPU), then
`pip install vllm-gr`.** This release targets that API pair only. Platform
dependencies are auto-detected by `setup.py` — no `[gpu]` / `[npu]` extras
needed.

For detailed instructions, see [quickstart](docs/quickstart.md) or [quick start ascend](docs/quickstart-ascend.md).

## Unified self-validation

Recommended command (this is subset of what PR gate runs):

```bash
tools/run_tests.sh
```

refer to [Run benchmarks](docs/benchmarks/README.md) for benchmark related test cases.

For accuracy and performance validation of the vLLM 0.14.1 to 0.22.1rc1
upgrade across L20, A100, and Ascend hardware, see the
[upgrade validation guide](docs/upgrade_validation_guide.md).

## Executing

- Please refer to [offline inference](examples/offline_inference/beam_search/README.md) for offline beam search example.
- Please refer to [online inference](examples/online_serving/beam_search/README.md) for online inference example.

## Uninstall

```bash
pip uninstall vllm-gr
```
