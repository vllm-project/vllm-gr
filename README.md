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

refer to [Run benchmarks](docs/benchmarks/README.md) for benchmark related test cases.

## Executing

- For the EngineCore-owned asynchronous Beam Search V1 path, start with the
  [OneRec V1 asynchronous scheduling quickstart](docs/onerec_v1_async_quickstart.md).
  It gives a beginner-friendly, end-to-end reproduction path; consult the
  [offline Beam Search examples](examples/offline_inference/beam_search/README.md#beam-search-v1)
  for the full configuration reference.
- Please refer to [offline inference](examples/offline_inference/beam_search/README.md) for offline beam search example.
- Please refer to [online inference](examples/online_serving/beam_search/README.md) for online inference example.

## NPU Serving with Constraint Tables

When serving on NPU with beam search enabled (e.g. `--beam-max-width` +
`--beam-worker-decision`), the platform auto-promotes the constraint backend
to Constraint Tables. In that mode you **must** pass
`--catalog-path <triples_document>`, where the document is produced by
`tools/build_constraint_triples.py` from the real catalog (e.g.
`video_constraint_triples.json`).

Without it, the tables are auto-built from the tokenizer's `<s_a_*>` /
`<s_b_*>` / `<s_c_*>` groups; a model whose groups exceed `max_valid_items`
(currently 500) fails fast at startup:

```text
ValueError: tokenizer constraint tables would be truncated: item groups
s_a/s_b/s_c have 8192/8192/8192 items but max_valid_items=500. ... Provide
--constraint-data built from the real catalog via tools/build_constraint_triples.py,
or pass valid_item_ids/prefix files covering the full universe.
```

The triples document is generated from the real recommendation catalog:

```bash
python tools/build_constraint_triples.py \
  --catalog /workspace/ckpt/OpenOneRec-RecIF/generated_data/video_catalog.json \
  --output /workspace/ckpt/OpenOneRec-RecIF/generated_data/video_constraint_triples.json \
  --begin "<|sid_begin|>" \
  --end "<|sid_end|>"
```

The catalog itself is converted from the OpenOneRec dataset's `sid2pid.json`
(= list of acceptable SIDs), see
[examples/online_serving/catalog/README.md](examples/online_serving/catalog/README.md):

```bash
python benchmarks/open_one_rec/tools/convert_catalog.py \
  --dataset-path /workspace/ckpt/OpenOneRec-RecIF/benchmark_data \
  --output-path /workspace/ckpt/OpenOneRec-RecIF/generated_data/video_catalog.json
```

`tests/system/run_benchmark_and_verify.sh` builds it on demand when the file
is missing (from `TRIPLES_PATH` or the catalog directory).

Example:

```bash
vllm-gr serve $REC_MODEL \
  --port 9312 \
  --trust-remote-code \
  --max-logprobs 1024 \
  --beam-max-width 128 \
  --beam-worker-decision \
  --catalog-path $REC_TRIPLES \  # video_constraint_triples.json
  --attention-backend CUSTOM
```

See `server.sh` for a complete working NPU serve script.

## Uninstall

```bash
pip uninstall vllm-gr
```
