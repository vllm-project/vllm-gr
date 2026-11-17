# vllm-gr

<div align="center">
  <img src="docs/assets/vLLM-GR.png" alt="vllm-gr logo" width="60%">
</div>

vLLM-based Generative Recommender Inference Engine

This repository focuses on vLLM-based generative recommendation inference, including the OneRec beam-search / catalog-constrained generation pipeline, the Pangu LLM passthrough path, example scripts, test suites, and GPU / NPU runtime adaptation code.

## Runtime version contract

**vLLM-GR 0.30.0 supports vLLM 0.30.0 only.** GR patches depend on vLLM
internals, and runtime initialization rejects a different public vLLM version.
Local build suffixes do not change that compatibility requirement.

| Target | Required runtime | Installation source |
| --- | --- | --- |
| CUDA | vLLM 0.30.0 | Pinned by `requirements/cuda.txt` |
| Ascend NPU | vLLM 0.30.0, vLLM-Ascend 0.30.0rc2, Triton-Ascend 3.6.0 | Prepared vendor image; see the [Ascend guide](docs/quickstart-ascend.md) |

The `+npu` and `+cpu` package suffixes identify build targets. A CPU package
does not provide a production Beam inference runtime. See the
[dependency and version policy](docs/dependency_version_policy.md) for release
tags, build overrides, and dependency constraints.

## Current Scope

The current implementation targets inference scenarios and does not aim for full training-phase feature coverage.

## Repository Contents

- `vllm_gr/` — GRLLM (offline beam search), beam-search/catalog patches, NPU runtime utils, ops
- [`examples/`](examples/README.md) — HTTP serving, catalog constraints, and offline inference
- `tools/` — unified test runner and local development utilities
- `scripts/` — wheel-build and device lifecycle helpers used by CI
- `tests/` — unit, contract, and Pangu system tests
- `requirements/` — platform dependency lists (auto-selected by `setup.py` as `cuda` / `npu` / `cpu`)

## Installation

Use the [CUDA quickstart](docs/quickstart.md) or the
[Ascend installation guide](docs/quickstart-ascend.md) to prepare the matching
runtime. Set `VLLM_GR_TARGET_DEVICE` explicitly when installing from source or
building wheels; `setup.py` selects the platform dependencies. No `[gpu]` extra
is needed.

For service security boundaries and vulnerability reporting, see the
[security policy](SECURITY.md) and the upstream
[vLLM security guide](https://docs.vllm.ai/en/latest/usage/security.html).

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

- [Run OneRec offline](examples/README.md#run-onerec-offline) with a local
  model snapshot.
- [Constrain SIDs to a catalog](examples/offline_inference/beam_search/README.md#run-with-a-constraint-table).
  See the [asynchronous scheduling quickstart](docs/onerec_v1_async_quickstart.md)
  for the full configuration.
- [Serve OneRec over HTTP](examples/online_serving/beam_search/README.md) or
  [gRPC](examples/online_serving/grpc/README.md).

## NPU Serving

The following is the serving configuration contract for the validated NPU
stack (vLLM 0.30.0 with vLLM-Ascend 0.30.0rc2).

The driver loads [default_config.yaml](vllm_gr/config/default_config.yaml) automatically. Set
`constraint_table.backend="npu"` in the Python `gr`
dictionary or its `--gr-config` JSON equivalent. The separate `--gr` flag
selects GR mode; for example, `vllm serve MODEL --gr --gr-config '{"result_width":2}'`.
Build the `.safetensors` artifact with the
same model/tokenizer snapshot used by the server and include the printed
artifact and tokenizer digests in the configuration.

Use the [HTTP example](examples/online_serving/beam_search/README.md) to
compile the table and verify the deployment.
The old frontend `--catalog-path` injection flow is retired; a triples JSON
is an artifact build input, not a replacement for the canonical configuration.

The [NPU installation guide](docs/quickstart-ascend.md) records the runtime
requirements and execution choices. Benchmark gates additionally require local dataset and
catalog paths plus a resolved benchmark policy; see
[benchmark instructions](docs/benchmarks/README.md).

For catalog-free Beam selection over the full vocabulary or opt-in SA/SB/SC
token families on this stack, see
[unconstrained Beam Search V1 on NPU](docs/npu_unconstrained_beam.md).

## Engine configuration

Each driver process loads the [common defaults](vllm_gr/config/default_config.yaml)
once. Each engine receives its own resolved settings, including its workers.
Pass native vLLM options by their usual names and plugin options under `gr`.
Only specify the values that differ for this launch.

```python
from vllm_gr import GRLLM

llm = GRLLM(
    model="/models/my-model",
    gr={"beam_width": [8, 8, 4], "candidate_width": [8, 4, 2]},
)
```

Serving defaults to unconstrained Beam search (`gr.constraint_mode: none`).
For table constraints, follow the build command in the common defaults, then set
`gr.constraint_mode: table` and supply the generated artifact and digests.
Explicit flags override `--config` YAML settings, which override the packaged
defaults. A user-supplied YAML file can import a base file with
`import: ./base.yaml`. Mappings merge by key and lists replace the inherited
list. Relative table paths in YAML resolve against the declaring file.
See [Configuration](docs/configuration.md) for the shared file format and startup examples.

## Uninstall

```bash
pip uninstall vllm-gr
```
