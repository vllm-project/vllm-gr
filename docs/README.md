---
hide:
  - navigation
  - toc
---

# Generative recommendation with vLLM-GR

Turn a user's context into ranked semantic IDs with native vLLM model execution,
GPU or Ascend acceleration, and explicit Beam resource ownership.

[Get started](getting_started/README.md) | [Explore the architecture](design/main-arch.md) | [Browse examples](examples/README.md)

[![vLLM-GR interfaces, EngineCore ownership and CUDA or Ascend Worker execution](assets/architecture.png)](assets/architecture.png)

## Build your first application

<div class="gr-cards" markdown="1">

- **[Run your first inference](getting_started/first-inference.md)**

    Prepare a supported runtime, load OneRec, and inspect the returned beams.

- **[Serve recommendation requests](serving/README.md)**

    Choose HTTP or gRPC, configure startup capacity, and verify SID outputs.

- **[Generate within a catalog](features/constraint-tables.md)**

    Compile a tokenizer-bound constraint artifact and select a catalog root.

- **[Tune execution](features/README.md)**

    Understand candidate widths, graph replay, Attention and resource updates.

</div>

## What vLLM-GR adds

vLLM-GR extends an installed vLLM runtime for generative recommendation.
Native vLLM supplies model loading, scheduling and paged KV allocation. GR
adds atomic Beam batches, persistent device state, constrained candidate
selection, specialized Attention and resource-update control.

| Capability | What you can do | Start here |
| --- | --- | --- |
| Beam generation | Retain and rank multiple continuations for each input | [Beam Search](features/beam-search.md) |
| Catalog constraints | Generate valid three-component SIDs from compiled Tries | [Constraint tables](features/constraint-tables.md) |
| Full-vocabulary generation | Run without a catalog, with backend-specific width limits | [Unconstrained Beam](features/unconstrained-beam-search.md) |
| Device execution | Use CUDA V1/V2 or the prepared Ascend V1 runtime | [Installation](getting_started/README.md#choose-your-runtime) |
| Serving and operations | Use Python, HTTP or gRPC and coordinate resource updates | [Serving](serving/README.md), [Hot updates](features/hot-update.md) |

## Find your path

| I want to... | Read |
| --- | --- |
| Install and run a model | [Getting started](getting_started/README.md) |
| Understand model-specific workflows | [Models](models/README.md) |
| Set engine and GR options | [Configuration](configuration/README.md) |
| Measure quality or latency | [Usage and benchmarks](usage/README.md) |
| Integrate a public interface | [Python API](api/README.md), [CLI](cli/README.md) |
| Change the implementation | [Developer guide](contributing/README.md), [Design](design/index.md) |
| Report a problem or contribute | [Community](community/README.md) |

## Runtime compatibility

**vLLM-GR 0.30.0 requires vLLM 0.30.0.** Ascend additionally requires the
prepared vendor stack described in its installation guide. CPU packages support
packaging and host checks; accelerator Beam inference requires CUDA or Ascend.
See [dependency versions](configuration/dependency_version_policy.md) before
changing a runtime. Feature guides document the supported backend combinations.
