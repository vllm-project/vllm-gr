# Development navigation

This guide maps runtime changes to their owning code and contracts. Commands run
from the repository root. For setup and review, use
[Contributing](https://github.com/JiusiServe/vllm-gr/blob/main/CONTRIBUTING.md); for test execution, use
[development utilities](tools/README.md).

## Scope and code map

vLLM-GR is a plugin and adaptation layer over an installed vLLM runtime. OneRec
Beam inference and ordinary Pangu LLM inference have different execution paths.
All paths below are relative to the repository root.

| Change | Start here | Owning contract or guide |
| --- | --- | --- |
| Engine arguments and device defaults | `vllm_gr/engine/arg_utils.py`, `vllm_gr/config/`, `vllm_gr/platforms/` | [EngineArgs configuration override](../design/runtime_patch_inventory.md#gr-patch-005-engineargs-configuration-override) |
| Imports, initialization, runtime compatibility | `vllm_gr/plugin.py`, `vllm_gr/_env_check.py`, `vllm_gr/patch.py` | [Patch inventory](../design/runtime_patch_inventory.md), [inventory data](../design/patch_inventory.yaml) |
| Configuration and public Beam API | `vllm_gr/config/`, `vllm_gr/entrypoints/llm.py` | [Offline configuration](https://github.com/JiusiServe/vllm-gr/blob/main/examples/offline_inference/beam_search/README.md), [online configuration](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md) |
| HTTP, gRPC, session transport, and scheduling | `vllm_gr/entrypoints/`, `vllm_gr/v1/engine/`, `vllm_gr/v1/core/sched/` | [Beam batch contract](../features/beam-search.md#batch-execution-contract), [async scheduling](../features/beam-search.md#asynchronous-scheduling), [gRPC/NumPy](../serving/README.md#grpc-and-numpy-transport) |
| Beam state and candidate selection | `vllm_gr/v1/beam/` | [Beam batch contract](../features/beam-search.md#batch-execution-contract), [GPU Beam state](../features/beam-search.md#worker-state-and-execution) |
| Worker, Attention, and graph execution | `vllm_gr/v1/worker/`, `vllm_gr/v1/attention/backends/`, `vllm_gr/ops/` | [Prefill capture](../features/cuda-graphs.md#prefill-capture-configuration), [ModelRunner V2](../features/cuda-graphs.md#modelrunner-v2-multi-batch-execution), [NPU Prefill](../features/attention.md#npu-bsnd-prefill) |
| Artifact loading and constrained Top-K | `vllm_gr/constraints/`, `tools/constraints/build_constraint_table.py` | [Artifact setup](../getting_started/quickstart.md), [multi-Trie contract](../features/constraint-tables.md#multiple-trie-roots) |
| Platform registration and activation | `vllm_gr/platforms/` | [Platform contract](../design/main-arch.md#platform-backend-contract) |
| Pangu compatibility | `vllm_gr/config/pangu_patch.py` | [Pangu evaluation and serving](tools/pangu1b.md) |

The sections below identify distinctions that are easy to miss across modules.
Detailed lifecycle and numerical rules stay in the linked contracts.

## Architecture and configuration

The [main architecture](../design/main-arch.md) owns initialization boundaries, request
flow and state ownership. Detailed contracts are grouped in the
[feature guides](../features/README.md). Use those before changing shared runtime
behavior; keep this document focused on code navigation and change workflow.

### Package ownership

Paths are relative to `vllm_gr/`. Internal imports use the implementation's
owning package. `entrypoints/gr.py` and `platforms/npu_admission.py` preserve
the original imports for `entrypoints/llm.py` and `platforms/ascend/admission.py`.
External providers and old plan pickles use `platforms/plan.py`
and the provider binding imports. Worker RPCs resolve `platforms/readiness.py`;
persisted Beam results resolve their sequence class in `entrypoints/beam_output.py`.
A package initializer may also define public entry points, including
the Beam submit/wait/close functions.

| Packages | Implementation owners |
| --- | --- |
| Package root, `config/` | Root holds public API, plugin bootstrap, version and process-wide patch orchestration. Configuration owns its parser, validation, default YAML and import-light backend plan codec. The general-plugin entry point and root exports remain stable. |
| `benchmarks/`, `engine/` | The installed benchmark registry serves the CLI. EngineArgs helpers configure native engine startup. |
| `entrypoints/`, `entrypoints/beam_search/` | Offline and online frontends share request/heap lifetime and detokenization. Beam request preparation, session transport adapters and lazy result fields live together. The standalone detokenizer can load without importing the Beam engine protocol. |
| `entrypoints/cli/`, `entrypoints/openai/` | CLI owns command dispatch, serve parser and banner. OpenAI owns HTTP request conversion and completion batching. |
| `entrypoints/grpc/`, `grpc/proto/` and `grpc/generated/` under `entrypoints/` | Server startup and request dispatch use the packaged schemas and generated bindings. The generator owns module names and private protobuf descriptors. |
| `constraints/` | Keep catalog-to-triples, triples-to-table and artifact serialization separate. The loader, device backends and versioned tables own different lifetimes. |
| `ops/`, `ops/cuda/`, `ops/npu/` | Public operator registrations and implementations, CUDA selection kernels and Ascend primitives. CUDA kernels share the packed score/index ordering. NPU projection owns its fixed tensors; worker hooks install and refresh it. |
| `platforms/`, `platforms/cuda/`, `platforms/ascend/` | Provider identity, discovery, admission, compatibility and startup installation. The CUDA provider exposes the worker-owned installer. |
| `sidecar/`, `sidecar/filesystems/`, `utils/` | Sidecar coordinates updates and storage adapters. Shared update status is importable from the package root. Device probes and memory cleanup retain their utility exports. |
| `v1/`, `v1/core/`, `v1/core/sched/` | Native V1 compatibility, core prefix hashes, and scheduler-owned session dispatch/retirement. Existing Ascend bootstrap hooks span several native initialization sites. |
| `v1/engine/` | EngineCore admission, wire protocol, client transport and cross-worker startup agreement. Readiness calls the worker to report its actual runner binding. |
| `v1/beam/`, `v1/metrics/` | Shared Beam contracts, execution geometry and session/pool primitives. Keep the CPU selection oracle for numerical parity and the metrics namespace for import compatibility. |
| `v1/attention/`, `v1/attention/backends/` | Backend selection, metadata builders, layout and execution. Cascade KV writes are shared across devices. Ascend layout restoration belongs to the consuming attention backend. CPU layout and device tensor construction have separate owners. |
| `v1/worker/`, `v1/worker/cuda/`, `v1/worker/npu/` | Worker owns service binding, the CUDA runner installer, model reload hooks and device execution. The Ascend runner constructs its batch views. CUDA workspace types share a directory with the loader that prewarms and accounts for them. |
| `v1/worker/cuda/v1/`, `v1/worker/cuda/v2/` | Generation-specific native inputs, hooks and graph lifetimes. The shared Prefill base still invokes the capture planner and error classifier. |

### Worker packages

The public installer is `vllm_gr.v1.worker.model_runner_patch.apply_worker_patches`.
The worker root owns cross-device dispatch, context, startup coordination,
diagnostics and the adapters shared by V1 and Ascend. Device implementations
live under `worker/cuda/` and `worker/npu/`. CUDA V1 and V2 use the same CUDA
stage, workspace, input and Decode support. Each version owns its native
integration and Prefill code in `cuda/v1/` or `cuda/v2/`.

Keep package initializers lightweight. Select backend implementations lazily
from the installer or the active dispatch branch. CUDA V2 can depend on shared
CUDA support and shared worker code. Its imports must stay independent of
CUDA V1 and Ascend integration.

### Configuration field maintenance

Declare each configuration field in `vllm_gr/config/models.py` with a dataclass
default, an attribute docstring, and `ConfigFieldSpec` in `field(metadata=...)`.
The specification binds its validator, canonical path, legacy key, required
input policy, and serialization policy. Parsers and writers consume these
declarations; add domain validation in `validation.py` and keep cross-field
constraints in the model. Input-specific path handling stays in `parser.py`.

Preserve the distinction between missing, null, and explicit false inputs.
Canonical inputs require `schema_version` even though direct construction has a
default. Table mode requires `gr.constraint_table.max_top_k` when constructing
a deployment. The standalone `ConstraintTableSpec` can omit this value.
The `validate-config` hook checks both configuration classes. See the
[standalone configuration tests](tools/README.md#targeted-tests-and-dependencies)
and extend the existing configuration round-trip tests for behavior changes.

## ModelRunner and graph boundaries

`beam_search` is the only Beam protocol, independently of native
ModelRunner V1/V2. The selection policy lives in `vllm_gr/_env_check.py`.

CUDA V1, CUDA V2, and NPU have different capture and storage lifecycles. Read
[Prefill capture](../features/cuda-graphs.md#prefill-capture-configuration) and
[V2 execution](../features/cuda-graphs.md#modelrunner-v2-multi-batch-execution) for CUDA, and
[NPU Prefill](../features/attention.md#npu-bsnd-prefill) for Ascend. Attention changes may also need the
[prefix](../features/attention.md#cuda-prefix-attention) and [short-suffix](../features/attention.md#cuda-short-suffix-attention)
contracts. Verify replay or fallback, buffer ownership, row mappings, and
cleanup for the affected runner; a graph-enabled setting is not replay evidence.

## Constraint and candidate boundaries

Start with `vllm_gr/constraints/backend.py` for output shapes and validity,
`vllm_gr/constraints/scoring.py` for comparable parent scores, and
`vllm_gr/constraints/loader.py` for artifact/tokenizer identity.
Canonical token IDs are tokenizer-global; validity comes from a structural mask,
not whether a score is finite. Read the
[multi-Trie contract](../features/constraint-tables.md#multiple-trie-roots) before changing routing
or platform support. Top-K ties need the declared parity contract, not an
assumption of identical slot ordering.

`vllm_gr/v1/beam/cpu_reference.py` checks candidate selection after generation;
it does not implement production scheduling, KV management, or CPU inference.

## Select regression coverage

Search by the changed symbol and behavior, then inspect the test fixtures. For
example, an initialization or transport change can start with:

```bash
rg -n 'initialize_runtime|re_apply_patches' tests vllm_gr
rg -n 'SUBMIT_GR_SESSION|BeamBatchRequest' tests vllm_gr
rg --files tests -g '*graph*' -g '*model_runner*'
```

| Boundary | Coverage location |
| --- | --- |
| Public API and entrypoints | `tests/entrypoints/` |
| Configuration and engine mode | `tests/engine/` |
| Scheduling, admission, and session lifecycle | `tests/v1/engine/` |
| CI policy and test levels | `tests/ci/` |
| Example scripts and benchmark helpers | `tests/examples/`, `tests/benchmarks/` |
| Protocol and upstream patch compatibility | `tests/contract/` |
| Provider registration and import isolation | `tests/platforms/` |
| Artifact, scoring, and backend behavior | `tests/constraints/`, `tests/tools/` |
| Worker, attention backends, and kernels | `tests/v1/worker/`, `tests/v1/attention/`, `tests/kernels/`, `tests/integration/` |
| OneRec/model integration and acceptance scripts | `tests/st/`, `tests/system/` |
| Wheel metadata and installation | `tests/packaging/` |

Directory names do not establish runtime requirements. Follow the
[validation guide](tools/README.md#validation-levels) for dependencies, slow-test
selection, hardware execution, and packaging isolation.

## Acceptance evidence

The [benchmark acceptance guide](tools/README.md#benchmark-acceptance) owns
policy selection and evidence requirements. Update the relevant contract and
examples when behavior changes; keep this page as navigation rather than a
second specification.
