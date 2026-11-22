# Design documents

Start with the [architecture overview](architecture_overview.md) to follow
a OneRec request from the frontend through EngineCore to the device worker.

The pages under `docs/design/feature/` follow individual operations from
request admission through device execution and online updates. Module design
covers the interfaces, data layouts and lifecycle that they share.
For startup settings and supported combinations, see
[configuration](../configuration/README.md) and
[feature compatibility](../user_guide/feature_compatibility.md).

## Feature design

### Requests and scheduling

- [Beam batch execution](feature/beam_batch_execution.md) defines what a frontend submits, what EngineCore admits and what the caller receives.
- [Asynchronous scheduling](feature/async_scheduling.md) explains how CUDA and Ascend workers carry beam state between stages and when a session releases its resources.

### Prefill execution

- [CUDA Prefill graph capture](feature/cuda_prefill_graph.md) covers token buckets in both model runners, padded replay and the fresh-prompt requirement for dense attention.
- [Ascend BSND Prefill graphs](feature/npu_prefill_bsnd.md) describes how complete prompts fit into fixed per-request regions and when the worker falls back.

### Decode attention

- [CUDA Beam Decode attention](feature/cuda_decode_attention.md) follows the shared prompt pages and per-beam suffix KV through attention and the final merge.
- [Ascend Decode execution](feature/npu_decode_execution.md) covers xllm and cascade selection, parent KV reordering and caller-owned XAttention output.

### Online updates

- [Model and catalog updates](feature/weight_update.md) describes rolling checkpoint reloads, online Trie replacement and recovery after an interrupted update.

## Module design

| Component | Responsibility |
| --- | --- |
| [GPU Beam state](module/beam_state.md) | Candidate history, completion records and buffer lifetime |
| [Constraint tables](module/constraint_tables.md) | Catalog layout and per-prompt Trie routing |
| [CUDA ModelRunner V2](module/model_runner_v2.md) | Native request integration, graph ownership and cleanup |
| [Platform backend](module/platform_backend.md) | Provider discovery, activation and dispatch boundaries |
| [Runtime patches](module/runtime_patches.md) | Upstream assumptions and the reason for each hook in the [patch inventory](patch_inventory.yaml) |

[Development navigation](../contributing/development.md) maps code changes to their owning modules and tests.
