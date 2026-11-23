# Development navigation

Use this table to find the implementation, tests and design guide for the area you are changing.

| Change area | Source | Regression tests | Design or guide |
| --- | --- | --- | --- |
| Plugin initialization and runtime patches | [Plugin entry point](../../vllm_gr/plugin.py), [startup defaults](../../vllm_gr/engine/startup.py), [patch registration](../../vllm_gr/patch.py) | [Initialization](../../tests/patch/test_patch_initialization.py), [upstream patch contracts](../../tests/contract/test_upstream_patch_contracts.py) | [Runtime patches](../design/module/runtime_patches.md) |
| Public configuration | [Configuration package](../../vllm_gr/config/) | [Parsing and validation](../../tests/config/), [engine integration](../../tests/engine/test_gr_config.py) | [Configuration guide](../configuration/README.md) |
| Offline Beam API | [GRLLM](../../vllm_gr/entrypoints/gr.py) | [Offline Beam search](../../tests/entrypoints/gr/test_offline_beam_search.py) | [OneRec quickstart](../user_guide/onerec_quickstart.md) |
| HTTP and gRPC serving | [OpenAI-compatible HTTP](../../vllm_gr/entrypoints/openai/), [gRPC](../../vllm_gr/entrypoints/grpc/) | [HTTP](../../tests/entrypoints/openai/), [gRPC](../../tests/entrypoints/grpc/) | [Serving APIs](../serving/README.md) |
| Request admission, scheduling and session lifecycle | [Engine](../../vllm_gr/v1/engine/), [scheduler](../../vllm_gr/v1/core/sched/) | [Engine and scheduler tests](../../tests/v1/engine/) | [Beam batch execution](../design/feature/beam_batch_execution.md), [async scheduling](../design/feature/async_scheduling.md) |
| Beam contracts, state and candidate selection | [Shared Beam contracts](../../vllm_gr/v1/beam/), [CUDA state](../../vllm_gr/v1/worker/gpu/beam_state.py), [NPU decisions](../../vllm_gr/v1/worker/npu/beam_decision.py) | [Beam regression tests](../../tests/v1/beam/) | [GPU Beam state](../design/module/beam_state.md) |
| Workers, attention and kernels | [Workers](../../vllm_gr/v1/worker/), [attention backends](../../vllm_gr/v1/attention/backends/), [operators](../../vllm_gr/ops/) | [Workers](../../tests/v1/worker/), [attention](../../tests/v1/attention/), [kernels](../../tests/kernels/) | [ModelRunner V2](../design/module/model_runner_v2.md), [CUDA prefill](../design/feature/gpu_prefill_graph.md), [NPU prefill](../design/feature/npu_prefill_bsnd.md) |
| Constraint artifacts and candidate scoring | [Constraints](../../vllm_gr/constraints/), [table builder](../../tools/constraints/build_constraint_table.py) | [Constraint tests](../../tests/constraints/) | [Constraint tables](../design/module/constraint_tables.md) |
| Platform registration and runtime compatibility | [Platform providers](../../vllm_gr/platforms/), [runtime compatibility](../../vllm_gr/platforms/compat.py) | [Platform tests](../../tests/platforms/) | [Platform backend](../design/module/platform_backend.md) |
| Model compatibility | [Pangu compatibility fixes](../../vllm_gr/config/pangu_patch.py) | [Pangu compatibility tests](../../tests/config/test_pangu_compat.py) | [Pangu evaluation and serving](../models/pangu.md) |

## Validate a change

- **Public configuration.** Check the packaged defaults and parser, then verify both `GRLLM` and `vllm serve --gr`.
- **Runtime changes.** Choose regression tests by fixture, [level and platform](ci/test_levels.md). Use the [local runner guide](ci/local_runner.md) to run them.
- **Wheel changes.** Follow [packaging validation](dependency_policy.md#packaging-validation).
- **OneRec accuracy or performance.** Follow the [benchmark acceptance guide](ci/benchmark_acceptance.md).
