# vLLM-GR documentation

Read the Markdown documentation directly on GitHub. Use the branch or tag that
matches the version of vLLM-GR you are running.

## Installation

- [CUDA quickstart](quickstart.md) and [Ascend NPU quickstart](quickstart-ascend.md):
  install the runtime and verify your environment.
- [Dependency and version policy](dependency_version_policy.md): check runtime
  compatibility, package variants, and release rules.

## Inference and serving

- [OneRec Beam Search V1 quickstart](onerec_v1_async_quickstart.md): build a
  Constraint Table and run an end-to-end example with asynchronous scheduling.
- [Offline Beam Search](../examples/offline_inference/beam_search/README.md)
  and [online HTTP serving](../examples/online_serving/beam_search/README.md):
  configure Beam Search V1 and run inference.
- [Pangu offline inference](../examples/offline_inference/pangu1b/README.md):
  run a model through the standard inference path.
- [gRPC and NumPy token inputs](grpc_numpy_upgrade.md) and
  [Nginx load balancing](data_parallelism.md): integrate and scale serving.
- [Catalog-constrained HTTP serving](../examples/online_serving/catalog/README.md),
  [Multi-Trie Constraint Tables](multi_root_constraint_table.md), and
  [unconstrained Beam Search on NPU](npu_unconstrained_beam.md): choose how to
  constrain generated candidates.
- [Security policy](../SECURITY.md): review deployment boundaries and report
  vulnerabilities.

## Benchmarks

- [OneRec benchmark overview](benchmarks/README.md): prepare the dataset, run
  throughput or accuracy benchmarks, and find troubleshooting steps and CI gates.
- [Standalone Qwen Beam V1 benchmark](../benchmarks/qwen/README.md): measure
  offline performance and collect timelines.
- [Benchmark policy in CI](ci/benchmark-policy-approvals.md): see how CI
  selects approved inputs and thresholds.

## Development and maintenance

- [Contributing](../CONTRIBUTING.md), [development navigation](development.md),
  [validation tools](tools/README.md), and [pre-commit](PRE_COMMIT.md):
  find the owning code and choose checks before opening a PR.
- [Beam batch execution](beam_batch_execution_contract.md),
  [decision helper](beam_search_decision_helper_contract.md),
  [persistent Beam state](beam_state_pr1.md), and
  [Beam Worker adapter](beam_stage_pr2.md): understand protocol and state
  ownership.
- [Asynchronous scheduling](async_scheduling.md),
  [scheduling validation](async_scheduling_validation.md), and
  [Beam Search metrics](beam_search_metrics.md): understand request progress,
  validation, and observability.
- [Platform backend contract](platform_backend_contract.md) and
  [runtime patch inventory](runtime_patch_inventory.md): find platform
  registration and patched upstream interfaces.
- [ModelRunner V2](modelrunner_v2_multibatch.md),
  [CUDA Prefill capture](prefill_graph_capture.md),
  [Beam prefix optimization](beam_prefix_optimization.md), and
  [short suffix attention](short_suffix_attention.md): understand CUDA execution.
- [NPU BSND Prefill graphs](npu_prefill_bsnd.md) and
  [XAttention output binding](npu_xattention_out.md): review optional NPU paths.
- [AI coding tools](agent_instructions.md) and
  [source attribution](legal/README.md): maintain project instructions and notices.
