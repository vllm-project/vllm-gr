---
name: vllm-gr-benchmark
description: Tune CUDA Beam performance using measured workloads, run, compare, or interpret vLLM-GR performance and OneRec accuracy benchmarks, reproduce benchmark gates, or investigate latency, throughput, cache behavior, and graph replay. Use when measured workload evidence is needed, rather than ordinary functional regression tests.
---

# Benchmark vLLM-GR

Paths below are relative to the repository root. Use the benchmark entry points,
documentation, and acceptance policy of the revision being measured.

## Choose a workload

Identify the question, target device, model, runner, and required evidence. Read
`docs/cli/bench.md` for the supported workloads, and the relevant guide below.
In a prepared environment, use `vllm bench --gr --list` and the runner's help to
confirm the available scenarios and arguments.

| Task | Scenario or guide |
| --- | --- |
| Tune CUDA inference configuration for a workload | `docs/configuration/gpu_tuning.md`; then choose the matching measurement scenario |
| Beam offline or HTTP latency | `beam-performance`; `docs/cli/bench.md#workloads` |
| Compare controlled Beam latency runs | `beam-performance-ab`; `benchmarks/beam_search/compare_latency_ab.py` |
| OneRec Recall and observed graph replay | `beam-recall`; `docs/cli/bench.md#workloads` |
| OneRec accuracy, offline throughput, or HTTP serving | `docs/benchmarking/README.md`, `docs/benchmarking/accuracy.md`, `docs/benchmarking/throughput.md` as applicable |
| Revision-isolated OneRec cold and cache-hit A/B | `ab-offline`; `tools/AB_Tests/README.md` |
| Synthetic Qwen Beam performance | `qwen-v1`; `benchmarks/qwen/README.md` |
| Reproduce or assess a CI benchmark gate | `docs/contributing/ci/benchmark_acceptance.md` |

Choose the smallest workload that answers the question. For dataset preparation,
read `docs/benchmarking/README.md`. A synthetic workload or smoke test does not
establish real-model Recall or production performance.

For configuration tuning, use the tuning guide to choose a feasible reference,
isolate the settings under test, and verify resource usage and quality. Start
with representative shapes before expanding the workload. Keep a reference
for each device; configuration sweeps do not require separate code checkouts.
When comparing devices with different effective configurations, interpret the
results as measurements of those deployments, not an isolated code speedup.

## Prepare comparable measurements

- Verify the measured revision, interpreter, imported GR package, benchmark
  driver, and device runtime. Use the target installation guide and
  `tools/ci/resolve_runtime.py` where applicable. Record the device identity,
  runner generation, and effective settings. Use assigned devices only.
- For new measurements in an A/B comparison, isolate each revision's checkout
  and runtime environment. Verify the package and driver actually belong to the
  intended checkout. Changing only the working directory can measure the same
  revision twice. Keep dependencies equivalent unless the runtime change is
  itself under investigation; record intentional differences. Reuse existing
  measurements if their provenance and the scenario's validator support the
  requested comparison. Treat historical measurements as evidence for their
  recorded revision and configuration. If evidence is insufficient, explain
  what is missing; run only the missing measurements needed for the task.
- Preserve model/tokenizer identity, dataset and catalog/constraint identity,
  ordered sample identities and input tokens. Reuse prepared inputs or verify
  logical identity with the benchmark's existing fingerprints. Match request
  geometry, beam and output budgets, batching, concurrency, cache policy, graph
  settings, warmup, and timing methodology apart from the variables under test.
- For cold/warm comparisons, verify actual cache reuse and comparable cache
  work. Reset and inspect cache counters at the points specified by the runner,
  outside its timed region. Do not remove public result materialization from
  the timed call or include separate profiling runs in reported latency.
- Where the environment, model configuration, or compiled operators are
  uncertain, run a minimal workload check before a long measurement. If the
  requested execution path fails or falls back, retain the evidence and record
  the actual path rather than treating it as the requested one.

## Execute and evaluate

Run the documented scenario when new measurements are needed. Keep separate raw
outputs for each measurement and any retries. Use its existing output and
workload validators and comparison tools. In particular:

- For Beam latency A/B, use `beam-performance-ab` to validate and compare
  recorded measurements, following its workload and retry contracts.
- For OneRec offline A/B, prepare and verify the pinned inputs as documented in
  `tools/AB_Tests/README.md`, then use `tools/AB_Tests/generate_summary.py` and
  compare the summaries from the updated checkout using
  `tools/AB_Tests/compare_ab.py`. Respect the comparator's evidence requirements
  for a qualified cache-hit performance verdict; report descriptive latency
  when measured cache work, sample identity, or environment evidence differs
  or is insufficient. Do not treat an invalid comparison as a result.
- For a CI gate, use the workload selected by the tested revision, branch, and
  device. Resolve its policy using `.github/benchmark-gate.yaml`,
  `tools/ci/resolve_benchmark_policy.py`, and the relevant workflow. Read the
  acceptance guide and applicable checked-in Recall baseline. Preserve the
  policy's input-identity checks, threshold semantics, retry procedure, and
  blocking/advisory rules. Record the tested SHA, policy identity, and effective
  thresholds. A custom local run does not establish CI gate acceptance.

Distinguish functional results, graph capture readiness, actual graph replay,
accuracy/Recall, and performance. For a replay claim, use observed execution
counters for the intended stage and batch geometry; configuration or capture
success alone is insufficient. Check fallback counts and workload completeness.
Use the selected scenario's aggregation semantics and report any evidence gaps.

## Report evidence

Include the question, revision(s), loaded runtime and driver identity, device,
model/tokenizer, input and catalog identity, beam/batch settings, runner,
effective graph and cache mode, exact commands, and raw output locations.
Present latency or throughput, accuracy, and replay evidence separately. For an
A/B verdict, identify controlled variables and intentional differences. Report
failures, skips, incomplete runs, and unverified paths explicitly. Do not claim
an improvement, acceptance, or replay without the corresponding evidence.
