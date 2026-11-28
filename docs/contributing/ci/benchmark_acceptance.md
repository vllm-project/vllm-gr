# OneRec benchmark acceptance

<a id="benchmark-acceptance"></a>

See [pull request checks](pr_merge_loop.md#pull-request-ci) for the CI review
workflow.

| Workload | When it runs | What it checks |
| --- | --- | --- |
| [OneRec five-task Recall](#gpu-five-task-recall) | GPU PR validation targeting `main` and eligible manual GPU runs | Per-task SID Recall@32, input identity and graph replay |
| [OneRec Video-100](#npu-video-100) | Buildkite [CUDA nightly](../../../.buildkite/test-cuda-nightly.yml) and [NPU nightly](../../../.buildkite/test-npu-nightly.yml) | Offline and HTTP accuracy and performance under the resolved policy |

The OneRec decode job must succeed for full validation to pass.

## Benchmark gates

The [policy file](../../../.github/benchmark-gate.yaml), its
[resolver](../../../tools/ci/resolve_benchmark_policy.py) and the tests are
checked out at the same workflow SHA. For a PR, GitHub supplies the test merge
commit. For a manual run, it supplies the selected revision. The policy job
validates the YAML and passes one set of resolved values to the benchmark jobs.
Its summary records the tested commit, policy SHA-256 and effective thresholds.

Video-100 reads the model name of the visible accelerator and selects its entry
in the [JSON baseline](../../../tests/baseline/onerec_video_benchmark.json).
The test fails if the detected model has no entry. The test log records the
detected model and selected entry.

### GPU five-task Recall

The GPU runner evaluates OneRec-1.7B with ModelRunner V1, batch size 4,
beam width 128, and 1,000 fixed samples per task. Video, ad, label_cond, and
interactive use SID-to-PID mapping. Product uses SID-to-IID.

CI checks SID Recall@32 for each task against the
[calibrated baseline](../../../benchmarks/beam_search/recall_baseline_batch4.json).
A first result below a task's floor triggers two fresh runs. The Recall check
fails when the same task is below its floor in all three runs. Equality passes.
PID Recall is diagnostic. Input identity, graph capture, and batch-size-4
replay are separate blocking checks.

#### Reproduce the CUDA gate

The checked-in baseline records the model revision, dataset and artifact
identities, runtime, and A100 calibration samples used for the vLLM 0.30.0
floors. Reproduce the check with the pinned snapshot and prepared data on a GPU.

```bash
V1_RECALL_BASELINE_FILE="$PWD/benchmarks/beam_search/recall_baseline_batch4.json" \
  bash tests/system/run_beam_v1_recall.sh
```

### OneRec Video-100

<a id="npu-video-100"></a>

The [Video-100 test](../../../tests/e2e/test_onerec_video_benchmark.py) evaluates
100 video prompts offline and over HTTP at beam width 128. Both measurements
must meet the pass@1, pass@32, and recall@32 floors in the
[Video-100 baseline](../../../tests/baseline/onerec_video_benchmark.json). Equality passes.
An accuracy result below a blocking threshold fails the pytest case.

The shared floors were calibrated using L20 and A100 measurements after the
parent-score correction in `0f5018516f82`. Applying the same floors on Ascend
is an acceptance policy decision. The test log records the thresholds used
for each measured metric.

The test downloads `benchmark_data` from the gated
`OpenOneRec/OpenOneRec-RecIF` dataset and uses the Hugging Face cache.
On a prepared Ascend runner with the model available, authenticate to
Hugging Face with a token that has access to the dataset and run:

```bash
python -m pytest -sv --run-slow --strict-test-levels \
  -m 'l4 and npu_test' tests/e2e/test_onerec_video_benchmark.py
```

## Changing a policy

Update the policy, resolver and corresponding tests together when their
contract changes. For a threshold change, review the YAML diff alongside the
effective settings in the **Resolve benchmark policy** summary. For a new
baseline, record its calibration revision, input identities and raw results.

When comparing performance or Recall, record both code revisions, imported
package, runtime versions, device, model/tokenizer/catalog, inputs, batch and
beam sizes, runner, graph settings, commands, and raw output locations. Hold
the workload constant and report latency, graph replay and Recall separately.
Use [benchmark setup](../../benchmarking/README.md) for data preparation.

## Local validation

```bash
python tools/ci/resolve_benchmark_policy.py \
  --policy-file .github/benchmark-gate.yaml --validate-only
python -m pytest -q tests/ci/test_benchmark_policy_ci.py \
  tests/ci/test_benchmark_branch_policy.py tests/ci/test_ready_label_ci.py
```

The resolver also accepts `--github-env` for shell consumers and
`--github-output` for GitHub Actions. Both use the same validated values.
The explicit `--bootstrap-strict` CLI option remains available for existing
standalone callers; CI does not select it automatically.

Independent runners should load the policy and resolver from the same
checkout they test and record the revision and effective settings.
