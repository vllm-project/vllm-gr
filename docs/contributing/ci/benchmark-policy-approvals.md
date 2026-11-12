# Benchmark policy in CI

CI reads `.github/benchmark-gate.yaml` and runs
`tools/ci/resolve_benchmark_policy.py` from the same checkout as the tests.
Every code checkout in `pre-commit.yml` uses the workflow's fixed `github.sha`:
the test merge commit for a pull request, or the selected commit for a manual
run. This includes both the PR's edits and the base changes in that merge.
A later base update cannot change the contents of the selected commit.

The policy job validates the YAML, applies the GPU/NPU and board settings,
and emits one `policy-json` output for the benchmark jobs. Its summary records
the commit, policy SHA-256, and effective thresholds. Missing or invalid policy
files fail the job. The normal CI path has no remote policy loader, old-base
bootstrap, or second policy resolution in the final gate.

## Changing a policy

A PR can update the policy, resolver, and tests together. These changes go
through ordinary code review, just like test assertions and workflow changes;
there is no separate policy approval protocol. Review threshold changes in the
PR diff and the effective values in the **Resolve benchmark policy** summary.

This follows the repository-local configuration and test commands used by
[vLLM's evaluation jobs](https://github.com/vllm-project/vllm/blob/main/.buildkite/test_areas/lm_eval.yaml)
and [vLLM-Omni's ready pipeline](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/cuda/test-ready.yml).
The CI service and device scheduling remain specific to vLLM-GR.

The `ready`, draft, documentation-only, and `abtest` routing rules are described
in [Contributing](https://github.com/JiusiServe/vllm-gr/blob/main/CONTRIBUTING.md#pull-request-ci). The required
**Benchmark blocking gate** consumes job results. Failed, cancelled, missing,
or skipped required validation does not pass. An eligible cached A/B result can
satisfy the A/B requirement; documentation-only and explicit A/B runs retain
their separate checks.

## Local validation and Jenkins

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

An independent Jenkins pipeline should check out its tested revision and load
both the policy and resolver from that checkout, recording the revision and
effective settings. Changes here do not reconfigure an external Jenkins job.
