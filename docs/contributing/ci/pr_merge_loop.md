# Pull request checks

## Pull request CI

Run the relevant [local tests](local_runner.md) or
[documentation checks](../pre_commit_setup.md#documentation-checks) and record
the commands, results and outstanding checks in the PR description.

| Workflow | When it runs |
| --- | --- |
| [PR validation](../../../.github/workflows/pre-commit.yml) | For a non-draft PR targeting `main`, applying `ready` starts validation. Opening, reopening or updating a PR with that label, or marking a labeled draft ready for review, also starts it. |
| [Wheel contents](../../../.github/workflows/wheel-contents.yml) | On eligible PR opens, updates and reopens targeting `main` or `release/v*`, and on eligible pushes to those branches. No label is needed. |
| [Wheel builds](../../../.github/workflows/build_wheel.yml) | On eligible non-draft PRs with `ready` targeting `main` or `release/v*`. The path filters are defined in the workflow. |

For a ready PR to `main`, check **Benchmark blocking gate**. Its inputs depend
on the complete diff:

- For eligible Markdown-only changes, the gate requires **Documentation
  checks** to pass. The [classifier](../../../tools/ci/docs_only.js) checks
  the changed paths against the PR's base revision.
- Full validation requires benchmark policy resolution, pre-commit (including
  L1/L2 tests), the Decode benchmark, and both Pangu accuracy jobs to succeed.
  It also requires a passing Beam performance A/B result for the current
  revision. The `abtest` label can trigger a separate A/B run. See the
  [OneRec benchmark guide](benchmark_acceptance.md) for the Decode workload.

### Rerun validation

After a fix, push the new commit with `ready` still applied. Re-running an
existing Actions run tests its original revision. Ask a maintainer if a label
or retry is needed.

## Before merging

1. Review the latest diff and resolve outstanding review comments. The
   [CODEOWNERS file](../../../.github/CODEOWNERS) assigns review for all paths
   to `@hsliuustc0106`. Check the PR's merge box for any required approvals.
2. Verify the gate on the latest head and base, and check any other required
   statuses shown by GitHub for the target branch. A passing A/B-only run
   covers just that workload.

After merge, [Test levels](../../../.github/workflows/test-levels.yml) selects
L3 on pushes to `main`, L4 nightly and L5 weekly. Inspect the GPU and NPU
lane logs to see which tests ran: a lane skips execution when the runner has
the other accelerator. See [test levels](test_levels.md#running-by-level)
to reproduce a run.
