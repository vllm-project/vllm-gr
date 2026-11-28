# Pull request checks

Use this guide to prepare a pull request and read its checks in GitHub.

## Pull request CI

Run the relevant [local tests](local_runner.md) or
[documentation checks](../pre_commit_setup.md#documentation-checks) before
marking the PR ready for review. Include the commands, outcomes and any
checks you could not run in the PR description.

GitHub shows the checks that apply to the target branch and changed files.
[PR validation](../../../.github/workflows/ci.yml) selects either
documentation checks or full validation. The
[wheel contents](../../../.github/workflows/wheel-contents.yml) and
[wheel build](../../../.github/workflows/build_wheel.yml) workflows have
their own triggers. Maintainers apply any required labels and arrange
hardware runs that need shared infrastructure.

### Rerun validation

After pushing a fix, check the results for the new head commit. A rerun
tests the original run's commit. If a check needs a retry and GitHub does
not give you that option, ask a maintainer in the PR.

## Before merging

1. Review the latest diff and resolve outstanding review comments. If the PR
   head has moved since review, check the new changes.
2. Open the checks for the latest PR revision. For changes to OneRec
   thresholds, review the
   [baseline and measurements](benchmark_acceptance.md).
3. Verify that the PR meets the target branch's required checks and review requirements.

After merge, [Buildkite](../../../.buildkite/pipeline.yml) runs L3 CUDA and
Ascend tests. The [test-level workflow](../../../.github/workflows/test-levels.yml)
selects L4 nightly and L5 weekly. Use those runs to follow up on regressions.
See [test levels](test_levels.md#running-by-level) for device coverage and
how to reproduce a run.
