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

[Buildkite](../../../.buildkite/pipeline.yml) checks the diff for PRs targeting
`main` before building the CI image or uploading the `ready` and `merge-test`
pipelines. A PR skips those steps if every changed path is under `docs/`,
ends in `.md`, or is the root `mkdocs.yml`.
Both paths of a rename must qualify. GitHub still runs the applicable
documentation and static checks. Buildkite runs its usual steps for mixed
changes, an unresolved diff, and pushes to `main`.

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

After merge, the [test-level workflow](../../../.github/workflows/test-levels.yml)
selects L3 on pushes to `main`, L4 nightly and L5 weekly. Use those runs to
follow up on regressions. See [test levels](test_levels.md#running-by-level)
for device coverage and how to reproduce a run.
