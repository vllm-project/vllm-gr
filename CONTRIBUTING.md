# Contributing to vLLM-GR

You may find information about contributing to vLLM-GR on [Contributing](https://vllm-gr.readthedocs.io/en/latest/contributing/)

## Pull request CI

All pull request validation, including pre-commit, tests, benchmarks and wheel
builds, requires the `ready` label. Run local checks before asking a maintainer
to add it. Create the repository label named `ready` if it does not exist.

- Opening a PR or pushing commits without `ready` skips validation and wheel
  builds. Only the lightweight `Benchmark blocking gate` runs on a GitHub-hosted
  runner and fails because validation has not passed.
- Adding `ready` starts CI on a non-draft PR. Further pushes automatically rerun
  CI while the label remains.
- Removing `ready` or converting the PR to draft cancels its previous PR runs
  and skips new validation. Marking a labeled draft ready for review starts CI.
- Adding or removing other labels neither reruns nor cancels validation. These
  events do not publish a check named `Benchmark blocking gate`.
- Existing branch and wheel path filters still apply. Push checks on configured
  target branches and manually dispatched workflows retain their existing behavior.
- To request a fresh PR run, remove and re-add `ready`.

Keep `Benchmark blocking gate` required in branch protection. It requires every
blocking validation job to succeed and rejects failed, cancelled or skipped
validation. Removing `ready` or converting a PR to draft therefore blocks merging
until validation runs successfully again. Cancelling CI stops the Pangu jobs as
well; their device cleanup steps still run after device mapping has succeeded.

This implementation targets `decode_graph`. The `main` branch has a different
CI implementation and must be updated separately. The `/retest` router also
runs from the default branch: enforcing the current `ready` label for that
manual command requires the separate router update on `main`.
