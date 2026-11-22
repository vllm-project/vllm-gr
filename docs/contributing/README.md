# Contributing to vLLM-GR

Use this guide to set up a checkout, validate a change and open a pull request.

<a id="development-setup"></a>

## Getting started

- **Editing documentation?** Run the [documentation checks](pre_commit_setup.md#documentation-checks)
  on the Markdown files you changed and verify their links. These checks run
  without an accelerator.
- **Changing runtime code?** Follow the [CUDA](../getting_started/installation/gpu.md)
  or [Ascend](../getting_started/installation/npu.md) installation guide for
  an editable checkout with development dependencies. To check your runtime,
  enable [GR CLI dispatch](../cli/README.md#command-form-and-installation)
  and run [`vllm check-env --gr`](../cli/check_env.md).

## Find a contributor guide

| Task | Guide |
| --- | --- |
| Find the code and tests for a change | [Development navigation](development.md) |
| Classify tests or run them locally | [Test levels and platform markers](ci/test_levels.md), [local test runner](ci/local_runner.md) |
| <a id="linting"></a>Install hooks and run formatting or documentation checks | [Pre-commit setup and scope](pre_commit_setup.md) |
| Add or move a documentation page | [Documentation guide](docs_guide.md) |
| Change a dependency or verify a package | [Dependency and packaging policy](dependency_policy.md) |
| <a id="pull-request-ci"></a>Prepare a PR, read CI results, or request a retry | [Pull request checks](ci/pr_merge_loop.md) |
| Reproduce a OneRec benchmark gate or review its baseline | [OneRec benchmark acceptance](ci/benchmark_acceptance.md) |

<a id="change-and-validation-workflow"></a>
<a id="issues-and-design-proposals"></a>

## Issues, RFCs and pull requests

Search existing issues and PRs first. Choose an [issue template](../../.github/ISSUE_TEMPLATE)
for a bug, feature request, new model or performance discussion. For a major
architecture change, use the [RFC template](../../.github/ISSUE_TEMPLATE/750-RFC.yml)
to discuss the design with maintainers before implementation.

<a id="commit-sign-off-and-pr-review"></a>

Sign off your commits with `git commit -s`. Update the affected tests,
documentation and examples, then fill in the [PR template](../../.github/PULL_REQUEST_TEMPLATE.md).

1. In **Purpose**, describe who is affected and what changes for them. Link a related issue or RFC when one exists.
2. In **Test Plan**, identify the behavior to cover and the hardware it requires.
3. In **Test Result**, give the commands you ran and their outcomes, including failures, skips and outstanding checks. For performance or accuracy comparisons, include the revisions, model, device, workload and configuration.

<a id="ai-coding-tools"></a>

## AI Usage Policy

- Disclose which tool helped and what it did in the PR, issue or review comment.
- Review the generated changes and verify technical claims and references yourself. You are responsible for what you submit.
- Describe the validation you performed and any checks you could not run.

See [project instructions](../../AGENTS.md) and [AI tool integration](agent_instructions.md)
for guidance on working with coding agents.
