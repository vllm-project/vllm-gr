# Developer guide

Use the same runtime and model assumptions as the behavior you are changing.
Start with a focused reproduction, read the owning contract, and choose the
smallest validation that can demonstrate the result.

| Task | Guide |
| --- | --- |
| Prepare a development environment and open a PR | [Contributing](https://github.com/JiusiServe/vllm-gr/blob/main/CONTRIBUTING.md) |
| Locate implementation owners and affected call paths | [Development navigation](development.md) |
| Understand runtime ownership and process boundaries | [Architecture](../design/main-arch.md) |
| Select host, device or real-model validation | [Validation levels](tools/README.md#validation-levels) |
| Format changed files without starting the full suite | [Pre-commit scope](PRE_COMMIT.md) |
| Build or edit the documentation | [Documentation guide](documentation.md) |
| Change benchmark acceptance policy | [CI policy](ci/benchmark-policy-approvals.md) |
| Maintain shared coding-agent instructions | [AI tool integration](agent_instructions.md) |

## Before submitting

Review the complete change and its affected contracts. Report actual commands
and outcomes, including failures, skips and hardware paths that were not run.
The [runtime patch inventories](../design/runtime_patch_inventory.md) and their
contract tests must stay aligned when upstream integration changes.

Device correctness, graph replay, recommendation quality and performance need
their own evidence. Use the [PR workflow](https://github.com/JiusiServe/vllm-gr/blob/main/CONTRIBUTING.md#pull-request-ci)
to understand which checks run for the submitted revision.
