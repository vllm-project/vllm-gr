# Maintaining agent instructions

[AGENTS.md](../../AGENTS.md) holds project-wide instructions and links to task
guides. [CLAUDE.md](../../CLAUDE.md) imports it with `@AGENTS.md`. Keep a
shared convention in one place so contributors using either tool see the same
guidance.

## Where to make a change

| Change | Owner |
| --- | --- |
| Project-wide convention or routing a task to its guide | [AGENTS.md](../../AGENTS.md) |
| Loading the shared instructions in Claude Code | [CLAUDE.md](../../CLAUDE.md) |
| Contributor setup, review and CI | [Contributing](README.md) |
| Code ownership and runtime behavior | [Development navigation](development.md) and its linked design documents |
| Selecting tests or hooks | [Test levels](ci/test_levels.md) and [pre-commit](pre_commit_setup.md) |

Keep enforceable rules in the configuration, scripts or tests that implement
them. Link to their owner when a contributor needs context.

## Verify a routing change

1. Run the [documentation checks](pre_commit_setup.md#documentation-checks) and follow the affected links.
2. If you changed how instructions load, start a fresh session in the checkout and inspect the active instructions.
3. Try a representative task. Locate the documentation checks or find the runner design for a CUDA graph change.
4. Record the tool version and which guide it selected.
