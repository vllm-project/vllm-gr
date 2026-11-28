# Maintaining agent instructions

[AGENTS.md](../../AGENTS.md) holds project-wide instructions and routes tasks
to guides and skills. [CLAUDE.md](../../CLAUDE.md) imports it with `@AGENTS.md`.
Keep a shared convention in one place so contributors using either tool see
the same guidance.

## Where to make a change

| Change | Owner |
| --- | --- |
| Project-wide convention or routing a task to its guide | [AGENTS.md](../../AGENTS.md) |
| Loading the shared instructions in Claude Code | [CLAUDE.md](../../CLAUDE.md) |
| Adding or updating a repository-local skill | [`.agents/skills/`](../../.agents/skills/) (`<skill-name>/SKILL.md`) |
| Contributor setup, review and CI | [Contributing](README.md) |
| Code ownership and runtime behavior | [Development navigation](development.md) and its linked design documents |
| Selecting tests or hooks | [Test levels](ci/test_levels.md) and [pre-commit](pre_commit_setup.md) |

Keep enforceable rules in the configuration, scripts or tests that implement
them. Link to their owner when a contributor needs context.

## Repository skills

Skills package reusable task workflows. Keep shared skills under
`.agents/skills/<skill-name>/SKILL.md` regardless of the agent a contributor
uses. Codex discovers them there. Other agents can follow the task links in
`AGENTS.md` to read the same files directly, including Claude Code through
`CLAUDE.md`. Keep one copy of each skill instead of adding agent-specific
skill directories.

Keep public contracts and human-facing procedures in their existing guides,
and call the repository tools for deterministic checks and measurements.
Skills should describe task decisions and reference those sources instead of
copying their commands, thresholds or implementation. Keep mandatory project
rules in `AGENTS.md`, even when a skill also uses them.

## Verify a routing change

1. Run the [documentation checks](pre_commit_setup.md#documentation-checks) and follow the affected links.
2. If you changed how instructions load, start a fresh session in the checkout
   and inspect the active instructions and available skills. For an agent that
   reads skills through task links, verify that the links resolve.
3. Try a representative task, such as a benchmark comparison or a public
   configuration change. Confirm that the selected workflow reaches the
   appropriate guide, tool and verification steps.
4. Record the tool version and which guide or skill it selected.
