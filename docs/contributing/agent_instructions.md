# AI coding tools and instruction maintenance

[AGENTS.md](https://github.com/JiusiServe/vllm-gr/blob/main/AGENTS.md) is the shared project instruction source.
[CLAUDE.md](https://github.com/JiusiServe/vllm-gr/blob/main/CLAUDE.md) imports it. Keep reusable development knowledge in
ordinary repository documents so developers and different tools can use it.

## Tool integration

| Tool | Repository entry point | Check in a fresh session |
| --- | --- | --- |
| Codex | Root `AGENTS.md` | Start in this checkout and confirm the active instruction sources include it |
| Claude Code | Root `CLAUDE.md` with `@AGENTS.md` | Use `/context` to inspect Memory files and confirm the import |
| Other agents | Their supported project-instruction mechanism | Verify that it actually loads the shared file |

Codex discovers guidance from the project root toward the starting directory;
overrides and user configuration can affect the result. Use the
[official discovery documentation](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
when diagnosing missing or stale instructions.

Recent Claude Code versions can also load `AGENTS.md` directly, depending on
settings and other instruction files. We retain the explicit import for sessions
that use `CLAUDE.md`. See the
[official shared-file guidance](https://code.claude.com/docs/en/memory#share-one-file-with-other-coding-tools).
This avoids maintaining a separate copy of the rules.

A Markdown link is navigation, not a portable automatic import. Root guidance
therefore tells agents when to read each task guide. A root `@AGENTS.md` import
does not recursively import every nested `AGENTS.md`. If directory-specific
instructions become necessary, verify loading from both the repository root
and that directory in each supported tool; add an appropriate local adapter
only where needed.

Keep permissions, model preferences, and personal memory in the tool's own
configuration. Instruction text does not execute hooks or enforce permissions.
Separate worktrees and environments can keep concurrent tool sessions from
overwriting one another's work.

## Where information belongs

| Information | Maintain it in |
| --- | --- |
| Short project-wide instructions and task routing | Root `AGENTS.md` |
| Tool-specific loading adapter | `CLAUDE.md` or the chosen tool's equivalent |
| Setup, change workflow, PR expectations, CI labels | [CONTRIBUTING.md](https://github.com/JiusiServe/vllm-gr/blob/main/CONTRIBUTING.md) |
| Code ownership and links to domain contracts | [Development navigation](development.md) |
| Test selection, runtime prerequisites, utility behavior | [Development utilities](tools/README.md) |
| Hook installation, scope, documentation checks | [Pre-commit guide](PRE_COMMIT.md) |
| Runtime behavior and invariants | The owning contract linked from development navigation |
| Enforced versions, formatting, checks, benchmark gates | Existing configuration, scripts, and tests |

Add guidance for a durable, non-obvious project convention. A one-off failure
usually belongs in a regression test or a code fix. If a rule already has an
owner, link there instead of copying it. Prefer a symbol search over an
exhaustive list of implementation and test filenames.

Keep the root brief enough to read on every task. Move domain detail into its
existing guide before adding more root instructions. Add a directory-specific
instruction file only when it has substantial additional rules and a verified
loading strategy for the team's tools. Do not pre-create wrappers for tools the
team does not use.

## Review changes to instructions

1. Check every command, path, and behavioral claim against the current code.
   Distinguish historical measurements from current contracts.
2. Remove duplicate or conflicting guidance; keep `CLAUDE.md` as an import
   unless there is an actual Claude-specific requirement.
3. Run [documentation checks](PRE_COMMIT.md#documentation-checks), including
   links and referenced command targets.
4. After changing loading or routing, use a fresh session to list active
   instruction sources and identify the relevant guide for a representative
   task. For a documentation edit it should select documentation checks; for a
   graph edit it should find the runner contract and device validation.
   Record the tool/version and observed behavior; file validation alone does
   not establish that a tool loaded the instructions.

## Upstream reference

This organization follows vLLM's separation of a small root instruction file,
task-specific guides, and human contributor documentation. See
[upstream AGENTS.md](https://github.com/vllm-project/vllm/blob/main/AGENTS.md),
[instruction maintenance](https://github.com/vllm-project/vllm/blob/main/docs/contributing/editing-agent-instructions.md),
and [test guidance](https://github.com/vllm-project/vllm/blob/main/tests/AGENTS.md).
Those are design references, not instructions governing this checkout.
vLLM-GR keeps its own dependency installation, formatting, runtime, and CI
configuration; copying upstream tool commands or repository policies would
change that contract.
