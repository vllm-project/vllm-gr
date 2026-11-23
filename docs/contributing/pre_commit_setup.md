# Pre-commit setup and scope

Install the repository hooks, then choose the checks that cover your changes.

## Install hooks

Run these commands from the repository root in your development environment.

```bash
python -m pip install pre-commit
pre-commit install
```

The [hook configuration](../../.pre-commit-config.yaml) installs both the
`pre-commit` and `commit-msg` Git hooks.

## Choose the scope

| Command | Scope |
| --- | --- |
| `pre-commit run` | Run applicable hooks on staged files |
| `pre-commit run --all-files` | Run applicable hooks across the checkout |
| `pre-commit run HOOK_ID --files PATH` | Run the named hook for the selected path |

The regex import, CUDA API, ShellCheck, and Excalidraw PNG hooks honor the
selected file list. To check tracked shell scripts directly, run
`bash tools/pre_commit/shellcheck.sh` from the repository root.

For targeted checks, specify the hook ID. `pre-commit run --files` can trigger
`run-tests` when the hook ID is omitted. The test hook receives no filenames
and runs the full pytest suite by default, including slow tests. See
[test levels](ci/test_levels.md) and [running tests locally](ci/local_runner.md)
to select runtime tests.

If a hook changes a file, review the diff and rerun that hook.

## Documentation checks

For standalone Markdown linting, use the repository configuration explicitly.

```bash
markdownlint --config tools/pre_commit/.markdownlint.yaml README.md
```

Replace the example path in the Bash array with the Markdown files you
edited or added.

```bash
files=(docs/contributing/pre_commit_setup.md)
pre-commit run markdownlint --files "${files[@]}"
pre-commit run check-characters --files "${files[@]}"
pre-commit run typos --files "${files[@]}"
pre-commit run end-of-file-fixer --files "${files[@]}"
pre-commit run trailing-whitespace --files "${files[@]}"
pre-commit run mixed-line-ending --files "${files[@]}"
git diff --check
```

[Documentation CI](../../.github/workflows/pre-commit.yml) also runs
`pre-commit run check-filenames --all-files` on all tracked paths, including
for deletion-only PRs. Verify changed links, anchors, and example commands as
described in the [documentation guide](docs_guide.md#checks).

## Mypy coverage

The default group checks Python source and stubs under `vllm_gr/` with
`check_untyped_defs` and `follow_imports = "silent"`. The checker uses
annotations from imports while reporting diagnostics for its explicit inputs.
The regeneration hook checks the generated gRPC bindings. Test helpers and
the CUDA and Ascend worker contracts use `follow_imports = "skip"` to
isolate runtime dependencies. The hook prints each group's file count.

See the [CUDA stage hooks contract](../design/module/model_runner_v2.md#cuda-stage-hooks-contract),
the [CUDA stage typing tests](../../tests/tools/test_cuda_stage_typing.py)
and the [checker routing tests](../../tests/tools/test_mypy_hook.py).
