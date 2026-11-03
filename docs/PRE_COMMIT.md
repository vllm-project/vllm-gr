# Pre-commit setup and scope

The source of truth is [.pre-commit-config.yaml](../.pre-commit-config.yaml).
Run commands from the repository root using the intended development environment.

## Install hooks

```bash
python -m pip install pre-commit
pre-commit install
```

The configuration installs both pre-commit and commit-message hooks. Its Python
version check follows the supported range in `pyproject.toml`. Hook environments
may need network access and tool downloads on their first run.

## Choose the scope

| Invocation | What it runs |
| --- | --- |
| `pre-commit run markdownlint --files docs/development.md` | One hook on the named Markdown file |
| `pre-commit run ruff-check --files vllm_gr/plugin.py` | One hook on the named Python file; may apply fixes |
| `pre-commit run --files <paths>` | Every applicable hook for those paths, including `run-tests` |
| `pre-commit run` | Applicable hooks for staged files, including the test hook when selected |
| `pre-commit run --all-files` | All applicable hooks, including the full slow test suite |
| `SKIP=run-tests pre-commit run --all-files` | Other applicable hooks; explicitly omits pytest coverage |

The local `run-tests` hook calls `tools/pre_commit/run_tests.sh`, which executes:

```bash
python3 -m pytest -x --run-slow --durations=10 -s tests
```

It uses `pass_filenames: false`, so selecting changed files does not restrict
pytest to those files. Use explicit hook IDs for documentation-only checks.
See [validation levels](tools/README.md#validation-levels) for targeted tests,
hardware requirements, and the platform-specific behavior of `tools/run_test.sh`.

## Documentation checks

For a Markdown edit, run the relevant hooks on the changed files. This example
checks one file; pass the complete changed Markdown file list for a real change:

```bash
pre-commit run markdownlint --files docs/development.md
pre-commit run check-characters --files docs/development.md
pre-commit run typos --files docs/development.md
pre-commit run end-of-file-fixer --files docs/development.md
pre-commit run trailing-whitespace --files docs/development.md
pre-commit run mixed-line-ending --files docs/development.md
git diff --check
```

Also verify relative links and referenced commands against the working tree;
Markdown linting does not establish that a link or test target exists. New
untracked files must be included explicitly in the hook commands; `git diff`
alone does not review their contents.

## Repository checks

Inspect `.pre-commit-config.yaml` for hook IDs, file filters, fix behavior, and
project-specific checks; formatting settings live in `pyproject.toml`. New
shared documentation must pass the character check. Generated gRPC bindings
are excluded by the hook configuration. The commit-message hook supplies DCO
sign-off using the configured Git identity.

## Failures and maintenance

When a hook modifies files, inspect the diff and rerun that hook. For an
environment failure, record the missing dependency or runtime and distinguish
it from a code/test failure. If a hook is skipped during local diagnosis, state
that explicitly; a skipped test hook does not establish a full validation pass.

Run `pre-commit autoupdate` only for an intentional hook-version update and
review the resulting configuration diff. It changes pinned versions; it is not
needed merely because the configuration changed. If an environment is corrupt,
`pre-commit clean` removes the local hook cache, and the next run rebuilds it.
Use that only when cache repair is needed.

CI invokes pre-commit in its prepared runtime and runs additional benchmark
and system gates. Local hook success does not replace those checks; see
[Contributing](../CONTRIBUTING.md#pull-request-ci).
