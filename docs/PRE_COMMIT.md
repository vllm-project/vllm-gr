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

## CUDA stage hooks contract

`CudaStageHooks` in `vllm_gr/v1/worker/gpu_stage_hooks.py` is an abstract base
class. Both CUDA runners subclass it, so an implementation that leaves a
required operation or policy abstract raises `TypeError` when the hooks object is
constructed, before any dispatch, on any device, with no type checker involved.
`validate_stage_hooks_type` additionally enforces the lifetime rule that one
Worker keeps one concrete hooks type.

The class-side `CudaStageHooksType` Protocol checks construction with one Worker
argument and the admission sizing operation. A `type[CudaStageHooks]` annotation
alone does not reject an incompatible subclass constructor. The runtime ABC
checks completeness; mypy checks signatures, return types and policy types.

The dedicated `cuda-stage` group in `tools/pre_commit/mypy.py` checks the contract,
both implementations, the shared core, and all execution/startup binding modules
together. Changing any member or the hook script selects this complete boundary,
even with `--follow-imports skip`. Native internals outside it remain dynamic.
`tests/tools/cuda_stage_typing_cases.py` supplies valid V1/V2 bindings and invalid
return, constructor and policy types, including the output-completion return
type. Narrow expected-error annotations combined with `--warn-unused-ignores`
make a weakened contract fail the same checker run.
The group disables both cache reads and writes to avoid a mypy 1.11.1 placeholder
serialization failure in these cyclic imports.
It retains mypy 1.11.1 in the tests/CUDA-stage hooks. The Beam/Engine hooks use
their newer checker with `--skip-cuda-stage`; file filters still select the
legacy hooks when only a production stage module or the hook script changes.

Runtime and checker-routing regressions live in `tests/tools/test_cuda_stage_typing.py`.
They remain CPU-only and do not launch checker subprocesses. The ordinary test
group skips production imports, so runtime test subclass declarations use local
`type: ignore[misc]` annotations for the inherited-`Any` diagnostic. The production
boundary is checked separately by each Python-version mypy pre-commit hook.

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

## Staged mypy coverage

`strict = true` in `pyproject.toml` applies to files actually selected by
`tools/pre_commit/mypy.py`; it does not mean the entire runtime is type checked.
The selector now matches the `vllm_gr` path component (including nested Python
files), rather than interpreting a shell glob as a regular expression. Obsolete
upstream directory groups and exclusions have been removed.

The default hook checks supplied test files, the hook itself, and every Python
module and stub under `vllm_gr/v1/beam` and `vllm_gr/v1/engine` (31 Python files
at initial onboarding). Newly added files in those directories are included
automatically; there is no per-file allowlist. Both CI and local hooks use
`--follow-imports skip --ignore-missing-imports`, so imported modules and missing
third-party packages are not fully checked. NumPy 2.2.6 (the CPU/CUDA requirement)
and msgspec are installed in the isolated hook environment to check array and
wire-structure types. The hook pins mypy 2.4.1: the previous 1.11.1 crashes while
serializing NumPy 2.2.6's type cache. The accelerator-dependent vLLM package is not installed:
its request base class has a single documented subclass exemption, and native
patch receivers/call-through values retain explicit opaque types. Locally owned
contracts and control flow are checked. Strict diagnostics are not globally
disabled. Python 3.11, 3.12, and 3.13 hooks use the same selection.

The `mypy-3.*` hooks check runtime files and the selector in that environment.
The `mypy-tests-3.*` hooks retain the previous test environment (mypy 1.11.1 and
its existing dependency set). This keeps existing test diagnostics stable while
onboarding the core; migrating the entire test suite's checker is a separate
change. Both groups still enforce strict checking on the supplied files.

Runtime files outside this first batch are printed as deferred. To audit them,
pass `--all-core` before an explicit list of tracked Python paths:

```bash
python tools/pre_commit/mypy.py 0 local --all-core vllm_gr/constraints/startup.py
```

This audit may fail on existing typing debt in other runtime subsystems. To
expand enforcement, fix a directory's errors, add it to `CORE_DIRS`, and run all
three mypy hooks on its tracked Python files. Beam and Engine are already
enforced in full; the initial 136 diagnostics are no longer deferred.
