# Contributing to vLLM-GR

## Development setup

Use the [CUDA guide](docs/quickstart.md) or
[Ascend guide](docs/quickstart-ascend.md) to prepare the platform runtime and
install the editable package with development dependencies. Follow the Python
range in `pyproject.toml` and set `VLLM_GR_TARGET_DEVICE` for the intended
platform. The Ascend path relies on the vendor runtime; a CPU package install
does not provide accelerator inference.

Install the [pre-commit hooks](docs/PRE_COMMIT.md). Before runtime validation,
confirm that the active interpreter imports this checkout, then run:

```bash
vllm-gr check-env
```

Use `vllm-gr check-env --npu` for Ascend. This checks compatibility in an
installed runtime; it does not run inference. Keep model and dataset paths in
the documented environment variables.

## Change and validation workflow

1. Use [development navigation](docs/development.md) to locate the owning code,
   callers, and contract. Keep the change focused and preserve unrelated work.
2. Update behavior, regression coverage, and the owning documentation together.
   Public changes also need the relevant examples and configuration updated.
3. Select checks using [validation levels](docs/tools/README.md#validation-levels).
   Follow its regression guidance and use hardware coverage for device behavior.
   For documentation-only work, use the
   [documentation checks](docs/PRE_COMMIT.md#documentation-checks).
4. Review the final diff and fill out the
   [PR template](.github/PULL_REQUEST_TEMPLATE.md) with the problem, resulting
   behavior, exact validation commands, results, and remaining gaps.

Keep passed, failed, skipped, and unrun checks distinct. The validation guide
describes the evidence needed for hardware and benchmark claims.

## AI coding tools

Project rules live in [AGENTS.md](AGENTS.md), with a shared-file import in
[CLAUDE.md](CLAUDE.md). See
[AI tool integration and instruction maintenance](docs/agent_instructions.md)
for loading checks, other tools, and where to add new guidance. Contributors
remain responsible for reviewing generated changes and their validation.

## Pull request CI

The [validation workflow](.github/workflows/pre-commit.yml) and
[wheel workflow](.github/workflows/build_wheel.yml) target PRs into `main` and
`decode_graph`. Full PR validation requires the `ready` label and a non-draft
PR. Wheel builds also apply their path filters. Run local checks before asking
maintainers to enable CI.

Adding `ready`, pushing while it remains, or marking a labeled draft ready for
review starts validation. Removing `ready` or converting the PR to draft
cancels its previous PR runs and skips new validation. Remove and re-add
`ready` to request a fresh run. Unrelated labels do not start full validation;
manual dispatches have their own job conditions.

Keep `Benchmark blocking gate` required in branch protection. It rejects
failed, cancelled, or skipped blocking jobs; a skipped run is not successful
validation. Cancelling CI also stops Pangu jobs, whose cleanup steps still run
after device mapping has succeeded.

For non-draft PRs into `decode_graph`, adding `abtest` requests a separate
Beam V1 performance A/B run. Remove and re-add it to rerun. Its success alone
does not satisfy full validation. A full `ready` run reuses an eligible A/B
result for the current revision or runs A/B when needed.

Benchmark policy depends on the PR base and device. Video-100/Pangu policy is
resolved from the PR base; Recall baseline provenance has separate rules.
Use the [benchmark guide](docs/tools/README.md#benchmark-acceptance) and inspect
the target revision's workflows when interpreting a result.
