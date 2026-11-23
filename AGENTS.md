# Working in vLLM-GR

These instructions apply throughout this checkout. [CLAUDE.md](CLAUDE.md)
imports this file so supported tools share the same project rules.

## Find the relevant guidance

vLLM-GR extends an installed vLLM with generative-recommendation inference.
Sibling vLLM, vLLM-Ascend, and vLLM-GR checkouts are references unless the task
explicitly targets them; they are not necessarily the imported runtime.

- For setup and PR workflow, read [CONTRIBUTING.md](CONTRIBUTING.md).
- Before changing runtime behavior, use [development navigation](docs/contributing/development.md)
  to find the owning code and read the relevant linked contract.
- Before selecting tests, read [validation levels](docs/contributing/ci/test_levels.md#validation-levels)
  and the affected tests' fixtures and device requirements.
- For documentation-only changes, use the
  [documentation checks](docs/contributing/pre_commit_setup.md#documentation-checks).
- When changing agent guidance or tool integration, read
  [AI instruction maintenance](docs/contributing/agent_instructions.md).

## Environment

Follow the [CUDA](docs/getting_started/installation/gpu.md) or [Ascend](docs/getting_started/installation/npu.md)
installation guide in an environment for the target device. Set
`VLLM_GR_TARGET_DEVICE` explicitly for platform-specific installs or wheels.
The deprecated `[gpu]`/`[npu]` extras do not provision those runtimes.

Dependency and Python constraints belong to `pyproject.toml`, `setup.py`,
and `requirements/`. Built-in runtime version pins live in
`vllm_gr/platforms/compat.py`. Preserve these constraints unless compatibility
is part of the task. In particular, the NPU requirements depend on a prepared
vendor runtime. Confirm the interpreter and import origin before runtime work:

```bash
python --version
python -c 'import vllm_gr; print(vllm_gr.__file__)'
```

## Editing

- Follow `pyproject.toml` and `.pre-commit-config.yaml` for formatting, headers,
  and import restrictions. Do not substitute upstream vLLM's tool settings.
- Write shared documentation in English and use characters accepted by
  `tools/pre_commit/check_english.py`; conversation language is unrestricted.
- Generate `vllm_gr/entrypoints/grpc/generated/` from its proto sources using
  `tools/codegen/generate_grpc_proto.py`. Do not hand-edit it or `vllm_gr/_version.py`.
- When changing runtime patch targets, ownership, activation, or removal, update
  both `docs/design/runtime_patch_inventory.md` and `docs/design/patch_inventory.yaml`, with
  the corresponding contract coverage.
- Keep transient task notes and machine-specific model/data paths out of shared
  instructions. Treat historical measurements as evidence for their recorded
  revision and configuration.

## Validation and reporting

`pre-commit run --all-files` includes the full pytest suite with `--run-slow`.
Use explicit hook IDs for documentation changes. `tools/run_test.sh` selects an
accelerator and may build NPU operators; check its documented prerequisites.

Host-oriented tests can still require Torch and vLLM through
`tests/conftest.py`. A CPU wheel or reference implementation does not establish
a production CPU Beam runtime. Choose regression coverage for the changed
behavior and use device tests when it affects kernels, inference, or graphs.

Report exact checks and outcomes, including failures, skips, and unrun coverage.
For hardware results include the runtime, device, model, runner, graph mode,
and output location. Keep functional, graph replay, accuracy, and performance
claims separate; preserve the accepted benchmark policy and input identity.
