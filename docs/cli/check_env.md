# vllm check-env

`vllm check-env --gr` checks the installed vLLM version and reports the
selected platform. On Ascend it also checks the vendor runtime packages.

## Usage

```bash
vllm check-env --gr          # Validate the selected platform
vllm check-env --gr --npu    # Require the Ascend NPU platform
```

## What it checks

Without `--npu`, the command validates the vLLM version and reports the
platform that vLLM auto-detection selected. See
[runtime compatibility](../getting_started/installation.md#runtime-compatibility)
for the supported package versions.

With `--npu`, or when auto-detection selects `npu`, the command also checks
`vllm-ascend` and `triton-ascend`. It imports the active `triton` module
to verify that it exposes `triton._C.libtriton.ascend`. A standard `triton`
wheel can replace the top-level module installed by `triton-ascend`, so the
check probes the module that Python actually loads.

## Output and exit status

On success the command prints a header followed by one line per validated
item. A CUDA report has this shape, where `<installed-version>` is the
resolved vLLM version.

```text
vLLM-GR environment check passed
  vllm: <installed-version>
  platform: cuda
```

A version mismatch or a missing or broken Ascend runtime raises an error with
the found and required versions, and the command exits non-zero. `--npu` on a
host whose auto-detection selects another device fails the platform check.

## Scope

The check covers the runtime pair for the selected platform only. Unselected
vendor runtimes are not validated, and the command does not import the serving
command's optional dependencies, such as the gRPC extra.

For the package contract behind these versions, see
[installation](../getting_started/installation.md) and
[dependency policy](../contributing/dependency_policy.md).
