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
platform that vLLM auto-detection selected. This GR release supports vLLM
0.31.0.

With `--npu`, or when auto-detection selects `npu`, the command additionally
validates the Ascend runtime:

| Item | Required |
| --- | --- |
| `vllm` | 0.31.0 |
| `vllm-ascend` | 0.31.0rc1 |
| `triton-ascend` | 3.2.2 |
| `triton` module | Provides `triton._C.libtriton.ascend` |

The Triton check imports the module that Python actually resolves. Installing
the `triton-ascend` distribution is not sufficient on its own: a standard
`triton` wheel installs the same top-level package and can overwrite it.

## Output and exit status

On success the command prints a header followed by one `name: value` line per
validated item:

```text
vLLM-GR environment check passed
  vllm: 0.31.0
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
[installation](../getting_started/installation/README.md) and
[dependency policy](../contributing/dependency_policy.md).
