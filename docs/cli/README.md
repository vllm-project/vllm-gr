# CLI reference

Use the `vllm` executable with `--gr` to select a vLLM-GR command.

## Commands

| Command | Purpose |
| --- | --- |
| `vllm serve ... --gr` | Serve a GR model over HTTP, or over gRPC with `--grpc`. See [serve](serve.md). |
| `vllm bench ... --gr` | Run a benchmark workload. See [benchmarks](bench.md). |
| `vllm check-env --gr` | Report the platform and check required runtime versions. See [check-env](check_env.md). |

The GR configuration keys used by these commands are documented in
[configuration](../configuration/README.md). This section describes the command
surface only.

## Command form and installation

<a id="enable-gr-dispatch"></a>
<a id="enable-vllm-gr-on-vllm-0300"></a>

### Enable GR dispatch

Apply the [CLI dispatch patch](patches/vllm-0.30.0-gr-cli.patch) to the
[supported vLLM installation](../getting_started/installation.md#runtime-compatibility).
Activate the target environment and set `gr_checkout` to the absolute path of
your vLLM-GR checkout. Use `python -m pip show vllm` to locate the installed
package.

For a wheel installation, set the patch target to the active interpreter's
site-packages directory.

```bash
patch_target=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
```

For an editable installation, set `patch_target` to the vLLM source root
shown as the editable project location. With the target set, check and apply
the patch.

```bash
gr_cli_patch="$gr_checkout/docs/cli/patches/vllm-0.30.0-gr-cli.patch"
patch --dry-run --forward -d "$patch_target" -p1 < "$gr_cli_patch" &&
  patch --forward -d "$patch_target" -p1 < "$gr_cli_patch"
```

Reapply the patch if a vLLM reinstall replaces the patched files. Put
`--gr` after the subcommand and before the optional `--` separator.
Commands without `--gr` use the native vLLM CLI.

## Help, versions, and diagnostics

GR commands use the shared [ModelRunner defaults](../configuration/README.md#modelrunner-selection).
On Ascend NPU, remove any inherited V2 override or explicitly set
`VLLM_USE_V2_MODEL_RUNNER=0` before running the serve help commands.

```bash
vllm --help                                      # Native command list
vllm --version                                   # Native vLLM version
vllm serve --gr --help
vllm serve --gr --help=all                        # All installed serving options
vllm bench --gr --help
vllm bench --gr --list
vllm bench qwen-v1 --gr --help
vllm check-env --gr                               # Detect the runtime platform
vllm check-env --gr --npu                         # Require Ascend NPU
```

The benchmark list and GR top-level help do not import the serving command.
`check-env` checks versions and the selected platform without importing the
serving command's optional dependencies.
