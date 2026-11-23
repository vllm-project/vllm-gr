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

### Enable `vllm ... --gr` on vLLM 0.30.0

The native `vllm` executable is provided by vLLM. Apply the
[vLLM 0.30.0 CLI patch](patches/vllm-0.30.0-gr-cli.patch) to enable
GR dispatch. With the target virtual environment active, set `gr_checkout` to
the absolute path of this GR checkout. For a wheel installation of vLLM 0.30.0,
run:

```bash
python -m pip show vllm
site_packages=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
gr_cli_patch="$gr_checkout/docs/cli/patches/vllm-0.30.0-gr-cli.patch"
patch --dry-run --forward -d "$site_packages" -p1 < "$gr_cli_patch" &&
  patch --forward -d "$site_packages" -p1 < "$gr_cli_patch"
```

For an editable vLLM 0.30.0 installation, use the vLLM source root in place of
`$site_packages`. Reapply the patch after reinstalling vLLM 0.30.0, if needed.
Commands without the GR switch continue through vLLM's normal CLI.

!!! important "GR dispatch and forwarded options"

    Apply the patch in the active vLLM environment. Put `--gr` after the
    subcommand and before the optional `--` separator. When upgrading a
    deployment, update its launch scripts to the forms below.

## Help, versions, and diagnostics

On Ascend, select ModelRunner V1 with `export VLLM_USE_V2_MODEL_RUNNER=0` before
running the serve help commands.

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
