# Update model weights and catalogs on running instances

Use a sidecar to roll out model weights or replace a constraint catalog on
running GR instances in a shared Pod. For validation and recovery behavior,
see the [sidecar design](../design/feature/weight_update.md).

## Serving prerequisites

This workflow requires a **gRPC inference service with a separate HTTP
management listener**, enabled by `gr.admin_port`. Point Sidecar at the
management port. The HTTP `/v1/completions` inference entrypoint does not
start these update endpoints. Check the
[gRPC device support](feature_compatibility.md#support-matrix) before choosing
this deployment path; Ascend-specific gRPC validation remains undocumented.

| Resource | Update behavior |
| --- | --- |
| Checkpoint | Pause and drain each group of instances, load compatible weights, then resume. |
| Catalog | Switch the constraint table while the CUDA asynchronous Beam service stays online. |

<a id="user-configuration-example"></a>

## Configuration

The user provides a Python configuration file exporting a `config: SidecarConfig`
object. The framework loads that object and manages startup and execution; the
configuration file does not construct a running Sidecar or call `run()`.

FileSystem, Update Mode, and Update Process implementations are passed as Python
objects, so users can supply custom implementations directly. A runnable example
is [examples/sidecar/config.py](../../examples/sidecar/config.py).

Copy the linked configuration to your deployment directory and edit the
instance URLs, HDFS sources, and local cache paths described below.

- `instances` lists service control endpoints; its length is independent of the
  physical GPU count. Tasks can target `all` or a list of instance IDs.
- `filesystem` is an implementation object. The Task passes `address` to both
  version lookup and download, together with the selected version for download.
- `target_dir` is the Task's shared local cache root; Task cache roots must not
  overlap. Sidecar and all target services must see the same absolute paths.
- Omitting `checker` selects `default_ckpt_checker` or `default_trie_checker`.
  Both compare against the service's reported local `checkpoint_path`/`trie_path`, which must
  also be readable by Sidecar.
- `update_mode` controls instance scheduling; `update_process` controls each
  instance's actions. The initial combinations are checkpoint + RollingUpdate +
  PausedUpdateProcess and Trie + SyncUpdate + OnlineUpdateProcess.
- No `state_dir` is configured. Shared local storage holds downloaded resources;
  completed versions and readiness are queried from each service. Overlapping
  Tasks share an instance lock within the Sidecar process.

<a id="user-startup-and-publication-guide"></a>

## Usage

### Prepare the Pod

Install vLLM-GR with its gRPC extra in the serving environment and provide a working
HDFS client/credentials for Sidecar. `HDFS_BIN` optionally selects the client;
`HDFS_USES_DFS_SUBCOMMAND=1` selects the Hadoop `hdfs dfs -...` dialect when needed.

Mount a shared local directory such as `/updates` at the **same absolute path**
in Sidecar and every target service container. Prepare the initial local Hugging
Face checkpoint and, for constrained serving, the canonical Trie artifact selected
by the existing GR config. Use an initial published checkpoint directory including
`_SUCCESS`: the default checker compares all relative file names/sizes, so adding
or omitting that marker on only one side causes rejection.

### Start the gRPC instances

Keep the existing gRPC entrypoint and inference arguments. The required startup
addition is **`gr.admin_port`**, unique per instance. It enables a separate HTTP
control listener in that gRPC service process; it is not the inference port and
does not replace the gRPC server. Without it, no update listener is started.

For instance 0, a minimal example is:

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve /updates/bootstrap/ckpt_1700000000 --gr \
  --grpc \
  --config /etc/vllm-gr/gr-config.yaml --gr.admin_port 9100 \
  --host 127.0.0.1 --port 9000 \
  --async-scheduling --no-enable-chunked-prefill
```

Retain your deployment's existing model, capacity, dtype, compile and other
inference flags. For instance `i`, use its GPU assignment, inference port
`9000+i`, and control port `9100+i`; configure those control URLs in
`SidecarConfig.instances`. The example configuration targets eight instances;
change the list to the number actually started.

The control host defaults to `127.0.0.1`, suitable for containers sharing a Pod
network. `gr.admin_host` overrides it. Set
`gr.trie_retention_seconds` on each service to override the 60-second old-table
retention period.

### Start Sidecar

Save your copy of the example configuration, with your HDFS sources, shared cache paths,
and actual instance control URLs, for example as `/etc/vllm-gr/sidecar-config.py`.
Start **one Sidecar process** for those targets:

```bash
vllm-gr-sidecar --config /etc/vllm-gr/sidecar-config.py
```

Equivalent module invocation:

```bash
python -m vllm_gr.sidecar --config /etc/vllm-gr/sidecar-config.py
```

For one discovery/update pass instead of continuous polling:

```bash
vllm-gr-sidecar --config /etc/vllm-gr/sidecar-config.py --once
```

Normal mode polls each Task at its configured interval. `--once` processes each
Task once and exits; an update failure returns a nonzero exit status. Sidecar
restart uses the same config/cache and reconciles the live instances, without
loading a progress journal. Locks coordinate Tasks in this process, not multiple
independent Sidecars targeting the same instances.

### Publish a version and inspect status

HDFS sources contain immutable version directories. Default discovery accepts a
suffix of at least ten timestamp digits (for example `ckpt_1700000000`) or
`_YYYYMMDD_HHMMSS` (for example `trie_20261106_120000`). Upload all files first and
create `_SUCCESS` last. Incomplete newer versions are skipped; published version
contents must not be overwritten.

```text
checkpoints/
└── ckpt_1700000000/
    ├── config.json
    ├── model.safetensors
    ├── ... other checkpoint files
    └── _SUCCESS
tries/
└── trie_20261106_120000/
    ├── table.safetensors
    └── _SUCCESS
```

The model structure/configuration stays fixed. Default checkpoint validation
compares file lists/sizes, not weight contents; default Trie validation compares
tensor names/count, not tensor values or file sizes. The server then loads the
weights into the existing model; it does not re-check architecture or tokenizer
compatibility, so an incompatible checkpoint fails during native loading. Only
single-file checkpoint loading has been exercised in the recorded GPU tests.

Inspect each instance's committed version and readiness through its control port:

```bash
curl -fsS 'http://127.0.0.1:9100/service/status'
```

Status has no query parameters and returns both resources in one response:

```json
{
  "service_state": "running",
  "checkpoint_valid": true,
  "checkpoint_version": "ckpt_1700000000",
  "trie_version": "trie_20261106_120000",
  "checkpoint_path": "/updates/checkpoints/ckpt_1700000000",
  "trie_path": "/updates/tries/trie_20261106_120000/table.safetensors"
}
```

Version and path fields are independent of service state. The status
contract, including `checkpoint_valid` and which component owns it, is defined
in [the update lifecycle](../design/feature/weight_update.md#update-lifecycle).

Sidecar compares the relevant version, reads its path for validation, and checks
service state and checkpoint validity. A paused instance already on the target
version only needs resume if `checkpoint_valid=true`.
After an interrupted rollout, restart Sidecar or allow its next poll to reconcile
the remaining instances. Keep published resources and local serving paths
available for validation and loading.
