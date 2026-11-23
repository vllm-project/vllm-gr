# Update checkpoints and Tries with Sidecar

Use Sidecar to roll out new checkpoints or Trie tables to running
[gRPC instances](../online_serving/grpc/README.md#serve-onerec-over-grpc).
Each instance exposes a separate HTTP management endpoint at its configured
`gr.admin_port`. For example, add `--gr.admin_port 9100` to a gRPC server
command and keep its inference port on `--port`. Online Trie switching
requires CUDA asynchronous Beam Search.

## Configure the instances and storage

Install vLLM-GR with the gRPC extra. Sidecar needs an HDFS client with
credentials for the source directories. Mount the shared cache at the same
absolute path in Sidecar and every target service container.

When using the default checkpoint checker, start the gRPC instances from an
initial checkpoint directory with the published file layout, including the
`_SUCCESS` marker. The checker compares relative file names and sizes,
including the marker, between the current and the new checkpoint.

From the repository root, copy the [configuration template](config.py):

```bash
cp examples/sidecar/config.py ./sidecar.local.py
```

Edit these fields in your copy:

- Replace the eight sample `Instance` entries with your running instances.
  Set each control URL to the host and `gr.admin_port` reachable from Sidecar.
- Set each task's `address` to the HDFS directory containing its checkpoint
  or Trie versions.
- Set each `target_dir` to a separate directory on the shared cache. Use one
  Sidecar process for these targets. Its default checkers also need to read
  the checkpoint and Trie paths currently reported by the services.

## Start Sidecar and publish a version

```bash
vllm-gr-sidecar --config ./sidecar.local.py
```

Leave it running to poll each task. To perform one discovery and update pass,
run the command with `--once`. This mode applies any eligible update before
exiting. A pass can also succeed when there is no newer ready version.

Upload each version into its own HDFS directory, such as
`ckpt_1700000000` or `trie_20261106_120000`. Write `_SUCCESS` last, after
all files have arrived. See the [publication guide](../../docs/user_guide/weight_update.md#publish-a-version-and-inspect-status)
for the required contents and the default checkers.

## Verify the rollout

Query each target's actual management URL, for example:

```bash
curl -fsS http://127.0.0.1:9100/service/status
```

After the rollout, each target should report `service_state` as `"running"`
and `checkpoint_valid` as `true`. Compare `checkpoint_version` or
`trie_version` with the HDFS version you published for that task.

Deployment and recovery details are available in the
[checkpoint and Trie hot-update guide](../../docs/user_guide/weight_update.md#user-startup-and-publication-guide).
