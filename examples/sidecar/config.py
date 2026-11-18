# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Composition example for servers implementing the sidecar control contract.

Run: python -m vllm_gr.sidecar --config examples/sidecar/config.py
gRPC servers: set VLLM_GR_ADMIN_PORT to the configured control port.
Hot updates are not exposed by the HTTP inference server.
Online Trie switching currently requires CUDA asynchronous beam_search_v1.
Checkpoint history defaults to two local copies, oldest downloads removed first.
Referenced and updating versions are protected and may temporarily exceed the limit.
Each Task cache directory must be owned by one Sidecar process.
"""

from pathlib import Path

from vllm_gr.sidecar import (
    HDFSDownloader,
    Instance,
    OnlineUpdateProcess,
    PausedUpdateProcess,
    RollingUpdate,
    SidecarConfig,
    SyncUpdate,
    UpdateTask,
    default_ckpt_checker,
    default_trie_checker,
)

config = SidecarConfig(
    instances=[Instance(f"node-{i}", f"http://127.0.0.1:{9100 + i}") for i in range(8)],
    tasks=[
        UpdateTask(
            name="ckpt",
            resource_type="checkpoint",
            checker=default_ckpt_checker,  # Optional; replace with your own (old_path, new_path) checker.
            filesystem=HDFSDownloader(),
            address="hdfs://cluster/models/gr/checkpoints",
            target_dir=Path("/updates/checkpoints"),
            max_local_checkpoints=2,  # Keep in-use versions even when temporarily over this limit.
            update_mode=RollingUpdate(max_concurrent_updates=1),
            update_process=PausedUpdateProcess(),
        ),
        UpdateTask(
            name="trie",
            resource_type="trie",
            checker=default_trie_checker,  # Omitting checker selects this default.
            filesystem=HDFSDownloader(),
            address="hdfs://cluster/models/gr/tries",
            target_dir=Path("/updates/tries"),
            poll_interval_seconds=60,
            update_mode=SyncUpdate(),
            update_process=OnlineUpdateProcess(),
        ),
    ],
)
