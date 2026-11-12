# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional, prompt-free artifacts for comparing Recall runs across devices."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def token_digest(tokens):
    """Hash the compact JSON representation of the actual input token IDs."""
    payload = json.dumps([int(token) for token in tokens], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_file_manifest(model):
    """Record all regular snapshot files, following HF's file symlinks."""
    root = Path(model)
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[path.relative_to(root).as_posix()] = {
            "sha256": digest.hexdigest(),
            "size": path.stat().st_size,
        }
    return result


def prefix_environment():
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "VLLM_BATCH_INVARIANT",
        "VLLM_GR_PREFIX_POLICY",
        "VLLM_GR_PREFIX_CALIBRATION",
        "VLLM_GR_PREFIX_CALIBRATION_PATH",
        "XDG_CACHE_HOME",
        "VLLM_ENABLE_PREFILL_CUDAGRAPH",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_GR_TARGET_DEVICE",
        "VLLM_GR_PREFILL_GRAPH_BUCKETS",
        "VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS",
        "VLLM_GR_PREFILL_GRAPH_KV_BOUND",
        "VLLM_GR_V1_STARTUP_CAPTURE_BATCH_SIZE",
    )
    values = {key: os.environ.get(key) for key in keys}
    override = values["VLLM_GR_PREFIX_CALIBRATION_PATH"]
    base = values["XDG_CACHE_HOME"] or str(Path.home() / ".cache")
    path = (
        Path(override).expanduser() if override else Path(base) / "vllm_gr/prefix_calibration.yaml"
    )
    if path.is_file():
        values["prefix_calibration_file_sha256_at_observation"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    else:
        values["prefix_calibration_file_sha256_at_observation"] = None
    return values


def effective_attention(log_text):
    """Extract actual backend choices reported by BeamAttention at startup."""
    pattern = re.compile(
        r"Beam attention: requested prefill=(\S+), effective prefill FA(\d+), decode FA(\d+)"
    )
    matches = sorted(set(pattern.findall(log_text)))
    return [
        {"requested_prefill": requested, "prefill_fa": int(prefill), "decode_fa": int(decode)}
        for requested, prefill, decode in matches
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matches = effective_attention(args.run_log.read_text(encoding="utf-8", errors="replace"))
    args.output.write_text(
        json.dumps({"effective_attention": matches}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
