#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Estimate HSTU model memory requirements without launching a vLLM engine.

Usage Examples
==============

1. Basic estimation (only static weights + KV cache shape + sequence estimates):

    python examples/hstu_memory_estimator.py \
        --model /workspace/ckpt/movielens_1m/800_truncated_new/converted

2. Specify UIH sequence lengths:

    python examples/hstu_memory_estimator.py \
        --model /workspace/ckpt/movielens_1m/800_truncated_new/converted \
        --sequence-lengths 1024,2048,4096

3. Override max candidates count (defaults to config.json value):

    python examples/hstu_memory_estimator.py \
        --model /workspace/ckpt/movielens_1m/800_truncated_new/converted \
        --max-candidates 256

4. Rough fit check against device memory (64 GiB NPU, 8% utilization):

    python examples/hstu_memory_estimator.py \
        --model /workspace/ckpt/movielens_1m/800_truncated_new/converted \
        --sequence-lengths 1024 \
        --device-memory-gib 64 \
        --gpu-memory-utilization 0.08

5. Override KV cache dtype (defaults to config.json torch_dtype or bfloat16):

    python examples/hstu_memory_estimator.py \
        --model /workspace/ckpt/movielens_1m/800_truncated_new/converted \
        --kv-cache-dtype float16

6. Full estimation with JSON report output:

    python examples/hstu_memory_estimator.py \
        --model /workspace/ckpt/movielens_1m/800_truncated_new/converted \
        --sequence-lengths 1024,4096 \
        --max-candidates 256 \
        --block-size 128 \
        --device-memory-gib 64 \
        --gpu-memory-utilization 0.08 \
        --output-json /tmp/hstu_memory_report.json

7. Match benchmark parameters for capacity scenario validation:

    python examples/hstu_memory_estimator.py \
        --model /workspace/ckpt/movielens_1m/800_truncated_new/converted \
        --sequence-lengths 4096 \
        --max-candidates 256 \
        --block-size 128 \
        --device-memory-gib 64 \
        --gpu-memory-utilization 0.1

Arguments
---------

--model (required)
    Path to the converted HSTU model directory containing config.json and
    weight files (*.safetensors, *.bin, *.pt, *.pth).

--sequence-lengths (default: "1024,2048,4096,8192")
    Comma-separated UIH (User Interaction History) sequence lengths to estimate.
    For each length, the tool calculates visible sequence length as:
        contextual_seq_len + UIH_seq_len + max_candidates

--max-candidates (default: from config.json)
    Override the number of candidate items for inference. Defaults to
    config.max_num_candidates_inference, then config.max_num_candidates.

--block-size (default: 128)
    Paged KV cache block size (tokens per block). Must match the value used
    at runtime for accurate block count estimates.

--kv-cache-dtype (default: from config.json or bfloat16)
    Override KV cache element dtype. Affects per-token KV byte calculations.
    Supported: float32, bfloat16, float16, float8_e4m3fn, fp8, int8, etc.

--device-memory-gib (optional)
    Total device HBM in GiB. When provided, prints a rough fit check showing
    whether weights + KV cache fit within the engine budget
    (device_memory * gpu_memory_utilization).

--gpu-memory-utilization (default: 0.9)
    Fraction of device HBM allocated to the vLLM engine. Only used when
    --device-memory-gib is provided.

--output-json (optional)
    Path to write a JSON report with all estimation data.

Output Sections
---------------

[Static Weights]    - Weight file count and total size on disk.
[Embedding Tables]  - Per-table row count, dimension, dtype, and byte size.
[KV Cache Shape]    - Per-layer and all-layers per-token KV byte cost.
[Sequence Estimates] - Per-seq-len visible length, KV bytes, and block counts.
[Rough Fit Check]   - Whether weights + KV fit in engine budget (if device memory given).

Notes
-----

- This tool reads config.json and weight files but does NOT launch a vLLM engine.
- The fit check is a lower/upper bound estimate; it does not include activation
  memory, allocator fragmentation, compiler caches, or runtime reservations.
- HSTU runtime may only write UIH tokens into paged KV cache, but the scheduler-
  visible sequence length still scales with contextual + UIH + candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from runtime_utils import add_repo_root_to_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
add_repo_root_to_path(__file__)

from vllm_gr.config.hstu_config import get_hstu_contextual_seq_len  # noqa: E402

DEFAULT_SEQUENCE_LENGTHS = "1024,2048,4096,8192"
DEFAULT_BLOCK_SIZE = 128
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9

_WEIGHT_FILE_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
)

_DTYPE_NBYTES = {
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "fp8": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate HSTU memory usage from config.json and model weight files "
            "without starting the vLLM runtime."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Converted HSTU model directory that contains config.json.",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=str,
        default=DEFAULT_SEQUENCE_LENGTHS,
        help="Comma-separated UIH sequence lengths to estimate.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help=(
            "Override inference candidate count. Defaults to "
            "config.max_num_candidates_inference, then config.max_num_candidates."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Paged KV block size used for the estimate (default: 128).",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=None,
        help="Override KV cache dtype. Defaults to config.kv_cache_dtype or torch_dtype.",
    )
    parser.add_argument(
        "--device-memory-gib",
        type=float,
        default=None,
        help=(
            "Optional device total memory. When provided, the script prints a rough "
            "fit check against gpu_memory_utilization."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Runtime memory utilization target for the rough fit check (default: 0.9).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional JSON report output path.",
    )
    return parser.parse_args()


def _parse_sequence_lengths(raw_value: str) -> list[int]:
    values: list[int] = []
    for part in raw_value.split(","):
        normalized = part.strip()
        if not normalized:
            continue
        values.append(int(normalized))
    if not values:
        raise ValueError("--sequence-lengths must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("--sequence-lengths values must be > 0")
    return values


def _load_model_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under model dir: {model_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dtype_nbytes(dtype_value: Any) -> int:
    normalized = str(dtype_value).strip().lower()
    try:
        return _DTYPE_NBYTES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype for estimation: {dtype_value!r}") from exc


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def _find_weight_files(model_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in _WEIGHT_FILE_PATTERNS:
        files.extend(sorted(model_dir.glob(pattern)))
    return sorted(set(files))


def _sum_weight_file_bytes(weight_files: list[Path]) -> int:
    return int(sum(path.stat().st_size for path in weight_files))


def _resolve_inference_candidates(
    config: dict[str, Any],
    override_max_candidates: int | None,
) -> int:
    if override_max_candidates is not None:
        return int(override_max_candidates)
    inferred = int(config.get("max_num_candidates_inference", 0) or 0)
    if inferred > 0:
        return inferred
    return int(config.get("max_num_candidates", 0) or 0)


def _estimate_embedding_bytes(config: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    embedding_tables = list(config.get("embedding_tables", []) or [])
    default_dtype = config.get("embedding_table_data_type", config.get("torch_dtype", "float16"))
    rows: list[dict[str, Any]] = []
    total_bytes = 0

    for table in embedding_tables:
        name = str(table["name"])
        num_embeddings = int(table["num_embeddings"])
        embedding_dim = int(table["embedding_dim"])
        dtype_name = str(table.get("data_type", default_dtype))
        bytes_per_elem = _dtype_nbytes(dtype_name)
        table_bytes = num_embeddings * embedding_dim * bytes_per_elem
        total_bytes += table_bytes
        rows.append(
            {
                "name": name,
                "num_embeddings": num_embeddings,
                "embedding_dim": embedding_dim,
                "dtype": dtype_name,
                "bytes_per_elem": bytes_per_elem,
                "bytes": table_bytes,
                "feature_names": list(table.get("feature_names", [])),
            }
        )

    return total_bytes, rows


def _estimate_kv_bytes_per_token(
    config: dict[str, Any],
    kv_cache_dtype: str,
) -> dict[str, int]:
    num_layers = int(config.get("hstu_attn_num_layers", 0) or 0)
    num_kv_heads = int(config.get("hstu_num_heads", 0) or 0)
    head_size_k = int(config.get("hstu_attn_qk_dim", 0) or 0)
    head_size_v = int(config.get("hstu_attn_linear_dim", 0) or 0)
    bytes_per_elem = _dtype_nbytes(kv_cache_dtype)

    per_layer_bytes = num_kv_heads * (head_size_k + head_size_v) * bytes_per_elem
    total_bytes = per_layer_bytes * num_layers
    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_size_k": head_size_k,
        "head_size_v": head_size_v,
        "bytes_per_elem": bytes_per_elem,
        "per_layer_per_token_bytes": per_layer_bytes,
        "all_layers_per_token_bytes": total_bytes,
    }


def _build_sequence_estimates(
    *,
    config: dict[str, Any],
    sequence_lengths: list[int],
    max_candidates: int,
    block_size: int,
    kv_bytes_per_token: int,
) -> list[dict[str, Any]]:
    contextual_seq_len = int(get_hstu_contextual_seq_len(config))
    rows: list[dict[str, Any]] = []
    for seq_len in sequence_lengths:
        visible_seq_len = contextual_seq_len + seq_len + max_candidates
        full_visible_blocks = math.ceil(visible_seq_len / block_size) if block_size > 0 else 0
        uih_only_blocks = math.ceil(seq_len / block_size) if block_size > 0 else 0
        full_visible_kv_bytes = visible_seq_len * kv_bytes_per_token
        uih_only_kv_bytes = seq_len * kv_bytes_per_token
        rows.append(
            {
                "seq_len": seq_len,
                "contextual_seq_len": contextual_seq_len,
                "max_candidates": max_candidates,
                "visible_seq_len": visible_seq_len,
                "full_visible_blocks": full_visible_blocks,
                "uih_only_blocks": uih_only_blocks,
                "full_visible_kv_bytes": full_visible_kv_bytes,
                "uih_only_kv_bytes": uih_only_kv_bytes,
            }
        )
    return rows


def _build_fit_checks(
    *,
    device_memory_gib: float | None,
    gpu_memory_utilization: float,
    weight_file_bytes: int,
    sequence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if device_memory_gib is None:
        return []

    device_total_bytes = int(device_memory_gib * (1024**3))
    engine_budget_bytes = int(device_total_bytes * gpu_memory_utilization)
    rows: list[dict[str, Any]] = []
    for sequence_row in sequence_rows:
        lower_bound_total = weight_file_bytes + int(sequence_row["uih_only_kv_bytes"])
        upper_bound_total = weight_file_bytes + int(sequence_row["full_visible_kv_bytes"])
        rows.append(
            {
                "seq_len": int(sequence_row["seq_len"]),
                "engine_budget_bytes": engine_budget_bytes,
                "lower_bound_total_bytes": lower_bound_total,
                "upper_bound_total_bytes": upper_bound_total,
                "lower_bound_fits_budget": lower_bound_total <= engine_budget_bytes,
                "upper_bound_fits_budget": upper_bound_total <= engine_budget_bytes,
            }
        )
    return rows


def main() -> int:
    args = _parse_args()
    model_dir = Path(args.model).expanduser().resolve()
    sequence_lengths = _parse_sequence_lengths(args.sequence_lengths)
    config = _load_model_config(model_dir)

    max_candidates = _resolve_inference_candidates(config, args.max_candidates)
    kv_cache_dtype = str(
        args.kv_cache_dtype or config.get("kv_cache_dtype") or config.get("torch_dtype", "bfloat16")
    )
    weight_files = _find_weight_files(model_dir)
    weight_file_bytes = _sum_weight_file_bytes(weight_files)
    embedding_bytes, embedding_rows = _estimate_embedding_bytes(config)
    kv_shape = _estimate_kv_bytes_per_token(config, kv_cache_dtype)
    sequence_rows = _build_sequence_estimates(
        config=config,
        sequence_lengths=sequence_lengths,
        max_candidates=max_candidates,
        block_size=int(args.block_size),
        kv_bytes_per_token=int(kv_shape["all_layers_per_token_bytes"]),
    )
    fit_rows = _build_fit_checks(
        device_memory_gib=args.device_memory_gib,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        weight_file_bytes=weight_file_bytes,
        sequence_rows=sequence_rows,
    )

    report = {
        "model_dir": str(model_dir),
        "config_path": str(model_dir / "config.json"),
        "torch_dtype": str(config.get("torch_dtype", "unknown")),
        "kv_cache_dtype": kv_cache_dtype,
        "contextual_seq_len": int(get_hstu_contextual_seq_len(config)),
        "max_candidates": max_candidates,
        "block_size": int(args.block_size),
        "weight_file_count": len(weight_files),
        "weight_file_bytes": weight_file_bytes,
        "weight_files": [str(path) for path in weight_files],
        "embedding_bytes": embedding_bytes,
        "embedding_tables": embedding_rows,
        "kv_shape": kv_shape,
        "sequence_estimates": sequence_rows,
        "fit_checks": fit_rows,
    }

    print("=" * 72)
    print(" HSTU Memory Estimator")
    print("=" * 72)
    print(f"model={model_dir}")
    print(f"torch_dtype={report['torch_dtype']} kv_cache_dtype={kv_cache_dtype}")
    print(
        "layers={layers} heads={heads} head_size_k={head_size_k} head_size_v={head_size_v}".format(
            layers=kv_shape["num_layers"],
            heads=kv_shape["num_kv_heads"],
            head_size_k=kv_shape["head_size_k"],
            head_size_v=kv_shape["head_size_v"],
        )
    )
    print(
        "contextual_seq_len={contextual_seq_len} max_candidates={max_candidates} block_size={block_size}".format(
            **report
        )
    )

    print("\n[Static Weights]")
    print(f"  weight_files={len(weight_files)} total={_format_bytes(weight_file_bytes)}")
    if not weight_files:
        print("  warning: no weight files were found under the model directory.")

    print("\n[Embedding Tables]")
    print(f"  total={_format_bytes(embedding_bytes)}")
    for row in embedding_rows:
        print(
            "  {name}: rows={num_embeddings} dim={embedding_dim} dtype={dtype} bytes={bytes_str}".format(
                name=row["name"],
                num_embeddings=row["num_embeddings"],
                embedding_dim=row["embedding_dim"],
                dtype=row["dtype"],
                bytes_str=_format_bytes(int(row["bytes"])),
            )
        )

    print("\n[KV Cache Shape]")
    print(
        "  per_layer_per_token={per_layer} all_layers_per_token={all_layers}".format(
            per_layer=_format_bytes(int(kv_shape["per_layer_per_token_bytes"])),
            all_layers=_format_bytes(int(kv_shape["all_layers_per_token_bytes"])),
        )
    )
    print(
        "  note: HSTU runtime may only write UIH tokens into paged KV cache, "
        "but scheduler-visible sequence length still scales with contextual + UIH + candidates."
    )

    print("\n[Sequence Estimates]")
    for row in sequence_rows:
        print(
            "  seq_len={seq_len} visible={visible_seq_len} "
            "uih_only_kv={uih_only_kv} full_visible_kv={full_visible_kv} "
            "uih_blocks={uih_blocks} full_blocks={full_blocks}".format(
                seq_len=row["seq_len"],
                visible_seq_len=row["visible_seq_len"],
                uih_only_kv=_format_bytes(int(row["uih_only_kv_bytes"])),
                full_visible_kv=_format_bytes(int(row["full_visible_kv_bytes"])),
                uih_blocks=row["uih_only_blocks"],
                full_blocks=row["full_visible_blocks"],
            )
        )

    if fit_rows:
        print("\n[Rough Fit Check]")
        engine_budget_bytes = int(args.device_memory_gib * (1024**3) * args.gpu_memory_utilization)
        print(
            "  device_memory={device_memory} gpu_memory_utilization={util:.2f} engine_budget={budget}".format(
                device_memory=_format_bytes(int(args.device_memory_gib * (1024**3))),
                util=args.gpu_memory_utilization,
                budget=_format_bytes(engine_budget_bytes),
            )
        )
        print(
            "  note: this is only a lower/upper bound using static weights + KV cache; "
            "it does not include temporary activations, allocator fragmentation, compiler caches, or runtime reservations."
        )
        for row in fit_rows:
            print(
                "  seq_len={seq_len} lower_bound={lower_bound} fits_lower={fits_lower} "
                "upper_bound={upper_bound} fits_upper={fits_upper}".format(
                    seq_len=row["seq_len"],
                    lower_bound=_format_bytes(int(row["lower_bound_total_bytes"])),
                    fits_lower=row["lower_bound_fits_budget"],
                    upper_bound=_format_bytes(int(row["upper_bound_total_bytes"])),
                    fits_upper=row["upper_bound_fits_budget"],
                )
            )

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
