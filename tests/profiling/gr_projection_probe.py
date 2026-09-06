# SPDX-License-Identifier: Apache-2.0
"""Isolated first-layer replay using the exact cached input-norm kernel.

Diagnostic only: executes trusted local Inductor output, loads checkpoint
weights, and runs no Scheduler, KV cache, or Beam operation. This is a frozen
OneRec-1.7B/1024-to-16-row reproduction, not a generic model runner.
Run with ``CUDA_VISIBLE_DEVICES=1 python -m tests.profiling.gr_projection_probe``.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from tests.profiling.gr_candidate_oracle import decode_record, require_validation_enabled

COMPILED_MODULE_SHA256 = "765e9986cfe7e655c7a5426423d2ad5d3cc29c915df9f4476d44acff76749acd"


def difference(left, right):
    assert left.shape == right.shape
    delta = (left.float() - right.float()).abs()
    return {
        "elements": left.numel(),
        "different_elements": int((left != right).sum()),
        "max_abs_difference": float(delta.max()),
    }


def main():
    require_validation_enabled()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiled-module",
        type=Path,
        required=True,
        help="trusted, frozen first-partition generated Python module",
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    module_path = args.compiled_module
    assert hashlib.sha256(module_path.read_bytes()).hexdigest() == COMPILED_MODULE_SHA256, (
        "compiled module differs from the frozen reproduction"
    )
    spec = importlib.util.spec_from_file_location("step5_exact_first_partition", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    capture_path = args.capture
    capture = json.loads(capture_path.read_text())
    assert capture["status"] == "complete" and capture["performance_evidence_valid"] is False
    assert len(capture["candidate_oracle_diagnostics"]) == 2
    # Decode only first-layer fields rather than the entire layerwise payload.
    records = [
        decode_record(
            {
                "diagnostic_payload_base64": encoded["diagnostic_payload_base64"],
                "forward": encoded["forward"][0],
                "attention": encoded["prefill_attention"][0],
            }
        )
        for encoded in capture["candidate_oracle_diagnostics"]
    ]
    ids_full = torch.tensor(records[0]["forward"]["input_ids"], dtype=torch.int32, device="cuda")
    ids_hit = torch.tensor(records[1]["forward"]["input_ids"], dtype=torch.int32, device="cuda")
    assert len(ids_full) == 1024 and len(ids_hit) == 16, "probe requires the frozen 1024/16 case"
    assert records[0]["forward"]["computed_prefix_tokens"] == 0
    assert records[1]["forward"]["computed_prefix_tokens"] == 1008
    assert all(
        record["attention"]["layer"] == "model.layers.0.self_attn.attn" for record in records
    )
    assert torch.equal(ids_full[-len(ids_hit) :], ids_hit)
    model = args.model
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    weight_digests = {}

    def weight(name):
        with safe_open(model / index[name], framework="pt", device="cpu") as handle:
            host = handle.get_tensor(name)
            weight_digests[name] = {
                "sha256": hashlib.sha256(host.view(torch.uint8).numpy().data).hexdigest(),
                "shape": list(host.shape),
                "dtype": str(host.dtype),
            }
            return host.to(device="cuda", dtype=torch.bfloat16)

    embedding = weight("model.embed_tokens.weight")
    norm = weight("model.layers.0.input_layernorm.weight")
    projection = torch.cat(
        [weight(f"model.layers.0.self_attn.{name}_proj.weight") for name in ("q", "k", "v")]
    )
    assert embedding.shape == (176384, 2048) and projection.shape == (4096, 2048)
    stream = torch._C._cuda_getCurrentRawStream(0)

    def normalize(ids):
        embedded = torch.empty((len(ids), 2048), dtype=torch.bfloat16, device="cuda")
        normalized = torch.empty_like(embedded)
        module.triton_red_fused__to_copy_embedding_rms_norm_0.run(
            ids, embedding, norm, embedded, normalized, len(ids), 2048, stream=stream
        )
        return embedded, normalized

    full_embedding, full_input = normalize(ids_full)
    hit_embedding, hit_input = normalize(ids_hit)
    # These are the same mm shape/weight strides as the cached first partition.
    full_qkv = torch.empty((len(ids_full), 4096), dtype=torch.bfloat16, device="cuda")
    hit_qkv = torch.empty((len(ids_hit), 4096), dtype=torch.bfloat16, device="cuda")
    module.extern_kernels.mm(full_input, projection.t(), out=full_qkv)
    module.extern_kernels.mm(hit_input, projection.t(), out=hit_qkv)
    replay_qkv = torch.empty_like(hit_qkv)
    # Hold the normalized values fixed: eliminate independently computed norm
    # inputs as an explanation for any full-M vs suffix-M difference.
    module.extern_kernels.mm(
        full_input[-len(ids_hit) :].contiguous(), projection.t(), out=replay_qkv
    )
    observed = [
        torch.tensor(
            record["attention"]["value_tail"], dtype=torch.bfloat16, device="cuda"
        ).reshape(16, 1024)
        for record in records
    ]
    report = {
        "scope": "isolated compiled input normalization plus dense projection; no attention/Beam/KV",
        "performance_evidence_valid": False,
        "module": str(module_path),
        "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "weights": weight_digests,
        "matmul_precision": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        },
        "full_rows": len(ids_full),
        "hit_rows": len(ids_hit),
        "embedding_tail": difference(full_embedding[-16:], hit_embedding),
        "normalized_input_tail": difference(full_input[-16:], hit_input),
        "raw_qkv_tail": difference(full_qkv[-16:], hit_qkv),
        "fixed_input_full_vs_suffix_projection": difference(full_qkv[-16:], replay_qkv),
        "independent_vs_fixed_suffix_projection": difference(hit_qkv, replay_qkv),
        "value_full_matches_model_capture": difference(full_qkv[-16:, 3072:], observed[0]),
        "value_hit_matches_model_capture": difference(hit_qkv[:, 3072:], observed[1]),
    }
    # FP64 accumulation is a numerical reference for the projection alone,
    # not a replacement model or a relaxed output acceptance threshold.
    fp64 = full_input[-16:].double() @ projection.double().t()
    rounded = fp64.to(torch.bfloat16)
    report["full_projection_vs_fp64_rounded"] = difference(full_qkv[-16:], rounded)
    report["hit_projection_vs_fp64_rounded"] = difference(hit_qkv, rounded)
    report["value_shape_difference"] = difference(full_qkv[-16:, 3072:], hit_qkv[:, 3072:])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
