# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert an original RECIF checkpoint into a runtime HF checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

VOCAB = 8192
NUM_SID_LEVELS = 3
FULL_VOCAB = NUM_SID_LEVELS * VOCAB
RECIF_ARCHITECTURE = "RecifForCausalLM"
ARCHITECTURE_KEYS = (
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_size",
    "num_experts",
)


class InsufficientDiskSpaceError(RuntimeError):
    """Raised when the output filesystem cannot hold the converted checkpoint."""


def architecture_from_config(config: Mapping[str, Any]) -> dict[str, int]:
    architecture: dict[str, int] = {}
    for key in ARCHITECTURE_KEYS:
        value = config.get(key)
        if value is None:
            raise ValueError(f"RECIF config is missing required field: {key}")
        architecture[key] = int(value)
    return architecture


def _split_qkv(qkv_w, num_heads, num_kv_heads, head_dim, hidden):
    heads_per_group = num_heads // num_kv_heads
    group = (heads_per_group + 2) * head_dim
    query_size = heads_per_group * head_dim
    qkv = qkv_w.reshape(num_kv_heads, group, hidden)
    query = qkv[:, :query_size, :].reshape(-1, hidden).contiguous()
    key = qkv[:, query_size : query_size + head_dim, :].reshape(-1, hidden).contiguous()
    value = qkv[:, query_size + head_dim :, :].reshape(-1, hidden).contiguous()
    return query, key, value


def _split_gated(weight):
    intermediate_size = weight.shape[0] // 2
    return (
        weight[:intermediate_size, :].contiguous(),
        weight[intermediate_size:, :].contiguous(),
    )


def megatron_to_hf(
    state_dict: Mapping[str, torch.Tensor],
    architecture: Mapping[str, int],
) -> dict[str, torch.Tensor]:
    layers = architecture["num_hidden_layers"]
    num_heads = architecture["num_attention_heads"]
    num_kv_heads = architecture["num_key_value_heads"]
    head_dim = architecture["head_dim"]
    hidden_size = architecture["hidden_size"]
    num_experts = architecture["num_experts"]
    embedding = state_dict["embedding.word_embeddings.weight"]
    if embedding.ndim != 2 or embedding.shape[0] < FULL_VOCAB or embedding.shape[1] != hidden_size:
        raise ValueError(
            "RECIF input embedding must contain at least "
            f"{FULL_VOCAB} vocabulary rows, got shape {tuple(embedding.shape)}"
        )
    converted = {
        "model.embed_tokens.weight": embedding[:FULL_VOCAB].contiguous(),
        "model.norm.weight": state_dict["decoder.final_layernorm.weight"],
    }
    for layer in range(layers):
        source = f"decoder.layers.{layer}"
        target = f"model.layers.{layer}"
        query, key, value = _split_qkv(
            state_dict[f"{source}.self_attention.linear_qkv.weight"],
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
        )
        converted[f"{target}.self_attn.q_proj.weight"] = query
        converted[f"{target}.self_attn.k_proj.weight"] = key
        converted[f"{target}.self_attn.v_proj.weight"] = value
        converted[f"{target}.self_attn.o_proj.weight"] = state_dict[
            f"{source}.self_attention.linear_proj.weight"
        ]
        converted[f"{target}.self_attn.q_norm.weight"] = state_dict[
            f"{source}.self_attention.q_layernorm.weight"
        ]
        converted[f"{target}.self_attn.k_norm.weight"] = state_dict[
            f"{source}.self_attention.k_layernorm.weight"
        ]
        converted[f"{target}.input_layernorm.weight"] = state_dict[
            f"{source}.self_attention.linear_qkv.layer_norm_weight"
        ]
        converted[f"{target}.post_attention_layernorm.weight"] = state_dict[
            f"{source}.pre_mlp_layernorm.weight"
        ]
        converted[f"{target}.mlp.gate.weight"] = state_dict[f"{source}.mlp.router.weight"]
        for expert in range(num_experts):
            gate, up = _split_gated(state_dict[f"{source}.mlp.experts.linear_fc1.weight{expert}"])
            converted[f"{target}.mlp.experts.{expert}.gate_proj.weight"] = gate
            converted[f"{target}.mlp.experts.{expert}.up_proj.weight"] = up
            converted[f"{target}.mlp.experts.{expert}.down_proj.weight"] = state_dict[
                f"{source}.mlp.experts.linear_fc2.weight{expert}"
            ]
    return converted


def _unwrap_state_dict(value: Any, source: Path) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} must contain a checkpoint mapping")
    for key in ("model", "state_dict", "module"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            value = nested
            break
    return value


def _load_checkpoint(
    path: Path,
    *,
    allow_unsafe_pickle: bool = False,
) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        if not allow_unsafe_pickle:
            raise
        warnings.warn(
            f"Loading {path} with weights_only=False. This can execute arbitrary "
            "code; only continue with a checkpoint from a trusted source.",
            RuntimeWarning,
            stacklevel=2,
        )
        value = torch.load(path, map_location="cpu", weights_only=False)
    return _unwrap_state_dict(value, path)


def merge_recif_heads(external: Mapping[str, Any], hidden_size: int) -> torch.Tensor:
    heads = external.get("lm_heads")
    if not isinstance(heads, Mapping):
        raise RuntimeError("external_rank0.pt must contain an 'lm_heads' mapping")
    weights = []
    for level in range(NUM_SID_LEVELS):
        name = f"heads.{level}.weight"
        weight = heads.get(name)
        if weight is None:
            weight = heads.get(f"lm_heads.{name}")
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(f"Missing RECIF SID head weight: {name}")
        expected = (VOCAB, hidden_size)
        if tuple(weight.shape) != expected:
            raise ValueError(f"Unexpected {name} shape {tuple(weight.shape)}; expected {expected}")
        weights.append(weight)
    return torch.cat(weights, dim=0).contiguous()


def convert_checkpoint(
    input_dir: Path,
    output_dir: Path,
    *,
    allow_unsafe_pickle: bool = False,
) -> None:
    required = ("config.json", "_model_rank0.pt", "external_rank0.pt")
    missing = [name for name in required if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Invalid RECIF checkpoint {input_dir}: missing {', '.join(missing)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _clean_output_dir(output_dir)

    config = json.loads((input_dir / "config.json").read_text(encoding="utf-8"))
    architecture = architecture_from_config(config)
    backbone = _load_checkpoint(
        input_dir / "_model_rank0.pt",
        allow_unsafe_pickle=allow_unsafe_pickle,
    )
    external = _load_checkpoint(
        input_dir / "external_rank0.pt",
        allow_unsafe_pickle=allow_unsafe_pickle,
    )
    weights = megatron_to_hf(backbone, architecture)
    weights["lm_head.weight"] = merge_recif_heads(external, architecture["hidden_size"])
    _ensure_disk_space(output_dir, weights)
    _save_safetensors(
        {name: weight.contiguous() for name, weight in weights.items()},
        output_dir / "model.safetensors",
    )

    config["architectures"] = [RECIF_ARCHITECTURE]
    config["vocab_size"] = FULL_VOCAB
    config["tie_word_embeddings"] = False
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for source in input_dir.glob("*.json"):
        if source.name != "config.json":
            shutil.copy2(source, output_dir / source.name)


def _estimate_safetensors_bytes(weights: Mapping[str, torch.Tensor]) -> int:
    """Approximate the on-disk size of the serialized weights."""

    return sum(tensor.numel() * tensor.element_size() for tensor in weights.values())


def _clean_output_dir(output_dir: Path) -> None:
    """Remove stale conversion artifacts directly under ``output_dir``.

    Only files and directories immediately inside the conversion output
    directory are removed; this function never touches any other path.
    """

    if not output_dir.exists():
        return
    removed = 0
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
        removed += 1
    if removed:
        print(f"Cleaned {removed} stale item(s) under {output_dir}")


def _ensure_disk_space(output_dir: Path, weights: Mapping[str, torch.Tensor]) -> None:
    """Fail fast (before serializing) when the target filesystem is too full."""

    required = _estimate_safetensors_bytes(weights) + 64 * 1024 * 1024
    free = shutil.disk_usage(output_dir).free
    if free < required:
        raise InsufficientDiskSpaceError(
            f"Insufficient disk space to write model.safetensors "
            f"(need ~{required / 2**30:.1f} GiB, only {free / 2**30:.1f} GiB "
            f"free on {output_dir}). Free up space on this filesystem and "
            "retry."
        )


def _save_safetensors(
    weights: Mapping[str, torch.Tensor],
    target: Path,
) -> None:
    """Write safetensors directly; remove the target if the write fails.

    No intermediate file is created on disk; the conversion itself stays in
    memory and only the final ``model.safetensors`` is materialized.
    """

    try:
        save_file(weights, target)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Allow a weights_only=False fallback for legacy checkpoints. "
            "This can execute arbitrary code and must only be used with trusted input."
        ),
    )
    args = parser.parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_dir.with_name(f"{input_dir.name}_hf")
    )
    try:
        convert_checkpoint(
            input_dir,
            output_dir,
            allow_unsafe_pickle=args.allow_unsafe_pickle,
        )
    except InsufficientDiskSpaceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    print(f"Converted RECIF checkpoint written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
