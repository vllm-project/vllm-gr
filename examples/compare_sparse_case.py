#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare one selected case across golden/native and converted sparse paths.

This inspection utility focuses on the highest-value early alignment checks:
1. config/table mapping sanity
2. touched sparse embedding rows for one case
3. lookup outputs (`seq_embeddings`) right after sparse preprocess
4. non-embedding payload tensors (`payload_features`) emitted by preprocess

It is intended for accuracy triage after `acc_test.py` already confirms the
output structure is aligned, but numerical error still remains.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GOLDEN_REPO = "/root/PROJECT/HSTU_Update/generative-recommenders"
DEFAULT_DATASET = "movielens-1m"
DEFAULT_SEED = 20260318


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one benchmark case across native/golden and converted "
            "sparse paths. Prints touched sparse rows, seq_embeddings, and payload_features."
        ),
    )
    parser.add_argument(
        "--native-model",
        type=str,
        required=True,
        help="Path to native/golden checkpoint directory (non_sparse.ckpt + sparse/).",
    )
    parser.add_argument(
        "--converted-model",
        type=str,
        required=True,
        help="Path to converted vllm-GR model directory.",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Benchmark input JSON used by acc_test / acc_output.",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        required=True,
        help="Case id to inspect.",
    )
    parser.add_argument(
        "--golden-repo",
        type=str,
        default=DEFAULT_GOLDEN_REPO,
        help=f"Path to generative-recommenders repo (default: {DEFAULT_GOLDEN_REPO}).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset name (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Optional native config.json to cap embedding table sizes.",
    )
    parser.add_argument(
        "--max-num-embeddings",
        type=int,
        default=None,
        help="Optional upper bound for native embedding tables.",
    )
    parser.add_argument(
        "--max-ids-per-feature",
        type=int,
        default=20,
        help="Max touched ids to print per feature (default: 20).",
    )
    parser.add_argument(
        "--preview-dim",
        type=int,
        default=8,
        help="Preview the first N embedding dims when printing rows (default: 8).",
    )
    parser.add_argument(
        "--preview-tokens",
        type=int,
        default=2,
        help="Preview the first N lookup output rows per feature (default: 2).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the full comparison report as JSON.",
    )
    return parser.parse_args()


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _find_case(payload: dict[str, Any], case_id: str) -> dict[str, Any]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("input JSON must contain list field: cases")
    for case in cases:
        if str(case.get("case_id")) == case_id:
            return case
    raise ValueError(f"case_id not found: {case_id}")


def _add_golden_repo_to_path(golden_repo: str) -> None:
    resolved = str(Path(golden_repo).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_config_json_path(model_path: str, config_json_arg: str | None) -> str | None:
    if config_json_arg:
        return config_json_arg
    inferred = os.path.join(model_path, "config.json")
    if os.path.isfile(inferred):
        return inferred
    return None


def _apply_num_embeddings_from_config(table_config: dict[str, Any], config_json_path: str) -> None:
    with open(config_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    embedding_tables = cfg.get("embedding_tables")
    if not isinstance(embedding_tables, list):
        return

    cfg_rows_by_name: dict[str, int] = {}
    for table in embedding_tables:
        if isinstance(table, dict) and "name" in table and "num_embeddings" in table:
            cfg_rows_by_name[str(table["name"])] = int(table["num_embeddings"])

    for table_name, emb_cfg in table_config.items():
        if table_name in cfg_rows_by_name:
            emb_cfg.num_embeddings = int(cfg_rows_by_name[table_name])


def _truncate_table_config_num_embeddings(
    table_config: dict[str, Any],
    max_num_embeddings: int | None,
) -> None:
    if max_num_embeddings is None:
        return
    if max_num_embeddings <= 0:
        raise ValueError(f"--max-num-embeddings must be > 0, got {max_num_embeddings}")
    for emb_cfg in table_config.values():
        emb_cfg.num_embeddings = min(int(emb_cfg.num_embeddings), max_num_embeddings)


def _build_feature_caps_from_native_table_config(table_config: dict[str, Any]) -> dict[str, int]:
    feature_caps: dict[str, int] = {}
    for emb_cfg in table_config.values():
        cap = int(emb_cfg.num_embeddings)
        for feature_name in emb_cfg.feature_names:
            key = str(feature_name)
            if key in feature_caps:
                feature_caps[key] = min(feature_caps[key], cap)
            else:
                feature_caps[key] = cap
    return feature_caps


def _build_feature_caps_from_converted_config(config: Any) -> dict[str, int]:
    feature_caps: dict[str, int] = {}
    for table in config.embedding_tables:
        cap = int(table["num_embeddings"])
        for feature_name in table["feature_names"]:
            key = str(feature_name)
            if key in feature_caps:
                feature_caps[key] = min(feature_caps[key], cap)
            else:
                feature_caps[key] = cap
    return feature_caps


def _build_feature_to_table_map_from_native_table_config(
    table_config: dict[str, Any],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for table_name, emb_cfg in table_config.items():
        for feature_name in emb_cfg.feature_names:
            mapping[str(feature_name)] = str(table_name)
    return mapping


def _build_feature_to_table_map_from_converted_config(config: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for table in config.embedding_tables:
        table_name = str(table["name"])
        for feature_name in table["feature_names"]:
            mapping[str(feature_name)] = table_name
    return mapping


def _validate_feature_value(
    *,
    case_id: str,
    key: str,
    value: Any,
    feature_caps: dict[str, int],
) -> int:
    ivalue = int(value)
    cap = feature_caps.get(key)
    if cap is not None and (ivalue < 0 or ivalue >= cap):
        raise ValueError(
            "feature index out of range: "
            f"case_id={case_id}, feature={key}, value={ivalue}, valid=[0,{cap - 1}]"
        )
    return ivalue


def _ordered_unique(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _extract_touched_feature_ids(
    *,
    case: dict[str, Any],
    case_id: str,
    uih_keys: list[str],
    contextual_keys: set[str],
    candidate_keys: list[str],
    feature_caps: dict[str, int],
) -> dict[str, list[int]]:
    user_history = case["user_history"]
    candidates = case["candidates"]
    context = case["context"]

    touched: dict[str, list[int]] = {}

    for key in uih_keys:
        if key in contextual_keys:
            values = [
                _validate_feature_value(
                    case_id=case_id,
                    key=key,
                    value=context.get(key, 0),
                    feature_caps=feature_caps,
                )
            ]
        else:
            values = [
                _validate_feature_value(
                    case_id=case_id,
                    key=key,
                    value=row.get(key, 0),
                    feature_caps=feature_caps,
                )
                for row in user_history
            ]
        touched[key] = _ordered_unique(values)

    for key in candidate_keys:
        values = [
            _validate_feature_value(
                case_id=case_id,
                key=key,
                value=row.get(key, 0),
                feature_caps=feature_caps,
            )
            for row in candidates
        ]
        touched[key] = _ordered_unique(values)

    return touched


def _build_sparse_kjts(
    *,
    case: dict[str, Any],
    case_id: str,
    uih_keys: list[str],
    contextual_keys: set[str],
    candidate_keys: list[str],
    feature_caps: dict[str, int],
):
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    user_history = case["user_history"]
    candidates = case["candidates"]
    context = case["context"]
    seq_len = len(user_history)
    num_candidates = len(candidates)

    uih_values: list[int] = []
    uih_lengths: list[int] = []
    for key in uih_keys:
        if key in contextual_keys:
            uih_values.append(
                _validate_feature_value(
                    case_id=case_id,
                    key=key,
                    value=context.get(key, 0),
                    feature_caps=feature_caps,
                )
            )
            uih_lengths.append(1)
        else:
            values = [
                _validate_feature_value(
                    case_id=case_id,
                    key=key,
                    value=row.get(key, 0),
                    feature_caps=feature_caps,
                )
                for row in user_history
            ]
            uih_values.extend(values)
            uih_lengths.append(seq_len)

    cand_values: list[int] = []
    cand_lengths: list[int] = []
    for key in candidate_keys:
        values = [
            _validate_feature_value(
                case_id=case_id,
                key=key,
                value=row.get(key, 0),
                feature_caps=feature_caps,
            )
            for row in candidates
        ]
        cand_values.extend(values)
        cand_lengths.append(num_candidates)

    uih_features_kjt = KeyedJaggedTensor(
        keys=uih_keys,
        lengths=torch.tensor(uih_lengths, dtype=torch.long),
        values=torch.tensor(uih_values, dtype=torch.long),
    )
    candidates_features_kjt = KeyedJaggedTensor(
        keys=candidate_keys,
        lengths=torch.tensor(cand_lengths, dtype=torch.long),
        values=torch.tensor(cand_values, dtype=torch.long),
    )
    return uih_features_kjt, candidates_features_kjt


def _find_weight_file(model_dir: str) -> str:
    for name in (
        "model.safetensors",
        "dense.safetensors",
        "model.pt",
        "dense.pt",
        "pytorch_model.bin",
    ):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No converted dense weight file found in: {model_dir}")


def _load_converted_weight_dict(model_dir: str) -> dict[str, torch.Tensor]:
    weight_path = _find_weight_file(model_dir)
    embedding_path = None
    if weight_path.endswith("dense.safetensors"):
        candidate_path = os.path.join(model_dir, "embedding.safetensors")
        if os.path.isfile(candidate_path):
            embedding_path = candidate_path
    elif weight_path.endswith("dense.pt"):
        candidate_path = os.path.join(model_dir, "embedding.pt")
        if os.path.isfile(candidate_path):
            embedding_path = candidate_path

    if weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(weight_path)
        if embedding_path is not None:
            state.update(load_file(embedding_path))
        return state

    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    if embedding_path is not None:
        embedding_state = torch.load(
            embedding_path,
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(state, dict) and isinstance(embedding_state, dict):
            state.update(embedding_state)
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unsupported weight file payload: {weight_path}")


def _get_embedding_weight(hstu_model: Any, table_name: str) -> torch.Tensor:
    state_dict = hstu_model.state_dict()
    preferred = f"_embedding_collection.embeddings.{table_name}.weight"
    if preferred in state_dict:
        return state_dict[preferred]

    suffix = f".{table_name}.weight"
    for key, value in state_dict.items():
        if key.endswith(suffix):
            return value

    raise KeyError(
        f"Unable to find embedding weight for table={table_name!r}; "
        f"available sparse-like keys include {[k for k in state_dict if 'embedding_collection' in k][:10]}"
    )


def _preview_vector(tensor: torch.Tensor, limit: int) -> list[float]:
    flat = tensor.detach().cpu().float().reshape(-1)
    return [float(x) for x in flat[:limit].tolist()]


def _preview_matrix(tensor: torch.Tensor, *, rows: int, dims: int) -> list[list[float]]:
    matrix = tensor.detach().cpu().float()
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    matrix = matrix[:rows, :dims]
    return [[float(x) for x in row.tolist()] for row in matrix]


def _summarize_tensor_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_f = left.detach().cpu().float()
    right_f = right.detach().cpu().float()
    if left_f.shape != right_f.shape:
        return {
            "shape_match": False,
            "left_shape": list(left_f.shape),
            "right_shape": list(right_f.shape),
        }
    diff = (left_f - right_f).abs()
    return {
        "shape_match": True,
        "left_shape": list(left_f.shape),
        "right_shape": list(right_f.shape),
        "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
    }


def _compare_touched_rows(
    *,
    native_hstu: Any,
    converted_hstu: Any,
    touched_feature_ids: dict[str, list[int]],
    native_feature_to_table: dict[str, str],
    converted_feature_to_table: dict[str, str],
    max_ids_per_feature: int,
    preview_dim: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    for feature_name, touched_ids in touched_feature_ids.items():
        native_table = native_feature_to_table.get(feature_name)
        converted_table = converted_feature_to_table.get(feature_name)
        entry: dict[str, Any] = {
            "feature_name": feature_name,
            "native_table": native_table,
            "converted_table": converted_table,
            "touched_ids_total": len(touched_ids),
            "touched_ids_reported": touched_ids[:max_ids_per_feature],
        }
        if native_table is None and converted_table is None:
            entry["classification"] = "non_embedding_feature"
            report[feature_name] = entry
            continue
        if native_table is None or converted_table is None:
            entry["classification"] = "embedding_mapping_mismatch"
            entry["error"] = "feature is not covered by one side's embedding table config"
            report[feature_name] = entry
            continue

        entry["classification"] = "embedding_feature"

        native_weight = _get_embedding_weight(native_hstu, native_table)
        converted_weight = _get_embedding_weight(converted_hstu, converted_table)
        entry["native_weight_dtype"] = str(native_weight.dtype)
        entry["converted_weight_dtype"] = str(converted_weight.dtype)
        entry["native_weight_shape"] = list(native_weight.shape)
        entry["converted_weight_shape"] = list(converted_weight.shape)

        rows: list[dict[str, Any]] = []
        for row_id in touched_ids[:max_ids_per_feature]:
            native_row = native_weight[row_id]
            converted_row = converted_weight[row_id]
            row_summary = {
                "row_id": int(row_id),
                **_summarize_tensor_difference(native_row, converted_row),
                "native_preview": _preview_vector(native_row, preview_dim),
                "converted_preview": _preview_vector(converted_row, preview_dim),
            }
            rows.append(row_summary)
        entry["rows"] = rows
        report[feature_name] = entry

    return report


def _compare_seq_embeddings(
    *,
    native_seq_embeddings: dict[str, Any],
    converted_seq_embeddings: dict[str, Any],
    preview_tokens: int,
    preview_dim: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    all_keys = sorted(set(native_seq_embeddings.keys()) | set(converted_seq_embeddings.keys()))

    for key in all_keys:
        native_value = native_seq_embeddings.get(key)
        converted_value = converted_seq_embeddings.get(key)
        entry: dict[str, Any] = {
            "present_in_native": native_value is not None,
            "present_in_converted": converted_value is not None,
        }
        if native_value is None or converted_value is None:
            report[key] = entry
            continue

        native_lengths = native_value.lengths.detach().cpu()
        converted_lengths = converted_value.lengths.detach().cpu()
        native_embedding = native_value.embedding.detach().cpu()
        converted_embedding = converted_value.embedding.detach().cpu()

        entry["native_lengths"] = [int(x) for x in native_lengths.tolist()]
        entry["converted_lengths"] = [int(x) for x in converted_lengths.tolist()]
        entry["lengths_match"] = bool(torch.equal(native_lengths, converted_lengths))
        entry["native_dtype"] = str(native_embedding.dtype)
        entry["converted_dtype"] = str(converted_embedding.dtype)
        entry.update(_summarize_tensor_difference(native_embedding, converted_embedding))
        entry["native_preview"] = _preview_matrix(
            native_embedding,
            rows=preview_tokens,
            dims=preview_dim,
        )
        entry["converted_preview"] = _preview_matrix(
            converted_embedding,
            rows=preview_tokens,
            dims=preview_dim,
        )
        report[key] = entry

    return report


def _compare_payload_features(
    *,
    native_payload_features: dict[str, torch.Tensor],
    converted_payload_features: dict[str, torch.Tensor],
    preview_dim: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    all_keys = sorted(set(native_payload_features.keys()) | set(converted_payload_features.keys()))

    for key in all_keys:
        native_value = native_payload_features.get(key)
        converted_value = converted_payload_features.get(key)
        entry: dict[str, Any] = {
            "present_in_native": native_value is not None,
            "present_in_converted": converted_value is not None,
        }
        if native_value is None or converted_value is None:
            report[key] = entry
            continue

        native_tensor = native_value.detach().cpu()
        converted_tensor = converted_value.detach().cpu()
        entry["native_dtype"] = str(native_tensor.dtype)
        entry["converted_dtype"] = str(converted_tensor.dtype)
        entry.update(_summarize_tensor_difference(native_tensor, converted_tensor))
        entry["native_preview"] = _preview_vector(native_tensor, preview_dim)
        entry["converted_preview"] = _preview_vector(converted_tensor, preview_dim)
        report[key] = entry

    return report


def _load_native_context(args: argparse.Namespace) -> dict[str, Any]:
    _add_golden_repo_to_path(args.golden_repo)

    from generative_recommenders.dlrm_v3.configs import (  # type: ignore
        get_embedding_table_config,
        get_hstu_configs,
    )
    from generative_recommenders.dlrm_v3.inference.inference_modules import (  # type: ignore
        set_is_inference,
    )
    from generative_recommenders.dlrm_v3.inference.model_family import (  # type: ignore
        HSTUModelFamily,
    )

    os.environ["WORLD_SIZE"] = "1"
    _set_seed(DEFAULT_SEED)
    set_is_inference(is_inference=True)

    hstu_config = get_hstu_configs(args.dataset)
    hstu_config.max_num_candidates = hstu_config.max_num_candidates_inference
    table_config = get_embedding_table_config(args.dataset)

    config_json_path = _resolve_config_json_path(args.native_model, args.config_json)
    if config_json_path is not None:
        _apply_num_embeddings_from_config(table_config, config_json_path)
    _truncate_table_config_num_embeddings(table_config, args.max_num_embeddings)

    model_family = HSTUModelFamily(
        hstu_config=hstu_config,
        table_config=table_config,
        output_trace=False,
        sparse_quant=False,
        compute_eval=False,
    )
    model_family.load(args.native_model)

    sparse_module = getattr(model_family.sparse, "module", None)
    if sparse_module is None or not hasattr(sparse_module, "_hstu_model"):
        raise RuntimeError("Unable to access golden sparse _hstu_model for comparison")

    return {
        "hstu_config": hstu_config,
        "table_config": table_config,
        "feature_caps": _build_feature_caps_from_native_table_config(table_config),
        "feature_to_table": _build_feature_to_table_map_from_native_table_config(table_config),
        "model_family": model_family,
        "hstu_model": sparse_module._hstu_model,
    }


def _load_converted_context(args: argparse.Namespace) -> dict[str, Any]:
    from vllm_gr.config.hstu_config import HSTUConfig
    from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

    config = HSTUConfig.from_pretrained(args.converted_model)
    model = HSTUForRecommendation(config=config, device=torch.device("cpu"))
    weights = _load_converted_weight_dict(args.converted_model)
    model.load_weights(iter(weights.items()))
    model.eval()

    return {
        "config": config,
        "feature_caps": _build_feature_caps_from_converted_config(config),
        "feature_to_table": _build_feature_to_table_map_from_converted_config(config),
        "model": model,
        "hstu_model": model.hstu,
    }


def _build_config_compare(
    native_ctx: dict[str, Any], converted_ctx: dict[str, Any]
) -> dict[str, Any]:
    native_hstu_config = native_ctx["hstu_config"]
    converted_config = converted_ctx["config"]

    native_uih_keys = list(native_hstu_config.hstu_uih_feature_names)
    converted_uih_keys = list(converted_config.hstu_uih_feature_names)
    native_candidate_keys = list(native_hstu_config.hstu_candidate_feature_names)
    converted_candidate_keys = list(converted_config.hstu_candidate_feature_names)
    native_contextual_keys = sorted(native_hstu_config.contextual_feature_to_max_length.keys())
    converted_contextual_keys = sorted(converted_config.contextual_feature_to_max_length.keys())
    native_tables = sorted(native_ctx["feature_to_table"].items())
    converted_tables = sorted(converted_ctx["feature_to_table"].items())

    return {
        "uih_keys_match": native_uih_keys == converted_uih_keys,
        "candidate_keys_match": native_candidate_keys == converted_candidate_keys,
        "contextual_keys_match": native_contextual_keys == converted_contextual_keys,
        "feature_table_map_match": native_tables == converted_tables,
        "native_uih_keys": native_uih_keys,
        "converted_uih_keys": converted_uih_keys,
        "native_candidate_keys": native_candidate_keys,
        "converted_candidate_keys": converted_candidate_keys,
        "native_contextual_keys": native_contextual_keys,
        "converted_contextual_keys": converted_contextual_keys,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print("== Case ==")
    case = report["case"]
    print(
        f"case_id={case['case_id']} user_history={case['user_history_len']} "
        f"candidates={case['num_candidates']}"
    )

    print("\n== Config Compare ==")
    config = report["config_compare"]
    print(f"uih_keys_match={config['uih_keys_match']}")
    print(f"candidate_keys_match={config['candidate_keys_match']}")
    print(f"contextual_keys_match={config['contextual_keys_match']}")
    print(f"feature_table_map_match={config['feature_table_map_match']}")

    print("\n== Touched Feature IDs ==")
    for feature_name, values in report["touched_feature_ids"].items():
        print(f"{feature_name}: {values}")

    print("\n== Touched Sparse Rows ==")
    for feature_name, entry in report["touched_rows"].items():
        print(
            f"[{feature_name}] classification={entry.get('classification')} "
            f"native_table={entry.get('native_table')} "
            f"converted_table={entry.get('converted_table')}"
        )
        if entry.get("classification") == "non_embedding_feature":
            print("  skipped: feature is not expected to come from an embedding table")
            continue
        if "error" in entry:
            print(f"  error={entry['error']}")
            continue
        print(
            f"  native_dtype={entry['native_weight_dtype']} "
            f"converted_dtype={entry['converted_weight_dtype']}"
        )
        for row in entry["rows"]:
            print(
                f"  row_id={row['row_id']} "
                f"mean_abs={row.get('mean_abs_error')} max_abs={row.get('max_abs_error')}"
            )

    print("\n== Seq Embeddings ==")
    for key, entry in report["seq_embeddings"].items():
        print(
            f"[{key}] present_native={entry['present_in_native']} "
            f"present_converted={entry['present_in_converted']}"
        )
        if not entry["present_in_native"] or not entry["present_in_converted"]:
            continue
        print(
            f"  lengths_match={entry['lengths_match']} "
            f"native_dtype={entry['native_dtype']} converted_dtype={entry['converted_dtype']} "
            f"mean_abs={entry.get('mean_abs_error')} max_abs={entry.get('max_abs_error')}"
        )

    print("\n== Payload Features ==")
    for key, entry in report["payload_features"].items():
        print(
            f"[{key}] present_native={entry['present_in_native']} "
            f"present_converted={entry['present_in_converted']}"
        )
        if not entry["present_in_native"] or not entry["present_in_converted"]:
            continue
        print(
            f"  native_dtype={entry['native_dtype']} converted_dtype={entry['converted_dtype']} "
            f"mean_abs={entry.get('mean_abs_error')} max_abs={entry.get('max_abs_error')}"
        )


def main() -> int:
    args = _parse_args()
    input_payload = _load_json(args.input_json)
    case = _find_case(input_payload, args.case_id)
    case_id = str(case["case_id"])

    native_ctx = _load_native_context(args)
    converted_ctx = _load_converted_context(args)

    config_compare = _build_config_compare(native_ctx, converted_ctx)

    native_hstu_config = native_ctx["hstu_config"]
    converted_config = converted_ctx["config"]
    native_uih_keys = list(native_hstu_config.hstu_uih_feature_names)
    native_candidate_keys = list(native_hstu_config.hstu_candidate_feature_names)
    native_contextual_keys = set(native_hstu_config.contextual_feature_to_max_length.keys())

    touched_feature_ids = _extract_touched_feature_ids(
        case=case,
        case_id=case_id,
        uih_keys=native_uih_keys,
        contextual_keys=native_contextual_keys,
        candidate_keys=native_candidate_keys,
        feature_caps=native_ctx["feature_caps"],
    )

    touched_rows = _compare_touched_rows(
        native_hstu=native_ctx["hstu_model"],
        converted_hstu=converted_ctx["hstu_model"],
        touched_feature_ids=touched_feature_ids,
        native_feature_to_table=native_ctx["feature_to_table"],
        converted_feature_to_table=converted_ctx["feature_to_table"],
        max_ids_per_feature=args.max_ids_per_feature,
        preview_dim=args.preview_dim,
    )

    from generative_recommenders.dlrm_v3.datasets.dataset import Samples  # type: ignore

    native_uih_kjt, native_candidates_kjt = _build_sparse_kjts(
        case=case,
        case_id=case_id,
        uih_keys=native_uih_keys,
        contextual_keys=native_contextual_keys,
        candidate_keys=native_candidate_keys,
        feature_caps=native_ctx["feature_caps"],
    )
    samples = Samples(
        uih_features_kjt=native_uih_kjt,
        candidates_features_kjt=native_candidates_kjt,
    )
    (
        native_seq_embeddings,
        native_payload_features,
        _,
        _,
        _,
        _,
        _,
    ) = native_ctx["model_family"].sparse.predict(samples)

    converted_uih_keys = list(converted_config.hstu_uih_feature_names)
    converted_candidate_keys = list(converted_config.hstu_candidate_feature_names)
    converted_contextual_keys = set(converted_config.contextual_feature_to_max_length.keys())
    converted_uih_kjt, converted_candidates_kjt = _build_sparse_kjts(
        case=case,
        case_id=case_id,
        uih_keys=converted_uih_keys,
        contextual_keys=converted_contextual_keys,
        candidate_keys=converted_candidate_keys,
        feature_caps=converted_ctx["feature_caps"],
    )
    (
        converted_seq_embeddings,
        converted_payload_features,
        _,
        _,
        _,
        _,
    ) = converted_ctx["hstu_model"].preprocess(
        uih_features=converted_uih_kjt,
        candidates_features=converted_candidates_kjt,
    )

    seq_embeddings = _compare_seq_embeddings(
        native_seq_embeddings=native_seq_embeddings,
        converted_seq_embeddings=converted_seq_embeddings,
        preview_tokens=args.preview_tokens,
        preview_dim=args.preview_dim,
    )
    payload_features = _compare_payload_features(
        native_payload_features=native_payload_features,
        converted_payload_features=converted_payload_features,
        preview_dim=args.preview_dim,
    )

    report = {
        "case": {
            "case_id": case_id,
            "user_history_len": len(case["user_history"]),
            "num_candidates": len(case["candidates"]),
        },
        "config_compare": config_compare,
        "touched_feature_ids": touched_feature_ids,
        "touched_rows": touched_rows,
        "seq_embeddings": seq_embeddings,
        "payload_features": payload_features,
    }

    _print_summary(report)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote report: {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[compare_sparse_case] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
