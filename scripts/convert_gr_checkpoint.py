#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Convert DLRM_v3 training checkpoints to HuggingFace-compatible format.

Meta's generative-recommenders saves training checkpoints in a split format:
    {path}/{batch_idx}/
    ├── sparse/            # torch.distributed.checkpoint (embeddings)
    └── non_sparse.ckpt    # torch.save dict with dense_dict, optimizer, etc.

This script converts them into vLLM-compatible flat format:
    {output_dir}/
    ├── config.json             # HSTUConfig (must be provided or generated)
    ├── model.safetensors       # flat state_dict with "hstu." prefix
    ├── dense.safetensors       # optional dense-only split weights
    ├── embedding.safetensors   # optional embedding-only split weights
    └── conversion_meta.json    # optional conversion metadata

Reuses the upstream checkpoint utilities ported in
vllm_gr.models.dlrm_v3.checkpoint for maximum code fidelity.

Dense weights
=============
Loaded from non_sparse.ckpt via torch.load(). The checkpoint contains a
``dense_dict`` with all non-embedding parameters.  Sparse keys are
filtered out with ``is_sparse_key()``, the ``hstu.`` prefix is added,
and tensors are cast to the target dtype (bfloat16 by default).

Sparse (embedding) weights
===========================
Saved as ``ShardedTensor`` via PyTorch Distributed Checkpoint (DCP) in the
``sparse/`` directory.  During training, embedding tables may be sharded
across N GPUs.  To load them in a single-process conversion script:

1. ``torch.distributed`` is initialized with the ``gloo`` backend
   (world_size=1, rank=0).
2. A ``DlrmHSTU`` model is created with ``EmbeddingCollection`` on CPU.
3. Dense weights are loaded into the model first (required by
   ``load_dense_state_dict``'s assertion).
4. ``load_sparse_checkpoint(model, path)`` calls DCP, which automatically
   reshards the N-rank checkpoint into the single-rank model.
5. Embedding tensors are extracted from the model's state_dict.

Alternatively, run via ``torchrun`` to let PyTorch handle distributed init::

    torchrun --nproc_per_node=1 scripts/convert_gr_checkpoint.py \\
        --input-path /path/to/checkpoint/{batch_idx} \\
        --output-dir /path/to/hf_model \\
        --config-json /path/to/config.json

Usage:
    python scripts/convert_gr_checkpoint.py \\
        --input-path /path/to/checkpoint/{batch_idx} \\
        --output-dir /path/to/hf_model \\
        --dataset movielens-1m

    # With existing config.json:
    python scripts/convert_gr_checkpoint.py \\
        --input-path /path/to/checkpoint/{batch_idx} \\
        --output-dir /path/to/hf_model \\
        --config-json /path/to/config.json

    # Dense-only (skip embeddings, matching upstream inference pattern):
    python scripts/convert_gr_checkpoint.py \\
        --input-path /path/to/checkpoint/{batch_idx} \\
        --output-dir /path/to/hf_model \\
        --config-json /path/to/config.json \\
        --dense-only
"""

import argparse
import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from vllm_gr.models.dlrm_v3.checkpoint import (
    is_sparse_key,
    load_dense_state_dict,
    load_sparse_checkpoint,
)

logger = logging.getLogger(__name__)


KNOWN_WEIGHT_ARTIFACTS = (
    "model.safetensors",
    "model.pt",
    "dense.safetensors",
    "dense.pt",
    "embedding.safetensors",
    "embedding.pt",
    "conversion_meta.json",
)


def _load_supported_dataset_choices() -> tuple[str, ...]:
    """Load supported dataset names without importing the heavy config package."""
    module_name = "vllm_gr_dataset_configs_for_conversion"
    module_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "vllm_gr",
        "config",
        "dataset_configs.py",
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load dataset configs from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.SUPPORTED_DATASETS)


DATASET_CHOICES = _load_supported_dataset_choices()


def _resolve_dtype_name(dtype_name: str, *, default: torch.dtype) -> torch.dtype:
    normalized = str(dtype_name).lower()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(normalized, default)


def _build_embedding_table_dtype_map(config: Any) -> Dict[str, torch.dtype]:
    default_dtype = _resolve_dtype_name(
        getattr(config, "embedding_table_data_type", "fp16"),
        default=torch.float16,
    )
    table_dtypes: Dict[str, torch.dtype] = {}

    for table_cfg in getattr(config, "embedding_tables", []) or []:
        table_name = str(table_cfg["name"])
        table_dtypes[table_name] = _resolve_dtype_name(
            table_cfg.get("data_type", default_dtype),
            default=default_dtype,
        )

    return table_dtypes


def _build_embedding_table_num_embeddings_map(config: Any) -> Dict[str, int]:
    table_num_embeddings: Dict[str, int] = {}
    for table_cfg in getattr(config, "embedding_tables", []) or []:
        table_num_embeddings[str(table_cfg["name"])] = int(table_cfg["num_embeddings"])
    return table_num_embeddings


def _resolve_sparse_tensor_target_dtype(
    config: Any,
    tensor_name: str,
    tensor: torch.Tensor,
) -> torch.dtype:
    if not torch.is_floating_point(tensor):
        return tensor.dtype

    table_dtypes = _build_embedding_table_dtype_map(config)
    for table_name, table_dtype in table_dtypes.items():
        if f".{table_name}." in tensor_name:
            return table_dtype
    return tensor.dtype


def _is_embedding_weight_tensor(tensor_name: str, tensor: torch.Tensor) -> bool:
    return (
        torch.is_tensor(tensor)
        and torch.is_floating_point(tensor)
        and ".embeddings." in tensor_name
        and tensor_name.endswith(".weight")
        and tensor.ndim >= 2
    )


def _cleanup_previous_weight_artifacts(output_dir: str) -> None:
    for artifact_name in KNOWN_WEIGHT_ARTIFACTS:
        artifact_path = os.path.join(output_dir, artifact_name)
        if os.path.exists(artifact_path):
            os.remove(artifact_path)
            logger.info("Removed stale artifact at %s", artifact_path)


def _init_single_process_distributed() -> None:
    """Initialize torch.distributed for single-process DCP loading.

    Uses gloo backend (CPU-only, no GPU required). Safe to call multiple
    times -- returns immediately if already initialized.
    """
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.distributed.init_process_group(backend="gloo")
    logger.info("Initialized single-process torch.distributed (gloo)")


def _destroy_distributed() -> None:
    """Destroy torch.distributed process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Destroyed torch.distributed process group")


def load_dense_weights(
    checkpoint_path: str,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load dense (non-embedding) weights from non_sparse.ckpt.

    Reads the checkpoint and filters out sparse keys using the upstream
    is_sparse_key function from checkpoint.py.

    Args:
        checkpoint_path: Path to checkpoint directory containing non_sparse.ckpt.
        device: Device to load tensors onto.

    Returns:
        Dictionary of dense parameter tensors.
    """
    non_sparse_ckpt = os.path.join(checkpoint_path, "non_sparse.ckpt")
    if not os.path.exists(non_sparse_ckpt):
        raise FileNotFoundError(f"non_sparse.ckpt not found at {non_sparse_ckpt}")

    logger.info("Loading dense weights from %s", non_sparse_ckpt)
    state = torch.load(non_sparse_ckpt, map_location=device, weights_only=False)
    dense_dict = state["dense_dict"]

    filtered = {}
    for k, v in dense_dict.items():
        if is_sparse_key(k, v):
            logger.debug("Skipping sparse key in dense_dict: %s", k)
            continue
        if isinstance(v, torch.nn.Parameter):
            v = v.data
        filtered[k] = v

    logger.info("Loaded %d dense tensors", len(filtered))
    return filtered


def load_sparse_weights(
    checkpoint_path: str,
    config_dir: str,
    dense_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Load sparse (embedding) weights via DCP and the upstream load_sparse_checkpoint.

    Creates a DlrmHSTU model, loads dense weights into it first (required by
    the upstream ``load_dense_state_dict`` assertion), then calls
    ``load_sparse_checkpoint()`` to load DCP shards into the model. Extracts
    and returns the embedding tensors.

    Requires ``torch.distributed`` to be initialized (call
    ``_init_single_process_distributed()`` first).

    Args:
        checkpoint_path: Path to checkpoint directory containing sparse/.
        config_dir: Path to directory with config.json (for model creation).
        dense_weights: Dense weights dict (keys without hstu. prefix) to
            pre-load into the model before sparse loading.

    Returns:
        Dictionary of sparse parameter tensors (keys without hstu. prefix).
    """
    sparse_path = os.path.join(checkpoint_path, "sparse")
    if not os.path.isdir(sparse_path):
        logger.info("No sparse/ directory found at %s, skipping", sparse_path)
        return {}

    from vllm_gr.config.hstu_config import HSTUConfig
    from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

    config = HSTUConfig.from_pretrained(config_dir)
    model = HSTUForRecommendation(config=config)
    hstu = model.hstu

    # Load dense weights into the model first -- load_sparse_checkpoint's
    # upstream SparseState depends on the model having consistent state.
    load_dense_state_dict(hstu, dense_weights)
    logger.info("Pre-loaded dense weights into model for sparse loading")

    sparse_keys_before = {k for k, v in hstu.state_dict().items() if is_sparse_key(k, v)}
    if not sparse_keys_before:
        raise RuntimeError(
            "Sparse checkpoint/config mismatch: checkpoint contains sparse weights, "
            "but the instantiated model exposes no sparse parameters. "
            "Check that config.json matches the checkpoint layout."
        )

    logger.info(
        "Loading sparse weights (%d embedding tensors) via DCP from %s",
        len(sparse_keys_before),
        sparse_path,
    )
    load_sparse_checkpoint(model=hstu, path=checkpoint_path)

    result = {}
    for k, v in hstu.state_dict().items():
        if k in sparse_keys_before:
            result[k] = v.detach().cpu()
    logger.info("Loaded %d sparse tensors", len(result))
    return result


def generate_config_from_dataset(
    dataset: str,
    output_dir: str,
) -> None:
    """Generate a config.json for the given dataset using vllm-GR dataset configs."""
    from vllm_gr.config.dataset_configs import (
        get_embedding_table_overrides,
        get_hstu_config_overrides,
    )
    from vllm_gr.config.hstu_config import HSTUConfig

    overrides = get_hstu_config_overrides(dataset)
    tables = get_embedding_table_overrides(dataset)
    if tables:
        overrides["embedding_tables"] = tables

    config = HSTUConfig(**overrides)
    _write_target_config(
        output_dir=output_dir,
        target_config=config,
        source_config=config,
    )
    logger.info("Generated config.json for dataset=%s at %s", dataset, output_dir)


def _materialize_source_config(
    *,
    config_json: Optional[str],
    dataset: Optional[str],
    output_dir: str,
) -> Optional[str]:
    if config_json is None and dataset is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    if config_json is not None:
        shutil.copy2(config_json, os.path.join(output_dir, "config.json"))
        logger.info("Copied source config.json from %s", config_json)
    else:
        generate_config_from_dataset(dataset or "", output_dir)
    return output_dir


def _load_hstu_config(config_dir: str) -> Any:
    from vllm_gr.config.hstu_config import HSTUConfig

    return HSTUConfig.from_pretrained(config_dir)


def _resolve_embedding_table_dim(config: Any) -> int:
    target_dim = int(getattr(config, "hstu_embedding_table_dim", 0))
    if target_dim <= 0:
        raise ValueError("config.hstu_embedding_table_dim must be a positive integer")

    embedding_tables = list(getattr(config, "embedding_tables", []) or [])
    if not embedding_tables:
        raise ValueError("config.embedding_tables must be non-empty")

    mismatched_tables = [
        (
            str(table_cfg.get("name", "<unknown>")),
            int(table_cfg.get("embedding_dim", -1)),
        )
        for table_cfg in embedding_tables
        if int(table_cfg.get("embedding_dim", -1)) != target_dim
    ]
    if mismatched_tables:
        raise ValueError(
            "All embedding_tables[*].embedding_dim values must match "
            "hstu_embedding_table_dim for conversion. "
            f"Expected {target_dim}, got {mismatched_tables!r}."
        )

    return target_dim


def _validate_crop_embedding_dim(
    config: Any,
    crop_embedding_dim: Optional[int],
) -> Tuple[int, Optional[int]]:
    source_dim = _resolve_embedding_table_dim(config)
    if crop_embedding_dim is None:
        return source_dim, None

    if crop_embedding_dim <= 0:
        raise ValueError(f"crop_embedding_dim must be a positive integer, got {crop_embedding_dim}")
    if crop_embedding_dim > source_dim:
        raise ValueError(
            "crop_embedding_dim cannot exceed the source embedding dimension: "
            f"crop_embedding_dim={crop_embedding_dim}, source_dim={source_dim}"
        )
    return source_dim, crop_embedding_dim


def _validate_crop_num_embeddings(
    crop_num_embeddings: Optional[int],
) -> Optional[int]:
    if crop_num_embeddings is None:
        return None
    if crop_num_embeddings <= 0:
        raise ValueError(
            f"crop_num_embeddings must be a positive integer, got {crop_num_embeddings}"
        )
    return int(crop_num_embeddings)


def _build_target_hstu_config(
    source_config: Any,
    crop_embedding_dim: Optional[int],
    crop_num_embeddings: Optional[int] = None,
) -> Any:
    config_dict = dict(source_config.to_dict())
    if crop_embedding_dim is not None:
        updated_tables = []
        for table_cfg in config_dict.get("embedding_tables", []) or []:
            updated_cfg = dict(table_cfg)
            updated_cfg["embedding_dim"] = int(crop_embedding_dim)
            updated_tables.append(updated_cfg)
        config_dict["embedding_tables"] = updated_tables
    if crop_num_embeddings is not None:
        updated_tables = []
        for table_cfg in config_dict.get("embedding_tables", []) or []:
            updated_cfg = dict(table_cfg)
            updated_cfg["num_embeddings"] = min(
                int(updated_cfg["num_embeddings"]),
                int(crop_num_embeddings),
            )
            updated_tables.append(updated_cfg)
        config_dict["embedding_tables"] = updated_tables
        if updated_tables:
            config_dict["hash_size"] = max(
                int(table_cfg["num_embeddings"]) for table_cfg in updated_tables
            )

    from vllm_gr.config.hstu_config import HSTUConfig

    return HSTUConfig(**config_dict)


def _ordered_feature_mapping_from_source(
    source_mapping: Any,
    target_mapping: Any,
) -> "OrderedDict[str, int]":
    ordered: "OrderedDict[str, int]" = OrderedDict()
    source_mapping = source_mapping or {}
    target_mapping = target_mapping or {}

    for key in source_mapping.keys():
        if key in target_mapping:
            ordered[str(key)] = int(target_mapping[key])

    for key, value in target_mapping.items():
        if key not in ordered:
            ordered[str(key)] = int(value)

    return ordered


def _build_order_preserved_config_dict(
    target_config: Any,
    *,
    source_config: Optional[Any] = None,
) -> Dict[str, Any]:
    config_dict = target_config.to_dict()

    if source_config is None:
        return config_dict

    contextual_max_key = "contextual_feature_to_max_length"
    contextual_min_key = "contextual_feature_to_min_uih_length"

    if contextual_max_key in config_dict:
        config_dict[contextual_max_key] = _ordered_feature_mapping_from_source(
            getattr(source_config, contextual_max_key, {}),
            config_dict.get(contextual_max_key, {}),
        )

    if contextual_min_key in config_dict:
        config_dict[contextual_min_key] = _ordered_feature_mapping_from_source(
            getattr(source_config, contextual_min_key, {}),
            config_dict.get(contextual_min_key, {}),
        )

    return config_dict


def _write_target_config(
    *,
    output_dir: str,
    target_config: Any,
    source_config: Optional[Any] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    config_dict = _build_order_preserved_config_dict(
        target_config,
        source_config=source_config,
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("Wrote target config.json to %s", output_dir)


def _crop_sparse_weights(
    sparse_weights: Dict[str, torch.Tensor],
    *,
    source_embedding_dim: int,
    crop_embedding_dim: Optional[int],
    source_table_num_embeddings: Optional[Dict[str, int]] = None,
    crop_num_embeddings: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if crop_embedding_dim is None and crop_num_embeddings is None:
        return sparse_weights

    source_table_num_embeddings = source_table_num_embeddings or {}
    cropped: Dict[str, torch.Tensor] = {}
    for tensor_name, tensor in sparse_weights.items():
        if _is_embedding_weight_tensor(tensor_name, tensor):
            table_name = tensor_name.split(".embeddings.", 1)[1].rsplit(".weight", 1)[0]
            if crop_num_embeddings is not None:
                expected_num_embeddings = source_table_num_embeddings.get(table_name)
                if (
                    expected_num_embeddings is not None
                    and tensor.shape[0] != expected_num_embeddings
                ):
                    raise RuntimeError(
                        "Sparse embedding tensor row-count mismatch during crop: "
                        f"{tensor_name} has rows={tensor.shape[0]}, "
                        f"expected {expected_num_embeddings}"
                    )
                tensor = tensor[: min(int(tensor.shape[0]), int(crop_num_embeddings))].contiguous()
            if tensor.shape[-1] != source_embedding_dim:
                raise RuntimeError(
                    "Sparse embedding tensor shape mismatch during crop: "
                    f"{tensor_name} has last_dim={tensor.shape[-1]}, "
                    f"expected {source_embedding_dim}"
                )
            if crop_embedding_dim is not None:
                tensor = tensor[..., :crop_embedding_dim].contiguous()
            cropped[tensor_name] = tensor
        else:
            cropped[tensor_name] = tensor

    return cropped


def _write_conversion_meta(
    *,
    output_dir: str,
    input_path: str,
    output_format: str,
    dense_only: bool,
    split_weights: bool,
    dense_artifact: str,
    embedding_artifact: Optional[str],
    config: Any,
    source_embedding_dim: int,
    crop_embedding_dim: Optional[int],
    crop_num_embeddings: Optional[int],
) -> None:
    table_dims = {
        str(table_cfg["name"]): int(table_cfg["embedding_dim"])
        for table_cfg in getattr(config, "embedding_tables", []) or []
    }
    table_rows = {
        str(table_cfg["name"]): int(table_cfg["num_embeddings"])
        for table_cfg in getattr(config, "embedding_tables", []) or []
    }
    meta = {
        "input_path": input_path,
        "output_format": output_format,
        "dense_only": dense_only,
        "split_weights": split_weights,
        "dense_artifact": dense_artifact,
        "embedding_artifact": embedding_artifact,
        "source_embedding_dim": source_embedding_dim,
        "target_embedding_dim": (
            int(crop_embedding_dim) if crop_embedding_dim is not None else int(source_embedding_dim)
        ),
        "hstu_embedding_table_dim": int(getattr(config, "hstu_embedding_table_dim")),
        "embedding_table_dims": table_dims,
        "crop_num_embeddings": crop_num_embeddings,
        "embedding_table_num_embeddings": table_rows,
    }
    meta_path = os.path.join(output_dir, "conversion_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("Wrote conversion metadata to %s", meta_path)


def convert(
    input_path: str,
    output_dir: str,
    config_json: Optional[str] = None,
    dataset: Optional[str] = None,
    dense_only: bool = False,
    output_format: str = "safetensors",
    target_dtype: str = "bfloat16",
    crop_embedding_dim: Optional[int] = None,
    crop_num_embeddings: Optional[int] = None,
    split_weights: bool = False,
) -> str:
    """Convert DLRM_v3 checkpoint to HuggingFace-compatible format.

    Args:
        input_path: Path to checkpoint dir (containing non_sparse.ckpt).
        output_dir: Output directory for converted model.
        config_json: Path to existing config.json to copy.
        dataset: Dataset name to generate config from.
        dense_only: If True, skip sparse (embedding) weights.
        output_format: "safetensors" or "pt".
        target_dtype: Target dtype for dense tensors (default: "bfloat16").
            Sparse embedding tensors follow config.json table dtype settings.
        crop_embedding_dim: Optional reduced embedding dimension for sparse
            embedding tables. Dense weights stay unchanged.
        crop_num_embeddings: Optional cap for embedding-table row count.
            Keeps the first N rows of each embedding table and updates
            config.json num_embeddings accordingly.
        split_weights: If True, save dense and embedding tensors to separate
            files (dense.safetensors + embedding.safetensors, or .pt).

    Returns:
        Path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    _cleanup_previous_weight_artifacts(output_dir)

    torch_dtype = _resolve_dtype_name(target_dtype, default=torch.bfloat16)
    source_config = None
    source_embedding_dim = 0
    target_config = None
    crop_num_embeddings = _validate_crop_num_embeddings(crop_num_embeddings)

    # Load dense weights
    dense_dict = load_dense_weights(input_path)

    dense_converted: Dict[str, torch.Tensor] = {}
    for k, v in dense_dict.items():
        dense_converted["hstu." + k] = v.to(torch_dtype)

    # Load sparse weights if requested
    has_sparse_dir = os.path.isdir(os.path.join(input_path, "sparse"))
    include_sparse = has_sparse_dir and not dense_only
    sparse_converted: Dict[str, torch.Tensor] = {}

    if (crop_embedding_dim is not None or crop_num_embeddings is not None) and dense_only:
        raise RuntimeError(
            "crop settings cannot be used together with --dense-only "
            "because no sparse embedding weights are converted."
        )

    if include_sparse:
        with tempfile.TemporaryDirectory(prefix="gr_convert_cfg_") as source_config_dir:
            materialized_source_config = _materialize_source_config(
                config_json=config_json,
                dataset=dataset,
                output_dir=source_config_dir,
            )
            if materialized_source_config is None:
                raise RuntimeError(
                    "Sparse checkpoint conversion requires config.json when sparse/ "
                    "weights are present. Provide --config-json, --dataset, or "
                    "use --dense-only."
                )
            source_config = _load_hstu_config(materialized_source_config)
            source_embedding_dim, crop_embedding_dim = _validate_crop_embedding_dim(
                source_config,
                crop_embedding_dim,
            )
            target_config = _build_target_hstu_config(
                source_config,
                crop_embedding_dim,
                crop_num_embeddings,
            )
            _write_target_config(
                output_dir=output_dir,
                target_config=target_config,
                source_config=source_config,
            )

            _init_single_process_distributed()
            try:
                sparse_dict = load_sparse_weights(
                    input_path,
                    materialized_source_config,
                    dense_dict,
                )
                sparse_dict = _crop_sparse_weights(
                    sparse_dict,
                    source_embedding_dim=source_embedding_dim,
                    crop_embedding_dim=crop_embedding_dim,
                    source_table_num_embeddings=_build_embedding_table_num_embeddings_map(
                        source_config
                    ),
                    crop_num_embeddings=crop_num_embeddings,
                )
                for k, v in sparse_dict.items():
                    sparse_target_dtype = _resolve_sparse_tensor_target_dtype(
                        target_config,
                        k,
                        v,
                    )
                    sparse_converted["hstu." + k] = v.to(sparse_target_dtype)
            except Exception as e:
                logger.error(
                    "Sparse loading failed: %s\n"
                    "Ensure torchrec is installed. You can also try:\n"
                    "  torchrun --nproc_per_node=1 scripts/convert_gr_checkpoint.py ...\n"
                    "Or use --dense-only to skip embeddings.",
                    e,
                )
                raise
            finally:
                _destroy_distributed()
    elif has_sparse_dir:
        if config_json or dataset:
            with tempfile.TemporaryDirectory(prefix="gr_convert_cfg_") as source_config_dir:
                materialized_source_config = _materialize_source_config(
                    config_json=config_json,
                    dataset=dataset,
                    output_dir=source_config_dir,
                )
                if materialized_source_config is not None:
                    source_config = _load_hstu_config(materialized_source_config)
                    source_embedding_dim, _ = _validate_crop_embedding_dim(
                        source_config,
                        crop_embedding_dim=None,
                    )
                    target_config = _build_target_hstu_config(
                        source_config,
                        None,
                        crop_num_embeddings,
                    )
                    _write_target_config(
                        output_dir=output_dir,
                        target_config=target_config,
                        source_config=source_config,
                    )
        elif crop_embedding_dim is not None or crop_num_embeddings is not None:
            raise RuntimeError(
                "crop settings require a config source via --config-json or --dataset."
            )
        logger.info("Sparse directory found but --dense-only specified, skipping embeddings")
    elif config_json or dataset:
        with tempfile.TemporaryDirectory(prefix="gr_convert_cfg_") as source_config_dir:
            materialized_source_config = _materialize_source_config(
                config_json=config_json,
                dataset=dataset,
                output_dir=source_config_dir,
            )
            if materialized_source_config is not None:
                source_config = _load_hstu_config(materialized_source_config)
                source_embedding_dim, crop_embedding_dim = _validate_crop_embedding_dim(
                    source_config,
                    crop_embedding_dim,
                )
                target_config = _build_target_hstu_config(
                    source_config,
                    crop_embedding_dim,
                    crop_num_embeddings,
                )
                _write_target_config(
                    output_dir=output_dir,
                    target_config=target_config,
                    source_config=source_config,
                )
    elif crop_embedding_dim is not None or crop_num_embeddings is not None:
        raise RuntimeError("crop settings require a config source via --config-json or --dataset.")

    if target_config is None and source_config is not None:
        target_config = source_config

    if has_sparse_dir and dense_only:
        logger.info("Sparse directory found but --dense-only specified, skipping embeddings")

    # Save
    dense_artifact: Optional[str] = None
    embedding_artifact: Optional[str] = None
    converted = {**dense_converted, **sparse_converted}

    if output_format == "safetensors":
        try:
            from safetensors.torch import save_file

            if split_weights:
                dense_artifact = "dense.safetensors"
                save_file(
                    dense_converted,
                    os.path.join(output_dir, dense_artifact),
                )
                if sparse_converted:
                    embedding_artifact = "embedding.safetensors"
                    save_file(
                        sparse_converted,
                        os.path.join(output_dir, embedding_artifact),
                    )
                out_file = os.path.join(output_dir, dense_artifact)
            else:
                dense_artifact = "model.safetensors"
                out_file = os.path.join(output_dir, dense_artifact)
                save_file(converted, out_file)
        except ImportError:
            raise RuntimeError(
                "Output format 'safetensors' requested, but safetensors is not "
                "installed. Install `safetensors` or use `--format pt`."
            )
    else:
        if split_weights:
            dense_artifact = "dense.pt"
            torch.save(dense_converted, os.path.join(output_dir, dense_artifact))
            if sparse_converted:
                embedding_artifact = "embedding.pt"
                torch.save(
                    sparse_converted,
                    os.path.join(output_dir, embedding_artifact),
                )
            out_file = os.path.join(output_dir, dense_artifact)
        else:
            dense_artifact = "model.pt"
            out_file = os.path.join(output_dir, dense_artifact)
            torch.save(converted, out_file)

    if target_config is not None:
        _write_conversion_meta(
            output_dir=output_dir,
            input_path=input_path,
            output_format=output_format,
            dense_only=dense_only,
            split_weights=split_weights,
            dense_artifact=dense_artifact or os.path.basename(out_file),
            embedding_artifact=embedding_artifact,
            config=target_config,
            source_embedding_dim=source_embedding_dim,
            crop_embedding_dim=crop_embedding_dim,
            crop_num_embeddings=crop_num_embeddings,
        )

    total_params = sum(v.numel() for v in converted.values())
    sparse_count = sum(1 for k in converted if "embedding_collection" in k)
    dense_count = len(converted) - sparse_count
    logger.info(
        "Converted %d tensors (%d parameters) -> %s",
        len(converted),
        total_params,
        out_file,
    )
    sparse_dtype_summary = (
        "per-table config"
        if sparse_count and target_config is not None
        else ("original tensor dtypes" if sparse_count else "n/a")
    )
    logger.info(
        "  Dense: %d tensors (target dtype=%s), Sparse: %d tensors (target dtype=%s)",
        dense_count,
        torch_dtype,
        sparse_count,
        sparse_dtype_summary,
    )
    if crop_embedding_dim is not None:
        logger.info(
            "  Embedding crop: source_dim=%d -> target_dim=%d "
            "(dense weights preserved at runtime dim=%d)",
            source_embedding_dim,
            crop_embedding_dim,
            source_embedding_dim,
        )
    if crop_num_embeddings is not None:
        logger.info(
            "  Embedding rows: capped each table at first %d rows",
            crop_num_embeddings,
        )
    if split_weights:
        logger.info(
            "  Split artifacts: dense=%s, embedding=%s",
            dense_artifact,
            embedding_artifact or "n/a",
        )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DLRM_v3 training checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to DLRM_v3 checkpoint directory (containing non_sparse.ckpt)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for converted HuggingFace model",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Path to existing config.json to copy into output directory",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=DATASET_CHOICES,
        help="Dataset name to auto-generate config.json",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        default=False,
        help=(
            "Only convert dense weights (skip sparse embeddings). "
            "Matches upstream inference pattern where embeddings come "
            "from an external feature store."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["safetensors", "pt"],
        default="safetensors",
        help="Output weight format (default: safetensors)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Target dtype for all tensors (default: bfloat16)",
    )
    parser.add_argument(
        "--crop-embedding-dim",
        type=int,
        default=None,
        help=(
            "Optionally crop sparse embedding table weights to this dimension. "
            "Dense weights stay unchanged; runtime pads embeddings back to "
            "hstu_embedding_table_dim before dense layers."
        ),
    )
    parser.add_argument(
        "--crop-num-embeddings",
        type=int,
        default=None,
        help=(
            "Optionally crop each embedding table to keep only the first N rows. "
            "Converted config.json num_embeddings values are reduced accordingly."
        ),
    )
    parser.add_argument(
        "--split-weights",
        action="store_true",
        default=False,
        help=(
            "Write dense and embedding weights into separate files "
            "(dense.safetensors + embedding.safetensors, or .pt)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    convert(
        input_path=args.input_path,
        output_dir=args.output_dir,
        config_json=args.config_json,
        dataset=args.dataset,
        dense_only=args.dense_only,
        output_format=args.format,
        target_dtype=args.dtype,
        crop_embedding_dim=args.crop_embedding_dim,
        crop_num_embeddings=args.crop_num_embeddings,
        split_weights=args.split_weights,
    )


if __name__ == "__main__":
    main()
