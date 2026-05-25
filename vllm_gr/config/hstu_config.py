# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure HSTU configuration definitions and validation helpers."""

from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig

from vllm_gr.config.multitask_types import MultitaskTaskType

_VALID_TASK_TYPES = {
    "binary_classification",
    "regression",
}


def _validate_feature_name_list(
    group_name: str,
    feature_names: List[str],
) -> List[str]:
    """Validate a configured feature-name list once at config init time."""
    validated: List[str] = []
    seen: set[str] = set()

    for key in feature_names:
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"{group_name} contains an invalid key: {key!r}"
            )
        if key in seen:
            raise ValueError(
                f"{group_name} contains duplicate key: {key!r}"
            )
        validated.append(key)
        seen.add(key)

    return validated


def _validate_feature_length_mapping(
    group_name: str,
    feature_lengths: Dict[str, int],
) -> Dict[str, int]:
    """Validate configured feature-length mappings once at config init time."""
    validated_keys = _validate_feature_name_list(
        group_name,
        list(feature_lengths.keys()),
    )
    validated: Dict[str, int] = {}
    for key in validated_keys:
        value = feature_lengths[key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{group_name}[{key!r}] must be a positive integer, got {value!r}"
            )
        validated[key] = value
    return validated


def _validate_merge_mapping(
    merge_mapping: List[List[str]],
) -> List[List[str]]:
    """Validate the UIH/candidate merge mapping shape and key names."""
    validated: List[List[str]] = []

    for pair in merge_mapping:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                "merge_uih_candidate_feature_mapping entries must be "
                "two-element lists"
            )
        validated.append(_validate_feature_name_list("merge_mapping_pair", pair))

    return validated


def _validate_embedding_tables(
    embedding_tables: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Validate embedding table definitions once at config init time."""
    if embedding_tables is None:
        return []

    validated_tables: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, table in enumerate(embedding_tables):
        if not isinstance(table, dict):
            raise ValueError(
                f"embedding_tables[{index}] must be a dict, got {type(table)!r}"
            )

        name = table.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"embedding_tables[{index}]['name'] must be a non-empty string"
            )
        if name in seen_names:
            raise ValueError(
                f"embedding_tables contains duplicate table name: {name!r}"
            )
        seen_names.add(name)

        num_embeddings = table.get("num_embeddings")
        if not isinstance(num_embeddings, int) or num_embeddings <= 0:
            raise ValueError(
                f"embedding_tables[{index}]['num_embeddings'] must be a positive "
                f"integer, got {num_embeddings!r}"
            )

        embedding_dim = table.get("embedding_dim")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(
                f"embedding_tables[{index}]['embedding_dim'] must be a positive "
                f"integer, got {embedding_dim!r}"
            )

        feature_names = _validate_feature_name_list(
            f"embedding_tables[{index}]['feature_names']",
            table.get("feature_names", [name]),
        )

        validated_table = dict(table)
        validated_table["name"] = name
        validated_table["num_embeddings"] = num_embeddings
        validated_table["embedding_dim"] = embedding_dim
        validated_table["feature_names"] = feature_names
        validated_tables.append(validated_table)

    return validated_tables


def _validate_feature_membership(
    feature_name: str,
    declared_features: List[str],
    group_name: str,
) -> None:
    if feature_name and feature_name not in declared_features:
        raise ValueError(
            f"{feature_name!r} must be declared in {group_name}"
        )


def _validate_embedding_feature_coverage(
    embedding_tables: List[Dict[str, Any]],
    expected_feature_names: List[str],
) -> None:
    if not expected_feature_names:
        return

    covered_feature_names = {
        feature_name
        for table in embedding_tables
        for feature_name in table["feature_names"]
    }
    missing = sorted(set(expected_feature_names) - covered_feature_names)
    if missing:
        raise ValueError(
            "embedding_tables must cover every declared embedding feature; "
            f"missing {missing!r}"
        )


def _validate_and_resolve_task_types(
    task_names: List[str],
    task_types: Optional[List[str]],
) -> List[MultitaskTaskType]:
    """Validate task types aligned with task_names and resolve enums."""
    if task_types is None:
        return [MultitaskTaskType.BINARY_CLASSIFICATION] * len(task_names)

    if not isinstance(task_types, list):
        raise ValueError("task_types must be a list of strings")

    if len(task_types) != len(task_names):
        raise ValueError(
            "task_types must have the same length as task_names: "
            f"len(task_types)={len(task_types)}, len(task_names)={len(task_names)}"
        )

    validated: List[MultitaskTaskType] = []
    for index, task_type in enumerate(task_types):
        if not isinstance(task_type, str) or task_type not in _VALID_TASK_TYPES:
            raise ValueError(
                f"task_types[{index}] must be one of "
                f"{sorted(_VALID_TASK_TYPES)!r}, got {task_type!r}"
            )
        validated.append(MultitaskTaskType[task_type.upper()])

    return validated


def _resolve_candidate_primary_id_feature_name(
    config: "HSTUConfig",
    *,
    require_resolution: bool,
) -> str:
    explicit = str(getattr(config, "candidate_primary_id_feature_name", "") or "")
    merge_mapping = list(
        getattr(config, "merge_uih_candidate_feature_mapping", []) or []
    )
    primary_uih_feature_name = str(
        getattr(config, "uih_post_id_feature_name", "") or ""
    )

    derived_candidates = [
        candidate_feature_name
        for uih_feature_name, candidate_feature_name in merge_mapping
        if uih_feature_name == primary_uih_feature_name
    ]

    if derived_candidates:
        first_candidate = derived_candidates[0]
        conflicting_candidates = [
            candidate
            for candidate in derived_candidates[1:]
            if candidate != first_candidate
        ]
        if conflicting_candidates:
            raise ValueError(
                "merge_uih_candidate_feature_mapping defines multiple candidate "
                "primary-id mappings for "
                f"{primary_uih_feature_name!r}: {derived_candidates!r}"
            )
        derived = first_candidate
    else:
        derived = ""

    if explicit:
        if derived and explicit != derived:
            raise ValueError(
                "candidate_primary_id_feature_name conflicts with "
                "merge_uih_candidate_feature_mapping for "
                f"{primary_uih_feature_name!r}: explicit={explicit!r}, "
                f"derived={derived!r}"
            )
        return explicit

    if derived:
        return derived

    if require_resolution:
        raise ValueError(
            "Unable to resolve candidate_primary_id_feature_name. Configure it "
            "explicitly or provide a merge_uih_candidate_feature_mapping entry "
            f"for the primary UIH feature {primary_uih_feature_name!r}."
        )

    return ""


def _validate_hstu_config_consistency(config: "HSTUConfig") -> None:
    _validate_feature_membership(
        config.uih_post_id_feature_name,
        config.user_embedding_feature_names,
        "user_embedding_feature_names",
    )
    _validate_feature_membership(
        config.uih_post_id_feature_name,
        config.hstu_uih_feature_names,
        "hstu_uih_feature_names",
    )

    if config.uih_action_time_feature_name:
        _validate_feature_membership(
            config.uih_action_time_feature_name,
            config.hstu_uih_feature_names,
            "hstu_uih_feature_names",
        )

    if config.uih_weight_feature_name:
        _validate_feature_membership(
            config.uih_weight_feature_name,
            config.hstu_uih_feature_names,
            "hstu_uih_feature_names",
        )

    if config.candidates_weight_feature_name:
        _validate_feature_membership(
            config.candidates_weight_feature_name,
            config.hstu_candidate_feature_names,
            "hstu_candidate_feature_names",
        )

    if config.candidates_watchtime_feature_name:
        _validate_feature_membership(
            config.candidates_watchtime_feature_name,
            config.hstu_candidate_feature_names,
            "hstu_candidate_feature_names",
        )

    if config.candidates_querytime_feature_name:
        _validate_feature_membership(
            config.candidates_querytime_feature_name,
            config.hstu_candidate_feature_names,
            "hstu_candidate_feature_names",
        )

    if getattr(config, "candidate_primary_id_feature_name", ""):
        _validate_feature_membership(
            config.candidate_primary_id_feature_name,
            config.hstu_candidate_feature_names,
            "hstu_candidate_feature_names",
        )
        _validate_feature_membership(
            config.candidate_primary_id_feature_name,
            config.item_embedding_feature_names,
            "item_embedding_feature_names",
        )

    missing_user_features = sorted(
        set(config.user_embedding_feature_names) - set(config.hstu_uih_feature_names)
    )
    if missing_user_features:
        raise ValueError(
            "user_embedding_feature_names must be a subset of "
            f"hstu_uih_feature_names; missing {missing_user_features!r}"
        )

    missing_item_features = sorted(
        set(config.item_embedding_feature_names)
        - set(config.hstu_candidate_feature_names)
    )
    if missing_item_features:
        raise ValueError(
            "item_embedding_feature_names must be a subset of "
            f"hstu_candidate_feature_names; missing {missing_item_features!r}"
        )

    for uih_feature_name, candidate_feature_name in (
        config.merge_uih_candidate_feature_mapping
    ):
        _validate_feature_membership(
            uih_feature_name,
            config.hstu_uih_feature_names,
            "hstu_uih_feature_names",
        )
        _validate_feature_membership(
            candidate_feature_name,
            config.hstu_candidate_feature_names,
            "hstu_candidate_feature_names",
        )

    for feature_name in config.contextual_feature_to_max_length:
        _validate_feature_membership(
            feature_name,
            config.hstu_uih_feature_names,
            "hstu_uih_feature_names",
        )

    for feature_name in config.contextual_feature_to_min_uih_length:
        _validate_feature_membership(
            feature_name,
            config.hstu_uih_feature_names,
            "hstu_uih_feature_names",
        )
        if feature_name not in config.contextual_feature_to_max_length:
            raise ValueError(
                "contextual_feature_to_min_uih_length keys must also be "
                "present in contextual_feature_to_max_length; "
                f"unexpected key {feature_name!r}"
            )

    _validate_embedding_feature_coverage(
        config.embedding_tables,
        config.user_embedding_feature_names + config.item_embedding_feature_names,
    )

    if config.embedding_tables:
        largest_table_size = max(
            table["num_embeddings"] for table in config.embedding_tables
        )
        if config.hash_size < largest_table_size:
            raise ValueError(
                "hash_size must be greater than or equal to every embedding table "
                f"size; got hash_size={config.hash_size}, "
                f"largest_table_size={largest_table_size}"
            )

    resolved_candidate_primary_id_feature_name = (
        _resolve_candidate_primary_id_feature_name(
            config,
            require_resolution=False,
        )
    )
    if resolved_candidate_primary_id_feature_name:
        _validate_feature_membership(
            resolved_candidate_primary_id_feature_name,
            config.item_embedding_feature_names,
            "item_embedding_feature_names",
        )


def _read_hstu_config_value(
    config: Any,
    key: str,
    default: Any,
) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def get_hstu_contextual_seq_len(config: Any) -> int:
    """Return the fixed contextual-prefix token count for one request."""
    contextual_mapping = _read_hstu_config_value(
        config,
        "contextual_feature_to_max_length",
        {},
    ) or {}
    return int(sum(int(length) for length in contextual_mapping.values()))


def get_hstu_full_sequence_len(
    *,
    max_seq_len: int,
    max_candidates: int,
    contextual_seq_len: int = 0,
) -> int:
    """Return the scheduler-visible HSTU full sequence length."""
    return int(contextual_seq_len) + int(max_seq_len) + int(max_candidates)


def get_hstu_max_model_len(
    config: Any,
    *,
    max_seq_len: Optional[int] = None,
    max_candidates: Optional[int] = None,
) -> int:
    """Derive HSTU max_model_len using full visible-sequence semantics."""
    resolved_max_seq_len = int(
        max_seq_len
        if max_seq_len is not None
        else _read_hstu_config_value(config, "max_seq_len", 0)
    )
    resolved_max_candidates = int(
        max_candidates
        if max_candidates is not None
        else max(
            int(_read_hstu_config_value(config, "max_num_candidates", 0)),
            int(_read_hstu_config_value(config, "max_num_candidates_inference", 0)),
        )
    )
    return get_hstu_full_sequence_len(
        max_seq_len=resolved_max_seq_len,
        max_candidates=resolved_max_candidates,
        contextual_seq_len=get_hstu_contextual_seq_len(config),
    )


class HSTUConfig(PretrainedConfig):
    """PretrainedConfig subclass for HSTU models (model_type="hstu")."""

    model_type = "hstu"
    architectures = ["HSTUEmbeddingModel"]
    is_encoder_decoder = False
    is_text_encoder = True
    is_attention_free = False

    def __init__(
        self,
        max_seq_len: int = 1024,
        max_num_candidates: int = 100,
        max_num_candidates_inference: int = 5,
        hstu_num_heads: int = 4,
        hstu_attn_linear_dim: int = 256,
        hstu_attn_qk_dim: int = 128,
        hstu_attn_num_layers: int = 5,
        hstu_embedding_table_dim: int = 192,
        hstu_preprocessor_hidden_dim: int = 256,
        hstu_transducer_embedding_dim: int = 512,
        hstu_group_norm: bool = False,
        hstu_input_dropout_ratio: float = 0.2,
        hstu_linear_dropout_rate: float = 0.2,
        contextual_feature_to_max_length: Optional[Dict[str, int]] = None,
        contextual_feature_to_min_uih_length: Optional[Dict[str, int]] = None,
        candidates_weight_feature_name: str = "",
        candidates_watchtime_feature_name: str = "",
        candidates_querytime_feature_name: str = "",
        user_embedding_feature_names: Optional[List[str]] = None,
        item_embedding_feature_names: Optional[List[str]] = None,
        uih_post_id_feature_name: str = "item_id",
        uih_action_time_feature_name: str = "",
        uih_weight_feature_name: str = "",
        hstu_uih_feature_names: Optional[List[str]] = None,
        hstu_candidate_feature_names: Optional[List[str]] = None,
        candidate_primary_id_feature_name: str = "",
        merge_uih_candidate_feature_mapping: Optional[List[List[str]]] = None,
        action_weights: Optional[List[int]] = None,
        action_embedding_init_std: float = 0.1,
        causal_multitask_weights: float = 0.2,
        enable_postprocessor: bool = True,
        use_layer_norm_postprocessor: bool = False,
        enable_kv_cache: bool = True,
        hash_size: int = 10_000_000,
        task_names: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None,
        embedding_tables: Optional[List[Dict[str, Any]]] = None,
        embedding_table_data_type: str = "fp16",
        is_inference: bool = True,
        is_text_encoder: bool = True,
        is_encoder_decoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.architectures = list(type(self).architectures)

        if "torch_dtype" not in kwargs:
            self.torch_dtype = "bfloat16"

        self.is_text_encoder = is_text_encoder
        self.is_encoder_decoder = is_encoder_decoder
        self.is_attention_free = False

        self.max_seq_len = max_seq_len
        self.max_num_candidates = max_num_candidates
        self.max_num_candidates_inference = max_num_candidates_inference

        self.hstu_num_heads = hstu_num_heads
        self.hstu_attn_linear_dim = hstu_attn_linear_dim
        self.hstu_attn_qk_dim = hstu_attn_qk_dim
        self.hstu_attn_num_layers = hstu_attn_num_layers
        self.hstu_embedding_table_dim = hstu_embedding_table_dim
        self.hstu_preprocessor_hidden_dim = hstu_preprocessor_hidden_dim
        self.hstu_transducer_embedding_dim = hstu_transducer_embedding_dim
        self.hstu_group_norm = hstu_group_norm
        self.hstu_input_dropout_ratio = hstu_input_dropout_ratio
        self.hstu_linear_dropout_rate = hstu_linear_dropout_rate

        self.contextual_feature_to_max_length = _validate_feature_length_mapping(
            "contextual_feature_to_max_length",
            contextual_feature_to_max_length or {},
        )
        self.contextual_feature_to_min_uih_length = _validate_feature_length_mapping(
            "contextual_feature_to_min_uih_length",
            contextual_feature_to_min_uih_length or {},
        )
        self.candidates_weight_feature_name = candidates_weight_feature_name
        self.candidates_watchtime_feature_name = candidates_watchtime_feature_name
        self.candidates_querytime_feature_name = candidates_querytime_feature_name

        self.user_embedding_feature_names = _validate_feature_name_list(
            "user_embedding_feature_names",
            user_embedding_feature_names or ["item_id"],
        )
        self.item_embedding_feature_names = _validate_feature_name_list(
            "item_embedding_feature_names",
            item_embedding_feature_names or ["item_candidate_id"],
        )
        self.uih_post_id_feature_name = uih_post_id_feature_name
        self.uih_action_time_feature_name = uih_action_time_feature_name
        self.uih_weight_feature_name = uih_weight_feature_name
        self.hstu_uih_feature_names = _validate_feature_name_list(
            "hstu_uih_feature_names",
            hstu_uih_feature_names or [self.uih_post_id_feature_name],
        )
        self.hstu_candidate_feature_names = _validate_feature_name_list(
            "hstu_candidate_feature_names",
            hstu_candidate_feature_names or list(self.item_embedding_feature_names),
        )
        self.candidate_primary_id_feature_name = candidate_primary_id_feature_name
        self.merge_uih_candidate_feature_mapping = _validate_merge_mapping(
            merge_uih_candidate_feature_mapping or []
        )
        self.action_weights = action_weights or []
        self.action_embedding_init_std = action_embedding_init_std
        self.causal_multitask_weights = causal_multitask_weights
        self.enable_postprocessor = enable_postprocessor
        self.use_layer_norm_postprocessor = use_layer_norm_postprocessor
        self.enable_kv_cache = bool(enable_kv_cache)

        if not isinstance(hash_size, int) or hash_size <= 0:
            raise ValueError(
                f"hash_size must be a positive integer, got {hash_size!r}"
            )
        self.hash_size = hash_size
        self.task_names = task_names or ["is_click"]
        self.task_types: List[MultitaskTaskType] = _validate_and_resolve_task_types(
            self.task_names,
            task_types,
        )
        self.embedding_table_data_type = embedding_table_data_type
        validated_embedding_tables = _validate_embedding_tables(embedding_tables)
        if validated_embedding_tables:
            self.embedding_tables = validated_embedding_tables
        else:
            self.embedding_tables = _validate_embedding_tables(
                [
                    {
                        "name": "item_embedding",
                        "embedding_dim": hstu_embedding_table_dim,
                        "num_embeddings": hash_size,
                        "feature_names": ["item_id", "item_candidate_id"],
                    }
                ]
            )
        if not is_inference:
            raise ValueError(
                "HSTUConfig only supports is_inference=True in this vllm-GR."
            )

        _validate_hstu_config_consistency(self)
        self.candidate_primary_id_feature_name = (
            _resolve_candidate_primary_id_feature_name(
                self,
                require_resolution=False,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        config_dict = super().to_dict()
        config_dict["task_types"] = [
            task_type.name.lower()
            for task_type in self.task_types
        ]
        return config_dict


__all__ = [
    "HSTUConfig",
    "get_hstu_contextual_seq_len",
    "get_hstu_full_sequence_len",
    "get_hstu_max_model_len",
]
