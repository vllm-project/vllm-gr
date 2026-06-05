# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Original dataset configs from Meta Platforms, Inc. (generative-recommenders)

"""
Dataset-specific configuration factory functions.

Ports the logic from generative_recommenders/dlrm_v3/configs.py to provide
HSTUConfig field overrides and embedding table configs per dataset.
"""

from typing import Any, Dict, List

HSTU_EMBEDDING_DIM = 512
HASH_SIZE = 10_000_000
HASH_SIZE_1B = 1_000_000_000

_MOVIELENS_DATASETS = {
    "movielens-1m",
    "movielens-20m",
    "movielens-13b",
    "movielens-18b",
}
_STREAMING_DATASETS = {
    "streaming",
    "streaming-400m",
    "streaming-200b",
    "streaming-100b",
    "sampled-streaming-100b",
}
_KUAIRAND_DATASETS = {
    "kuairand-1k",
    "kuairand-27k",
}
_SYNTHETIC_DATASETS = {
    "debug",
}

SUPPORTED_DATASETS = sorted(
    _MOVIELENS_DATASETS | _STREAMING_DATASETS | _KUAIRAND_DATASETS | _SYNTHETIC_DATASETS
)


def _resolve_dataset_family(dataset: str) -> str:
    """Return the canonical dataset family for supported presets."""
    if dataset in _MOVIELENS_DATASETS:
        return "movielens"
    if dataset in _STREAMING_DATASETS:
        return "streaming"
    if dataset in _KUAIRAND_DATASETS:
        return "kuairand"
    if dataset in _SYNTHETIC_DATASETS:
        return "synthetic"
    raise ValueError(f"Unsupported dataset: {dataset!r}. Supported datasets: {SUPPORTED_DATASETS}")


def get_hstu_config_overrides(dataset: str) -> Dict[str, Any]:
    """Return HSTUConfig field overrides for a given dataset.

    Aligns with generative_recommenders/dlrm_v3/configs.py get_hstu_configs().
    """
    overrides: Dict[str, Any] = {
        "hstu_num_heads": 4,
        "hstu_attn_linear_dim": 128,
        "hstu_attn_qk_dim": 128,
        "hstu_attn_num_layers": 5,
        "hstu_embedding_table_dim": HSTU_EMBEDDING_DIM,
        "hstu_preprocessor_hidden_dim": 256,
        "hstu_transducer_embedding_dim": HSTU_EMBEDDING_DIM,
        "hstu_input_dropout_ratio": 0.2,
        "hstu_linear_dropout_rate": 0.1,
        "hash_size": HASH_SIZE,
    }
    dataset_family = _resolve_dataset_family(dataset)

    if dataset_family == "movielens":
        if dataset != "movielens-1m":
            overrides["hash_size"] = HASH_SIZE_1B
        overrides["max_num_candidates"] = 10
        overrides["max_num_candidates_inference"] = (
            2048 if dataset in {"movielens-13b", "movielens-18b"} else 5
        )
        overrides["contextual_feature_to_max_length"] = (
            {"user_id": 1, "sex": 1, "age_group": 1, "occupation": 1, "zip_code": 1}
            if dataset == "movielens-1m"
            else {"user_id": 1}
        )
        overrides["contextual_feature_to_min_uih_length"] = (
            {"user_id": 20, "sex": 20, "age_group": 20, "occupation": 20, "zip_code": 20}
            if dataset == "movielens-1m"
            else {"user_id": 20}
        )
        overrides["candidates_weight_feature_name"] = "item_action_weights"
        overrides["candidates_watchtime_feature_name"] = "item_movie_rating"
        overrides["candidates_querytime_feature_name"] = "item_query_time"
        overrides["user_embedding_feature_names"] = (
            ["movie_id", "user_id", "sex", "age_group", "occupation", "zip_code"]
            if dataset == "movielens-1m"
            else ["movie_id", "user_id"]
        )
        overrides["item_embedding_feature_names"] = ["item_movie_id"]
        overrides["uih_post_id_feature_name"] = "movie_id"
        overrides["uih_action_time_feature_name"] = "action_timestamp"
        overrides["uih_weight_feature_name"] = "item_weights"
        overrides["hstu_uih_feature_names"] = (
            [
                "user_id",
                "sex",
                "age_group",
                "occupation",
                "zip_code",
                "movie_id",
                "movie_rating",
                "action_timestamp",
                "item_weights",
                "dummy_watch_time",
            ]
            if dataset == "movielens-1m"
            else [
                "user_id",
                "movie_id",
                "movie_rating",
                "action_timestamp",
                "item_weights",
                "dummy_watch_time",
            ]
        )
        overrides["hstu_candidate_feature_names"] = [
            "item_movie_id",
            "item_movie_rating",
            "item_query_time",
            "item_action_weights",
            "item_dummy_watchtime",
        ]
        overrides["merge_uih_candidate_feature_mapping"] = [
            ["movie_id", "item_movie_id"],
            ["movie_rating", "item_movie_rating"],
            ["action_timestamp", "item_query_time"],
            ["item_weights", "item_action_weights"],
            ["dummy_watch_time", "item_dummy_watchtime"],
        ]
        overrides["action_weights"] = [1, 2, 4, 8, 16]
        overrides["task_names"] = ["rating"]
        overrides["task_types"] = ["binary_classification"]

    elif dataset_family == "streaming":
        overrides["hash_size"] = HASH_SIZE_1B
        overrides["max_num_candidates"] = 32
        overrides["max_num_candidates_inference"] = 2048
        overrides["contextual_feature_to_max_length"] = {"user_id": 1}
        overrides["contextual_feature_to_min_uih_length"] = {"user_id": 20}
        overrides["candidates_weight_feature_name"] = "item_action_weights"
        overrides["candidates_watchtime_feature_name"] = "item_candidate_rating"
        overrides["candidates_querytime_feature_name"] = "item_query_time"
        overrides["user_embedding_feature_names"] = [
            "item_id",
            "user_id",
            "item_category_id",
        ]
        overrides["item_embedding_feature_names"] = [
            "item_candidate_id",
            "item_candidate_category_id",
        ]
        overrides["uih_post_id_feature_name"] = "item_id"
        overrides["uih_action_time_feature_name"] = "action_timestamp"
        overrides["uih_weight_feature_name"] = "item_weights"
        overrides["hstu_uih_feature_names"] = [
            "user_id",
            "item_id",
            "item_rating",
            "action_timestamp",
            "item_weights",
            "dummy_watch_time",
            "item_category_id",
        ]
        overrides["hstu_candidate_feature_names"] = [
            "item_candidate_id",
            "item_candidate_rating",
            "item_query_time",
            "item_action_weights",
            "item_dummy_watchtime",
            "item_candidate_category_id",
        ]
        overrides["merge_uih_candidate_feature_mapping"] = [
            ["item_id", "item_candidate_id"],
            ["item_rating", "item_candidate_rating"],
            ["action_timestamp", "item_query_time"],
            ["item_weights", "item_action_weights"],
            ["dummy_watch_time", "item_dummy_watchtime"],
            ["item_category_id", "item_candidate_category_id"],
        ]
        overrides["action_weights"] = [1, 2, 4, 8, 16]
        overrides["action_embedding_init_std"] = 5.0
        overrides["task_names"] = ["rating"]
        overrides["task_types"] = ["binary_classification"]

    elif dataset_family == "kuairand":
        overrides["contextual_feature_to_max_length"] = {
            "user_id": 1,
            "user_active_degree": 1,
            "follow_user_num_range": 1,
            "fans_user_num_range": 1,
            "friend_user_num_range": 1,
            "register_days_range": 1,
        }
        overrides["candidates_weight_feature_name"] = "item_action_weight"
        overrides["candidates_watchtime_feature_name"] = "item_target_watchtime"
        overrides["candidates_querytime_feature_name"] = "item_query_time"
        overrides["user_embedding_feature_names"] = [
            "video_id",
            "user_id",
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ]
        overrides["item_embedding_feature_names"] = ["item_video_id"]
        overrides["uih_post_id_feature_name"] = "video_id"
        overrides["uih_action_time_feature_name"] = "action_timestamp"
        overrides["uih_weight_feature_name"] = "action_weight"
        overrides["hstu_uih_feature_names"] = [
            "user_id",
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
            "video_id",
            "action_timestamp",
            "action_weight",
            "watch_time",
        ]
        overrides["hstu_candidate_feature_names"] = [
            "item_video_id",
            "item_action_weight",
            "item_target_watchtime",
            "item_query_time",
        ]
        overrides["merge_uih_candidate_feature_mapping"] = [
            ["video_id", "item_video_id"],
            ["action_timestamp", "item_query_time"],
            ["action_weight", "item_action_weight"],
            ["watch_time", "item_target_watchtime"],
        ]
        overrides["action_weights"] = [1, 2, 4, 8, 16, 32, 64, 128]
        overrides["task_names"] = [
            "is_click",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_hate",
            "long_view",
            "is_profile_enter",
        ]
        overrides["task_types"] = [
            "binary_classification",
        ] * len(overrides["task_names"])

    else:
        overrides["contextual_feature_to_max_length"] = {
            "viewer_id": 1,
            "dummy_contextual": 1,
        }
        overrides["contextual_feature_to_min_uih_length"] = {
            "viewer_id": 128,
            "dummy_contextual": 128,
        }
        overrides["candidates_weight_feature_name"] = "item_action_weight"
        overrides["candidates_watchtime_feature_name"] = "item_target_watchtime"
        overrides["candidates_querytime_feature_name"] = "item_query_time"
        overrides["user_embedding_feature_names"] = [
            "uih_post_id",
            "uih_owner_id",
            "viewer_id",
            "dummy_contextual",
        ]
        overrides["item_embedding_feature_names"] = [
            "item_post_id",
            "item_owner_id",
        ]
        overrides["uih_post_id_feature_name"] = "uih_post_id"
        overrides["uih_action_time_feature_name"] = "uih_action_time"
        overrides["hstu_uih_feature_names"] = [
            "uih_post_id",
            "uih_action_time",
            "uih_weight",
            "uih_owner_id",
            "uih_watchtime",
            "uih_surface_type",
            "uih_video_length",
            "viewer_id",
            "dummy_contextual",
        ]
        overrides["hstu_candidate_feature_names"] = [
            "item_post_id",
            "item_owner_id",
            "item_surface_type",
            "item_video_length",
            "item_action_weight",
            "item_target_watchtime",
            "item_query_time",
        ]
        overrides["merge_uih_candidate_feature_mapping"] = [
            ["uih_post_id", "item_post_id"],
            ["uih_owner_id", "item_owner_id"],
            ["uih_action_time", "item_query_time"],
            ["uih_weight", "item_action_weight"],
            ["uih_watchtime", "item_target_watchtime"],
            ["uih_video_length", "item_video_length"],
            ["uih_surface_type", "item_surface_type"],
        ]
        overrides["task_names"] = ["vvp100"]
        overrides["task_types"] = ["binary_classification"]

    return overrides


def get_embedding_table_overrides(dataset: str) -> List[Dict[str, Any]]:
    """Return embedding_tables list for HSTUConfig for a given dataset.

    Aligns with generative_recommenders/dlrm_v3/configs.py
    get_embedding_table_config().
    """
    dim = HSTU_EMBEDDING_DIM
    dataset_family = _resolve_dataset_family(dataset)

    if dataset_family == "movielens":
        if dataset == "movielens-1m":
            return [
                {
                    "name": "movie_id",
                    "embedding_dim": dim,
                    "num_embeddings": HASH_SIZE,
                    "feature_names": ["movie_id", "item_movie_id"],
                },
                {
                    "name": "user_id",
                    "embedding_dim": dim,
                    "num_embeddings": HASH_SIZE,
                    "feature_names": ["user_id"],
                },
                {
                    "name": "sex",
                    "embedding_dim": dim,
                    "num_embeddings": HASH_SIZE,
                    "feature_names": ["sex"],
                },
                {
                    "name": "age_group",
                    "embedding_dim": dim,
                    "num_embeddings": HASH_SIZE,
                    "feature_names": ["age_group"],
                },
                {
                    "name": "occupation",
                    "embedding_dim": dim,
                    "num_embeddings": HASH_SIZE,
                    "feature_names": ["occupation"],
                },
                {
                    "name": "zip_code",
                    "embedding_dim": dim,
                    "num_embeddings": HASH_SIZE,
                    "feature_names": ["zip_code"],
                },
            ]
        else:
            return [
                {
                    "name": "movie_id",
                    "embedding_dim": dim,
                    "num_embeddings": HASH_SIZE_1B,
                    "feature_names": ["movie_id", "item_movie_id"],
                },
                {
                    "name": "user_id",
                    "embedding_dim": dim,
                    "num_embeddings": 3_000_000,
                    "feature_names": ["user_id"],
                },
            ]

    elif dataset_family == "streaming":
        return [
            {
                "name": "item_id",
                "embedding_dim": dim,
                "num_embeddings": HASH_SIZE_1B,
                "feature_names": ["item_id", "item_candidate_id"],
            },
            {
                "name": "item_category_id",
                "embedding_dim": dim,
                "num_embeddings": 128,
                "weight_init_max": 1.0,
                "weight_init_min": -1.0,
                "feature_names": ["item_category_id", "item_candidate_category_id"],
            },
            {
                "name": "user_id",
                "embedding_dim": dim,
                "num_embeddings": HASH_SIZE,
                "feature_names": ["user_id"],
            },
        ]

    elif dataset_family == "kuairand":
        return [
            {
                "name": "video_id",
                "embedding_dim": dim,
                "num_embeddings": HASH_SIZE,
                "feature_names": ["video_id", "item_video_id"],
            },
            {
                "name": "user_id",
                "embedding_dim": dim,
                "num_embeddings": HASH_SIZE,
                "feature_names": ["user_id"],
            },
            {
                "name": "user_active_degree",
                "embedding_dim": dim,
                "num_embeddings": 8,
                "feature_names": ["user_active_degree"],
            },
            {
                "name": "follow_user_num_range",
                "embedding_dim": dim,
                "num_embeddings": 9,
                "feature_names": ["follow_user_num_range"],
            },
            {
                "name": "fans_user_num_range",
                "embedding_dim": dim,
                "num_embeddings": 9,
                "feature_names": ["fans_user_num_range"],
            },
            {
                "name": "friend_user_num_range",
                "embedding_dim": dim,
                "num_embeddings": 8,
                "feature_names": ["friend_user_num_range"],
            },
            {
                "name": "register_days_range",
                "embedding_dim": dim,
                "num_embeddings": 8,
                "feature_names": ["register_days_range"],
            },
        ]

    return [
        {
            "name": "post_id",
            "embedding_dim": dim,
            "num_embeddings": HASH_SIZE,
            "feature_names": [
                "uih_post_id",
                "item_post_id",
                "uih_owner_id",
                "item_owner_id",
            ],
        },
        {
            "name": "viewer_id",
            "embedding_dim": dim,
            "num_embeddings": HASH_SIZE,
            "feature_names": ["viewer_id"],
        },
        {
            "name": "dummy_contextual",
            "embedding_dim": dim,
            "num_embeddings": HASH_SIZE,
            "feature_names": ["dummy_contextual"],
        },
    ]
