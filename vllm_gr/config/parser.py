# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Final

from vllm_gr.config.models import GRConfig
from vllm_gr.config.validation import (
    validate_attention_backend,
    validate_catalog_path,
    validate_vocab_offset,
    validate_vocab_slice_size,
)

CATALOG_PATH_KEY: Final = "vllm_gr_catalog_path"
ATTENTION_BACKEND_KEY: Final = "vllm_gr_attention_backend"
VOCAB_OFFSET_KEY: Final = "vllm_gr_vocab_offset"
VOCAB_SLICE_SIZE_KEY: Final = "vllm_gr_vocab_slice_size"
GR_CONFIG_KEY_PREFIX: Final = "vllm_gr_"
KNOWN_GR_CONFIG_KEYS: Final = frozenset(
    {
        CATALOG_PATH_KEY,
        ATTENTION_BACKEND_KEY,
        VOCAB_OFFSET_KEY,
        VOCAB_SLICE_SIZE_KEY,
    }
)


def _reject_unknown_gr_keys(additional_config: Mapping[object, object]) -> None:
    unknown_keys = sorted(
        key
        for key in additional_config
        if isinstance(key, str)
        and key.startswith(GR_CONFIG_KEY_PREFIX)
        and key not in KNOWN_GR_CONFIG_KEYS
    )
    if unknown_keys:
        unknown = ", ".join(repr(key) for key in unknown_keys)
        raise ValueError(f"unknown vLLM-GR additional_config key(s): {unknown}")


def parse_gr_config(vllm_config: object | None) -> GRConfig:
    if vllm_config is None:
        return GRConfig()
    additional_config = getattr(vllm_config, "additional_config", None)
    if additional_config is None:
        return GRConfig()
    if not isinstance(additional_config, Mapping):
        # Upstream also accepts opaque SupportsHash configuration objects.
        # They cannot carry primitive vllm_gr_* keys, so leave them untouched
        # and treat them as an empty GR configuration on the read path.
        return GRConfig()
    _reject_unknown_gr_keys(additional_config)
    return GRConfig(
        catalog_path=validate_catalog_path(additional_config.get(CATALOG_PATH_KEY)),
        attention_backend=validate_attention_backend(additional_config.get(ATTENTION_BACKEND_KEY)),
        vocab_offset=validate_vocab_offset(additional_config.get(VOCAB_OFFSET_KEY)),
        vocab_slice_size=validate_vocab_slice_size(additional_config.get(VOCAB_SLICE_SIZE_KEY)),
    )


def write_gr_config(vllm_config: object, gr_config: GRConfig) -> None:
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, MutableMapping):
        raise ValueError("vLLM-GR additional_config must be a mutable mapping")
    _reject_unknown_gr_keys(additional_config)

    catalog_path = validate_catalog_path(gr_config.catalog_path)
    attention_backend = validate_attention_backend(gr_config.attention_backend)
    vocab_offset = validate_vocab_offset(gr_config.vocab_offset)
    vocab_slice_size = validate_vocab_slice_size(gr_config.vocab_slice_size)

    if catalog_path is None:
        additional_config.pop(CATALOG_PATH_KEY, None)
    else:
        additional_config[CATALOG_PATH_KEY] = catalog_path

    if attention_backend is None:
        additional_config.pop(ATTENTION_BACKEND_KEY, None)
    else:
        additional_config[ATTENTION_BACKEND_KEY] = attention_backend.value

    if vocab_offset is None:
        additional_config.pop(VOCAB_OFFSET_KEY, None)
    else:
        additional_config[VOCAB_OFFSET_KEY] = vocab_offset

    if vocab_slice_size is None:
        additional_config.pop(VOCAB_SLICE_SIZE_KEY, None)
    else:
        additional_config[VOCAB_SLICE_SIZE_KEY] = vocab_slice_size
