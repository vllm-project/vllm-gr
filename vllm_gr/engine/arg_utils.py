# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Engine arguments for generative recommendation.

This module extends vLLM's ``EngineArgs`` / ``AsyncEngineArgs`` with
GR-specific parameters while keeping the underlying vLLM configuration
contract intact.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)


class GRCommonEngineArgsMixin:
    """
    GR Engine Arguments - Inherits from vLLM EngineArgs.

    Extends vLLM's EngineArgs with GR-specific parameters while maintaining
    full compatibility with vLLM's configuration system.

    Inherited from EngineArgs (commonly used):
        model: Model name or path
        tensor_parallel_size: Tensor parallelism degree
        gpu_memory_utilization: GPU memory utilization fraction
        disable_log_stats: Whether to disable logging statistics
        max_model_len: Maximum scheduler-visible model length

    GR-specific parameters:
        max_seq_len: Maximum user history sequence length
        max_candidates: Maximum number of candidate items

    Feature names (uih, candidate, primary_id, task_names) are read
    directly from HSTUConfig (hf_config), which is the single source of
    truth.  Override via hf_overrides or config.json, not here.

    Model architecture parameters (num_layers, embedding_dim, etc.) live in
    config.json / HSTUConfig and are loaded via hf_config, following vLLM's
    convention.  They are NOT duplicated here.
    """

    # GR-specific: Sequence configuration
    max_seq_len: int = 512
    max_candidates: int = 1000

    # GR-specific: HSTU NPU fused runtime policy
    enable_npu_fused_prefill: Optional[bool] = None
    force_hammer_npu: Optional[bool] = None

    # GR-specific: Dataset identifier (drives config overrides)
    dataset: Optional[str] = None

    # GR-specific: explicit model family for backend/runtime defaults.
    # Keep this separate from HuggingFace's ``model_type`` so future GR
    # model families can coexist without assuming unknown models are HSTU.
    model_family: Optional[str] = None

    # GR-specific: explicit opt-in for demo/example flows that intentionally
    # provide only config.json and rely on randomly initialized weights.
    # Keep the default strict so local model directories without weights fail
    # fast unless the caller clearly requests dummy loading.
    allow_missing_weights: bool = False

    # Resolved by ``_apply_model_family_defaults``: "gr" or "llm".
    model_category: Optional[str] = None

    def __post_init__(self):
        """Post-init processing to ensure proper configuration."""
        super().__post_init__()

        # Merge dataset overrides into hf_overrides (in-memory, never disk)
        self._apply_dataset_overrides()

        self.model_family = self._resolve_model_family()
        self._apply_model_family_defaults()

    def _is_ascend_platform(self) -> bool:
        """Return True when the active vLLM platform is Ascend/NPU."""
        return current_platform.device_type == "npu"

    def _select_worker_cls_for_model_family(
        self,
        model_family: Optional[str],
    ) -> Optional[str]:
        """Choose a worker only when the model family defines one."""
        from vllm_gr.adapters.registry import get_worker_cls_path

        if model_family is None:
            return None
        return get_worker_cls_path(model_family, is_npu=self._is_ascend_platform())

    def _has_model_weights(self) -> bool:
        """Check if the model path contains HuggingFace-style weight files."""
        if not self.model:
            return False

        if not os.path.isdir(self.model):
            return False

        hf_patterns = [
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
        ]
        for pattern in hf_patterns:
            if glob.glob(os.path.join(self.model, pattern)):
                return True

        return False

    @staticmethod
    def _normalize_model_family(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        return normalized or None

    def _read_model_type_from_hf_overrides(self) -> Optional[str]:
        if isinstance(self.hf_overrides, dict):
            model_type = self.hf_overrides.get("model_type")
            if isinstance(model_type, str):
                return self._normalize_model_family(model_type)
        return None

    def _read_model_type_from_local_config(self) -> Optional[str]:
        config = self._read_local_config_dict()
        if config is None:
            return None
        model_type = config.get("model_type")
        if isinstance(model_type, str):
            return self._normalize_model_family(model_type)
        return None

    def _read_local_config_dict(self) -> Optional[Dict[str, Any]]:
        if not self.model or not os.path.isdir(self.model):
            return None

        config_path = os.path.join(self.model, "config.json")
        if not os.path.exists(config_path):
            return None

        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read local model config at {config_path}: {exc}"
            ) from exc

    def _resolve_model_family(self) -> Optional[str]:
        """Resolve the GR model family without assuming unknown == HSTU."""
        explicit = self._normalize_model_family(self.model_family)
        if explicit is not None:
            return explicit

        override_model_type = self._read_model_type_from_hf_overrides()
        if override_model_type is not None:
            return override_model_type

        config_model_type = self._read_model_type_from_local_config()
        if config_model_type is not None:
            return config_model_type

        if not self.model:
            # The in-repo no-path flow is still the current HSTU bootstrap path.
            return "hstu"

        return None

    def _apply_model_family_defaults(self) -> None:
        """Resolve model category and apply family-specific engine defaults."""
        from vllm_gr.adapters.pangu_llm._defaults import apply_pangu_llm_defaults
        from vllm_gr.adapters.registry import apply_family_defaults
        from vllm_gr.config.model_taxonomy import ModelCategory, classify_model

        self.model_category = classify_model(
            model_family=self.model_family,
            local_config=self._read_local_config_dict(),
            hf_overrides=(
                self.hf_overrides if isinstance(self.hf_overrides, dict) else None
            ),
        ).value

        if self.model_category == ModelCategory.LLM.value:
            apply_pangu_llm_defaults(self)
            return

        if self.model_family is not None:
            self._ensure_family_registered(self.model_family)
            apply_family_defaults(self.model_family, self)

    @staticmethod
    def _ensure_family_registered(model_family: str) -> None:
        """Trigger Layer 1 registration for the given family if not already done.

        Uses dynamic discovery: imports ``vllm_gr.adapters.<model_family>``
        which triggers ``register_family()`` in that package's ``__init__.py``.
        No hardcoded family list — new models only need to create the adapter
        package directory with a ``register_family()`` call.
        """
        from vllm_gr.adapters.registry import get_registered_families

        if model_family in get_registered_families():
            return

        import importlib

        target = f"vllm_gr.adapters.{model_family}"
        try:
            importlib.import_module(target)
        except ModuleNotFoundError as exc:
            if exc.name == target:
                pass  # adapter package not installed — family stays unregistered
            else:
                raise

    def _apply_dataset_overrides(self) -> None:
        """Merge dataset-specific overrides into ``hf_overrides`` (in-memory).

        Unlike the previous implementation that mutated ``config.json`` on
        disk, this uses vLLM's ``hf_overrides`` mechanism so that overrides
        are applied in-memory when HuggingFace loads the config.  This is
        safe for concurrent processes and read-only model directories.

        User-supplied ``hf_overrides`` take precedence over dataset defaults.
        """
        if not self.dataset:
            return

        from vllm_gr.config.dataset_configs import (
            get_embedding_table_overrides,
            get_hstu_config_overrides,
        )

        overrides = get_hstu_config_overrides(self.dataset)
        tables = get_embedding_table_overrides(self.dataset)
        if tables:
            overrides["embedding_tables"] = tables

        if isinstance(self.hf_overrides, dict):
            merged = dict(overrides)
            merged.update(self.hf_overrides)
            self.hf_overrides = merged
        elif self.hf_overrides is None:
            self.hf_overrides = overrides

        logger.info("Applied dataset=%s overrides via hf_overrides", self.dataset)

    def create_engine_config(self, usage_context=None, headless: bool = False):
        """
        Create the vLLM engine config for GR models.

        ``initialize_runtime`` must run in the current process before the base
        vLLM engine config path so custom registrations and plugins are active.
        Dataset overrides have already been merged into ``hf_overrides`` during
        ``__post_init__``.
        """
        from vllm_gr.config.model_taxonomy import ModelCategory, is_gr_category
        from vllm_gr.engine.engine_factory import attach_model_category_to_vllm_config
        from vllm_gr.plugin import initialize_runtime

        strict_gr = is_gr_category(getattr(self, "model_category", None))
        initialize_runtime(strict_gr=strict_gr)
        logger.info(
            "Creating engine config for model: %s (category=%s)",
            self.model,
            getattr(self, "model_category", None),
        )
        vllm_config = super().create_engine_config(
            usage_context=usage_context,
            headless=headless,
        )
        attach_model_category_to_vllm_config(vllm_config, self)
        return vllm_config


@dataclass
class GREngineArgs(GRCommonEngineArgsMixin, EngineArgs):
    """
    GR Engine arguments for offline/synchronous recommendation runtime.

    Feature names and task metadata are sourced from the model config
    (HSTUConfig / hf_config) rather than duplicated in CLI/runtime args.
    """

    max_seq_len: int = 512
    max_candidates: int = 1000
    enable_npu_fused_prefill: Optional[bool] = None
    force_hammer_npu: Optional[bool] = None
    dataset: Optional[str] = None
    model_family: Optional[str] = None
    model_category: Optional[str] = None
    allow_missing_weights: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GREngineArgs":
        """Create engine args from a dictionary, ignoring unknown fields."""
        # Filter to only include valid fields.
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class GRAsyncEngineArgs(GRCommonEngineArgsMixin, AsyncEngineArgs):
    """
    GR async engine arguments for online serving.

    The async variant shares the same GR policy as ``GREngineArgs`` while
    inheriting the extra async-serving flags from vLLM's ``AsyncEngineArgs``.
    Feature names and task metadata remain sourced from the model config.
    """

    max_seq_len: int = 512
    max_candidates: int = 1000
    enable_npu_fused_prefill: Optional[bool] = None
    force_hammer_npu: Optional[bool] = None
    dataset: Optional[str] = None
    model_family: Optional[str] = None
    model_category: Optional[str] = None
    allow_missing_weights: bool = False

    @classmethod
    def from_cli_args(cls, args):
        """Build async engine args from parsed CLI args."""
        return super().from_cli_args(args)
