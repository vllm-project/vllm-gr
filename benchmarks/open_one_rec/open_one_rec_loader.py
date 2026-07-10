"""
Base loader for OpenOneRec task datasets.

Provides common functionality for data loading, sampling, and file path
resolution.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC
from dataclasses import dataclass
from importlib import import_module
from json import JSONDecodeError
from typing import Any, Dict, Optional

from .tasks.item_understand.config import ITEM_UNDERSTAND_CONFIG
from .tasks.label_pred.config import LABEL_PRED_CONFIG
from .tasks.rec_reason.config import REC_REASON_CONFIG
from .tasks.recommendation.config import (
    AD_CONFIG,
    INTERACTIVE_CONFIG,
    LABEL_COND_CONFIG,
    PRODUCT_CONFIG,
    VIDEO_CONFIG,
)

# Imported from: https://github.com/Kuaishou-OneRec/OpenOneRec
# Original commit: ae668a6a30bd71bde7a3f459fe0ac958182ce1d2
# Date imported: 2026-01-09
# License: Apache 2.0 License (inherited from source)


logger = logging.getLogger(__name__)


def _require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenOneRec benchmark requires pandas to read parquet dataset files."
        ) from exc
    return pd


def _read_parquet(path: str, columns: Optional[list[str]] = None):
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        pd = _require_pandas()
        return pd.read_parquet(path, columns=columns)

    read_errors = []

    try:
        parquet_file = pq.ParquetFile(path)
        table = parquet_file.read(columns=columns, use_threads=False)
        return table.to_pandas(ignore_metadata=True)
    except Exception as exc:
        read_errors.append(("ParquetFile.read", exc))

    try:
        pd = _require_pandas()
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        read_errors.append(("pandas.read_parquet", exc))

    error_summary = "; ".join(f"{name}: {type(exc).__name__}: {exc}" for name, exc in read_errors)
    raise RuntimeError(f"Failed to read parquet file {path}. Tried: {error_summary}")


@dataclass
class TaskRegistration:
    """Task registration information"""

    name: str
    config: Dict[str, Any]
    evaluator_path: str
    category: str  # "general", "recommendation", "caption"

    @property
    def evaluator_class(self):
        module_path, class_name = self.evaluator_path.split(":")
        module = import_module(module_path)
        return getattr(module, class_name)


# ========================================
# Unified Task Registry
# ========================================

TASK_REGISTRY: Dict[str, TaskRegistration] = {
    "label_cond": TaskRegistration(
        name="label_cond",
        config=LABEL_COND_CONFIG,
        evaluator_path="benchmarks.open_one_rec.tasks.recommendation.evaluator:RecommendationEvaluator",
        category="recommendation",
    ),
    "video": TaskRegistration(
        name="video",
        config=VIDEO_CONFIG,
        evaluator_path="benchmarks.open_one_rec.tasks.recommendation.evaluator:RecommendationEvaluator",
        category="recommendation",
    ),
    "product": TaskRegistration(
        name="product",
        config=PRODUCT_CONFIG,
        evaluator_path="benchmarks.open_one_rec.tasks.recommendation.evaluator:RecommendationEvaluator",
        category="recommendation",
    ),
    "ad": TaskRegistration(
        name="ad",
        config=AD_CONFIG,
        evaluator_path="benchmarks.open_one_rec.tasks.recommendation.evaluator:RecommendationEvaluator",
        category="recommendation",
    ),
    "interactive": TaskRegistration(
        name="interactive",
        config=INTERACTIVE_CONFIG,
        evaluator_path="benchmarks.open_one_rec.tasks.recommendation.evaluator:RecommendationEvaluator",
        category="recommendation",
    ),
    "label_pred": TaskRegistration(
        name="label_pred",
        config=LABEL_PRED_CONFIG,
        evaluator_path="benchmarks.open_one_rec.tasks.label_pred.evaluator:LabelPredEvaluator",
        category="recommendation",
    ),
    "item_understand": TaskRegistration(
        name="item_understand",
        config=ITEM_UNDERSTAND_CONFIG,
        evaluator_path="benchmarks.open_one_rec.tasks.item_understand.evaluator:ItemUnderstandEvaluator",
        category="caption",
    ),
    "rec_reason": TaskRegistration(
        name="rec_reason",
        config=REC_REASON_CONFIG,
        evaluator_path="benchmarks.open_one_rec.tasks.rec_reason.evaluator:RecoReasonEvaluator",
        category="caption",
    ),
}


# ========================================
# Factory Functions
# ========================================


def get_loader(
    task_name: str,
    data_dir: str,
    tokenizer: Optional[Any] = None,
    enable_thinking: Optional[bool] = None,
    custom_input_len: Optional[int] = None,
):
    """
    Get loader instance for a task

    Replaces loader_factory.get_loader()

    Args:
        task_name: Name of the task
        data_dir: Data directory path
        tokenizer: Tokenizer instance (optional, required for message-based formats)
        enable_thinking: Enable thinking mode (optional, overrides task config if set)
        custom_input_len: Set the input length to a predefined length

    Returns:
        Loader instance

    Raises:
        ValueError: If task_name is not registered
    """
    if task_name not in TASK_REGISTRY:
        available_tasks = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {available_tasks}")

    reg = TASK_REGISTRY[task_name]

    # Create loader instance with aligned parameters
    return BaseLoader(
        task_config=reg.config,
        data_dir=data_dir,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        custom_input_len=custom_input_len,
    )


class DataLoaderWrapper:
    """Wrapper for unified data loading interface"""

    def __init__(
        self,
        tokenizer: Any,
        data_dir: str,
        enable_thinking: Optional[bool] = None,
        custom_input_len: Optional[int] = None,
    ):
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required. Pass the tokenizer created by the "
                "benchmark (or AutoTokenizer.from_pretrained(...)) explicitly."
            )

        if custom_input_len is not None:
            custom_input_len = int(custom_input_len)
            if custom_input_len <= 0:
                raise ValueError("custom_input_len must be a positive integer")

        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.enable_thinking = enable_thinking
        self.custom_input_len = custom_input_len
        self._loader_cache = {}

    def load_data(
        self,
        task_name: str,
        split: str = "test",
        sample_size: Optional[Any] = None,
    ):
        """Load data using new loader system"""
        if task_name not in self._loader_cache:
            self._loader_cache[task_name] = get_loader(
                task_name=task_name,
                data_dir=self.data_dir,
                tokenizer=self.tokenizer,
                enable_thinking=self.enable_thinking,
                custom_input_len=self.custom_input_len,
            )

        loader = self._loader_cache[task_name]
        return loader.load_data(split=split, sample_size=sample_size)


class BaseLoader(ABC):
    """Base class for all task data loaders"""

    def __init__(
        self,
        task_config: Dict[str, Any],
        data_dir: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        enable_thinking: Optional[bool] = None,
        custom_input_len: Optional[int] = None,
    ):
        """Initialize base loader"""
        if custom_input_len is not None:
            custom_input_len = int(custom_input_len)
            if custom_input_len <= 0:
                raise ValueError("custom_input_len must be a positive integer")

        self.task_config = task_config
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.custom_input_len = custom_input_len
        self.task_name = task_config.get("name", "unknown")

        # Validate tokenizer is provided for messages-based format
        if self.tokenizer is None:
            raise ValueError(
                f"{self.task_name} requires tokenizer for messages-based format. "
                f"Please provide a tokenizer when initializing the Benchmark.\n"
                f"Example: Benchmark(task_types=['{self.task_name}'], tokenizer=your_tokenizer)"
            )

    def load_data(
        self, split: str = "test", sample_size: Optional[Any] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load data for the task in messages-based format

        Args:
            split: Dataset split (default "test")
            sample_size: Override sample size (can be int, "full", or None to use task config)

        Returns:
            Dictionary mapping sample_id to sample data:
            {
                sample_id: {
                    "prompt": "formatted prompt from apply_chat_template",
                    "ground_truth": "answer",
                    "metadata": {
                        "row_index": idx,
                        "messages": [...]
                    }
                }
            }
        """
        # Determine effective sample size
        if sample_size is not None:
            if sample_size == "full":
                effective_sample_size = self.task_config.get("size")
            else:
                effective_sample_size = int(sample_size)
        else:
            effective_sample_size = self.task_config.get("sample_size")

        full_size = self.task_config.get("size")

        # Try to load cached sample dataframe
        df = None
        if (
            effective_sample_size is not None
            and full_size is not None
            and effective_sample_size < full_size
        ):
            df = self._load_sample_dataframe(split, effective_sample_size)

        # If no cache, load and sample original data
        if df is None:
            df = self._load_dataframe(split)

            # Perform sampling if needed
            if effective_sample_size is not None and effective_sample_size < len(df):
                df = self._sample_data(df, effective_sample_size)

                # Save sampled data
                if full_size is not None and effective_sample_size < full_size:
                    self._save_sample_data(df, split, effective_sample_size)

        if "messages" not in df.columns:
            raise ValueError(
                f"{self.task_name} requires 'messages' column in data file. "
                f"Found columns: {list(df.columns)}\n"
                f"Please ensure your data is in messages-based format."
            )

        if "metadata" not in df.columns:
            raise ValueError(
                f"{self.task_name} requires 'metadata' column in data file. "
                f"Found columns: {list(df.columns)}\n"
                f"Please ensure your data is in messages-based format."
            )

        logger.info(f"Processing {self.task_name} data in messages-based format")

        result = self._process_dataframe(df)

        return result

    @staticmethod
    def _is_empty_value(value) -> bool:
        """Check if a value is None, NaN, or empty"""
        if value is None:
            return True

        if isinstance(value, float):
            try:
                pd = _require_pandas()
                return pd.isna(value)
            except (ValueError, TypeError):
                return False

        if isinstance(value, str):
            return len(value.strip()) == 0

        try:
            if hasattr(value, "__len__"):
                return len(value) == 0
        except (ValueError, TypeError):
            pass

        return False

    @staticmethod
    def _convert_messages_format(messages: list) -> list:
        """
        Convert message format.

        {"role": "user", "content": [{"type": "text", "text": "..."}]}
        ->
        {"role": "user", "content": "..."}
        """
        converted = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                # Extract text from content list
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                converted.append({"role": msg.get("role"), "content": "".join(text_parts)})
            else:
                # Already in old format
                converted.append(msg)
        return converted

    def _load_custom_chat_template(self):
        """Load custom chat template based on configuration"""
        if not self.tokenizer:
            return

        prompt_config = self.task_config.get("prompt_config", {})
        custom_template = prompt_config.get("custom_chat_template")

        template_path = os.path.join(os.path.dirname(__file__), custom_template)

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"✗ Custom chat template not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()
        logger.info(f"✓ Loaded custom chat template: {custom_template}")

    def _get_data_file_path(self, split: str) -> str:
        """Get data file path for the given split"""
        if self.data_dir:
            base_dir = self.data_dir
        else:
            base_dir = "./data"

        data_file = os.path.join(base_dir, self.task_name, f"{self.task_name}_{split}.parquet")

        if os.path.exists(data_file):
            return data_file

        return data_file

    def _get_sample_data_file_path(self, split: str, sample_size: int) -> str:
        """Get sample data file path"""
        if self.data_dir:
            base_dir = self.data_dir
        else:
            base_dir = "./data"

        task_sample_file = os.path.join(
            base_dir,
            self.task_name,
            f"{self.task_name}_{split}_sample_{sample_size}.parquet",
        )

        if os.path.exists(task_sample_file):
            return task_sample_file

        root_sample_file = os.path.join(
            base_dir,
            f"{self.task_name}_{split}_sample_{sample_size}.parquet",
        )
        if os.path.exists(root_sample_file):
            return root_sample_file

        # Preserve the original first-write layout under <data_dir>/<task_name>/.
        # Only fall back to the root-level cache path when that cache already
        # exists and we are reading legacy data.
        return task_sample_file

    def _load_dataframe(self, split: str) -> Any:
        """Load DataFrame from data file"""
        data_file = self._get_data_file_path(split)

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        logger.info(f"Loading data file: {data_file}")

        df = _read_parquet(data_file, columns=["messages", "metadata"])

        return df

    def _sample_data(self, df: Any, sample_size: int) -> Any:
        """Reproducibly sample data (take head) from DataFrame"""
        if sample_size >= len(df):
            return df

        logger.info(f"Sampling {sample_size} samples (total: {len(df)})")
        return df.sample(sample_size, random_state=42)

    def _save_sample_data(self, df: Any, split: str, sample_size: int):
        """Save sample data in parquet format"""
        sample_file = self._get_sample_data_file_path(split, sample_size)

        sample_dir = os.path.dirname(sample_file)
        if sample_dir:
            os.makedirs(sample_dir, exist_ok=True)

        df.to_parquet(sample_file, index=False)
        logger.info(f"Sample data saved to: {sample_file}")

    def _load_sample_dataframe(self, split: str, sample_size: int) -> Optional[Any]:
        """Load sample dataframe from cache if exists"""
        sample_file = self._get_sample_data_file_path(split, sample_size)

        if not os.path.exists(sample_file):
            return None

        logger.info(f"Loading sample data from cache: {sample_file}")

        try:
            df = _read_parquet(sample_file, columns=["messages", "metadata"])
        except Exception as exc:
            logger.warning(
                "Failed to load sample cache %s: %s: %s. Falling back to source dataset.",
                sample_file,
                type(exc).__name__,
                exc,
            )
            return None
        return df

    def _process_dataframe(self, df: Any) -> Dict[str, Dict[str, Any]]:
        """Process DataFrame and convert to model input format"""
        self._load_custom_chat_template()

        result = {}

        prompt_config = self.task_config.get("prompt_config", {})
        # Command-line parameter has higher priority than config
        if self.enable_thinking is not None:
            enable_thinking = self.enable_thinking
        else:
            enable_thinking = prompt_config.get("enable_thinking", False)

        logger.info(f"Auto Thinking: {'✓ Enabled' if enable_thinking else '✗ Disabled'}")

        for idx, row in df.iterrows():
            sample_id = str(idx)

            messages = row.get("messages")
            if self._is_empty_value(messages):
                logger.info(f"Sample {sample_id}: messages is empty, skipping")
                continue

            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except JSONDecodeError:
                    logger.info(f"Sample {sample_id}: failed to parse messages, skipping")
                    continue

            messages = self._convert_messages_format(messages)

            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except Exception as e:
                logger.info(f"Sample {sample_id}: failed to apply chat template: {e}, skipping")
                continue

            if self.custom_input_len is not None:
                formatted_prompt = self._enforce_prompt_token_length(formatted_prompt, sample_id)

            metadata_raw = row.get("metadata")
            if self._is_empty_value(metadata_raw):
                logger.info(f"Sample {sample_id}: metadata is empty, skipping")
                continue

            if isinstance(metadata_raw, str):
                try:
                    metadata_dict = json.loads(metadata_raw)
                except JSONDecodeError:
                    logger.info(f"Sample {sample_id}: failed to parse metadata, skipping")
                    continue
            elif isinstance(metadata_raw, dict):
                metadata_dict = metadata_raw
            else:
                logger.info(f"Sample {sample_id}: invalid metadata format, skipping")
                continue

            answer = metadata_dict.get("answer")
            if self._is_empty_value(answer):
                logger.info(f"Sample {sample_id}: answer is empty in metadata, skipping")
                continue

            ground_truth_str = str(answer).strip()

            result_item = {
                "prompt": formatted_prompt,
                "ground_truth": ground_truth_str,
                "metadata": self._make_metadata_serializable(idx, metadata_dict),
            }

            result[sample_id] = result_item

        logger.info(f"Loaded {len(result)} samples for {self.task_name}")

        return result

    def _tokenize_prompt(self, prompt: str) -> list[int]:
        """Tokenize prompt text for length normalization."""
        tokenized = self.tokenizer(prompt, add_special_tokens=False)
        token_ids = getattr(tokenized, "input_ids", None)
        if token_ids is None and isinstance(tokenized, dict):
            token_ids = tokenized.get("input_ids", [])
        return list(token_ids or [])

    def _enforce_prompt_token_length(self, prompt: str, sample_id: str) -> str:
        """Trim or repeat prompt tokens to match custom_input_len exactly."""
        if self.custom_input_len is None:
            return prompt

        target_len = int(self.custom_input_len)
        if target_len <= 0:
            raise ValueError("custom_input_len must be a positive integer")

        prompt_token_ids = self._tokenize_prompt(prompt)
        current_len = len(prompt_token_ids)

        if current_len == target_len:
            return prompt

        if current_len == 0:
            logger.warning(
                "Sample %s: prompt tokenized to empty input; cannot enforce custom_input_len=%s",
                sample_id,
                target_len,
            )
            return prompt

        if current_len < target_len:
            repeat_count = (target_len + current_len - 1) // current_len
            adjusted_token_ids = (prompt_token_ids * repeat_count)[:target_len]
            action = "extended"
        else:
            adjusted_token_ids = prompt_token_ids[:target_len]
            action = "trimmed"

        adjusted_prompt = self.tokenizer.decode(
            adjusted_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        final_len = len(self._tokenize_prompt(adjusted_prompt))
        if final_len != target_len:
            logger.warning(
                "Sample %s: requested custom_input_len=%s, got %s after %s",
                sample_id,
                target_len,
                final_len,
                action,
            )
        else:
            logger.debug(
                "Sample %s: prompt %s to %s tokens",
                sample_id,
                action,
                target_len,
            )

        return adjusted_prompt

    def _make_metadata_serializable(
        self,
        idx: Any,
        metadata_dict: dict,
    ) -> dict:
        """Build the per-sample metadata dict.

        Drops the ``answer`` key (already promoted to the top-level
        ``ground_truth`` field by the caller) and injects the row index.
        Returns a new dict — does NOT mutate ``metadata_dict`` in place.
        """
        return {
            "row_index": int(idx) if hasattr(idx, "__int__") else str(idx),
            **{k: v for k, v in metadata_dict.items() if k != "answer"},
        }
