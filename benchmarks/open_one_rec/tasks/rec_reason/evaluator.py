# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Adapted from https://github.com/Kuaishou-OneRec/OpenOneRec

"""
Recommendation Reason Evaluator

Evaluates model predictions on Recommendation Reason task using LLM-based multi-dimensional evaluation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Tuple

from benchmarks.open_one_rec.base_evaluator import BaseEval, EvaluationError
from benchmarks.open_one_rec.tasks.rec_reason.utils import (
    evaluate_reasoning,
    extract_after_think,
)

logger = logging.getLogger(__name__)


class _OpenAIJudge:
    """Provide the scorer's blocking interface using the configured OpenAI SDK."""

    def __init__(
        self, client_type, model_name: str, endpoint: str, api_key: str, use_async: bool
    ) -> None:
        self.client_type = client_type
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_key = api_key
        self.use_async = use_async

    def generate(self, prompt: str) -> str:
        if self.use_async:
            return asyncio.run(self._generate_async(prompt))
        with self.client_type(base_url=self.endpoint, api_key=self.api_key) as client:
            response = client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
        return self._response_text(response)

    async def _generate_async(self, prompt: str) -> str:
        # Each worker owns its client and event loop until the request completes.
        async with self.client_type(base_url=self.endpoint, api_key=self.api_key) as client:
            response = await client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
        return self._response_text(response)

    @staticmethod
    def _response_text(response) -> str:
        if not response.choices:
            raise ValueError("Judge response contains no choices")
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Judge response contains no text content")
        return content


class RecoReasonEvaluator(BaseEval):
    """Recommendation Reason task evaluator"""

    reuse_cached_metrics = False

    @property
    def required_metrics(self) -> List[str]:
        """Define required overall metrics for Recommendation Reason evaluation"""
        return ["llm_score"]

    def _get_vllm_client(self, eval_config: Dict[str, Any]):
        """
        Create vLLM OpenAI-compatible client.

        Args:
            eval_config: Evaluation configuration with vLLM endpoint settings

        Returns:
            Blocking Judge adapter configured for the vLLM endpoint

        Raises:
            ImportError: If openai package not available
            ValueError: If endpoint configuration invalid
        """
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("OpenAI client not installed. Install with: pip install openai")

        # Get vLLM endpoint configuration
        vllm_endpoint = eval_config.get("vllm_endpoint", "http://localhost:8000/v1")
        vllm_api_key = eval_config.get("vllm_api_key", "not-needed")
        use_async = eval_config.get("vllm_async", False)

        logger.info(f"Creating vLLM client for endpoint: {vllm_endpoint}")

        return _OpenAIJudge(
            client_type=AsyncOpenAI if use_async else OpenAI,
            model_name=eval_config.get("llm_judge_model", "gemini"),
            endpoint=vllm_endpoint,
            api_key=vllm_api_key,
            use_async=use_async,
        )

    def _compute_metrics_from_scratch(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Compute all metrics from scratch

        Returns:
            Tuple of (metrics, per_sample_metrics)
        """
        total_samples = len(self.samples)

        # Prepare data for evaluation
        sample_ids = list(self.samples.keys())
        predictions = []
        references = []

        for sample_id in sample_ids:
            sample = self.samples[sample_id]

            # Get ground truth
            ground_truth = sample.get("ground_truth", "")
            references.append(ground_truth)

            # Get model prediction (first generation)
            generations = sample.get("generations", [])
            if not generations:
                prediction = ""
            else:
                # Extract text after </think> tag if present
                prediction = extract_after_think(generations[0])
            predictions.append(prediction)

        # Get evaluation config
        eval_config = self.task_config.get("evaluation_config", {})

        # Build per-sample metrics
        per_sample_metrics = {}
        for sample_id in sample_ids:
            per_sample_metrics[sample_id] = {}

        # Build overall metrics
        metrics = {
            "num_samples": total_samples,
        }

        # LLM Evaluation (if enabled)
        llm_eval_enabled = eval_config.get("llm_eval_enabled", False)
        if llm_eval_enabled:
            logger.info("LLM evaluation enabled, starting multi-dimensional evaluation...")
            llm_metrics, llm_per_sample = self._evaluate_reasoning(
                sample_ids=sample_ids,
                predictions=predictions,
                references=references,
                eval_config=eval_config,
            )

            # Merge LLM metrics into overall metrics
            metrics.update(llm_metrics)

            # Merge LLM per-sample metrics
            for sample_id in sample_ids:
                if sample_id in llm_per_sample:
                    per_sample_metrics[sample_id].update(llm_per_sample[sample_id])

        # Save debug information if requested
        if self.debug and self.predictions_dir:
            self._save_debug_info(metrics, per_sample_metrics, predictions, references)

        return metrics, per_sample_metrics

    def _evaluate_reasoning(
        self, sample_ids: list, predictions: list, references: list, eval_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Perform LLM-based multi-dimensional evaluation.

        Args:
            sample_ids: List of sample IDs
            predictions: List of prediction texts
            references: List of reference texts
            eval_config: Evaluation configuration

        Returns:
            Tuple of (llm_metrics, llm_per_sample_metrics)
        """
        try:
            llm_client = self._get_vllm_client(eval_config)
        except Exception as e:
            raise EvaluationError(f"Cannot initialize recommendation-reason Judge: {e}") from e

        # Get LLM eval config
        llm_judge_model = eval_config.get("llm_judge_model", "gemini")
        llm_max_workers = eval_config.get("llm_max_workers", 3)
        llm_max_samples = eval_config.get("llm_max_samples", 300)

        if not callable(getattr(llm_client, "generate", None)):
            raise EvaluationError(f"{self.task_name}: Judge client must provide generate(prompt)")

        # Prepare data as dicts
        predictions_dict = {id: pred for id, pred in zip(sample_ids, predictions)}
        references_dict = {id: ref for id, ref in zip(sample_ids, references)}

        # Get model name for cache file naming
        model_name = getattr(llm_client, "model_name", llm_judge_model)

        # Run LLM evaluation
        try:
            llm_metrics, llm_per_sample = evaluate_reasoning(
                predictions=predictions_dict,
                references=references_dict,
                llm_client=llm_client,
                max_workers=llm_max_workers,
                max_samples=llm_max_samples,
                model_name=model_name,
                save_dir=self.predictions_dir,
            )

        except EvaluationError:
            raise
        except Exception as e:
            raise EvaluationError(f"{self.task_name}: Judge evaluation failed: {e}") from e

        self._require_judge_metrics(llm_metrics, "llm_score", "llm_eval_num_samples", 1, 5)
        return llm_metrics, llm_per_sample

    def _save_debug_info(
        self,
        metrics: Dict[str, Any],
        per_sample_metrics: Dict[str, Dict[str, Any]],
        predictions: list,
        references: list,
    ):
        """
        Save detailed debug information to file

        Args:
            metrics: Overall metrics
            per_sample_metrics: Per-sample metrics
            predictions: List of predictions
            references: List of references
        """
        # Prepare debug info
        debug_info = {
            "overall_metrics": metrics,
            "per_sample_metrics": per_sample_metrics,
            "sample_count": len(predictions),
        }

        # Add some examples
        sample_ids = list(self.samples.keys())
        debug_info["examples"] = []
        for i in range(min(10, len(sample_ids))):
            sample_id = sample_ids[i]
            debug_info["examples"].append(
                {
                    "sample_id": sample_id,
                    "prediction": predictions[i][:500] + "..."
                    if len(predictions[i]) > 500
                    else predictions[i],
                    "reference": references[i][:500] + "..."
                    if len(references[i]) > 500
                    else references[i],
                    "llm_score": per_sample_metrics[sample_id].get("llm_score"),
                    "llm_reason": per_sample_metrics[sample_id].get("llm_reason"),
                }
            )

        # Save to file using base class method
        self._save_debug_json(debug_info, filename="debug.json")

        # Print summary statistics
        logger.info(f"Total samples: {metrics['num_samples']}")

        # Print LLM eval metrics if available
        if metrics.get("llm_score") is not None:
            logger.info(f"LLM Eval Score: {metrics['llm_score']:.4f}")
