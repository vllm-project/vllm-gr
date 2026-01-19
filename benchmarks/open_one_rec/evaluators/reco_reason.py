# Imported from: https://https://github.com/Kuaishou-OneRec/OpenOneRec
# Original commit: ae668a6a30bd71bde7a3f459fe0ac958182ce1d2
# Date imported: 2026‑01‑09
# License: Apache 2.0 License (inherited from source)
"""
Recommendation Reason Evaluator

Evaluates model predictions on Recommendation Reason task using LLM-based multi-dimensional evaluation.
"""

from typing import Dict, Any, Tuple, List

from benchmarks.open_one_rec.evaluators.base_evaluator import BaseEval
from benchmarks.open_one_rec.utils import extract_after_think

import logging

logger = logging.getLogger(__name__)


class RecoReasonEvaluator(BaseEval):
    """Recommendation Reason task evaluator"""

    @property
    def required_metrics(self) -> List[str]:
        """Define required overall metrics for Recommendation Reason evaluation"""
        return ["llm_score"]

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
