# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Adapted from https://github.com/Kuaishou-OneRec/OpenOneRec

"""
Recommendation Reason LLM Evaluation Utilities

Provides functions for extracting refined reasoning and multi-dimensional LLM evaluation.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import regex as re
from tqdm import tqdm

from benchmarks.open_one_rec.base_evaluator import EvaluationError
from benchmarks.open_one_rec.prompts import load_prompt

logger = logging.getLogger(__name__)


EVALUATION_PROMPT = load_prompt("rec_reason_evaluation")


def extract_refined_reasoning(text: str) -> str:
    """
    Extract the refined reasoning section from the full text.

    Finds the last occurrence of "精炼推理" and extracts the text after it.

    Args:
        text: Full text containing the reasoning

    Returns:
        Extracted refined reasoning text, or original text if pattern not found
    """
    if not text:
        return ""

    # Find the last occurrence of "精炼推理"
    keyword = "精炼推理"
    last_pos = text.rfind(keyword)

    if last_pos != -1:
        # Extract text after "精炼推理"
        after_keyword = text[last_pos + len(keyword) :]
        # Remove leading punctuation, whitespace, and markdown symbols
        after_keyword = re.sub(r"^[\s\*#：:\n]+", "", after_keyword)
        if after_keyword.strip():
            return after_keyword.strip()

    # If "精炼推理" not found, return original text
    return text.strip()


def extract_after_think(text: str) -> str:
    """Extract text after the last  comprehend tag if present."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract JSON from LLM response.

    Args:
        response: LLM response text

    Returns:
        Parsed JSON dict or None if parsing fails
    """
    if not response:
        return None

    try:
        response = response.strip()
        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        return json.loads(response.strip())
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.info(f"Failed to parse JSON: {response[:200]}...")
        return None


def _valid_sample_score(score: Any) -> bool:
    return type(score) is int and 1 <= score <= 5


def evaluate_single(
    gt_reasoning: str, model_reasoning: str, llm_client
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Evaluate a single sample using LLM.

    Args:
        gt_reasoning: Ground truth refined reasoning
        model_reasoning: Model-generated refined reasoning
        llm_client: LLM client instance

    Returns:
        Tuple of (evaluation_result, error_message)
    """
    prompt = EVALUATION_PROMPT.format(gt_reasoning, model_reasoning)

    try:
        response = llm_client.generate(prompt)
        result = extract_json_from_response(response)

        if isinstance(result, dict) and "llm_score" in result:
            score = result["llm_score"]
            if not _valid_sample_score(score):
                return None, f"Invalid Judge score: {score!r}; expected an integer in [1, 5]"
            return result, None

        return None, f"Failed to parse JSON or missing 'llm_score': {response[:100]}"

    except Exception as e:
        return None, f"API error: {str(e)}"


def evaluate_batch(
    gt_reasonings: Dict[str, str],
    model_reasonings: Dict[str, str],
    llm_client,
    max_workers: int = 5,
    desc: str = "Evaluating reasoning",
) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Evaluate multiple samples in parallel.

    Args:
        gt_reasonings: Dict of {sample_id: gt_reasoning}
        model_reasonings: Dict of {sample_id: model_reasoning}
        llm_client: LLM client instance
        max_workers: Number of concurrent workers
        desc: Progress bar description

    Returns:
        Tuple of (results, errors)
    """
    results = {}
    errors = {}

    common_ids = gt_reasonings.keys() & model_reasonings.keys()

    def process_single(sample_id: str):
        gt = gt_reasonings[sample_id]
        model = model_reasonings[sample_id]
        result, error = evaluate_single(gt, model, llm_client)
        return sample_id, result, error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, sid): sid for sid in common_ids}

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            sample_id, result, error = future.result()
            if result is not None:
                results[sample_id] = result
            if error is not None:
                errors[sample_id] = error

    # Statistics
    total_attempted = len(common_ids)
    total_success = len(results)

    logger.info(f"{desc} statistics: {total_attempted} attempted, {total_success} successful")

    return results, errors


def calculate_metrics(eval_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Calculate micro and macro metrics from evaluation results.

    Args:
        eval_results: Dict of {sample_id: evaluation_result}

    Returns:
        Dict with micro/macro scores
    """
    if not eval_results:
        return {}

    scores = []
    for sample_id, result in eval_results.items():
        score = result.get("llm_score")
        if not _valid_sample_score(score):
            raise EvaluationError(
                f"{sample_id}: Invalid Judge score: {score!r}; expected an integer in [1, 5]"
            )
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return {
        "micro_llm_score": avg_score,
        "macro_llm_score": avg_score,
        "llm_score": avg_score,
        "llm_eval_num_samples": len(scores),
    }


def get_per_sample_metrics(eval_results: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-sample metrics from evaluation results.

    Args:
        eval_results: Dict of {sample_id: evaluation_result}

    Returns:
        Dict of {sample_id: {llm_score, llm_reason}}
    """
    per_sample = {}

    for sample_id, result in eval_results.items():
        sample_metrics = {}

        if "llm_score" in result:
            sample_metrics["llm_score"] = result["llm_score"]

        if "llm_reason" in result:
            sample_metrics["llm_reason"] = result["llm_reason"]

        per_sample[sample_id] = sample_metrics

    return per_sample


def get_cache_path(save_dir: str, model_name: str) -> str:
    """Get the diagnostic path with the model name encoded as one filename."""
    return os.path.join(save_dir, f"llm_eval_{quote(model_name, safe='')}.json")


def save_eval_results(
    save_dir: str,
    sample_ids: List[str],
    gt_reasonings: Dict[str, str],
    model_reasonings: Dict[str, str],
    eval_results: Dict[str, Dict],
    model_name: str,
):
    """
    Save evaluation results to file.

    Args:
        save_dir: Directory to save the file
        sample_ids: List of sample IDs
        gt_reasonings: Dict of {sample_id: gt_reasoning}
        model_reasonings: Dict of {sample_id: model_reasoning}
        eval_results: Dict of {sample_id: evaluation_result}
        model_name: Model name for filename
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = get_cache_path(save_dir, model_name)

    detailed_results = {}

    for sample_id in sample_ids:
        detailed_results[sample_id] = {
            "gt_reasoning": gt_reasonings.get(sample_id, ""),
            "model_reasoning": model_reasonings.get(sample_id, ""),
            "eval_result": eval_results.get(sample_id, {}),
        }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation results saved to {save_path}")


def evaluate_reasoning(
    predictions: Dict[str, str],
    references: Dict[str, str],
    llm_client,
    max_workers: int = 5,
    max_samples: Optional[int] = None,
    model_name: str = "unknown",
    save_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Main evaluation function for recommendation reasoning.

    Args:
        predictions: Dict of {sample_id: prediction_text}
        references: Dict of {sample_id: reference_text}
        llm_client: LLM client instance for evaluation
        max_workers: Number of concurrent workers
        max_samples: Maximum samples to evaluate (None for all)
        model_name: Model name for cache file naming
        save_dir: Directory to save results

    Returns:
        Tuple of (metrics, per_sample_metrics)
    """
    # Select samples
    all_sample_ids = sorted(set(predictions.keys()) & set(references.keys()))

    if max_samples is not None and max_samples < len(all_sample_ids):
        sample_ids = all_sample_ids[:max_samples]
        logger.info(f"Selected {len(sample_ids)} samples for LLM evaluation")
    else:
        sample_ids = all_sample_ids
        logger.info(f"Evaluating all {len(sample_ids)} samples")

    if not sample_ids:
        raise EvaluationError("Recommendation-reason Judge produced zero valid scores")

    logger.info("Extracting refined reasoning...")
    gt_reasonings = {}
    model_reasonings = {}
    for sample_id in sample_ids:
        reference = extract_refined_reasoning(references[sample_id])
        prediction = extract_refined_reasoning(extract_after_think(predictions[sample_id]))
        if not reference:
            raise EvaluationError(f"{sample_id}: empty reference for recommendation-reason Judge")
        if not prediction:
            raise EvaluationError(f"{sample_id}: empty prediction for recommendation-reason Judge")
        gt_reasonings[sample_id] = reference
        model_reasonings[sample_id] = prediction

    # Previous score files contain diagnostics only. Sample IDs do not identify
    # the prediction or reference used in an earlier run.
    eval_results, errors = evaluate_batch(gt_reasonings, model_reasonings, llm_client, max_workers)
    if missing_scores := [sample_id for sample_id in sample_ids if sample_id not in eval_results]:
        sample_id = missing_scores[0]
        reason = errors.get(sample_id, "missing Judge score")
        raise EvaluationError(
            f"Judge failed for {len(missing_scores)} sample(s); {sample_id}: {reason}"
        )

    # Calculate metrics
    metrics = calculate_metrics(eval_results)
    per_sample_metrics = get_per_sample_metrics(eval_results)

    # Save results
    if save_dir:
        save_eval_results(
            save_dir, sample_ids, gt_reasonings, model_reasonings, eval_results, model_name
        )

    # Print summary
    logger.info(f"LLM evaluation completed: {metrics.get('llm_eval_num_samples', 0)} samples")
    logger.info(f"  LLM Score: {metrics.get('llm_score', 0):.4f}")

    return metrics, per_sample_metrics
