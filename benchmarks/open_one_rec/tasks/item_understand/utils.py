# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Adapted from https://github.com/Kuaishou-OneRec/OpenOneRec

import json
import logging
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from benchmarks.open_one_rec.base_evaluator import EvaluationError
from benchmarks.open_one_rec.prompts import load_prompt

logger = logging.getLogger(__name__)


WIP_EXTRACTION_PROMPT = load_prompt("item_understand_wip_extraction")


WIP_MATCHING_PROMPT = load_prompt("item_understand_wip_matching")


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract JSON from LLM response (simplified version for well-behaved LLMs).
    """
    if not response:
        return None

    try:
        response = response.rstrip("```").lstrip("```json")
        return json.loads(response.strip())
    except json.JSONDecodeError:
        logger.error(response)
        return None


def _valid_wip(wip: Any) -> bool:
    if not isinstance(wip, dict):
        return False
    text = wip.get("info_point")
    score = wip.get("importance_score")
    return isinstance(text, str) and bool(text.strip()) and type(score) is int and 1 <= score <= 5


def extract_wips_single(text: str, llm_client) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Extract WIPs from a single text using LLM.

    Args:
        text: Input text to extract WIPs from
        llm_client: LLM client instance (with built-in retry mechanism)

    Returns:
        Tuple of (wips_list, error_message)
        - wips_list: List of WIP dicts if successful, None if failed
        - error_message: Error message if failed, None if successful
    """
    prompt = WIP_EXTRACTION_PROMPT.format(text)

    try:
        response = llm_client.generate(prompt)
        result = extract_json_from_response(response)

        wips = result.get("wips") if isinstance(result, dict) else None
        if isinstance(wips, list) and all(_valid_wip(wip) for wip in wips):
            return wips, None

        return None, "Failed to parse JSON from response"

    except Exception as e:
        return None, f"API error: {str(e)}"


def extract_wips_batch(
    texts: Dict[str, str], llm_client, max_workers: int = 5, desc: str = "Extracting WIPs"
) -> Tuple[Dict[str, List[Dict]], Dict[str, str]]:
    """
    Extract WIPs from multiple texts in parallel.

    Args:
        texts: Dict of {sample_id: text}
        llm_client: LLM client instance (with built-in retry mechanism)
        max_workers: Number of concurrent workers
        desc: Progress bar description

    Returns:
        Tuple of (results, errors):
        - results: Dict of {sample_id: wips_list}
        - errors: Dict of {sample_id: error_message}
    """
    results = {}
    errors = {}

    def process_single(sample_id: str, text: str):
        wips, error = extract_wips_single(text, llm_client)
        return sample_id, wips, error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, sid, text): sid for sid, text in texts.items()}

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            sample_id, wips, error = future.result()
            if wips is not None:
                results[sample_id] = wips
            if error:
                errors[sample_id] = error

    # Statistics: count valid (non-empty) extraction results
    total_attempted = len(texts)
    total_parsed = len(results)
    valid_results = sum(1 for wips in results.values() if wips)  # Count non-empty lists

    logger.info(
        f"{desc} statistics: {total_attempted} attempted, {total_parsed} parsed, {valid_results} valid (non-empty)"
    )

    return results, errors


def _count_wips(wips: List[Dict]) -> Counter[tuple[str, int]]:
    return Counter((wip["info_point"], wip["importance_score"]) for wip in wips)


def match_wips_single(
    gt_wips: List[Dict], model_wips: List[Dict], llm_client
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Match WIPs between ground truth and model generation.

    Args:
        gt_wips: Ground truth WIPs list
        model_wips: Model-generated WIPs list
        llm_client: LLM client instance (with built-in retry mechanism)

    Returns:
        Tuple of (match_result, error_message)
    """
    gt_str = json.dumps(gt_wips, ensure_ascii=False, indent=2)
    model_str = json.dumps(model_wips, ensure_ascii=False, indent=2)
    prompt = WIP_MATCHING_PROMPT.format(gt_str, model_str)

    try:
        response = llm_client.generate(prompt)
        result = extract_json_from_response(response)

        if isinstance(result, dict):
            matches = result.get("matches")
            unmatched_model = result.get("unmatched_model_wips")
            unmatched_gt = result.get("unmatched_gt_wips")
            if (
                isinstance(matches, list)
                and isinstance(unmatched_model, list)
                and isinstance(unmatched_gt, list)
                and (matches or unmatched_model or unmatched_gt)
                and all(_valid_wip(wip) for wip in unmatched_model + unmatched_gt)
                and all(
                    isinstance(match, dict)
                    and _valid_wip(match.get("gt_wip"))
                    and _valid_wip(match.get("model_wip"))
                    for match in matches
                )
            ):
                for side, expected_wips, reported_wips in (
                    ("reference", gt_wips, [m["gt_wip"] for m in matches] + unmatched_gt),
                    ("model", model_wips, [m["model_wip"] for m in matches] + unmatched_model),
                ):
                    expected = _count_wips(expected_wips)
                    reported = _count_wips(reported_wips)
                    if expected != reported:
                        missing = sum((expected - reported).values())
                        unexpected = sum((reported - expected).values())
                        return None, (
                            f"{side} WIP coverage mismatch: {missing} missing, "
                            f"{unexpected} unexpected occurrence(s)"
                        )
                return result, None

        return None, "Failed to parse match JSON from response"

    except Exception as e:
        return None, f"API error: {str(e)}"


def match_wips_batch(
    gt_wips_dict: Dict[str, List[Dict]],
    model_wips_dict: Dict[str, List[Dict]],
    llm_client,
    max_workers: int = 5,
) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Match WIPs for multiple samples in parallel.

    Args:
        gt_wips_dict: Dict of {sample_id: gt_wips_list}
        model_wips_dict: Dict of {sample_id: model_wips_list}
        llm_client: LLM client instance (with built-in retry mechanism)
        max_workers: Number of concurrent workers

    Returns:
        Tuple of (results, errors)
    """
    results = {}
    errors = {}

    # Only match samples that have both GT and model WIPs (and both are non-empty)
    common_ids = {
        id
        for id in (set(gt_wips_dict.keys()) & set(model_wips_dict.keys()))
        if gt_wips_dict[id] and model_wips_dict[id]
    }

    def process_single(sample_id: str):
        gt_wips = gt_wips_dict[sample_id]
        model_wips = model_wips_dict[sample_id]
        match_result, error = match_wips_single(gt_wips, model_wips, llm_client)
        return sample_id, match_result, error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, sid): sid for sid in common_ids}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Matching WIPs"):
            sample_id, match_result, error = future.result()
            if match_result is not None:
                results[sample_id] = match_result
            if error is not None:
                errors[sample_id] = error

    # Statistics: count valid (non-empty) match results
    total_attempted = len(common_ids)
    total_parsed = len(results)
    valid_results = 0

    for sample_id, match_result in results.items():
        # Check if result is not empty (has at least one non-empty field)
        if match_result:
            matches = match_result.get("matches", [])
            unmatched_model = match_result.get("unmatched_model_wips", [])
            unmatched_gt = match_result.get("unmatched_gt_wips", [])

            # Consider valid if result has any content
            if matches or unmatched_model or unmatched_gt:
                valid_results += 1

    logger.info(
        f"Matching statistics: {total_attempted} attempted, {total_parsed} parsed, {valid_results} valid (non-empty)"
    )

    return results, errors


def get_wip_score_int(wip: Optional[Dict]) -> int:
    """Get importance score from WIP, defaulting to 1."""
    if not wip:
        return 1
    return wip.get("importance_score", 1)


def calculate_unweighted_metrics(
    match_results: Dict[str, Dict], core_threshold: int = 5
) -> Dict[str, Any]:
    """
    Calculate unweighted metrics (count-based) with macro and per-sample versions.

    Args:
        match_results: Dict of {sample_id: match_result}
        core_threshold: Threshold for core WIPs (importance_score >= threshold)

    Returns:
        Dict with macro F1, core versions, and per-sample F1s (unweighted)
    """
    if not match_results:
        return {}

    # Per-sample metrics (for macro calculation)
    per_sample = {}

    for sample_id, result in match_results.items():
        if not result:
            per_sample[sample_id] = {"overall_f1": 0.0, "core_f1": 0.0}
            continue

        # Sample-level counts
        sample_tp = len(result.get("matches", []))
        sample_fp = len(result.get("unmatched_model_wips", []))
        sample_fn = len(result.get("unmatched_gt_wips", []))

        sample_core_tp = 0
        sample_core_fp = 0
        sample_core_fn = 0

        # Core: count only WIPs with importance_score >= threshold
        for match in result.get("matches", []):
            gt_wip = match.get("gt_wip", {})
            if get_wip_score_int(gt_wip) >= core_threshold:
                sample_core_tp += 1

        for fp_wip in result.get("unmatched_model_wips", []):
            if get_wip_score_int(fp_wip) >= core_threshold:
                sample_core_fp += 1

        for fn_wip in result.get("unmatched_gt_wips", []):
            if get_wip_score_int(fn_wip) >= core_threshold:
                sample_core_fn += 1

        # Calculate per-sample F1s
        sample_overall_f1 = (
            2 * sample_tp / (2 * sample_tp + sample_fp + sample_fn)
            if (2 * sample_tp + sample_fp + sample_fn) > 0
            else 0.0
        )
        sample_core_f1 = (
            2 * sample_core_tp / (2 * sample_core_tp + sample_core_fp + sample_core_fn)
            if (2 * sample_core_tp + sample_core_fp + sample_core_fn) > 0
            else 0.0
        )

        per_sample[sample_id] = {
            "overall_f1": sample_overall_f1,
            "core_f1": sample_core_f1,
        }

    # Calculate macro F1 (average of per-sample F1s)
    valid_samples = [v for v in per_sample.values() if v]
    macro_f1 = (
        sum(s["overall_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0
    )
    macro_core_f1 = (
        sum(s["core_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0
    )

    return {
        "macro_wip_unweighted_f1": macro_f1,
        "macro_wip_unweighted_core_f1": macro_core_f1,
        "per_sample": per_sample,
    }


def calculate_importance_weighted_metrics(
    match_results: Dict[str, Dict], core_threshold: int = 5
) -> Dict[str, Any]:
    """
    Calculate importance-weighted metrics (weighted by importance_score only) with macro and per-sample versions.

    Args:
        match_results: Dict of {sample_id: match_result}
        core_threshold: Threshold for core WIPs (importance_score >= threshold)

    Returns:
        Dict with macro F1, core versions, and per-sample F1s (importance-weighted)
    """
    if not match_results:
        return {}

    # Per-sample metrics (for macro calculation)
    per_sample = {}

    for sample_id, result in match_results.items():
        if not result:
            per_sample[sample_id] = {"overall_f1": 0.0, "core_f1": 0.0}
            continue

        # Sample-level metrics
        sample_tp, sample_fp, sample_fn = 0.0, 0.0, 0.0
        sample_core_tp, sample_core_fp, sample_core_fn = 0.0, 0.0, 0.0

        # TP from matches (use GT score)
        for match in result.get("matches", []):
            gt_wip = match.get("gt_wip")
            gt_score = get_wip_score_int(gt_wip)
            sample_tp += gt_score
            if gt_score >= core_threshold:
                sample_core_tp += gt_score

        # FP from unmatched model WIPs
        for fp_wip in result.get("unmatched_model_wips", []):
            fp_score = get_wip_score_int(fp_wip)
            sample_fp += fp_score
            if fp_score >= core_threshold:
                sample_core_fp += fp_score

        # FN from unmatched GT WIPs
        for fn_wip in result.get("unmatched_gt_wips", []):
            fn_score = get_wip_score_int(fn_wip)
            sample_fn += fn_score
            if fn_score >= core_threshold:
                sample_core_fn += fn_score

        # Calculate per-sample F1s
        sample_overall_f1 = (
            2 * sample_tp / (2 * sample_tp + sample_fp + sample_fn)
            if (2 * sample_tp + sample_fp + sample_fn) > 0
            else 0.0
        )
        sample_core_f1 = (
            2 * sample_core_tp / (2 * sample_core_tp + sample_core_fp + sample_core_fn)
            if (2 * sample_core_tp + sample_core_fp + sample_core_fn) > 0
            else 0.0
        )

        per_sample[sample_id] = {
            "overall_f1": sample_overall_f1,
            "core_f1": sample_core_f1,
        }

    # Calculate macro F1 (average of per-sample F1s)
    valid_samples = [v for v in per_sample.values() if v]
    macro_f1 = (
        sum(s["overall_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0
    )
    macro_core_f1 = (
        sum(s["core_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0
    )

    return {
        "macro_wip_importance_weighted_f1": macro_f1,
        "macro_wip_importance_weighted_core_f1": macro_core_f1,
        "per_sample": per_sample,
    }


def calculate_double_weighted_metrics(
    match_results: Dict[str, Dict],
    core_threshold: int = 5,
) -> Dict[str, Any]:
    """
    Calculate double-weighted metrics using V6.2 logic (importance_score × match_quality) with macro and per-sample versions.

    NOTE: This function now uses pre-computed match_quality from match results (no BERTScore computation here).

    V6.2 Logic:
    - For matched pairs:
        - TP = gt_score × match_quality
        - FN = gt_score × (1 - match_quality)
        - FP = model_score × (1 - match_quality)
    - For unmatched GT WIPs: FN += gt_score (complete miss)
    - For unmatched model WIPs: FP += model_score (complete hallucination)

    Args:
        match_results: Dict of {sample_id: match_result} (with pre-computed match_quality)
        core_threshold: Threshold for core WIPs (importance_score >= threshold)

    Returns:
        Dict with macro F1, core versions, and per-sample F1s (double-weighted)
    """
    if not match_results:
        return {}

    # Per-sample metrics (for macro calculation)
    per_sample = {}

    for sample_id, result in match_results.items():
        if not result:
            per_sample[sample_id] = {"overall_f1": 0.0, "core_f1": 0.0}
            continue

        # Sample-level metrics
        sample_tp, sample_fp, sample_fn = 0.0, 0.0, 0.0
        sample_core_tp, sample_core_fp, sample_core_fn = 0.0, 0.0, 0.0

        # Process matched pairs using pre-computed match_quality
        matches = result.get("matches", [])
        for match in matches:
            gt_wip = match.get("gt_wip", {})
            model_wip = match.get("model_wip", {})
            match_quality = match.get("match_quality")

            # Skip if match_quality not computed
            if match_quality is None:
                continue

            gt_score = get_wip_score_int(gt_wip)
            model_score = get_wip_score_int(model_wip)

            # V6.2 formulas for all WIPs
            tp_contrib = gt_score * match_quality
            fn_contrib = gt_score * (1 - match_quality)
            fp_contrib = model_score * (1 - match_quality)

            sample_tp += tp_contrib
            sample_fn += fn_contrib
            sample_fp += fp_contrib

            # Core: V6.2 formulas only for WIPs with importance_score >= threshold
            if gt_score >= core_threshold:
                sample_core_tp += tp_contrib
                sample_core_fn += fn_contrib
            if model_score >= core_threshold:
                sample_core_fp += fp_contrib

        # Complete misses (unmatched GT WIPs)
        for fn_wip in result.get("unmatched_gt_wips", []):
            fn_score = get_wip_score_int(fn_wip)
            sample_fn += fn_score
            if fn_score >= core_threshold:
                sample_core_fn += fn_score

        # Complete hallucinations (unmatched model WIPs)
        for fp_wip in result.get("unmatched_model_wips", []):
            fp_score = get_wip_score_int(fp_wip)
            sample_fp += fp_score
            if fp_score >= core_threshold:
                sample_core_fp += fp_score

        # Calculate per-sample F1s
        sample_overall_f1 = (
            2 * sample_tp / (2 * sample_tp + sample_fp + sample_fn)
            if (2 * sample_tp + sample_fp + sample_fn) > 0
            else 0.0
        )
        sample_core_f1 = (
            2 * sample_core_tp / (2 * sample_core_tp + sample_core_fp + sample_core_fn)
            if (2 * sample_core_tp + sample_core_fp + sample_core_fn) > 0
            else 0.0
        )

        per_sample[sample_id] = {
            "overall_f1": sample_overall_f1,
            "core_f1": sample_core_f1,
        }

    # Calculate macro F1 (average of per-sample F1s)
    valid_samples = [v for v in per_sample.values() if v]
    macro_f1 = (
        sum(s["overall_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0
    )
    macro_core_f1 = (
        sum(s["core_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0
    )

    return {
        "macro_wip_double_weighted_f1": macro_f1,
        "macro_wip_double_weighted_core_f1": macro_core_f1,
        "per_sample": per_sample,
    }


def save_wip_detailed_results(
    save_dir: str,
    sample_ids: List[str],
    gt_wips_dict: Dict[str, List[Dict]],
    model_wips_dict: Dict[str, List[Dict]],
    match_results_dict: Dict[str, Dict],
    per_sample_f1s_dict: Dict[str, Dict],
    predictions_dict: Dict[str, str],
    references_dict: Dict[str, str],
    filename: str = "wip_results.json",
):
    """
    Save detailed WIP evaluation results to file.

    This saves:
    - prediction: Model prediction text for each sample
    - reference: Ground truth reference text for each sample
    - gt_wips: Ground truth information points for each sample
    - model_wips: Model-generated information points for each sample
    - match_result: Matching results (matches, unmatched_gt_wips, unmatched_model_wips)
    - per_sample_f1s: 6 F1 scores for each sample (3 types × 2 versions)

    Args:
        save_dir: Directory to save the file
        sample_ids: List of sample IDs
        gt_wips_dict: Dict of {sample_id: gt_wips_list}
        model_wips_dict: Dict of {sample_id: model_wips_list}
        match_results_dict: Dict of {sample_id: match_result}
        per_sample_f1s_dict: Dict of {sample_id: f1_scores_dict} with 6 F1 scores
        predictions_dict: Dict of {sample_id: prediction_text}
        references_dict: Dict of {sample_id: reference_text}
        filename: Output filename
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    detailed_results = {}

    for sample_id in sample_ids:
        detailed_results[sample_id] = {
            "prediction": predictions_dict.get(sample_id, ""),
            "reference": references_dict.get(sample_id, ""),
            "gt_wips": gt_wips_dict.get(sample_id, []),
            "model_wips": model_wips_dict.get(sample_id, []),
            "match_result": match_results_dict.get(sample_id, {}),
            "f1_scores": per_sample_f1s_dict.get(sample_id, {}),
        }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    logger.info(f"WIP detailed results saved to {save_path}")


def extract_after_think(text: str) -> str:
    """Extract text after the last と思 tag"""
    if "と思" in text:
        return text.split("と思")[-1].strip()
    return text


def _compute_bertscore(
    match_results: Dict[str, Dict], bertscore_model: str, bertscore_num_layers: int
) -> None:
    """Compute match quality from the current run's matched pairs."""
    matches = [match for result in match_results.values() for match in result["matches"]]
    if not matches:
        return

    import torch
    from bert_score import BERTScorer

    scorer = BERTScorer(
        model_type=bertscore_model,
        num_layers=bertscore_num_layers,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lang="zh",
        rescale_with_baseline=False,
    )
    _, _, f1 = scorer.score(
        [match["model_wip"]["info_point"] for match in matches],
        [match["gt_wip"]["info_point"] for match in matches],
    )
    for match, score in zip(matches, f1.tolist(), strict=True):
        if not 0 <= score <= 1 and not math.isclose(score, 1.0, abs_tol=1e-6):
            raise EvaluationError(f"Invalid WIP match quality: {score!r}")
        match["match_quality"] = min(score, 1.0)


def _check_wip_errors(stage: str, errors: Dict[str, str]) -> None:
    if errors:
        sample_id, reason = next(iter(errors.items()))
        raise EvaluationError(f"{stage} failed for {len(errors)} sample(s); {sample_id}: {reason}")


def evaluate_wip(
    predictions: Dict[str, str],
    references: Dict[str, str],
    llm_client,
    max_workers: int = 5,
    max_samples: Optional[int] = None,
    model_name: str = "unknown",
    save_dir: Optional[str] = None,
    bertscore_model: str = "bert-base-chinese",
    bertscore_num_layers: int = 9,
    core_threshold: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Main WIP evaluation function with six types of F1 metrics (3 types × 2 versions).

    Computes:
    1. Unweighted F1 (count-based) - overall and core
    2. Importance-weighted F1 (weighted by importance_score only) - overall and core
    3. Double-weighted F1 (V6.2 logic: importance_score × match_quality) - overall and core

    Args:
        predictions: Dict of {sample_id: prediction_text}
        references: Dict of {sample_id: reference_text}
        llm_client: LLM client instance for judge (with built-in retry mechanism)
        max_workers: Number of concurrent workers
        max_samples: Maximum samples to evaluate (None for all)
        model_name: Model name for cache file naming
        save_dir: Directory to save results
        bertscore_model: BERT model for computing match quality
        bertscore_num_layers: Number of layers for BERTScore
        core_threshold: Threshold for core WIPs (importance_score >= threshold)

    Returns:
        Tuple of (metrics, per_sample_metrics):
        - metrics: Dict with 6 F1 scores and sample count (flattened)
        - per_sample_metrics: Dict of {sample_id: {6 F1 scores}}
    """
    # Select samples (sorted by sample_id for consistency)
    all_sample_ids = sorted(set(predictions.keys()) & set(references.keys()))

    if max_samples is not None and max_samples < len(all_sample_ids):
        sample_ids = all_sample_ids[:max_samples]
        logger.info(f"Selected {len(sample_ids)} samples for WIP evaluation (sorted by sample_id)")
    else:
        sample_ids = all_sample_ids
        logger.info(f"Evaluating all {len(sample_ids)} samples for WIP")

    selected_predictions = {id: predictions[id] for id in sample_ids}
    selected_references = {id: references[id] for id in sample_ids}

    # Reference and prediction WIPs must both come from the current run.
    # The old Parquet and JSON files were indexed by sample ID alone.
    gt_wips, errors = extract_wips_batch(
        selected_references, llm_client, max_workers, "Extracting GT WIPs"
    )
    _check_wip_errors("Reference WIP extraction", errors)
    if missing_references := [sample_id for sample_id, wips in gt_wips.items() if not wips]:
        raise EvaluationError(
            f"Reference WIP extraction returned no points for {missing_references[0]}"
        )

    model_wips, errors = extract_wips_batch(
        {key: extract_after_think(value) for key, value in selected_predictions.items()},
        llm_client,
        max_workers,
        "Extracting Model WIPs",
    )
    _check_wip_errors("Prediction WIP extraction", errors)

    match_results, errors = match_wips_batch(gt_wips, model_wips, llm_client, max_workers)
    _check_wip_errors("WIP matching", errors)
    for sample_id, wips in model_wips.items():
        if not wips:
            match_results[sample_id] = {
                "matches": [],
                "unmatched_model_wips": [],
                "unmatched_gt_wips": gt_wips[sample_id],
            }

    if not match_results:
        raise EvaluationError("WIP Judge produced zero valid scores")
    _compute_bertscore(match_results, bertscore_model, bertscore_num_layers)

    # Step 4: Calculate three types of metrics (each with overall, core, and per-sample)
    logger.info("Computing all metrics (unweighted, importance-weighted, double-weighted)...")

    # 4.1: Unweighted metrics
    unweighted_metrics = calculate_unweighted_metrics(match_results, core_threshold=core_threshold)

    # 4.2: Importance-weighted metrics
    importance_metrics = calculate_importance_weighted_metrics(
        match_results, core_threshold=core_threshold
    )

    # 4.3: Double-weighted metrics (using pre-computed BERTScore)
    double_metrics = calculate_double_weighted_metrics(match_results, core_threshold=core_threshold)

    # Flattened overall metrics (6 F1 scores: 3 types × 2 versions)
    metrics = {
        # Macro F1 (average of per-sample F1s)
        "macro_wip_unweighted_f1": unweighted_metrics.get("macro_wip_unweighted_f1", 0.0),
        "macro_wip_unweighted_core_f1": unweighted_metrics.get("macro_wip_unweighted_core_f1", 0.0),
        "macro_wip_importance_weighted_f1": importance_metrics.get(
            "macro_wip_importance_weighted_f1", 0.0
        ),
        "macro_wip_importance_weighted_core_f1": importance_metrics.get(
            "macro_wip_importance_weighted_core_f1", 0.0
        ),
        "macro_wip_double_weighted_f1": double_metrics.get("macro_wip_double_weighted_f1", 0.0),
        "macro_wip_double_weighted_core_f1": double_metrics.get(
            "macro_wip_double_weighted_core_f1", 0.0
        ),
        "wip_num_samples": len(match_results),
    }

    # Merge per-sample metrics from all three types (6 F1 scores per sample, same as before since per-sample is used for macro)
    per_sample_metrics = {}
    for sample_id in sample_ids:
        unweighted_per_sample = unweighted_metrics.get("per_sample", {}).get(
            sample_id, {"overall_f1": 0.0, "core_f1": 0.0}
        )
        importance_per_sample = importance_metrics.get("per_sample", {}).get(
            sample_id, {"overall_f1": 0.0, "core_f1": 0.0}
        )
        double_per_sample = double_metrics.get("per_sample", {}).get(
            sample_id, {"overall_f1": 0.0, "core_f1": 0.0}
        )

        per_sample_metrics[sample_id] = {
            "wip_unweighted_f1": unweighted_per_sample["overall_f1"],
            "wip_unweighted_core_f1": unweighted_per_sample["core_f1"],
            "wip_importance_weighted_f1": importance_per_sample["overall_f1"],
            "wip_importance_weighted_core_f1": importance_per_sample["core_f1"],
            "wip_double_weighted_f1": double_per_sample["overall_f1"],
            "wip_double_weighted_core_f1": double_per_sample["core_f1"],
        }

    # Step 5: Save detailed results to file
    if save_dir:
        logger.info("Saving WIP detailed results...")
        save_wip_detailed_results(
            save_dir=save_dir,
            sample_ids=sample_ids,
            gt_wips_dict=gt_wips,
            model_wips_dict=model_wips,
            match_results_dict=match_results,
            per_sample_f1s_dict=per_sample_metrics,
            predictions_dict=selected_predictions,
            references_dict=selected_references,
            filename=f"wip_{model_name}.json",
        )

    logger.info(f"WIP evaluation completed: {metrics['wip_num_samples']} samples")
    logger.info(
        f"  Macro Unweighted F1: {metrics['macro_wip_unweighted_f1']:.4f} (Core: {metrics['macro_wip_unweighted_core_f1']:.4f})"
    )
    logger.info(
        f"  Macro Importance-weighted F1: {metrics['macro_wip_importance_weighted_f1']:.4f} (Core: {metrics['macro_wip_importance_weighted_core_f1']:.4f})"
    )
    logger.info(
        f"  Macro Double-weighted F1: {metrics['macro_wip_double_weighted_f1']:.4f} (Core: {metrics['macro_wip_double_weighted_core_f1']:.4f})"
    )

    return metrics, per_sample_metrics
