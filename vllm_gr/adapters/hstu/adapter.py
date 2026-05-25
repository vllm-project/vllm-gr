# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
HSTUModelAdapter — concrete adapter for HSTU-based recommendation models.

Implements all ``GRModelAdapter`` core methods and opts into the
``GRPoolingModel``, ``GRScoringModel``, and ``GRPrefixCacheSupport``
capability protocols.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_gr.adapters.base import GRModelAdapter
from vllm_gr.adapters.capabilities import (
    GRPrefixCacheSupport,
    GRPoolingModel,
    GRScoringModel,
)
from vllm_gr.adapters.hstu.feature_extractor import (
    compute_schema_fingerprint,
    extract_candidate_features,
    extract_context_features,
    extract_uih_features,
    validate_parsed_features,
)
from vllm_gr.inputs.hstu_prompt_identity import (
    build_hstu_cache_salt,
    build_hstu_prompt_token_ids,
)


class HSTUModelAdapter(GRModelAdapter, GRPoolingModel, GRScoringModel, GRPrefixCacheSupport):
    """Adapter for HSTU generative recommendation models.

    Reads all feature names, task metadata, and cache configuration from
    ``HSTUConfig`` (``hf_config``).
    """

    model_type: str = "hstu"
    payload_type: type = None  # set in __init__ from registry
    is_scoring_model: bool = True

    def __init__(self, vllm_config, gr_config):
        super().__init__(vllm_config, gr_config)
        # Resolve payload type via registry (lazily imported, avoids circular import).
        from vllm_gr.adapters.registry import get_payload_type
        self.payload_type = get_payload_type("hstu")

        self._resolve_feature_config()

    def _resolve_feature_config(self) -> None:
        """Read feature names and task metadata from HSTUConfig."""
        hf = self._hf_config
        cfg = self._gr_config

        configured_context_feature_names = list(dict.fromkeys(
            list(getattr(hf, "contextual_feature_to_max_length", {}).keys())
            + list(getattr(hf, "contextual_feature_to_min_uih_length", {}).keys())
        ))
        self.context_feature_names = configured_context_feature_names
        raw_uih_feature_names = list(
            getattr(hf, "hstu_uih_feature_names", None)
            or getattr(hf, "user_embedding_feature_names", None)
            or []
        )
        # HSTU configs often list context features inside hstu_uih_feature_names,
        # but prompts carry them separately under prompt["context"] rather than
        # duplicating them into each user_history record.
        self.uih_feature_names = [
            key for key in raw_uih_feature_names
            if key not in self.context_feature_names
        ]
        self.candidate_feature_names = list(
            getattr(hf, "hstu_candidate_feature_names", None)
            or getattr(hf, "item_embedding_feature_names", None)
            or []
        )
        self.full_uih_feature_names = list(
            dict.fromkeys(
                self.context_feature_names + self.uih_feature_names
            )
        )
        self.primary_id_feature_name = (
            getattr(hf, "uih_post_id_feature_name", None) or "item_id"
        )
        self.candidate_primary_id_key = self._resolve_candidate_primary_id_key()
        self.task_names = list(getattr(hf, "task_names", None) or [])
        self.primary_task = self.task_names[0] if self.task_names else ""

        self.max_seq_len = cfg.max_seq_len
        self.max_candidates = cfg.max_candidates

    def _resolve_candidate_primary_id_key(self) -> str:
        if not self.candidate_feature_names:
            return (
                getattr(self._hf_config, "candidate_primary_id_feature_name", None)
                or self.primary_id_feature_name
            )
        return self._resolve_candidate_primary_id_key_from_config(
            self._hf_config, self.candidate_feature_names
        )

    @staticmethod
    def _resolve_candidate_primary_id_key_from_config(hf_config, candidate_feature_names: list[str]) -> str:
        explicit = str(getattr(hf_config, "candidate_primary_id_feature_name", "") or "")
        if explicit:
            return explicit

        merge_mapping = list(
            getattr(hf_config, "merge_uih_candidate_feature_mapping", []) or []
        )
        primary_uih_id_key = getattr(hf_config, "uih_post_id_feature_name", "")
        for uih_key, candidate_key in merge_mapping:
            if uih_key == primary_uih_id_key:
                return candidate_key

        raise ValueError(
            "Unable to resolve candidate primary id feature. Configure "
            "`candidate_primary_id_feature_name` explicitly or provide a "
            "`merge_uih_candidate_feature_mapping` entry that maps "
            f"{primary_uih_id_key!r} to one of {candidate_feature_names!r}."
        )

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def parse_prompt(self, prompt: dict) -> "GRParsedInput":
        """Extract and validate UIH / candidate / context features.

        Returns a GRParsedInput instance with feature data in
        ``raw_features`` and metadata in ``metadata``.
        """
        from vllm_gr.inputs.base import GRParsedInput

        user_id = str(prompt.get("user_id", ""))
        user_history = prompt.get("user_history", [])
        candidates = prompt.get("candidates", [])
        recent_history = user_history[-self.max_seq_len:]

        uih_keys, uih_values_per_key, uih_lengths_per_key = extract_uih_features(
            recent_history, self.uih_feature_names,
        )
        cand_keys, cand_values_per_key, cand_lengths_per_key = extract_candidate_features(
            candidates, self.candidate_feature_names,
        )

        context = prompt.get("context") or {}
        ctx_keys, ctx_values_per_key = extract_context_features(
            context, feature_names=self.context_feature_names,
        )

        uih_keys, cand_keys, ctx_keys, schema_fingerprint = compute_schema_fingerprint(
            uih_keys=uih_keys, cand_keys=cand_keys, ctx_keys=ctx_keys,
        )
        validate_parsed_features(
            uih_keys=uih_keys,
            uih_values_per_key=uih_values_per_key,
            uih_lengths_per_key=uih_lengths_per_key,
            cand_keys=cand_keys,
            cand_values_per_key=cand_values_per_key,
            cand_lengths_per_key=cand_lengths_per_key,
            ctx_keys=ctx_keys,
            ctx_values_per_key=ctx_values_per_key,
        )

        return GRParsedInput(
            user_id=user_id,
            schema_fingerprint=schema_fingerprint,
            raw_features={
                "uih_keys": uih_keys,
                "uih_values_per_key": uih_values_per_key,
                "uih_lengths_per_key": uih_lengths_per_key,
                "cand_keys": cand_keys,
                "cand_values_per_key": cand_values_per_key,
                "cand_lengths_per_key": cand_lengths_per_key,
                "ctx_keys": ctx_keys,
                "ctx_values_per_key": ctx_values_per_key,
                "full_uih_keys": list(self.full_uih_feature_names),
            },
            metadata={
                "history_len": len(recent_history),
                "num_candidates": len(candidates),
            },
            original_prompt=prompt,
        )

    def build_payload(self, parsed: "GRParsedInput") -> Any:
        """Build ``GRFeaturesPayload`` from parsed features."""
        from vllm_gr.engine import GRFeaturesPayload

        rf = parsed.raw_features
        original_candidates = [
            dict(c) for c in parsed.original_prompt.get("candidates", [])
        ]

        return GRFeaturesPayload(
            uih_keys=rf["uih_keys"],
            uih_values_per_key=rf["uih_values_per_key"],
            uih_lengths_per_key=rf["uih_lengths_per_key"],
            cand_keys=rf["cand_keys"],
            cand_values_per_key=rf["cand_values_per_key"],
            cand_lengths_per_key=rf["cand_lengths_per_key"],
            ctx_keys=rf["ctx_keys"],
            ctx_values_per_key=rf["ctx_values_per_key"],
            full_uih_keys=rf["full_uih_keys"],
            user_id=parsed.user_id,
            original_candidates=original_candidates,
        )

    def decode_payload(self, payload: Any) -> dict[str, Any]:
        """Decode ``GRFeaturesPayload`` into per-feature tensors."""
        uih_keys = list(payload.uih_keys)
        cand_keys = list(payload.cand_keys)
        ctx_keys = list(payload.ctx_keys)
        (
            uih_keys, cand_keys, ctx_keys, schema_fingerprint,
        ) = compute_schema_fingerprint(
            uih_keys=uih_keys, cand_keys=cand_keys, ctx_keys=ctx_keys,
        )
        validate_parsed_features(
            uih_keys=uih_keys,
            uih_values_per_key=list(payload.uih_values_per_key),
            uih_lengths_per_key=list(payload.uih_lengths_per_key),
            cand_keys=cand_keys,
            cand_values_per_key=list(payload.cand_values_per_key),
            cand_lengths_per_key=list(payload.cand_lengths_per_key),
            ctx_keys=ctx_keys,
            ctx_values_per_key=list(payload.ctx_values_per_key),
        )

        full_uih_keys = list(
            getattr(payload, "full_uih_keys", []) or (uih_keys + ctx_keys)
        )
        combined_uih_keys = uih_keys + ctx_keys
        if (
            len(full_uih_keys) != len(combined_uih_keys)
            or set(full_uih_keys) != set(combined_uih_keys)
        ):
            raise ValueError(
                "GR payload full_uih_keys do not match decoded UIH/context keys: "
                f"full_uih_keys={full_uih_keys!r}, combined_keys={combined_uih_keys!r}"
            )

        return {
            "uih_keys": uih_keys,
            "uih_values_per_key": [
                torch.tensor(values, dtype=torch.int64)
                for values in payload.uih_values_per_key
            ],
            "uih_lengths_per_key": torch.tensor(
                payload.uih_lengths_per_key, dtype=torch.int64,
            ),
            "cand_keys": cand_keys,
            "cand_values_per_key": [
                torch.tensor(values, dtype=torch.int64)
                for values in payload.cand_values_per_key
            ],
            "cand_lengths_per_key": torch.tensor(
                payload.cand_lengths_per_key, dtype=torch.int64,
            ),
            "ctx_keys": ctx_keys,
            "ctx_values_per_key": [
                torch.tensor(values, dtype=torch.int64)
                for values in payload.ctx_values_per_key
            ],
            "full_uih_keys": full_uih_keys,
            "schema_fingerprint": schema_fingerprint,
        }

    def build_model_kwargs(
        self,
        batch_features: list[tuple[str, dict[str, Any]]],
        device,
    ) -> dict[str, Any]:
        """Build KJT model kwargs for HSTU forward pass."""
        from vllm_gr.adapters.hstu.kjt_builder import build_hstu_model_kwargs

        return build_hstu_model_kwargs(batch_features, device)

    def process_output(self, raw_output: Any, request_meta: dict) -> Any:
        """Convert 2D pooling scores ``(num_tasks, num_candidates)`` into
        ``GRRequestOutput`` with ``RecommendedItem`` list."""
        from vllm_gr.outputs import GRRequestOutput, RecommendedItem

        candidates = request_meta.get("original_candidates", [])
        user_id = request_meta.get("user_id")
        task_names = request_meta.get("task_names", [])
        primary_task = request_meta.get("primary_task", "")
        primary_id_key = request_meta.get("primary_id_key", "item_id")
        request_id = request_meta.get("request_id", "")
        finish_reason = request_meta.get("finish_reason")

        if not candidates or raw_output is None:
            return GRRequestOutput(
                request_id=request_id,
                user_id=user_id,
                recommendations=[],
                all_scores=raw_output,
                finished=finish_reason is not None,
            )

        scores = raw_output.float()
        if scores.dim() != 2:
            raise ValueError(
                "GR pooling output must be 2D (num_tasks, num_candidates): "
                f"shape={tuple(scores.shape)}"
            )

        all_task_scores: dict[str, list[float]] | None = None
        if task_names:
            if primary_task and primary_task in task_names:
                primary_idx = task_names.index(primary_task)
            else:
                primary_idx = 0
            primary_scores = scores[primary_idx]
            all_task_scores = {
                name: scores[i].tolist()
                for i, name in enumerate(task_names)
            }
        else:
            primary_scores = scores[0]

        if len(candidates) != len(primary_scores):
            raise ValueError(
                "GR candidate/score count mismatch: "
                f"candidates={len(candidates)}, scores={len(primary_scores)}"
            )

        recommendations = []
        for idx in range(len(candidates)):
            candidate = candidates[idx]
            item_task_scores = None
            if all_task_scores is not None:
                item_task_scores = {t: all_task_scores[t][idx] for t in all_task_scores}
            recommendations.append(RecommendedItem(
                item_id=candidate.get(primary_id_key, idx),
                score=float(primary_scores[idx]),
                features=candidate,
                task_scores=item_task_scores,
            ))

        return GRRequestOutput(
            request_id=request_id,
            user_id=user_id,
            recommendations=recommendations,
            all_scores=raw_output,
            finished=finish_reason is not None,
        )

    # ------------------------------------------------------------------
    # Optional methods
    # ------------------------------------------------------------------

    def build_request_lengths(self, features: list[dict]) -> tuple | None:
        """Build ``(num_targets, history_lengths, contextual_seq_len)``."""
        hf = self._hf_config
        if not bool(getattr(hf, "enable_kv_cache", False)):
            return None
        if not features:
            return None

        num_targets: list[int] = []
        history_lengths: list[int] = []
        for feat in features:
            cand_lengths = feat.get("cand_lengths_per_key")
            if cand_lengths is not None:
                num_targets.append(
                    int(cand_lengths.max().item()) if cand_lengths.numel() > 0 else 0
                )
            else:
                num_targets.append(0)

            uih_lengths = feat.get("uih_lengths_per_key")
            if uih_lengths is not None:
                history_lengths.append(
                    int(uih_lengths.max().item()) if uih_lengths.numel() > 0 else 0
                )
            else:
                history_lengths.append(0)

        contextual_seq_len = int(
            sum(getattr(hf, "contextual_feature_to_max_length", {}).values())
        )
        device = features[0].get("uih_lengths_per_key", torch.tensor([])).device
        return (
            torch.tensor(num_targets, dtype=torch.int32, device=device),
            torch.tensor(history_lengths, dtype=torch.int32, device=device),
            contextual_seq_len,
        )

    def build_prompt_token_ids(self, parsed: "GRParsedInput") -> list[int] | None:
        """Build stable synthetic token IDs for HSTU prefix-cache hashing."""
        hf = self._hf_config
        contextual_feature_to_max_length = {
            str(k): int(v)
            for k, v in (getattr(hf, "contextual_feature_to_max_length", {}) or {}).items()
        }
        contextual_feature_to_min_uih_length = {
            str(k): int(v)
            for k, v in (getattr(hf, "contextual_feature_to_min_uih_length", {}) or {}).items()
        }
        rf = parsed.raw_features
        return build_hstu_prompt_token_ids(
            uih_keys=rf["uih_keys"],
            uih_values_per_key=rf["uih_values_per_key"],
            cand_keys=rf["cand_keys"],
            cand_values_per_key=rf["cand_values_per_key"],
            ctx_keys=rf["ctx_keys"],
            ctx_values_per_key=rf["ctx_values_per_key"],
            history_len=parsed.metadata["history_len"],
            num_candidates=parsed.metadata["num_candidates"],
            contextual_feature_to_max_length=contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=contextual_feature_to_min_uih_length,
        )

    def build_cache_salt(self, parsed: "GRParsedInput") -> str | None:
        """Build cache-salt string for HSTU prefix-cache identity."""
        hf = self._hf_config
        contextual_feature_to_max_length = {
            str(k): int(v)
            for k, v in (getattr(hf, "contextual_feature_to_max_length", {}) or {}).items()
        }
        contextual_feature_to_min_uih_length = {
            str(k): int(v)
            for k, v in (getattr(hf, "contextual_feature_to_min_uih_length", {}) or {}).items()
        }
        model_type = getattr(hf, "model_type", None) or "hstu"
        return build_hstu_cache_salt(
            schema_fingerprint=parsed.schema_fingerprint,
            contextual_feature_to_max_length=contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=contextual_feature_to_min_uih_length,
            primary_id_feature_name=self.primary_id_feature_name,
            model_type=model_type,
        )

    def extract_request_meta(self, payload: Any) -> dict:
        """Extract metadata from a GRFeaturesPayload for output processing."""
        return {
            "user_id": getattr(payload, "user_id", None),
            "original_candidates": list(getattr(payload, "original_candidates", [])),
            "task_names": self.task_names,
            "primary_task": self.primary_task,
            "primary_id_key": self.candidate_primary_id_key,
        }


# ── Layer-2 registration (heavy: torch/torchrec now loaded) ──────
from vllm_gr.adapters.registry import register_adapter_class

register_adapter_class("hstu", HSTUModelAdapter)
