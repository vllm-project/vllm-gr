# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FuXiAlphaModelAdapter — concrete adapter for FuXi-alpha recommendation models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_gr.inputs.base import GRParsedInput

from vllm_gr.adapters.base import GRModelAdapter
from vllm_gr.adapters.capabilities import GRPoolingModel, GRScoringModel
from vllm_gr.adapters.fuxi.feature_extractor import (
    FUXI_SCHEMA_FINGERPRINT,
    extract_fuxi_candidate_ids,
    extract_fuxi_history_features,
)
from vllm_gr.adapters.fuxi.payload_builder import build_fuxi_model_kwargs


class FuXiAlphaModelAdapter(GRModelAdapter, GRPoolingModel, GRScoringModel):
    """Adapter for FuXi-alpha generative recommendation models (RAB-off)."""

    model_type: str = "fuxi"
    payload_type: type = None
    is_scoring_model: bool = True

    def __init__(self, vllm_config, gr_config):
        super().__init__(vllm_config, gr_config)
        from vllm_gr.adapters.registry import get_payload_type

        self.payload_type = get_payload_type("fuxi")

        hf = self._hf_config
        self.max_seq_len: int = int(getattr(hf, "max_seq_len", gr_config.max_seq_len))
        self.max_candidates: int = int(
            getattr(hf, "max_num_candidates_inference", gr_config.max_candidates)
        )
        self.use_rating: bool = bool(getattr(hf, "use_rating_embedding", False))
        self.use_timestamp_payload: bool = bool(getattr(hf, "use_timestamp_payload", False))
        self.item_id_key: str = getattr(hf, "item_id_feature_name", None) or "item_id"
        self.rating_key: str = getattr(hf, "rating_feature_name", None) or "rating"
        self.ts_key: str = getattr(hf, "timestamp_feature_name", None) or "timestamp"
        self.task_names: list[str] = list(getattr(hf, "task_names", None) or ["score"])
        self.primary_task: str = self.task_names[0] if self.task_names else "score"

    def validate_prompt(self, prompt: dict) -> None:
        if "context" in prompt and prompt["context"]:
            raise ValueError("FuXi prompts do not support the 'context' field")

        user_history = prompt.get("user_history")
        if not isinstance(user_history, list) or not user_history:
            raise ValueError("FuXi prompts require a non-empty user_history list")

        candidates = prompt.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("FuXi prompts require a non-empty candidates list")

        if len(user_history) > self.max_seq_len:
            raise ValueError(
                f"user_history length {len(user_history)} exceeds max_seq_len {self.max_seq_len}"
            )

        if len(candidates) > self.max_candidates:
            raise ValueError(
                f"candidates length {len(candidates)} exceeds max_candidates {self.max_candidates}"
            )

    def parse_prompt(self, prompt: dict) -> "GRParsedInput":
        from vllm_gr.inputs.base import GRParsedInput

        user_id = str(prompt.get("user_id", ""))
        user_history = prompt.get("user_history", [])
        candidates = prompt.get("candidates", [])
        recent_history = user_history[-self.max_seq_len :]

        past_ids, past_ratings, past_timestamps = extract_fuxi_history_features(
            recent_history,
            item_id_key=self.item_id_key,
            rating_key=self.rating_key,
            ts_key=self.ts_key,
            use_rating=self.use_rating,
            use_timestamp=self.use_timestamp_payload,
        )
        candidate_ids = extract_fuxi_candidate_ids(
            candidates,
            item_id_key=self.item_id_key,
        )

        return GRParsedInput(
            user_id=user_id,
            schema_fingerprint=FUXI_SCHEMA_FINGERPRINT,
            raw_features={
                "past_ids": past_ids,
                "past_ratings": past_ratings,
                "past_timestamps": past_timestamps,
                "history_len": len(recent_history),
                "candidate_ids": candidate_ids,
                "num_candidates": len(candidates),
            },
            metadata={
                "history_len": len(recent_history),
                "num_candidates": len(candidates),
            },
            original_prompt=prompt,
        )

    def build_payload(self, parsed: "GRParsedInput") -> Any:
        from vllm_gr.adapters.fuxi.payload import FuXiFeaturesPayload

        rf = parsed.raw_features
        original_candidates = [
            dict(record) for record in parsed.original_prompt.get("candidates", [])
        ]

        return FuXiFeaturesPayload(
            past_ids=list(rf["past_ids"]),
            past_ratings=list(rf["past_ratings"]),
            past_timestamps=list(rf["past_timestamps"]),
            history_len=int(rf["history_len"]),
            candidate_ids=list(rf["candidate_ids"]),
            num_candidates=int(rf["num_candidates"]),
            user_id=parsed.user_id,
            original_candidates=original_candidates,
        )

    def decode_payload(self, payload: Any) -> dict[str, Any]:
        history_len = int(payload.history_len)
        if history_len != len(payload.past_ids):
            raise ValueError(
                "FuXi payload history_len mismatch: "
                f"history_len={history_len}, len(past_ids)={len(payload.past_ids)}"
            )
        num_candidates = int(payload.num_candidates)
        if num_candidates != len(payload.candidate_ids):
            raise ValueError(
                "FuXi payload num_candidates mismatch: "
                f"num_candidates={num_candidates}, "
                f"len(candidate_ids)={len(payload.candidate_ids)}"
            )

        return {
            "past_ids": list(payload.past_ids),
            "past_ratings": list(payload.past_ratings),
            "past_timestamps": list(payload.past_timestamps),
            "history_len": history_len,
            "num_candidates": num_candidates,
            "candidate_ids": list(payload.candidate_ids),
            "past_lengths": torch.tensor(history_len, dtype=torch.int64),
            "schema_fingerprint": FUXI_SCHEMA_FINGERPRINT,
            "user_id": getattr(payload, "user_id", ""),
            "original_candidates": list(getattr(payload, "original_candidates", []) or []),
        }

    def build_model_kwargs(
        self,
        batch_features: list[tuple[str, dict[str, Any]]],
        device,
    ) -> dict[str, Any]:
        return build_fuxi_model_kwargs(
            batch_features,
            device,
            use_rating=self.use_rating,
            use_timestamp=self.use_timestamp_payload,
        )

    def process_output(self, raw_output: Any, request_meta: dict) -> Any:
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
        if scores.dim() == 2:
            if primary_task and primary_task in task_names:
                primary_idx = task_names.index(primary_task)
            else:
                primary_idx = 0
            primary_scores = scores[primary_idx]
        elif scores.dim() == 1:
            primary_scores = scores
        else:
            raise ValueError(f"FuXi pooling output must be 1D or 2D: shape={tuple(scores.shape)}")

        if len(candidates) != len(primary_scores):
            raise ValueError(
                "GR candidate/score count mismatch: "
                f"candidates={len(candidates)}, scores={len(primary_scores)}"
            )

        recommendations = []
        for idx, candidate in enumerate(candidates):
            recommendations.append(
                RecommendedItem(
                    item_id=candidate.get(primary_id_key, idx),
                    score=float(primary_scores[idx]),
                    features=candidate,
                    task_scores=None,
                )
            )

        return GRRequestOutput(
            request_id=request_id,
            user_id=user_id,
            recommendations=recommendations,
            all_scores=raw_output,
            finished=finish_reason is not None,
        )

    def build_request_lengths(self, features: list[dict]) -> tuple | None:
        hf = self._hf_config
        if not bool(getattr(hf, "enable_kv_cache", True)):
            return None
        if not features:
            return None

        history_lengths: list[int] = []
        for feat in features:
            past_lengths = feat.get("past_lengths")
            if past_lengths is not None:
                history_lengths.append(int(past_lengths.item()))
            else:
                history_lengths.append(int(feat.get("history_len", 0)))

        device = features[0].get("past_lengths", torch.tensor(0)).device
        batch_size = len(features)
        return (
            torch.zeros(batch_size, dtype=torch.int32, device=device),
            torch.tensor(history_lengths, dtype=torch.int32, device=device),
            0,
        )

    def extract_request_meta(self, payload: Any) -> dict:
        return {
            "user_id": getattr(payload, "user_id", None),
            "original_candidates": list(getattr(payload, "original_candidates", []) or []),
            "task_names": self.task_names,
            "primary_task": self.primary_task,
            "primary_id_key": self.item_id_key,
        }


from vllm_gr.adapters.registry import register_adapter_class  # noqa: E402

register_adapter_class("fuxi", FuXiAlphaModelAdapter)
