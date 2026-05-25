# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Core tests for GR input/output processor behavior."""

import os
import sys
import importlib
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_vllm_processor_runtime() -> bool:
    required_modules = [
        "vllm.v1.engine",
        "vllm.v1.engine.input_processor",
        "vllm.v1.engine.output_processor",
        "vllm.outputs",
    ]
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            return False
    return True


_HAS_VLLM_RUNTIME = _has_vllm_processor_runtime()
_SKIP_REASON = (
    "vLLM processor/runtime submodules are unavailable in this environment"
)
_INT32_MAX = (1 << 31) - 1


def _make_adapter(
    *,
    uih_feature_names=None,
    candidate_feature_names=None,
    context_feature_to_max_length=None,
    context_feature_to_min_uih_length=None,
    merge_uih_candidate_feature_mapping=None,
    uih_post_id_feature_name="item_id",
    candidate_primary_id_feature_name=None,
    task_names=None,
    max_seq_len=512,
    max_candidates=1000,
):
    """Build an HSTUModelAdapter with the given feature config."""
    from vllm_gr.adapters.hstu.adapter import HSTUModelAdapter

    if uih_feature_names is None:
        uih_feature_names = ["item_id"]
    if candidate_feature_names is None:
        candidate_feature_names = ["item_candidate_id"]
    if merge_uih_candidate_feature_mapping is None:
        merge_uih_candidate_feature_mapping = [
            [uih_feature_names[0], candidate_feature_names[0]],
        ]
    if context_feature_to_max_length is None:
        context_feature_to_max_length = {}
    if context_feature_to_min_uih_length is None:
        context_feature_to_min_uih_length = {}
    if task_names is None:
        task_names = ["is_click"]

    hf_config_kwargs = dict(
        hstu_uih_feature_names=uih_feature_names,
        user_embedding_feature_names=uih_feature_names,
        hstu_candidate_feature_names=candidate_feature_names,
        item_embedding_feature_names=candidate_feature_names,
        uih_post_id_feature_name=uih_post_id_feature_name,
        merge_uih_candidate_feature_mapping=merge_uih_candidate_feature_mapping,
        task_names=task_names,
        contextual_feature_to_max_length=context_feature_to_max_length,
        contextual_feature_to_min_uih_length=context_feature_to_min_uih_length,
    )
    if candidate_primary_id_feature_name is not None:
        hf_config_kwargs["candidate_primary_id_feature_name"] = candidate_primary_id_feature_name

    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(**hf_config_kwargs),
        ),
    )
    gr_config = types.SimpleNamespace(
        max_seq_len=max_seq_len,
        max_candidates=max_candidates,
    )
    return HSTUModelAdapter(vllm_config, gr_config)


def _make_processor_with_adapter(**adapter_kwargs):
    """Build a GRInputProcessor wired to an HSTUModelAdapter."""
    from vllm_gr.engine.gr_input_processor import GRInputProcessor

    adapter = _make_adapter(**adapter_kwargs)
    processor = GRInputProcessor.__new__(GRInputProcessor)
    processor._adapter = adapter
    processor._validate_lora = lambda _lora_request: None
    return processor


# ── Feature Extraction ────────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestExtractFeatures(unittest.TestCase):

    def test_extract_multi_key_features(self):
        from vllm_gr.gr_schema import extract_features

        records = [
            {"item_id": 1, "timestamp": 1000, "rating": 4},
            {"item_id": 2, "timestamp": 2000, "rating": 5},
            {"item_id": 3, "timestamp": 3000, "rating": 3},
        ]

        keys, values, lengths = extract_features(
            records=records,
            feature_names=["item_id", "timestamp", "rating"],
            group_name="features",
        )

        self.assertEqual(len(keys), 3)
        item_idx = keys.index("item_id")
        self.assertEqual(values[item_idx], [1, 2, 3])
        for length in lengths:
            self.assertEqual(length, 3)

    def test_extract_missing_key_rejected(self):
        from vllm_gr.gr_schema import extract_features

        records = [{"item_id": 1}, {"item_id": 2}]
        with self.assertRaisesRegex(ValueError, "do not match configured schema"):
            extract_features(
                records=records,
                feature_names=["item_id", "nonexistent_key"],
                group_name="features",
            )

    def test_extract_missing_value_rejected(self):
        from vllm_gr.gr_schema import extract_features

        records = [
            {"item_id": 1, "rating": 5},
            {"item_id": 2},
        ]

        with self.assertRaisesRegex(ValueError, "do not match configured schema"):
            extract_features(
                records=records,
                feature_names=["item_id", "rating"],
                group_name="features",
            )

    def test_process_gr_prompt_preserves_all_candidates(self):
        from vllm.pooling_params import PoolingParams

        processor = _make_processor_with_adapter(
            uih_feature_names=["item_id"],
            candidate_feature_names=["item_candidate_id"],
            max_candidates=1,
        )

        request = processor._process_gr_prompt(
            request_id="req-no-truncation",
            prompt={
                "user_id": "u1",
                "user_history": [{"item_id": 1}],
                "candidates": [
                    {"item_candidate_id": 10},
                    {"item_candidate_id": 20},
                    {"item_candidate_id": 30},
                ],
            },
            params=PoolingParams(task="embed"),
        )

        self.assertEqual(
            request.gr_features.cand_values_per_key,
            [[10, 20, 30]],
        )
        self.assertEqual(len(request.gr_features.original_candidates), 3)

    def test_process_gr_prompt_separates_contextual_features_from_history(self):
        from vllm.pooling_params import PoolingParams

        processor = _make_processor_with_adapter(
            uih_feature_names=["uih_post_id", "uih_owner_id"],
            candidate_feature_names=["item_post_id"],
            context_feature_to_max_length={"viewer_id": 2, "dummy_contexual": 2},
            context_feature_to_min_uih_length={"viewer_id": 2, "dummy_contexual": 2},
            merge_uih_candidate_feature_mapping=[
                ["uih_post_id", "item_post_id"],
            ],
            uih_post_id_feature_name="uih_post_id",
        )

        request = processor._process_gr_prompt(
            request_id="req-contextual-split",
            prompt={
                "user_id": "u1",
                "user_history": [
                    {"uih_post_id": 11, "uih_owner_id": 101},
                    {"uih_post_id": 12, "uih_owner_id": 102},
                ],
                "candidates": [{"item_post_id": 21}],
                "context": {
                    "viewer_id": 7,
                    "dummy_contexual": 1,
                },
            },
            params=PoolingParams(task="embed"),
        )

        self.assertEqual(
            request.gr_features.uih_keys,
            ["uih_post_id", "uih_owner_id"],
        )
        self.assertEqual(
            request.gr_features.ctx_keys,
            ["viewer_id", "dummy_contexual"],
        )
        self.assertEqual(
            request.gr_features.full_uih_keys,
            ["viewer_id", "dummy_contexual", "uih_post_id", "uih_owner_id"],
        )
        self.assertEqual(request.gr_features.uih_values_per_key, [[11, 12], [101, 102]])
        self.assertEqual(request.gr_features.ctx_values_per_key, [[7], [1]])

    def test_process_gr_prompt_builds_stable_prompt_identity(self):
        from vllm.pooling_params import PoolingParams

        processor = _make_processor_with_adapter(
            uih_feature_names=["item_id", "timestamp"],
            candidate_feature_names=["item_candidate_id", "item_query_time"],
            context_feature_to_max_length={"viewer_id": 2},
            context_feature_to_min_uih_length={"viewer_id": 2},
        )

        prompt = {
            "user_id": "u-cache",
            "user_history": [
                {"item_id": 11, "timestamp": 101},
                {"item_id": 12, "timestamp": 102},
                {"item_id": 13, "timestamp": 103},
            ],
            "candidates": [
                {"item_candidate_id": 21, "item_query_time": 1001},
                {"item_candidate_id": 22, "item_query_time": 1002},
            ],
            "context": {"viewer_id": 7},
        }

        request_a = processor._process_gr_prompt(
            request_id="req-a",
            prompt=prompt,
            params=PoolingParams(task="embed"),
        )
        request_b = processor._process_gr_prompt(
            request_id="req-b",
            prompt=prompt,
            params=PoolingParams(task="embed"),
        )

        self.assertEqual(request_a.prompt_token_ids, request_b.prompt_token_ids)
        self.assertEqual(request_a.cache_salt, request_b.cache_salt)
        self.assertTrue(request_a.cache_salt.startswith("hstu-prefix-v1:"))
        self.assertEqual(
            len(request_a.prompt_token_ids),
            2 + len(prompt["user_history"]) + len(prompt["candidates"]),
        )
        self.assertTrue(
            all(0 <= token_id <= _INT32_MAX for token_id in request_a.prompt_token_ids)
        )

    def test_process_gr_prompt_prompt_identity_changes_with_context_and_candidates(self):
        from vllm.pooling_params import PoolingParams

        processor = _make_processor_with_adapter(
            uih_feature_names=["item_id", "timestamp"],
            candidate_feature_names=["item_candidate_id", "item_query_time"],
            context_feature_to_max_length={"viewer_id": 2},
            context_feature_to_min_uih_length={"viewer_id": 2},
        )

        base_prompt = {
            "user_id": "u-diff",
            "user_history": [
                {"item_id": 11, "timestamp": 101},
                {"item_id": 12, "timestamp": 102},
                {"item_id": 13, "timestamp": 103},
            ],
            "candidates": [
                {"item_candidate_id": 21, "item_query_time": 1001},
                {"item_candidate_id": 22, "item_query_time": 1002},
            ],
            "context": {"viewer_id": 7},
        }
        context_changed_prompt = {
            **base_prompt,
            "context": {"viewer_id": 8},
        }
        candidate_changed_prompt = {
            **base_prompt,
            "candidates": [
                {"item_candidate_id": 99, "item_query_time": 1001},
                {"item_candidate_id": 22, "item_query_time": 1002},
            ],
        }

        base_request = processor._process_gr_prompt(
            request_id="req-base",
            prompt=base_prompt,
            params=PoolingParams(task="embed"),
        )
        context_request = processor._process_gr_prompt(
            request_id="req-context",
            prompt=context_changed_prompt,
            params=PoolingParams(task="embed"),
        )
        candidate_request = processor._process_gr_prompt(
            request_id="req-candidate",
            prompt=candidate_changed_prompt,
            params=PoolingParams(task="embed"),
        )

        self.assertNotEqual(
            base_request.prompt_token_ids[:2],
            context_request.prompt_token_ids[:2],
        )
        self.assertEqual(
            base_request.prompt_token_ids[2:5],
            candidate_request.prompt_token_ids[2:5],
        )
        self.assertNotEqual(
            base_request.prompt_token_ids[5:],
            candidate_request.prompt_token_ids[5:],
        )


# ── Output Construction ──────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestGRRequestState(unittest.TestCase):

    @staticmethod
    def _make_state(user_id="user_123", candidates=None,
                    primary_id_key="item_candidate_id"):
        from vllm_gr.engine.gr_output_processor import GRRequestState

        if candidates is None:
            candidates = [
                {"item_candidate_id": 10},
                {"item_candidate_id": 20},
                {"item_candidate_id": 30},
                {"item_candidate_id": 40},
                {"item_candidate_id": 50},
            ]

        adapter = _make_adapter()
        return GRRequestState(
            request_id="test-001",
            external_req_id="test-001",
            parent_req=None,
            request_index=0,
            lora_request=None,
            output_kind=None,
            prompt=None,
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            logprobs_processor=None,
            detokenizer=None,
            max_tokens_param=None,
            arrival_time=0.0,
            queue=None,
            log_stats=False,
            stream_interval=1,
            is_gr_request=True,
            gr_request_meta={
                "user_id": user_id,
                "original_candidates": candidates,
                "primary_id_key": primary_id_key,
                "task_names": [],
                "primary_task": "",
                "_adapter": adapter,
            },
        )

    def test_make_gr_output_basic(self):
        state = self._make_state()
        scores = torch.tensor([[0.1, 0.5, 0.3, 0.8, 0.2]])

        from vllm_gr.outputs import GRRequestOutput

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(output.user_id, "user_123")
        self.assertEqual(len(output.recommendations), 5)
        self.assertEqual(output.recommendations[0].item_id, 10)
        self.assertAlmostEqual(output.recommendations[0].score, 0.1)

    def test_make_gr_output_preserves_all_candidate_scores(self):
        candidates = [{"item_candidate_id": i} for i in range(10)]
        state = self._make_state(candidates=candidates)
        scores = torch.tensor([[0.1, 0.9, 0.3, 0.8, 0.5, 0.2, 0.7, 0.4, 0.6, 0.15]])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(len(output.recommendations), 10)
        self.assertEqual(output.recommendations[0].item_id, 0)
        self.assertEqual(output.recommendations[1].item_id, 1)
        self.assertAlmostEqual(output.recommendations[1].score, 0.9)

    def test_make_gr_output_rejects_candidate_score_mismatch(self):
        state = self._make_state(candidates=[{"item_candidate_id": 10}, {"item_candidate_id": 20}])
        scores = torch.tensor([[0.1]])

        with self.assertRaisesRegex(ValueError, "candidate/score count mismatch"):
            state._make_gr_output(scores, finish_reason="stop")

    def test_make_gr_output_rejects_non_2d_scores(self):
        state = self._make_state()

        with self.assertRaisesRegex(ValueError, "must be 2D"):
            state._make_gr_output(torch.tensor([0.1, 0.5]), finish_reason="stop")

    def test_make_request_output_keeps_empty_candidate_requests_in_gr_path(self):
        state = self._make_state(candidates=[])

        output = state.make_request_output(
            new_token_ids=[],
            pooling_output=torch.tensor([0.42]),
            finish_reason="stop",
            stop_reason=None,
        )

        from vllm_gr.outputs import GRRequestOutput

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(output.user_id, "user_123")
        self.assertEqual(output.recommendations, [])

    def test_make_gr_output_uses_explicit_primary_id_key(self):
        state = self._make_state(
            candidates=[
                {"rank_feature": 999, "item_candidate_id": 10},
                {"rank_feature": 555, "item_candidate_id": 20},
            ],
            primary_id_key="item_candidate_id",
        )

        output = state._make_gr_output(
            torch.tensor([[0.1, 0.2]]),
            finish_reason="stop",
        )

        self.assertEqual(output.item_ids, [10, 20])


# ── Multi-Task Output ────────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestMultiTaskOutput(unittest.TestCase):

    def _make_state_with_tasks(self, task_names, primary_task=""):
        from vllm_gr.engine.gr_output_processor import GRRequestState

        candidates = [
            {"item_candidate_id": 10},
            {"item_candidate_id": 20},
            {"item_candidate_id": 30},
        ]

        adapter = _make_adapter(task_names=task_names)

        return GRRequestState(
            request_id="mt-001",
            external_req_id="mt-001",
            parent_req=None,
            request_index=0,
            lora_request=None,
            output_kind=None,
            prompt=None,
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            logprobs_processor=None,
            detokenizer=None,
            max_tokens_param=None,
            arrival_time=0.0,
            queue=None,
            log_stats=False,
            stream_interval=1,
            is_gr_request=True,
            gr_request_meta={
                "user_id": "user_mt",
                "original_candidates": candidates,
                "primary_id_key": "item_candidate_id",
                "task_names": task_names,
                "primary_task": primary_task,
                "_adapter": adapter,
            },
        )

    def test_primary_task_selection(self):
        state = self._make_state_with_tasks(
            task_names=["is_click", "is_purchase"],
            primary_task="is_purchase",
        )
        scores = torch.tensor([
            [0.9, 0.1, 0.5],   # is_click
            [0.2, 0.8, 0.6],   # is_purchase
        ])

        output = state._make_gr_output(scores, finish_reason="stop")

        self.assertEqual(output.recommendations[0].item_id, 10)
        self.assertAlmostEqual(output.recommendations[0].score, 0.2)
        self.assertEqual(output.recommendations[1].item_id, 20)
        self.assertAlmostEqual(output.recommendations[1].score, 0.8)

    def test_per_item_task_scores(self):
        state = self._make_state_with_tasks(
            task_names=["is_click", "is_purchase"],
            primary_task="is_click",
        )
        scores = torch.tensor([
            [0.9, 0.1, 0.5],
            [0.2, 0.8, 0.6],
        ])

        output = state._make_gr_output(scores, finish_reason="stop")

        first = output.recommendations[0]
        self.assertIn("is_click", first.task_scores)
        self.assertIn("is_purchase", first.task_scores)
        self.assertAlmostEqual(first.task_scores["is_click"], 0.9, places=4)


# ── GRFeaturesPayload ────────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestGRFeaturesPayload(unittest.TestCase):

    def test_creation_and_serialization(self):
        """Payload round-trips through msgspec."""
        import msgspec
        from vllm_gr.engine import GRFeaturesPayload

        payload = GRFeaturesPayload(
            uih_keys=["item_id", "timestamp"],
            uih_values_per_key=[[1, 2, 3], [1000, 2000, 3000]],
            uih_lengths_per_key=[3, 3],
            cand_keys=["item_candidate_id"],
            cand_values_per_key=[[10, 20]],
            cand_lengths_per_key=[2],
            user_id="user_123",
        )

        self.assertEqual(payload.uih_keys, ["item_id", "timestamp"])
        self.assertEqual(payload.user_id, "user_123")

        data = msgspec.msgpack.encode(payload)
        decoded = msgspec.msgpack.decode(data, type=GRFeaturesPayload)
        self.assertEqual(decoded.uih_keys, ["item_id", "timestamp"])
        self.assertEqual(decoded.user_id, "user_123")


if __name__ == "__main__":
    unittest.main()
