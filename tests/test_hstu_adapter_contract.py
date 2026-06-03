# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for HSTUModelAdapter method contracts."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import unittest
from types import SimpleNamespace

try:
    import torchrec  # noqa: F401
    import vllm  # noqa: F401

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


@unittest.skipUnless(_VLLM_AVAILABLE, "vLLM runtime not available")
class TestHSTUAdapterContract(unittest.TestCase):
    @staticmethod
    def _make_adapter() -> None:
        from vllm_gr.adapters.hstu.adapter import HSTUModelAdapter

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hstu_uih_feature_names=["item_id"],
                    user_embedding_feature_names=["item_id"],
                    hstu_candidate_feature_names=["item_candidate_id"],
                    item_embedding_feature_names=["item_candidate_id"],
                    uih_post_id_feature_name="item_id",
                    candidate_primary_id_feature_name="item_candidate_id",
                    merge_uih_candidate_feature_mapping=[
                        ["item_id", "item_candidate_id"],
                    ],
                    task_names=["is_click"],
                    task_types=["binary_classification"],
                    contextual_feature_to_max_length={},
                    contextual_feature_to_min_uih_length={},
                ),
            ),
        )
        gr_config = SimpleNamespace(max_seq_len=512, max_candidates=1000)
        return HSTUModelAdapter(vllm_config, gr_config)

    def setUp(self) -> None:
        self.adapter = self._make_adapter()

    def test_model_type_is_hstu(self) -> None:
        self.assertEqual(self.adapter.model_type, "hstu")

    def test_parse_prompt_extracts_features(self) -> None:
        prompt = {
            "user_id": "u1",
            "user_history": [{"item_id": 1}, {"item_id": 2}],
            "candidates": [{"item_candidate_id": 10}, {"item_candidate_id": 20}],
        }
        parsed = self.adapter.parse_prompt(prompt)
        # parse_prompt now returns GRParsedInput
        from vllm_gr.inputs.base import GRParsedInput

        self.assertIsInstance(parsed, GRParsedInput)
        self.assertEqual(parsed.user_id, "u1")
        self.assertEqual(parsed.raw_features["uih_keys"], ["item_id"])
        self.assertEqual(parsed.raw_features["cand_keys"], ["item_candidate_id"])
        self.assertEqual(parsed.metadata["history_len"], 2)
        self.assertEqual(parsed.metadata["num_candidates"], 2)
        self.assertTrue(len(parsed.schema_fingerprint) > 0, "schema_fingerprint must be non-empty")
        self.assertIn("candidates", parsed.original_prompt)

    def test_parse_prompt_treats_context_features_as_separate_from_history(self) -> None:
        """Configured context keys should not be required inside each history row."""

        from vllm_gr.adapters.hstu.adapter import HSTUModelAdapter

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hstu_uih_feature_names=[
                        "user_id",
                        "sex",
                        "age_group",
                        "item_id",
                        "action_timestamp",
                    ],
                    user_embedding_feature_names=["item_id", "user_id", "sex", "age_group"],
                    hstu_candidate_feature_names=["item_candidate_id", "item_query_time"],
                    item_embedding_feature_names=["item_candidate_id"],
                    uih_post_id_feature_name="item_id",
                    candidate_primary_id_feature_name="item_candidate_id",
                    merge_uih_candidate_feature_mapping=[
                        ["item_id", "item_candidate_id"],
                    ],
                    task_names=["is_click"],
                    contextual_feature_to_max_length={
                        "user_id": 1,
                        "sex": 1,
                        "age_group": 1,
                    },
                    contextual_feature_to_min_uih_length={},
                ),
            ),
        )
        gr_config = SimpleNamespace(max_seq_len=512, max_candidates=1000)
        adapter = HSTUModelAdapter(vllm_config, gr_config)

        prompt = {
            "user_id": "u1",
            "user_history": [
                {"item_id": 1, "action_timestamp": 100},
                {"item_id": 2, "action_timestamp": 200},
            ],
            "candidates": [{"item_candidate_id": 10, "item_query_time": 300}],
            "context": {"user_id": 7, "sex": 1, "age_group": 3},
        }
        parsed = adapter.parse_prompt(prompt)

        self.assertEqual(parsed.raw_features["uih_keys"], ["item_id", "action_timestamp"])
        self.assertEqual(parsed.raw_features["ctx_keys"], ["user_id", "sex", "age_group"])
        self.assertEqual(
            parsed.raw_features["full_uih_keys"],
            ["user_id", "sex", "age_group", "item_id", "action_timestamp"],
        )

    def test_build_payload_returns_hstu_features_payload(self) -> None:
        from vllm_gr.adapters.hstu.payload import HSTUFeaturesPayload

        parsed = self.adapter.parse_prompt(
            {
                "user_id": "u1",
                "user_history": [{"item_id": 1}],
                "candidates": [{"item_candidate_id": 10}],
            }
        )
        payload = self.adapter.build_payload(parsed)
        self.assertIsInstance(payload, HSTUFeaturesPayload)
        self.assertEqual(payload.user_id, "u1")

    def test_decode_payload_round_trip(self) -> None:
        parsed = self.adapter.parse_prompt(
            {
                "user_id": "u1",
                "user_history": [{"item_id": 1}, {"item_id": 2}],
                "candidates": [{"item_candidate_id": 10}],
            }
        )
        payload = self.adapter.build_payload(parsed)
        decoded = self.adapter.decode_payload(payload)

        self.assertEqual(decoded["uih_keys"], ["item_id"])
        self.assertEqual(decoded["cand_keys"], ["item_candidate_id"])
        self.assertIn("schema_fingerprint", decoded)
        self.assertIn("full_uih_keys", decoded)

    def test_build_model_kwargs_returns_kjt_dict(self) -> None:
        import torch

        parsed = self.adapter.parse_prompt(
            {
                "user_id": "u1",
                "user_history": [{"item_id": 1}],
                "candidates": [{"item_candidate_id": 10}],
            }
        )
        payload = self.adapter.build_payload(parsed)
        decoded = self.adapter.decode_payload(payload)

        batch_features = [("req-1", decoded)]
        result = self.adapter.build_model_kwargs(batch_features, torch.device("cpu"))
        self.assertIn("uih_features", result)
        self.assertIn("candidates_features", result)

    def test_extract_request_meta(self) -> None:
        parsed = self.adapter.parse_prompt(
            {
                "user_id": "u1",
                "user_history": [{"item_id": 1}],
                "candidates": [{"item_candidate_id": 10}],
            }
        )
        payload = self.adapter.build_payload(parsed)
        meta = self.adapter.extract_request_meta(payload)
        self.assertEqual(meta["user_id"], "u1")
        self.assertEqual(meta["task_names"], ["is_click"])
        self.assertIn("original_candidates", meta)

    def test_build_cache_salt_returns_string(self) -> None:
        parsed = self.adapter.parse_prompt(
            {
                "user_id": "u1",
                "user_history": [{"item_id": 1}],
                "candidates": [{"item_candidate_id": 10}],
            }
        )
        salt = self.adapter.build_cache_salt(parsed)
        self.assertIsInstance(salt, str)
        self.assertTrue(len(salt) > 0)

    def test_build_prompt_token_ids_returns_list(self) -> None:
        parsed = self.adapter.parse_prompt(
            {
                "user_id": "u1",
                "user_history": [{"item_id": 1}],
                "candidates": [{"item_candidate_id": 10}],
            }
        )
        token_ids = self.adapter.build_prompt_token_ids(parsed)
        self.assertIsInstance(token_ids, list)
        self.assertTrue(len(token_ids) > 0)
        self.assertTrue(all(isinstance(t, int) for t in token_ids))

    def test_process_output_returns_gr_request_output(self) -> None:
        import torch

        from vllm_gr.outputs import GRRequestOutput

        meta = {
            "request_id": "req-1",
            "user_id": "u1",
            "original_candidates": [{"item_candidate_id": 10}, {"item_candidate_id": 20}],
            "task_names": ["is_click"],
            "primary_task": "is_click",
            "primary_id_key": "item_candidate_id",
            "finish_reason": None,
        }
        scores = torch.tensor([[0.9, 0.1]])
        result = self.adapter.process_output(scores, meta)
        self.assertIsInstance(result, GRRequestOutput)
        self.assertEqual(result.user_id, "u1")
        self.assertEqual(len(result.recommendations), 2)

    def test_build_request_lengths_returns_none_when_kv_cache_disabled(self) -> None:
        """When enable_kv_cache is False (the default), returns None."""
        features = [
            {
                "cand_lengths_per_key": __import__("torch").tensor([2]),
                "uih_lengths_per_key": __import__("torch").tensor([5]),
            }
        ]
        result = self.adapter.build_request_lengths(features)
        self.assertIsNone(result)

    def test_build_request_lengths_returns_none_for_empty_features(self) -> None:
        """Returns None when features list is empty, even with kv cache enabled."""
        # Build an adapter with enable_kv_cache=True

        from vllm_gr.adapters.hstu.adapter import HSTUModelAdapter

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hstu_uih_feature_names=["item_id"],
                    user_embedding_feature_names=["item_id"],
                    hstu_candidate_feature_names=["item_candidate_id"],
                    item_embedding_feature_names=["item_candidate_id"],
                    uih_post_id_feature_name="item_id",
                    candidate_primary_id_feature_name="item_candidate_id",
                    merge_uih_candidate_feature_mapping=[
                        ["item_id", "item_candidate_id"],
                    ],
                    task_names=["is_click"],
                    enable_kv_cache=True,
                    contextual_feature_to_max_length={"country": 1},
                ),
            ),
        )
        gr_config = SimpleNamespace(max_seq_len=512, max_candidates=1000)
        adapter = HSTUModelAdapter(vllm_config, gr_config)
        result = adapter.build_request_lengths([])
        self.assertIsNone(result)

    def test_build_request_lengths_builds_tensors_with_kv_cache(self) -> None:
        """With enable_kv_cache=True and non-empty features, returns correct tensors."""
        import torch

        from vllm_gr.adapters.hstu.adapter import HSTUModelAdapter

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hstu_uih_feature_names=["item_id"],
                    user_embedding_feature_names=["item_id"],
                    hstu_candidate_feature_names=["item_candidate_id"],
                    item_embedding_feature_names=["item_candidate_id"],
                    uih_post_id_feature_name="item_id",
                    candidate_primary_id_feature_name="item_candidate_id",
                    merge_uih_candidate_feature_mapping=[
                        ["item_id", "item_candidate_id"],
                    ],
                    task_names=["is_click"],
                    enable_kv_cache=True,
                    contextual_feature_to_max_length={"country": 1, "age": 1},
                ),
            ),
        )
        gr_config = SimpleNamespace(max_seq_len=512, max_candidates=1000)
        adapter = HSTUModelAdapter(vllm_config, gr_config)

        features = [
            {
                "cand_lengths_per_key": torch.tensor([3]),
                "uih_lengths_per_key": torch.tensor([10]),
            },
            {
                "cand_lengths_per_key": torch.tensor([2]),
                "uih_lengths_per_key": torch.tensor([5]),
            },
        ]
        result = adapter.build_request_lengths(features)
        self.assertIsNotNone(result)
        num_targets, history_lengths, contextual_seq_len = result

        torch.testing.assert_close(num_targets, torch.tensor([3, 2], dtype=torch.int32))
        torch.testing.assert_close(history_lengths, torch.tensor([10, 5], dtype=torch.int32))
        self.assertEqual(contextual_seq_len, 2)  # country=1 + age=1

    def test_build_request_lengths_handles_missing_lengths_gracefully(self) -> None:
        """Features missing cand/uih lengths get 0 values."""

        from vllm_gr.adapters.hstu.adapter import HSTUModelAdapter

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hstu_uih_feature_names=["item_id"],
                    user_embedding_feature_names=["item_id"],
                    hstu_candidate_feature_names=["item_candidate_id"],
                    item_embedding_feature_names=["item_candidate_id"],
                    uih_post_id_feature_name="item_id",
                    candidate_primary_id_feature_name="item_candidate_id",
                    merge_uih_candidate_feature_mapping=[
                        ["item_id", "item_candidate_id"],
                    ],
                    task_names=["is_click"],
                    enable_kv_cache=True,
                    contextual_feature_to_max_length={},
                ),
            ),
        )
        gr_config = SimpleNamespace(max_seq_len=512, max_candidates=1000)
        adapter = HSTUModelAdapter(vllm_config, gr_config)

        # Feature dict with no length keys at all
        features = [{}]
        result = adapter.build_request_lengths(features)
        self.assertIsNotNone(result)
        num_targets, history_lengths, contextual_seq_len = result
        self.assertEqual(num_targets[0].item(), 0)
        self.assertEqual(history_lengths[0].item(), 0)
        self.assertEqual(contextual_seq_len, 0)


if __name__ == "__main__":
    unittest.main()
