# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


"""Tests for FuXiAlphaModelAdapter method contracts."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import math
import unittest
from types import SimpleNamespace

import torch


class TestFuXiAdapterContract(unittest.TestCase):
    @staticmethod
    def _make_adapter() -> None:
        from vllm_gr.adapters.fuxi.adapter import FuXiAlphaModelAdapter

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    max_seq_len=64,
                    max_num_candidates_inference=50,
                    use_rating_embedding=False,
                    use_timestamp_payload=False,
                    item_id_feature_name="item_id",
                    rating_feature_name="rating",
                    timestamp_feature_name="timestamp",
                    task_names=["score"],
                    enable_kv_cache=True,
                ),
            ),
        )
        gr_config = SimpleNamespace(max_seq_len=64, max_candidates=50)
        return FuXiAlphaModelAdapter(vllm_config, gr_config)

    def setUp(self) -> None:
        self.adapter = self._make_adapter()

    def test_model_type_is_fuxi(self) -> None:
        self.assertEqual(self.adapter.model_type, "fuxi")

    def test_parse_prompt_extracts_features(self) -> None:
        prompt = {
            "user_id": "u1",
            "user_history": [{"item_id": 1}, {"item_id": 2}],
            "candidates": [{"item_id": 10}, {"item_id": 20}],
        }
        parsed = self.adapter.parse_prompt(prompt)
        self.assertEqual(parsed.raw_features["past_ids"], [1, 2])
        self.assertEqual(parsed.raw_features["candidate_ids"], [10, 20])
        self.assertEqual(parsed.metadata["history_len"], 2)
        self.assertEqual(parsed.metadata["num_candidates"], 2)

    def test_validate_prompt_rejects_context(self) -> None:
        with self.assertRaises(ValueError):
            self.adapter.validate_prompt(
                {
                    "user_id": "u1",
                    "user_history": [{"item_id": 1}],
                    "candidates": [{"item_id": 10}],
                    "context": {"user_id": 1},
                }
            )

    def test_build_payload_returns_fuxi_features_payload(self) -> None:
        from vllm_gr.adapters.fuxi.payload import FuXiFeaturesPayload

        parsed = self.adapter.parse_prompt(
            {
                "user_id": "u1",
                "user_history": [{"item_id": 1}],
                "candidates": [{"item_id": 10}],
            }
        )
        payload = self.adapter.build_payload(parsed)
        self.assertIsInstance(payload, FuXiFeaturesPayload)
        self.assertEqual(payload.user_id, "u1")

    def test_decode_payload_round_trip(self) -> None:
        parsed = self.adapter.parse_prompt(
            {
                "user_id": "u1",
                "user_history": [{"item_id": 1}, {"item_id": 2}],
                "candidates": [{"item_id": 10}],
            }
        )
        payload = self.adapter.build_payload(parsed)
        decoded = self.adapter.decode_payload(payload)
        self.assertEqual(decoded["past_ids"], [1, 2])
        self.assertEqual(decoded["candidate_ids"], [10])
        self.assertEqual(decoded["schema_fingerprint"], "fuxi:v1")

    def test_build_model_kwargs_shapes(self) -> None:
        parsed = self.adapter.parse_prompt(
            {
                "user_id": "u1",
                "user_history": [{"item_id": 1}, {"item_id": 2}],
                "candidates": [{"item_id": 10}, {"item_id": 20}, {"item_id": 30}],
            }
        )
        payload = self.adapter.build_payload(parsed)
        decoded = self.adapter.decode_payload(payload)
        kwargs = self.adapter.build_model_kwargs([("r1", decoded)], torch.device("cpu"))

        self.assertEqual(kwargs["past_lengths"].tolist(), [2])
        self.assertEqual(kwargs["past_ids"].shape, (1, 2))
        self.assertEqual(kwargs["candidate_ids"].shape, (1, 3))
        self.assertEqual(kwargs["candidate_lengths"].tolist(), [3])

    def test_build_request_lengths_returns_zero_targets(self) -> None:
        parsed = self.adapter.parse_prompt(
            {
                "user_id": "u1",
                "user_history": [{"item_id": 1}, {"item_id": 2}],
                "candidates": [{"item_id": 10}],
            }
        )
        decoded = self.adapter.decode_payload(self.adapter.build_payload(parsed))
        num_targets, history_lengths, contextual_seq_len = self.adapter.build_request_lengths(
            [decoded]
        )
        self.assertEqual(num_targets.tolist(), [0])
        self.assertEqual(history_lengths.tolist(), [2])
        self.assertEqual(contextual_seq_len, 0)

    def test_extract_request_meta(self) -> None:
        parsed = self.adapter.parse_prompt(
            {
                "user_id": "u1",
                "user_history": [{"item_id": 1}],
                "candidates": [{"item_id": 10}],
            }
        )
        payload = self.adapter.build_payload(parsed)
        meta = self.adapter.extract_request_meta(payload)
        self.assertEqual(meta["task_names"], ["score"])
        self.assertEqual(meta["primary_task"], "score")
        self.assertEqual(meta["primary_id_key"], "item_id")


class TestFuXiDefaults(unittest.TestCase):
    def test_apply_fuxi_defaults_disables_prefix_cache(self) -> None:
        from vllm_gr.adapters.fuxi._defaults import apply_fuxi_defaults

        engine_args = SimpleNamespace(
            trust_remote_code=False,
            skip_tokenizer_init=False,
            runner=None,
            enforce_eager=False,
            disable_hybrid_kv_cache_manager=False,
            kv_cache_dtype="auto",
            enable_prefix_caching=True,
            max_model_len=None,
            max_seq_len=64,
            dtype="auto",
            model=None,
            allow_missing_weights=False,
            load_format=None,
            _is_ascend_platform=lambda: False,
            _has_model_weights=lambda: True,
            _read_local_config_dict=lambda: None,
        )
        apply_fuxi_defaults(engine_args)
        self.assertFalse(engine_args.enable_prefix_caching)


class TestFuXiModuleForward(unittest.TestCase):
    def test_forward_output_shape(self) -> None:
        from vllm_gr.config.fuxi_config import FuXiConfig
        from vllm_gr.models.fuxi.fuxi_module import FuXiAlphaInferenceModel

        cfg = FuXiConfig(
            max_seq_len=8,
            item_embedding_dim=16,
            num_blocks=1,
            num_heads=2,
            dqk=8,
            dv=8,
            vocab_size=128,
            enable_kv_cache=False,
        )
        model = FuXiAlphaInferenceModel(cfg, vllm_config=None)
        model.eval()

        past_lengths = torch.tensor([3], dtype=torch.int64)
        past_ids = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.int64)
        candidate_ids = torch.tensor([[10, 20, 0]], dtype=torch.int64)
        candidate_lengths = torch.tensor([2], dtype=torch.int64)

        with torch.no_grad():
            scores = model(
                past_lengths=past_lengths,
                past_ids=past_ids,
                past_payloads={},
                candidate_ids=candidate_ids,
                candidate_lengths=candidate_lengths,
            )

        self.assertEqual(tuple(scores.shape), (1, 2))
        self.assertTrue(torch.isfinite(scores).all().item())

    def test_uvqk_shape_is_rab_off(self) -> None:
        from vllm_gr.config.fuxi_config import FuXiConfig
        from vllm_gr.models.fuxi.fuxi_module import FuXiBlockJagged

        cfg = FuXiConfig(
            max_seq_len=8,
            item_embedding_dim=16,
            num_blocks=1,
            num_heads=2,
            dqk=8,
            dv=8,
            vocab_size=128,
            enable_kv_cache=False,
        )
        block = FuXiBlockJagged(
            embedding_dim=16,
            linear_hidden_dim=cfg.dv,
            attention_dim=cfg.dqk,
            dropout_ratio=0.0,
            attn_dropout_ratio=0.0,
            num_heads=cfg.num_heads,
            linear_activation="silu",
            relative_attention_bias_module=None,
            vllm_config=None,
            prefix="_fuxi._attention_layers.0",
            layer_idx=0,
            enable_kv_cache=False,
        )
        expected_out = cfg.dv * cfg.num_heads * 2 + cfg.dqk * cfg.num_heads * 2
        self.assertEqual(block._uvqk.shape, (16, expected_out))
        self.assertEqual(block.attn_layer.impl.scale, 1.0)


class TestFuXiPreprocessors(unittest.TestCase):
    def test_input_preprocessor_matches_fuxi_alpha_flow(self) -> None:
        from vllm_gr.models.fuxi.input_features_preprocessors import (
            LearnablePositionalEmbeddingInputFeaturesPreprocessor,
        )

        preproc = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=8,
            embedding_dim=4,
            dropout_rate=0.0,
        )
        past_lengths = torch.tensor([3], dtype=torch.int64)
        past_ids = torch.tensor([[11, 12, 13, 0, 0]], dtype=torch.int64)
        past_embeddings = torch.randn(1, 5, 4)
        _, user_embeddings, valid_mask = preproc(
            past_lengths,
            past_ids,
            past_embeddings,
            {},
        )
        expected = past_embeddings * 2.0 + preproc._pos_emb(
            torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
        )
        expected = expected * valid_mask
        self.assertTrue(torch.allclose(user_embeddings, expected))

    def test_get_current_embeddings_uses_left_padded_last_valid(self) -> None:
        from vllm_gr.models.fuxi.utils import get_current_embeddings

        encoded = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [9.0, 9.0],
                    [9.0, 9.0],
                ]
            ]
        )
        current = get_current_embeddings(torch.tensor([3]), encoded)
        self.assertEqual(current.tolist(), [[3.0, 0.0]])


class TestFuXiPayloadPadding(unittest.TestCase):
    def test_truncate_or_pad_seq_matches_fuxi_alpha_dataset(self) -> None:
        from vllm_gr.adapters.fuxi.payload_builder import _truncate_or_pad_seq

        self.assertEqual(_truncate_or_pad_seq([101, 102], 3), [101, 102, 0])
        self.assertEqual(_truncate_or_pad_seq([], 2), [0, 0])
        self.assertEqual(_truncate_or_pad_seq([1, 2, 3], 3), [1, 2, 3])

    def test_payload_builder_left_aligns_history(self) -> None:
        from vllm_gr.adapters.fuxi.payload_builder import build_fuxi_model_kwargs

        kwargs = build_fuxi_model_kwargs(
            [
                (
                    "r1",
                    {
                        "history_len": 2,
                        "num_candidates": 1,
                        "past_ids": [101, 102],
                        "past_ratings": [4, 5],
                        "past_timestamps": [1, 2],
                        "candidate_ids": [201],
                    },
                ),
                (
                    "r2",
                    {
                        "history_len": 3,
                        "num_candidates": 2,
                        "past_ids": [201, 202, 203],
                        "past_ratings": [3, 4, 5],
                        "past_timestamps": [3, 4, 5],
                        "candidate_ids": [301, 302],
                    },
                ),
            ],
            torch.device("cpu"),
            use_rating=False,
            use_timestamp=False,
        )
        self.assertEqual(kwargs["past_ids"][0].tolist(), [101, 102, 0])
        self.assertEqual(kwargs["past_ids"][1].tolist(), [201, 202, 203])
        self.assertEqual(kwargs["candidate_ids"][0].tolist(), [201, 0])
        self.assertEqual(kwargs["candidate_ids"][1].tolist(), [301, 302])


class TestFuXiKvCacheParity(unittest.TestCase):
    def test_kv_cache_and_fallback_match(self) -> None:
        from vllm_gr.config.fuxi_config import FuXiConfig
        from vllm_gr.models.fuxi.fuxi_module import FuXiAlphaInferenceModel

        torch.manual_seed(0)
        cfg_off = FuXiConfig(
            max_seq_len=8,
            item_embedding_dim=16,
            num_blocks=1,
            num_heads=2,
            dqk=8,
            dv=8,
            vocab_size=128,
            enable_kv_cache=False,
        )
        model_off = FuXiAlphaInferenceModel(cfg_off, vllm_config=None)
        model_off.eval()

        torch.manual_seed(0)
        cfg_on = FuXiConfig(
            max_seq_len=8,
            item_embedding_dim=16,
            num_blocks=1,
            num_heads=2,
            dqk=8,
            dv=8,
            vocab_size=128,
            enable_kv_cache=True,
        )
        model_on = FuXiAlphaInferenceModel(cfg_on, vllm_config=None)
        model_on.eval()

        past_lengths = torch.tensor([3], dtype=torch.int64)
        past_ids = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.int64)
        candidate_ids = torch.tensor([[10, 20, 0]], dtype=torch.int64)
        candidate_lengths = torch.tensor([2], dtype=torch.int64)

        with torch.no_grad():
            scores_off = model_off(
                past_lengths=past_lengths,
                past_ids=past_ids,
                past_payloads={},
                candidate_ids=candidate_ids,
                candidate_lengths=candidate_lengths,
            )
            scores_on = model_on(
                past_lengths=past_lengths,
                past_ids=past_ids,
                past_payloads={},
                candidate_ids=candidate_ids,
                candidate_lengths=candidate_lengths,
            )

        self.assertTrue(torch.allclose(scores_off, scores_on, atol=1e-2, rtol=1e-2))
        for value in scores_off.reshape(-1).tolist():
            self.assertTrue(math.isfinite(value))
