# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end output regression test for the adapter-driven output pipeline.

Validates that GROutputProcessor + HSTUModelAdapter produce correct
GRRequestOutput objects with the same score mapping as before the
refactoring.  This tests the exact output path used by both the sync
and async engines.
"""

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


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestOutputProcessorAdapterE2E(unittest.TestCase):
    """End-to-end test: adapter -> GROutputProcessor -> GRRequestOutput."""

    @staticmethod
    def _make_adapter_and_processor():
        from vllm_gr.adapters.hstu.adapter import HSTUModelAdapter
        from vllm_gr.engine.gr_output_processor import GROutputProcessor

        vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(
                    hstu_uih_feature_names=["item_id"],
                    user_embedding_feature_names=["item_id"],
                    hstu_candidate_feature_names=["item_candidate_id"],
                    item_embedding_feature_names=["item_candidate_id"],
                    uih_post_id_feature_name="item_id",
                    candidate_primary_id_feature_name="item_candidate_id",
                    merge_uih_candidate_feature_mapping=[
                        ["item_id", "item_candidate_id"],
                    ],
                    task_names=["is_click", "is_purchase"],
                    contextual_feature_to_max_length={},
                    contextual_feature_to_min_uih_length={},
                ),
            ),
        )
        gr_config = types.SimpleNamespace(max_seq_len=512, max_candidates=1000)
        adapter = HSTUModelAdapter(vllm_config, gr_config)
        processor = GROutputProcessor(
            tokenizer=None,
            log_stats=False,
            stream_interval=1,
            adapter=adapter,
        )
        return adapter, processor

    def test_single_request_output_matches_scores(self):
        """Full output path: scores -> GRRequestOutput with correct recommendations."""
        from vllm_gr.outputs import GRRequestOutput

        adapter, _ = self._make_adapter_and_processor()

        # Simulate what GROutputProcessor.add_request does:
        # Build a GRRequestState with adapter in meta
        from vllm_gr.engine.gr_output_processor import GRRequestState

        candidates = [
            {"item_candidate_id": 101},
            {"item_candidate_id": 202},
            {"item_candidate_id": 303},
        ]
        meta = {
            "user_id": "user-e2e",
            "original_candidates": candidates,
            "primary_id_key": "item_candidate_id",
            "task_names": ["is_click", "is_purchase"],
            "primary_task": "is_click",
            "request_id": "req-e2e-001",
            "_adapter": adapter,
        }

        state = GRRequestState(
            request_id="req-e2e-001",
            external_req_id="req-e2e-001",
            parent_req=None,
            request_index=0,
            lora_request=None,
            output_kind=None,
            prompt=None,
            prompt_token_ids=[0],
            prompt_embeds=None,
            logprobs_processor=None,
            detokenizer=None,
            max_tokens_param=None,
            arrival_time=0.0,
            queue=None,
            log_stats=False,
            stream_interval=1,
            is_gr_request=True,
            gr_request_meta=meta,
        )

        # Simulate model output: 2 tasks x 3 candidates
        scores = torch.tensor([
            [0.9, 0.1, 0.5],   # is_click
            [0.2, 0.8, 0.6],   # is_purchase
        ])

        output = state.make_request_output(
            new_token_ids=[],
            pooling_output=scores,
            finish_reason="stop",
            stop_reason=None,
        )

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(output.user_id, "user-e2e")
        self.assertEqual(output.request_id, "req-e2e-001")
        self.assertTrue(output.finished)
        self.assertEqual(len(output.recommendations), 3)

        # Verify primary task (is_click) scores
        self.assertEqual(output.recommendations[0].item_id, 101)
        self.assertAlmostEqual(output.recommendations[0].score, 0.9)
        self.assertEqual(output.recommendations[1].item_id, 202)
        self.assertAlmostEqual(output.recommendations[1].score, 0.1)
        self.assertEqual(output.recommendations[2].item_id, 303)
        self.assertAlmostEqual(output.recommendations[2].score, 0.5)

        # Verify per-item task scores
        self.assertIn("is_click", output.recommendations[0].task_scores)
        self.assertIn("is_purchase", output.recommendations[0].task_scores)
        self.assertAlmostEqual(
            output.recommendations[0].task_scores["is_click"], 0.9
        )
        self.assertAlmostEqual(
            output.recommendations[0].task_scores["is_purchase"], 0.2
        )
        self.assertAlmostEqual(
            output.recommendations[1].task_scores["is_purchase"], 0.8
        )

    def test_multi_request_output_boundaries(self):
        """Multiple requests produce independent, correctly bounded outputs."""
        from vllm_gr.outputs import GRRequestOutput

        adapter, _ = self._make_adapter_and_processor()
        from vllm_gr.engine.gr_output_processor import GRRequestState

        for req_idx in range(3):
            candidates = [
                {"item_candidate_id": req_idx * 10 + i}
                for i in range(2)
            ]
            meta = {
                "user_id": f"user-{req_idx}",
                "original_candidates": candidates,
                "primary_id_key": "item_candidate_id",
                "task_names": ["is_click"],
                "primary_task": "is_click",
                "request_id": f"req-{req_idx}",
                "_adapter": adapter,
            }
            state = GRRequestState(
                request_id=f"req-{req_idx}",
                external_req_id=f"req-{req_idx}",
                parent_req=None,
                request_index=0,
                lora_request=None,
                output_kind=None,
                prompt=None,
                prompt_token_ids=[0],
                prompt_embeds=None,
                logprobs_processor=None,
                detokenizer=None,
                max_tokens_param=None,
                arrival_time=0.0,
                queue=None,
                log_stats=False,
                stream_interval=1,
                is_gr_request=True,
                gr_request_meta=meta,
            )

            scores = torch.tensor([[0.7, 0.3]])
            output = state.make_request_output(
                new_token_ids=[],
                pooling_output=scores,
                finish_reason="stop",
                stop_reason=None,
            )

            self.assertIsInstance(output, GRRequestOutput)
            self.assertEqual(output.user_id, f"user-{req_idx}")
            self.assertEqual(len(output.recommendations), 2)
            self.assertEqual(
                output.recommendations[0].item_id,
                req_idx * 10,
            )
            self.assertAlmostEqual(output.recommendations[0].score, 0.7)

    def test_empty_candidates_produces_empty_recommendations(self):
        """Edge case: no candidates -> empty recommendations list."""
        from vllm_gr.outputs import GRRequestOutput

        adapter, _ = self._make_adapter_and_processor()
        from vllm_gr.engine.gr_output_processor import GRRequestState

        meta = {
            "user_id": "user-empty",
            "original_candidates": [],
            "primary_id_key": "item_candidate_id",
            "task_names": [],
            "primary_task": "",
            "request_id": "req-empty",
            "_adapter": adapter,
        }
        state = GRRequestState(
            request_id="req-empty",
            external_req_id="req-empty",
            parent_req=None,
            request_index=0,
            lora_request=None,
            output_kind=None,
            prompt=None,
            prompt_token_ids=[0],
            prompt_embeds=None,
            logprobs_processor=None,
            detokenizer=None,
            max_tokens_param=None,
            arrival_time=0.0,
            queue=None,
            log_stats=False,
            stream_interval=1,
            is_gr_request=True,
            gr_request_meta=meta,
        )

        output = state.make_request_output(
            new_token_ids=[],
            pooling_output=torch.tensor([]),
            finish_reason="stop",
            stop_reason=None,
        )

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(output.user_id, "user-empty")
        self.assertEqual(output.recommendations, [])

    def test_extract_request_meta_via_adapter_round_trip(self):
        """Verify adapter.extract_request_meta -> process_output round trip."""
        from vllm_gr.adapters.hstu.adapter import HSTUModelAdapter

        vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(
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
                    contextual_feature_to_max_length={},
                ),
            ),
        )
        gr_config = types.SimpleNamespace(max_seq_len=512, max_candidates=1000)
        adapter = HSTUModelAdapter(vllm_config, gr_config)

        # Build a payload via the full adapter path
        prompt = {
            "user_id": "user-roundtrip",
            "user_history": [{"item_id": 1}, {"item_id": 2}],
            "candidates": [
                {"item_candidate_id": 10},
                {"item_candidate_id": 20},
            ],
        }
        parsed = adapter.parse_prompt(prompt)
        payload = adapter.build_payload(parsed)

        # Extract meta (simulates what GROutputProcessor.add_request does)
        meta = adapter.extract_request_meta(payload)
        meta["request_id"] = "req-roundtrip"
        meta["_adapter"] = adapter

        # Feed scores through process_output
        scores = torch.tensor([[0.8, 0.2]])
        output = adapter.process_output(scores, meta)

        from vllm_gr.outputs import GRRequestOutput

        self.assertIsInstance(output, GRRequestOutput)
        self.assertEqual(output.user_id, "user-roundtrip")
        self.assertEqual(len(output.recommendations), 2)
        self.assertEqual(output.recommendations[0].item_id, 10)
        self.assertAlmostEqual(output.recommendations[0].score, 0.8)
        self.assertEqual(output.recommendations[1].item_id, 20)
        self.assertAlmostEqual(output.recommendations[1].score, 0.2)


if __name__ == "__main__":
    unittest.main()
