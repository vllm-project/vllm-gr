# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for capability protocol isinstance checks."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import unittest
from types import SimpleNamespace

try:
    import torchrec  # noqa: F401
    import vllm  # noqa: F401

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from vllm_gr.adapters.capabilities import (
    GREmbeddingModel,
    GRPoolingModel,
    GRPrefixCacheSupport,
    GRScoringModel,
)


@unittest.skipUnless(_VLLM_AVAILABLE, "vLLM runtime not available")
class TestGRCapabilities(unittest.TestCase):
    """Verify that HSTUModelAdapter satisfies expected capability protocols."""

    def _make_hstu_adapter(self) -> None:
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
                    contextual_feature_to_max_length={},
                    contextual_feature_to_min_uih_length={},
                ),
            )
        )
        gr_config = SimpleNamespace(max_seq_len=512, max_candidates=1000)
        return HSTUModelAdapter(vllm_config, gr_config)

    def test_hstu_is_pooling_model(self) -> None:
        adapter = self._make_hstu_adapter()
        self.assertIsInstance(adapter, GRPoolingModel)

    def test_hstu_is_scoring_model(self) -> None:
        adapter = self._make_hstu_adapter()
        self.assertIsInstance(adapter, GRScoringModel)

    def test_hstu_is_not_embedding_model(self) -> None:
        """HSTU is a scoring model, not an embedding model."""
        adapter = self._make_hstu_adapter()
        self.assertNotIsInstance(adapter, GREmbeddingModel)

    def test_scoring_and_embedding_are_distinguishable(self) -> None:
        """Scoring and embedding protocols must produce different isinstance results."""

        class _FakeScoringAdapter:
            is_scoring_model = True

            def build_model_kwargs(self, batch_features, device) -> None:
                return {}

        class _FakeEmbeddingAdapter:
            is_embedding_model = True

            def build_model_kwargs(self, batch_features, device) -> None:
                return {}

        self.assertIsInstance(_FakeScoringAdapter(), GRScoringModel)
        self.assertNotIsInstance(_FakeScoringAdapter(), GREmbeddingModel)
        self.assertIsInstance(_FakeEmbeddingAdapter(), GREmbeddingModel)
        self.assertNotIsInstance(_FakeEmbeddingAdapter(), GRScoringModel)

    def test_hstu_supports_prefix_cache(self) -> None:
        adapter = self._make_hstu_adapter()
        self.assertIsInstance(adapter, GRPrefixCacheSupport)

    def test_plain_adapter_is_not_pooling_model(self) -> None:
        """A base adapter that doesn't implement the protocol is not detected."""

        class _Plain:
            pass

        self.assertNotIsInstance(_Plain(), GRPoolingModel)

    def test_adapter_without_capability_marker_is_neither_scoring_nor_embedding(self) -> None:
        """A class with build_model_kwargs but no marker matches neither."""

        class _NoMarker:
            def build_model_kwargs(self, batch_features, device) -> None:
                return {}

        self.assertNotIsInstance(_NoMarker(), GRScoringModel)
        self.assertNotIsInstance(_NoMarker(), GREmbeddingModel)

    def test_embedding_model_protocol_exists(self) -> None:
        """GREmbeddingModel is defined (for future use)."""
        self.assertTrue(callable(GREmbeddingModel))


if __name__ == "__main__":
    unittest.main()
