# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


"""Contract tests for HSTU prompt identity with vLLM native prefix caching."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import importlib
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _has_prefix_cache_runtime() -> bool:
    required_modules = [
        "vllm.pooling_params",
        "vllm.utils.hashing",
        "vllm.v1.core.kv_cache_utils",
        "vllm.v1.request",
        "vllm.v1.engine.input_processor",
    ]
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            return False
    return True


_HAS_PREFIX_CACHE_RUNTIME = _has_prefix_cache_runtime()
_SKIP_REASON = "vLLM prefix-cache runtime submodules are unavailable"


@unittest.skipUnless(_HAS_PREFIX_CACHE_RUNTIME, _SKIP_REASON)
class TestHSTUPrefixCacheContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from vllm.utils.hashing import sha256_cbor
        from vllm.v1.core.kv_cache_utils import init_none_hash

        init_none_hash(sha256_cbor)

    @staticmethod
    def _make_processor() -> None:
        from vllm_gr.adapters.hstu.adapter import HSTUModelAdapter
        from vllm_gr.engine.gr_input_processor import GRInputProcessor

        vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(
                    hstu_uih_feature_names=["item_id", "timestamp"],
                    user_embedding_feature_names=["item_id", "timestamp"],
                    hstu_candidate_feature_names=["item_candidate_id", "item_query_time"],
                    item_embedding_feature_names=["item_candidate_id", "item_query_time"],
                    uih_post_id_feature_name="item_id",
                    candidate_primary_id_feature_name="item_candidate_id",
                    merge_uih_candidate_feature_mapping=[
                        ["item_id", "item_candidate_id"],
                    ],
                    task_names=["is_click"],
                    contextual_feature_to_max_length={"viewer_id": 2},
                    contextual_feature_to_min_uih_length={"viewer_id": 0},
                ),
            ),
        )
        gr_config = types.SimpleNamespace(max_seq_len=512, max_candidates=8)
        adapter = HSTUModelAdapter(vllm_config, gr_config)

        processor = GRInputProcessor.__new__(GRInputProcessor)
        processor._adapter = adapter
        processor._validate_lora = lambda _lora_request: None
        return processor

    @staticmethod
    def _shared_prefix_block_count(left: list[bytes], right: list[bytes]) -> int:
        shared = 0
        for left_hash, right_hash in zip(left, right):
            if left_hash != right_hash:
                break
            shared += 1
        return shared

    def _build_request_hashes(
        self,
        *,
        prompt: dict[str, object],
        block_size: int = 2,
        cache_salt_override: str | None = None,
    ) -> tuple[list[int], str, list[bytes]]:
        from vllm.pooling_params import PoolingParams
        from vllm.utils.hashing import sha256_cbor
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher

        from vllm_gr.request import GRRequest

        processor = self._make_processor()
        engine_request = processor._process_gr_prompt(
            request_id="req-prefix-cache",
            prompt=prompt,
            params=PoolingParams(task="embed"),
        )
        if cache_salt_override is not None:
            engine_request.cache_salt = cache_salt_override

        block_hasher = get_request_block_hasher(
            block_size=block_size,
            caching_hash_fn=sha256_cbor,
        )
        request = GRRequest.from_engine_core_request(engine_request, block_hasher)
        return (
            list(request.prompt_token_ids or []),
            str(request.cache_salt),
            list(request.block_hashes),
        )

    def _build_request(
        self,
        *,
        request_id: str,
        prompt: dict[str, object],
        block_size: int = 2,
        cache_salt_override: str | None = None,
    ) -> None:
        from vllm.pooling_params import PoolingParams
        from vllm.utils.hashing import sha256_cbor
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher

        from vllm_gr.request import GRRequest

        processor = self._make_processor()
        engine_request = processor._process_gr_prompt(
            request_id=request_id,
            prompt=prompt,
            params=PoolingParams(task="embed"),
        )
        if cache_salt_override is not None:
            engine_request.cache_salt = cache_salt_override

        block_hasher = get_request_block_hasher(
            block_size=block_size,
            caching_hash_fn=sha256_cbor,
        )
        return GRRequest.from_engine_core_request(engine_request, block_hasher)

    @staticmethod
    def _make_kv_cache_config(block_size: int, num_blocks: int) -> None:
        import torch
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
        )

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        )

    def _prime_prefix_cache(
        self,
        *,
        manager,
        request,
    ) -> None:
        computed_blocks, num_computed_tokens = manager.get_computed_blocks(request)
        self.assertEqual(num_computed_tokens, 0)
        self.assertFalse(computed_blocks.blocks[0])

        new_blocks = manager.allocate_slots(
            request,
            request.num_tokens,
            num_computed_tokens,
            computed_blocks,
        )
        self.assertIsNotNone(new_blocks)
        manager.free(request)

    def test_shared_context_and_history_preserve_prefix_block_hashes(self) -> None:
        base_prompt = {
            "user_id": "u-prefix",
            "user_history": [
                {"item_id": 11, "timestamp": 101},
                {"item_id": 12, "timestamp": 102},
                {"item_id": 13, "timestamp": 103},
                {"item_id": 14, "timestamp": 104},
            ],
            "candidates": [
                {"item_candidate_id": 21, "item_query_time": 1001},
                {"item_candidate_id": 22, "item_query_time": 1002},
            ],
            "context": {"viewer_id": 7},
        }
        candidate_changed_prompt = {
            **base_prompt,
            "candidates": [
                {"item_candidate_id": 99, "item_query_time": 1001},
                {"item_candidate_id": 98, "item_query_time": 1002},
            ],
        }

        base_tokens, _base_salt, base_hashes = self._build_request_hashes(
            prompt=base_prompt,
            block_size=2,
        )
        changed_tokens, _changed_salt, changed_hashes = self._build_request_hashes(
            prompt=candidate_changed_prompt,
            block_size=2,
        )

        self.assertEqual(base_tokens[:6], changed_tokens[:6])
        self.assertNotEqual(base_tokens[6:], changed_tokens[6:])
        self.assertEqual(
            self._shared_prefix_block_count(base_hashes, changed_hashes),
            3,
        )

    def test_context_change_breaks_first_prefix_block_hash(self) -> None:
        base_prompt = {
            "user_id": "u-context",
            "user_history": [
                {"item_id": 11, "timestamp": 101},
                {"item_id": 12, "timestamp": 102},
                {"item_id": 13, "timestamp": 103},
                {"item_id": 14, "timestamp": 104},
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

        _base_tokens, _base_salt, base_hashes = self._build_request_hashes(
            prompt=base_prompt,
            block_size=2,
        )
        _changed_tokens, _changed_salt, changed_hashes = self._build_request_hashes(
            prompt=context_changed_prompt,
            block_size=2,
        )

        self.assertNotEqual(base_hashes[0], changed_hashes[0])
        self.assertEqual(
            self._shared_prefix_block_count(base_hashes, changed_hashes),
            0,
        )

    def test_cache_salt_change_invalidates_entire_block_hash_chain(self) -> None:
        prompt = {
            "user_id": "u-salt",
            "user_history": [
                {"item_id": 11, "timestamp": 101},
                {"item_id": 12, "timestamp": 102},
                {"item_id": 13, "timestamp": 103},
                {"item_id": 14, "timestamp": 104},
            ],
            "candidates": [
                {"item_candidate_id": 21, "item_query_time": 1001},
                {"item_candidate_id": 22, "item_query_time": 1002},
            ],
            "context": {"viewer_id": 7},
        }

        tokens_a, salt_a, hashes_a = self._build_request_hashes(
            prompt=prompt,
            block_size=2,
        )
        tokens_b, salt_b, hashes_b = self._build_request_hashes(
            prompt=prompt,
            block_size=2,
            cache_salt_override="hstu-prefix-v999:manual-override",
        )

        self.assertEqual(tokens_a, tokens_b)
        self.assertNotEqual(salt_a, salt_b)
        self.assertNotEqual(hashes_a[0], hashes_b[0])
        self.assertEqual(self._shared_prefix_block_count(hashes_a, hashes_b), 0)

    def test_kv_cache_manager_hits_shared_hstu_prefix_blocks(self) -> None:
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        block_size = 2
        manager = KVCacheManager(
            self._make_kv_cache_config(block_size, num_blocks=8),
            max_model_len=512,
            enable_caching=True,
            hash_block_size=block_size,
        )
        base_prompt = {
            "user_id": "u-manager",
            "user_history": [
                {"item_id": 11, "timestamp": 101},
                {"item_id": 12, "timestamp": 102},
                {"item_id": 13, "timestamp": 103},
                {"item_id": 14, "timestamp": 104},
            ],
            "candidates": [
                {"item_candidate_id": 21, "item_query_time": 1001},
                {"item_candidate_id": 22, "item_query_time": 1002},
            ],
            "context": {"viewer_id": 7},
        }
        candidate_changed_prompt = {
            **base_prompt,
            "candidates": [
                {"item_candidate_id": 99, "item_query_time": 1001},
                {"item_candidate_id": 98, "item_query_time": 1002},
            ],
        }

        request_a = self._build_request(
            request_id="req-a",
            prompt=base_prompt,
            block_size=block_size,
        )
        request_b = self._build_request(
            request_id="req-b",
            prompt=candidate_changed_prompt,
            block_size=block_size,
        )

        self._prime_prefix_cache(
            manager=manager,
            request=request_a,
        )

        computed_blocks, num_computed_tokens = manager.get_computed_blocks(request_b)

        self.assertEqual(request_a.num_tokens, 8)
        self.assertEqual(request_b.num_tokens, 8)
        self.assertEqual(num_computed_tokens, 6)
        self.assertEqual(computed_blocks.get_block_ids(), ([1, 2, 3],))

    def test_kv_cache_manager_context_change_breaks_prefix_hit(self) -> None:
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        block_size = 2
        manager = KVCacheManager(
            self._make_kv_cache_config(block_size, num_blocks=8),
            max_model_len=512,
            enable_caching=True,
            hash_block_size=block_size,
        )
        base_prompt = {
            "user_id": "u-context-manager",
            "user_history": [
                {"item_id": 11, "timestamp": 101},
                {"item_id": 12, "timestamp": 102},
                {"item_id": 13, "timestamp": 103},
                {"item_id": 14, "timestamp": 104},
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

        request_a = self._build_request(
            request_id="req-context-a",
            prompt=base_prompt,
            block_size=block_size,
        )
        request_b = self._build_request(
            request_id="req-context-b",
            prompt=context_changed_prompt,
            block_size=block_size,
        )

        self._prime_prefix_cache(
            manager=manager,
            request=request_a,
        )

        computed_blocks, num_computed_tokens = manager.get_computed_blocks(request_b)

        self.assertEqual(num_computed_tokens, 0)
        self.assertFalse(computed_blocks.blocks[0])

    def test_kv_cache_manager_cache_salt_change_breaks_prefix_hit(self) -> None:
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        block_size = 2
        manager = KVCacheManager(
            self._make_kv_cache_config(block_size, num_blocks=8),
            max_model_len=512,
            enable_caching=True,
            hash_block_size=block_size,
        )
        prompt = {
            "user_id": "u-salt-manager",
            "user_history": [
                {"item_id": 11, "timestamp": 101},
                {"item_id": 12, "timestamp": 102},
                {"item_id": 13, "timestamp": 103},
                {"item_id": 14, "timestamp": 104},
            ],
            "candidates": [
                {"item_candidate_id": 21, "item_query_time": 1001},
                {"item_candidate_id": 22, "item_query_time": 1002},
            ],
            "context": {"viewer_id": 7},
        }

        request_a = self._build_request(
            request_id="req-salt-a",
            prompt=prompt,
            block_size=block_size,
        )
        request_b = self._build_request(
            request_id="req-salt-b",
            prompt=prompt,
            block_size=block_size,
            cache_salt_override="hstu-prefix-v999:manual-override",
        )

        self._prime_prefix_cache(
            manager=manager,
            request=request_a,
        )

        computed_blocks, num_computed_tokens = manager.get_computed_blocks(request_b)

        self.assertEqual(num_computed_tokens, 0)
        self.assertFalse(computed_blocks.blocks[0])


if __name__ == "__main__":
    unittest.main()
