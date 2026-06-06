# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Core tests for engine args and GR request data types."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import importlib
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_MODEL_DIR = os.path.join(ROOT, "examples", "configs", "movielens-1m")

sys.path.insert(0, ROOT)


def _has_vllm_engine_runtime() -> bool:
    required_modules = [
        "vllm.engine.arg_utils",
        "vllm.platforms",
        "vllm.outputs",
        "vllm.pooling_params",
        "vllm.v1.request",
        "vllm.v1.engine.input_processor",
        "vllm.v1.engine.output_processor",
        "vllm.transformers_utils.model_arch_config_convertor",
        "vllm.entrypoints.llm",
    ]
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            return False
    return True


_HAS_VLLM_RUNTIME = _has_vllm_engine_runtime()
_SKIP_REASON = "vLLM engine/runtime submodules are unavailable in this environment"


# ── Engine Args ───────────────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestGREngineArgs(unittest.TestCase):
    def test_default_values(self) -> None:
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(model=DEMO_MODEL_DIR, allow_missing_weights=True)
        self.assertEqual(args.model, DEMO_MODEL_DIR)
        self.assertEqual(args.max_seq_len, 512)
        self.assertEqual(args.max_candidates, 1000)

    def test_custom_values(self) -> None:
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(
            model=DEMO_MODEL_DIR,
            max_seq_len=256,
            max_candidates=500,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.8,
            allow_missing_weights=True,
        )
        self.assertEqual(args.max_seq_len, 256)
        self.assertEqual(args.max_candidates, 500)

    def test_worker_cls_injection(self) -> None:
        from vllm_gr.engine.arg_utils import GREngineArgs

        with mock.patch.object(
            GREngineArgs,
            "_is_ascend_platform",
            return_value=False,
        ):
            args = GREngineArgs(
                model=DEMO_MODEL_DIR,
                model_family="hstu",
                allow_missing_weights=True,
            )
        self.assertEqual(args.worker_cls, "vllm_gr.worker.gr_worker.GRWorker")

    def test_hstu_defaults_do_not_force_disable_prefix_caching(self) -> None:
        from vllm_gr.engine.arg_utils import GREngineArgs

        with mock.patch.object(
            GREngineArgs,
            "_is_ascend_platform",
            return_value=False,
        ):
            args = GREngineArgs(
                model="test-model",
                model_family="hstu",
                enable_prefix_caching=True,
            )

        self.assertTrue(args.enable_prefix_caching)

    def test_hstu_defaults_disable_hybrid_kv_cache_manager(self) -> None:
        from vllm_gr.engine.arg_utils import GREngineArgs

        with mock.patch.object(
            GREngineArgs,
            "_is_ascend_platform",
            return_value=True,
        ):
            args = GREngineArgs(
                model="test-model",
                model_family="hstu",
            )

        self.assertTrue(args.disable_hybrid_kv_cache_manager)

    def test_unknown_model_family_does_not_apply_hstu_defaults(self) -> None:
        from vllm_gr.engine.arg_utils import GREngineArgs

        with mock.patch.object(
            GREngineArgs,
            "_read_model_type_from_local_config",
            return_value=None,
        ):
            args = GREngineArgs(
                model=DEMO_MODEL_DIR,
                allow_missing_weights=True,
            )
        self.assertNotEqual(
            args.worker_cls,
            "vllm_gr.worker.gr_worker.GRWorker",
        )
        self.assertFalse(args.trust_remote_code)

    def test_feature_names_not_on_engine_args(self) -> None:
        from vllm_gr.engine.arg_utils import GREngineArgs

        fields = {f.name for f in GREngineArgs.__dataclass_fields__.values()}
        self.assertNotIn("uih_feature_names", fields)
        self.assertNotIn("candidate_feature_names", fields)

    def test_dataset_overrides_use_hf_overrides(self) -> None:
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(
            model=DEMO_MODEL_DIR,
            dataset="movielens-1m",
            allow_missing_weights=True,
        )
        self.assertIsInstance(args.hf_overrides, dict)
        self.assertEqual(
            args.hf_overrides["uih_post_id_feature_name"],
            "movie_id",
        )

    def test_dataset_overrides_preserve_user_hf_overrides(self) -> None:
        from vllm_gr.engine.arg_utils import GREngineArgs

        args = GREngineArgs(
            model=DEMO_MODEL_DIR,
            dataset="movielens-1m",
            hf_overrides={"hstu_attn_num_layers": 99},
            allow_missing_weights=True,
        )
        self.assertEqual(args.hf_overrides["hstu_attn_num_layers"], 99)
        self.assertEqual(
            args.hf_overrides["uih_post_id_feature_name"],
            "movie_id",
        )


# ── GRConfig ──────────────────────────────────────────────────────


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestGRConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        from vllm_gr.config.gr_config import GRConfig

        cfg = GRConfig()
        self.assertEqual(cfg.max_seq_len, 512)
        self.assertEqual(cfg.max_candidates, 1000)

    def test_no_feature_name_fields(self) -> None:
        from vllm_gr.config.gr_config import GRConfig

        fields = {f.name for f in GRConfig.__dataclass_fields__.values()}
        self.assertNotIn("uih_feature_names", fields)
        self.assertNotIn("candidate_feature_names", fields)


class TestGRPrompt(unittest.TestCase):
    def test_basic_gr_prompt(self) -> None:
        from vllm_gr.inputs.data import GRPrompt, is_gr_prompt

        prompt: GRPrompt = {
            "user_id": 123,
            "user_history": [{"item_id": 1}, {"item_id": 2}],
            "candidates": [{"item_id": 10}, {"item_id": 20}],
        }
        self.assertTrue(is_gr_prompt(prompt))

    def test_gr_prompt_with_optional_fields(self) -> None:
        from vllm_gr.inputs.data import is_gr_prompt

        prompt = {
            "user_id": "user_abc",
            "user_history": [{"item_id": 1, "timestamp": 1000}],
            "candidates": [{"item_id": 10, "category_id": 1}],
            "context": {"device_type": 0},
        }
        self.assertTrue(is_gr_prompt(prompt))

    def test_is_gr_prompt_accepts_registered_model_type_without_hstu_shape(self) -> None:
        from vllm_gr.inputs.data import is_gr_prompt

        prompt = {
            "user_id": "user_abc",
            "model_type": "hstu",
        }
        self.assertTrue(is_gr_prompt(prompt))

    def test_is_gr_prompt_rejects_unknown_model_type_without_legacy_shape(self) -> None:
        from vllm_gr.inputs.data import is_gr_prompt

        prompt = {
            "user_id": "user_abc",
            "model_type": "nonexistent_family_xyz",
        }
        self.assertFalse(is_gr_prompt(prompt))

    def test_is_gr_prompt_false_cases(self) -> None:
        from vllm_gr.inputs.data import is_gr_prompt

        self.assertFalse(is_gr_prompt("not a dict"))
        self.assertFalse(is_gr_prompt(None))
        self.assertFalse(is_gr_prompt({}))
        self.assertFalse(is_gr_prompt({"user_history": []}))
        self.assertFalse(is_gr_prompt({"candidates": []}))

    def test_is_gr_prompt_user_id_only_accepted(self) -> None:
        """Without model_type, user_id alone is sufficient.


        Field validation (user_history, candidates, etc.) is deferred
        to the adapter's validate_prompt(), keeping is_gr_prompt()
        model-family-agnostic.
        """
        from vllm_gr.inputs.data import is_gr_prompt

        self.assertTrue(is_gr_prompt({"user_id": 1}))
        self.assertTrue(is_gr_prompt({"user_id": 1, "user_history": []}))

    def test_is_gr_prompt_rejects_invalid_model_type(self) -> None:
        """When model_type is present but invalid, reject outright.

        An explicit model_type means the caller wants registry-based
        matching.  A bad value (empty, whitespace, non-string) should
        not silently fall through to the generic path.
        """
        from vllm_gr.inputs.data import is_gr_prompt

        self.assertFalse(is_gr_prompt({"user_id": 1, "model_type": ""}))
        self.assertFalse(is_gr_prompt({"user_id": 1, "model_type": "   "}))
        self.assertFalse(is_gr_prompt({"user_id": 1, "model_type": 123}))

    def test_ensure_registered_model_family_propagates_broken_import(self) -> None:
        """A broken adapter dependency must propagate, not be silently
        swallowed as 'unregistered family'."""
        from unittest import mock

        from vllm_gr.inputs.data import _ensure_registered_model_family

        exc = ModuleNotFoundError("torchrec")
        exc.name = "torchrec"
        with mock.patch("importlib.import_module", side_effect=exc):
            # torchrec is NOT the target module, so this should re-raise
            with self.assertRaises(ModuleNotFoundError):
                _ensure_registered_model_family("broken_dep_adapter")

    def test_ensure_registered_model_family_silences_missing_adapter(self) -> None:
        """When the adapter package itself is missing, no error is raised."""
        from unittest import mock

        from vllm_gr.inputs.data import _ensure_registered_model_family

        target = "vllm_gr.adapters.nonexistent_pkg_xyz"
        exc = ModuleNotFoundError(target)
        exc.name = target
        with mock.patch("importlib.import_module", side_effect=exc):
            # Should NOT raise — the adapter package simply doesn't exist
            _ensure_registered_model_family("nonexistent_pkg_xyz")


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestGRRequest(unittest.TestCase):
    def test_without_features(self) -> None:
        from vllm.pooling_params import PoolingParams

        from vllm_gr.request import GRRequest, is_gr_request

        request = GRRequest(
            request_id="req1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=None,
            pooling_params=PoolingParams(task="embed"),
            eos_token_id=None,
        )
        self.assertIsNone(request.gr_features)
        self.assertFalse(is_gr_request(request))

    def test_with_features(self) -> None:
        from vllm.pooling_params import PoolingParams

        from vllm_gr.adapters.hstu.payload import HSTUFeaturesPayload
        from vllm_gr.engine.gr_features_codec import (
            decode_gr_features_payload,
            encode_gr_features_payload,
        )
        from vllm_gr.request import GRRequest, is_gr_request

        payload = HSTUFeaturesPayload(
            uih_keys=["item_id"],
            uih_values_per_key=[[1, 2, 3]],
            uih_lengths_per_key=[3],
            cand_keys=["item_candidate_id"],
            cand_values_per_key=[[10, 20]],
            cand_lengths_per_key=[2],
            user_id="user_123",
        )
        request = GRRequest(
            gr_features=encode_gr_features_payload(payload),
            request_id="req2",
            prompt_token_ids=[1, 2, 3],
            sampling_params=None,
            pooling_params=PoolingParams(task="embed"),
            eos_token_id=None,
        )
        self.assertTrue(is_gr_request(request))
        decoded = decode_gr_features_payload(request.gr_features, HSTUFeaturesPayload)
        self.assertEqual(decoded.user_id, "user_123")

    def test_is_gr_request_false_for_non_request(self) -> None:
        from vllm_gr.request import is_gr_request

        self.assertFalse(is_gr_request(None))
        self.assertFalse(is_gr_request({}))


@unittest.skipUnless(_HAS_VLLM_RUNTIME, _SKIP_REASON)
class TestGROutputProcessorStats(unittest.TestCase):
    def test_prefix_cache_counters_accumulate_scheduler_stats(self) -> None:
        from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats

        from vllm_gr.engine.gr_output_processor import GROutputProcessor

        processor = GROutputProcessor(
            tokenizer=None,
            log_stats=False,
            stream_interval=1,
        )

        stats_a = SchedulerStats(
            prefix_cache_stats=PrefixCacheStats(
                requests=1,
                queries=50,
                hits=0,
            )
        )
        stats_b = SchedulerStats(
            prefix_cache_stats=PrefixCacheStats(
                requests=1,
                queries=50,
                hits=32,
            )
        )

        processor.update_scheduler_stats(stats_a)
        processor.update_scheduler_stats(stats_b)

        self.assertEqual(
            processor.get_prefix_cache_counters(),
            {
                "requests": 2,
                "queries": 100,
                "hits": 32,
                "connector_requests": 0,
                "connector_queries": 0,
                "connector_hits": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
