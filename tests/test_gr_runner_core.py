# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "vllm_gr" / "worker" / "gr_runner_mixin.py"


def _install_stub_module(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


class _FakeKeyedJaggedTensor:
    def __init__(
        self,
        *,
        keys: list[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        self._keys = list(keys)
        self._values = values
        self._lengths = lengths

    @classmethod
    def from_lengths_sync(
        cls,
        *,
        keys: list[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
    ) -> "_FakeKeyedJaggedTensor":
        return cls(keys=keys, values=values, lengths=lengths)

    def keys(self) -> list[str]:
        return self._keys

    def values(self) -> torch.Tensor:
        return self._values

    def lengths(self) -> torch.Tensor:
        return self._lengths


@dataclass
class _FakeModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    kv_connector_output: object | None = None
    pooler_output: list[torch.Tensor | None] | None = None


class _FakePoolingMetadata:
    def __init__(self, prompt_lens: list[int]) -> None:
        self.prompt_lens = torch.tensor(prompt_lens, dtype=torch.int64)
        self.cursor_device = None

    def build_pooling_cursor(
        self,
        _num_scheduled_tokens_np: np.ndarray,
        _seq_lens_cpu: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.cursor_device = device


class _FakeSeqLensBuffer:
    def __init__(self, cpu_tensor: torch.Tensor) -> None:
        self.cpu = cpu_tensor


def _load_gr_runner_module():
    vllm_pkg = types.ModuleType("vllm")
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    sequence_module = types.ModuleType("vllm.sequence")
    sequence_module.IntermediateTensors = object
    outputs_module = types.ModuleType("vllm.v1.outputs")
    outputs_module.ModelRunnerOutput = _FakeModelRunnerOutput

    torchrec_pkg = types.ModuleType("torchrec")
    torchrec_sparse_pkg = types.ModuleType("torchrec.sparse")
    torchrec_jt_module = types.ModuleType("torchrec.sparse.jagged_tensor")
    torchrec_jt_module.KeyedJaggedTensor = _FakeKeyedJaggedTensor

    _install_stub_module("vllm", vllm_pkg)
    _install_stub_module("vllm.logger", logger_module)
    _install_stub_module("vllm.sequence", sequence_module)
    _install_stub_module("vllm.v1.outputs", outputs_module)
    _install_stub_module("torchrec", torchrec_pkg)
    _install_stub_module("torchrec.sparse", torchrec_sparse_pkg)
    _install_stub_module("torchrec.sparse.jagged_tensor", torchrec_jt_module)

    spec = importlib.util.spec_from_file_location(
        "gr_runner_mixin_core_test_module",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _BaseRunner:
    def _update_states(self, scheduler_output):
        self.seen_scheduler_output = scheduler_output


class TestGRRunnerCore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "vllm",
                "vllm.logger",
                "vllm.sequence",
                "vllm.v1.outputs",
                "torchrec",
                "torchrec.sparse",
                "torchrec.sparse.jagged_tensor",
            )
        }
        cls.runner_module = _load_gr_runner_module()

    @classmethod
    def tearDownClass(cls):
        for name, module in cls._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _make_runner(self, req_ids: list[str] | None = None):
        mixin_cls = self.runner_module.GRFeatureRunnerMixin

        class _Runner(mixin_cls, _BaseRunner):
            pass

        runner = _Runner()
        runner._adapter = self._make_adapter()
        runner.input_batch = SimpleNamespace(req_ids=req_ids or ["r1"])
        return runner

    @staticmethod
    def _make_adapter():
        """Create a lightweight adapter that matches HSTUModelAdapter's API."""
        from vllm_gr.adapters.hstu.kjt_builder import build_hstu_model_kwargs

        class _FakeAdapter:
            def decode_payload(self, payload):
                from vllm_gr.gr_schema import (
                    attach_schema_fingerprint,
                    validate_context_group,
                    validate_feature_group,
                )
                uih_keys = list(payload.uih_keys)
                cand_keys = list(payload.cand_keys)
                ctx_keys = list(payload.ctx_keys)
                uih_keys, cand_keys, ctx_keys, schema_fingerprint = \
                    attach_schema_fingerprint(
                        uih_keys=uih_keys, cand_keys=cand_keys, ctx_keys=ctx_keys,
                    )
                validate_feature_group(
                    "uih", uih_keys,
                    list(payload.uih_values_per_key),
                    list(payload.uih_lengths_per_key),
                )
                validate_feature_group(
                    "cand", cand_keys,
                    list(payload.cand_values_per_key),
                    list(payload.cand_lengths_per_key),
                )
                validate_context_group(
                    ctx_keys, list(payload.ctx_values_per_key),
                )
                full_uih_keys = list(
                    getattr(payload, "full_uih_keys", []) or (uih_keys + ctx_keys)
                )
                return {
                    "uih_keys": uih_keys,
                    "uih_values_per_key": [
                        torch.tensor(v, dtype=torch.int64)
                        for v in payload.uih_values_per_key
                    ],
                    "uih_lengths_per_key": torch.tensor(
                        payload.uih_lengths_per_key, dtype=torch.int64,
                    ),
                    "cand_keys": cand_keys,
                    "cand_values_per_key": [
                        torch.tensor(v, dtype=torch.int64)
                        for v in payload.cand_values_per_key
                    ],
                    "cand_lengths_per_key": torch.tensor(
                        payload.cand_lengths_per_key, dtype=torch.int64,
                    ),
                    "ctx_keys": ctx_keys,
                    "ctx_values_per_key": [
                        torch.tensor(v, dtype=torch.int64)
                        for v in payload.ctx_values_per_key
                    ],
                    "full_uih_keys": full_uih_keys,
                    "schema_fingerprint": schema_fingerprint,
                }

            def build_model_kwargs(self, batch_features, device):
                return build_hstu_model_kwargs(batch_features, device)

            def build_request_lengths(self, features):
                return None

        return _FakeAdapter()

    def test_build_kwargs_batches_features_by_key(self):
        runner = self._make_runner(req_ids=["r1", "r2"])

        payload1 = SimpleNamespace(
            uih_keys=["item_id"],
            uih_values_per_key=[[1, 2]],
            uih_lengths_per_key=[2],
            cand_keys=["item_candidate_id"],
            cand_values_per_key=[[11, 12]],
            cand_lengths_per_key=[2],
            ctx_keys=["country"],
            ctx_values_per_key=[[86]],
        )
        payload2 = SimpleNamespace(
            uih_keys=["item_id"],
            uih_values_per_key=[[3]],
            uih_lengths_per_key=[1],
            cand_keys=["item_candidate_id"],
            cand_values_per_key=[[13]],
            cand_lengths_per_key=[1],
            ctx_keys=["country"],
            ctx_values_per_key=[[99]],
        )

        runner._gr_features_cache["r1"] = runner._adapter.decode_payload(payload1)
        runner._gr_features_cache["r2"] = runner._adapter.decode_payload(payload2)

        model_kwargs = runner._build_gr_model_kwargs()

        self.assertEqual(set(model_kwargs), {"uih_features", "candidates_features"})
        self.assertEqual(
            model_kwargs["uih_features"].keys(),
            ["item_id", "country"],
        )
        self.assertEqual(
            model_kwargs["candidates_features"].keys(),
            ["item_candidate_id"],
        )
        torch.testing.assert_close(
            model_kwargs["uih_features"].values(),
            torch.tensor([1, 2, 3, 86, 99], dtype=torch.int64),
        )
        torch.testing.assert_close(
            model_kwargs["uih_features"].lengths(),
            torch.tensor([2, 1, 1, 1], dtype=torch.int64),
        )
        torch.testing.assert_close(
            model_kwargs["candidates_features"].values(),
            torch.tensor([11, 12, 13], dtype=torch.int64),
        )
        torch.testing.assert_close(
            model_kwargs["candidates_features"].lengths(),
            torch.tensor([2, 1], dtype=torch.int64),
        )

    def test_build_kwargs_fails_fast_for_decode_errors_and_mixed_schema(self):
        with self.subTest("decode_errors"):
            runner = self._make_runner()
            runner._gr_feature_errors["r1"] = "bad payload"

            with self.assertRaisesRegex(ValueError, "decode failed"):
                runner._build_gr_model_kwargs()

        with self.subTest("mixed_schema"):
            runner = self._make_runner(req_ids=["r1", "r2"])
            payload1 = SimpleNamespace(
                uih_keys=["item_id"],
                uih_values_per_key=[[1, 2]],
                uih_lengths_per_key=[2],
                cand_keys=["item_candidate_id"],
                cand_values_per_key=[[11, 12]],
                cand_lengths_per_key=[2],
                ctx_keys=["country"],
                ctx_values_per_key=[[86]],
            )
            payload2 = SimpleNamespace(
                uih_keys=["item_id"],
                uih_values_per_key=[[3, 4]],
                uih_lengths_per_key=[2],
                cand_keys=["item_candidate_id"],
                cand_values_per_key=[[13, 14]],
                cand_lengths_per_key=[2],
                ctx_keys=["country", "gender"],
                ctx_values_per_key=[[86], [1]],
            )

            runner._gr_features_cache["r1"] = runner._adapter.decode_payload(payload1)
            runner._gr_features_cache["r2"] = runner._adapter.decode_payload(payload2)

            with self.assertRaisesRegex(ValueError, "Mixed GR batch schema mismatch"):
                runner._build_gr_model_kwargs()

    def test_build_kwargs_handles_empty_history_and_candidates(self):
        runner = self._make_runner(req_ids=["r1"])

        payload = SimpleNamespace(
            uih_keys=["item_id"],
            uih_values_per_key=[[]],
            uih_lengths_per_key=[0],
            cand_keys=["item_candidate_id"],
            cand_values_per_key=[[]],
            cand_lengths_per_key=[0],
            ctx_keys=[],
            ctx_values_per_key=[],
        )

        runner._gr_features_cache["r1"] = runner._adapter.decode_payload(payload)

        model_kwargs = runner._build_gr_model_kwargs()

        self.assertEqual(model_kwargs["uih_features"].keys(), ["item_id"])
        self.assertEqual(
            model_kwargs["candidates_features"].keys(),
            ["item_candidate_id"],
        )
        self.assertEqual(model_kwargs["uih_features"].values().numel(), 0)
        self.assertEqual(model_kwargs["candidates_features"].values().numel(), 0)
        torch.testing.assert_close(
            model_kwargs["uih_features"].lengths(),
            torch.tensor([0], dtype=torch.int64),
        )
        torch.testing.assert_close(
            model_kwargs["candidates_features"].lengths(),
            torch.tensor([0], dtype=torch.int64),
        )

    def test_build_kwargs_preserves_full_uih_schema_order(self):
        runner = self._make_runner(req_ids=["r1"])

        payload = SimpleNamespace(
            uih_keys=["uih_post_id", "uih_owner_id"],
            uih_values_per_key=[[10, 20], [100, 200]],
            uih_lengths_per_key=[2, 2],
            cand_keys=["item_post_id"],
            cand_values_per_key=[[50, 60]],
            cand_lengths_per_key=[2],
            ctx_keys=["viewer_id", "dummy_contexual"],
            ctx_values_per_key=[[7], [1]],
            full_uih_keys=[
                "viewer_id",
                "uih_post_id",
                "uih_owner_id",
                "dummy_contexual",
            ],
        )

        runner._gr_features_cache["r1"] = runner._adapter.decode_payload(payload)

        model_kwargs = runner._build_gr_model_kwargs()

        self.assertEqual(
            model_kwargs["uih_features"].keys(),
            ["viewer_id", "uih_post_id", "uih_owner_id", "dummy_contexual"],
        )
        torch.testing.assert_close(
            model_kwargs["uih_features"].values(),
            torch.tensor([7, 10, 20, 100, 200, 1], dtype=torch.int64),
        )
        torch.testing.assert_close(
            model_kwargs["uih_features"].lengths(),
            torch.tensor([1, 2, 2, 1], dtype=torch.int64),
        )

    def test_pool_accepts_per_request_hidden_state_list(self):
        mixin_cls = self.runner_module.GRFeatureRunnerMixin

        class _Runner(mixin_cls):
            def __init__(self):
                self.input_batch = SimpleNamespace(
                    num_reqs=2,
                    pooling_params=[object(), object()],
                    req_ids=["req-1", "req-2"],
                    req_id_to_index={"req-1": 0, "req-2": 1},
                )
                self.seq_lens = _FakeSeqLensBuffer(
                    torch.tensor([2, 3], dtype=torch.int64)
                )
                self._pooling_metadata = _FakePoolingMetadata([2, 3])
                self.sync_called = False

                class _FakeModel:
                    @staticmethod
                    def pooler(hidden_states, pooling_metadata):
                        return hidden_states

                self.model = _FakeModel()

            def _sync_device(self):
                self.sync_called = True

        runner = _Runner()
        runner.input_batch.get_pooling_metadata = lambda: runner._pooling_metadata
        outputs = [
            torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float32),
        ]

        result = runner._pool(
            hidden_states=outputs,
            num_scheduled_tokens=5,
            num_scheduled_tokens_np=np.array([2, 3], dtype=np.int32),
            kv_connector_output=None,
        )

        self.assertEqual(result.req_ids, ["req-1", "req-2"])
        self.assertEqual(len(result.pooler_output), 2)
        self.assertEqual(result.pooler_output[0].device.type, "cpu")
        self.assertEqual(result.pooler_output[1].device.type, "cpu")
        self.assertEqual(runner._pooling_metadata.cursor_device.type, outputs[0].device.type)
        self.assertTrue(runner.sync_called)

    def test_pool_fails_fast_when_output_count_mismatches_request_count(self):
        mixin_cls = self.runner_module.GRFeatureRunnerMixin

        class _Runner(mixin_cls):
            def __init__(self):
                self.input_batch = SimpleNamespace(
                    num_reqs=2,
                    pooling_params=[object(), object()],
                    req_ids=["req-1", "req-2"],
                    req_id_to_index={"req-1": 0, "req-2": 1},
                )
                self.seq_lens = _FakeSeqLensBuffer(
                    torch.tensor([2, 3], dtype=torch.int64)
                )
                self._pooling_metadata = _FakePoolingMetadata([2, 3])

                class _FakeModel:
                    @staticmethod
                    def pooler(hidden_states, pooling_metadata):
                        return hidden_states[:1]

                self.model = _FakeModel()

        runner = _Runner()
        runner.input_batch.get_pooling_metadata = lambda: runner._pooling_metadata
        outputs = [
            torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float32),
        ]

        with self.assertRaisesRegex(ValueError, "output count mismatch"):
            runner._pool(
                hidden_states=outputs,
                num_scheduled_tokens=5,
                num_scheduled_tokens_np=np.array([2, 3], dtype=np.int32),
                kv_connector_output=None,
            )

    def test_update_states_cleans_finished_request_caches(self):
        runner = self._make_runner()
        runner._gr_features_cache["done"] = {"schema_fingerprint": "x"}
        runner._gr_feature_errors["done"] = "bad payload"

        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[],
            finished_req_ids=["done"],
        )

        runner._update_states(scheduler_output)

        self.assertNotIn("done", runner._gr_features_cache)
        self.assertNotIn("done", runner._gr_feature_errors)
        self.assertIs(runner.seen_scheduler_output, scheduler_output)


if __name__ == "__main__":
    unittest.main()
