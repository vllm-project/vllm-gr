# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for GRConfig pickle round-trip."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import pickle  # noqa: pickle
import unittest

from vllm_gr.config.gr_config import GRConfig


class TestGRConfigPickle(unittest.TestCase):
    def test_pickle_round_trip_default(self) -> None:
        cfg = GRConfig()
        data = pickle.dumps(cfg)
        restored = pickle.loads(data)
        self.assertEqual(restored.max_seq_len, cfg.max_seq_len)
        self.assertEqual(restored.max_candidates, cfg.max_candidates)

    def test_pickle_round_trip_custom(self) -> None:
        cfg = GRConfig(max_seq_len=1024, max_candidates=500)
        data = pickle.dumps(cfg)
        restored = pickle.loads(data)
        self.assertEqual(restored.max_seq_len, 1024)
        self.assertEqual(restored.max_candidates, 500)

    def test_equality_after_pickle(self) -> None:
        cfg = GRConfig(max_seq_len=256, max_candidates=200)
        restored = pickle.loads(pickle.dumps(cfg))
        self.assertEqual(cfg, restored)


if __name__ == "__main__":
    unittest.main()
