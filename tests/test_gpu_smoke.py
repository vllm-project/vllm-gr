# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


"""GPU smoke test for the minimal GR runtime path."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import unittest

try:
    from tests.smoke_test_utils import smoke_harness
except ImportError:
    from smoke_test_utils import smoke_harness


class TestGPUSmoke(unittest.TestCase):
    def test_gpu_smoke_runs_grllm_worker_runner_hstu_chain(self) -> None:
        with smoke_harness("cuda") as harness:
            llm = harness.create_llm()
            outputs = llm.recommend(harness.prompt(), use_tqdm=False)
            engine = llm.get_engine()
            worker = engine.executor.worker

            self.assertEqual(worker.__class__.__name__, "GRWorker")
            self.assertEqual(worker.model_runner.__class__.__name__, "GRModelRunner")
            self.assertEqual(worker.model_runner.model.backend, "cuda")
            self.assertEqual(outputs[0].item_ids, [11, 12])
            self.assertTrue(outputs[0].finished)

            llm.shutdown()

            self.assertIsNone(llm.llm_engine)
            self.assertEqual(harness.state["active_platform"], None)
            self.assertIn(("gr_model_runner", "cuda"), harness.state["trace"])
            self.assertIn(("hstu", "cuda"), harness.state["trace"])


if __name__ == "__main__":
    unittest.main()
