# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU smoke test for the minimal GR runtime path."""

from __future__ import annotations

import unittest

try:
    from tests.smoke_test_utils import smoke_harness
except ImportError:
    from smoke_test_utils import smoke_harness


class TestNPUSmoke(unittest.TestCase):

    def test_npu_smoke_runs_grllm_worker_runner_hstu_chain(self):
        with smoke_harness("npu") as harness:
            llm = harness.create_llm()
            outputs = llm.recommend(harness.prompt(), use_tqdm=False)
            engine = llm.get_engine()
            worker = engine.executor.worker

            self.assertEqual(worker.__class__.__name__, "GRNPUWorker")
            self.assertEqual(
                worker.model_runner.__class__.__name__,
                "GRNPUModelRunner",
            )
            self.assertEqual(worker.model_runner.model.backend, "npu")
            self.assertEqual(outputs[0].item_ids, [11, 12])
            self.assertTrue(outputs[0].finished)

            llm.shutdown()

            self.assertIsNone(llm.llm_engine)
            self.assertEqual(harness.state["active_platform"], None)
            self.assertIn(("gr_npu_model_runner", "npu"), harness.state["trace"])
            self.assertIn(("hstu", "npu"), harness.state["trace"])


if __name__ == "__main__":
    unittest.main()
