# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for GRRequestOutput — requires the full vLLM runtime.

Phase 2 output types (GRModelOutput, ScoringOutput, etc.) are tested in
test_outputs_phase2.py via isolated subprocesses so they do not pollute
this file's import state.  This file only covers GRRequestOutput, which
genuinely needs vLLM.
"""

import unittest

try:
    from vllm_gr.outputs import (
        GRModelOutput,
        GRRequestOutput,
        RecommendedItem,
    )
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


@unittest.skipUnless(_VLLM_AVAILABLE, "vLLM runtime not available")
class TestGRRequestOutputUnchanged(unittest.TestCase):
    """Verify GRRequestOutput is NOT affected by the new types."""

    def test_gr_request_output_not_gr_model_output(self):
        out = GRRequestOutput(request_id="r1")
        self.assertNotIsInstance(out, GRModelOutput)

    def test_gr_request_output_isinstance_check(self):
        """The isinstance checks in async_gr_engine / gr.py / serving.py
        must continue to work."""
        out = GRRequestOutput(
            request_id="r1",
            user_id="u1",
            recommendations=[RecommendedItem(item_id="a", score=1.0)],
            finished=True,
        )
        self.assertIsInstance(out, GRRequestOutput)
        self.assertTrue(out.finished)


if __name__ == "__main__":
    unittest.main()
