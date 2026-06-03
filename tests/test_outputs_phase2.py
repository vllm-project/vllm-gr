# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for Phase 2 output types (GRModelOutput hierarchy).

These types depend only on dataclasses + torch, not on vLLM.
We run each assertion in an isolated subprocess so that the
``sys.modules`` mock never leaks into the parent process — other
test files (e.g. test_outputs_hierarchy.py) see a pristine import
state regardless of execution order.
"""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import subprocess
import sys
import textwrap
import unittest

# Each entry is (description, inline script).
# The subprocess script mocks ``vllm.outputs`` minimally, imports the
# Phase-2 classes, and asserts expected behaviour.  If any assertion
# fails the subprocess exits non-zero and this test reports failure.
_SUBPROCESS_TESTS = {
    "scoring_default": textwrap.dedent("""\
        import sys
        from unittest.mock import MagicMock, patch
        stub = MagicMock()
        with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
            from vllm_gr.outputs import ScoringOutput
        o = ScoringOutput()
        assert o.recommendations == [], o.recommendations
        assert o.all_scores is None
        assert o.request_meta == {}
    """),
    "scoring_with_values": textwrap.dedent("""\
        import sys
        from unittest.mock import MagicMock, patch
        stub = MagicMock()
        with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
            from vllm_gr.outputs import RecommendedItem, ScoringOutput
        out = ScoringOutput(
            recommendations=[RecommendedItem(item_id="a", score=0.9)],
            request_meta={"user": "u1"},
        )
        assert len(out.recommendations) == 1
        assert abs(out.recommendations[0].score - 0.9) < 1e-9
        assert out.request_meta["user"] == "u1"
    """),
    "scoring_is_gr_model_output": textwrap.dedent("""\
        import sys
        from unittest.mock import MagicMock, patch
        stub = MagicMock()
        with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
            from vllm_gr.outputs import GRModelOutput, ScoringOutput
        assert isinstance(ScoringOutput(), GRModelOutput)
    """),
    "embedding_default": textwrap.dedent("""\
        import sys
        from unittest.mock import MagicMock, patch
        stub = MagicMock()
        with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
            from vllm_gr.outputs import EmbeddingOutput
        o = EmbeddingOutput()
        assert o.embeddings is None
        assert o.item_ids == []
        assert o.similarities == []
        assert o.request_meta == {}
    """),
    "embedding_with_values": textwrap.dedent("""\
        import sys
        from unittest.mock import MagicMock, patch
        stub = MagicMock()
        with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
            from vllm_gr.outputs import EmbeddingOutput
        out = EmbeddingOutput(item_ids=["a", "b"], similarities=[0.9, 0.7])
        assert out.item_ids == ["a", "b"]
        assert abs(out.similarities[0] - 0.9) < 1e-9
    """),
    "embedding_is_gr_model_output": textwrap.dedent("""\
        import sys
        from unittest.mock import MagicMock, patch
        stub = MagicMock()
        with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
            from vllm_gr.outputs import EmbeddingOutput, GRModelOutput
        assert isinstance(EmbeddingOutput(), GRModelOutput)
    """),
    "text_default": textwrap.dedent("""\
        import sys
        from unittest.mock import MagicMock, patch
        stub = MagicMock()
        with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
            from vllm_gr.outputs import TextOutput
        o = TextOutput()
        assert o.text == ""
        assert o.token_ids == []
        assert o.request_meta == {}
    """),
    "text_with_values": textwrap.dedent("""\
        import sys
        from unittest.mock import MagicMock, patch
        stub = MagicMock()
        with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
            from vllm_gr.outputs import TextOutput
        out = TextOutput(text="hello", token_ids=[1, 2, 3])
        assert out.text == "hello"
        assert out.token_ids == [1, 2, 3]
    """),
    "text_is_gr_model_output": textwrap.dedent("""\
        import sys
        from unittest.mock import MagicMock, patch
        stub = MagicMock()
        with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
            from vllm_gr.outputs import GRModelOutput, TextOutput
        assert isinstance(TextOutput(), GRModelOutput)
    """),
    "scoring_independent_defaults": textwrap.dedent("""\
        import sys
        from unittest.mock import MagicMock, patch
        stub = MagicMock()
        with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
            from vllm_gr.outputs import RecommendedItem, ScoringOutput
        o1 = ScoringOutput()
        o2 = ScoringOutput()
        o1.recommendations.append(RecommendedItem(item_id="x", score=1.0))
        assert len(o2.recommendations) == 0, f"shared mutable default: {o2.recommendations}"
    """),
    "embedding_independent_defaults": textwrap.dedent("""\
        import sys
        from unittest.mock import MagicMock, patch
        stub = MagicMock()
        with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
            from vllm_gr.outputs import EmbeddingOutput
        o1 = EmbeddingOutput()
        o2 = EmbeddingOutput()
        o1.similarities.append(0.5)
        assert len(o2.similarities) == 0, f"shared mutable default: {o2.similarities}"
    """),
}


class TestPhase2Outputs(unittest.TestCase):
    def _run_subprocess(self, script: str) -> None:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )

    for _name, _script in _SUBPROCESS_TESTS.items():
        # Create a separate test method for each case so failures are
        # individually identifiable in pytest / unittest output.
        locals()[f"test_{_name}"] = lambda self, s=_script: self._run_subprocess(s)


if __name__ == "__main__":
    unittest.main()
