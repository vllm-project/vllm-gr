# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for Phase 3.S deferred items."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import subprocess
import sys
import textwrap
import unittest


class TestHSTUFeaturesPayloadSerialization(unittest.TestCase):
    """Verify HSTUFeaturesPayload serialization round-trip."""

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

    def test_serialization_round_trip(self) -> None:
        """HSTUFeaturesPayload must serialize/deserialize correctly."""
        self._run_subprocess(
            textwrap.dedent("""\
            import sys
            import msgspec
            from unittest.mock import MagicMock, patch
            stub = MagicMock()
            with patch.dict(sys.modules, {"vllm": stub, "vllm.v1": stub.v1, "vllm.v1.engine": stub.v1.engine}):
                from vllm_gr.adapters.hstu.payload import HSTUFeaturesPayload

                enc = msgspec.msgpack.Encoder()

                original = HSTUFeaturesPayload(
                    uih_keys=["a", "b"],
                    uih_values_per_key=[[1, 2], [3, 4]],
                    uih_lengths_per_key=[2, 2],
                    cand_keys=["c"],
                    cand_values_per_key=[[5]],
                    cand_lengths_per_key=[1],
                    ctx_keys=["d"],
                    ctx_values_per_key=[[6]],
                    full_uih_keys=["d", "a", "b"],
                    user_id="u1",
                    original_candidates=[{"id": 10}],
                )

                encoded = enc.encode(original)

                dec_hstu = msgspec.msgpack.Decoder(HSTUFeaturesPayload)
                decoded = dec_hstu.decode(encoded)
                assert decoded.uih_keys == ["a", "b"]
                assert decoded.user_id == "u1"
                assert decoded.original_candidates == [{"id": 10}]

                encoded2 = enc.encode(decoded)
                assert encoded == encoded2
        """)
        )


class TestOutputsPackageReexport(unittest.TestCase):
    """Verify vllm_gr.outputs package re-exports all public types."""

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

    def test_request_types_importable(self) -> None:
        self._run_subprocess(
            textwrap.dedent("""\
            import sys
            from unittest.mock import MagicMock, patch
            stub = MagicMock()
            with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
                from vllm_gr.outputs import RecommendedItem, GRRequestOutput
                assert RecommendedItem is not None
                assert GRRequestOutput is not None
        """)
        )

    def test_base_types_importable(self) -> None:
        self._run_subprocess(
            textwrap.dedent("""\
            import sys
            from unittest.mock import MagicMock, patch
            stub = MagicMock()
            with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
                from vllm_gr.outputs import GRModelOutput, ScoringOutput, EmbeddingOutput, TextOutput
                assert GRModelOutput is not None
                assert isinstance(ScoringOutput(), GRModelOutput)
                assert isinstance(EmbeddingOutput(), GRModelOutput)
                assert isinstance(TextOutput(), GRModelOutput)
        """)
        )

    def test_submodule_direct_import(self) -> None:
        self._run_subprocess(
            textwrap.dedent("""\
            import sys
            from unittest.mock import MagicMock, patch
            stub = MagicMock()
            with patch.dict(sys.modules, {"vllm": stub, "vllm.outputs": stub.outputs}):
                from vllm_gr.outputs.request import RecommendedItem, GRRequestOutput
                from vllm_gr.outputs.base import GRModelOutput, ScoringOutput
                assert RecommendedItem is not None
                assert ScoringOutput is not None
        """)
        )


if __name__ == "__main__":
    unittest.main()
