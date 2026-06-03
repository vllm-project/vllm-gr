# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for GRModelAdapter interface constraints."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import unittest
from types import SimpleNamespace


class TestGRModelAdapterInterface(unittest.TestCase):
    """Verify that GRModelAdapter defines the expected interface."""

    def _make_adapter(self) -> None:
        from vllm_gr.adapters.base import GRModelAdapter

        class _TestAdapter(GRModelAdapter):
            model_type = "test"
            payload_type = dict

            def parse_prompt(self, prompt) -> None:
                from vllm_gr.inputs.base import GRParsedInput

                return GRParsedInput()

            def build_payload(self, parsed) -> None:
                return {}

            def decode_payload(self, payload) -> None:
                return {}

            def build_model_kwargs(self, batch_features, device) -> None:
                return {}

            def process_output(self, raw_output, request_meta) -> None:
                return None

        vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace()))
        gr_config = SimpleNamespace()
        return _TestAdapter(vllm_config, gr_config)

    def test_adapter_has_model_type(self) -> None:
        adapter = self._make_adapter()
        self.assertEqual(adapter.model_type, "test")

    def test_adapter_has_payload_type(self) -> None:
        adapter = self._make_adapter()
        self.assertEqual(adapter.payload_type, dict)

    def test_adapter_stores_configs(self) -> None:
        adapter = self._make_adapter()
        self.assertIsNotNone(adapter._vllm_config)
        self.assertIsNotNone(adapter._hf_config)
        self.assertIsNotNone(adapter._gr_config)

    def test_validate_prompt_default_is_noop(self) -> None:
        adapter = self._make_adapter()
        # Should not raise
        adapter.validate_prompt({"user_id": 1})

    def test_build_request_lengths_default_returns_none(self) -> None:
        adapter = self._make_adapter()
        self.assertIsNone(adapter.build_request_lengths([]))

    def test_build_prompt_token_ids_default_returns_none(self) -> None:
        adapter = self._make_adapter()
        from vllm_gr.inputs.base import GRParsedInput

        parsed = GRParsedInput()
        self.assertIsNone(adapter.build_prompt_token_ids(parsed))

    def test_build_cache_salt_default_returns_none(self) -> None:
        adapter = self._make_adapter()
        from vllm_gr.inputs.base import GRParsedInput

        parsed = GRParsedInput()
        self.assertIsNone(adapter.build_cache_salt(parsed))

    def test_extract_request_meta_default_returns_empty_dict(self) -> None:
        adapter = self._make_adapter()
        self.assertEqual(adapter.extract_request_meta(None), {})

    def test_base_class_raises_not_implemented(self) -> None:
        from vllm_gr.adapters.base import GRModelAdapter

        vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace()))
        gr_config = SimpleNamespace()
        base = GRModelAdapter(vllm_config, gr_config)

        with self.assertRaises(NotImplementedError):
            base.parse_prompt({})
        with self.assertRaises(NotImplementedError):
            base.build_payload({})
        with self.assertRaises(NotImplementedError):
            base.decode_payload(None)
        with self.assertRaises(NotImplementedError):
            base.build_model_kwargs([], None)
        with self.assertRaises(NotImplementedError):
            base.process_output(None, {})


if __name__ == "__main__":
    unittest.main()
