# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for driver-side Constraint Tables startup wiring (N1).

Covers the platform adapter factory (torch override, CUDA/NPU fail-fast), the
vocab-slicing rejection, the triples-document compilation, and the
compile -> inject -> worker-digest-ack full chain.  These run without a vLLM /
GPU / NPU environment: the platform provider is faked and the only real work is
NumPy/torch compilation.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm_gr.config import GRConfig
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.constraints.startup import (
    compile_constraint_table_resource,
    create_constraint_execution,
    install_constraint_table,
    prepare_constraint_table,
    validate_constraint_table_platform,
)


def _make_tokenizer() -> MagicMock:
    result = MagicMock()
    result.convert_tokens_to_ids.side_effect = lambda values: [int(value) for value in values]
    result.get_vocab.return_value = {str(i): i for i in range(10000)}
    return result


def _make_group_tokenizer() -> SimpleNamespace:
    """Tokenizer with small s_a/s_b/s_c groups (IDs < vocab_size=1000)."""
    added_vocab = {
        "<s_a_0>": 10,
        "<s_a_1>": 11,
        "<s_b_0>": 20,
        "<s_b_1>": 21,
        "<s_b_2>": 22,
        "<s_c_0>": 30,
        "<s_c_1>": 31,
    }
    return SimpleNamespace(get_added_vocab=lambda: dict(added_vocab))


@pytest.fixture  # type: ignore[misc]
def tokenizer() -> MagicMock:
    return _make_tokenizer()


@pytest.fixture  # type: ignore[misc]
def triples_path(tmp_path: Path) -> Path:
    document = compile_relations_json([["1", "2", "3"], ["1", "4", "5"]])
    path = tmp_path / "constraint_triples.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _fake_activate_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_type: str,
    adapter: Any,
) -> None:
    """Route activate_current_backend to a fake provider without importing vLLM."""

    def fake_activate() -> Any:
        return SimpleNamespace(
            device_type=device_type,
            create_constrained_topk_execution=lambda device=None: adapter,
        )

    monkeypatch.setattr(
        "vllm_gr.platforms.loader.activate_current_backend",
        fake_activate,
    )


class TestCreateConstraintExecution:
    def test_torch_override_returns_torch_adapter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_GR_CONSTRAINT_EXECUTION", "torch")

        adapter = create_constraint_execution(device="cpu")

        from vllm_gr.constraints.execution import TorchConstrainedTopKExecution

        assert isinstance(adapter, TorchConstrainedTopKExecution)
        assert adapter.is_available()

    def test_torch_override_rejects_unknown_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_GR_CONSTRAINT_EXECUTION", "cuda")

        with pytest.raises(ValueError, match="VLLM_GR_CONSTRAINT_EXECUTION"):
            create_constraint_execution()

    def test_provider_without_adapter_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_activate_provider(monkeypatch, device_type="cuda", adapter=None)

        with pytest.raises(RuntimeError, match="does not implement Constraint Table"):
            create_constraint_execution()

    def test_unavailable_adapter_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_activate_provider(
            monkeypatch,
            device_type="npu",
            adapter=SimpleNamespace(is_available=lambda: False),
        )

        with pytest.raises(RuntimeError, match="custom operator is unavailable"):
            create_constraint_execution()

    def test_available_adapter_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = SimpleNamespace(is_available=lambda: True)
        _fake_activate_provider(monkeypatch, device_type="npu", adapter=adapter)

        assert create_constraint_execution(device="npu:0") is adapter


class TestValidateConstraintTablePlatform:
    def test_accepts_default_config(self) -> None:
        validate_constraint_table_platform(GRConfig())  # must not raise

    def test_rejects_vocab_offset(self) -> None:
        with pytest.raises(ValueError, match="vocab slicing"):
            validate_constraint_table_platform(GRConfig(vocab_offset=1024))

    def test_rejects_vocab_slice_size(self) -> None:
        with pytest.raises(ValueError, match="vocab slicing"):
            validate_constraint_table_platform(GRConfig(vocab_slice_size=1024))

    def test_cuda_accepts_complete_vocab_slice(self) -> None:
        validate_constraint_table_platform(
            GRConfig(vocab_offset=1000, vocab_slice_size=1024),
            device_type="cuda",
        )

    def test_npu_still_rejects_vocab_slice(self) -> None:
        with pytest.raises(ValueError, match="platform 'npu'"):
            validate_constraint_table_platform(
                GRConfig(vocab_offset=1000, vocab_slice_size=1024),
                device_type="npu",
            )


class TestCompileConstraintTableResource:
    def test_rejects_non_object_document(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text(json.dumps([["1", "2", "3"]]), encoding="utf-8")

        with pytest.raises(ValueError, match="must be a JSON object"):
            compile_constraint_table_resource(str(path), _make_tokenizer(), vocab_size=100)

    @pytest.mark.parametrize("scalar", [None, 42, 3.5, "catalog", True])  # type: ignore[misc]
    def test_rejects_scalar_document_with_value_error(self, tmp_path: Path, scalar: object) -> None:
        """A scalar JSON catalog must raise ValueError, not TypeError.

        Regression: the old message built ``list(document)[:3]`` which raised
        TypeError for null / numbers before the documented ValueError.
        """
        path = tmp_path / "scalar.json"
        path.write_text(json.dumps(scalar), encoding="utf-8")

        with pytest.raises(ValueError, match="must be a JSON object"):
            compile_constraint_table_resource(str(path), _make_tokenizer(), vocab_size=100)

    def test_compiles_and_reports_size_stats(
        self, triples_path: Path, tokenizer: MagicMock
    ) -> None:
        resource = compile_constraint_table_resource(str(triples_path), tokenizer, vocab_size=100)

        assert resource.digest
        assert resource.total_elements > 0
        assert resource.total_bytes > 0
        assert resource.first_token_ids.tolist() == [1]

    def test_bytes_cap_rejects_oversized_resource(
        self,
        monkeypatch: pytest.MonkeyPatch,
        triples_path: Path,
        tokenizer: MagicMock,
    ) -> None:
        # Shrink the cap so a small catalog trips it without allocating 512 MiB.
        monkeypatch.setattr("vllm_gr.constraints.compiler.MAX_RESOURCE_BYTES", 16)

        with pytest.raises(ValueError, match="exceeds"):
            compile_constraint_table_resource(str(triples_path), tokenizer, vocab_size=100)


def _available_adapter() -> Any:
    """A fake adapter that satisfies the startup capability probe."""
    return SimpleNamespace(is_available=lambda: True)


class TestInstallConstraintTable:
    def test_compiles_injects_and_acks_digest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        triples_path: Path,
        tokenizer: MagicMock,
    ) -> None:
        expected = compile_constraint_table_resource(
            str(triples_path), tokenizer, vocab_size=100
        ).digest
        _fake_activate_provider(monkeypatch, device_type="npu", adapter=_available_adapter())
        engine_core = MagicMock()
        engine_core.collective_rpc.return_value = [{"digest": expected, "cached": True}]
        engine = SimpleNamespace(engine_core=engine_core)

        resource = install_constraint_table(
            engine,
            catalog_path=str(triples_path),
            tokenizer=tokenizer,
            vocab_size=100,
        )

        assert resource.digest == expected
        positional, keyword = engine_core.collective_rpc.call_args
        assert positional[0] == "set_gr_constraint_resources"
        assert keyword["args"][0]["digest"] == expected

    def test_fails_when_worker_acks_wrong_digest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        triples_path: Path,
        tokenizer: MagicMock,
    ) -> None:
        _fake_activate_provider(monkeypatch, device_type="npu", adapter=_available_adapter())
        engine_core = MagicMock()
        engine_core.collective_rpc.return_value = [{"digest": "deadbeef"}]
        engine = SimpleNamespace(engine_core=engine_core)

        with pytest.raises(RuntimeError, match="expected"):
            install_constraint_table(
                engine,
                catalog_path=str(triples_path),
                tokenizer=tokenizer,
                vocab_size=100,
            )

    def test_install_fails_fast_when_platform_lacks_adapter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        triples_path: Path,
        tokenizer: MagicMock,
    ) -> None:
        """The startup probe must reject CUDA (adapter=None) through install."""
        _fake_activate_provider(monkeypatch, device_type="cuda", adapter=None)
        engine = SimpleNamespace(engine_core=MagicMock())

        with pytest.raises(RuntimeError, match="does not implement Constraint Table"):
            install_constraint_table(
                engine,
                catalog_path=str(triples_path),
                tokenizer=tokenizer,
                vocab_size=100,
            )
        engine.engine_core.collective_rpc.assert_not_called()


class TestPrepareConstraintTable:
    def test_compiles_without_touching_workers(
        self, monkeypatch: pytest.MonkeyPatch, triples_path: Path
    ) -> None:
        _fake_activate_provider(monkeypatch, device_type="npu", adapter=_available_adapter())

        resource = prepare_constraint_table(
            catalog_path=str(triples_path),
            tokenizer=_make_tokenizer(),
            vocab_size=100,
            gr_config=GRConfig(),
        )

        assert resource.digest

    def test_prepare_fails_fast_when_platform_lacks_adapter(
        self, monkeypatch: pytest.MonkeyPatch, triples_path: Path
    ) -> None:
        """CUDA (adapter=None) must fail at prepare time, not at first step."""
        _fake_activate_provider(monkeypatch, device_type="cuda", adapter=None)

        with pytest.raises(RuntimeError, match="does not implement Constraint Table"):
            prepare_constraint_table(
                catalog_path=str(triples_path),
                tokenizer=_make_tokenizer(),
                vocab_size=100,
                gr_config=GRConfig(),
            )

    def test_prepare_fails_fast_when_adapter_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, triples_path: Path
    ) -> None:
        """An NPU host with a missing custom op must fail at prepare time."""
        _fake_activate_provider(
            monkeypatch,
            device_type="npu",
            adapter=SimpleNamespace(is_available=lambda: False),
        )

        with pytest.raises(RuntimeError, match="custom operator is unavailable"):
            prepare_constraint_table(
                catalog_path=str(triples_path),
                tokenizer=_make_tokenizer(),
                vocab_size=100,
                gr_config=GRConfig(),
            )

    def test_auto_builds_from_tokenizer_groups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """catalog_path=None builds the tables from s_a/s_b/s_c groups (OneRec default)."""
        _fake_activate_provider(monkeypatch, device_type="npu", adapter=_available_adapter())

        resource = prepare_constraint_table(
            catalog_path=None,
            tokenizer=_make_group_tokenizer(),
            vocab_size=1000,
            gr_config=GRConfig(),
        )

        assert resource is not None
        assert resource.vocab_size == 1000
        assert resource.first_token_ids.tolist() == [10, 11]
        assert resource.max_prefix1_degree == 3
        assert resource.max_prefix2_degree == 2
        assert resource.digest

    def test_sentinel_path_selects_tokenizer_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The AUTO_CONSTRAINT_TABLE_PATH sentinel behaves like catalog_path=None."""
        from vllm_gr.config.models import AUTO_CONSTRAINT_TABLE_PATH

        _fake_activate_provider(monkeypatch, device_type="npu", adapter=_available_adapter())

        resource = prepare_constraint_table(
            catalog_path=AUTO_CONSTRAINT_TABLE_PATH,
            tokenizer=_make_group_tokenizer(),
            vocab_size=1000,
            gr_config=GRConfig(),
        )

        assert resource is not None
        assert resource.first_token_ids.tolist() == [10, 11]

    def test_auto_build_fails_without_token_groups(
        self, monkeypatch: pytest.MonkeyPatch, tokenizer: MagicMock
    ) -> None:
        """A tokenizer without s_a/s_b/s_c groups must fail, not fall back."""
        _fake_activate_provider(monkeypatch, device_type="npu", adapter=_available_adapter())
        tokenizer.get_added_vocab.return_value = {}

        with pytest.raises(ValueError, match="requires a tokenizer with"):
            prepare_constraint_table(
                catalog_path=None,
                tokenizer=tokenizer,
                vocab_size=1000,
                gr_config=GRConfig(),
            )
