# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for RuntimeSelection normalization and serialization."""

from __future__ import annotations

import pytest

from vllm_gr._runtime_errors import (
    GRCompatibilityError,
    GRInconsistencyError,
    GRPreconditionError,
    GRPropagationError,
)
from vllm_gr._runtime_selection import (
    CURRENT_SCHEMA_VERSION,
    GRBeamConfig,
    GRBeamRequestOptions,
    RuntimeIntentInput,
    RuntimeSelection,
    normalize_runtime_selection,
    parse_vllm_gr_mode,
    read_vllm_gr_mode_from_env,
    selection_from_user_mode,
)
from vllm_gr._static_descriptors import (
    CAPABILITY_GR_BEAM,
    DIAG_SOURCE_API,
    DIAG_SOURCE_AUTO,
    DIAG_SOURCE_CLI,
    FAMILY_GR_RECOMMEND,
    FAMILY_LLM,
    FAMILY_NATIVE,
    MODEL_ARCH_HSTU,
    USER_MODE_GR_BEAM,
    USER_MODE_GR_RECOMMEND,
    USER_MODE_NATIVE,
)


class TestUserModeMapping:
    def test_native_mode(self) -> None:
        selection = selection_from_user_mode(USER_MODE_NATIVE, source="cli")
        assert selection.family == FAMILY_NATIVE
        assert selection.capabilities == frozenset()
        assert selection.model_arch is None

    def test_gr_recommend_mode(self) -> None:
        selection = selection_from_user_mode(USER_MODE_GR_RECOMMEND, source="cli")
        assert selection.family == FAMILY_GR_RECOMMEND
        assert selection.model_arch == MODEL_ARCH_HSTU

    def test_gr_beam_mode(self) -> None:
        selection = selection_from_user_mode(USER_MODE_GR_BEAM, source="cli")
        assert selection.family == FAMILY_LLM
        assert selection.capabilities == frozenset({CAPABILITY_GR_BEAM})


class TestValidation:
    def test_llm_without_capability_fails(self) -> None:
        selection = RuntimeSelection(family=FAMILY_LLM)
        with pytest.raises(GRCompatibilityError, match="at least one capability"):
            selection.validate()

    def test_gr_recommend_with_gr_beam_fails(self) -> None:
        selection = RuntimeSelection(
            family=FAMILY_GR_RECOMMEND,
            model_arch=MODEL_ARCH_HSTU,
            capabilities=frozenset({CAPABILITY_GR_BEAM}),
        )
        with pytest.raises(GRCompatibilityError, match="cannot carry capabilities"):
            selection.validate()

    def test_native_with_gr_beam_fails(self) -> None:
        selection = RuntimeSelection(
            family=FAMILY_NATIVE,
            capabilities=frozenset({CAPABILITY_GR_BEAM}),
        )
        with pytest.raises(GRCompatibilityError, match="cannot carry capabilities"):
            selection.validate()

    def test_unknown_family_fails(self) -> None:
        selection = RuntimeSelection(family="nonexistent")
        with pytest.raises(GRCompatibilityError, match="Unknown family"):
            selection.validate()

    def test_unknown_capability_fails(self) -> None:
        selection = RuntimeSelection(
            family=FAMILY_LLM,
            capabilities=frozenset({"imaginary_cap"}),
        )
        with pytest.raises(GRCompatibilityError, match="Unknown capabilities"):
            selection.validate()

    def test_gr_recommend_without_model_arch_fails(self) -> None:
        selection = RuntimeSelection(family=FAMILY_GR_RECOMMEND)
        with pytest.raises(GRCompatibilityError, match="requires model_arch"):
            selection.validate()

    def test_invalid_model_arch_for_family(self) -> None:
        selection = RuntimeSelection(
            family=FAMILY_GR_RECOMMEND,
            model_arch="bogus_arch",
        )
        with pytest.raises(GRCompatibilityError, match="is invalid for"):
            selection.validate()

    def test_model_arch_on_wrong_family_fails(self) -> None:
        selection = RuntimeSelection(
            family=FAMILY_NATIVE,
            model_arch=MODEL_ARCH_HSTU,
        )
        with pytest.raises(GRCompatibilityError, match="is invalid for"):
            selection.validate()


class TestNormalization:
    def test_default_is_native(self) -> None:
        selection = normalize_runtime_selection(RuntimeIntentInput())
        assert selection.family == FAMILY_NATIVE

    def test_cli_env_same_uses_cli_diagnostics(self) -> None:
        selection = normalize_runtime_selection(
            RuntimeIntentInput(
                cli_mode=USER_MODE_GR_BEAM,
                env_mode=USER_MODE_GR_BEAM,
            )
        )
        assert selection.family == FAMILY_LLM
        assert selection.diagnostics.source == DIAG_SOURCE_CLI

    def test_api_env_same_uses_api_diagnostics(self) -> None:
        selection = normalize_runtime_selection(
            RuntimeIntentInput(
                api_mode=USER_MODE_GR_RECOMMEND,
                env_mode=USER_MODE_GR_RECOMMEND,
            )
        )
        assert selection.family == FAMILY_GR_RECOMMEND
        assert selection.diagnostics.source == DIAG_SOURCE_API

    def test_cli_env_conflict_fails(self) -> None:
        with pytest.raises(GRInconsistencyError, match="cli=.*env="):
            normalize_runtime_selection(
                RuntimeIntentInput(
                    cli_mode=USER_MODE_NATIVE,
                    env_mode=USER_MODE_GR_BEAM,
                )
            )

    def test_api_env_conflict_fails(self) -> None:
        with pytest.raises(GRInconsistencyError, match="api=.*env="):
            normalize_runtime_selection(
                RuntimeIntentInput(
                    api_mode=USER_MODE_NATIVE,
                    env_mode=USER_MODE_GR_RECOMMEND,
                )
            )

    def test_cli_api_conflict_fails(self) -> None:
        with pytest.raises(GRInconsistencyError, match="Conflicting explicit"):
            normalize_runtime_selection(
                RuntimeIntentInput(
                    cli_mode=USER_MODE_NATIVE,
                    api_mode=USER_MODE_GR_BEAM,
                )
            )

    def test_auto_with_explicit_fails(self) -> None:
        with pytest.raises(GRInconsistencyError, match="--auto cannot be combined"):
            normalize_runtime_selection(
                RuntimeIntentInput(cli_mode=USER_MODE_NATIVE, use_auto=True)
            )

    def test_auto_gr_recommend(self) -> None:
        selection = normalize_runtime_selection(
            RuntimeIntentInput(
                use_auto=True,
                auto_inferred_family=FAMILY_GR_RECOMMEND,
            )
        )
        assert selection.family == FAMILY_GR_RECOMMEND
        assert selection.model_arch == MODEL_ARCH_HSTU

    def test_auto_llm_without_capability_fails(self) -> None:
        with pytest.raises(GRCompatibilityError, match="explicit capability"):
            normalize_runtime_selection(
                RuntimeIntentInput(
                    use_auto=True,
                    auto_inferred_family=FAMILY_LLM,
                )
            )

    def test_env_gr_beam(self) -> None:
        selection = normalize_runtime_selection(
            RuntimeIntentInput(env_mode=USER_MODE_GR_BEAM)
        )
        assert selection.family == FAMILY_LLM
        assert CAPABILITY_GR_BEAM in selection.capabilities

    def test_auto_with_env_mode_fails(self) -> None:
        with pytest.raises(GRInconsistencyError, match="--auto cannot be combined"):
            normalize_runtime_selection(
                RuntimeIntentInput(env_mode=USER_MODE_NATIVE, use_auto=True)
            )

    def test_auto_without_inferred_family_fails(self) -> None:
        with pytest.raises(GRInconsistencyError, match="no family could be inferred"):
            normalize_runtime_selection(RuntimeIntentInput(use_auto=True))

    def test_auto_native(self) -> None:
        selection = normalize_runtime_selection(
            RuntimeIntentInput(
                use_auto=True,
                auto_inferred_family=FAMILY_NATIVE,
            )
        )
        assert selection.family == FAMILY_NATIVE
        assert selection.model_arch is None
        assert selection.diagnostics.source == DIAG_SOURCE_AUTO

    def test_cli_mode_alone(self) -> None:
        selection = normalize_runtime_selection(
            RuntimeIntentInput(cli_mode=USER_MODE_NATIVE)
        )
        assert selection.family == FAMILY_NATIVE
        assert selection.diagnostics.source == DIAG_SOURCE_CLI

    def test_cli_api_same_value_prefers_cli(self) -> None:
        selection = normalize_runtime_selection(
            RuntimeIntentInput(
                cli_mode=USER_MODE_GR_RECOMMEND,
                api_mode=USER_MODE_GR_RECOMMEND,
            )
        )
        assert selection.family == FAMILY_GR_RECOMMEND
        assert selection.diagnostics.source == DIAG_SOURCE_CLI


class TestEnvParsing:
    def test_parse_empty_as_unset(self) -> None:
        assert parse_vllm_gr_mode("") is None
        assert parse_vllm_gr_mode("   ") is None
        assert parse_vllm_gr_mode(None) is None

    def test_parse_case_insensitive(self) -> None:
        assert parse_vllm_gr_mode("GR_BEAM") == USER_MODE_GR_BEAM

    def test_unknown_mode_fails(self) -> None:
        with pytest.raises(GRInconsistencyError, match="Unknown VLLM_GR_MODE"):
            parse_vllm_gr_mode("unknown")

    def test_read_from_env_mapping(self) -> None:
        assert (
            read_vllm_gr_mode_from_env({"VLLM_GR_MODE": "gr_recommend"})
            == USER_MODE_GR_RECOMMEND
        )


class TestSerialization:
    def test_runtime_selection_round_trip(self) -> None:
        original = selection_from_user_mode(USER_MODE_GR_BEAM, source="env")
        restored = RuntimeSelection.deserialize(original.serialize())
        assert restored == original

    def test_runtime_selection_schema_mismatch_fails(self) -> None:
        original = selection_from_user_mode(USER_MODE_NATIVE, source="cli")
        payload = original.serialize().replace(
            f'"schema_version": {CURRENT_SCHEMA_VERSION}',
            '"schema_version": 999',
        )
        with pytest.raises(GRPropagationError, match="schema_version"):
            RuntimeSelection.deserialize(payload)

    def test_gr_beam_config_round_trip(self) -> None:
        config = GRBeamConfig(token_catalog_path="/tmp/catalog.json")
        restored = GRBeamConfig.deserialize(config.serialize())
        assert restored == config

    def test_gr_beam_config_optional_catalog(self) -> None:
        config = GRBeamConfig()
        restored = GRBeamConfig.deserialize(config.serialize())
        assert restored.token_catalog_path is None

    def test_deserialize_invalid_json(self) -> None:
        with pytest.raises(GRPropagationError, match="JSON is invalid"):
            RuntimeSelection.deserialize("not json")

    def test_deserialize_non_object_json_array(self) -> None:
        with pytest.raises(GRPropagationError, match="must be a JSON object"):
            RuntimeSelection.deserialize("[]")

    def test_deserialize_non_object_json_string(self) -> None:
        with pytest.raises(GRPropagationError, match="must be a JSON object"):
            GRBeamConfig.deserialize('"x"')

    def test_deserialize_missing_family(self) -> None:
        payload = '{"schema_version": 1, "model_arch": null, "capabilities": [], "diagnostics": {"source": "cli"}}'
        with pytest.raises(GRPropagationError, match="missing string 'family'"):
            RuntimeSelection.deserialize(payload)

    def test_deserialize_non_string_capabilities(self) -> None:
        payload = '{"schema_version": 1, "family": "native", "model_arch": null, "capabilities": [123], "diagnostics": {"source": "cli"}}'
        with pytest.raises(GRPropagationError, match="must be a string list"):
            RuntimeSelection.deserialize(payload)

    def test_gr_beam_config_schema_mismatch(self) -> None:
        config = GRBeamConfig(token_catalog_path="/tmp/x")
        payload = config.serialize().replace(
            f'"schema_version": {CURRENT_SCHEMA_VERSION}',
            '"schema_version": 999',
        )
        with pytest.raises(GRPropagationError, match="schema_version"):
            GRBeamConfig.deserialize(payload)

    def test_gr_beam_config_blank_catalog_path_fails(self) -> None:
        config = GRBeamConfig(token_catalog_path="   ")
        with pytest.raises(GRPreconditionError, match="cannot be empty"):
            config.validate()


class TestBeamRequestOptions:
    def test_invalid_beam_width(self) -> None:
        with pytest.raises(Exception, match="beam_width"):
            GRBeamRequestOptions(beam_width=0).validate()
