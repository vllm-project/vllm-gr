# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hardware-golden comparator must fail closed without importing vLLM."""

from copy import deepcopy

import pytest

from tests.system.run_full_vocab_beam_alignment import (
    compare_reports,
    load_prompts,
    native_engine_kwargs,
    native_prompts,
)


def _reports():
    baseline = dict(
        implementation="native",
        prompt_sha256="same",
        beam_width=1,
        steps=3,
        dtype="bfloat16",
        eos_token_id=0,
        outputs=[[dict(token_ids=[8, 0], finish_reason="stop", cumulative_logprob=-1.25)]],
    )
    actual = deepcopy(baseline)
    actual["implementation"] = "gr"
    return baseline, actual


def test_raw_token_alignment():
    baseline, actual = _reports()
    compare_reports(baseline, actual, 1e-4)
    actual["outputs"][0][0]["cumulative_logprob"] += 1e-5
    compare_reports(baseline, actual, 1e-4)


@pytest.mark.parametrize("fault", ["tokens", "score", "stop", "count", "prompt", "dtype", "nan"])
def test_alignment_rejects_mismatch(fault):
    baseline, actual = _reports()
    sequence = actual["outputs"][0][0]
    if fault == "tokens":
        sequence["token_ids"] = [8]  # Removing EOS is not acceptable postprocessing.
    elif fault == "score":
        sequence["cumulative_logprob"] = -0.5
    elif fault == "stop":
        sequence["finish_reason"] = "length"
    elif fault == "count":
        actual["outputs"][0] = []
    elif fault == "prompt":
        actual["prompt_sha256"] = "different"
    elif fault == "dtype":
        actual["dtype"] = "float16"
    else:
        sequence["cumulative_logprob"] = float("nan")
    with pytest.raises(AssertionError):
        compare_reports(baseline, actual, 1e-4)


@pytest.mark.parametrize("document", ["[]", "[[]]", "[[true]]", "[[-1]]", "[1]", "{}"])
def test_prompt_validation(tmp_path, document):
    path = tmp_path / "prompts.json"
    path.write_text(document)
    with pytest.raises(ValueError, match="token-ID"):
        load_prompts(path)


def test_prompt_ids_are_not_modified(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text("[[4, 3, 2], [9, 2]]")
    assert load_prompts(path) == [[4, 3, 2], [9, 2]]


class LegacyEngineArgs:
    def __init__(self, model, dtype, max_logprobs):
        pass


class CurrentEngineArgs:
    def __init__(self, model, dtype, max_logprobs, *, async_scheduling=None):
        pass


@pytest.mark.parametrize("engine_args_type", [LegacyEngineArgs, CurrentEngineArgs])
def test_native_options_follow_engine_args_fields_without_version_detection(engine_args_type):
    common = dict(model="model", dtype="bfloat16", max_logprobs=256)
    expected = dict(common)
    if engine_args_type is CurrentEngineArgs:
        expected["async_scheduling"] = False
    actual = native_engine_kwargs(common, engine_args_type)
    assert actual == expected
    engine_args_type(**actual)  # Both fake native signatures accept the complete result.
    assert common == dict(model="model", dtype="bfloat16", max_logprobs=256)


@pytest.mark.parametrize("engine_args_type", [LegacyEngineArgs, CurrentEngineArgs])
def test_native_options_do_not_silently_drop_unsupported_configuration(engine_args_type):
    with pytest.raises(ValueError, match="does not support.*unknown_option"):
        native_engine_kwargs({"unknown_option": True}, engine_args_type)


def test_native_prompt_schema_is_compatible_with_old_and_current_token_prompts():
    original = [[7, 1], [8, 2]]
    converted = native_prompts(original)
    assert converted == [{"prompt_token_ids": [7, 1]}, {"prompt_token_ids": [8, 2]}]
    converted[0]["prompt_token_ids"].append(99)
    assert original == [[7, 1], [8, 2]]
