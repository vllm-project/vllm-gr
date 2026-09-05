# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from tests.profiling.gr_prefill_diagnostic import compare_attention_layers, tensor_difference


def layers():
    reference = [
        {
            "layer": "layer.0",
            "computed_prefix_tokens": 0,
            "actual_tokens": 4,
            "max_seq_len": 4,
            "causal": True,
            "flash_attn_version": 2,
            "query_tail": [[1, 2]],
            "key_tail": [[3, 4]],
            "value_tail": [[5, 6]],
            "output_tail": [[7, 8]],
        }
    ]
    candidate = copy.deepcopy(reference)
    candidate[0].update(
        computed_prefix_tokens=3,
        actual_tokens=1,
        cached_key_different_elements=[0],
        cached_value_different_elements=[0],
        cached_key_max_abs_difference=[0.0],
        cached_value_max_abs_difference=[0.0],
    )
    return reference, candidate


def test_localizes_pre_attention_difference_without_root_cause_claim():
    reference, candidate = layers()
    candidate[0]["query_tail"][0][1] = 3
    candidate[0]["output_tail"][0][1] = 9
    report = compare_attention_layers(reference, candidate, 1)
    assert report["first_differing_attention_input_layer"] == "layer.0"
    assert report["first_differing_attention_output_layer"] == "layer.0"
    assert report["cached_prefix_elementwise_equal_at_all_captured_layers"]
    assert report["layers"][0]["query_tail"]["max_abs_difference"] == 1


def test_changed_prefix_is_reported_not_hidden_by_identical_tail():
    reference, candidate = layers()
    candidate[0].update(cached_key_different_elements=[1], cached_key_max_abs_difference=[0.5])
    report = compare_attention_layers(reference, candidate, 1)
    assert not report["cached_prefix_elementwise_equal_at_all_captured_layers"]
    assert report["first_differing_attention_input_layer"] is None


@pytest.mark.parametrize(
    "left,right",
    [([[1, 2]], [1, 2]), ([[1], [2, 3]], [[1], [2, 3]]), ([float("nan")], [0]), ([], [])],
)
def test_malformed_snapshot_cannot_look_equal(left, right):
    with pytest.raises(AssertionError):
        tensor_difference(left, right)


def test_missing_layer_and_wrong_prefix_extent_are_rejected():
    reference, candidate = layers()
    with pytest.raises(AssertionError, match="attention layers"):
        compare_attention_layers(reference, candidate, 28)
    candidate[0]["computed_prefix_tokens"] = 2
    with pytest.raises(AssertionError, match="extent"):
        compare_attention_layers(reference, candidate, 1)
