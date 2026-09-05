# SPDX-License-Identifier: Apache-2.0
"""Localize Prefill differences in diagnostic captures, not a parity verdict.

Run as ``python -m tests.profiling.gr_prefill_diagnostic CAPTURE --layers 28``.
The capture must contain one full-Prefill request followed by a cache hit, with
per-layer device snapshots and pre-write cached-Prefix comparisons. This tool
does not validate model mathematics or treat its output as performance data.
"""

import argparse
import json
import math

from tests.profiling.gr_candidate_oracle import (
    compare_prefill_records,
    decode_record,
    require_validation_enabled,
)


def _flatten(value):
    if isinstance(value, list):
        return [element for item in value for element in _flatten(item)]
    assert type(value) in (int, float) and math.isfinite(value), "nonfinite/invalid snapshot"
    return [value]


def tensor_difference(reference, candidate):
    def shape(value):
        if not isinstance(value, list):
            return ()
        assert value, "empty snapshot"
        children = [shape(item) for item in value]
        assert all(child == children[0] for child in children), "ragged snapshot"
        return (len(value),) + children[0]

    assert shape(reference) == shape(candidate), "snapshot shape mismatch"
    left, right = _flatten(reference), _flatten(candidate)
    assert left, "empty snapshot"
    errors = [abs(a - b) for a, b in zip(left, right)]
    return {
        "elements": len(left),
        "different_elements": sum(error != 0 for error in errors),
        "max_abs_difference": max(errors),
    }


def compare_attention_layers(reference, candidate, expected_layers):
    require_validation_enabled()
    assert type(expected_layers) is int and expected_layers > 0
    assert len(reference) == len(candidate) == expected_layers, "missing/extra attention layers"
    names = [layer["layer"] for layer in reference]
    assert len(set(names)) == expected_layers, "duplicate attention layer"
    assert names == [layer["layer"] for layer in candidate], "attention layer order mismatch"
    reports = []
    for left, right in zip(reference, candidate):
        assert left["computed_prefix_tokens"] == 0, "reference is not full Prefill"
        assert right["computed_prefix_tokens"] > 0, "candidate is not a Prefix hit"
        assert left["max_seq_len"] == right["max_seq_len"], "KV length mismatch"
        assert left["actual_tokens"] == left["max_seq_len"], "reference Q/KV mismatch"
        assert right["actual_tokens"] + right["computed_prefix_tokens"] == right["max_seq_len"], (
            "hit Q/KV extent mismatch"
        )
        assert left["causal"] is True and right["causal"] is True, "noncausal Prefill"
        assert left["flash_attn_version"] == right["flash_attn_version"], "FA version mismatch"
        report = {"layer": left["layer"]}
        for field in ("query_tail", "key_tail", "value_tail", "output_tail"):
            report[field] = tensor_difference(left[field], right[field])
        for field in ("key", "value"):
            count = right[f"cached_{field}_different_elements"]
            error = right[f"cached_{field}_max_abs_difference"]
            assert len(count) == len(error) == 1, "invalid Prefix comparison shape"
            assert type(count[0]) is int and count[0] >= 0, "invalid Prefix difference count"
            assert math.isfinite(error[0]) and error[0] >= 0, "invalid Prefix error"
            assert (count[0] == 0) == (error[0] == 0), "inconsistent Prefix comparison"
            report[f"cached_{field}_different_elements"] = count[0]
            report[f"cached_{field}_max_abs_difference"] = error[0]
        reports.append(report)
    first_input = next(
        (
            row["layer"]
            for row in reports
            if any(
                row[field]["different_elements"]
                for field in ("query_tail", "key_tail", "value_tail")
            )
        ),
        None,
    )
    first_output = next(
        (row["layer"] for row in reports if row["output_tail"]["different_elements"]), None
    )
    return {
        "first_differing_attention_input_layer": first_input,
        "first_differing_attention_output_layer": first_output,
        "cached_prefix_elementwise_equal_at_all_captured_layers": all(
            row["cached_key_different_elements"] == row["cached_value_different_elements"] == 0
            for row in reports
        ),
        "layers": reports,
        "scope": "observed QKV/output tails and common-block-table Prefix before hit writes; "
        "not bitwise equality, std-FA metadata validation, or a root-cause verdict",
    }


def main():
    require_validation_enabled()
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--layers", type=int, required=True)
    args = parser.parse_args()
    with open(args.capture) as stream:
        capture = json.load(stream)
    assert capture["status"] == "complete", "capture not complete"
    assert capture["performance_evidence_valid"] is False, "expected diagnostic-only capture"
    records = [decode_record(record) for record in capture["candidate_oracle_diagnostics"]]
    assert len(records) == 2, "expected exactly one full Prefill and one hit"
    output = {
        "capture": args.capture,
        "prefill_candidates": compare_prefill_records(*records),
        "attention": compare_attention_layers(
            records[0]["prefill_attention"],
            records[1]["prefill_attention"],
            args.layers,
        ),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
