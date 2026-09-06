# SPDX-License-Identifier: Apache-2.0
"""Independent CPU Beam selection/ABI oracle for captured model candidates.

No production Beam code is imported. This validates selection and lineage,
not the model logits, constraint sampler mathematics, or KV contents.
"""

import argparse
import base64
import json
import math
import struct


def require_validation_enabled():
    # Assertions are deliberate diagnostic failures, not optional debug checks.
    # Never emit PASS from an interpreter that has removed them.
    if not __debug__:
        raise RuntimeError("candidate oracle requires assertions; do not run with python -O")


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reshape(flat, shape):
    if len(shape) == 1:
        return flat
    stride = math.prod(shape[1:])
    return [reshape(flat[i * stride : (i + 1) * stride], shape[1:]) for i in range(shape[0])]


def decode_record(record):
    require_validation_enabled()
    payload = base64.b64decode(record["diagnostic_payload_base64"], validate=True)

    def decode(value):
        if isinstance(value, dict) and set(value) == {"tensor"}:
            field = value["tensor"]
            assert field["dtype"] in ("float32", "int32"), "unsupported diagnostic dtype"
            shape = field["shape"]
            assert isinstance(shape, list) and shape, "invalid diagnostic shape"
            assert all(type(size) is int and size > 0 for size in shape), "invalid diagnostic shape"
            assert type(field["words"]) is int and field["words"] == math.prod(shape), (
                "diagnostic shape/word mismatch"
            )
            assert type(field["offset"]) is int and field["offset"] >= 0, (
                "invalid diagnostic offset"
            )
            assert 4 * (field["offset"] + field["words"]) <= len(payload), (
                "diagnostic tensor outside payload"
            )
            fmt = "f" if field["dtype"] == "float32" else "i"
            values = list(
                struct.unpack_from("<" + fmt * field["words"], payload, field["offset"] * 4)
            )
            return reshape(values, field["shape"])
        if isinstance(value, dict):
            return {
                key: decode(item)
                for key, item in value.items()
                if key != "diagnostic_payload_base64"
            }
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value

    return decode(record)


def final_fields(compact):
    require_validation_enabled()
    width, steps = compact["shape"]
    assert type(width) is int and width > 0 and type(steps) is int and steps > 0, (
        "invalid final ABI shape"
    )
    payload = base64.b64decode(compact["payload_base64"], validate=True)
    size = width * steps
    assert len(payload) == 4 * (3 * size + width), "final ABI byte size mismatch"
    sequences = reshape(list(struct.unpack_from("<" + "i" * size, payload, 0)), [width, steps])
    parents = reshape(list(struct.unpack_from("<" + "i" * size, payload, size * 4)), [width, steps])
    logprobs = reshape(
        list(struct.unpack_from("<" + "f" * size, payload, size * 8)), [width, steps]
    )
    scores = list(struct.unpack_from("<" + "f" * width, payload, size * 12))
    return sequences, parents, logprobs, scores


def assert_production(stage, expected):
    production = stage["production"]
    for field in ("tokens", "parents", "scores", "sequence", "active"):
        assert isinstance(production[field], list) and len(production[field]) == 1, (
            "production B dimension must be one",
            field,
        )
    assert production["tokens"][0] == [row["sequence"][-1] for row in expected], (
        "tokens",
        stage["stage"],
    )
    assert production["parents"][0] == [row["parents"][-1] for row in expected], (
        "parents",
        stage["stage"],
    )
    assert production["scores"][0] == [row["score"] for row in expected], ("scores", stage["stage"])
    assert production["sequence"][0] == [row["sequence"] for row in expected], (
        "sequence",
        stage["stage"],
    )
    assert production["active"][0] == [int(math.isfinite(row["score"])) for row in expected], (
        "active",
        stage["stage"],
    )


def candidate_rows(stage):
    sampler = stage["sampler"]
    tokens = sampler["tokens"]
    locals_ = sampler["local_logprobs"]
    mask = sampler["valid_mask"]
    counts = sampler["valid_counts"]
    assert len(tokens) == len(locals_)
    if mask is None:
        # Prefill uses the normalized wire scores; a finite entry is active.
        # Raw Decode scores cannot use this fallback (-1e20 padding is finite).
        assert stage["stage"] == 0, "raw Decode candidates require a valid mask"
        mask = [[math.isfinite(value) for value in row] for row in locals_]
    assert len(mask) == len(tokens)
    if counts is not None:
        assert len(counts) == len(tokens), "valid count row mismatch"
        assert all(type(count) is int and count >= 0 for count in counts), "invalid valid count"
    for row_index, (ids, scores, valid) in enumerate(zip(tokens, locals_, mask)):
        assert len(ids) == len(scores) == len(valid)
        assert all(type(token) is int and token >= 0 for token in ids), "invalid candidate token"
        assert all(type(value) in (int, bool) and value in (0, 1) for value in valid), (
            "invalid candidate mask"
        )
        assert any(valid), "empty candidate row is outside this successful-session diagnostic"
        if counts is not None:
            assert sum(bool(x) for x in valid) == min(int(counts[row_index]), len(ids)), (
                "valid count mismatch"
            )
        valid_scores = [score for score, active in zip(scores, valid) if active]
        assert all(math.isfinite(score) for score in valid_scores)
        assert valid_scores == sorted(valid_scores, reverse=True), (
            "candidate input is not score-sorted"
        )
    return tokens, locals_, mask


def evaluate(record):
    require_validation_enabled()
    width, steps, result_width = record["beam_width"], record["steps"], record["result_width"]
    for name in ("beam_width", "steps", "result_width", "prefix_len"):
        assert type(record[name]) is int and record[name] > 0, f"invalid {name}"
    assert result_width <= width, "result_width exceeds beam_width"
    assert record["final_compact"]["shape"] == [result_width, steps], "final ABI shape mismatch"
    stages, forwards = record["stages"], record["forward"]
    assert len(stages) == len(forwards) == steps, "stage/forward count mismatch"
    assert [stage["stage"] for stage in stages] == list(range(steps))
    assert [forward["step_before"] for forward in forwards] == list(range(steps))
    computed_prefix = forwards[0].get("computed_prefix_tokens", 0)
    assert 0 <= computed_prefix < record["prefix_len"], "invalid computed Prefix extent"
    assert forwards[0]["positions"] == list(range(computed_prefix, record["prefix_len"])), (
        "Prefill position mismatch"
    )
    assert len(forwards[0]["input_ids"]) == record["prefix_len"] - computed_prefix
    tokens, locals_, mask = candidate_rows(stages[0])
    assert len(tokens) == 1 and len(tokens[0]) == width
    # This oracle case requires W active bootstrap survivors. Do not silently
    # claim coverage of underfilled/inactive cases from an unrelated smoke.
    assert all(mask[0]), "underfilled Prefill is outside this diagnostic case"
    expected = [
        {
            "sequence": [token],
            "score": f32(score),
            "parents": [0],
            "local": [f32(score)],
            "transport": [f32(score)],
        }
        for token, score in zip(tokens[0], locals_[0])
    ]
    assert_production(stages[0], expected)
    candidate_count = width
    for index in range(1, steps):
        forward = forwards[index]
        assert forward["input_ids"] == [row["sequence"][-1] for row in expected], (
            "Forward row/token mismatch",
            index,
        )
        assert forward["positions"] == [record["prefix_len"] + index - 1] * width, (
            "Forward positions",
            index,
        )
        tokens, locals_, mask = candidate_rows(stages[index])
        assert len(tokens) == width
        ranked = []
        for parent, (ids, scores, valid) in enumerate(zip(tokens, locals_, mask)):
            assert len(ids) == width
            for candidate, (token, local, active) in enumerate(zip(ids, scores, valid)):
                if active:
                    score = f32(expected[parent]["score"] + local)
                    ranked.append((score, parent * width + candidate, parent, token, local))
        candidate_count += len(ranked)
        assert len(ranked) >= width, "underfilled Decode is outside this diagnostic case"
        ranked.sort(key=lambda row: (-row[0], row[1]))
        selected = ranked[:width]
        # Group top-W by parent while preserving the global selection rank.
        selected.sort(key=lambda row: row[2])
        if index == steps - 1:
            # FinalSelect restores score order; stable ties use grouped rank.
            selected.sort(key=lambda row: -row[0])
            selected = selected[:result_width]
        next_expected = []
        for score, _flat, parent, token, local in selected:
            previous = expected[parent]
            next_expected.append(
                {
                    "sequence": previous["sequence"] + [token],
                    "score": score,
                    "parents": previous["parents"] + [parent],
                    "local": previous["local"] + [f32(local)],
                    "transport": previous["transport"] + [f32(score - previous["score"])],
                }
            )
        expected = next_expected
        assert_production(stages[index], expected)
    sequences, parents, logprobs, scores = final_fields(record["final_compact"])
    assert sequences == [row["sequence"] for row in expected], "final sequence mismatch"
    assert parents == [row["parents"] for row in expected], "final parent lineage mismatch"
    assert scores == [row["score"] for row in expected], "final score mismatch"
    assert logprobs == [row["transport"] for row in expected], "final transported logprob mismatch"
    max_raw_error = max(
        abs(actual - raw)
        for actuals, row in zip(logprobs, expected)
        for actual, raw in zip(actuals, row["local"])
    )
    assert max_raw_error <= 1e-5, ("raw selected logprob error", max_raw_error)
    return {
        "status": "PASS",
        "request_id": record["request_id"],
        "stages": steps,
        "beam_width": width,
        "result_width": result_width,
        "prefill_computed_prefix_tokens": computed_prefix,
        "candidates_enumerated": candidate_count,
        "max_selected_raw_local_logprob_error": max_raw_error,
        "scope": "selection, Forward row-token linkage, final ABI; not model/sampler/KV mathematics",
    }


def compare_prefill_records(reference, candidate):
    """Locate a difference before Decode; not a cache/model correctness verdict.

    Callers must establish the same model/configuration. The CLI compares two
    requests within one capture, and this function verifies their visible
    prompt suffix. Cached KV contents are not available in this diagnostic.
    """
    evaluate(reference)
    evaluate(candidate)
    for field in ("beam_width", "result_width", "steps", "prefix_len"):
        assert reference[field] == candidate[field], f"comparison {field} mismatch"
    first = reference["forward"][0]
    second = candidate["forward"][0]
    assert first.get("computed_prefix_tokens", 0) == 0, "reference needs full Prefill"
    offset = second.get("computed_prefix_tokens", 0)
    assert first["input_ids"][offset:] == second["input_ids"], "comparison prompt suffix mismatch"
    left_ids, left_scores, _ = candidate_rows(reference["stages"][0])
    right_ids, right_scores, _ = candidate_rows(candidate["stages"][0])
    assert len(set(left_ids[0])) == len(left_ids[0]), "duplicate reference Prefill token"
    assert len(set(right_ids[0])) == len(right_ids[0]), "duplicate candidate Prefill token"
    left = dict(zip(left_ids[0], left_scores[0]))
    right = dict(zip(right_ids[0], right_scores[0]))
    shared = sorted(left.keys() & right.keys())
    deltas = [abs(left[token] - right[token]) for token in shared]
    exact = left_ids == right_ids and left_scores == right_scores
    return {
        "observation": "identical_prefill_candidates" if exact else "prefill_candidates_differ",
        "reference_request_id": reference["request_id"],
        "candidate_request_id": candidate["request_id"],
        "candidate_computed_prefix_tokens": offset,
        "shared_token_count": len(shared),
        "shared_tokens_with_score_difference": sum(delta != 0 for delta in deltas),
        "max_shared_token_score_difference": max(deltas, default=None),
        "reference_only_tokens": sorted(left.keys() - right.keys()),
        "candidate_only_tokens": sorted(right.keys() - left.keys()),
        "candidate_token_order_equal": left_ids == right_ids,
        "scope": "Prefill sampler output before persistent Decode; not a root-cause or parity verdict",
    }


def main():
    require_validation_enabled()
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument(
        "--compare-prefill-first-last",
        action="store_true",
        help="compare first/full-Prefill and last request in the same capture",
    )
    args = parser.parse_args()
    with open(args.capture) as stream:
        capture = json.load(stream)
    assert capture["status"] == "complete"
    records = capture["candidate_oracle_diagnostics"]
    assert records, "no model candidate diagnostic evidence"
    decoded = [decode_record(record) for record in records]
    reports = [evaluate(record) for record in decoded]
    output = {"capture": args.capture, "reports": reports}
    if args.compare_prefill_first_last:
        assert len(decoded) >= 2, "comparison needs at least two captured requests"
        output["prefill_comparison"] = compare_prefill_records(decoded[0], decoded[-1])
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
