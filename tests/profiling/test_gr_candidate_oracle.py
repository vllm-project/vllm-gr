# SPDX-License-Identifier: Apache-2.0
import base64
import copy
import struct
import subprocess
import sys
import unittest

from tests.profiling.gr_candidate_oracle import compare_prefill_records, decode_record, evaluate


def fixture():
    def stage(index, tokens, local, parents, scores, sequence):
        width = len(tokens[0])
        return {
            "stage": index,
            "sampler": {
                "tokens": tokens,
                "local_logprobs": local,
                "valid_mask": [[1] * width for _ in tokens],
                "valid_counts": [width] * len(tokens),
            },
            "production": {
                "tokens": [[row[-1] for row in sequence]],
                "parents": [parents],
                "scores": [scores],
                "sequence": [sequence],
                "active": [[1, 1]],
            },
        }

    sequences = [11, 30, 50, 11, 30, 51]
    parents = [0, 1, 1, 0, 1, 1]
    logprobs = [-0.5, -0.25, -0.25, -0.5, -0.25, -0.5]
    scores = [-1.0, -1.25]
    payload = struct.pack("<12i8f", *sequences, *parents, *logprobs, *scores)
    return {
        "request_id": "synthetic",
        "beam_width": 2,
        "result_width": 2,
        "steps": 3,
        "prefix_len": 2,
        "forward": [
            {"step_before": 0, "input_ids": [1, 2], "positions": [0, 1]},
            {"step_before": 1, "input_ids": [10, 11], "positions": [2, 2]},
            {"step_before": 2, "input_ids": [20, 30], "positions": [3, 3]},
        ],
        "stages": [
            stage(0, [[10, 11]], [[-0.25, -0.5]], [0, 0], [-0.25, -0.5], [[10], [11]]),
            stage(
                1,
                [[20, 21], [30, 31]],
                [[-0.5, -1.0], [-0.25, -2.0]],
                [0, 1],
                [-0.75, -0.75],
                [[10, 20], [11, 30]],
            ),
            stage(
                2,
                [[40, 41], [50, 51]],
                [[-0.75, -1.0], [-0.25, -0.5]],
                [1, 1],
                [-1.0, -1.25],
                [[11, 30, 50], [11, 30, 51]],
            ),
        ],
        "final_compact": {"shape": [2, 3], "payload_base64": base64.b64encode(payload).decode()},
    }


class OracleTests(unittest.TestCase):
    def test_result_width_must_fit_beam_and_match_final_shape(self):
        for width, error in (
            (3, "exceeds beam_width"),
            (1, "ABI shape"),
            (True, "invalid result_width"),
        ):
            with self.subTest(width=width):
                record = fixture()
                record["result_width"] = width
                with self.assertRaisesRegex(AssertionError, error):
                    evaluate(record)

    def test_extra_production_batch_is_not_ignored(self):
        for field in ("tokens", "parents", "scores", "sequence", "active"):
            with self.subTest(field=field):
                record = fixture()
                record["stages"][1]["production"][field].append([999])
                with self.assertRaisesRegex(AssertionError, "B dimension"):
                    evaluate(record)

    def test_nonbinary_mask_and_invalid_count_are_rejected(self):
        record = fixture()
        record["stages"][1]["sampler"]["valid_mask"][0][0] = 2
        with self.assertRaisesRegex(AssertionError, "candidate mask"):
            evaluate(record)
        record = fixture()
        record["stages"][1]["sampler"]["valid_counts"].append(2)
        with self.assertRaisesRegex(AssertionError, "count row"):
            evaluate(record)

    def test_empty_unselected_parent_is_not_accepted_as_success(self):
        record = fixture()
        record["stages"][2]["sampler"]["valid_mask"][0] = [0, 0]
        record["stages"][2]["sampler"]["valid_counts"][0] = 0
        with self.assertRaisesRegex(AssertionError, "empty candidate row"):
            evaluate(record)

    def test_optimized_python_cannot_emit_pass(self):
        for entry in ("evaluate({})", "main()"):
            with self.subTest(entry=entry):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-O",
                        "-c",
                        f"from tests.profiling.gr_candidate_oracle import evaluate, main; {entry}",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires assertions", result.stderr)

    def test_tensor_descriptor_cannot_silently_drop_payload_words(self):
        record = {
            "diagnostic_payload_base64": base64.b64encode(struct.pack("<3i", 1, 2, 999)).decode(),
            "value": {"tensor": {"dtype": "int32", "shape": [1, 2], "words": 3, "offset": 0}},
        }
        with self.assertRaisesRegex(AssertionError, "shape/word mismatch"):
            decode_record(record)
        record["value"]["tensor"]["words"] = 2
        self.assertEqual(decode_record(record)["value"], [[1, 2]])
        record["value"]["tensor"]["offset"] = 2
        with self.assertRaisesRegex(AssertionError, "outside payload"):
            decode_record(record)

    def test_hand_computed_ties_and_parent_grouping(self):
        result = evaluate(fixture())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["candidates_enumerated"], 10)

    def test_production_parent_is_not_oracle_input(self):
        record = fixture()
        record["stages"][1]["production"]["parents"][0][1] = 0
        with self.assertRaisesRegex(AssertionError, "parents"):
            evaluate(record)

    def test_next_actual_input_is_verified_in_order(self):
        record = fixture()
        record["forward"][2]["input_ids"].reverse()
        with self.assertRaisesRegex(AssertionError, "Forward row/token"):
            evaluate(record)

    def test_raw_invalid_padding_needs_mask(self):
        record = fixture()
        record["stages"][1]["sampler"]["valid_mask"] = None
        with self.assertRaisesRegex(AssertionError, "valid mask"):
            evaluate(record)

    def test_final_logprob_is_independently_verified(self):
        record = fixture()
        final = record["final_compact"]
        payload = bytearray(base64.b64decode(final["payload_base64"]))
        struct.pack_into("<f", payload, 12 * 4, -0.25)
        final["payload_base64"] = base64.b64encode(payload).decode()
        with self.assertRaisesRegex(AssertionError, "transported logprob"):
            evaluate(record)

    def test_does_not_mutate_captured_production_data(self):
        record = fixture()
        saved = copy.deepcopy(record)
        evaluate(record)
        self.assertEqual(record, saved)

    def test_cache_hit_uses_actual_native_computed_prefix_extent(self):
        record = fixture()
        record["prefix_len"] = 6
        record["forward"][0]["computed_prefix_tokens"] = 4
        record["forward"][0]["positions"] = [4, 5]
        record["forward"][1]["positions"] = [6, 6]
        record["forward"][2]["positions"] = [7, 7]
        self.assertEqual(evaluate(record)["prefill_computed_prefix_tokens"], 4)

    def test_cache_hit_missing_prefix_extent_is_not_silently_inferred(self):
        record = fixture()
        record["prefix_len"] = 6
        record["forward"][0]["positions"] = [4, 5]
        with self.assertRaisesRegex(AssertionError, "Prefill position"):
            evaluate(record)

    def test_prefill_comparison_checks_visible_prompt_suffix(self):
        reference = fixture()
        candidate = fixture()
        candidate["forward"][0].update(computed_prefix_tokens=1, input_ids=[2], positions=[1])
        report = compare_prefill_records(reference, candidate)
        self.assertEqual(report["observation"], "identical_prefill_candidates")
        self.assertEqual(report["shared_token_count"], 2)
        candidate["forward"][0]["input_ids"] = [99]
        with self.assertRaisesRegex(AssertionError, "prompt suffix"):
            compare_prefill_records(reference, candidate)

    def test_prefill_comparison_reports_pre_decode_token_difference(self):
        reference = fixture()
        candidate = fixture()
        # Change an unselected initial branch consistently. The candidate's
        # selection/lineage is still correct, but its Prefill candidates differ.
        candidate["stages"][0]["sampler"]["tokens"][0][0] = 12
        candidate["stages"][0]["production"]["tokens"][0][0] = 12
        candidate["stages"][0]["production"]["sequence"][0][0][0] = 12
        candidate["stages"][1]["production"]["sequence"][0][0][0] = 12
        candidate["forward"][1]["input_ids"][0] = 12
        report = compare_prefill_records(reference, candidate)
        self.assertEqual(report["observation"], "prefill_candidates_differ")
        self.assertEqual(report["reference_only_tokens"], [10])
        self.assertEqual(report["candidate_only_tokens"], [12])
        self.assertEqual(report["shared_tokens_with_score_difference"], 0)

    def test_prefill_comparison_requires_valid_selection_evidence(self):
        candidate = fixture()
        candidate["stages"][1]["production"]["scores"][0][0] = 0
        with self.assertRaisesRegex(AssertionError, "scores"):
            compare_prefill_records(fixture(), candidate)

    def test_prefill_comparison_requires_full_reference(self):
        reference = fixture()
        reference["forward"][0].update(computed_prefix_tokens=1, input_ids=[2], positions=[1])
        with self.assertRaisesRegex(AssertionError, "full Prefill"):
            compare_prefill_records(reference, fixture())


if __name__ == "__main__":
    unittest.main()
