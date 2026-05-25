# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from vllm_gr.gr_schema import (
    build_schema_fingerprint,
    extract_context_features,
    extract_features,
    validate_context_group,
    validate_feature_group,
)


class TestGRSchemaContract(unittest.TestCase):

    def test_extract_features_requires_every_configured_key(self):
        with self.assertRaisesRegex(ValueError, "do not match configured schema"):
            extract_features(
                records=[{"item_id": 1}, {"item_id": 2, "rating": 5}],
                feature_names=["item_id", "rating"],
                group_name="uih",
            )

    def test_extract_features_rejects_fully_mismatched_schema(self):
        with self.assertRaisesRegex(ValueError, "item_id missing in records"):
            extract_features(
                records=[{"movie_id": 1}, {"movie_id": 2}],
                feature_names=["item_id"],
                group_name="cand",
            )

    def test_extract_features_preserves_configured_key_order(self):
        keys, values, lengths = extract_features(
            records=[
                {"item_id": 1, "timestamp": 100},
                {"item_id": 2, "timestamp": 200},
            ],
            feature_names=["item_id", "timestamp"],
            group_name="uih",
        )

        self.assertEqual(keys, ["item_id", "timestamp"])
        self.assertEqual(values, [[1, 2], [100, 200]])
        self.assertEqual(lengths, [2, 2])

    def test_extract_features_preserves_schema_for_empty_records(self):
        keys, values, lengths = extract_features(
            records=[],
            feature_names=["item_id", "timestamp"],
            group_name="uih",
        )

        self.assertEqual(keys, ["item_id", "timestamp"])
        self.assertEqual(values, [[], []])
        self.assertEqual(lengths, [0, 0])

    def test_context_features_must_be_scalar_ints(self):
        with self.assertRaisesRegex(ValueError, "convertible to int"):
            extract_context_features({"country": "cn", "bad": object()})

    def test_validate_context_group_rejects_non_scalar_payloads(self):
        with self.assertRaisesRegex(ValueError, "exactly one value"):
            validate_context_group(["country"], [[1, 2]])

    def test_validate_feature_group_rejects_cross_feature_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same number of records"):
            validate_feature_group(
                "uih",
                ["item_id", "timestamp"],
                [[1, 2], [100]],
                [2, 1],
            )

    def test_schema_fingerprint_is_stable(self):
        fingerprint = build_schema_fingerprint(
            uih_keys=["item_id", "timestamp"],
            cand_keys=["item_candidate_id"],
            ctx_keys=["country"],
        )
        self.assertEqual(
            fingerprint,
            "uih:item_id,timestamp|cand:item_candidate_id|ctx:country",
        )


if __name__ == "__main__":
    unittest.main()
