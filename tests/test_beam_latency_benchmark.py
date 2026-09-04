# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from benchmarks.open_one_rec.beam_latency import (
    _compare_mode_results,
    _digest_sequences,
    _make_sampling_params,
    _markdown_summary,
    _percentile,
    _prepare_prompt_ids,
    _stable_digests,
    _summarize,
    _validate_args,
    _validate_beam_output,
    _write_result,
)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "\n":
            return [99]
        return {"short": [1, 2, 3], "long": list(range(10))}[text]


def test_percentile_and_summary_are_deterministic():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert _percentile(values, 50.0) == 3.0
    assert _percentile(values, 90.0) == pytest.approx(4.6)

    summary = _summarize(values)
    assert summary["count"] == 5
    assert summary["mean_ms"] == 3.0
    assert summary["p95_ms"] == pytest.approx(4.8)
    assert summary["std_ms"] == pytest.approx(2**0.5)


def test_prompt_normalization_accounts_for_serving_marker():
    tokenizer = _Tokenizer()
    prompt_ids, info = _prepare_prompt_ids(
        tokenizer,
        "short",
        6,
        begin_token="<|sid_begin|>",
    )
    assert prompt_ids == [99, 99, 1, 2, 3]
    assert info == {
        "source_prompt_tokens": 3,
        "submitted_prompt_tokens": 5,
        "begin_tokens_appended_by_serving": 1,
        "model_input_tokens": 6,
        "adjustment": "left_pad_newline",
    }

    prompt_ids, info = _prepare_prompt_ids(
        tokenizer,
        "long",
        4,
        begin_token="<|sid_begin|>",
    )
    assert prompt_ids == [7, 8, 9]
    assert info["adjustment"] == "left_truncate"


def test_beam_output_validation_and_digest_cover_all_beams():
    sequences = [SimpleNamespace(tokens=[10, 11, 12, 99]) for _ in range(2)]
    output = [SimpleNamespace(sequences=sequences)]
    validation = _validate_beam_output(
        output,
        beam_width=2,
        model_input_tokens=2,
        decode_steps=1,
        end_token_id=99,
    )
    assert validation["beam_count"] == 2
    assert validation["sequence_length"] == 4
    assert validation["output_token_digest"] == _digest_sequences(sequences)

    with pytest.raises(AssertionError, match="does not end"):
        _validate_beam_output(
            [
                SimpleNamespace(
                    sequences=[SimpleNamespace(tokens=[10, 11, 12, 98]) for _ in range(2)]
                )
            ],
            beam_width=2,
            model_input_tokens=2,
            decode_steps=1,
            end_token_id=99,
        )


def test_sampling_params_falls_back_for_older_compatibility_class():
    class LegacyParams:
        def __init__(self, beam_width, max_tokens, temperature, ignore_eos):
            self.beam_width = beam_width
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.ignore_eos = ignore_eos

    params = _make_sampling_params(
        LegacyParams,
        beam_width=2,
        max_tokens=5,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
    )
    assert params.beam_width == 2
    assert not hasattr(params, "begin_token")


def test_compare_results_reports_latency_and_output_parity():
    def make_result(median, digests):
        medians = {
            "prefill_miss": median,
            "prefill_hit": median / 2,
            "decode": median / 4,
            "e2e_miss": median * 2,
            "e2e_hit": median,
            "overhead": median / 4,
        }
        records = [{"output_token_digest": digest} for digest in digests]
        return {
            "contract": {"input_lengths": [1024]},
            "lengths": {
                "1024": {
                    "median_ms": medians,
                    "raw": {"miss": records[:1], "hit": records[1:]},
                }
            },
        }

    comparison = _compare_mode_results(
        make_result(10.0, ["same", "same"]),
        make_result(8.0, ["same", "same"]),
    )
    row = comparison["1024"]
    assert row["delta_graph_minus_eager_ms"]["e2e_miss"] == -4.0
    assert row["graph_improvement_percent"]["e2e_miss"] == 20.0
    assert row["output_digests_identical"] is True
    assert row["output_stability_pass"] is True

    markdown = _markdown_summary(
        {
            "mode": "compare",
            "environment": {
                "device": {"name": "Test GPU", "total_memory_mib": 1234},
                "vllm": "0.22.1",
                "vllm_gr_version": "0.22.1",
                "vllm_gr_commit": "abc123",
                "vllm_gr_dirty": False,
            },
            "contract": {
                "beam_width": 128,
                "max_tokens_including_markers": 5,
                "measured_pairs_per_length": 5,
                "input_lengths": [1024],
            },
            "comparison": comparison,
        }
    )
    assert "Test GPU (1234 MiB)" in markdown
    assert "| 1024 |" in markdown
    assert "20.00%" in markdown

    reordered = _compare_mode_results(
        make_result(10.0, ["first", "second"]),
        make_result(8.0, ["second", "first"]),
    )
    assert reordered["1024"]["output_digests_identical"] is False

    assert _stable_digests([{"output_token_digest": "a"}, {"output_token_digest": "a"}]) == {
        "count": 1,
        "digests": ["a"],
        "stable": True,
    }


def test_argument_validation_rejects_duplicate_lengths():
    args = SimpleNamespace(
        input_lengths=[1024, 1024],
        beam_width=2,
        max_tokens=5,
        warmups=0,
        runs=1,
        gpu_memory_utilization=0.7,
    )
    with pytest.raises(ValueError, match="duplicates"):
        _validate_args(args)


def test_quiet_result_writing_still_persists_json(tmp_path, capsys):
    output = tmp_path / "result.json"
    _write_result({"status": "ok"}, output, print_result=False)

    assert output.read_text(encoding="utf-8") == '{\n  "status": "ok"\n}\n'
    assert capsys.readouterr().out == ""
