# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.beam_search.v1_performance_gate import validate_online_response
from benchmarks.beam_search.v1_recall_diagnostics import (
    effective_attention,
    model_file_manifest,
    prefix_environment,
    token_digest,
)
from benchmarks.beam_search.v1_recall_gate import graph_delta, infer_batch
from benchmarks.open_one_rec.tools.convert_catalog import process_json

VALID = ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>", "<|sid_end|>"]


class Tokenizer:
    def convert_ids_to_tokens(self, tokens: list[int]) -> list[str]:
        return [VALID[token - 1] if 1 <= token <= 5 else "<invalid>" for token in tokens]

    def batch_decode(self, rows: list[list[int]], **_kwargs: Any) -> list[str]:
        return ["".join(self.convert_ids_to_tokens(row)) for row in rows]


def test_online_requires_complete_sid_and_consistent_usage() -> None:
    valid = "".join(VALID)
    body: dict[str, Any] = {
        "choices": [{"text": valid}, {"text": valid}],
        "usage": {"prompt_tokens": 1023, "completion_tokens": 10, "total_tokens": 1033},
    }
    validate_online_response(body, index=0, beam_width=2, prompt_tokens=1023)
    for bad in ("", valid[: -len(VALID[-1])], valid.replace("<s_b_2>", "<s_c_2>")):
        body["choices"][1]["text"] = bad
        with pytest.raises(RuntimeError, match="SID output"):
            validate_online_response(body, index=0, beam_width=2, prompt_tokens=1023)
    body["choices"][1]["text"] = valid
    body["usage"]["completion_tokens"] = 0
    with pytest.raises(RuntimeError, match="token usage"):
        validate_online_response(body, index=0, beam_width=2, prompt_tokens=1023)


def test_recall_rejects_malformed_or_truncated_beam_after_rank_32() -> None:
    output = SimpleNamespace(
        sequences=[SimpleNamespace(tokens=[99, *range(1, 6)]) for _ in range(33)]
    )
    llm = SimpleNamespace(beam_search_v1=lambda *_args, **_kwargs: [output])
    params = SimpleNamespace(beam_width=33)
    assert len(infer_batch(llm, Tokenizer(), [[99]], params)[0]) == 33
    for suffix in ([1, 2, 3, 4], [1, 2, 4, 4, 5], [1, 2, 3, 4, 3]):
        output.sequences[-1].tokens = [99, *suffix]
        with pytest.raises(RuntimeError, match="SID"):
            infer_batch(llm, Tokenizer(), [[99]], params)


@pytest.mark.l1  # type: ignore[misc]
def test_recall_diagnostics_record_scores_without_changing_predictions() -> None:
    output = SimpleNamespace(
        sequences=[
            SimpleNamespace(tokens=[99, *range(1, 6)], cum_logprob=-0.25),
            SimpleNamespace(tokens=[99, *range(1, 6)], cum_logprob=-0.75),
        ]
    )
    llm = SimpleNamespace(beam_search_v1=lambda *_args, **_kwargs: [output])
    params = SimpleNamespace(beam_width=2)
    original = infer_batch(llm, Tokenizer(), [[99]], params)
    details: list[Any] = []
    assert infer_batch(llm, Tokenizer(), [[99]], params, diagnostics=details) == original
    assert details == [
        [
            {"token_ids": [1, 2, 3, 4, 5], "score": -0.25},
            {"token_ids": [1, 2, 3, 4, 5], "score": -0.75},
        ]
    ]
    assert token_digest([99]) != token_digest([98])
    assert token_digest([99]) == token_digest((99,))


@pytest.mark.l1  # type: ignore[misc]
def test_recall_diagnostics_hash_snapshot_and_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    before = model_file_manifest(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights changed")
    after = model_file_manifest(tmp_path)
    assert before["config.json"] == after["config.json"]
    assert before["model.safetensors"]["sha256"] != after["model.safetensors"]["sha256"]
    calibration = tmp_path / "prefix.yaml"
    monkeypatch.setenv("VLLM_GR_PREFIX_CALIBRATION_PATH", str(calibration))
    assert prefix_environment()["prefix_calibration_file_sha256_at_observation"] is None
    calibration.write_text("calibration", encoding="utf-8")
    assert prefix_environment()["prefix_calibration_file_sha256_at_observation"] is not None


@pytest.mark.l1  # type: ignore[misc]
def test_recall_diagnostics_extract_effective_attention_version() -> None:
    log = """startup
Beam attention: requested prefill=auto, effective prefill FA3, decode FA2
Beam attention: requested prefill=auto, effective prefill FA3, decode FA2
"""
    assert effective_attention(log) == [
        {"requested_prefill": "auto", "prefill_fa": 3, "decode_fa": 2}
    ]
    assert effective_attention("no attention log") == []


@pytest.mark.parametrize("catalog", [{}, {"bad": 1}, {"549755813888": 1}])  # type: ignore[misc]
def test_catalog_rejects_invalid_or_empty_input_without_overwriting_output(
    tmp_path: Path, catalog: dict[str, int]
) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "catalog.json"
    source.write_text(json.dumps(catalog), encoding="utf-8")
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError):
        process_json(str(source), str(output))
    assert output.read_text(encoding="utf-8") == "existing"


def test_recall_graph_gate_requires_every_b4_batch_to_replay() -> None:
    before: list[dict[str, Any]] = [{"gr_prefill_dispatches": {}}]
    after: list[dict[str, Any]] = [
        {
            "runner": "GPURunner",
            "gr_prefill_buckets": [16],
            "gr_prefill_captures": [[16, 4]],
            "gr_prefill_kv_bounds": {"16:4": 8192},
            "gr_prefill_dispatches": {"replay:4": 2},
        }
    ]
    assert graph_delta(before, after, 2)["expected_b4_replays"] == 2
    after[0]["gr_prefill_dispatches"] = {"replay:4": 1, "fallback:4:no_graph": 1}
    with pytest.raises(RuntimeError, match="expected 2 replays"):
        graph_delta(before, after, 2)


pytestmark = pytest.mark.cpu_test
