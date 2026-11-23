# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.beam_search.measure_latency import validate_online_response
from benchmarks.beam_search.measure_onerec_recall import (
    RECALL_PREFILL_BUCKETS,
    graph_delta,
    infer_batch,
    recall_gr_config,
    recall_prefill_buckets,
)
from benchmarks.beam_search.recall_diagnostics import (
    effective_attention,
    model_file_manifest,
    prefix_environment,
    token_digest,
)
from benchmarks.open_one_rec.tools.convert_catalog import process_json
from tests.helpers.paths import REPO_ROOT

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
    llm = SimpleNamespace(beam_search=lambda *_args, **_kwargs: [output])
    params = SimpleNamespace(beam_width=33)
    assert len(infer_batch(llm, Tokenizer(), [[99]], params)[0]) == 33
    for suffix in ([1, 2, 3, 4], [1, 2, 4, 4, 5], [1, 2, 3, 4, 3]):
        output.sequences[-1].tokens = [99, *suffix]
        with pytest.raises(RuntimeError, match="SID"):
            infer_batch(llm, Tokenizer(), [[99]], params)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "kind", ["missing", "directory", "absolute", "relative", "empty", "unset"]
)
def test_recall_shell_checks_baseline_before_data(tmp_path: Path, kind: str) -> None:
    bash = os.environ.get("TEST_BASH") or shutil.which("bash")
    if bash is None:
        raise pytest.skip.Exception("Bash is required to test the Recall launcher")

    data = tmp_path / "missing-data"
    output = tmp_path / "output"
    baseline = tmp_path / "baseline.json"
    env = {
        **os.environ,
        "V1_RECALL_DATASET_PATH": data.as_posix(),
        "V1_RECALL_OUTPUT_DIR": output.as_posix(),
    }
    if kind == "absolute":
        baseline.write_text("{}", encoding="utf-8")
    elif kind == "directory":
        baseline.mkdir()
    if kind == "unset":
        env.pop("V1_RECALL_BASELINE_FILE", None)
    else:
        env["V1_RECALL_BASELINE_FILE"] = {
            "relative": "benchmarks/beam_search/recall_baseline_batch4.json",
            "empty": "",
        }.get(kind, baseline.as_posix())

    completed = subprocess.run(
        [bash, (REPO_ROOT / "tests/system/run_beam_v1_recall.sh").as_posix()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 1, completed.stderr
    if kind in ("missing", "directory"):
        assert "V1_RECALL_BASELINE_FILE" in completed.stderr
        assert baseline.as_posix() in completed.stderr
        assert "Missing OneRec accuracy input" not in completed.stderr
    else:
        assert f"Missing OneRec accuracy input: {data.as_posix()}/sid2pid.json" in completed.stderr
    assert not output.exists()


@pytest.mark.l1  # type: ignore[misc]
def test_recall_diagnostics_record_scores_without_changing_predictions() -> None:
    output = SimpleNamespace(
        sequences=[
            SimpleNamespace(tokens=[99, *range(1, 6)], cum_logprob=-0.25),
            SimpleNamespace(tokens=[99, *range(1, 6)], cum_logprob=-0.75),
        ]
    )
    llm = SimpleNamespace(beam_search=lambda *_args, **_kwargs: [output])
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
    tmp_path: Path,
) -> None:
    from vllm_gr.config.models import DeploymentOptions

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    before = model_file_manifest(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights changed")
    after = model_file_manifest(tmp_path)
    assert before["config.json"] == after["config.json"]
    assert before["model.safetensors"]["sha256"] != after["model.safetensors"]["sha256"]
    calibration = tmp_path / "prefix.yaml"
    settings = DeploymentOptions(prefix_calibration_path=str(calibration))
    assert prefix_environment(settings)["prefix_calibration_file_sha256_at_observation"] is None
    calibration.write_text("calibration", encoding="utf-8")
    assert prefix_environment(settings)["prefix_calibration_file_sha256_at_observation"] is not None


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


@pytest.mark.l1  # type: ignore[misc]
def test_recall_gr_config_preserves_calibrated_b4_capture(tmp_path: Path) -> None:
    from vllm_gr.config import gr_config_from_mapping
    from vllm_gr.v1.worker.prefill_capture_plan import plan_prefill_capture
    from vllm_gr.v1.worker.prefill_graph_common import (
        get_default_buckets,
        get_prefill_graph_capture_num_reqs,
    )

    baseline_path = REPO_ROOT / "benchmarks/beam_search/recall_baseline_batch4.json"
    profile = json.loads(baseline_path.read_text(encoding="utf-8"))["graph_profile"]
    assert ",".join(map(str, RECALL_PREFILL_BUCKETS)) == profile["prefill_bucket_grid"]
    token_budget = 22808
    buckets = recall_prefill_buckets(token_budget)
    assert buckets[-2:] == [22528, 23040]
    artifact = tmp_path / "catalog.safetensors"
    options = recall_gr_config(
        beam_width=128,
        batch_size=4,
        buckets=buckets,
        artifact=artifact,
        artifact_digest="sha256:" + "a" * 64,
        tokenizer_digest="sha256:" + "b" * 64,
    )
    config = gr_config_from_mapping({"gr": options, "attention-config": {"backend": "CUSTOM"}})
    assert config.deployment.candidate_width == config.deployment.beam_width == (128, 128, 128)
    assert str(config.deployment.prefill_kv_bound) == profile["prefill_kv_bound"]
    assert config.constraint_table is not None
    assert config.constraint_table.path == str(artifact)
    assert config.constraint_table.max_top_k == 128
    host = SimpleNamespace(additional_config={"vllm_gr": {"gr": options}})
    captures = plan_prefill_capture(
        get_default_buckets(host),
        get_prefill_graph_capture_num_reqs(host),
        max_num_seqs=128,
        max_num_batched_tokens=token_budget,
        max_model_len=8192,
        total_token_buckets=True,
    )
    assert captures[token_budget] == (4,)
    assert set(captures.values()) == {(4,)}


pytestmark = pytest.mark.cpu_test
