# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.profiling import profile_onerec_b1_l1024_sid3 as profile


def _args(**overrides) -> argparse.Namespace:
    values = {
        "batch_size": 1,
        "backend": "gpu",
        "execution": "eager",
        "input_len": 1024,
        "beam_width": 128,
        "generation_steps": 3,
        "runs": 10,
        "warmups": 1,
        "slo_ms": None,
        "gpu_memory_utilization": 0.7,
        "dtype": "bfloat16",
        "catalog": Path("catalog.json"),
        "prefix_cache": "miss",
        "async_scheduling": False,
        "persistent": False,
        "seed": 0,
        "base_sha": "b" * 40,
        "pr_324_sha": "c" * 40,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_v1_model_runner_must_be_explicit() -> None:
    profile._require_v1_model_runner({"VLLM_USE_V2_MODEL_RUNNER": "0"})
    with pytest.raises(RuntimeError, match="ModelRunner V1"):
        profile._require_v1_model_runner({})
    with pytest.raises(RuntimeError, match="ModelRunner V1"):
        profile._require_v1_model_runner({"VLLM_USE_V2_MODEL_RUNNER": "1"})


def test_backend_execution_validation() -> None:
    profile._validate_args(_args(backend="npu", execution="cascade"))
    with pytest.raises(ValueError, match="NPU-only"):
        profile._validate_args(_args(backend="gpu", execution="cascade"))
    with pytest.raises(ValueError, match="GPU-only"):
        profile._validate_args(_args(backend="npu", execution="compiled"))
    with pytest.raises(ValueError, match="batch_size=1"):
        profile._validate_args(_args(batch_size=2))
    with pytest.raises(ValueError, match="complete 40-hex"):
        profile._validate_args(_args(base_sha="short"))
    with pytest.raises(ValueError, match="requires async scheduling"):
        profile._validate_args(_args(persistent=True))


def test_source_shas_are_required_cli_arguments() -> None:
    with pytest.raises(SystemExit):
        profile._parse_args([])


@pytest.mark.parametrize(
    ("backend", "execution", "expected_eager", "expected_npu_mode"),
    [
        ("gpu", "eager", True, None),
        ("gpu", "compiled", False, None),
        ("gpu", "graph", False, None),
        ("npu", "eager", True, "xllm"),
        ("npu", "graph", False, "xllm"),
        ("npu", "cascade", True, "cascade"),
    ],
)
def test_engine_kwargs_encode_backend_and_execution(
    backend: str,
    execution: str,
    expected_eager: bool,
    expected_npu_mode: str | None,
) -> None:
    kwargs = profile._engine_kwargs(_args(backend=backend, execution=execution))
    assert kwargs["enforce_eager"] is expected_eager
    assert kwargs["beam_graph_enabled"] is (execution == "graph")
    assert kwargs["enable_prefix_caching"] is True
    assert kwargs["beam_max_width"] == 128
    assert kwargs["beam_max_decode_steps"] == 3
    assert kwargs["disable_log_stats"] is False
    if expected_npu_mode is None:
        assert "additional_config" not in kwargs
    else:
        assert kwargs["additional_config"]["vllm_gr_npu_decode_execution"] == expected_npu_mode


def test_persistent_engine_kwargs_enable_final_only_runtime() -> None:
    kwargs = profile._engine_kwargs(_args(persistent=True, async_scheduling=True))
    assert kwargs["async_scheduling"] is True
    assert kwargs["additional_config"] == {
        "vllm_gr_persistent_request": True,
        "vllm_gr_persistent_runner_state": True,
        "vllm_gr_scheduling_mode": "async",
    }


def test_persistent_npu_kwargs_preserve_decode_mode() -> None:
    kwargs = profile._engine_kwargs(
        _args(
            backend="npu",
            execution="cascade",
            persistent=True,
            async_scheduling=True,
        )
    )
    assert kwargs["additional_config"] == {
        "vllm_gr_npu_decode_execution": "cascade",
        "vllm_gr_persistent_request": True,
        "vllm_gr_persistent_runner_state": True,
        "vllm_gr_scheduling_mode": "async",
    }


def test_disabled_prefix_cache_is_encoded_in_engine_config() -> None:
    kwargs = profile._engine_kwargs(_args(prefix_cache="disabled"))
    assert kwargs["enable_prefix_caching"] is False


def test_summary_keeps_tail_and_goodput() -> None:
    summary = profile._summary([1.0, 2.0, 3.0, 4.0], slo_ms=2.5)
    assert summary["count"] == 4
    assert summary["median_ms"] == 2.5
    assert summary["p99_ms"] == pytest.approx(3.97)
    assert summary["throughput_requests_per_second"] == 400.0
    assert summary["goodput_requests_per_second"] == 200.0
    assert profile._summary([1.0])["goodput_requests_per_second"] is None


def test_digests_and_atomic_json_output(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes(b'{"version":1}\n')
    assert profile._sha256_file(catalog) == (
        "50208d78350a7a160dec59a82df1499b6ca7da33e54c5eb11c97e342118e68bb"
    )
    output = tmp_path / "nested" / "baseline.json"
    profile._write_result(output, {"samples_ms": [1.25, 2.5]})
    assert json.loads(output.read_text(encoding="utf-8")) == {"samples_ms": [1.25, 2.5]}
    assert not output.with_suffix(".json.tmp").exists()


def test_catalog_triples_are_resolved_to_token_ids(tmp_path: Path) -> None:
    catalog = tmp_path / "triples.json"
    catalog.write_text(
        json.dumps(
            {
                "format_version": 1,
                "triples": [
                    {"a": "a1", "b": "b2", "c": "c3"},
                    {"a": "a1", "b": "b2", "c": "c3"},
                    {"a": "a4", "b": "b5", "c": "c6"},
                ],
            }
        ),
        encoding="utf-8",
    )
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: {
            "a1": 1,
            "b2": 2,
            "c3": 3,
            "a4": 4,
            "b5": 5,
            "c6": 6,
        }[token]
    )

    assert profile._catalog_triple_token_ids(catalog, tokenizer) == {
        (1, 2, 3),
        (4, 5, 6),
    }


def test_catalog_triple_validation_rejects_unknown_tokens(tmp_path: Path) -> None:
    catalog = tmp_path / "triples.json"
    catalog.write_text(
        json.dumps({"triples": [{"a": "a", "b": "b", "c": "unknown"}]}),
        encoding="utf-8",
    )
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: -1 if token == "unknown" else 1
    )

    with pytest.raises(profile.EvidenceUnavailableError, match="unknown tokenizer token"):
        profile._catalog_triple_token_ids(catalog, tokenizer)


def test_metric_delta_is_stable_for_missing_keys() -> None:
    assert profile._metric_delta({"queries": 3.0}, {"queries": 8.0, "hits": 2.0}) == {
        "hits": 2.0,
        "queries": 5.0,
    }


def test_git_status_probe_failure_is_not_reported_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def command(argv, cwd=None):
        del cwd
        if tuple(argv) == ("git", "rev-parse", "HEAD"):
            return "a" * 40
        if tuple(argv) == ("git", "branch", "--show-current"):
            return "main"
        return None

    monkeypatch.setattr(profile, "_run_command", command)
    with pytest.raises(profile.EvidenceUnavailableError, match="worktree status"):
        profile._source_metadata(
            tmp_path,
            _args(),
        )


def test_driver_cli_evidence_is_mandatory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile, "_run_command", lambda _argv: None)
    with pytest.raises(profile.EvidenceUnavailableError, match="driver/device evidence"):
        profile._driver_accelerator_info("gpu")


def test_trace_evidence_requires_files_and_stage_markers(tmp_path: Path) -> None:
    with pytest.raises(profile.EvidenceUnavailableError, match="without a trace"):
        profile._validate_trace_evidence([], "gpu", "eager")

    trace = tmp_path / "trace.json"
    trace.write_text('{"name":"prefill_beam_bootstrap"}', encoding="utf-8")
    with pytest.raises(profile.EvidenceUnavailableError, match="decode_result_d2h"):
        profile._validate_trace_evidence([trace], "gpu", "graph")

    trace.write_text(
        '{"events":["prefill_beam_bootstrap","decode_result_d2h"]}',
        encoding="utf-8",
    )
    evidence = profile._validate_trace_evidence([trace], "gpu", "eager")
    assert evidence["observed_stage_markers"] == [
        "decode_result_d2h",
        "prefill_beam_bootstrap",
    ]


@pytest.mark.parametrize("execution", ["eager", "graph", "cascade"])
def test_trace_evidence_uses_npu_stage_markers(tmp_path: Path, execution: str) -> None:
    trace = tmp_path / f"npu-{execution}.json"
    trace.write_text(
        '{"events":["gr_npu_prefill_stage","gr_npu_decode_stage"]}',
        encoding="utf-8",
    )
    evidence = profile._validate_trace_evidence([trace], "npu", execution)
    assert evidence["required_stage_markers"] == [
        "gr_npu_prefill_stage",
        "gr_npu_decode_stage",
    ]


def test_memory_component_breakdown_is_an_explicit_failed_gate() -> None:
    gate = profile._memory_breakdown_gate()
    assert gate["status"] == "FAIL"
    assert set(gate["measurements"]) == {
        "native_kv_bytes",
        "beam_kv_bytes",
        "workspace_bytes",
        "graph_pool_bytes",
        "gr_state_bytes",
    }
    assert set(gate["measurements"].values()) == {None}


def test_worker_trace_coverage_is_an_explicit_failed_gate() -> None:
    gate = profile._trace_coverage_gate()
    assert gate["status"] == "FAIL"
    assert "EngineCore Scheduler/Request/metadata" in gate["limitation"]


class _FlatLogprobs:
    start_indices = [0, 2]
    end_indices = [2, 3]
    token_ids = [7, 3, 9]
    logprobs = [-0.5, -1.25, -0.125]
    ranks = [1, 2, None]
    decoded_tokens = ["seven", None, "nine"]


def _logprob(value: float, rank: int | None, decoded_token: str | None) -> SimpleNamespace:
    return SimpleNamespace(logprob=value, rank=rank, decoded_token=decoded_token)


def test_flat_and_dict_logprobs_have_the_same_canonical_form() -> None:
    ordinary = [
        {
            3: _logprob(-1.25, 2, None),
            7: _logprob(-0.5, 1, "seven"),
        },
        {9: _logprob(-0.125, None, "nine")},
    ]
    assert profile._canonical_logprobs(_FlatLogprobs()) == profile._canonical_logprobs(ordinary)


def test_output_fingerprints_cover_tokens_scores_logprobs_and_abi() -> None:
    sequence = SimpleNamespace(
        tokens=[1, 2, 3],
        cum_logprob=-0.75,
        logprobs=_FlatLogprobs(),
        text="result",
        finish_reason="stop",
        stop_reason=3,
    )
    fingerprints = profile._output_fingerprints([SimpleNamespace(sequences=[sequence])])
    assert set(fingerprints) == {
        "token_digest",
        "score_digest",
        "logprob_digest",
        "full_abi_digest",
        "parent_lineage_available",
        "limitation",
    }
    assert fingerprints["parent_lineage_available"] is False
    assert (
        len(
            {
                fingerprints["token_digest"],
                fingerprints["score_digest"],
                fingerprints["logprob_digest"],
                fingerprints["full_abi_digest"],
            }
        )
        == 4
    )


def _diagnostic(
    *,
    backend: str = "gpu",
    runner_class: str = "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
    dispatches: int = 0,
) -> dict:
    return {
        "backend": backend,
        "runner_class": runner_class,
        "memory": {},
        "beam_graph": {
            "captured_widths": [128],
            "full_dispatches": dispatches,
        },
    }


def test_worker_diagnostics_uses_named_rpc_and_rejects_missing_evidence() -> None:
    engine = SimpleNamespace(
        collective_rpc=lambda method: (
            [_diagnostic()] if method == "get_gr_profile_runtime_stats" else []
        )
    )
    llm = SimpleNamespace(llm_engine=engine)
    assert profile._worker_diagnostics(llm, "gpu") == [_diagnostic()]

    broken = SimpleNamespace(llm_engine=SimpleNamespace(collective_rpc=lambda method: []))
    with pytest.raises(profile.EvidenceUnavailableError, match="no snapshots"):
        profile._worker_diagnostics(broken, "gpu")


def test_runner_v2_detection_uses_complete_class_path() -> None:
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            collective_rpc=lambda method: [
                _diagnostic(runner_class="vllm.v1.worker.gpu_model_runner_v2.ModelRunner")
            ]
        )
    )
    with pytest.raises(RuntimeError, match="required V1 class"):
        profile._worker_diagnostics(llm, "gpu")


def test_v1_engine_config_must_be_effectively_disabled() -> None:
    profile._validate_v1_engine_config(SimpleNamespace(use_v2_model_runner=False))
    with pytest.raises(RuntimeError, match="use_v2_model_runner"):
        profile._validate_v1_engine_config(SimpleNamespace(use_v2_model_runner=True))


def test_graph_evidence_requires_exact_capture_and_dispatch_delta() -> None:
    args = _args(execution="graph", runs=3, profile_dir=Path("trace"))
    profile._validate_graph_evidence(
        [_diagnostic(dispatches=10)], [_diagnostic(dispatches=14)], args
    )
    with pytest.raises(profile.EvidenceUnavailableError, match="dispatch evidence"):
        profile._validate_graph_evidence(
            [_diagnostic(dispatches=10)], [_diagnostic(dispatches=13)], args
        )
    missing = _diagnostic(dispatches=14)
    missing["beam_graph"]["captured_widths"] = [64]
    with pytest.raises(profile.EvidenceUnavailableError, match="was not captured"):
        profile._validate_graph_evidence([_diagnostic(dispatches=10)], [missing], args)


@pytest.mark.parametrize(
    ("mode", "queries", "hits"),
    [
        ("disabled", 0.0, 0.0),
        ("miss", 10.0, 0.0),
        ("miss", 10.0, 3.0),
        ("hit", 10.0, 5.0),
    ],
)
def test_prefix_evidence_accepts_observed_mode(mode: str, queries: float, hits: float) -> None:
    profile._validate_prefix_evidence(
        mode,
        {"vllm:prefix_cache_queries": queries, "vllm:prefix_cache_hits": hits},
    )
    expected_rate = hits / queries if queries else None
    assert (
        profile._prefix_hit_rate(
            {"vllm:prefix_cache_queries": queries, "vllm:prefix_cache_hits": hits}
        )
        == expected_rate
    )


@pytest.mark.parametrize(
    ("mode", "events"),
    [
        ("disabled", [{"computed_tokens": 0, "cache_enabled": False}]),
        ("miss", [{"computed_tokens": 0, "cache_enabled": True}]),
        ("hit", [{"computed_tokens": 1008, "cache_enabled": True}]),
    ],
)
def test_direct_prefix_observer_accepts_actual_cache_mode(mode, events) -> None:
    profile._validate_prefix_evidence(
        mode,
        {"vllm:prefix_cache_queries": 0.0, "vllm:prefix_cache_hits": 0.0},
        events,
        1,
    )


def test_direct_prefix_observer_rejects_missing_or_wrong_evidence() -> None:
    delta = {"vllm:prefix_cache_queries": 0.0, "vllm:prefix_cache_hits": 0.0}
    with pytest.raises(profile.EvidenceUnavailableError, match="recorded no queries"):
        profile._validate_prefix_evidence("hit", delta, [], 1)
    with pytest.raises(profile.EvidenceUnavailableError, match="computed tokens"):
        profile._validate_prefix_evidence(
            "miss", delta, [{"computed_tokens": 16, "cache_enabled": True}], 1
        )
    with pytest.raises(profile.EvidenceUnavailableError, match="count mismatch"):
        profile._validate_prefix_evidence(
            "hit", delta, [{"computed_tokens": 16, "cache_enabled": True}], 2
        )


def test_inproc_prefix_observer_preserves_native_result() -> None:
    blocks = object()
    manager = SimpleNamespace(get_computed_blocks=lambda _request: (blocks, 1008))
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(enable_prefix_caching=True)
            ),
            engine_core=SimpleNamespace(
                engine_core=SimpleNamespace(
                    scheduler=SimpleNamespace(kv_cache_manager=manager)
                )
            )
        )
    )
    events = profile._install_inproc_prefix_observer(llm)
    request = SimpleNamespace(
        request_id="native-0",
        num_prompt_tokens=1024,
        num_computed_tokens=0,
        gr_issued_steps=0,
        gr_persistent_runner_state=True,
    )

    assert manager.get_computed_blocks(request) == (blocks, 1008)
    assert events == [
        {
            "request_id": "native-0",
            "query_tokens": 1024,
            "computed_tokens": 1008,
            "num_computed_tokens_before": 0,
            "gr_issued_steps": 0,
            "persistent": True,
            "cache_enabled": True,
        }
    ]


def test_prefix_metrics_must_exist() -> None:
    with pytest.raises(profile.EvidenceUnavailableError, match="metrics are unavailable"):
        profile._metric_totals([])


def test_main_persists_environment_failure_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tmp_path / "catalog.json"
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "failure.json"
    catalog.write_text("{}", encoding="utf-8")
    prompt.write_text("prompt", encoding="utf-8")

    def fail(_args: argparse.Namespace) -> dict:
        raise profile.EnvironmentNotValidatedError("NPU accelerator is unavailable")

    monkeypatch.setattr(profile, "_run", fail)
    with pytest.raises(RuntimeError, match="accelerator"):
        profile.main(
            [
                "--model",
                "model",
                "--catalog",
                str(catalog),
                "--prompt",
                str(prompt),
                "--output-json",
                str(output),
                "--backend",
                "npu",
                "--execution",
                "eager",
                "--prefix-cache",
                "miss",
                "--implementation",
                "baseline",
                "--base-sha",
                "b" * 40,
                "--pr-324-sha",
                "c" * 40,
            ]
        )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "failed"
    assert evidence["validation_status"] == "ENVIRONMENT_NOT_VALIDATED"
    assert evidence["error"] == {
        "type": "EnvironmentNotValidatedError",
        "message": "NPU accelerator is unavailable",
    }


def test_failure_classification_does_not_use_message_substrings() -> None:
    evidence = profile._failure_result(
        _args(implementation="baseline"),
        RuntimeError("CUDA result mismatch"),
    )
    assert evidence["validation_status"] == "FAIL"
    import_failure = profile._failure_result(
        _args(implementation="baseline"),
        ImportError("torch-npu import failed outside preflight"),
    )
    assert import_failure["validation_status"] == "FAIL"
