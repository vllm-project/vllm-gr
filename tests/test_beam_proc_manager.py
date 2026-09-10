# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_gr.entrypoints.openai import beam_proc_manager


def test_export_catalog_preserves_integer_token_ids(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    output_path = tmp_path / "catalog-token-ids.json"
    catalog_path.write_text(json.dumps([[1, 2, 3], [4, 5]]), encoding="utf-8")
    tokenizer = MagicMock()

    entry_count = beam_proc_manager._export_catalog_token_ids(
        catalog_path,
        output_path,
        tokenizer,
    )

    assert entry_count == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == [[1, 2, 3], [4, 5]]
    tokenizer.convert_tokens_to_ids.assert_not_called()


def test_beam_proc_catalog_skip_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = beam_proc_manager.BeamProcConfig(
        config_path=tmp_path / "beam-proc-config.json",
        enginecore_beamproc_addr="ipc://enginecore.sock",
        enginecore_beamproc_tx_addr="ipc://enginecore-tx.sock",
        enginecore_beamproc_rx_addr="ipc://enginecore-rx.sock",
        apiserver_output_addr="ipc://apiserver.sock",
        ready_path=tmp_path / "beam-proc.ready",
    )
    beam_proc_manager._CATALOG_RESULTS.clear()
    monkeypatch.setattr(
        beam_proc_manager,
        "ensure_beam_proc_config",
        lambda: prepared,
    )

    with patch.object(beam_proc_manager.logger, "info") as log_info:
        assert beam_proc_manager.prepare_beam_proc_catalog(None, None) is None
        assert beam_proc_manager.prepare_beam_proc_catalog(None, None) is None

    log_info.assert_called_once_with("BeamProc catalog export skipped: no catalog_path configured")


def test_beam_proc_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(beam_proc_manager.ENV_ENABLED, raising=False)
    monkeypatch.delenv(beam_proc_manager.ENV_STARTED, raising=False)
    beam_proc_manager._HANDLE = None

    assert beam_proc_manager.ensure_beam_proc_started() is None


def test_beam_proc_not_started_twice_in_child(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(beam_proc_manager.ENV_ENABLED, "1")
    monkeypatch.setenv(beam_proc_manager.ENV_STARTED, "1")
    monkeypatch.setenv(beam_proc_manager.ENV_ENGINECORE_ADDR, "ipc://enginecore.sock")
    beam_proc_manager._HANDLE = None

    with patch("subprocess.Popen") as mock_popen:
        handle = beam_proc_manager.ensure_beam_proc_started()

    assert handle is None
    mock_popen.assert_not_called()


def test_dead_beam_proc_handle_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = MagicMock()
    process.poll.return_value = 17
    handle = beam_proc_manager.BeamProcHandle(
        process=process,
        config_path=tmp_path / "beam-proc-config.json",
        enginecore_beamproc_addr="ipc://enginecore.sock",
        enginecore_beamproc_tx_addr="ipc://enginecore-tx.sock",
        enginecore_beamproc_rx_addr="ipc://enginecore-rx.sock",
        apiserver_output_addr="ipc://apiserver.sock",
    )
    monkeypatch.setattr(beam_proc_manager, "_HANDLE", handle)

    with pytest.raises(RuntimeError, match="status 17.*dead endpoint"):
        beam_proc_manager.ensure_beam_proc_started()

    process.poll.assert_called_once_with()


def test_beam_proc_exit_code_reports_owned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = MagicMock()
    process.poll.return_value = 23
    monkeypatch.setattr(
        beam_proc_manager,
        "_HANDLE",
        beam_proc_manager.BeamProcHandle(
            process=process,
            config_path=tmp_path / "beam-proc-config.json",
            enginecore_beamproc_addr="ipc://enginecore.sock",
            enginecore_beamproc_tx_addr="ipc://enginecore-tx.sock",
            enginecore_beamproc_rx_addr="ipc://enginecore-rx.sock",
            apiserver_output_addr="ipc://apiserver.sock",
        ),
    )

    assert beam_proc_manager.beam_proc_exit_code() == 23


def test_beam_proc_starts_with_generated_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "beam-proc"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setenv(beam_proc_manager.ENV_ENABLED, "1")
    monkeypatch.delenv(beam_proc_manager.ENV_BINARY, raising=False)
    monkeypatch.setattr(beam_proc_manager, "_PACKAGED_BINARY", binary)
    monkeypatch.delenv(beam_proc_manager.ENV_CONFIG, raising=False)
    monkeypatch.delenv(beam_proc_manager.ENV_STARTED, raising=False)
    beam_proc_manager._HANDLE = None

    process = MagicMock()
    process.pid = 123
    process.poll.return_value = 0

    with (
        patch("subprocess.Popen", return_value=process) as mock_popen,
        patch.object(beam_proc_manager, "_wait_for_beam_proc_ready"),
    ):
        handle = beam_proc_manager.ensure_beam_proc_started()

    assert handle is not None
    assert handle.process is process
    assert handle.config_path.exists()
    config = json.loads(handle.config_path.read_text(encoding="utf-8"))
    assert config["enginecore_beamproc_addr"].startswith("ipc://")
    assert config["apiserver_output_addr"].startswith("ipc://")
    assert config["queue_capacity"] == 64
    assert config["zmq_hwm"] == 64
    assert mock_popen.call_args.args[0] == [str(binary), str(handle.config_path)]
    assert mock_popen.call_args.kwargs["cwd"] == str(binary.parent)
    assert (
        mock_popen.call_args.kwargs["env"][beam_proc_manager.ENV_ENGINECORE_ADDR]
        == handle.enginecore_beamproc_addr
    )
    assert (
        mock_popen.call_args.kwargs["env"][beam_proc_manager.ENV_APISERVER_OUTPUT_ADDR]
        == handle.apiserver_output_addr
    )
    assert mock_popen.call_args.kwargs["env"][beam_proc_manager.ENV_STARTED] == "1"

    handle.shutdown()
    beam_proc_manager._HANDLE = None
    monkeypatch.delenv(beam_proc_manager.ENV_STARTED, raising=False)


def test_beam_proc_uses_existing_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "beam-proc"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "num_workers": 1,
                "log_level": "debug",
                "log_file": str(tmp_path / "beam.log"),
                "enginecore_beamproc_addr": "ipc://enginecore.sock",
                "apiserver_output_addr": "ipc://apiserver.sock",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(beam_proc_manager.ENV_ENABLED, "true")
    monkeypatch.setenv(beam_proc_manager.ENV_BINARY, str(binary))
    monkeypatch.setenv(beam_proc_manager.ENV_CONFIG, str(config_path))
    monkeypatch.delenv(beam_proc_manager.ENV_STARTED, raising=False)
    beam_proc_manager._HANDLE = None

    process = MagicMock()
    process.pid = 456
    process.poll.return_value = 0

    with (
        patch("subprocess.Popen", return_value=process),
        patch.object(beam_proc_manager, "_wait_for_beam_proc_ready"),
    ):
        handle = beam_proc_manager.ensure_beam_proc_started()

    assert handle is not None
    assert handle.config_path == config_path
    assert handle.enginecore_beamproc_addr == "ipc://enginecore.sock"
    assert handle.apiserver_output_addr == "ipc://apiserver.sock"
    assert handle.temp_dir is None

    handle.shutdown()
    beam_proc_manager._HANDLE = None
    monkeypatch.delenv(beam_proc_manager.ENV_STARTED, raising=False)


def test_beam_proc_ready_acknowledgment(tmp_path: Path) -> None:
    ready_path = tmp_path / "beam-proc.ready"
    ready_path.write_text("READY\n", encoding="utf-8")
    process = MagicMock()
    process.poll.return_value = None

    beam_proc_manager._wait_for_beam_proc_ready(process, ready_path)


def test_beam_proc_exit_before_ready_fails_fast(tmp_path: Path) -> None:
    process = MagicMock()
    process.poll.return_value = 23

    with pytest.raises(RuntimeError, match="status 23 before READY"):
        beam_proc_manager._wait_for_beam_proc_ready(
            process,
            tmp_path / "missing.ready",
        )
