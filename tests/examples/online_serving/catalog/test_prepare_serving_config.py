# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.online_serving.catalog import prepare_serving_config as serving_config
from vllm_gr.config.parser import load_gr_config, read_settings
from vllm_gr.constraints.artifact import load_constraint_table_artifact

pytestmark = pytest.mark.cpu_test


@pytest.fixture  # type: ignore[misc]
def local_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sample = json.loads(serving_config.SAMPLE_CATALOG.read_text(encoding="utf-8"))
    vocab = {
        token: index
        for index, token in enumerate(dict.fromkeys(token for row in sample for token in row), 4)
    }
    tokenizer = SimpleNamespace(
        get_vocab=lambda: vocab,
        convert_tokens_to_ids=lambda values: [vocab[value] for value in values],
    )
    transformers = Mock()
    transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    transformers.AutoConfig.from_pretrained.return_value = SimpleNamespace(vocab_size=20)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    model = tmp_path / "local model"
    model.mkdir()
    return model


@pytest.mark.l1  # type: ignore[misc]
def test_prepare_serving_config_reuses_inputs_and_publishes_loadable_tables(
    tmp_path: Path, local_model: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workdir = tmp_path / "serving output"
    workdir.mkdir()
    raw_catalog = workdir / "catalog.json"
    sample = json.loads(serving_config.SAMPLE_CATALOG.read_text(encoding="utf-8"))
    raw_catalog.write_text(json.dumps([sample[0]]), encoding="utf-8")
    raw_input = raw_catalog.read_bytes()

    args = ["--model", str(local_model), "--workdir", str(workdir)]
    config = workdir / "vllm_gr_config.yaml"
    assert serving_config.main(args) == 0
    previous = read_settings(config)["gr"]["constraint_table"]
    assert serving_config.main([*args, "--raw-catalog", str(raw_catalog)]) == 0
    current = read_settings(config)["gr"]["constraint_table"]
    assert previous["path"] != current["path"]
    assert previous["digests"]["artifact"] != current["digests"]["artifact"]
    assert Path(previous["path"]).is_file()

    triples = workdir / "constraint_triples.json"
    triples.write_bytes(Path(current["path"]).with_name(triples.name).read_bytes())
    triples_input = triples.read_bytes()
    assert (
        serving_config.main([*args, "--triples", str(triples), "--constraint-backend", "npu"]) == 0
    )
    assert read_settings(config)["attention-config"] == {"backend": "CUSTOM"}
    constraint = load_gr_config(config).constraint
    assert constraint is not None
    spec = constraint.table
    assert spec is not None and spec.path is not None
    assert spec.backend.value == "npu"
    assert Path(spec.path).is_absolute()
    assert spec.artifact_digest and spec.artifact_digest == current["digests"]["artifact"]
    assert spec.tokenizer_digest and spec.tokenizer_digest == current["digests"]["tokenizer"]
    load_constraint_table_artifact(
        spec.path,
        expected_digest=spec.artifact_digest,
        expected_tokenizer_digest=spec.tokenizer_digest,
        expected_vocab_size=20,
    )
    assert str(config) in capsys.readouterr().out
    assert raw_catalog.read_bytes() == raw_input
    assert triples.read_bytes() == triples_input

    with pytest.raises(SystemExit) as exc:
        serving_config.main([*args, "--raw-catalog", str(raw_catalog), "--triples", str(triples)])
    assert exc.value.code == 2
    invalid = workdir / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    published = config.read_bytes()
    assert serving_config.main([*args, "--triples", str(invalid)]) == 1
    assert config.read_bytes() == published
    assert Path(spec.path).is_file()


@pytest.mark.l1  # type: ignore[misc]
def test_publish_failure_discards_only_this_attempt(tmp_path: Path, local_model: Path) -> None:
    workdir = tmp_path / "serving"
    args = ["--model", str(local_model), "--workdir", str(workdir)]
    assert serving_config.main(args) == 0
    target = workdir / "vllm_gr_config.yaml"
    old_config = target.rename(workdir / "previous.yaml")
    old_yaml = old_config.read_bytes()
    table = Path(read_settings(old_config)["gr"]["constraint_table"]["path"])
    old_table = table.read_bytes()
    target.mkdir()
    entries = set(workdir.iterdir())

    assert serving_config.main(args) == 1
    assert set(workdir.iterdir()) == entries
    assert target.is_dir()
    assert old_config.read_bytes() == old_yaml
    assert table.read_bytes() == old_table


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("after_replace", [False, True], ids=["before-replace", "after-replace"])  # type: ignore[misc]
def test_interrupted_publish_preserves_referenced_tables(
    tmp_path: Path, local_model: Path, monkeypatch: pytest.MonkeyPatch, after_replace: bool
) -> None:
    workdir = tmp_path / "serving"
    config, old_table = serving_config.prepare_serving_config(
        model=local_model, workdir=workdir, backend="cuda"
    )
    old_yaml = config.read_bytes()
    old_entries = set(workdir.iterdir())
    replace = Path.replace

    def interrupt_replace(source: Path, target: Path) -> Path:
        if target != config:
            return replace(source, target)
        if after_replace:
            replace(source, target)
        raise KeyboardInterrupt

    with monkeypatch.context() as interrupt:
        interrupt.setattr(Path, "replace", interrupt_replace)
        with pytest.raises(KeyboardInterrupt):
            serving_config.prepare_serving_config(
                model=local_model, workdir=workdir, backend="cuda"
            )

    active_table = Path(read_settings(config)["gr"]["constraint_table"]["path"])
    assert old_table.is_file()
    assert active_table.is_file()
    if after_replace:
        assert active_table != old_table
        assert set(workdir.iterdir()) - old_entries == {active_table.parent}
    else:
        assert config.read_bytes() == old_yaml
        assert set(workdir.iterdir()) == old_entries
