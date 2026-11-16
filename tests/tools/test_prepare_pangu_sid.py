# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Offline conversion tests using real SentencePiece and safetensors assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

import pytest
import torch
import transformers
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, LlamaTokenizer

from tools.model_conversion import prepare_pangu_sid as converter

sentencepiece = pytest.importorskip(
    "sentencepiece", reason="The offline Pangu SID tests train a real tiny SentencePiece model"
)
sp_pb2 = pytest.importorskip(
    "sentencepiece.sentencepiece_model_pb2", reason="SentencePiece protobuf support is required"
)

HIDDEN_SIZE = 8
STAGE_SIZE = 3
REVISION = "a" * 40
ORDINARY_TEXTS = [
    "Hello, world!",
    "Pangu model beam search.",
    "\u4f60\u597d\uff0c\u8fd9\u662f\u63a8\u8350\u6d4b\u8bd5\u3002",
    "1234567890",
    "\nA B\tC",
]


class SourceFactory(Protocol):
    def __call__(self, *, head: str = "zero", dtype: torch.dtype = torch.float32) -> Path: ...


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _snapshot(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    # NumPy cannot expose BF16 directly. Compare the stored bits, including dtype.
    return bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())


def _load_checkpoint(directory: Path) -> dict[str, torch.Tensor]:
    """Reload every tensor using only the published HF shard index."""
    index = _json(directory / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    names = set(weight_map.values())
    assert names == {path.name for path in directory.glob("*.safetensors")}
    tensors: dict[str, torch.Tensor] = {}
    for name in sorted(names):
        with safe_open(directory / name, framework="pt", device="cpu") as handle:
            assert handle.metadata()["format"] == "pt"
            for key in handle.keys():
                assert key not in tensors
                assert weight_map[key] == name
                tensors[key] = handle.get_tensor(key)
    assert set(tensors) == set(weight_map)
    assert index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    return tensors


# The test mypy hook uses follow_imports=skip, including pytest decorators.
@pytest.fixture(scope="module")  # type: ignore[misc]
def tiny_tokenizer_assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = Path(tmp_path_factory.mktemp("pangu-sentencepiece"))
    model_prefix = directory / "tiny"
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(ORDINARY_TEXTS * 4),
        model_prefix=str(model_prefix),
        vocab_size=96,
        hard_vocab_limit=False,
        model_type="bpe",
        character_coverage=1.0,
        num_threads=1,
        shuffle_input_sentence=False,
        minloglevel=2,
    )
    # Transformers 5 uses a separate backend for native SentencePiece files.
    tokenizer_class = getattr(transformers, "SentencePieceBackend", LlamaTokenizer)
    tokenizer = tokenizer_class(
        vocab_file=str(model_prefix.with_suffix(".model")),
        legacy=True,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<unk>",
    )
    assets = directory / "assets"
    tokenizer.save_pretrained(assets)
    # The public snapshot includes this file; Transformers 5 omits it on save.
    _write_json(assets / "special_tokens_map.json", tokenizer.special_tokens_map)
    processor = sentencepiece.SentencePieceProcessor(
        model_file=str(model_prefix.with_suffix(".model"))
    )
    assert len(tokenizer) == tokenizer.vocab_size == processor.get_piece_size()
    assert (assets / "tokenizer.model").is_file()
    return assets


@pytest.fixture  # type: ignore[misc]
def source_factory(
    tmp_path: Path,
    tiny_tokenizer_assets: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SourceFactory:
    # These constants are the only conversion behavior replaced by the fixture.
    # The tokenizer, checkpoint conversion, serialization, and validation are real.
    import shutil

    old_tokenizer = AutoTokenizer.from_pretrained(
        tiny_tokenizer_assets, use_fast=False, local_files_only=True
    )
    vocab_size = len(old_tokenizer)
    monkeypatch.setattr(converter, "PUBLIC_VOCAB_SIZE", vocab_size)
    monkeypatch.setattr(converter, "GROUP_SIZE", STAGE_SIZE)
    monkeypatch.setattr(converter, "CHUNK_ROWS", 5)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    created = 0

    def make_source(*, head: str = "zero", dtype: torch.dtype = torch.float32) -> Path:
        nonlocal created
        created += 1
        source = tmp_path / f"source-{created}"
        shutil.copytree(tiny_tokenizer_assets, source)
        _write_json(
            source / "config.json",
            {
                "architectures": ["PanguEmbeddedForCausalLM"],
                "model_type": "pangu",
                "vocab_size": vocab_size,
                "hidden_size": HIDDEN_SIZE,
                "tie_word_embeddings": True,
                "initializer_range": 0.02,
                "bos_token_id": old_tokenizer.bos_token_id,
                "eos_token_id": old_tokenizer.eos_token_id,
                "pad_token_id": old_tokenizer.pad_token_id,
            },
        )
        _write_json(
            source / "generation_config.json",
            {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "do_sample": False},
        )
        embedding = (
            torch.arange(vocab_size * HIDDEN_SIZE, dtype=torch.float32)
            .reshape(vocab_size, HIDDEN_SIZE)
            .div(17)
            .add(0.125)
            .to(dtype)
        )
        first = {
            converter.EMBEDDING_KEY: embedding,
            "model.layers.0.self_attn.q_proj.weight": torch.arange(64, dtype=dtype).reshape(8, 8),
        }
        second = {"model.norm.weight": torch.linspace(0.5, 1.5, HIDDEN_SIZE, dtype=dtype)}
        if head == "zero":
            second[converter.HEAD_KEY] = torch.zeros_like(embedding)
        elif head == "equal":
            second[converter.HEAD_KEY] = embedding.clone()
        elif head == "different":
            second[converter.HEAD_KEY] = embedding + 1
        else:
            assert head == "absent"
        shards = {
            "model-00001-of-00002.safetensors": first,
            "model-00002-of-00002.safetensors": second,
        }
        for name, tensors in shards.items():
            save_file(tensors, source / name, metadata={"format": "pt"})
        _write_json(
            source / "model.safetensors.index.json",
            {
                "metadata": {
                    "total_size": sum(
                        tensor.numel() * tensor.element_size()
                        for tensors in shards.values()
                        for tensor in tensors.values()
                    )
                },
                "weight_map": {key: name for name, tensors in shards.items() for key in tensors},
            },
        )
        return source

    return make_source


def _args(source: Path, output: Path, **overrides: Any) -> argparse.Namespace:
    values = {
        "source": str(source),
        "output": str(output),
        "source_revision": REVISION,
        "seed": 1234,
        "alignment": 32,
        # Deliberately force several small shards without a large model fixture.
        "shard_mib": 1 / 1024,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])  # type: ignore[misc]
def test_conversion_preserves_source_and_extends_real_tokenizer(
    tmp_path: Path, source_factory: SourceFactory, dtype: torch.dtype
) -> None:
    source = source_factory(dtype=dtype)
    before = _snapshot(source)
    original = _load_checkpoint(source)
    old_config = _json(source / "config.json")
    old_tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False, local_files_only=True)
    old_proto = sp_pb2.ModelProto()
    old_proto.ParseFromString((source / "tokenizer.model").read_bytes())
    output = tmp_path / "new-parent" / "converted"

    assert converter.run(_args(source, output)) == output

    assert _snapshot(source) == before
    assert not list(output.parent.glob(output.name + ".incomplete-*"))
    tensors = _load_checkpoint(output)
    new_config = _json(output / "config.json")
    token_ids = _json(output / "sid_token_ids.json")["tokens"]
    old_size = old_config["vocab_size"]
    sid_tokens = [f"<s_{stage}_{i}>" for stage in "abc" for i in range(STAGE_SIZE)]
    sid_tokens += [converter.BEGIN_TOKEN, converter.END_TOKEN]
    unpadded_size = old_size + len(sid_tokens)
    padded_size = math.ceil(unpadded_size / 32) * 32
    assert padded_size > unpadded_size  # Exercise anonymous alignment capacity.
    assert new_config == {**old_config, "vocab_size": padded_size}
    assert (output / "generation_config.json").read_bytes() == (
        source / "generation_config.json"
    ).read_bytes()
    embedding = tensors[converter.EMBEDDING_KEY]
    assert embedding.shape == (padded_size, HIDDEN_SIZE)
    assert embedding.dtype == dtype
    assert _raw_bytes(embedding[:old_size]) == _raw_bytes(original[converter.EMBEDDING_KEY])
    assert torch.isfinite(embedding).all()
    assert torch.all(torch.count_nonzero(embedding[old_size:], dim=1) > 0)
    assert _raw_bytes(tensors[converter.HEAD_KEY]) == _raw_bytes(embedding)
    for key, tensor in original.items():
        if key not in {converter.EMBEDDING_KEY, converter.HEAD_KEY}:
            assert tensors[key].dtype == tensor.dtype
            assert _raw_bytes(tensors[key]) == _raw_bytes(tensor)
    index = _json(output / "model.safetensors.index.json")
    assert index["weight_map"][converter.EMBEDDING_KEY] != index["weight_map"][converter.HEAD_KEY]
    assert len(set(index["weight_map"].values())) >= 3

    tokenizer = AutoTokenizer.from_pretrained(output, use_fast=False, local_files_only=True)
    assert len(tokenizer) == tokenizer.vocab_size == unpadded_size
    for token, token_id in old_tokenizer.get_vocab().items():
        assert tokenizer.get_vocab()[token] == token_id
    new_proto = sp_pb2.ModelProto()
    new_proto.ParseFromString((output / "tokenizer.model").read_bytes())
    assert [piece.SerializeToString() for piece in new_proto.pieces[:old_size]] == [
        piece.SerializeToString() for piece in old_proto.pieces
    ]
    assert len(new_proto.pieces) == unpadded_size
    assert token_ids == {token: old_size + i for i, token in enumerate(sid_tokens)}
    sentinels = {converter.BEGIN_TOKEN, converter.END_TOKEN}
    for token in sid_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        assert ids == [token_ids[token]]
        assert tokenizer.decode(ids, skip_special_tokens=False) == token
        assert tokenizer.decode(ids, skip_special_tokens=True) == (
            "" if token in sentinels else token
        )
        assert tokenizer.added_tokens_decoder[ids[0]].special == (token in sentinels)
    expected_ids = [token_ids[token] for token in sid_tokens]
    assert tokenizer.encode("".join(sid_tokens), add_special_tokens=False) == expected_ids
    assert (
        tokenizer.convert_ids_to_tokens(expected_ids, skip_special_tokens=True) == sid_tokens[:-2]
    )
    for text in ORDINARY_TEXTS:
        assert tokenizer.encode(text) == old_tokenizer.encode(text)
    assert not any(token_id >= unpadded_size for token_id in tokenizer.get_vocab().values())

    manifest = _json(output / "conversion_manifest.json")
    assert manifest["trained"] is False
    assert manifest["recommendation_accuracy_claimed"] is False
    assert manifest["source"]["declared_revision"] == REVISION
    assert manifest["source"]["revision_verified"] is False
    assert manifest["weights"]["source_lm_head_kind"] == "all_zero_placeholder"
    for name, digest in manifest["source"]["file_sha256"].items():
        assert before[name] == digest
    for name, digest in manifest["output_file_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(  # type: ignore[misc]
    "head,expected_kind", [("absent", "absent"), ("equal", "same_as_embedding")]
)
def test_tied_head_is_materialized_from_embedding(
    tmp_path: Path, source_factory: SourceFactory, head: str, expected_kind: str
) -> None:
    source = source_factory(head=head)
    before = _snapshot(source)
    output = tmp_path / "converted"
    converter.run(_args(source, output))
    tensors = _load_checkpoint(output)
    assert torch.equal(tensors[converter.HEAD_KEY], tensors[converter.EMBEDDING_KEY])
    assert torch.count_nonzero(tensors[converter.HEAD_KEY]) > 0
    manifest = _json(output / "conversion_manifest.json")
    assert manifest["weights"]["source_lm_head_kind"] == expected_kind
    assert _snapshot(source) == before


def test_seed_controls_only_the_initialized_rows(
    tmp_path: Path, source_factory: SourceFactory
) -> None:
    source = source_factory()
    old_size = _json(source / "config.json")["vocab_size"]
    checkpoints = []
    for name, seed in [("first", 42), ("repeat", 42), ("different", 43)]:
        output = tmp_path / name
        converter.run(_args(source, output, seed=seed, source_revision=None))
        manifest = _json(output / "conversion_manifest.json")
        assert manifest["source"]["declared_revision"] is None
        assert manifest["source"]["revision_verified"] is False
        checkpoints.append(_load_checkpoint(output))
    first, repeat, different = checkpoints
    for key in first:
        assert _raw_bytes(first[key]) == _raw_bytes(repeat[key])
        if key in {converter.EMBEDDING_KEY, converter.HEAD_KEY}:
            assert _raw_bytes(first[key][:old_size]) == _raw_bytes(different[key][:old_size])
            assert _raw_bytes(first[key][old_size:]) != _raw_bytes(different[key][old_size:])
        else:
            assert _raw_bytes(first[key]) == _raw_bytes(different[key])


@pytest.mark.parametrize("kind", ["file", "directory"])  # type: ignore[misc]
def test_existing_output_is_never_overwritten(
    tmp_path: Path, source_factory: SourceFactory, kind: str
) -> None:
    source = source_factory()
    before = _snapshot(source)
    output = tmp_path / "existing"
    if kind == "file":
        output.write_bytes(b"keep this existing file")
    else:
        output.mkdir()
        (output / "sentinel").write_bytes(b"keep this existing directory")
    with pytest.raises(FileExistsError):
        converter.run(_args(source, output))
    assert (output if kind == "file" else output / "sentinel").read_bytes().startswith(b"keep")
    assert _snapshot(source) == before
    assert not list(tmp_path.glob(output.name + ".incomplete-*"))


@pytest.mark.parametrize(  # type: ignore[misc]
    "location", ["same", "inside", "normalized-inside"]
)
def test_overlapping_source_and_output_are_rejected(
    source_factory: SourceFactory, location: str
) -> None:
    source = source_factory()
    before = _snapshot(source)
    output = {
        "same": source,
        "inside": source / "converted",
        "normalized-inside": source / ".." / source.name / "converted",
    }[location]
    with pytest.raises((ValueError, FileExistsError)):
        converter.run(_args(source, output))
    assert _snapshot(source) == before
    assert not list(source.parent.glob("*.incomplete-*"))


def test_symlink_alias_cannot_create_output_inside_source(
    tmp_path: Path, source_factory: SourceFactory
) -> None:
    source = source_factory()
    before = _snapshot(source)
    alias = tmp_path / "source-alias"
    try:
        alias.symlink_to(source, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable on this platform")
    with pytest.raises(ValueError):
        converter.run(_args(source, alias / "converted"))
    assert _snapshot(source) == before


def test_dangling_output_symlink_is_not_replaced(
    tmp_path: Path, source_factory: SourceFactory
) -> None:
    source = source_factory()
    output = tmp_path / "dangling"
    missing = tmp_path / "missing-target"
    try:
        output.symlink_to(missing, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable on this platform")
    with pytest.raises(FileExistsError):
        converter.run(_args(source, output))
    assert output.is_symlink()
    assert not missing.exists()


@pytest.mark.parametrize(  # type: ignore[misc]
    "overrides", [{"alignment": 0}, {"shard_mib": 0}, {"source_revision": "main"}]
)
def test_invalid_arguments_fail_without_publishing(
    tmp_path: Path, source_factory: SourceFactory, overrides: dict[str, Any]
) -> None:
    source = source_factory()
    before = _snapshot(source)
    output = tmp_path / "converted"
    with pytest.raises(ValueError):
        converter.run(_args(source, output, **overrides))
    assert not output.exists()
    assert not list(tmp_path.glob(output.name + ".incomplete-*"))
    assert _snapshot(source) == before


@pytest.mark.parametrize(  # type: ignore[misc]
    "defect", ["index", "duplicate", "untied", "vocab"]
)
def test_invalid_source_is_rejected_before_conversion(
    tmp_path: Path, source_factory: SourceFactory, defect: str
) -> None:
    source = source_factory()
    if defect == "index":
        path = source / "model.safetensors.index.json"
        index = _json(path)
        index["weight_map"][converter.EMBEDDING_KEY] = "missing.safetensors"
        _write_json(path, index)
    elif defect == "duplicate":
        tensor = load_file(source / "model-00001-of-00002.safetensors")[converter.EMBEDDING_KEY]
        save_file({converter.EMBEDDING_KEY: tensor}, source / "duplicate.safetensors")
    else:
        path = source / "config.json"
        config = _json(path)
        config["tie_word_embeddings" if defect == "untied" else "vocab_size"] = (
            False if defect == "untied" else config["vocab_size"] + 1
        )
        _write_json(path, config)
    before = _snapshot(source)
    output = tmp_path / "converted"
    with pytest.raises(ValueError):
        converter.run(_args(source, output))
    assert not output.exists()
    assert not list(tmp_path.glob(output.name + ".incomplete-*"))
    assert _snapshot(source) == before


def test_bad_tied_head_leaves_failure_evidence_without_final_output(
    tmp_path: Path, source_factory: SourceFactory
) -> None:
    source = source_factory(head="different")
    before = _snapshot(source)
    output = tmp_path / "converted"
    with pytest.raises(ValueError, match="lm_head"):
        converter.run(_args(source, output))
    assert not output.exists()
    attempts = list(tmp_path.glob(output.name + ".incomplete-*"))
    assert len(attempts) == 1
    failure = _json(attempts[0] / "conversion_failure.json")
    assert failure["status"] == "failed"
    assert failure["error_type"] == "ValueError"
    assert "lm_head" in failure["error"]
    assert not (attempts[0] / "conversion_manifest.json").exists()
    # A real tokenizer conversion completed before the malformed head was read.
    assert (attempts[0] / "sid_token_ids.json").is_file()
    assert _snapshot(source) == before


pytestmark = pytest.mark.cpu_test
