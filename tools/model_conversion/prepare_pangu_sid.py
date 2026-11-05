#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare an untrained SID checkpoint from local public openPangu assets.

The conversion runs offline on CPU and imports the tokenizer's trusted local
Python implementation. Install torch, safetensors, transformers, sentencepiece
and protobuf before running it. It does not download or train a model.

Example::

    python tools/model_conversion/prepare_pangu_sid.py \
        --source /models/openPangu-public \
        --output /models/openPangu-public-sid

The source must be an untouched local snapshot of
``openpangu/openPangu-Embedded-1B-V1.1``. Optional revision metadata is declared
by the caller and is not authenticated against local files or Hugging Face.
The manifest records SHA-256 hashes of actual input assets for reproducibility.

The destination must not exist and must not contain, or be inside, the source.
A failed conversion leaves an ``.incomplete-*`` directory for inspection; the
final destination appears only after the saved tokenizer and weights pass
validation. Newly initialized SID rows are untrained and intended for execution
smoke tests, with no recommendation accuracy claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PUBLIC_REPO = "openpangu/openPangu-Embedded-1B-V1.1"
PUBLIC_REVISION = "146d3a854e2930b5e3dfbf9dc1187a6c78f8decd"
PUBLIC_VOCAB_SIZE = 153376
EMBEDDING_KEY = "model.embed_tokens.weight"
HEAD_KEY = "lm_head.weight"
GROUP_SIZE = 8192
CHUNK_ROWS = 256
BEGIN_TOKEN = "<|sid_begin|>"
END_TOKEN = "<|sid_end|>"


def progress(phase: str, **fields: Any) -> None:
    print(json.dumps({"phase": phase, **fields}, sort_keys=True), flush=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    """Hash raw CPU bytes without converting BF16 to a different dtype."""
    import torch

    digest = hashlib.sha256()
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    for offset in range(0, raw.numel(), 8 * 1024 * 1024):
        digest.update(raw[offset : offset + 8 * 1024 * 1024].numpy().tobytes())
    return digest.hexdigest()


def metadata_files(source: Path) -> list[Path]:
    """Copy only public model assets; old weight indexes are rebuilt below."""
    names = {"LICENSE", "Open Source Software Notice", "README.md", "README_EN.md"}
    result = []
    for item in source.iterdir():
        if not item.is_file() or item.name.endswith(".safetensors.index.json"):
            continue
        if item.name in names or item.suffix in {".json", ".py"}:
            result.append(item)
        elif item.name == "tokenizer.model":
            result.append(item)
    return sorted(result)


def inspect_source(source: Path) -> tuple[dict[str, Any], dict[str, str]]:
    from safetensors import safe_open

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["PanguEmbeddedForCausalLM"]:
        raise ValueError("Expected the public PanguEmbeddedForCausalLM architecture")
    if config.get("vocab_size") != PUBLIC_VOCAB_SIZE:
        raise ValueError(f"Source is not the original public {PUBLIC_VOCAB_SIZE}-row vocabulary")
    if config.get("tie_word_embeddings") is not True:
        raise ValueError("This conversion expects public Pangu tied embeddings")
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError("Source contains no public safetensors checkpoint")
    weight_map: dict[str, str] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in weight_map:
                    raise ValueError(f"Duplicate checkpoint tensor: {key}")
                weight_map[key] = file.name
    if EMBEDDING_KEY not in weight_map:
        raise ValueError(f"Missing public embedding tensor: {EMBEDDING_KEY}")
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        expected_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        if weight_map != expected_map:
            raise ValueError("Source safetensors and weight index disagree")
    return config, weight_map


def extend_tokenizer(
    source: Path, output: Path, alignment: int
) -> tuple[dict[str, int], dict[str, Any]]:
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2
    from transformers import AutoTokenizer

    progress("load_source_tokenizer_start")
    old_tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=True, use_fast=False, local_files_only=True
    )
    progress("load_source_tokenizer_done", vocab_size=len(old_tokenizer))
    if old_tokenizer.vocab_size != PUBLIC_VOCAB_SIZE:
        raise ValueError("The public tokenizer's SentencePiece size is unexpected")
    if len(old_tokenizer) != PUBLIC_VOCAB_SIZE:
        raise ValueError("The source has extra tokens; use an untouched public snapshot")
    proto = sp_pb2.ModelProto()
    proto.ParseFromString((source / "tokenizer.model").read_bytes())
    if len(proto.pieces) != PUBLIC_VOCAB_SIZE:
        raise ValueError("SentencePiece and public model vocabulary disagree")
    original_pieces = [piece.SerializeToString() for piece in proto.pieces]
    sid_tokens = [f"<s_{stage}_{i}>" for stage in "abc" for i in range(GROUP_SIZE)]
    sid_tokens += [BEGIN_TOKEN, END_TOKEN]
    existing_vocab = old_tokenizer.get_vocab()
    if any(token in existing_vocab for token in sid_tokens):
        raise ValueError("The source already includes SID tokens")
    sid_end_size = PUBLIC_VOCAB_SIZE + len(sid_tokens)
    padded_size = math.ceil(sid_end_size / alignment) * alignment
    # Extra checkpoint rows are anonymous alignment capacity, matching the
    # usual HF arrangement where len(tokenizer) can be smaller than vocab_size.
    # They are not SID candidates and need no invented tokenizer symbols.
    tokens = sid_tokens
    token_ids = {token: PUBLIC_VOCAB_SIZE + i for i, token in enumerate(tokens)}
    for token in tokens:
        piece = proto.pieces.add()
        piece.piece = token
        piece.score = 0.0
        # The HF added-token table also makes adjacency independent of the
        # SentencePiece whitespace normalizer and guarantees single-token SIDs.
        piece.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED
    proto.trainer_spec.vocab_size = len(proto.pieces)
    (output / "tokenizer.model").write_bytes(proto.SerializeToString())
    if [piece.SerializeToString() for piece in proto.pieces[:PUBLIC_VOCAB_SIZE]] != original_pieces:
        raise AssertionError("An original public SentencePiece entry was modified")

    config_path = output / "tokenizer_config.json"
    tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder = tokenizer_config.setdefault("added_tokens_decoder", {})
    sentinels = {BEGIN_TOKEN, END_TOKEN}
    for token, token_id in token_ids.items():
        if str(token_id) in decoder:
            raise ValueError(f"Token id already present in added-token config: {token_id}")
        decoder[str(token_id)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": token in sentinels,
        }
    specials = tokenizer_config.setdefault("additional_special_tokens", [])
    specials.extend([BEGIN_TOKEN, END_TOKEN])
    write_json(config_path, tokenizer_config)
    added_path = output / "added_tokens.json"
    added = json.loads(added_path.read_text(encoding="utf-8")) if added_path.exists() else {}
    added.update(token_ids)
    write_json(added_path, added)
    special_path = output / "special_tokens_map.json"
    special_map = json.loads(special_path.read_text(encoding="utf-8"))
    special_map.setdefault("additional_special_tokens", []).extend([BEGIN_TOKEN, END_TOKEN])
    write_json(special_path, special_map)

    progress("load_extended_tokenizer_start")
    tokenizer = AutoTokenizer.from_pretrained(
        output, trust_remote_code=True, use_fast=False, local_files_only=True
    )
    progress("load_extended_tokenizer_done", vocab_size=len(tokenizer))
    if len(tokenizer) != sid_end_size or tokenizer.vocab_size != sid_end_size:
        raise AssertionError("Reloaded tokenizer and actual vocabulary disagree")
    reloaded_vocab = tokenizer.get_vocab()
    for token, token_id in existing_vocab.items():
        if reloaded_vocab.get(token) != token_id:
            raise AssertionError(f"Original token id changed: {token!r}")
    # These symbols have no whitespace/boundary/normalization behavior.
    # One concatenated call validates every token and every adjacent boundary
    # without repeatedly traversing HF tokenizer setup for 24578 inputs.
    progress("bulk_sid_encode_start", token_count=len(sid_tokens))
    actual = tokenizer.encode("".join(sid_tokens), add_special_tokens=False)
    expected = [token_ids[token] for token in sid_tokens]
    if actual != expected:
        mismatch = next(
            (i for i, (left, right) in enumerate(zip(actual, expected)) if left != right),
            min(len(actual), len(expected)),
        )
        raise AssertionError(
            f"Bulk SID encoding differs at index {mismatch}; "
            f"actual_count={len(actual)}, expected_count={len(expected)}"
        )
    progress("bulk_sid_encode_done", token_count=len(actual))
    individual_tokens = [f"<s_{stage}_{i}>" for stage in "abc" for i in (0, 1, GROUP_SIZE - 1)] + [
        BEGIN_TOKEN,
        END_TOKEN,
    ]
    individual_checks = []
    for token in individual_tokens:
        expected = [token_ids[token]]
        actual = tokenizer.encode(token, add_special_tokens=False)
        if actual != expected:
            raise AssertionError(f"SID is not atomic: {token}: {actual} != {expected}")
        decoded = tokenizer.decode(actual, skip_special_tokens=False)
        decoded_skip = tokenizer.decode(actual, skip_special_tokens=True)
        expected_skip = "" if token in sentinels else token
        if decoded != token or decoded_skip != expected_skip:
            raise AssertionError(
                f"SID decode behavior differs: {token}: {decoded!r}, skip={decoded_skip!r}"
            )
        individual_checks.append(
            {
                "token": token,
                "ids": actual,
                "decoded": decoded,
                "decoded_skip_special_tokens": decoded_skip,
            }
        )
    # SA/SB/SC are ordinary recommendation tokens. Only the two structural
    # sentinels are special, so skipping special IDs must retain all 24576 SID
    # symbols. Validate the HF token conversion path for the whole SID list.
    retained_tokens = tokenizer.convert_ids_to_tokens(
        [token_ids[token] for token in sid_tokens], skip_special_tokens=True
    )
    if retained_tokens != sid_tokens[:-2]:
        raise AssertionError("Skipping special tokens removes or changes normal SID symbols")
    reloaded_added_tokens = tokenizer.added_tokens_decoder
    for token in sid_tokens:
        added_token = reloaded_added_tokens[token_ids[token]]
        if added_token.special != (token in sentinels):
            raise AssertionError(f"Reloaded special-token flag is incorrect: {token}")
    adjacency_checks = []
    for i in (0, 1, GROUP_SIZE - 1):
        sequence = [BEGIN_TOKEN] + [f"<s_{stage}_{i}>" for stage in "abc"] + [END_TOKEN]
        actual = tokenizer.encode("".join(sequence), add_special_tokens=False)
        expected = [token_ids[token] for token in sequence]
        if actual != expected:
            raise AssertionError("Adjacent SID tokens did not preserve the intended sequence")
        adjacency_checks.append({"tokens": sequence, "ids": actual})
    # Added symbols can intentionally change text containing those exact
    # symbols. Ordinary public text should retain its original segmentation.
    ordinary_texts = [
        "Hello, world!",
        "Pangu model beam search.",
        "\u4f60\u597d\uff0c\u8fd9\u662f\u63a8\u8350\u6d4b\u8bd5\u3002",
        "1234567890",
        "\nA B\tC",
    ]
    for text in ordinary_texts:
        if tokenizer.encode(text) != old_tokenizer.encode(text):
            raise AssertionError(f"Ordinary public text changed tokenization: {text!r}")
    report = {
        "original_vocab_size": PUBLIC_VOCAB_SIZE,
        "sid_vocab_size": len(sid_tokens),
        "tokenizer_vocab_size": sid_end_size,
        "padded_vocab_size": padded_size,
        "padding_rows": padded_size - sid_end_size,
        "padding_rows_have_no_tokenizer_symbols": True,
        "stage_size": GROUP_SIZE,
        "stages": {
            stage: {
                "first_id": token_ids[f"<s_{stage}_0>"],
                "last_id": token_ids[f"<s_{stage}_{GROUP_SIZE - 1}>"],
            }
            for stage in "abc"
        },
        "begin_token": {"text": BEGIN_TOKEN, "id": token_ids[BEGIN_TOKEN]},
        "end_token": {"text": END_TOKEN, "id": token_ids[END_TOKEN]},
        "all_original_ids_preserved": True,
        "all_original_sentencepiece_entries_preserved": True,
        "all_sid_concatenated_ids_verified": True,
        "individual_sid_checks": individual_checks,
        "adjacency_checks": adjacency_checks,
        "ordinary_text_checks": len(ordinary_texts),
        "sid_tokens_are_special": False,
        "sentinel_tokens_are_special": True,
        "skip_special_tokens_retains_all_sid_symbols_verified": True,
    }
    write_json(output / "sid_token_ids.json", {"tokens": token_ids, "layout": report})
    progress("tokenizer_validation_done")
    return token_ids, report


class ShardWriter:
    """Write bounded CPU tensor batches and construct a standard HF index."""

    def __init__(self, output: Path, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("Shard size must be positive")
        self.output = output
        self.max_bytes = max_bytes
        self.pending: dict[str, Any] = {}
        self.pending_bytes = 0
        self.total_bytes = 0
        self.shards: list[str] = []
        self.weight_map: dict[str, str] = {}

    def add(self, key: str, tensor: Any, *, singleton: bool = False) -> None:
        size = tensor.numel() * tensor.element_size()
        if self.pending and (singleton or self.pending_bytes + size > self.max_bytes):
            self.flush()
        self.pending[key] = tensor
        self.pending_bytes += size
        if singleton or self.pending_bytes >= self.max_bytes:
            self.flush()

    def flush(self) -> None:
        from safetensors.torch import save_file

        if not self.pending:
            return
        # Final HF-style names are assigned only after the count is known.
        name = f"conversion-shard-{len(self.shards) + 1:05d}.safetensors"
        save_file(self.pending, self.output / name, metadata={"format": "pt"})
        for key in self.pending:
            self.weight_map[key] = name
        self.total_bytes += self.pending_bytes
        self.shards.append(name)
        self.pending.clear()
        self.pending_bytes = 0

    def finish(self) -> None:
        self.flush()
        names = {}
        count = len(self.shards)
        for i, name in enumerate(self.shards, 1):
            final = f"model-{i:05d}-of-{count:05d}.safetensors"
            (self.output / name).rename(self.output / final)
            names[name] = final
        self.weight_map = {key: names[value] for key, value in self.weight_map.items()}
        write_json(
            self.output / "model.safetensors.index.json",
            {
                "metadata": {"total_size": self.total_bytes},
                "weight_map": self.weight_map,
            },
        )


def convert_weights(
    source: Path,
    output: Path,
    weight_map: dict[str, str],
    config: dict[str, Any],
    padded_size: int,
    seed: int,
    shard_bytes: int,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    progress("convert_weights_start", source_tensor_count=len(weight_map))
    writer = ShardWriter(output, shard_bytes)
    inventory: dict[str, Any] = {}
    old_size = int(config["vocab_size"])
    hidden = int(config["hidden_size"])
    std = float(config.get("initializer_range", 0.02))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Invalid public initializer_range")
    with safe_open(source / weight_map[EMBEDDING_KEY], framework="pt", device="cpu") as handle:
        original = handle.get_slice(EMBEDDING_KEY)
        if original.get_shape() != [old_size, hidden]:
            raise ValueError("Public embedding shape disagrees with config")
        sample = original[0:1]
        if sample.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ValueError(f"Unsupported public embedding dtype: {sample.dtype}")
        expanded = torch.empty((padded_size, hidden), dtype=sample.dtype, device="cpu")
        for start in range(0, old_size, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, old_size)
            expanded[start:stop].copy_(original[start:stop])
        old_hash = tensor_sha256(handle.get_tensor(EMBEDDING_KEY))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for start in range(old_size, padded_size, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, padded_size)
        values = torch.randn((stop - start, hidden), generator=generator, dtype=torch.float32)
        expanded[start:stop].copy_((values * std).to(expanded.dtype))
    if not torch.isfinite(expanded).all().item():
        raise ValueError("Expanded public embedding contains NaN or Inf")
    if torch.any(torch.count_nonzero(expanded[old_size:], dim=1) == 0).item():
        raise AssertionError("An initialized SID or padding row is all zero")
    if tensor_sha256(expanded[:old_size]) != old_hash:
        raise AssertionError("Original embedding prefix changed during initialization")
    expanded_hash = tensor_sha256(expanded)

    placeholder_kind = "absent"
    if HEAD_KEY in weight_map:
        zero_head = True
        equal_head = True
        with safe_open(source / weight_map[HEAD_KEY], framework="pt", device="cpu") as handle:
            head = handle.get_slice(HEAD_KEY)
            if head.get_shape() != [old_size, hidden]:
                raise ValueError("Public lm_head shape disagrees with tied embedding")
            for start in range(0, old_size, CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, old_size)
                chunk = head[start:stop]
                zero_head = zero_head and torch.count_nonzero(chunk).item() == 0
                equal_head = equal_head and torch.equal(chunk, expanded[start:stop])
        if not zero_head and not equal_head:
            raise ValueError(
                "Source tied lm_head is neither the public zero placeholder nor embedding"
            )
        placeholder_kind = "all_zero_placeholder" if zero_head else "same_as_embedding"

    # Save aliases in separate files: safetensors forbids two names sharing
    # storage inside one file. Both files contain identical nonzero weights,
    # while config keeps the runtime model tied. No zero output head survives.
    writer.add(EMBEDDING_KEY, expanded, singleton=True)
    writer.add(HEAD_KEY, expanded, singleton=True)
    inventory[EMBEDDING_KEY] = {
        "operation": "preserve_public_prefix_and_initialize_new_rows",
        "dtype": str(expanded.dtype),
        "shape": list(expanded.shape),
        "source_prefix_sha256": old_hash,
        "output_prefix_sha256": old_hash,
        "output_sha256": expanded_hash,
    }
    inventory[HEAD_KEY] = {
        "operation": "materialize_tied_embedding_alias",
        "source_kind": placeholder_kind,
        "dtype": str(expanded.dtype),
        "shape": list(expanded.shape),
        "output_sha256": expanded_hash,
    }
    del expanded
    gc.collect()
    for key in sorted(weight_map):
        if key in {EMBEDDING_KEY, HEAD_KEY}:
            continue
        with safe_open(source / weight_map[key], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
            digest = tensor_sha256(tensor)
            inventory[key] = {
                "operation": "unchanged",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "source_sha256": digest,
            }
            writer.add(key, tensor)
    writer.finish()

    # Re-read the written checkpoint, rather than trusting the in-memory
    # conversion. Verify every original tensor and both materialized aliases.
    for key, item in inventory.items():
        with safe_open(output / writer.weight_map[key], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
            digest = tensor_sha256(tensor)
            expected = item.get("source_sha256", item.get("output_sha256"))
            if digest != expected:
                raise AssertionError(f"Written tensor changed: {key}")
            item["verified_output_sha256"] = digest
    write_json(output / "conversion_tensor_inventory.json", inventory)
    progress("convert_weights_done", output_tensor_count=len(writer.weight_map))
    return {
        "source_tensor_count": len(weight_map),
        "output_tensor_count": len(writer.weight_map),
        "all_other_tensors_bitwise_preserved": True,
        "old_embedding_prefix_bitwise_preserved": True,
        "embedding_shape": [padded_size, hidden],
        "source_lm_head_kind": placeholder_kind,
        "output_lm_head_equals_embedding": True,
        "all_initialized_rows_finite_and_nonzero": True,
        "initialization": {
            "seed": seed,
            "distribution": "normal",
            "std": std,
            "generation_dtype": "torch.float32",
            "chunk_rows": CHUNK_ROWS,
        },
        "shards": len(writer.shards),
        "total_tensor_bytes": writer.total_bytes,
    }


def run(args: argparse.Namespace) -> Path:
    import torch

    source = Path(args.source).resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(f"Source is not a model directory: {source}")
    requested_destination = Path(args.output).absolute()
    if requested_destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite destination: {requested_destination}")
    destination = requested_destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite destination: {destination}")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must not contain one another")
    if args.source_revision is not None and (
        len(args.source_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("--source-revision must be a full lowercase 40-character commit hash")
    shard_bytes = int(args.shard_mib * 1024 * 1024)
    if args.alignment <= 0 or shard_bytes <= 0:
        raise ValueError("Alignment and shard size must be positive")
    torch.set_num_threads(1)
    config, weight_map = inspect_source(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=destination.name + ".incomplete-", dir=destination.parent)
    )
    started = time.time()
    try:
        assets = metadata_files(source)
        source_file_set = set(assets + [source / name for name in weight_map.values()])
        source_index = source / "model.safetensors.index.json"
        if source_index.exists():
            source_file_set.add(source_index)
        source_files = sorted(source_file_set)
        source_hashes = {path.name: sha256_file(path) for path in source_files}
        for path in assets:
            shutil.copy2(path, temporary / path.name)
        _, token_report = extend_tokenizer(source, temporary, args.alignment)
        config["vocab_size"] = token_report["padded_vocab_size"]
        config["tie_word_embeddings"] = True
        write_json(temporary / "config.json", config)
        # Public EOS/BOS/pad IDs and generation behavior remain untouched.
        weight_report = convert_weights(
            source,
            temporary,
            weight_map,
            {**config, "vocab_size": PUBLIC_VOCAB_SIZE},
            config["vocab_size"],
            args.seed,
            shard_bytes,
        )
        for path in source_files:
            if sha256_file(path) != source_hashes[path.name]:
                raise AssertionError(f"Source changed during conversion: {path}")
        output_hashes = {
            path.name: sha256_file(path) for path in sorted(temporary.iterdir()) if path.is_file()
        }
        versions = {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "safetensors", "sentencepiece", "protobuf")
        }
        manifest = {
            "format_version": 1,
            "status": "converted_and_cpu_validated",
            "purpose": "Untrained synthetic SID checkpoint for Beam execution smoke tests only",
            "trained": False,
            "recommendation_accuracy_claimed": False,
            "source": {
                "repo_id": PUBLIC_REPO,
                "declared_revision": args.source_revision,
                "revision_verified": False,
                "revision_note": (
                    "Caller-declared provenance label; local snapshot identity is not authenticated"
                ),
                "local_directory": str(source),
                "file_sha256": source_hashes,
            },
            "output_directory": str(destination),
            "tokenizer": token_report,
            "weights": weight_report,
            "output_file_sha256": output_hashes,
            "runtime": {"python": sys.version, "packages": versions},
            "converter_sha256": sha256_file(Path(__file__).resolve()),
            "elapsed_seconds": time.time() - started,
        }
        write_json(temporary / "conversion_manifest.json", manifest)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite destination: {destination}")
        temporary.rename(destination)
        return destination
    except BaseException as exc:
        write_json(
            temporary / "conversion_failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source": str(source),
                "requested_output": str(destination),
                "elapsed_seconds": time.time() - started,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, help="Untouched local public HF snapshot")
    parser.add_argument(
        "--source-revision",
        default=None,
        help=(
            "Optional declared commit hash; local snapshot identity is not verified. "
            f"Validated public snapshot: {PUBLIC_REVISION}"
        ),
    )
    parser.add_argument(
        "--output", required=True, help="New model directory; must not already exist"
    )
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument(
        "--alignment",
        "--pad-to-multiple",
        type=int,
        default=256,
        help="Pad checkpoint vocabulary rows to this multiple (default: 256)",
    )
    parser.add_argument(
        "--shard-mib", type=int, default=256, help="Maximum shard MiB, except individual tensors"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    progress("starting", source=args.source, output=args.output)
    output = run(args)
    progress(
        "converted_and_cpu_validated",
        output=str(output),
        manifest=str(output / "conversion_manifest.json"),
    )


if __name__ == "__main__":
    main()
