# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Configurable comparisons of the old resource and its replacement."""

from pathlib import Path

import pytest

from vllm_gr.sidecar import default_ckpt_checker, default_trie_checker


def test_checkpoint_same_files_and_sizes_allow_new_weights(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    for directory, content in [(old, b"old"), (new, b"new")]:
        (directory / "shards").mkdir(parents=True)
        (directory / "shards" / "model.bin").write_bytes(content)
    assert default_ckpt_checker(old, new)


@pytest.mark.parametrize("change", ["missing", "renamed", "added", "resized"])  # type: ignore[misc]
def test_checkpoint_rejects_file_list_or_size_changes(tmp_path: Path, change: str) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    for directory in (old, new):
        directory.mkdir()
        (directory / "model.bin").write_bytes(b"old")
    if change == "missing":
        (new / "model.bin").unlink()
    elif change == "renamed":
        (new / "model.bin").rename(new / "other.bin")
    elif change == "added":
        (new / "extra.bin").write_bytes(b"new")
    else:
        (new / "model.bin").write_bytes(b"larger")
    assert not default_ckpt_checker(old, new)


def test_trie_compares_names_without_requiring_same_tensor_sizes(tmp_path: Path) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    old, new = tmp_path / "old.safetensors", tmp_path / "new.safetensors"
    save_file({"first": np.zeros(1), "second": np.zeros(2)}, old)
    save_file({"first": np.ones(4), "second": np.ones(5)}, new)
    assert default_trie_checker(old, new)
    save_file({"first": np.ones(4), "renamed": np.ones(5)}, new)
    assert not default_trie_checker(old, new)
    save_file({"first": np.ones(4)}, new)
    assert not default_trie_checker(old, new)


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
