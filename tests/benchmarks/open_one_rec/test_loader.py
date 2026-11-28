# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import errno
import json
import stat
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from benchmarks.open_one_rec.open_one_rec_loader import (
    TASK_REGISTRY,
    BaseLoader,
    DataLoaderWrapper,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_registered_tasks_match_downloadable_recommendation_tasks() -> None:
    from tools.AB_Tests.download_onerec_dataset import TASK_FILES

    expected = {"ad", "product", "label_cond", "video", "interactive", "label_pred"}
    assert set(TASK_REGISTRY) == set(TASK_FILES) == expected


@pytest.fixture  # type: ignore[misc]
def video_parquet(tmp_path: Path) -> tuple[Path, pd.DataFrame, DataLoaderWrapper]:
    source = tmp_path / "video" / "video_test.parquet"
    source.parent.mkdir()
    rows = pd.DataFrame(
        [
            {
                "messages": json.dumps([{"role": "user", "content": f" question {i} "}]),
                "metadata": json.dumps({"answer": f"answer {i}", "row_index": -1}),
            }
            for i in range(10)
        ]
    )
    rows.to_parquet(source, index=False)

    def render(messages: list[dict[str, str]], **kwargs: bool) -> str:
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return f"user:{messages[0]['content']}\nassistant:"

    return (
        source,
        rows,
        DataLoaderWrapper(tokenizer=Mock(apply_chat_template=render), data_dir=str(tmp_path)),
    )


@pytest.mark.parametrize("cache_at_root", [False, True])  # type: ignore[misc]
def test_sample_cache_preserves_source_rows(
    video_parquet: tuple[Path, pd.DataFrame, DataLoaderWrapper],
    monkeypatch: pytest.MonkeyPatch,
    cache_at_root: bool,
) -> None:
    source, rows, loader = video_parquet
    cache = (
        source.parent.parent if cache_at_root else source.parent
    ) / "video_test_sample_3.parquet"
    sampled = rows.sample(3, random_state=42)
    expected = [
        (
            str(index),
            index,
            [{"role": "user", "content": f" question {index} "}],
            f"user: question {index} \nassistant:",
            f"answer {index}",
        )
        for index in sampled.index
    ]

    cold = loader.load_data("video", sample_size=3)
    cold_cache = source.parent / cache.name
    assert stat.S_IMODE(cold_cache.stat().st_mode) == stat.S_IMODE(source.stat().st_mode)
    with monkeypatch.context() as patch:
        read_source = Mock(side_effect=AssertionError("source read on cache hit"))
        patch.setattr(loader._loader_cache["video"], "_load_dataframe", read_source)
        warm = loader.load_data("video", sample_size=3)
        read_source.assert_not_called()
    assert list(warm) == list(cold) == list(map(str, sampled.index))

    if cache_at_root:
        cold_cache.unlink()
    sampled.to_parquet(cache, index=False)
    migrated = loader.load_data("video", sample_size=3)

    for result in (cold, warm, migrated):
        assert [
            (
                sample_id,
                sample["metadata"]["row_index"],
                sample["chat_messages"],
                sample["prompt"],
                sample["ground_truth"],
            )
            for sample_id, sample in result.items()
        ] == expected
        assert all(
            sample["chat_template_kwargs"] == {"enable_thinking": False}
            for sample in result.values()
        )

    source.unlink()
    assert loader.load_data("video", sample_size=3) == migrated


def test_loader_defaults_to_downloaded_dataset_root(
    video_parquet: tuple[Path, pd.DataFrame, DataLoaderWrapper],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, rows, wrapper = video_parquet
    data_dir = tmp_path / "data" / "benchmark_data"
    data_dir.mkdir(parents=True)
    source.parent.rename(data_dir / "video")
    monkeypatch.chdir(tmp_path)

    loader = BaseLoader(TASK_REGISTRY["video"].config, tokenizer=wrapper.tokenizer)
    samples = loader.load_data(sample_size=3)

    assert list(samples) == list(map(str, rows.sample(3, random_state=42).index))
    assert (data_dir / "video" / "video_test_sample_3.parquet").is_file()


def test_exact_sample_ids_use_source_order_requested(
    video_parquet: tuple[Path, pd.DataFrame, DataLoaderWrapper],
) -> None:
    _, _, loader = video_parquet
    loader.load_data("video", sample_size=3)
    result = loader.load_data("video", sample_ids=["7", 2, "0"])
    assert list(result) == ["7", "2", "0"]
    assert [sample["ground_truth"] for sample in result.values()] == [
        "answer 7",
        "answer 2",
        "answer 0",
    ]


def test_legacy_sample_cache_requires_source(
    video_parquet: tuple[Path, pd.DataFrame, DataLoaderWrapper],
) -> None:
    source, rows, loader = video_parquet
    rows.sample(3, random_state=42).to_parquet(
        source.with_name("video_test_sample_3.parquet"), index=False
    )
    source.unlink()
    with pytest.raises(FileNotFoundError) as exc:
        loader.load_data("video", sample_size=3)
    assert str(source) in str(exc.value)


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EROFS])  # type: ignore[misc]
def test_sample_cache_write_failure_keeps_samples_and_cleans_up(
    video_parquet: tuple[Path, pd.DataFrame, DataLoaderWrapper],
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    source, rows, loader = video_parquet
    replace = Mock(side_effect=OSError(error_number, "read-only dataset"))
    monkeypatch.setattr("benchmarks.open_one_rec.open_one_rec_loader.os.replace", replace)
    result = loader.load_data("video", sample_size=3)
    assert list(result) == list(map(str, rows.sample(3, random_state=42).index))
    replace.assert_called_once()
    assert list(source.parent.iterdir()) == [source]
