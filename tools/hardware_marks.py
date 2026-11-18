# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Hardware resource markers for pytest decorators and parameter cases."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest


def _load_markers() -> dict[str, str]:
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with path.open("rb") as source:
        markers = tomllib.load(source)["tool"]["pytest"]["ini_options"]["markers"]
    entries = (marker.split(":", 1) for marker in markers)
    return {name.strip(): description for name, description in entries}


_MARKERS = _load_markers()
PLATFORM_MARKERS = {"cuda": "gpu_test", "npu": "npu_test"}
PLATFORM_SKUS = {
    platform: tuple(name for name, desc in _MARKERS.items() if f"[{platform}-sku]" in desc)
    for platform in PLATFORM_MARKERS
}
CARD_MARKERS = {
    int(name.removeprefix("cards_")): name
    for name, desc in _MARKERS.items()
    if "[hardware-cards]" in desc
}
HARDWARE_PLATFORM_FIXTURE = "_hardware_platform"
SkuSpec = str | list[str] | tuple[str, ...]
Test = TypeVar("Test", bound=Callable[..., object] | type)


def _marks_by_platform(
    cuda_skus: tuple[str, ...], res: dict[str, SkuSpec] | None, num_cards: int | dict[str, int]
) -> dict[str, tuple[int, list[pytest.MarkDecorator]]]:
    if res is not None and cuda_skus:
        raise ValueError("Use either positional CUDA SKUs or res")
    resources: dict[str, SkuSpec] = {"cuda": cuda_skus} if res is None else res
    if not resources or resources.keys() - PLATFORM_SKUS.keys():
        raise ValueError(f"res must use one or more platforms from {tuple(PLATFORM_SKUS)}")
    counts = num_cards if isinstance(num_cards, dict) else dict.fromkeys(resources, num_cards)
    if counts.keys() - resources.keys():
        raise ValueError("num_cards has a platform missing from res")

    marks_by_platform = {}
    for platform, resource in resources.items():
        skus = (resource,) if isinstance(resource, str) else tuple(resource)
        if not skus or set(skus).difference(PLATFORM_SKUS[platform]):
            raise ValueError(f"Expected {platform} SKUs from {PLATFORM_SKUS[platform]}, got {skus}")
        count = counts.get(platform, 1)
        if type(count) is not int or count not in CARD_MARKERS:
            raise ValueError(f"num_cards must be one of {tuple(CARD_MARKERS)}, got {count!r}")
        marks = [getattr(pytest.mark, name) for name in (PLATFORM_MARKERS[platform], *skus)]
        marks_by_platform[platform] = count, marks
    return marks_by_platform


def hardware_marks(
    *cuda_skus: str, res: dict[str, SkuSpec] | None = None, num_cards: int | dict[str, int] = 1
) -> list[pytest.MarkDecorator]:
    """Return resource marks for one item with a single card count."""
    by_platform = _marks_by_platform(cuda_skus, res, num_cards)
    counts = {count for count, _ in by_platform.values()}
    if len(counts) != 1:
        raise ValueError(
            "Use hardware_test or separate pytest.param cases for different card counts"
        )
    return [
        *(mark for _, marks in by_platform.values() for mark in marks),
        getattr(pytest.mark, CARD_MARKERS[counts.pop()]),
    ]


def hardware_test(
    *cuda_skus: str, res: dict[str, SkuSpec] | None = None, num_cards: int | dict[str, int] = 1
) -> Callable[[Test], Test]:
    """Split mixed card counts into separately selectable platform cases."""
    by_platform = _marks_by_platform(cuda_skus, res, num_cards)
    if len({count for count, _ in by_platform.values()}) > 1:
        params = [
            pytest.param(
                platform, id=platform, marks=[*marks, getattr(pytest.mark, CARD_MARKERS[count])]
            )
            for platform, (count, marks) in by_platform.items()
        ]

        def split(test: Test) -> Test:
            pytest.mark.usefixtures(HARDWARE_PLATFORM_FIXTURE)(test)
            pytest.mark.parametrize(HARDWARE_PLATFORM_FIXTURE, params, indirect=True)(test)
            return test

        return split

    marks = hardware_marks(*cuda_skus, res=res, num_cards=num_cards)

    def decorate(test: Test) -> Test:
        for mark in marks:
            mark(test)
        return test

    return decorate


def matches_sku(platform: str, device_name: str, allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return True
    # Token matching handles NVIDIA A100-SXM4-80GB without treating L40S as L40.
    tokens = re.findall(r"[A-Z0-9]+", device_name.upper())
    if platform == "npu":
        compact = "".join(tokens).removeprefix("ASCEND")
        for prefix, sku in (
            ("910B", "A2"),
            ("91093", "A3"),
            ("910C", "A3"),
            ("950", "A5"),
            ("310P", "310P"),
        ):
            if compact.startswith(prefix):
                return sku in allowed
    return any(sku in tokens for sku in allowed)
