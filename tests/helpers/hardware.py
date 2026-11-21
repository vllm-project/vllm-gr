# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Hardware declarations and probes for test selection, independent of vLLM.

Registration describes available runner types, not validated model support.
Torch and torch_npu are imported only when probing a selected execution target.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import pytest
import regex as re

SKUS = {
    "cuda": ("L20", "A100", "H800", "H100", "L4", "B200"),
    "npu": ("A2", "A3", "A5"),
}
LEGACY_MARKERS = {"host": "cpu_test", "cuda": "gpu_test", "npu": "npu_test"}
PLATFORM_MARKERS = tuple(LEGACY_MARKERS) + tuple(LEGACY_MARKERS.values())
CARD_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8)
_VARIANT_FIXTURE = "_gr_hardware_variant"
_TestFunction = TypeVar("_TestFunction", bound=Callable[..., Any])


@dataclass(frozen=True)
class Hardware:
    backend: str
    models: tuple[str, ...]
    num_cards: int = 1


@dataclass(frozen=True)
class DeviceInventory:
    backend: str
    names: tuple[str, ...]


def _requirements(
    res: Mapping[str, str | Sequence[str]], num_cards: int | Mapping[str, int]
) -> list[Hardware]:
    if not res:
        raise ValueError("res must declare at least one accelerator platform")
    if isinstance(num_cards, Mapping) and set(num_cards) - set(res):
        raise ValueError("num_cards contains a platform absent from res")
    requirements = []
    for backend, value in res.items():
        if backend not in SKUS:
            raise ValueError(f"Unknown accelerator platform {backend!r}; use cuda or npu")
        models = (value,) if isinstance(value, str) else tuple(value)
        if not models or any(model not in SKUS[backend] for model in models):
            raise ValueError(f"Invalid {backend} models {models!r}; registered: {SKUS[backend]}")
        cards = num_cards.get(backend, 1) if isinstance(num_cards, Mapping) else num_cards
        if isinstance(cards, bool) or not isinstance(cards, int) or cards not in CARD_COUNTS:
            raise ValueError(f"num_cards must be one of {CARD_COUNTS}, got {cards!r}")
        requirements.append(Hardware(backend, tuple(dict.fromkeys(models)), cards))
    return requirements


def _marks(requirement: Hardware) -> list[pytest.MarkDecorator]:
    return [
        pytest.mark.hardware(requirement),
        getattr(pytest.mark, requirement.backend),
        getattr(pytest.mark, LEGACY_MARKERS[requirement.backend]),
        getattr(pytest.mark, f"cards_{requirement.num_cards}"),
        *(getattr(pytest.mark, model) for model in requirement.models),
    ]


def hardware_marks(
    *, res: Mapping[str, str | Sequence[str]], num_cards: int | Mapping[str, int] = 1
) -> list[pytest.MarkDecorator]:
    """Marks for one pytest.param; platforms must have the same card count.

    For different counts use hardware_test, or a separate pytest.param for each
    platform, so a CUDA four-card item cannot match an NPU two-card selection.
    """
    requirements = _requirements(res, num_cards)
    if len({requirement.num_cards for requirement in requirements}) != 1:
        raise ValueError("Different platform card counts require hardware_test or separate params")
    return [mark for requirement in requirements for mark in _marks(requirement)]


def hardware_test(
    *, res: Mapping[str, str | Sequence[str]], num_cards: int | Mapping[str, int] = 1
) -> Callable[[_TestFunction], _TestFunction]:
    """Collect one variant per platform, with independent SKU/card marks."""
    requirements = _requirements(res, num_cards)
    params = [
        pytest.param(requirement, marks=_marks(requirement), id=requirement.backend)
        for requirement in requirements
    ]

    def decorate(function: _TestFunction) -> _TestFunction:
        function = pytest.mark.usefixtures(_VARIANT_FIXTURE)(function)
        return cast(
            _TestFunction,
            pytest.mark.parametrize(_VARIANT_FIXTURE, params, indirect=True)(function),
        )

    return decorate


def parse_target(value: str) -> Hardware:
    """Parse host, cuda:H100, or npu:A3; reject unknown or ambiguous targets."""
    if value == "host":
        return Hardware("host", ())
    backend, separator, model = value.partition(":")
    if not separator or backend not in SKUS or model not in SKUS[backend]:
        raise ValueError(f"Invalid test hardware {value!r}; use host, cuda:<SKU>, or npu:<SKU>")
    return Hardware(backend, (model,))


def model_from_name(backend: str, name: str) -> str | None:
    normalized = re.sub(r"[^A-Z0-9]", "", name.upper())
    if backend == "cuda":
        # Boundaries prevent L4 from matching L40, or A100 from matching A1000.
        for sku in SKUS[backend]:
            if re.search(rf"(?<![A-Z0-9]){sku}(?![A-Z0-9])", name.upper()):
                return sku
    elif backend == "npu":
        # torch_npu names the SoC, not the Atlas generation. Unknown names fail
        # closed; do not infer a generation merely from 'Ascend' or '910'.
        if normalized.startswith("ASCEND910B"):
            return "A2"
        if normalized.startswith(("ASCEND910939", "ASCEND910C")):
            return "A3"
        if normalized.startswith("ASCEND950"):
            return "A5"
    return None


def probe_devices(backend: str) -> DeviceInventory:
    """Inspect runtime-visible logical devices, honoring visibility variables."""
    torch = importlib.import_module("torch")

    if backend == "cuda":
        if torch.version.hip:
            raise ValueError("A ROCm runtime cannot validate a CUDA target")
        device = torch.cuda
    elif backend == "npu":
        importlib.import_module("torch_npu")

        device = torch.npu
    else:
        raise ValueError(f"Cannot probe accelerator platform {backend!r}")
    if not device.is_available():
        raise ValueError(f"No usable {backend} runtime/device")
    return DeviceInventory(
        backend, tuple(device.get_device_name(index) for index in range(device.device_count()))
    )


def validate_inventory(requirement: Hardware, inventory: DeviceInventory) -> None:
    """Require enough visible devices, all of an allowed, recognized SKU.

    Restrict visibility first on a heterogeneous machine. Merely counting a
    matching subset would allow the engine to execute on a different device.
    """
    models = tuple(model_from_name(inventory.backend, name) for name in inventory.names)
    if (
        inventory.backend != requirement.backend
        or len(models) < requirement.num_cards
        or any(model not in requirement.models for model in models)
    ):
        raise ValueError(
            f"Required {requirement.backend} {requirement.models} with at least "
            f"{requirement.num_cards} visible devices; found {inventory.backend} {inventory.names}. "
            "Check runner routing and device visibility."
        )
