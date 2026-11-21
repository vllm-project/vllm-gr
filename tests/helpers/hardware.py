# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Hardware declarations and probes for test selection, independent of vLLM.

Registration describes available runner types, not validated model support.
Torch and torch_npu are imported only when probing a selected execution target.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import regex as re

SKUS = {
    "cuda": ("L20", "A100", "H800", "H100", "L4", "B200"),
    "npu": ("A2", "A3", "A5"),
}
PLATFORMS = {"cuda": "gpu_test", "npu": "npu_test"}
CARD_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8)


@dataclass(frozen=True)
class Hardware:
    backend: str
    models: tuple[str, ...]
    num_cards: int = 1


@dataclass(frozen=True)
class DeviceInventory:
    backend: str
    names: tuple[str, ...]


def parse_target(value: str) -> Hardware:
    """Parse host, cuda:H100, or npu:A3; reject unknown or ambiguous targets."""
    if value == "host":
        return Hardware("host", ())
    backend, separator, model = value.partition(":")
    if not separator or backend not in SKUS or model not in SKUS[backend]:
        raise ValueError(f"Invalid test hardware {value!r}; use host, cuda:<SKU>, or npu:<SKU>")
    return Hardware(backend, (model,))


def marker_expression(target: Hardware) -> str:
    host = "(cpu_test and not (gpu_test or npu_test))"
    if target.backend == "host":
        return host
    all_models = " or ".join(model for models in SKUS.values() for model in models)
    return f"{host} or ({PLATFORMS[target.backend]} and ({target.models[0]} or not ({all_models})))"


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
