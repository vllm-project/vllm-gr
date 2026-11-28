# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Discover and cache an independent GR platform, following vLLM-Omni.

Importing this package must not load vllm, torch, or any device backend:
the winning class is imported only during resolution, and the interface
contract (which subclasses vLLM's Platform) is deferred for the same
reason. Probes read vLLM's current_platform lazily inside their bodies.
"""

from __future__ import annotations

import logging
import os
from importlib import metadata
from inspect import signature
from threading import RLock
from typing import TYPE_CHECKING

from .errors import AmbiguousBackendError, UnsupportedBackendError
from .identity import qualified_class_name

if TYPE_CHECKING:
    from vllm.platforms.interface import Platform

    from .interface import GRPlatform

logger = logging.getLogger(__name__)
PLATFORM_PLUGINS_GROUP = "vllm_gr.platform_plugins"
_lock = RLock()
_current_gr_platform: GRPlatform | None = None
_native_platform: Platform | None = None
_resolving = False


def _after_fork() -> None:
    global _lock, _current_gr_platform, _native_platform, _resolving
    _lock = RLock()
    _current_gr_platform = _native_platform = None
    _resolving = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def cuda_platform_plugin() -> str | None:
    from vllm.platforms import current_platform

    if current_platform.is_cuda() and current_platform.device_type == "cuda":
        return "vllm_gr.platforms.cuda.platform.CudaGRPlatform"
    return None


def ascend_platform_plugin() -> str | None:
    from vllm.platforms import current_platform

    if current_platform.device_type == "npu":
        from .identity import qualified_class_name

        if any(
            qualified_class_name(cls) == "vllm_ascend.platform.NPUPlatform"
            for cls in type(current_platform).__mro__
        ):
            return "vllm_gr.platforms.ascend.platform.AscendGRPlatform"
    return None


BUILTIN_PROBES = {"cuda": cuda_platform_plugin, "ascend": ascend_platform_plugin}


def _origin(entry: metadata.EntryPoint) -> str:
    dist = entry.dist
    return f"{dist.name if dist is not None else '<unknown distribution>'}:{entry.name}"


def _resolve(native: Platform) -> GRPlatform:
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from .interface import GRPlatform, UnspecifiedGRPlatform

    matches: list[tuple[metadata.EntryPoint, str]] = []
    for entry in metadata.entry_points(group=PLATFORM_PLUGINS_GROUP):
        try:
            probe = entry.load()
        except Exception:
            logger.warning("Could not load GR probe %s", _origin(entry), exc_info=True)
            continue
        # An old probe(identity) must not silently fall back to a built-in.
        try:
            signature(probe).bind()
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"GR platform probe {_origin(entry)} must accept zero arguments"
            ) from exc
        try:
            path = probe()
        except Exception:
            logger.warning("GR platform probe %s failed", _origin(entry), exc_info=True)
            continue
        if path is not None:
            if not isinstance(path, str) or not path:
                raise TypeError(
                    f"GR platform probe {_origin(entry)} must return a class path or None"
                )
            matches.append((entry, path))

    if len(matches) > 1:
        origins = ", ".join(sorted(_origin(entry) for entry, _ in matches))
        raise AmbiguousBackendError(f"Multiple external GR platforms match: {origins}")

    selected_entry: metadata.EntryPoint | None = None
    if matches:
        selected_entry, path = matches[0]
        origin = _origin(selected_entry)
    else:
        builtins = [(name, path) for name, probe in BUILTIN_PROBES.items() if (path := probe())]
        if not builtins:
            return UnspecifiedGRPlatform()
        if len(builtins) > 1:
            raise AmbiguousBackendError(f"Multiple built-in GR integrations match: {builtins}")
        origin, path = builtins[0]

    platform_cls = resolve_obj_by_qualname(path)
    if not isinstance(platform_cls, type) or not issubclass(platform_cls, GRPlatform):
        raise TypeError(f"{origin} selected {path}, expected a GRPlatform subclass")
    if (
        type(platform_cls.api_version) is not int
        or platform_cls.api_version != GRPlatform.api_version
    ):
        raise RuntimeError(
            f"GR platform {origin} requires platform API {platform_cls.api_version!r}; "
            f"this vllm-gr supports version {GRPlatform.api_version}"
        )
    native_type = getattr(platform_cls, "native_platform_type", None)
    if native_type is not None and not isinstance(native, native_type):
        raise RuntimeError(
            f"{origin} selected {path} for incompatible native platform {qualified_class_name(type(native))}"
        )
    try:
        platform = platform_cls()
    except Exception as exc:
        raise RuntimeError(f"Could not initialize selected GR platform {origin}") from exc
    if selected_entry is not None and selected_entry.dist is not None:
        platform.plugin_distribution = selected_entry.dist.name
    logger.debug("Selected GR platform %s from %s", path, origin)
    return platform


def get_current_gr_platform(*, required: bool = True) -> GRPlatform:
    from vllm.platforms import current_platform

    from .interface import UnspecifiedGRPlatform

    global _current_gr_platform, _native_platform, _resolving
    with _lock:
        if _current_gr_platform is None or _native_platform is not current_platform:
            if _resolving:
                raise RuntimeError("Recursive GR platform discovery")
            _resolving = True
            try:
                selected = _resolve(current_platform)
                _current_gr_platform, _native_platform = selected, current_platform
            finally:
                _resolving = False
        if required and isinstance(_current_gr_platform, UnspecifiedGRPlatform):
            raise UnsupportedBackendError(
                f"No GR integration for {qualified_class_name(type(current_platform))} "
                f"({current_platform._enum.name}, device_type={current_platform.device_type!r})"
            )
        return _current_gr_platform


if TYPE_CHECKING:
    current_gr_platform: GRPlatform


def __getattr__(name: str):
    if name == "current_gr_platform":
        return get_current_gr_platform(required=False)
    raise AttributeError(name)
