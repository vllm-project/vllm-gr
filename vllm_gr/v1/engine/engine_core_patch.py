# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_FORK."""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# EngineCore.__init__ wrapper
# ---------------------------------------------------------------------------


def make_patched_engine_core_init(original_init):
    """Wrap EngineCore.__init__ to add beam state."""

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.beam_cache: dict[str, dict] = {}

    return patched_init


# ---------------------------------------------------------------------------
# Helpers: add an enum member to EngineCoreRequestType at runtime
# ---------------------------------------------------------------------------


def _add_enum_member(name: str, value: bytes) -> None:
    """Dynamically add a member to EngineCoreRequestType if absent."""
    if hasattr(EngineCoreRequestType, name):
        return
    member = object.__new__(EngineCoreRequestType)
    member._name_ = name
    member._value_ = value
    EngineCoreRequestType._value2member_map_[value] = member
    EngineCoreRequestType._member_map_[name] = member
    EngineCoreRequestType._member_names_.append(name)
    type.__setattr__(EngineCoreRequestType, name, member)


# ---------------------------------------------------------------------------
# Child-process patch applicator — called from patched run_engine_core
# ---------------------------------------------------------------------------


def apply_engine_core_child_patches():
    """Apply all patches inside the EngineCore child process.

    This must be called at the start of run_engine_core (the child process
    entry point) because multiprocessing 'spawn' starts a fresh Python
    interpreter that does not inherit monkey patches from the parent.
    """
    import vllm.v1.engine as engine_mod
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    from vllm_gr.v1.engine.core import _cache_beam_request, _handle_beam_fork, process_input_sockets
    from vllm_gr.v1.engine.types import BeamForkRequest

    # Inject types into vllm.v1.engine module
    engine_mod.BeamForkRequest = BeamForkRequest

    # Add new enum members: ADD_BATCH, BEAM_FORK
    _add_enum_member("ADD_BATCH", b"\x05")
    _add_enum_member("BEAM_FORK", b"\x06")

    # Wrap EngineCore.__init__ to add beam state
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)

    # Bind _cache_beam_request and _handle_beam_fork on both classes
    EngineCore._cache_beam_request = _cache_beam_request
    EngineCore._handle_beam_fork = _handle_beam_fork
    EngineCoreProc._cache_beam_request = _cache_beam_request
    EngineCoreProc._handle_beam_fork = _handle_beam_fork

    # Replace EngineCoreProc.process_input_sockets
    EngineCoreProc.process_input_sockets = process_input_sockets

    logger.info("Engine-core patches (ADD_BATCH, BEAM_FORK) applied in child process.")


def run_engine_core(*args, **kwargs):
    """Wrapper for EngineCoreProc.run_engine_core that applies patches
    before the engine starts.  This is a module-level function so that
    multiprocessing 'spawn' can pickle it by qualified name.

    In the spawned child process, all modules are freshly imported, so
    EngineCoreProc.run_engine_core is the original unpatched version.
    We apply our patches first, then delegate to the original.
    """
    from vllm.v1.engine.core import EngineCoreProc

    # Save reference to original before applying patches
    original_run = EngineCoreProc.run_engine_core
    apply_engine_core_child_patches()
    original_run(*args, **kwargs)
