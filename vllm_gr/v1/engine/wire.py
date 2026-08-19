# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wire-level EngineCore request type extensions owned by vLLM-GR."""

from __future__ import annotations

from vllm.v1.engine import EngineCoreRequestType

ADD_BATCH_WIRE_VALUE = b"\x06"
BEAM_REQUEST_STEP_UPDATE_WIRE_VALUE = b"\x07"
BEAM_CONTROLLER_START_WIRE_VALUE = b"\x08"
BEAM_CONTROLLER_CANCEL_WIRE_VALUE = b"\x09"
BEAM_CONSTRAINT_RESOURCE_REGISTER_WIRE_VALUE = b"\x0a"


def _add_enum_member(name: str, value: bytes) -> None:
    """Dynamically add a member to EngineCoreRequestType if absent.

    This uses object.__new__() to bypass enum's normal creation pathway,
    which is necessary because we're monkey-patching at runtime and cannot
    modify the original class definition in vLLM's codebase.

    While this approach manipulates enum internals, it:
    - Is a standard technique used by enum extension libraries (e.g. aenum)
    - Includes validation to detect incompatibilities early
    - Is contained to initialization time (not runtime performance critical)
    - Is tested to ensure correctness

    Args:
        name: The name of the new enum member (e.g., "ADD_BATCH")
        value: The byte value for the enum member.

    Raises:
        RuntimeError: If the enum's internal structure is incompatible.
    """
    # Skip if already added with the expected wire value.
    if hasattr(EngineCoreRequestType, name):
        existing = getattr(EngineCoreRequestType, name)
        if existing.value != value:
            raise RuntimeError(
                f"EngineCoreRequestType.{name} already exists with wire value "
                f"{existing.value!r}, expected {value!r}"
            )
        return

    # Validate that the enum has expected internal structure.
    required_attrs = ["_value2member_map_", "_member_map_", "_member_names_"]
    for attr in required_attrs:
        if not hasattr(EngineCoreRequestType, attr):
            raise RuntimeError(
                f"EngineCoreRequestType missing expected attribute '{attr}'. "
                f"This indicates an incompatible Python/vLLM version. "
                f"Please report this issue with your Python version."
            )

    # Verify the value type matches existing members.
    if EngineCoreRequestType._value2member_map_:
        existing_value_type = type(next(iter(EngineCoreRequestType._value2member_map_.keys())))
        if not isinstance(value, existing_value_type):
            raise TypeError(
                f"Value type mismatch: expected {existing_value_type.__name__}, "
                f"got {type(value).__name__}"
            )
        if value in EngineCoreRequestType._value2member_map_:
            existing = EngineCoreRequestType._value2member_map_[value]
            raise RuntimeError(
                f"EngineCoreRequestType wire value {value!r} is already used by "
                f"{existing.name}; cannot register {name}"
            )

    # Create the member instance bypassing enum's __new__.
    member = object.__new__(EngineCoreRequestType)
    member._name_ = name
    member._value_ = value

    # Register in enum's internal maps.
    EngineCoreRequestType._value2member_map_[value] = member
    EngineCoreRequestType._member_map_[name] = member
    EngineCoreRequestType._member_names_.append(name)

    # Set as class attribute (must use type.__setattr__ to bypass enum's __setattr__).
    type.__setattr__(EngineCoreRequestType, name, member)


def register_engine_core_request_types() -> None:
    """Register the vLLM-GR EngineCore request types in the current process."""
    _add_enum_member("ADD_BATCH", ADD_BATCH_WIRE_VALUE)
    _add_enum_member(
        "BEAM_REQUEST_STEP_UPDATE",
        BEAM_REQUEST_STEP_UPDATE_WIRE_VALUE,
    )
    _add_enum_member("BEAM_CONTROLLER_START", BEAM_CONTROLLER_START_WIRE_VALUE)
    _add_enum_member("BEAM_CONTROLLER_CANCEL", BEAM_CONTROLLER_CANCEL_WIRE_VALUE)
    _add_enum_member(
        "BEAM_CONSTRAINT_RESOURCE_REGISTER",
        BEAM_CONSTRAINT_RESOURCE_REGISTER_WIRE_VALUE,
    )
