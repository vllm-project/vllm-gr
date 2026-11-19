# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Build generated gRPC classes without registering symbols in the default pool."""

from google.protobuf import message_factory
from google.protobuf.descriptor import Descriptor, FileDescriptor
from google.protobuf.internal import api_implementation
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
from google.protobuf.message import Message


def _reduce_nested_message(message: Message) -> tuple[type[Message], tuple[()], object]:
    return type(message), (), message.__getstate__()


def _message_class(descriptor: Descriptor, module_name: str, qualname: str) -> type[Message]:
    message_class = message_factory.GetMessageClass(descriptor)
    message_class.__module__ = module_name
    message_class.__qualname__ = qualname
    if descriptor.containing_type is not None and api_implementation.Type() == "python":
        # Protobuf resolves nested pickles in the default symbol database.
        setattr(message_class, "__reduce__", _reduce_nested_message)
    for name, nested in descriptor.nested_types_by_name.items():
        setattr(message_class, name, _message_class(nested, module_name, f"{qualname}.{name}"))
    return message_class


def build_top_descriptors_and_messages(
    file_descriptor: FileDescriptor, module_name: str, namespace: dict[str, object]
) -> None:
    """Populate the generated module from descriptors already in its private pool."""
    # The stock builder registers message files in the default pool when the
    # pure-Python protobuf implementation is active.
    for name, enum in file_descriptor.enum_types_by_name.items():
        namespace[f"_{name.upper()}"] = enum
        namespace[name] = EnumTypeWrapper(enum)
        namespace.update((value.name, value.number) for value in enum.values)

    for name, extension in file_descriptor.extensions_by_name.items():
        namespace[f"{name.upper()}_FIELD_NUMBER"] = extension.number
        namespace[name] = extension

    for name, service in file_descriptor.services_by_name.items():
        namespace[f"_{name.upper()}"] = service

    for name, descriptor in file_descriptor.message_types_by_name.items():
        namespace[name] = _message_class(descriptor, module_name, name)
