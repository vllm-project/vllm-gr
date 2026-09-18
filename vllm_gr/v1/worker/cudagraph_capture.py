# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared error classification for optional GR CUDA graph capture."""


def is_recoverable_capture_error(error: RuntimeError) -> bool:
    """Allow CUDA's capture-unsupported errors, never arbitrary model failures.

    Inspect the chain: endCapture can mask the original error. Capture
    invalidation is fatal on supported PyTorch 2.11: endCapture raises before
    cleaning allocator/RNG state. Synchronization alone cannot repair it.
    """
    messages = []
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        # PyTorch appends generic advice about device-side assertions to
        # every CUDA error; only the error headline describes the failure.
        messages.append(str(current).split("\n", 1)[0].lower())
        current = current.__cause__ or current.__context__
    fatal = (
        "illegal memory",
        "device-side assert",
        "misaligned address",
        "out of memory",
        "operation failed due to a previous error during capture",
        "already recording to mempool_id",
    )
    if any(word in message for message in messages for word in fatal):
        return False
    # Every error in the chain must be understood. An outer shape/lifecycle
    # failure must not be hidden by an earlier capture-unsupported context.
    return all(
        "operation not permitted when stream is capturing" in message
        or "operation not supported when stream is capturing" in message
        for message in messages
    )
