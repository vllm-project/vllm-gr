# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-gr: High-throughput serving plugin for HSTU generative recommendation models."""

__version__ = "0.1.0"


# try:
#     from . import patch  # noqa: F401
# except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
#     if exc.name != "vllm":
#         raise
#     # Allow importing vllm_gr without vllm (e.g., documentation builds)
#     patch = None  # type: ignore

# Empty package initializer