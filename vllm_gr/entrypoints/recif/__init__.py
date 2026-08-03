# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RECIF-specific online serving endpoint."""

from vllm_gr.entrypoints.recif.api_router import patch_recif_api_router

__all__ = ["patch_recif_api_router"]
