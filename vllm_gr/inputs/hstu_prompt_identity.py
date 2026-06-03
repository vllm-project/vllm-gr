# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Re-export from new location for backward compatibility.
from vllm_gr.inputs.hstu.prompt_identity import (  # noqa: F401
    build_hstu_cache_salt,
    build_hstu_prompt_token_ids,
)
