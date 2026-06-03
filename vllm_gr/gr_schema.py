# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Re-export from new location for backward compatibility.
from vllm_gr.inputs.hstu.schema import (  # noqa: F401
    attach_schema_fingerprint,
    build_schema_fingerprint,
    extract_context_features,
    extract_features,
    validate_context_group,
    validate_feature_group,
)
