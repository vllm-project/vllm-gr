# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
def dense_prefill_mode() -> str | None:
    from vllm.config import get_current_vllm_config_or_none

    from vllm_gr.config.models import deployment_for

    config = get_current_vllm_config_or_none()
    if (
        config is None
        or config.cache_config.enable_prefix_caching
        or config.scheduler_config.enable_chunked_prefill
        # A KV connector can load earlier prompt tokens into the paged cache.
        or getattr(config, "kv_transfer_config", None) is not None
    ):
        return None
    mode = deployment_for(config).dense_prefill
    return mode if mode != "off" else None
